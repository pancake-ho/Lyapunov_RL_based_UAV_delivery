import os
import numpy as np
import random
from collections import defaultdict, deque
import copy
import time
import math
import matplotlib.pyplot as plt
import sys
import csv
import datetime
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from pathlib import Path
import logging

from environment_pt import Env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = "./results"
Path(DIR).mkdir(parents=True, exist_ok=True)

class ActionTracer:
    def __init__(self, fig_name: str, label: str):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(DIR).mkdir(parents=True, exist_ok=True)
        self.path = os.path.join(DIR, f"trace_{fig_name}_{label}_{ts}.csv")
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "fig","label","episode","t_small","t_large",
            "alpha","node_qmax","cap_alpha",
            "M_req","q_req","M_eff","q_eff",
            "stall","Q","reward"
        ])

    def log(self, *, fig, label, episode, t_small, t_large,
            alpha, node_qmax, cap_alpha, M_req, q_req, M_eff, q_eff, stall, Q, reward):
        self.w.writerow([
            fig, label, episode, t_small, t_large,
            -1 if alpha is None else int(alpha),
            int(node_qmax) if node_qmax is not None else 0,
            float(cap_alpha) if cap_alpha is not None else 0.0,
            int(M_req), int(q_req), int(M_eff), int(q_eff),
            int(stall), int(Q), float(reward)
        ])

    def close(self):
        try: self.f.close()
        except: pass



# -------------------------------------------------
# Model definitions (S-Model for scheduling, MQ-Model for M,q selection)
# -------------------------------------------------
class QNetS(nn.Module):
    def __init__(self, input_dim=10, output_dim=5, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QNetMQ(nn.Module):
    def __init__(self, input_dim=2, output_dim=28, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------------------------
# Replay Buffer
# -------------------------------------------------
class Replay:
    def __init__(self, capacity=200_000):
        from collections import deque
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d):
        self.buf.append((s,a,r,s2,d))
    def sample(self, n):
        import random, numpy as np
        if len(self.buf) < n: return None
        batch = random.sample(self.buf, n)
        s,a,r,s2,d = map(np.array, zip(*batch))
        return s,a,r,s2,d
    def __len__(self): return len(self.buf)

def dqn_step(optim, behav_net, targ_net, replay: Replay, batch=256, gamma=0.99, double=True):
    import torch, torch.nn.functional as F, numpy as np
    data = replay.sample(batch)
    if data is None: return 0.0
    s,a,r,s2,d = data
    dev = next(behav_net.parameters()).device
    s  = torch.tensor(s,  dtype=torch.float32, device=dev)
    a  = torch.tensor(a,  dtype=torch.long,    device=dev)
    r  = torch.tensor(r,  dtype=torch.float32, device=dev)
    s2 = torch.tensor(s2, dtype=torch.float32, device=dev)
    d  = torch.tensor(d,  dtype=torch.float32, device=dev)
    q  = behav_net(s).gather(1, a.view(-1,1)).squeeze(1)
    with torch.no_grad():
        if double:
            a2 = behav_net(s2).argmax(1)
            q2 = targ_net(s2).gather(1, a2.view(-1,1)).squeeze(1)
        else:
            q2 = targ_net(s2).max(1)[0]
        y  = r + gamma * (1.0 - d) * q2
    loss = F.smooth_l1_loss(q, y)
    optim.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(behav_net.parameters(), 5.0)
    optim.step()
    return float(loss.item())

# -------------------------------------------------
# Agent
# -------------------------------------------------
class HRLAgent:
    def __init__(self, gamma=0.99, epsilon=0.05, learning_rate=1e-3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.behavior_S_model = QNetS().to(DEVICE)
        self.target_S_model = QNetS().to(DEVICE)
        self.behavior_MQ_model = QNetMQ().to(DEVICE)
        self.target_MQ_model = QNetMQ().to(DEVICE)

        self.update_target_S_model()
        self.update_target_MQ_model()

        self.S_optimizer = optim.Adam(self.behavior_S_model.parameters(), lr=self.learning_rate)
        self.MQ_optimizer = optim.Adam(self.behavior_MQ_model.parameters(), lr=self.learning_rate)

        self.S_replay = Replay(20000)
        self.MQ_replay = Replay(20000)

        # Define action space list (M,q combinations)
        self.action_list = self._build_action_list()

    def _build_action_list(self) -> List[Tuple[int, int]]:
        action_list = [(0, 0)]
        for M in range(1, 10):
            for q in range(1, 4):
                action_list.append((M, q))
        return action_list

    def update_target_MQ_model(self):
        self.target_MQ_model.load_state_dict(self.behavior_MQ_model.state_dict())

    def update_target_S_model(self):
        self.target_S_model.load_state_dict(self.behavior_S_model.state_dict())

    @staticmethod
    def _flatten_node_state(node_state) -> np.ndarray:
        dists, quals = node_state
        x = np.array(dists + quals, dtype=np.float32)
        if x.shape[0] < 10:
            x = np.pad(x, (0, 10 - x.shape[0]), mode='constant')
        return x.reshape(1, -1)

    # -------------------------- Scheduling (S) --------------------------
    def get_alpha(self, env):
        if env.large_time >= env.horizon_small or env.large_time >= len(env.distances):
            return None, None, None
        res = env.check_in_cell(env.large_time)
        if res is None:
            return None, None, None
        idxs, node_state = res
        x_np = self._flatten_node_state(node_state)
        x = torch.from_numpy(x_np).to(device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            q_value = self.behavior_S_model(x)
        q_val = q_value.detach().cpu().numpy().flatten()
        for i, gi in enumerate(idxs):
            if gi < 0:
                q_val[i] = -1e30
        if np.random.rand() <= self.epsilon:
            valid = [i for i, gi in enumerate(idxs) if gi >= 0]
            link_idx = np.random.choice(valid) if valid else None
        else:
            link_idx = int(np.argmax(q_val))
        if link_idx is None:
            return None, node_state, None
        env_link_node_index = idxs[link_idx]
        return env_link_node_index, node_state, link_idx

    def test_get_alpha(self, env):
        if env.large_time >= env.horizon_small or env.large_time >= len(env.distances):
            return None, None, None
        res = env.check_in_cell(env.large_time)
        if res is None:
            return None, None, None
        idxs, node_state = res
        x_np = self._flatten_node_state(node_state)
        x = torch.from_numpy(x_np).to(device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            q_value = self.target_S_model(x)
        q_val = q_value.detach().cpu().numpy().flatten()
        for i, gi in enumerate(idxs):
            if gi < 0:
                q_val[i] = -1e30
        link_idx = int(np.argmax(q_val))
        if link_idx is None:
            return None, node_state, None
        env_link_node_index = idxs[link_idx]
        return env_link_node_index, node_state, link_idx

    @staticmethod
    def select_alpha_heuristic(env, mode: str):
        t = min(env.small_time, env.horizon_small - 1)
        cand = []
        for i in range(env.num_node):
            d = env.distances[t][i]
            if d <= 1.0:
                if mode == "nearest":
                    cand.append((d, i))
                elif mode == "highest_quality":
                    if env.node_quality[i] > 0:
                        cand.append((-env.node_quality[i], d, i))
        if not cand: return None
        if mode == "nearest":
            cand.sort(key=lambda x: x[0]); return cand[0][1]
        else:
            cand.sort(key=lambda x: (x[0], x[1])); return cand[0][2]


    # --------------------------- M,q selection ---------------------------
    def get_Mq(self, prev_state: np.ndarray, alpha: Optional[int], env) -> Tuple[int, int]:
        if alpha is None or alpha < 0:
            return (0, 0)
        s = prev_state.copy()
        if s[0, 1] == 0.0:
            s[0, 1] = 2.0
        x = torch.from_numpy(s).to(device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.behavior_MQ_model(x)
        # feasible mask by capacity
                # agent_pt.py — get_Mq / test_get_Mq 공통 패치 핵심
        env.calculate_capacity(env.large_time)
        cap_node = float(env.ChannelCapacity_list[alpha])
        node_qmax = int(env.node_quality[alpha])  # 추가

        qv = q_values.detach().cpu().numpy().flatten()
        for ai, (M, q) in enumerate(self.action_list):
            if (M * q) > cap_node or q > node_qmax:  # ← 캐시 제약
                qv[ai] = -1e30
        best = int(np.argmax(qv))
        return self.action_list[best]


    def test_get_Mq(self, prev_state: np.ndarray, alpha: Optional[int], env, cap_list=None) -> Tuple[int, int]:
        if alpha is None or alpha < 0:
            return (0, 0)
        s = prev_state.copy()
        if s[0, 1] == 0.0:
            s[0, 1] = 2.0
        if cap_list is None:
            env.calculate_capacity(env.large_time)
            cap_node = float(env.ChannelCapacity_list[alpha])
        else:
            cap_node = float(cap_list[alpha])

        node_qmax = int(env.node_quality[alpha])     # <-- 추가
        x = torch.from_numpy(s).to(device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.target_MQ_model(x)
        qv = q_values.detach().cpu().numpy().flatten()
        for ai, (M, q) in enumerate(self.action_list):
            if (M * q) > cap_node or q > node_qmax:  # <-- 캐시 제약까지 함께 마스킹
                qv[ai] = -1e30
        best = int(np.argmax(qv))
        return self.action_list[best]


def pretrain_dqn(
    episodes=1500,                 # 총 에피소드(큰/작은 시간척도 포함)
    sync_every=150,               # 타깃 네트 동기화 주기
    eval_every=150,               # 진행 로그 주기
    s_out="./results/pretrained_S.pt",
    mq_out="./results/pretrained_MQ.pt",
    seed=42
):
    import random, numpy as np, torch, logging, os
    logger = logging.getLogger("HRL"); logger.setLevel(logging.INFO)
    if not logger.handlers:
        _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler(); sh.setFormatter(_fmt); logger.addHandler(sh)
        fh = logging.FileHandler(os.path.join(DIR, "pretrain.log"), mode="w", encoding="utf-8")
        fh.setFormatter(_fmt); logger.addHandler(fh)

    env = Env()                 # 논문식 Env(보상/제약 내장)  :contentReference[oaicite:5]{index=5}
    agent = HRLAgent()

    # 별도 리플레이(클래스 내부 버퍼와 무관)
    Sbuf, MQbuf = Replay(200_000), Replay(200_000)
    Sopt, MQopt  = agent.S_optimizer, agent.MQ_optimizer

    def eps_fn(t, T):  # ε 선형 감소 (0.20→0.02)
        hi, lo = 0.20, 0.02
        return max(lo, hi - (hi - lo) * (t / max(1,T)))

    for ep in range(1, episodes+1):
        # 다양한 조건 노출: SNR/케이스 무작위
        env.snr_set_db(random.choice([15.0, 20.0, 25.0, 30.0]))   # :contentReference[oaicite:6]{index=6}
        env.set_case(random.choice([1,2,3]))                      # :contentReference[oaicite:7]{index=7}
        obs = env.reset()

        epsilon = eps_fn(ep, episodes)
        done = False; ep_R = 0.0; stall = 0; qsum = 0; csum = 0
        alpha, sS, aS = None, None, None

        while not done and env.small_time < env.horizon_small:
            # === 큰 시간척도: α 선택 (S 네트) ===
            if (env.small_time % env.agent_T) == 0:
                # 후보 추출 → S-입력(10차원) 만들기   :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
                res = env.check_in_cell(env.large_time)
                if res is None:
                    alpha, sS, aS = None, None, None
                else:
                    idxs, node_state = res
                    x_np = HRLAgent._flatten_node_state(node_state)        # (1,10)  :contentReference[oaicite:10]{index=10}
                    x_t  = torch.from_numpy(x_np).to(DEVICE, torch.float32)
                    with torch.no_grad():
                        qS = agent.behavior_S_model(x_t).cpu().numpy().flatten()
                    # 불가 후보 마스킹
                    for i, gi in enumerate(idxs):
                        if gi < 0: qS[i] = -1e30
                    valid = [i for i,gi in enumerate(idxs) if gi >= 0]
                    if (len(valid) == 0):
                        alpha, sS, aS = None, None, None
                    else:
                        if random.random() < epsilon:
                            link_idx = random.choice(valid)
                        else:
                            link_idx = int(np.argmax(qS))
                        alpha = idxs[link_idx]
                        sS = x_np.copy()                 # S-state (학습 전)
                        aS = link_idx                    # 액션 인덱스(0..4)

            # === 작은 시간척도: (M,q) 선택 (MQ 네트) ===
            prev = np.asarray(obs, dtype=np.float32).reshape(1,-1)
            # 제약 마스크는 MQ 네트 내부에서 처리함  :contentReference[oaicite:11]{index=11}
            if random.random() < epsilon:
                if alpha is None or alpha < 0:
                    M,q = 0,0
                else:
                    env.calculate_capacity(env.large_time)
                    cap   = float(env.ChannelCapacity_list[alpha])
                    qmax  = int(env.node_quality[alpha])
                    cand = [(M_,q_) for (M_,q_) in agent.action_list if (M_*q_<=cap and q_<=qmax)]
                    M,q = random.choice(cand) if cand else (0,0)
            else:
                M,q = agent.get_Mq(prev, alpha, env)    # behavior 네트 사용  :contentReference[oaicite:12]{index=12}

            # === 환경 전개(보상: 논문식) ===
            K_pred  = int(env.last_K)   # 이전 재생 품질(예측)  :contentReference[oaicite:13]{index=13}
            K_learn = int(q)            # 실제 재생 품질(선택)
            obs2, r, done, info = env.step(alpha, int(M), int(q), K_pred=K_pred, K_learn=K_learn)  # :contentReference[oaicite:14]{index=14}
            ep_R += r; stall += int(info["stall"]); qsum += int(info["M_eff"])*int(info["q_eff"]); csum += int(info["M_eff"])

            # === 리플레이 적재 ===
            # S-transition: α를 선택한 큰 슬롯에서만 적재
            if sS is not None and aS is not None:
                # 다음 큰 슬롯 상태 sS' 구성
                res2 = env.check_in_cell(env.large_time)
                if res2 is not None:
                    _, node_state2 = res2
                    sS2 = HRLAgent._flatten_node_state(node_state2)
                else:
                    sS2 = np.zeros_like(sS)
                Sbuf.push(sS.squeeze(0), int(aS), float(r), sS2.squeeze(0), int(done))
                sS, aS = None, None  # 같은 큰 슬롯에서 중복 적재 방지

            # MQ-transition: 매 작은 슬롯 적재
            a_idx = agent.action_list.index((int(M), int(q)))
            MQbuf.push(prev.squeeze(0), int(a_idx), float(r), np.asarray(obs2, dtype=np.float32), int(done))

            # === 최적화 스텝 ===
            dqn_step(Sopt,  agent.behavior_S_model,  agent.target_S_model,  Sbuf, batch=256, gamma=0.99, double=True)
            dqn_step(MQopt, agent.behavior_MQ_model, agent.target_MQ_model, MQbuf, batch=256, gamma=0.99, double=True)

            obs = obs2

        # === 타깃 동기화/로그/체크포인트 ===
        if (ep % sync_every) == 0:
            agent.update_target_S_model(); agent.update_target_MQ_model()   # :contentReference[oaicite:15]{index=15}
        if (ep % eval_every) == 0:
            avg_q = 0.0 if csum==0 else (qsum/csum)
            logger.info(f"[pretrain] ep={ep} | R={ep_R:.2f} | stall={stall} | avgQ={avg_q:.3f} | eps={epsilon:.3f}")
        if (ep % eval_every) == 0 or ep==episodes:
            torch.save({"state_dict": agent.behavior_S_model.state_dict()}, s_out)
            torch.save({"state_dict": agent.behavior_MQ_model.state_dict()}, mq_out)

    logger.info(f"[pretrain] saved S→{s_out}, MQ→{mq_out}")

    # ------------------------------- Train/Eval Loops (요약) -------------------------------
    # (여기서는 평가/플롯에 필요한 핵심 루틴만 발췌)

def _eval_case_fig23(env: Env, agent: HRLAgent, mode: str, case: int, episodes: int = 30):
    # 30dB로 고정 (논문 Fig2–3)
    env.snr_set_db(30.0)  # dB
    avg_buf_acc = 0.0
    avg_qual_acc = 0.0

    tracer = ActionTracer("Fig23", f"{mode}_case{case}")

    for ep in range(episodes):
        env.set_case(case)       # <-- PT 환경에서 케이스 설정
        obs = env.reset()
        # 집계 변수
        buffer_events = 0
        qual_sum = 0
        chunk_sum = 0
        done = False
        # 작은 시간척도 반복
        while not done and env.small_time < env.horizon_small:
            # 스케줄링(큰 시간척도) 선택
            if (env.small_time % env.agent_T) == 0:
                if mode == "proposed":
                    alpha, _, _ = agent.test_get_alpha(env)
                elif mode in ("nearest", "highest_quality"):
                    alpha = HRLAgent.select_alpha_heuristic(env, mode)
                else:
                    alpha = None
            # 전송(M,q)은 항상 DQN (논문 Fig2–3 비교군 정의)
            M, q = agent.test_get_Mq(np.asarray(obs, dtype=np.float32).reshape(1, -1), alpha, env)
            obs, reward, done, info = env.step(alpha, M, q, K_pred=int(env.last_K), K_learn=int(q))
            # 집계
            buffer_events += int(info["stall"])
            qual_sum += int(info["M_eff"]) * int(info["q_eff"])
            chunk_sum += int(info["M_eff"])
             # --- NEW: trace log per step
            node_qmax = (env.node_quality[alpha] if (alpha is not None and 0 <= alpha < env.num_node) else 0)
            tracer.log(
                fig="Fig23",
                label=f"{mode}_case{case}",
                episode=ep,
                t_small=env.small_time,
                t_large=env.large_time,
                alpha=alpha,
                node_qmax=node_qmax,
                cap_alpha=info.get("cap_alpha", 0.0),
                M_req=M, q_req=q,
                M_eff=info.get("M_eff", 0),
                q_eff=info.get("q_eff", 0),
                stall=info.get("stall", 0),
                Q=info.get("Q", 0),
                reward=reward
            )
        small_T = max(1, env.small_time)
        avg_buf_acc += buffer_events / small_T
        avg_qual_acc += (0.0 if chunk_sum == 0 else (qual_sum / chunk_sum))

    tracer.close()  # <-- NEW
    return (avg_buf_acc / episodes), (avg_qual_acc / episodes)


def _eval_fig45(env: Env, agent: HRLAgent, mode: str, snr_db: float, episodes: int = 30):
    # 1) SNR 설정
    env.snr_set_db(float(snr_db))

    avg_buf_acc = 0.0
    avg_qual_acc = 0.0

    tracer = ActionTracer("Fig45", f"{mode}_{int(snr_db)}dB")   # <-- NEW

    for ep in range(episodes):
        # 2) 랜덤·미지 캐싱 정책: case 무작위 선택
        env.set_case(random.choice([1, 2, 3]))
        obs = env.reset()

        buffer_events = 0
        qual_sum = 0
        chunk_sum = 0
        done = False

        while not done and env.small_time < env.horizon_small:
            # 큰 시간척도에서만 스케줄링(α) 갱신
            if (env.small_time % env.agent_T) == 0:
                if mode == "proposed":
                    alpha, _, _ = agent.test_get_alpha(env)
                elif mode in ("nearest", "highest_quality"):
                    alpha = HRLAgent.select_alpha_heuristic(env, mode)
                else:
                    alpha = None

            # 3) MQ는 항상 DQN (논문 비교군 정의)
            prev = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            M, q = agent.test_get_Mq(prev, alpha, env)

            # 5) env.step 시그니처 정합 (K_pred=K_learn=q)
            obs, reward, done, info = env.step(alpha, M, q, K_pred=int(env.last_K), K_learn=int(q))
            # 4) 지표 집계
            buffer_events += int(info["stall"])
            qual_sum += int(info["M_eff"]) * int(info["q_eff"])
            chunk_sum += int(info["M_eff"])

            # --- NEW: trace log
            node_qmax = (env.node_quality[alpha] if (alpha is not None and 0 <= alpha < env.num_node) else 0)
            tracer.log(
                fig="Fig45",
                label=f"{mode}_{int(snr_db)}dB",
                episode=ep,
                t_small=env.small_time,
                t_large=env.large_time,
                alpha=alpha,
                node_qmax=node_qmax,
                cap_alpha=info.get("cap_alpha", 0.0),
                M_req=M, q_req=q,
                M_eff=info.get("M_eff", 0),
                q_eff=info.get("q_eff", 0),
                stall=info.get("stall", 0),
                Q=info.get("Q", 0),
                reward=reward
            )

        small_T = max(1, env.small_time)
        avg_buf_acc += buffer_events / small_T
        avg_qual_acc += (0.0 if chunk_sum == 0 else (qual_sum / chunk_sum))

    tracer.close()
    return (avg_buf_acc / episodes), (avg_qual_acc / episodes)


def _eval_fig67(env: Env, agent: "HRLAgent", method: str, case_id: int, episodes: int = 10):
    # 30 dB 고정
    env.snr_set_db(30.0)

    avg_buf_acc, avg_qual_acc = 0.0, 0.0

    tracer = ActionTracer("Fig67", f"{method}_case{case_id}")  # <-- NEW

    for ep in range(episodes):
        # 케이스 설정 → 리셋
        env.set_case(case_id)
        obs = env.reset()

        stall_cnt = 0
        qual_sum = 0
        chunk_sum = 0
        done = False
        alpha = None

        while not done and env.small_time < env.horizon_small:
            # 큰 시간척도마다 α 갱신
            if (env.small_time % env.agent_T) == 0:
                alpha, _, _ = agent.test_get_alpha(env)

            # 전송 정책 선택
            if method == "proposed":
                prev = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                M, q = agent.test_get_Mq(prev, alpha, env)

            elif method == "most_chunk":
                # q=1, 용량 내 최대 M
                if alpha is None or alpha < 0:
                    M, q = 0, 0
                else:
                    env.calculate_capacity(env.large_time)
                    cap = float(env.ChannelCapacity_list[alpha])
                    q = 1
                    # q<=node_qmax(=1이므로 자동 만족). M*q ≤ cap 최대 M
                    M = max([M_ for (M_, q_) in agent.action_list if (q_ == 1 and (M_ * 1) <= cap) ]
                            or [0])

            elif method == "highest_quality_chunk":
                if alpha is None or alpha < 0:
                    M, q = 0, 0
                else:
                    env.calculate_capacity(env.large_time)
                    cap = float(env.ChannelCapacity_list[alpha])
                    qmax = int(env.node_quality[alpha])
                    if qmax <= 0:
                        M, q = 0, 0
                    else:
                        M = max([M_ for (M_, q_) in agent.action_list if (q_ == qmax and (M_ * qmax) <= cap)]
                                or [0])
                        q = qmax if M > 0 else 0
            else:
                M, q = 0, 0

            # step 호출 (K_pred=K_learn=q)
            obs, reward, done, info = env.step(alpha, M, q, K_pred=int(env.last_K), K_learn=int(q))

            # 지표 집계
            stall_cnt += int(info["stall"])
            qual_sum  += int(info["M_eff"]) * int(info["q_eff"])
            chunk_sum += int(info["M_eff"])

            # --- NEW: trace log
            node_qmax = (env.node_quality[alpha] if (alpha is not None and 0 <= alpha < env.num_node) else 0)
            tracer.log(
                fig="Fig67",
                label=f"{method}_case{case_id}",
                episode=ep,
                t_small=env.small_time,
                t_large=env.large_time,
                alpha=alpha,
                node_qmax=node_qmax,
                cap_alpha=info.get("cap_alpha", 0.0),
                M_req=M, q_req=q,
                M_eff=info.get("M_eff", 0),
                q_eff=info.get("q_eff", 0),
                stall=info.get("stall", 0),
                Q=info.get("Q", 0),
                reward=reward
            )

        small_T = max(1, env.small_time)
        avg_buf_acc  += (stall_cnt / small_T)
        avg_qual_acc += (0.0 if chunk_sum == 0 else (qual_sum / chunk_sum))

    tracer.close()
    return (avg_buf_acc / episodes), (avg_qual_acc / episodes)


# -------------------------------------------------
# 실험 실행 및 플로팅 (Fig.2~7)
# -------------------------------------------------

def run_all_figures(pretrained_S: str = None, pretrained_MQ: str = None):
    logger = logging.getLogger("HRL")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(_fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(os.path.join(DIR, "run.log"), mode="w", encoding="utf-8")
        fh.setFormatter(_fmt)
        logger.addHandler(fh)

    env = Env()

    agent = HRLAgent()

    def _load(model, path):
        if path and os.path.exists(path):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                sd = obj.get("model_state_dict", obj.get("state_dict", obj))
                model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(obj.state_dict())
            return True
        return False

    s_ok = _load(agent.behavior_S_model, pretrained_S)
    if s_ok:
        agent.update_target_S_model()
    mq_ok = _load(agent.behavior_MQ_model, pretrained_MQ)
    if mq_ok:
        agent.update_target_MQ_model()
    logger.info(f"[pretrain] S-loaded={s_ok} from '{pretrained_S}', MQ-loaded={mq_ok} from '{pretrained_MQ}'")

    # ---------------- Fig.2~3 (3 cases) ----------------
    cases = [1, 2, 3]
    modes = ["proposed", "nearest", "highest_quality"]

    buf23  = {m: [] for m in modes}
    qual23 = {m: [] for m in modes}

    for mode in modes:
        for case in cases:
            logger.info(f"[fig23] mode={mode}, case={case} (testing 30 episodes)")
            avg_buf, avg_q = _eval_case_fig23(env, agent, mode, case, episodes=100)
            logger.info(f"[fig23] mode={mode}, case={case} results | avg_buf_rate={avg_buf:.4f}, avg_quality={avg_q:.4f}")
            buf23[mode].append(avg_buf)
            qual23[mode].append(avg_q)

    # (옵션) 간단 검증
    for m in modes:
        assert len(buf23[m]) == len(cases), f"buf length mismatch for {m}"
        assert len(qual23[m]) == len(cases), f"qual length mismatch for {m}"

    # --- Plotting Fig.2 (Buffering Rate vs Case) ---
    fig2 = plt.figure()
    for m in modes:
        plt.plot(cases, buf23[m], marker='o', label=m)
    plt.xlabel("Case"); plt.ylabel("Buffering Rate")
    plt.title("Buffering Rate in Various Caching Environments (Fig.2)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig2.png")); plt.close(fig2)

    # --- Plotting Fig.3 (Avg. Quality vs Case) ---
    fig3 = plt.figure()
    for m in modes:
        plt.plot(cases, qual23[m], marker='o', label=m)
    plt.xlabel("Case"); plt.ylabel("Avg. Quality Level")
    plt.title("Avg. Quality Level in Various Caching Environments (Fig.3)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig3.png")); plt.close(fig3)

    # --------------- Fig.4~5 (SNR sweep) ---------------
    snrs = [15.0, 20.0, 25.0, 30.0]
    methods = ["proposed", "nearest", "highest_quality"]
    buf45  = {m: [] for m in methods}
    qual45 = {m: [] for m in methods}

    for mode in methods:
        for snr in snrs:
            logger.info(f"[fig45] mode={mode}, SNR={snr}dB (testing 30 episodes)")
            avg_buf, avg_q = _eval_fig45(env, agent, mode, snr, episodes=100)
            logger.info(f"[fig45] mode={mode}, SNR={snr}dB results | "
                        f"avg_buf_rate={avg_buf:.4f}, avg_quality={avg_q:.4f}")
            # ✅ append는 한 번만
            buf45[mode].append(avg_buf)
            qual45[mode].append(avg_q)

    # 간단 검증
    for m in methods:
        assert len(buf45[m]) == len(snrs), f"buf length mismatch for {m}"
        assert len(qual45[m]) == len(snrs), f"qual length mismatch for {m}"

    # Fig.4
    fig4 = plt.figure()
    for m in methods:
        plt.plot(snrs, buf45[m], marker='o', label=m)
    plt.xlabel("SNR (dB)"); plt.ylabel("Buffering Rate")
    plt.title("Buffering Rate vs SNR (Fig.4)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig4.png")); plt.close(fig4)

    # Fig.5
    fig5 = plt.figure()
    for m in methods:
        plt.plot(snrs, qual45[m], marker='o', label=m)
    plt.xlabel("SNR (dB)"); plt.ylabel("Avg. Quality Level")
    plt.title("Average Quality vs SNR (Fig.5)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig5.png")); plt.close(fig5)

    # --------------- Fig.6~7 (3 cases, per-episode) ---------------
    methods = ["proposed", "most_chunk", "highest_quality_chunk"]
    cases = [1, 2, 3]
    buf67  = {m: [] for m in methods}
    qual67 = {m: [] for m in methods}

    for case in cases:
        logging.info(f"[fig67] case={case}")
        for m in methods:
            logging.info(f"[fig67] method={m}, case={case} (testing 10 episodes)")
            avg_buf, avg_q = _eval_fig67(env, agent, method=m, case_id=case, episodes=100)  # <-- 함수명 소문자
            buf67[m].append(avg_buf)
            qual67[m].append(avg_q)
            logging.info(f"[fig67] method={m}, case={case} | avg_buf_rate={avg_buf:.6f}, avg_quality={avg_q:.4f}")

    # 간단한 길이 검증
    for m in methods:
        assert len(buf67[m]) == len(cases), f"buf length mismatch for {m}"
        assert len(qual67[m]) == len(cases), f"qual length mismatch for {m}"

    # --- Plotting Fig.6 ---
    fig6 = plt.figure()
    for m in methods:
        y = buf67[m]                                    # <-- 이름 정합
        plt.plot(cases, y, marker='o', label=m)
    plt.xlabel("Case")
    plt.ylabel("Buffering Rate")
    plt.title("Buffering Rate in Various Caching Environments (Fig.6)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig6.png")); plt.close(fig6)

    # --- Plotting Fig.7 ---
    fig7 = plt.figure()
    for m in methods:
        y = qual67[m]                                   # <-- 이름 정합
        plt.plot(cases, y, marker='o', label=m)
    plt.xlabel("Case")
    plt.ylabel("Avg. Quality Level")
    plt.title("Avg. Quality Level in Various Caching Environments (Fig.7)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(DIR, "Fig7.png")); plt.close(fig7)

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true", help="Run DQN pretraining and save .pt files")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--s_out", type=str, default="./results/pretrained_S.pt")
    parser.add_argument("--mq_out", type=str, default="./results/pretrained_MQ.pt")
    args = parser.parse_args()

    if args.pretrain:
        pretrain_dqn(episodes=args.episodes, s_out=args.s_out, mq_out=args.mq_out)
    else:
        s_path = os.environ.get("S_MODEL", "./results/pretrained_S.pt")
        mq_path = os.environ.get("MQ_MODEL", "./results/pretrained_MQ.pt")
        run_all_figures(pretrained_S=s_path, pretrained_MQ=mq_path)