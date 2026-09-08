#!/usr/bin/env python3
# ndt_ppo_baseline.py
"""
Baseline: Ladipo, Okegbile, Cai, "Network Digital Twin-enhanced QoE Optimization
for Adaptive Video Streaming in 6G IoV Networks" (VTC2025-Fall),
re-implemented on the RSU-only version of the UAV-assisted vehicular video
delivery scenario (frame/slot timing, fixed reserved RB = W_R/J_R & P_R/J_R,
chunk ladder S_k/U_k, playback queue Q_n with demand b chunks/slot,
stall = 1{Q_n(t) < b}, Z_n = Q_tilde - Q_n logged for debugging).

Usage:
  python ndt_ppo_baseline.py --episodes 200 --log_slots eval --log_text
  python ndt_ppo_baseline.py --policy btr --eval_episodes 3
"""
import argparse, json, math, os, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================== args
def get_args():
    p = argparse.ArgumentParser("NDT-GRU + PPO baseline (RSU-only)")
    s = p.add_argument_group("scenario")
    s.add_argument("--num_rsu", type=int, default=2)
    s.add_argument("--rsu_spacing", type=float, default=400.0, help="[m], region width")
    s.add_argument("--num_users", type=int, default=8)
    s.add_argument("--v_min", type=float, default=5.0, help="[m/s]")
    s.add_argument("--v_max", type=float, default=20.0, help="[m/s]")
    s.add_argument("--slot", type=float, default=1.0, help="Delta [s]")
    s.add_argument("--T", type=int, default=10, help="slots per frame")
    s.add_argument("--frames", type=int, default=30, help="frames per episode")
    r = p.add_argument_group("radio")
    r.add_argument("--J_R", type=int, default=3, help="max concurrent RSU users")
    r.add_argument("--W_R", type=float, default=20e6, help="[Hz]")
    r.add_argument("--P_R", type=float, default=40.0, help="[W]")
    r.add_argument("--H_R", type=float, default=8.0)
    r.add_argument("--H_n", type=float, default=1.5)
    r.add_argument("--alpha_R", type=float, default=3.0)
    r.add_argument("--beta_R_dB", type=float, default=-70.0, help="path gain at 1 m [dB]")
    r.add_argument("--N0_dBmHz", type=float, default=-166.9897)
    r.add_argument("--gap_dB", type=float, default=3.0103, help="SNR gap Gamma")
    r.add_argument("--fading", choices=["none", "rayleigh"], default="rayleigh")
    r.add_argument("--fading_rho", type=float, default=0.0, help="AR(1) slot-to-slot correlation")
    v = p.add_argument_group("video")
    v.add_argument("--bitrates_mbps", default="0.5,1,2,4")
    v.add_argument("--psnr", default="0.55,0.72,0.86,1.0", help="PSNR per level; U_k = PSNR_k / PSNR_max")
    v.add_argument("--chunk_sec", type=float, default=1.0, help="video seconds per chunk")
    v.add_argument("--b", type=int, default=1, help="playback demand [chunks/slot]")
    v.add_argument("--Lmax", type=int, default=3)
    v.add_argument("--Q_init", type=int, default=3)
    v.add_argument("--Q_tilde", type=int, default=100, help="Z = Q_tilde - Q (logging / obs scale)")
    q = p.add_argument_group("qoe")
    q.add_argument("--beta1", type=float, default=1.0)
    q.add_argument("--beta2", type=float, default=0.5)
    q.add_argument("--beta3", type=float, default=2.0)
    q.add_argument("--fit_beta_csv", default="", help="CSV with Vq,Qv,Re,engagement_sec,n_segments -> fits beta")
    g = p.add_argument_group("gru")
    g.add_argument("--gru_hist", type=int, default=8, help="history length T")
    g.add_argument("--gru_horizon", type=int, default=4, help="prediction horizon N")
    g.add_argument("--gru_hidden", type=int, default=32)
    g.add_argument("--gru_epochs", type=int, default=30)
    g.add_argument("--gru_lr", type=float, default=1e-3)
    g.add_argument("--gru_pretrain_episodes", type=int, default=10)
    g.add_argument("--alpha_base", type=float, default=1.5)
    g.add_argument("--gamma1", type=float, default=1.0)
    g.add_argument("--gamma2", type=float, default=1.0)
    g.add_argument("--cap_by_prediction", action="store_true",
                   help="paper constraint (13a): chunks also capped by predicted bandwidth")
    o = p.add_argument_group("policy")
    o.add_argument("--policy", choices=["ppo", "btr", "random"], default="ppo")
    o.add_argument("--episodes", type=int, default=200)
    o.add_argument("--hidden", default="128,64", help="paper: (128,64)")
    o.add_argument("--lr_actor", type=float, default=1e-4, help="paper: 1e-5")
    o.add_argument("--lr_critic", type=float, default=3e-4, help="paper: 1e-4")
    o.add_argument("--gamma", type=float, default=0.95)
    o.add_argument("--lam", type=float, default=0.95)
    o.add_argument("--clip", type=float, default=0.2)
    o.add_argument("--ppo_epochs", type=int, default=4)
    o.add_argument("--minibatch", type=int, default=64, help="paper: 64")
    o.add_argument("--entropy", type=float, default=0.01)
    e = p.add_argument_group("run")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", default="cpu")
    e.add_argument("--out_dir", default="runs/ndt_ppo")
    e.add_argument("--eval_episodes", type=int, default=3)
    e.add_argument("--log_slots", choices=["eval", "all", "none"], default="eval")
    e.add_argument("--log_text", action="store_true", help="also write human-readable slot log")
    return p.parse_args()


# ============================================================== environment
class RSUVideoEnv:
    """1-D road, M RSU regions, users reflect at road ends.
    Region membership fixed at frame start; position/channel change every slot."""

    def __init__(self, a, rng):
        self.a, self.rng = a, rng
        self.M = a.num_rsu
        self.road = a.num_rsu * a.rsu_spacing
        self.rsu_x = (np.arange(self.M) + 0.5) * a.rsu_spacing
        self.S = np.array([float(x) for x in a.bitrates_mbps.split(",")]) * 1e6 * a.chunk_sec  # bits/chunk
        psnr = np.array([float(x) for x in a.psnr.split(",")])
        self.U = psnr / psnr.max()
        self.K = len(self.S)
        self.Wbar, self.Pbar = a.W_R / a.J_R, a.P_R / a.J_R          # fixed reserved RB
        self.beta = 10 ** (a.beta_R_dB / 10)
        N0 = 10 ** ((a.N0_dBmHz - 30) / 10)
        self.noise = 10 ** (a.gap_dB / 10) * N0 * self.Wbar
        self.n_slots = a.frames * a.T
        self.bhat = None

    def reset(self):
        a, N = self.a, self.a.num_users
        self.x = self.rng.uniform(0, self.road, N)
        self.dir = self.rng.choice([-1.0, 1.0], N)
        self.v = self.rng.uniform(a.v_min, a.v_max, N)
        self.Q = np.full(N, float(a.Q_init))
        self.k_last = np.zeros(N, int)
        self.qoe_last = np.zeros(N)
        self.vq_last = np.zeros(N)
        self.h = (self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N)) / math.sqrt(2)
        self.hist = [deque(maxlen=a.gru_hist) for _ in range(N)]
        self.t, self.frame = 0, -1
        self._new_frame()
        self._observe_channel()
        for n in range(N):
            for _ in range(a.gru_hist):
                self.hist[n].append(self.feat[n])

    def _new_frame(self):
        self.frame += 1
        self.region = np.minimum((self.x // self.a.rsu_spacing).astype(int), self.M - 1)
        self.members = [np.where(self.region == m)[0] for m in range(self.M)]

    def _observe_channel(self):
        a = self.a
        dx = self.x - self.rsu_x[self.region]
        self.d = np.sqrt((a.H_R - a.H_n) ** 2 + dx ** 2)
        if a.fading == "rayleigh":
            w = (self.rng.standard_normal(len(self.x)) + 1j * self.rng.standard_normal(len(self.x))) / math.sqrt(2)
            self.h = a.fading_rho * self.h + math.sqrt(1 - a.fading_rho ** 2) * w
            xi = np.abs(self.h) ** 2
        else:
            xi = 1.0
        g = self.beta * xi / self.d ** a.alpha_R
        self.C = self.Wbar * np.log2(1 + self.Pbar * g / self.noise)      # bit/s, actual this slot
        # handover likelihood: how much of the distance to the region edge one frame of travel covers
        left = self.region * a.rsu_spacing
        d_edge = np.where(self.dir > 0, left + a.rsu_spacing - self.x, self.x - left)
        reach = self.v * a.slot * a.T
        self.ho = np.clip(1 - d_edge / np.maximum(reach, 1e-6), 0, 1)
        self.dx_rel = dx / a.rsu_spacing
        self.feat = np.stack([np.log1p(self.C / 1e6), self.v / a.v_max, self.dx_rel, self.ho], 1)

    def snapshot(self):
        return dict(x=self.x.copy(), v=self.v.copy(), dir=self.dir.copy(), d=self.d.copy(),
                    C=self.C.copy(), ho=self.ho.copy(), Q=self.Q.copy(), k_last=self.k_last.copy(),
                    bhat=self.bhat.copy(), members=[m.copy() for m in self.members])

    def step(self, actions):
        """actions[m] = list of (user_id, k_req), k_req in 0..K (0 = scheduled but idle), len <= J_R."""
        a, N = self.a, self.a.num_users
        l = np.zeros(N, int); k_req = np.zeros(N, int); sched = np.zeros(N, bool)
        for m in range(self.M):
            assert len(actions[m]) <= a.J_R, "RSU capacity violated"
            for n, kk in actions[m]:
                sched[n] = True
                k_req[n] = kk
                if kk <= 0:
                    continue
                rate = self.C[n]
                if a.cap_by_prediction:
                    rate = min(rate, self.bhat[n, 0] * 1e6)
                l[n] = min(a.Lmax, int(rate * a.slot // self.S[kk - 1]))
        delivered = l > 0
        k_del = np.where(delivered, k_req, 0)
        dep = np.minimum(self.Q, a.b)
        stall = self.Q < a.b
        Q_next = self.Q - dep + l
        # ---- QoE factors (paper eq. 6/7/9 in slot form)
        Vq = np.where(delivered, self.U[np.maximum(k_del - 1, 0)], 0.0)
        Qv = np.where(delivered & (self.k_last > 0), np.abs(k_del - self.k_last) / max(self.K - 1, 1), 0.0)
        Re = stall.astype(float) * a.slot
        qoe = a.beta1 * Vq - a.beta2 * Qv - a.beta3 * Re
        rewards = np.array([qoe[ids].sum() for ids in self.members])
        rec = dict(sched=sched, k=k_req, l=l, Q_next=Q_next, stall=stall, Vq=Vq, Qv=Qv, Re=Re, qoe=qoe)
        # ---- transition
        self.Q = Q_next
        self.k_last = np.where(delivered, k_del, self.k_last)
        self.qoe_last, self.vq_last = qoe, Vq
        for n in range(N):
            self.hist[n].append(self.feat[n])
        self.x = self.x + self.dir * self.v * a.slot
        over, under = self.x > self.road, self.x < 0
        self.x[over] = 2 * self.road - self.x[over]; self.dir[over] = -1
        self.x[under] = -self.x[under]; self.dir[under] = 1
        self.t += 1
        done = self.t >= self.n_slots
        if not done:
            if self.t % a.T == 0:
                self._new_frame()
            self._observe_channel()
        return rewards, rec, done


# ============================================================== GRU bandwidth predictor
class BWPredictor(nn.Module):
    def __init__(self, in_dim, hidden, horizon):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)      # eq. (2): W_o h_t + b_o

    def forward(self, x):                            # x: [B, T, in_dim]
        out, _ = self.gru(x)
        return self.head(out[:, -1])                 # log1p(Mbps), [B, N]

    @torch.no_grad()
    def predict_mbps(self, hist):
        x = torch.tensor(np.array([np.array(h) for h in hist]), dtype=torch.float32, device=DEV)
        return np.expm1(self(x).clamp(min=0).cpu().numpy())


def make_gru_dataset(episodes, hist, H):
    X, Y, C = [], [], []
    for Fe in episodes:                              # [n_slots, N, 4]
        for t in range(hist, Fe.shape[0] - H + 1):
            X.append(Fe[t - hist:t].transpose(1, 0, 2))   # [N, hist, 4]
            Y.append(Fe[t:t + H, :, 0].T)                 # [N, H] future log-rate
            C.append(Fe[t - 1][:, [1, 3]])                # [N, 2] (v/vmax, ho) at last input step
    return np.concatenate(X), np.concatenate(Y), np.concatenate(C)


def train_gru(pred, X, Y, C, a):
    opt = torch.optim.Adam(pred.parameters(), lr=a.gru_lr)
    X, Y, C = (torch.tensor(z, dtype=torch.float32, device=DEV) for z in (X, Y, C))
    n = len(X)
    for ep in range(a.gru_epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(0, n, 256):
            idx = perm[s:s + 256]
            yhat = pred(X[idx])
            alpha = a.alpha_base + a.gamma1 * C[idx, 0] + a.gamma2 * C[idx, 1]      # eq. (5)
            lam = torch.where(yhat > Y[idx], alpha[:, None].expand_as(yhat), torch.ones_like(yhat))  # eq. (4)
            loss = (lam * (Y[idx] - yhat) ** 2).mean()                               # eq. (3)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == a.gru_epochs - 1:
            with torch.no_grad():
                rmse = ((np.expm1(pred(X).cpu().numpy()) - np.expm1(Y.cpu().numpy())) ** 2).mean() ** 0.5
            print(f"[GRU] epoch {ep:3d} weighted-loss {tot / n:.4f}  RMSE {rmse:.2f} Mbps")


# ============================================================== PPO actor-critic
def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Per-user shared encoder + region context (mean-pool). Actions per region:
    sequential pick of <=J_R users (resource allocation) and a quality level per picked user."""

    def __init__(self, obs_dim, hidden, K):
        super().__init__()
        H = hidden[-1]
        self.phi = mlp([obs_dim, *hidden])
        self.ctx = mlp([H + 1, H])
        self.mix = mlp([2 * H, H, H])
        self.score = nn.Linear(H, 1)
        self.qhead = nn.Linear(H, K + 1)
        self.vhead = mlp([H, H, 1])

    def encode(self, obs, mask, load):
        e = torch.tanh(self.phi(obs))                              # [B,N,H]
        m = mask.unsqueeze(-1)
        c = (e * m).sum(1) / m.sum(1).clamp(min=1)
        c = torch.tanh(self.ctx(torch.cat([c, load.unsqueeze(-1)], -1)))
        h = torch.tanh(self.mix(torch.cat([e, c.unsqueeze(1).expand_as(e)], -1)))
        return h, c


def run_policy(net, obs, mask, load, J, picks=None, quals=None, deterministic=False):
    h, c = net.encode(obs, mask, load)
    scores = net.score(h).squeeze(-1)                              # [B,N]
    qlogits = net.qhead(h)                                         # [B,N,K+1]
    value = net.vhead(c).squeeze(-1)
    B, N = mask.shape
    avail = mask.bool().clone()
    need_sample = mask.sum(1) > J                                  # if |N_m| <= J everyone is scheduled
    out_p = torch.full((B, J), -1, dtype=torch.long, device=obs.device)
    out_q = torch.full((B, J), -1, dtype=torch.long, device=obs.device)
    logp = torch.zeros(B, device=obs.device); ent = torch.zeros(B, device=obs.device)
    ar = torch.arange(B, device=obs.device)
    for j in range(J):
        active = avail.any(1)
        if not active.any():
            break
        logits = scores.masked_fill(~avail, -1e9)
        d = Categorical(logits=logits)
        if picks is not None: idx = picks[:, j].clamp(min=0)
        elif deterministic:    idx = logits.argmax(1)
        else:                  idx = d.sample()
        sel = active & need_sample
        logp = logp + torch.where(sel, d.log_prob(idx), torch.zeros_like(logp))
        ent = ent + torch.where(sel, d.entropy(), torch.zeros_like(ent))
        dq = Categorical(logits=qlogits[ar, idx])
        if quals is not None: kk = quals[:, j].clamp(min=0)
        elif deterministic:    kk = qlogits[ar, idx].argmax(1)
        else:                  kk = dq.sample()
        logp = logp + torch.where(active, dq.log_prob(kk), torch.zeros_like(logp))
        ent = ent + torch.where(active, dq.entropy(), torch.zeros_like(ent))
        out_p[:, j] = torch.where(active, idx, torch.full_like(idx, -1))
        out_q[:, j] = torch.where(active, kk, torch.full_like(kk, -1))
        avail[ar, idx] = avail[ar, idx] & ~active
    return out_p, out_q, logp, ent, value


def gae(rew, val, gamma, lam):
    T = len(rew); adv = np.zeros(T); last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(val)


def ppo_update(net, opt, buf, a):
    obs = torch.tensor(np.array(buf["obs"]), dtype=torch.float32, device=DEV)
    mask = torch.tensor(np.array(buf["mask"]), dtype=torch.float32, device=DEV)
    load = torch.tensor(np.array(buf["load"]), dtype=torch.float32, device=DEV)
    picks = torch.tensor(np.array(buf["picks"]), dtype=torch.long, device=DEV)
    quals = torch.tensor(np.array(buf["quals"]), dtype=torch.long, device=DEV)
    old_logp = torch.tensor(np.array(buf["logp"]), dtype=torch.float32, device=DEV)
    adv = torch.tensor(np.array(buf["adv"]), dtype=torch.float32, device=DEV)
    ret = torch.tensor(np.array(buf["ret"]), dtype=torch.float32, device=DEV)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    idx = np.arange(len(obs)); stats = []
    for _ in range(a.ppo_epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), a.minibatch):
            mb = idx[s:s + a.minibatch]
            _, _, logp, ent, val = run_policy(net, obs[mb], mask[mb], load[mb], a.J_R, picks[mb], quals[mb])
            ratio = (logp - old_logp[mb]).exp()
            pl = -torch.min(ratio * adv[mb], ratio.clamp(1 - a.clip, 1 + a.clip) * adv[mb]).mean()  # eq. (18)
            vl = 0.5 * (ret[mb] - val).pow(2).mean()
            loss = pl + 0.5 * vl - a.entropy * ent.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            stats.append((pl.item(), vl.item(), ent.mean().item()))
    return np.mean(stats, 0)


# ============================================================== observation / heuristics / logging
def build_obs(env, bhat, a):
    return np.concatenate([
        np.log1p(bhat) / 4.0,                        # predicted BW (N steps)
        (env.Q / a.Q_tilde)[:, None],                # buffer occupancy
        (env.Q < a.b).astype(float)[:, None],        # currently stalling
        (env.k_last / env.K)[:, None],               # last segment version
        env.ho[:, None],                             # handover likelihood
        (env.qoe_last / 5.0)[:, None],               # past QoE
        env.vq_last[:, None],                        # past video quality
        env.dx_rel[:, None], (env.v / a.v_max)[:, None], env.dir[:, None],
    ], 1)


def heuristic_actions(env, bhat, a, mode, rng):
    acts = []
    for m in range(env.M):
        ids = env.members[m]
        if mode == "random":
            chosen = rng.permutation(ids)[:a.J_R]
            acts.append([(int(n), int(rng.integers(1, env.K + 1))) for n in chosen])
        else:  # BTR: lowest buffer first; highest version whose chunk fits the predicted BW
            chosen = ids[np.argsort(env.Q[ids])][:a.J_R]
            lst = []
            for n in chosen:
                feas = np.where(env.S <= bhat[n, 0] * 1e6 * a.slot)[0]
                lst.append((int(n), int(feas.max() + 1) if len(feas) else 1))
            acts.append(lst)
    return acts


class SlotLogger:
    """One JSON line per slot: per region -> per vehicle: state | decision | next state."""

    def __init__(self, jsonl_path, txt_path=None):
        self.fj = open(jsonl_path, "w")
        self.ft = open(txt_path, "w") if txt_path else None

    def log(self, ep, frame, slot, snap, rec, rewards, a):
        regions = []
        for m, ids in enumerate(snap["members"]):
            users = []
            for n in ids:
                n = int(n)
                users.append(dict(
                    id=n, pos=round(float(snap["x"][n]), 1), v=round(float(snap["v"][n]), 1),
                    dir=int(snap["dir"][n]), dist=round(float(snap["d"][n]), 1),
                    C_mbps=round(float(snap["C"][n]) / 1e6, 2),
                    bhat_mbps=[round(float(x), 2) for x in snap["bhat"][n]],
                    ho=round(float(snap["ho"][n]), 2), Q=int(snap["Q"][n]),
                    Z=int(a.Q_tilde - snap["Q"][n]), k_last=int(snap["k_last"][n]),
                    stall=int(snap["Q"][n] < a.b),
                    sched=int(rec["sched"][n]), k=int(rec["k"][n]), l=int(rec["l"][n]),
                    Q_next=int(rec["Q_next"][n]), Z_next=int(a.Q_tilde - rec["Q_next"][n]),
                    Vq=round(float(rec["Vq"][n]), 3), Qv=round(float(rec["Qv"][n]), 3),
                    Re=round(float(rec["Re"][n]), 2), QoE=round(float(rec["qoe"][n]), 3)))
            regions.append(dict(rsu=m, n_users=len(ids), reward=round(float(rewards[m]), 3), users=users))
        record = dict(episode=ep, frame=frame, slot=slot, slot_in_frame=slot % a.T, regions=regions)
        self.fj.write(json.dumps(record) + "\n")
        if self.ft:
            self.ft.write(f"[ep {ep} | frame {frame} | slot {slot} (in-frame {slot % a.T})]\n")
            for rg in regions:
                self.ft.write(f"  RSU{rg['rsu']}  |N|={rg['n_users']}  J_R={a.J_R}  reward={rg['reward']}\n")
                for u in rg["users"]:
                    self.ft.write(
                        f"    veh{u['id']:3d}: pos={u['pos']:7.1f} v={u['v']:4.1f} dir={u['dir']:+d} "
                        f"d={u['dist']:6.1f} C={u['C_mbps']:6.2f} bhat={u['bhat_mbps'][0]:6.2f} ho={u['ho']:.2f} "
                        f"Q={u['Q']:3d} Z={u['Z']:3d} k_last={u['k_last']} stall={u['stall']}"
                        f" | sched={u['sched']} k={u['k']} l={u['l']}"
                        f" | Q'={u['Q_next']:3d} Z'={u['Z_next']:3d} Vq={u['Vq']:.2f} Qv={u['Qv']:.2f} "
                        f"Re={u['Re']:.1f} QoE={u['QoE']:+.3f}\n")
            self.ft.write("\n")

    def close(self):
        self.fj.close()
        if self.ft: self.ft.close()


class Metrics:
    def __init__(self, a):
        self.a = a; self.n = 0; self.slots = 0
        self.qoe = self.stall = self.util = self.chunks = self.switch = self.sched = self.viol = self.Q = 0.0

    def update(self, rec):
        self.n += len(rec["qoe"]); self.slots += 1
        self.qoe += rec["qoe"].sum(); self.stall += rec["stall"].sum()
        self.util += (rec["Vq"] * rec["l"]).sum(); self.chunks += rec["l"].sum()
        self.switch += (rec["Qv"] > 0).sum(); self.sched += rec["sched"].sum()
        self.viol += (rec["Q_next"] > self.a.Q_tilde).sum(); self.Q += rec["Q_next"].sum()

    def summary(self):
        return dict(avg_qoe=self.qoe / self.n, stall_ratio=self.stall / self.n,
                    util_per_chunk=self.util / max(self.chunks, 1), chunks_per_user_slot=self.chunks / self.n,
                    switch_rate=self.switch / self.n, served_ratio=self.sched / self.n,
                    Q_violation_rate=self.viol / self.n, avg_Q=self.Q / self.n)


# ============================================================== episode runner
def run_episode(env, a, predictor, net=None, mode="ppo", deterministic=False,
                logger=None, ep=0, rng=None, collect_feats=None):
    env.reset()
    N, M, J = a.num_users, env.M, a.J_R
    keep_buf = (mode == "ppo" and not deterministic)
    buf = {k: [[] for _ in range(M)] for k in ["obs", "mask", "load", "picks", "quals", "logp", "value", "reward"]}
    met = Metrics(a)
    feats = []
    done = False
    while not done:
        bhat = predictor.predict_mbps(env.hist)          # NDT prediction from history up to t-1
        env.bhat = bhat
        feats.append(env.feat.copy())
        snap = env.snapshot()
        frame, slot = env.frame, env.t
        if mode == "ppo":
            obs_full = build_obs(env, bhat, a)
            obs = np.zeros((M, N, obs_full.shape[1]), np.float32); mask = np.zeros((M, N), np.float32)
            load = np.zeros(M, np.float32)
            for m in range(M):
                ids = env.members[m]
                obs[m, :len(ids)] = obs_full[ids]; mask[m, :len(ids)] = 1; load[m] = len(ids) / J
            with torch.no_grad():
                picks, quals, logp, _, value = run_policy(
                    net, torch.tensor(obs, device=DEV), torch.tensor(mask, device=DEV),
                    torch.tensor(load, device=DEV), J, deterministic=deterministic)
            picks, quals = picks.cpu().numpy(), quals.cpu().numpy()
            actions = [[(int(env.members[m][p]), int(q)) for p, q in zip(picks[m], quals[m]) if p >= 0]
                       for m in range(M)]
        else:
            actions = heuristic_actions(env, bhat, a, mode, rng)
        rewards, rec, done = env.step(actions)
        if keep_buf:
            for m in range(M):
                buf["obs"][m].append(obs[m]); buf["mask"][m].append(mask[m]); buf["load"][m].append(load[m])
                buf["picks"][m].append(picks[m]); buf["quals"][m].append(quals[m])
                buf["logp"][m].append(float(logp[m])); buf["value"][m].append(float(value[m]))
                buf["reward"][m].append(float(rewards[m]))
        met.update(rec)
        if logger: logger.log(ep, frame, slot, snap, rec, rewards, a)
    if collect_feats is not None:
        collect_feats.append(np.stack(feats))
    flat = None
    if keep_buf:
        flat = {k: [] for k in ["obs", "mask", "load", "picks", "quals", "logp", "adv", "ret"]}
        for m in range(M):
            adv, ret = gae(buf["reward"][m], buf["value"][m], a.gamma, a.lam)
            for k in ["obs", "mask", "load", "picks", "quals", "logp"]:
                flat[k] += buf[k][m]
            flat["adv"] += list(adv); flat["ret"] += list(ret)
    return met.summary(), flat


# ============================================================== optional: beta fitting (paper eq. 11-12)
def fit_beta(csv_path, chunk_sec):
    """CSV columns: Vq, Qv, Re, engagement_sec, n_segments. QoE_ref = 5*e / (S*tau + Re)."""
    from scipy.optimize import curve_fit
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    qoe_ref = 5 * d["engagement_sec"] / (d["n_segments"] * chunk_sec + d["Re"])
    f = lambda X, b1, b2, b3: b1 * X[0] - b2 * X[1] - b3 * X[2]
    popt, _ = curve_fit(f, (d["Vq"], d["Qv"], d["Re"]), qoe_ref, p0=[1, 0.5, 1])
    return [float(x) for x in popt]


# ============================================================== main
def main():
    global DEV
    a = get_args()
    DEV = torch.device(a.device)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.fit_beta_csv:
        a.beta1, a.beta2, a.beta3 = fit_beta(a.fit_beta_csv, a.chunk_sec)
        print(f"[beta] fitted beta1={a.beta1:.3f} beta2={a.beta2:.3f} beta3={a.beta3:.3f}")
    json.dump(vars(a), open(os.path.join(a.out_dir, "args.json"), "w"), indent=2)

    env = RSUVideoEnv(a, rng)
    predictor = BWPredictor(4, a.gru_hidden, a.gru_horizon).to(DEV)

    # ---- 1) GRU pretraining on random-policy traces (paper trains predictor first)
    feats = []
    print(f"[GRU] collecting {a.gru_pretrain_episodes} random episodes ...")
    for ep in range(a.gru_pretrain_episodes):
        run_episode(env, a, predictor, mode="random", rng=rng, collect_feats=feats)
    X, Y, C = make_gru_dataset(feats, a.gru_hist, a.gru_horizon)
    print(f"[GRU] dataset {len(X)} windows")
    train_gru(predictor, X, Y, C, a)
    torch.save(predictor.state_dict(), os.path.join(a.out_dir, "gru.pt"))

    # ---- 2) PPO training
    net = None
    if a.policy == "ppo":
        hidden = [int(h) for h in a.hidden.split(",")]
        net = ActorCritic(a.gru_horizon + 9, hidden, env.K).to(DEV)
        critic_ids = {id(p) for p in net.vhead.parameters()}
        opt = torch.optim.Adam([
            {"params": [p for p in net.parameters() if id(p) not in critic_ids], "lr": a.lr_actor},
            {"params": list(net.vhead.parameters()), "lr": a.lr_critic}])
        flog = open(os.path.join(a.out_dir, "train_log.csv"), "w")
        flog.write("episode,avg_qoe,stall_ratio,served_ratio,util_per_chunk,switch_rate,pol_loss,val_loss,entropy,sec\n")
        t0 = time.time()
        for ep in range(a.episodes):
            logger = None
            if a.log_slots == "all":
                logger = SlotLogger(os.path.join(a.out_dir, f"slots_train_ep{ep:04d}.jsonl"),
                                    os.path.join(a.out_dir, f"slots_train_ep{ep:04d}.txt") if a.log_text else None)
            s, buf = run_episode(env, a, predictor, net, "ppo", False, logger, ep)
            if logger: logger.close()
            pl, vl, en = ppo_update(net, opt, buf, a)
            flog.write(f"{ep},{s['avg_qoe']:.4f},{s['stall_ratio']:.4f},{s['served_ratio']:.4f},"
                       f"{s['util_per_chunk']:.4f},{s['switch_rate']:.4f},{pl:.4f},{vl:.4f},{en:.4f},{time.time() - t0:.0f}\n")
            flog.flush()
            if ep % 10 == 0 or ep == a.episodes - 1:
                print(f"[PPO] ep {ep:4d} avg_qoe {s['avg_qoe']:+.3f} stall {s['stall_ratio']:.3f} "
                      f"served {s['served_ratio']:.2f} util {s['util_per_chunk']:.2f} ent {en:.3f} ({time.time() - t0:.0f}s)")
        flog.close()
        torch.save(net.state_dict(), os.path.join(a.out_dir, "ppo.pt"))

    # ---- 3) evaluation with slot-level logging
    results = []
    for ep in range(a.eval_episodes):
        logger = None
        if a.log_slots != "none":
            logger = SlotLogger(os.path.join(a.out_dir, f"slots_eval_ep{ep:02d}.jsonl"),
                                os.path.join(a.out_dir, f"slots_eval_ep{ep:02d}.txt") if a.log_text else None)
        s, _ = run_episode(env, a, predictor, net, a.policy, True, logger, ep, rng)
        if logger: logger.close()
        results.append(s)
        print(f"[EVAL:{a.policy}] ep {ep} " + " ".join(f"{k}={v:.3f}" for k, v in s.items()))
    mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    json.dump(dict(per_episode=results, mean=mean), open(os.path.join(a.out_dir, "eval_metrics.json"), "w"), indent=2)
    print("[EVAL mean] " + " ".join(f"{k}={v:.3f}" for k, v in mean.items()))


DEV = torch.device("cpu")
if __name__ == "__main__":
    main()