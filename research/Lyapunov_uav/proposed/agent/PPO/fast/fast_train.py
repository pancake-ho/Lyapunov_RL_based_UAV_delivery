from __future__ import annotations

import argparse
import sys
import time
import csv
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EnvConfig
from env.env import Env

from agent.PPO.common import (
    infer_flat_dim,
    split_env_reset,
    split_env_step,
    set_seed,
    ensure_dir,
)
from agent.PPO.common.utils import ScalarLogger, save_json
from agent.PPO.fast.fast_agent import FastPPOAgent, FastPPOConfig


def _find_proposed_root(start: Optional[Path] = None) -> Path:
    """
    proposed root를 자동 탐색함.

    기대 구조:
        proposed/
            config.py
            env/
            agent/
    """
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent

    for parent in [cur, *cur.parents]:
        if (parent / "config.py").exists() and (parent / "env").exists() and (parent / "agent").exists():
            return parent

    raise RuntimeError(
        "proposed root를 찾지 못했습니다.\n"
        "fast_train.py가 research/Lyapunov_uav/proposed/agent/PPO/fast/ 아래에 있는지 확인하세요."
    )

PROPOSED_ROOT = _find_proposed_root()
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-Timescale PPO Agent Training")

    # mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="train이면 PPO 학습, eval이면 평가만 수행.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="eval mode에서 불러올 checkpoint path."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="train mode에서 --checkpoint를 불러와 이어서 학습."
    )

    # experiment
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="저장 directory 이름. None이면 자동 생성.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help=(
            "train mode에서 학습할 episode 수. "
            "현재 기준 episode 1개 = slow-timescale round 1개."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20
    )
    parser.add_argument(
        "--save-every-episodes",
        type=int,
        default=50,
        help="train mode에서 몇 episode마다 checkpoint를 저장할지. 0이면 주기 저장 비활성화.",
    )

    # PPO rollout/update
    parser.add_argument(
        "--rollout-slots",
        type=int,
        default=1024,
        help=(
            "PPO update 1회 전에 수집할 fast slot transition 수."
        ),
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-0.5)

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Actor/Critic MLP hidden dimensions.",
    )
    parser.add_argument(
        "--no-obs-norm",
        action="store_true",
    )
    parser.add_argument(
        "--no-adv-norm",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='auto, cpu, cuda, cuda:0 등.',
    )
    parser.add_argument(
        "--deterministic-torch",
        action="store_true",
        help="재현성 우선. CUDA deterministic 옵션을 켠다.",
    )

    # env
    parser.add_argument("--num-user", type=int, default=None)
    parser.add_argument("--num-rsu", type=int, default=None)
    parser.add_argument("--num-uav", type=int, default=None)

    parser.add_argument(
        "--slow-T",
        dest="slow_T",
        type=int,
        default=None,
        help=(
            "slow-timescale round 길이. "
            "현재 fast-only 학습에서는 episode 하나의 fast slot 길이와 동일하다."
        ),
    )

    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)

    # reward coeffs
    parser.add_argument(
        "--V",
        type=float,
        default=1.0,
        help="env fast reward의 quality term 계수 V.",
    )

    parser.add_argument(
        "--plot-every-episodes",
        type=int,
        default=50,
        help="몇 episode마다 reward/metric plot을 저장할지. 0이면 학습 중 plot 저장 비활성화.",
    )
    parser.add_argument(
        "--plot-smooth-window",
        type=int,
        default=20,
        help="reward moving average window size.",
    )

    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    """
    CLI args를 EnvConfig에 반영함.
    """
    cfg = EnvConfig(seed=int(args.seed))

    updates: Dict[str, Any] = {}

    if args.num_user is not None:
        updates["num_user"] = int(args.num_user)

    if args.num_rsu is not None and args.num_uav is not None:
        updates["num_rsu"] = int(args.num_rsu)
        updates["num_uav"] = int(args.num_uav)
    elif args.num_rsu is not None:
        updates["num_rsu"] = int(args.num_rsu)
        updates["num_uav"] = int(args.num_rsu)
    elif args.num_uav is not None:
        updates["num_rsu"] = int(args.num_uav)
        updates["num_uav"] = int(args.num_uav)

    if args.slow_T is not None:
        updates["slow_T"] = int(args.slow_T)

    if args.layer is not None:
        updates["layer"] = int(args.layer)
        updates["quality_weights"] = tuple(float(i + 1) for i in range(int(args.layer))
                                           )
    if args.chunk is not None:
        updates["chunk"] = int(args.chunk)

    if len(updates) > 0:
        cfg = replace(cfg, **updates)

    return cfg


def build_ppo_config(args: argparse.Namespace) -> FastPPOConfig:
    """
    CLI args를 FastPPOConfig로 변환함.
    """
    return FastPPOConfig(
        rollout_steps=int(args.rollout_slots),
        update_epochs=int(args.update_epochs),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        lr=float(args.lr),
        max_grad_norm=float(args.max_grad_norm),
        clip_coef=float(args.clip_coef),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        normalize_obs=not bool(args.no_obs_norm),
        normalize_adv=not bool(args.no_adv_norm),
        hidden_dims=tuple(int(x) for x in args.hidden_dims),
        init_log_std=float(args.init_log_std),
        device=str(args.device),
    )


def make_run_dir(args: argparse.Namespace, env_cfg: EnvConfig) -> Path:
    """
    실행 결과를 저장하는 directory를 생성함.
    """
    if args.run_name is None:
        run_name = (
            f"fast_ppo_{args.mode}"
            f"_ep{int(args.num_episodes)}"
            f"_slowT{int(env_cfg.slow_T)}"
        )
    else:
        run_name = str(args.run_name)

    run_dir = PROPOSED_ROOT / "fast" / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "figs")

    return run_dir


def sample_random_slow_action(env: Env, rng: np.random.Generator) -> Dict[str, Any]:
    """
    Fast-only PPO 학습용 random slow-timescale action 생성.
    """
    cfg = env.cfg

    m = int(cfg.num_rsu)
    n = int(cfg.num_user)
    u = int(cfg.num_uav)

    if m <= 0 or n <= 0 or u <= 0:
        raise ValueError(
            f"유효하지 않은 숫자입니다: num_rsu={m}, num_uav={u}, num_user={n}"
        )

    # 고정 random 확률
    rsu_user_prob = 0.7
    uav_hire_prob = 0.5
    uav_user_prob = 0.7

    rsu_scheduling = np.zeros((m, n), dtype=np.int32)
    uav_hiring = np.zeros(u, dtype=np.int32)
    uav_scheduling = np.zeros((u, n), dtype=np.int32)

    for user_idx in range(n):
        if rng.random() < rsu_user_prob:
            rsu_idx = int(rng.integers(low=0, high=m))
            rsu_scheduling[rsu_idx, user_idx] = 1
    
    # RSU-USER
    rsu_served_user = (rsu_scheduling.sum(axis=0) > 0)

    requested_content = np.asarray(env.requested_content, dtype=np.int32)
    uav_cached_content = np.asarray(env.uav_cached_content, dtype=np.int32)

    if requested_content.shape != (n,):
        raise ValueError(
            f"requested_content shape mismatch: expected {(n,)}, got {requested_content.shape}"
        )
    if uav_cached_content.shape != (u,):
        raise ValueError(
            f"uav_cached_content shape mismatch: expected {(u,)}, got {uav_cached_content.shape}"
        )

    uav_user_cap = int(getattr(cfg, "uav_user_cap", n))
    uav_user_cap = max(1, min(uav_user_cap, n))

    # UAV / UAV-USER
    for uav_idx in range(u):
        if rng.random() >= uav_hire_prob:
            continue

         # residual user 중심:
        # RSU에 scheduling되지 않은 user만 UAV candidate로 둔다.
        residual_mask = ~rsu_served_user

        # UAV는 하나의 content를 caching하고,
        # 같은 content를 요청하는 user만 지원 가능하다고 둔다.
        cache_match_mask = requested_content == uav_cached_content[uav_idx]

        candidate_mask = residual_mask & cache_match_mask
        candidate_users = np.flatnonzero(candidate_mask)

        # content/cache까지 맞추면 후보가 너무 적을 수 있으므로,
        # 후보가 없으면 residual user 중에서라도 random candidate를 만든다.
        if candidate_users.size == 0:
            candidate_users = np.flatnonzero(residual_mask)

        # 그래도 후보가 없으면 이 UAV는 고용하지 않는다.
        if candidate_users.size == 0:
            continue

        selected_mask = rng.random(candidate_users.size) < uav_user_prob
        selected_users = candidate_users[selected_mask]

        # 확률 샘플링 결과가 비면 candidate 중 1명은 강제로 선택.
        if selected_users.size == 0:
            selected_users = np.asarray(
                [int(rng.choice(candidate_users))],
                dtype=np.int64,
            )

        # UAV 1대당 candidate user 수 제한.
        if selected_users.size > uav_user_cap:
            selected_users = rng.choice(
                selected_users,
                size=uav_user_cap,
                replace=False,
            )

        uav_hiring[uav_idx] = 1
        uav_scheduling[uav_idx, np.asarray(selected_users, dtype=np.int64)] = 1

    # scheduling 대상이 없는 UAV는 hiring 제거.
    active_hire = (uav_scheduling.sum(axis=1) > 0).astype(np.int32)
    uav_hiring = uav_hiring * active_hire
    uav_scheduling = uav_scheduling * uav_hiring[:, None]

    return {
        "rsu_scheduling": rsu_scheduling.astype(np.int32),
        "uav_hiring": uav_hiring.astype(np.int32),
        "uav_scheduling": uav_scheduling.astype(np.int32),
    }



def attach_existing_slow_context(env: Env, obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    FastActionCodec이 obs에서 slow decision을 읽을 수 있도록,
    env가 이미 들고 있는 slow-timescale context만 obs에 붙임.

    여기서 slow action을 새로 만들지는 않음.
    """
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict, got {type(obs)}")

    merged = dict(obs)

    if hasattr(env, "rsu_scheduling"):
        merged["rsu_scheduling"] = np.asarray(env.rsu_scheduling, dtype=np.int32).copy()
    if hasattr(env, "uav_hiring"):
        merged["uav_hiring"] = np.asarray(env.uav_hiring, dtype=np.int32).copy()
    if hasattr(env, "uav_scheduling"):
        merged["uav_scheduling"] = np.asarray(env.uav_scheduling, dtype=np.int32).copy()

    return merged


def save_configs(
    run_dir: Path,
    args: argparse.Namespace,
    env_cfg: EnvConfig,
    ppo_cfg: FastPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> None:
    """
    실행 설정 저장.
    """
    save_json(vars(args), run_dir / "args.json")
    save_json(asdict(env_cfg), run_dir / "env_config.json")
    save_json(asdict(ppo_cfg), run_dir / "ppo_config.json")
    save_json(
        {
            "proposed_root": str(PROPOSED_ROOT),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        run_dir / "run_info.json",
    )


def reset_env(
    env: Env,
    slow_rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    env.reset() 결과를 Gymnasium/Gym 호환 형태로 정리.

    fast-only PPO에서는 매 episode를 하나의 slow-timescale round로 보고,
    reset 직후 random slow action을 무조건 생성해서 env에 주입한다.
    """
    obs, info = split_env_reset(env.reset())

    slow_action = sample_random_slow_action(
        env=env,
        rng=slow_rng,
    )
    env.apply_slow_action(slow_action)

    # apply_slow_action 이후 slow context가 갱신되므로 fast obs를 다시 가져온다.
    obs = env.get_fast_obs()
    obs = attach_existing_slow_context(env, obs)

    info = dict(info)
    info["random_slow_action"] = {
        "rsu_scheduling": np.asarray(slow_action["rsu_scheduling"], dtype=np.int32).copy(),
        "uav_hiring": np.asarray(slow_action["uav_hiring"], dtype=np.int32).copy(),
        "uav_scheduling": np.asarray(slow_action["uav_scheduling"], dtype=np.int32).copy(),
    }
    info["random_rsu_links"] = int(np.sum(slow_action["rsu_scheduling"]))
    info["random_hired_uav"] = int(np.sum(slow_action["uav_hiring"]))
    info["random_uav_links"] = int(np.sum(slow_action["uav_scheduling"]))

    return obs, info


def _read_scalar_csv(csv_path: Path) -> Dict[str, list[float]]:
    """
    ScalarLogger가 저장한 csv를 읽어서 column별 float list로 반환.
    숫자로 변환 안 되는 값은 건너뜀.
    """
    data: Dict[str, list[float]] = {}

    if not csv_path.exists():
        return data

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return data

        for name in reader.fieldnames:
            data[name] = []

        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                try:
                    data.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass

    return data


def _moving_average(values: list[float], window: int) -> np.ndarray:
    """
    reward curve smoothing용 moving average.
    window <= 1이면 원본 반환.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    if window <= 1 or arr.size < window:
        return arr

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="valid")


def save_training_plots(
    run_dir: Path,
    smooth_window: int = 20,
) -> None:
    """
    episodes.csv / updates.csv 기반으로 학습 그래프를 png로 저장.
    """
    figures_dir = run_dir / "figures"
    ensure_dir(figures_dir)

    episodes_csv = run_dir / "logs" / "episodes.csv"
    updates_csv = run_dir / "logs" / "updates.csv"

    ep_data = _read_scalar_csv(episodes_csv)
    update_data = _read_scalar_csv(updates_csv)

    # 1) Episode reward
    if "episode" in ep_data and "episode_reward" in ep_data:
        x = np.asarray(ep_data["episode"], dtype=np.float32)
        y = np.asarray(ep_data["episode_reward"], dtype=np.float32)

        plt.figure()
        plt.plot(x, y, label="episode_reward")

        y_ma = _moving_average(ep_data["episode_reward"], smooth_window)
        if y_ma.size > 0 and y_ma.size <= x.size:
            x_ma = x[-y_ma.size:]
            plt.plot(x_ma, y_ma, label=f"moving_avg_{smooth_window}")

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Fast PPO Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(figures_dir / "episode_reward.png", dpi=200)
        plt.close()

    # 2) Episode components
    component_keys = [
        "episode_delivery",
        "episode_quality",
        "episode_stall",
        "episode_consumed_soc",
        "episode_charged_soc",
        "episode_outage",
    ]

    if "episode" in ep_data:
        x = np.asarray(ep_data["episode"], dtype=np.float32)

        plt.figure()
        plotted = False
        for key in component_keys:
            if key not in ep_data:
                continue
            y = np.asarray(ep_data[key], dtype=np.float32)
            if y.size != x.size:
                continue
            plt.plot(x, y, label=key)
            plotted = True

        if plotted:
            plt.xlabel("Episode")
            plt.ylabel("Value")
            plt.title("Fast PPO Episode Metrics")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(figures_dir / "episode_metrics.png", dpi=200)

        plt.close()

    # 3) PPO update losses
    if "update" in update_data:
        x = np.asarray(update_data["update"], dtype=np.float32)

        loss_keys = [
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "explained_variance",
        ]

        plt.figure()
        plotted = False
        for key in loss_keys:
            if key not in update_data:
                continue
            y = np.asarray(update_data[key], dtype=np.float32)
            if y.size != x.size:
                continue
            plt.plot(x, y, label=key)
            plotted = True

        if plotted:
            plt.xlabel("Update")
            plt.ylabel("Value")
            plt.title("Fast PPO Update Metrics")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(figures_dir / "update_metrics.png", dpi=200)

        plt.close()


def extract_info_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    """
    env.step() info에서 로그용 metric만 추출.
    """
    reward_components = info.get("reward_components", {})
    fast_components = reward_components.get("fast_reward_components", {})

    stall = info.get("stall", 0.0)
    try:
        stall_sum = float(np.sum(np.asarray(stall, dtype=np.float32)))
    except Exception:
        stall_sum = 0.0

    return {
        "sum_delivery": float(fast_components.get("sum_delivery", 0.0)),
        "sum_quality": float(fast_components.get("sum_quality", 0.0)),
        "sum_consumed_soc": float(fast_components.get("sum_consumed_soc", 0.0)),
        "sum_charged_soc": float(fast_components.get("sum_charged_soc", 0.0)),
        "num_hired_uav": float(fast_components.get("num_hired_uav", 0.0)),
        "num_charging_uav": float(fast_components.get("num_charging_uav", 0.0)),
        "num_outage_uav": float(fast_components.get("num_outage_uav", 0.0)),
        "stall_sum": stall_sum,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(int(args.seed), deterministic=bool(args.deterministic_torch))

    env_cfg = build_env_config(args)
    ppo_cfg = build_ppo_config(args)

    slow_rng = np.random.default_rng(int(args.seed) + 10007)

    env = Env(env_cfg)
    obs, reset_info = reset_env(env, slow_rng=slow_rng)

    obs_dim = infer_flat_dim(obs)
    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if args.resume:
        if args.checkpoint is None:
            raise ValueError("--resume 사용 시 --checkpoint가 필요합니다.")
        agent.load(
            path=args.checkpoint,
            strict=True,
            load_optimizer=True,
        )

    run_dir = make_run_dir(args, env_cfg)
    save_configs(
        run_dir=run_dir,
        args=args,
        env_cfg=env_cfg,
        ppo_cfg=ppo_cfg,
        obs_dim=obs_dim,
        action_dim=agent.action_dim,
    )

    episode_logger = ScalarLogger(run_dir / "logs" / "episodes.csv")
    update_logger = ScalarLogger(run_dir / "logs" / "updates.csv")

    print("=" * 100, flush=True)
    print("[FAST PPO TRAIN START]", flush=True)
    print(f"proposed_root : {PROPOSED_ROOT}", flush=True)
    print(f"run_dir       : {run_dir}", flush=True)
    print(f"device        : {agent.device}", flush=True)
    print(f"obs_dim       : {obs_dim}", flush=True)
    print(f"action_dim    : {agent.action_dim}", flush=True)
    print(f"slow_T        : {env_cfg.slow_T}", flush=True)
    print(f"rollout_slots : {ppo_cfg.rollout_steps}", flush=True)
    print(f"reset_info    : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    global_slot = 0
    update_idx = 0
    episode_idx = 0

    last_obs = obs
    last_done = False

    while episode_idx < int(args.num_episodes):
        obs, reset_info = reset_env(env, slow_rng=slow_rng)

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0

        for _ in range(int(env_cfg.slow_T)):
            selected = agent.select_action(
                obs=obs,
                deterministic=False,
                update_norm=True,
            )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )
            next_obs = attach_existing_slow_context(env, next_obs_raw)

            is_round_boundary = bool(info.get("is_round_boundary", False))
            done = bool(terminated or truncated or is_round_boundary)

            agent.store_transition(
                obs_vec=selected["obs_vec"],
                raw_action=selected["raw_action"],
                reward=float(reward),
                done=done,
                value=float(selected["value"]),
                log_prob=float(selected["log_prob"]),
            )

            metrics = extract_info_metrics(info)

            ep_reward += float(reward)
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_steps += 1
            global_slot += 1

            last_obs = next_obs
            last_done = done

            if agent.buffer.is_full:
                agent.finish_rollout(
                    last_obs=last_obs,
                    last_done=last_done,
                )
                update_logs = agent.update()
                update_idx += 1

                update_row = {
                    "update": update_idx,
                    "global_slot": global_slot,
                    "episode": episode_idx + 1,
                    **update_logs,
                }
                update_logger.write(update_row)

                print(
                    "[UPDATE] "
                    f"update={update_idx} "
                    f"slot={global_slot} "
                    f"episode={episode_idx + 1} "
                    f"policy_loss={update_logs['policy_loss']:.6f} "
                    f"value_loss={update_logs['value_loss']:.6f} "
                    f"entropy={update_logs['entropy']:.6f} "
                    f"ev={update_logs['explained_variance']:.4f}",
                    flush=True,
                )

            obs = next_obs

            if done:
                break

        episode_idx += 1

        episode_row = {
            "episode": episode_idx,
            "global_slot": global_slot,
            "episode_steps": ep_steps,
            "episode_reward": ep_reward,
            "episode_delivery": ep_delivery,
            "episode_quality": ep_quality,
            "episode_stall": ep_stall,
            "episode_consumed_soc": ep_consumed_soc,
            "episode_charged_soc": ep_charged_soc,
            "episode_outage": ep_outage,
        }
        episode_logger.write(episode_row)

        print(
            "[EPISODE] "
            f"ep={episode_idx}/{args.num_episodes} "
            f"steps={ep_steps} "
            f"reward={ep_reward:.4f} "
            f"delivery={ep_delivery:.4f} "
            f"quality={ep_quality:.4f} "
            f"stall={ep_stall:.4f}",
            flush=True,
        )

        if (
            int(args.plot_every_episodes) > 0
            and episode_idx % int(args.plot_every_episodes) == 0
        ):
            save_training_plots(
                run_dir=run_dir,
                smooth_window=int(args.plot_smooth_window),
            )
            print(
                f"[PLOT] saved figures to {run_dir / 'figures'}",
                flush=True,
            )

        if (
            int(args.save_every_episodes) > 0
            and episode_idx % int(args.save_every_episodes) == 0
        ):
            ckpt_path = run_dir / "checkpoints" / f"fast_ppo_ep{episode_idx}.pt"
            agent.save(
                ckpt_path,
                extra={
                    "episode": episode_idx,
                    "global_slot": global_slot,
                    "update_idx": update_idx,
                    "env_config": asdict(env_cfg),
                    "ppo_config": asdict(ppo_cfg),
                },
            )
            print(f"[SAVE] {ckpt_path}", flush=True)

    if len(agent.buffer) > 0:
        agent.finish_rollout(
            last_obs=last_obs,
            last_done=last_done,
        )
        update_logs = agent.update()
        update_idx += 1

        update_logger.write(
            {
                "update": update_idx,
                "global_slot": global_slot,
                "episode": episode_idx,
                **update_logs,
            }
        )

        print(
            "[FINAL UPDATE] "
            f"update={update_idx} "
            f"slot={global_slot} "
            f"policy_loss={update_logs['policy_loss']:.6f} "
            f"value_loss={update_logs['value_loss']:.6f}",
            flush=True,
        )

    final_ckpt_path = run_dir / "checkpoints" / "fast_ppo_final.pt"
    agent.save(
        final_ckpt_path,
        extra={
            "episode": episode_idx,
            "global_slot": global_slot,
            "update_idx": update_idx,
            "env_config": asdict(env_cfg),
            "ppo_config": asdict(ppo_cfg),
        },
    )

    save_json(
        {
            "episode": episode_idx,
            "global_slot": global_slot,
            "update_idx": update_idx,
            "final_checkpoint": str(final_ckpt_path),
            "run_dir": str(run_dir),
        },
        run_dir / "train_summary.json",
    )

    save_training_plots(
        run_dir=run_dir,
        smooth_window=int(args.plot_smooth_window),
    )

    print("=" * 100, flush=True)
    print("[FAST PPO TRAIN DONE]", flush=True)
    print(f"final checkpoint: {final_ckpt_path}", flush=True)
    print("=" * 100, flush=True)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(int(args.seed), deterministic=True)

    env_cfg = build_env_config(args)
    ppo_cfg = build_ppo_config(args)

    slow_rng = np.random.default_rng(int(args.seed) + 20011)

    env = Env(env_cfg)
    obs, reset_info = reset_env(env, slow_rng=slow_rng)

    obs_dim = infer_flat_dim(obs)
    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if args.checkpoint is not None:
        agent.load(
            path=args.checkpoint,
            strict=True,
            load_optimizer=False,
        )

    run_dir = make_run_dir(args, env_cfg)
    save_configs(
        run_dir=run_dir,
        args=args,
        env_cfg=env_cfg,
        ppo_cfg=ppo_cfg,
        obs_dim=obs_dim,
        action_dim=agent.action_dim,
    )

    eval_logger = ScalarLogger(run_dir / "logs" / "eval_episodes.csv")

    print("=" * 100, flush=True)
    print("[FAST PPO EVAL START]", flush=True)
    print(f"proposed_root : {PROPOSED_ROOT}", flush=True)
    print(f"run_dir       : {run_dir}", flush=True)
    print(f"device        : {agent.device}", flush=True)
    print(f"obs_dim       : {obs_dim}", flush=True)
    print(f"action_dim    : {agent.action_dim}", flush=True)
    print(f"slow_T        : {env_cfg.slow_T}", flush=True)
    print(f"checkpoint    : {args.checkpoint}", flush=True)
    print(f"reset_info    : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    rewards = []
    deliveries = []
    qualities = []
    stalls = []

    for episode_idx in range(1, int(args.eval_episodes) + 1):
        obs, reset_info = reset_env(env, slow_rng=slow_rng)

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0

        for _ in range(int(env_cfg.slow_T)):
            selected = agent.select_action(
                obs=obs,
                deterministic=True,
                update_norm=False,
            )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )
            next_obs = attach_existing_slow_context(env, next_obs_raw)

            metrics = extract_info_metrics(info)

            ep_reward += float(reward)
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_steps += 1

            done = bool(
                terminated
                or truncated
                or bool(info.get("is_round_boundary", False))
            )

            obs = next_obs

            if done:
                break

        rewards.append(ep_reward)
        deliveries.append(ep_delivery)
        qualities.append(ep_quality)
        stalls.append(ep_stall)

        eval_logger.write(
            {
                "episode": episode_idx,
                "episode_steps": ep_steps,
                "episode_reward": ep_reward,
                "episode_delivery": ep_delivery,
                "episode_quality": ep_quality,
                "episode_stall": ep_stall,
                "episode_consumed_soc": ep_consumed_soc,
                "episode_charged_soc": ep_charged_soc,
                "episode_outage": ep_outage,
            }
        )

        print(
            "[EVAL] "
            f"ep={episode_idx}/{args.eval_episodes} "
            f"steps={ep_steps} "
            f"reward={ep_reward:.4f} "
            f"delivery={ep_delivery:.4f} "
            f"quality={ep_quality:.4f} "
            f"stall={ep_stall:.4f}",
            flush=True,
        )

    summary = {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "delivery_mean": float(np.mean(deliveries)) if deliveries else 0.0,
        "quality_mean": float(np.mean(qualities)) if qualities else 0.0,
        "stall_mean": float(np.mean(stalls)) if stalls else 0.0,
        "checkpoint": args.checkpoint,
        "run_dir": str(run_dir),
    }

    save_json(summary, run_dir / "eval_summary.json")

    print("=" * 100, flush=True)
    print("[FAST PPO EVAL DONE]", flush=True)
    print(
        f"reward_mean={summary['reward_mean']:.4f}, "
        f"delivery_mean={summary['delivery_mean']:.4f}, "
        f"quality_mean={summary['quality_mean']:.4f}, "
        f"stall_mean={summary['stall_mean']:.4f}",
        flush=True,
    )
    print("=" * 100, flush=True)


def main() -> None:
    args = parse_args()

    if int(args.num_episodes) <= 0:
        raise ValueError("--num-episodes는 양수여야 합니다.")
    if int(args.eval_episodes) <= 0:
        raise ValueError("--eval-episodes는 양수여야 합니다.")
    if int(args.rollout_slots) <= 0:
        raise ValueError("--rollout-slots는 양수여야 합니다.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size는 양수여야 합니다.")
    if int(args.update_epochs) <= 0:
        raise ValueError("--update-epochs는 양수여야 합니다.")

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError(f"지원하지 않는 mode입니다: {args.mode}")


if __name__ == "__main__":
    main()