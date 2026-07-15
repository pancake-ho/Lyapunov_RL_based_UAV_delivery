from __future__ import annotations

import csv
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        if (
            (parent / "config.py").exists()
            and (parent / "env").exists()
            and (parent / "agent").exists()
        ):
            return parent

    raise RuntimeError(
        "proposed root를 찾지 못했습니다.\n"
        "fast_train.py가 research/Lyapunov_uav/proposed/agent/PPO/fast/ 아래에 있는지 확인하세요."
    )


PROPOSED_ROOT = _find_proposed_root()

if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))


from config import EnvConfig
from env.env import Env

from agent.PPO.config import FastTrainConfig, get_fast_ppo_config
from agent.PPO.common import (
    infer_fast_obs_dim,
    split_env_reset,
    split_env_step,
    set_seed,
    ensure_dir,
)
from agent.PPO.common.utils import ScalarLogger, save_json
from agent.PPO.fast.fast_agent import FastPPOAgent, FastPPOConfig as AgentPPOConfig


def build_env_config() -> EnvConfig:
    """
    EnvConfig는 proposed/config.py 기준 그대로 사용한다.
    fast_train.py에서는 시스템 상수를 override하지 않는다.
    """
    return EnvConfig()


def build_agent_ppo_config(train_cfg: FastTrainConfig) -> AgentPPOConfig:
    hidden_dims = train_cfg.hidden_dims

    if hidden_dims is None:
        hidden_dims = [256, 256]

    return AgentPPOConfig(
        rollout_steps=int(
            train_cfg.rollout_slots
        ),
        update_epochs=int(
            train_cfg.update_epochs
        ),
        batch_size=int(
            train_cfg.batch_size
        ),

        gamma=float(
            train_cfg.gamma
        ),
        gae_lambda=float(
            train_cfg.gae_lambda
        ),

        lr=float(
            train_cfg.lr
        ),
        max_grad_norm=float(
            train_cfg.max_grad_norm
        ),

        clip_coef=float(
            train_cfg.clip_coef
        ),
        value_coef=float(
            train_cfg.value_coef
        ),

        categorical_entropy_coef=float(
            train_cfg
            .categorical_entropy_coef
        ),
        power_entropy_coef=float(
            train_cfg
            .power_entropy_coef
        ),

        target_kl=(
            None
            if train_cfg.target_kl is None
            else float(
                train_cfg.target_kl
            )
        ),

        normalize_obs=bool(
            train_cfg.obs_norm
        ),
        normalize_adv=bool(
            train_cfg.adv_norm
        ),

        hidden_dims=tuple(
            int(x)
            for x
            in train_cfg.hidden_dims
        ),

        init_log_std=float(
            train_cfg.init_log_std
        ),

        use_value_huber_loss=bool(
            train_cfg
            .use_value_huber_loss
        ),
        use_value_clip=bool(
            train_cfg.use_value_clip
        ),
        value_clip_coef=float(
            train_cfg.value_clip_coef
        ),

        fail_on_nan=bool(
            train_cfg.fail_on_nan
        ),
        device=str(
            train_cfg.device
        ),
    )


def make_run_dir(train_cfg: FastTrainConfig, env_cfg: EnvConfig) -> Path:
    if train_cfg.run_name is None:
        run_name = (
            f"fast_ppo_{train_cfg.mode}"
            f"_ep{int(train_cfg.num_episodes)}"
            f"_slowT{int(env_cfg.slow_T)}"
        )
    else:
        run_name = str(train_cfg.run_name)

    output_root = Path(train_cfg.output_root)

    if not output_root.is_absolute():
        output_root = PROPOSED_ROOT / output_root

    run_dir = output_root / run_name

    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "figures")

    return run_dir


def get_episode_move_prob(
    train_cfg: FastTrainConfig,
    episode_idx: int,
) -> float:
    """
    1-based episode index에 해당하는 mobility probability를 반환한다.
    """
    episode_idx = int(episode_idx)

    selected_prob = float(
        train_cfg.mobility_curriculum[0][1]
    )

    for start_episode, move_prob in (
        train_cfg.mobility_curriculum
    ):
        if episode_idx < int(start_episode):
            break
        selected_prob = float(move_prob)

    return selected_prob


def sample_random_slow_action(
    env: Env,
    rng: np.random.Generator,
    train_cfg: FastTrainConfig,
) -> Dict[str, Any]:
    """
    Fast-only PPO 학습용 random slow-timescale action 생성.
    slow policy는 아직 학습하지 않고, PPO/config.py의 확률값으로 round마다 slow condition을 샘플링한다.
    """
    if train_cfg.slow_decision_mode != "random":
        raise ValueError("현재 fast-only 학습에서는 slow_decision_mode='random'만 지원합니다.")

    cfg = env.cfg

    m = int(cfg.num_rsu)
    n = int(cfg.num_user)
    u = int(cfg.num_uav)

    if m <= 0 or n <= 0 or u <= 0:
        raise ValueError(
            f"유효하지 않은 숫자입니다: num_rsu={m}, num_uav={u}, num_user={n}"
        )
    
    if m != u:
        raise ValueError(
            "현재 fast-only random slow action은 num_rsu == num_uav 가정을 사용합니다. "
            f"num_rsu={m}, num_uav={u}"
        )

    user_region = np.asarray(env.user_region, dtype=np.int32)

    if user_region.shape != (n,):
        raise ValueError(
            f"user_region shape mismatch: expected {(n,)}, got {user_region.shape}"
        )

    rsu_user_prob = float(train_cfg.random_rsu_user_prob)
    uav_hire_prob = float(train_cfg.random_uav_hire_prob)
    uav_user_prob = float(train_cfg.random_uav_user_prob)

    rsu_scheduling = np.zeros((m, n), dtype=np.int32)
    uav_hiring = np.zeros(u, dtype=np.int32)
    uav_scheduling = np.zeros((u, n), dtype=np.int32)

    # 1) RSU scheduling: user의 현재 region RSU만 후보로 사용
    for user_idx in range(n):
        region_idx = int(user_region[user_idx])
        if region_idx < 0 or region_idx >= m:
            continue

        if rng.random() < rsu_user_prob:
            rsu_scheduling[region_idx, user_idx] = 1

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

    # 2) UAV scheduling: UAV index == region index
    for uav_idx in range(u):
        if rng.random() >= uav_hire_prob:
            continue

        same_region_mask = user_region == uav_idx
        residual_mask = ~rsu_served_user
        cache_match_mask = requested_content == uav_cached_content[uav_idx]

        candidate_mask = same_region_mask & residual_mask & cache_match_mask
        candidate_users = np.flatnonzero(candidate_mask)

        if candidate_users.size == 0:
            continue

        selected_mask = rng.random(candidate_users.size) < uav_user_prob
        selected_users = candidate_users[selected_mask]

        if selected_users.size == 0:
            selected_users = np.asarray(
                [int(rng.choice(candidate_users))],
                dtype=np.int64,
            )

        if selected_users.size > uav_user_cap:
            selected_users = rng.choice(
                selected_users,
                size=uav_user_cap,
                replace=False,
            )

        uav_hiring[uav_idx] = 1
        uav_scheduling[uav_idx, np.asarray(selected_users, dtype=np.int64)] = 1

    # scheduling 없는 UAV는 hiring 제거
    active_hire = (uav_scheduling.sum(axis=1) > 0).astype(np.int32)
    uav_hiring = uav_hiring * active_hire
    uav_scheduling = uav_scheduling * uav_hiring[:, None]

    return {
        "rsu_scheduling": rsu_scheduling.astype(np.int32),
        "uav_hiring": uav_hiring.astype(np.int32),
        "uav_scheduling": uav_scheduling.astype(np.int32),
    }


def save_configs(
    run_dir: Path,
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
    ppo_cfg: AgentPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> None:
    save_json(train_cfg.to_dict(), run_dir / "train_config.json")
    save_json(env_cfg.as_dict() if hasattr(env_cfg, "as_dict") else asdict(env_cfg), run_dir / "env_config.json")
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
    train_cfg: FastTrainConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    fast-only episode reset.

    절차:
        1) env.reset()
        2) 현재 user_region/content 기준 random slow action 생성
        3) env.apply_slow_action()
        4) env.get_fast_obs() 반환

    반환 obs에는 raw slow scheduling matrix를 붙이지 않는다.
    get_fast_obs()가 이미 connection state를 제공하기 때문이다.
    """
    obs, info = split_env_reset(env.reset())

    slow_action = sample_random_slow_action(
        env=env,
        rng=slow_rng,
        train_cfg=train_cfg,
    )

    env.apply_slow_action(slow_action)

    obs = env.get_fast_obs()

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


def apply_random_slow_action_for_current_round(
    env: Env,
    slow_rng: np.random.Generator,
    train_cfg: FastTrainConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    round boundary에서 random slow-timescale decision을 새로 적용한다.

    현재 fast-only 학습에서는 slow policy를 아직 학습하지 않으므로,
    각 round마다 random slow condition을 샘플링한다.
    """
    slow_action = sample_random_slow_action(
        env=env,
        rng=slow_rng,
        train_cfg=train_cfg,
    )

    applied_slow_action = env.apply_slow_action(slow_action)
    obs = env.get_fast_obs()

    info = {
        "random_slow_action": {
            "rsu_scheduling": np.asarray(applied_slow_action.rsu_scheduling, dtype=np.int32).copy(),
            "uav_hiring": np.asarray(applied_slow_action.uav_hiring, dtype=np.int32).copy(),
            "uav_scheduling": np.asarray(applied_slow_action.uav_scheduling, dtype=np.int32).copy(),
        },
        "random_rsu_links": int(np.sum(applied_slow_action.rsu_scheduling)),
        "random_hired_uav": int(np.sum(applied_slow_action.uav_hiring)),
        "random_uav_links": int(np.sum(applied_slow_action.uav_scheduling)),
    }

    return obs, info


def _read_scalar_csv(csv_path: Path) -> Dict[str, list[float]]:
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
    figures_dir = run_dir / "figures"
    ensure_dir(figures_dir)

    episodes_csv = run_dir / "logs" / "episodes.csv"
    updates_csv = run_dir / "logs" / "updates.csv"

    ep_data = _read_scalar_csv(episodes_csv)
    update_data = _read_scalar_csv(updates_csv)

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

    component_keys = [
        "episode_delivery",
        "episode_quality",
        "episode_stall",
        "episode_consumed_soc",
        "episode_charged_soc",
        "episode_outage",
        "episode_min_soc",
        "episode_mean_soc",
        "episode_max_B",
        "episode_mean_B",
        "episode_charging_slots",
        "episode_outage_slots",
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


def extract_info_metrics(
    info: Dict[str, Any],
) -> Dict[str, float]:
    reward_components = info.get(
        "reward_components",
        {},
    )
    fast_components = reward_components.get(
        "fast_reward_components",
        {},
    )

    stall_arr = np.asarray(
        info.get("stall", []),
        dtype=np.float32,
    )
    playback_arr = np.asarray(
        info.get("playback", []),
        dtype=np.float32,
    )

    stall_sum = (
        float(np.sum(stall_arr))
        if stall_arr.size > 0
        else 0.0
    )

    prev_connection_state = info.get(
        "prev_connection_state",
        {},
    )
    connection_type = np.asarray(
        prev_connection_state.get(
            "connection_type",
            [],
        ),
        dtype=np.int32,
    )

    if (
        stall_arr.shape
        == playback_arr.shape
        == connection_type.shape
    ):
        scheduled_mask = (
            connection_type > 0
        )
        unscheduled_mask = (
            ~scheduled_mask
        )

        scheduled_stall = float(
            np.sum(
                stall_arr[scheduled_mask]
            )
        )
        scheduled_playback = float(
            np.sum(
                playback_arr[scheduled_mask]
            )
        )

        unscheduled_stall = float(
            np.sum(
                stall_arr[unscheduled_mask]
            )
        )
        unscheduled_playback = float(
            np.sum(
                playback_arr[unscheduled_mask]
            )
        )
    else:
        scheduled_stall = 0.0
        scheduled_playback = 0.0
        unscheduled_stall = 0.0
        unscheduled_playback = 0.0

    next_E = np.asarray(
        info.get("next_E", []),
        dtype=np.float32,
    )
    next_B = np.asarray(
        info.get("next_B", []),
        dtype=np.float32,
    )
    charging_state = np.asarray(
        info.get("charging_state", []),
        dtype=np.float32,
    )
    outage = np.asarray(
        info.get("outage", []),
        dtype=np.float32,
    )

    return {
        "sum_delivery": float(
            fast_components.get(
                "sum_delivery",
                0.0,
            )
        ),
        "sum_quality": float(
            fast_components.get(
                "sum_quality",
                0.0,
            )
        ),
        "sum_quality_degradation": float(
            fast_components.get(
                "sum_quality_degradation",
                0.0,
            )
        ),
        "quality_per_chunk": float(
            fast_components.get(
                "quality_per_chunk",
                0.0,
            )
        ),
        "quality_degradation_per_chunk": float(
            fast_components.get(
                "quality_degradation_per_chunk",
                0.0,
            )
        ),

        "video_delivery_term": float(
            fast_components.get(
                "video_delivery_term",
                0.0,
            )
        ),
        "battery_consume_term": float(
            fast_components.get(
                "battery_consume_term",
                0.0,
            )
        ),
        "battery_charge_term": float(
            fast_components.get(
                "battery_charge_term",
                0.0,
            )
        ),
        "quality_degradation_term": float(
            fast_components.get(
                "quality_degradation_term",
                0.0,
            )
        ),

        "delivery_term_share": float(
            fast_components.get(
                "delivery_term_share",
                0.0,
            )
        ),
        "battery_consume_term_share": float(
            fast_components.get(
                "battery_consume_term_share",
                0.0,
            )
        ),
        "battery_charge_term_share": float(
            fast_components.get(
                "battery_charge_term_share",
                0.0,
            )
        ),
        "quality_term_share": float(
            fast_components.get(
                "quality_term_share",
                0.0,
            )
        ),

        "sum_consumed_soc": float(
            fast_components.get(
                "sum_consumed_soc",
                0.0,
            )
        ),
        "sum_charged_soc": float(
            fast_components.get(
                "sum_charged_soc",
                0.0,
            )
        ),

        "num_hired_uav": float(
            fast_components.get(
                "num_hired_uav",
                0.0,
            )
        ),
        "num_charging_uav": float(
            fast_components.get(
                "num_charging_uav",
                0.0,
            )
        ),
        "num_outage_uav": float(
            fast_components.get(
                "num_outage_uav",
                0.0,
            )
        ),

        "stall_sum": stall_sum,
        "scheduled_stall": (
            scheduled_stall
        ),
        "scheduled_playback": (
            scheduled_playback
        ),
        "scheduled_stall_rate": (
            scheduled_stall
            / scheduled_playback
            if scheduled_playback > 0.0
            else 0.0
        ),
        "unscheduled_stall": (
            unscheduled_stall
        ),
        "unscheduled_playback": (
            unscheduled_playback
        ),
        "unscheduled_stall_rate": (
            unscheduled_stall
            / unscheduled_playback
            if unscheduled_playback > 0.0
            else 0.0
        ),

        "min_soc": (
            float(np.min(next_E))
            if next_E.size > 0
            else 0.0
        ),
        "mean_soc": (
            float(np.mean(next_E))
            if next_E.size > 0
            else 0.0
        ),
        "max_B": (
            float(np.max(next_B))
            if next_B.size > 0
            else 0.0
        ),
        "mean_B": (
            float(np.mean(next_B))
            if next_B.size > 0
            else 0.0
        ),
        "charging_slots": (
            float(np.sum(charging_state))
            if charging_state.size > 0
            else 0.0
        ),
        "outage_slots": (
            float(np.sum(outage))
            if outage.size > 0
            else 0.0
        ),
    }


def train(train_cfg: FastTrainConfig) -> None:
    set_seed(
        int(train_cfg.seed),
        deterministic=bool(train_cfg.deterministic_torch),
    )

    env_cfg = build_env_config()
    ppo_cfg = build_agent_ppo_config(train_cfg)

    slow_rng = np.random.default_rng(int(train_cfg.seed) + 10007)

    env = Env(env_cfg)

    obs, reset_info = reset_env(
        env=env,
        slow_rng=slow_rng,
        train_cfg=train_cfg,
    )

    obs_dim = infer_fast_obs_dim(obs)

    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if train_cfg.legacy_transfer:
        if train_cfg.checkpoint is None:
            raise ValueError(
                "legacy_transfer=True이면 "
                "checkpoint가 필요합니다."
            )

        transfer_info = (
            agent.load_legacy_transfer(
                train_cfg.checkpoint
            )
        )

        print(
            "[LEGACY TRANSFER] "
            f"critic tensors="
            f"{transfer_info['num_transferred_tensors']}",
            flush=True,
        )

    elif train_cfg.resume:
        if train_cfg.checkpoint is None:
            raise ValueError(
                "resume=True이면 "
                "checkpoint가 필요합니다."
            )

        # 새 mixed policy checkpoint만 resume 가능
        agent.load(
            path=train_cfg.checkpoint,
            strict=True,
            load_optimizer=True,
        )

    run_dir = make_run_dir(train_cfg, env_cfg)

    save_configs(
        run_dir=run_dir,
        train_cfg=train_cfg,
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

    while episode_idx < int(train_cfg.num_episodes):
        current_episode = episode_idx + 1

        current_move_prob = (
            get_episode_move_prob(
                train_cfg=train_cfg,
                episode_idx=current_episode,
            )
        )

        env.cfg.move_prob = float(
            current_move_prob
        )

        obs, reset_info = reset_env(
            env=env,
            slow_rng=slow_rng,
            train_cfg=train_cfg,
        )

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0
        ep_min_soc = float("inf")
        ep_mean_soc_sum = 0.0
        ep_max_B = 0.0
        ep_mean_B_sum = 0.0
        ep_charging_slots = 0.0
        ep_outage_slots = 0.0

        ep_quality_degradation = 0.0

        ep_video_delivery_term = 0.0
        ep_battery_consume_term = 0.0
        ep_battery_charge_term = 0.0
        ep_quality_degradation_term = 0.0

        ep_scheduled_stall = 0.0
        ep_scheduled_playback = 0.0
        ep_unscheduled_stall = 0.0
        ep_unscheduled_playback = 0.0

        ep_active_action_dims = 0.0
        ep_active_action_ratio = 0.0
        ep_action_saturation_ratio = 0.0

        ep_horizon = int(env_cfg.slow_T) * int(train_cfg.rounds_per_episode)

        ep_service_rate = 0.0
        ep_mean_requested_chunks = 0.0

        ep_layer_ratios = np.zeros(
            int(env_cfg.layer) + 1,
            dtype=np.float64,
        )

        for _ in range(ep_horizon):
            selected = agent.select_action(obs)

            ep_service_rate += float(
                selected["service_rate"]
            )

            ep_mean_requested_chunks += float(
                selected[
                    "mean_requested_chunks"
                ]
            )

            for layer_idx in range(
                1,
                int(env_cfg.layer) + 1,
            ):
                ep_layer_ratios[
                    layer_idx
                ] += float(
                    selected[
                        f"layer_{layer_idx}_ratio"
                    ]
                )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )

            is_round_boundary = bool(info.get("is_round_boundary", False))

            ep_done = bool(terminated or truncated or ep_steps + 1 >= ep_horizon)

            # reward scaling
            raw_reward = float(reward)
            ppo_reward = raw_reward * float(train_cfg.ppo_reward_scale)

            agent.store_transition(
                obs_vec=selected["obs_vec"],
                raw_action=selected["raw_action"],
                action_mask=selected["action_mask"],
                reward=ppo_reward,
                done=ep_done,
                value=float(selected["value"]),
                log_prob=float(selected["log_prob"]),
            )

            metrics = extract_info_metrics(info)

            ep_reward += raw_reward
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_min_soc = min(ep_min_soc, metrics["min_soc"])
            ep_mean_soc_sum += metrics["mean_soc"]
            ep_max_B = max(ep_max_B, metrics["max_B"])
            ep_mean_B_sum += metrics["mean_B"]
            ep_charging_slots += metrics["charging_slots"]
            ep_outage_slots += metrics["outage_slots"]

            ep_quality_degradation += (
                metrics["sum_quality_degradation"]
            )

            ep_video_delivery_term += (
                metrics["video_delivery_term"]
            )
            ep_battery_consume_term += (
                metrics["battery_consume_term"]
            )
            ep_battery_charge_term += (
                metrics["battery_charge_term"]
            )
            ep_quality_degradation_term += (
                metrics["quality_degradation_term"]
            )

            ep_scheduled_stall += (
                metrics["scheduled_stall"]
            )
            ep_scheduled_playback += (
                metrics["scheduled_playback"]
            )
            ep_unscheduled_stall += (
                metrics["unscheduled_stall"]
            )
            ep_unscheduled_playback += (
                metrics["unscheduled_playback"]
            )

            ep_active_action_dims += float(
                selected["active_action_dims"]
            )
            ep_active_action_ratio += float(
                selected["active_action_ratio"]
            )
            ep_action_saturation_ratio += float(
                selected["action_saturation_ratio"]
            )

            ep_steps += 1
            global_slot += 1

            next_obs = next_obs_raw

            if is_round_boundary and not ep_done:
                next_obs, slow_info = apply_random_slow_action_for_current_round(
                    env=env,
                    slow_rng=slow_rng,
                    train_cfg=train_cfg,
                )
                info["next_random_slow_action"] = slow_info 

            last_obs = next_obs
            last_done = ep_done

            if agent.buffer.is_full:
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
                        "episode": episode_idx + 1,
                        **update_logs,
                    }
                )

                print(
                    "[UPDATE] "
                    f"update={update_idx} "
                    f"slot={global_slot} "
                    f"episode={episode_idx + 1} "
                    f"policy_loss={update_logs['policy_loss']:.6f} "
                    f"value_loss={update_logs['value_loss']:.6f} "
                    f"entropy={update_logs['entropy']:.6f} "
                    f"kl={update_logs['approx_kl']:.6f} "
                    f"clipfrac={update_logs['clipfrac']:.4f} "
                    f"ev={update_logs['explained_variance']:.4f}",
                    flush=True,
                )

            obs = next_obs

            if ep_done:
                break

        episode_idx += 1

        ep_quality_per_chunk = (
            ep_quality / ep_delivery
            if ep_delivery > 0.0
            else 0.0
        )

        ep_quality_degradation_per_chunk = (
            ep_quality_degradation
            / ep_delivery
            if ep_delivery > 0.0
            else 0.0
        )

        ep_scheduled_stall_rate = (
            ep_scheduled_stall
            / ep_scheduled_playback
            if ep_scheduled_playback > 0.0
            else 0.0
        )

        ep_unscheduled_stall_rate = (
            ep_unscheduled_stall
            / ep_unscheduled_playback
            if ep_unscheduled_playback > 0.0
            else 0.0
        )

        ep_active_action_dims_mean = (
            ep_active_action_dims
            / max(ep_steps, 1)
        )

        ep_active_action_ratio_mean = (
            ep_active_action_ratio
            / max(ep_steps, 1)
        )

        ep_action_saturation_ratio_mean = (
            ep_action_saturation_ratio
            / max(ep_steps, 1)
        )

        ep_service_rate /= max(
            ep_steps,
            1,
        )

        ep_mean_requested_chunks /= max(
            ep_steps,
            1,
        )

        ep_layer_ratios /= float(
            max(ep_steps, 1)
        )

        if ep_steps > 0:
            ep_mean_soc = ep_mean_soc_sum / float(ep_steps)
            ep_mean_B = ep_mean_B_sum / float(ep_steps)
        else:
            ep_mean_soc = 0.0
            ep_mean_B = 0.0

        if not np.isfinite(ep_min_soc):
            ep_min_soc = 0.0

        episode_logger.write(
            {
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
                "episode_min_soc": ep_min_soc,
                "episode_mean_soc": ep_mean_soc,
                "episode_max_B": ep_max_B,
                "episode_mean_B": ep_mean_B,
                "episode_charging_slots": ep_charging_slots,
                "episode_outage_slots": ep_outage_slots,
                "move_prob": float(current_move_prob),
                "episode_quality_per_chunk":
                    ep_quality_per_chunk,

                "episode_quality_degradation":
                    ep_quality_degradation,

                "episode_quality_degradation_per_chunk":
                    ep_quality_degradation_per_chunk,

                "episode_scheduled_stall_rate":
                    ep_scheduled_stall_rate,

                "episode_unscheduled_stall_rate":
                    ep_unscheduled_stall_rate,

                "episode_video_delivery_term":
                    ep_video_delivery_term,

                "episode_battery_consume_term":
                    ep_battery_consume_term,

                "episode_battery_charge_term":
                    ep_battery_charge_term,

                "episode_quality_degradation_term":
                    ep_quality_degradation_term,

                "episode_active_action_dims_mean":
                    ep_active_action_dims_mean,

                "episode_active_action_ratio_mean":
                    ep_active_action_ratio_mean,

                "episode_action_saturation_ratio_mean":
                    ep_action_saturation_ratio_mean,
    
                "episode_service_rate":
                    ep_service_rate,

                "episode_mean_requested_chunks":
                    ep_mean_requested_chunks,

                "episode_layer_1_ratio":
                    float(ep_layer_ratios[1]),

                "episode_layer_2_ratio":
                    float(ep_layer_ratios[2]),

                "episode_layer_3_ratio":
                    float(ep_layer_ratios[3]),

                "episode_layer_4_ratio":
                    float(ep_layer_ratios[4]),
            }
        )

        print(
            "[EPISODE] "
            f"ep={episode_idx}/{train_cfg.num_episodes} "
            f"steps={ep_steps} "
            f"reward={ep_reward:.4f} "
            f"delivery={ep_delivery:.4f} "
            f"quality={ep_quality:.4f} "
            f"stall={ep_stall:.4f}",
            f"min_soc={ep_min_soc:.2f} ",
            f"charging_slots={ep_charging_slots:.0f} ",
            f"outage_slots={ep_outage_slots:.0f}",
            flush=True,
        )

        if (
            int(train_cfg.plot_every_episodes) > 0
            and episode_idx % int(train_cfg.plot_every_episodes) == 0
        ):
            save_training_plots(
                run_dir=run_dir,
                smooth_window=int(train_cfg.plot_smooth_window),
            )
            print(f"[PLOT] saved figures to {run_dir / 'figures'}", flush=True)

        if (
            int(train_cfg.save_every_episodes) > 0
            and episode_idx % int(train_cfg.save_every_episodes) == 0
        ):
            ckpt_path = run_dir / "checkpoints" / f"fast_ppo_ep{episode_idx}.pt"
            agent.save(
                ckpt_path,
                extra={
                    "episode": episode_idx,
                    "global_slot": global_slot,
                    "update_idx": update_idx,
                    "env_config": env_cfg.as_dict() if hasattr(env_cfg, "as_dict") else asdict(env_cfg),
                    "train_config": train_cfg.to_dict(),
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
            "env_config": env_cfg.as_dict() if hasattr(env_cfg, "as_dict") else asdict(env_cfg),
            "train_config": train_cfg.to_dict(),
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
        smooth_window=int(train_cfg.plot_smooth_window),
    )

    print("=" * 100, flush=True)
    print("[FAST PPO TRAIN DONE]", flush=True)
    print(f"final checkpoint: {final_ckpt_path}", flush=True)
    print("=" * 100, flush=True)


def evaluate(train_cfg: FastTrainConfig) -> None:
    set_seed(int(train_cfg.seed), deterministic=True)

    env_cfg = build_env_config()

    target_move_prob = float(
        train_cfg.mobility_curriculum[-1][1]
    )
    env_cfg.move_prob = target_move_prob

    ppo_cfg = build_agent_ppo_config(train_cfg)

    slow_rng = np.random.default_rng(int(train_cfg.seed) + 20011)

    env = Env(env_cfg)

    obs, reset_info = reset_env(
        env=env,
        slow_rng=slow_rng,
        train_cfg=train_cfg,
    )

    obs_dim = infer_fast_obs_dim(obs)

    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if train_cfg.checkpoint is None:
        raise ValueError(
            "eval mode에서는 checkpoint가 반드시 필요합니다."
        )

    if train_cfg.checkpoint is not None:
        agent.load(
            path=train_cfg.checkpoint,
            strict=True,
            load_optimizer=False,
        )

    run_dir = make_run_dir(train_cfg, env_cfg)

    save_configs(
        run_dir=run_dir,
        train_cfg=train_cfg,
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
    print(f"checkpoint    : {train_cfg.checkpoint}", flush=True)
    print(f"reset_info    : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    rewards: list[float] = []
    deliveries: list[float] = []
    qualities: list[float] = []
    stalls: list[float] = []

    for episode_idx in range(1, int(train_cfg.eval_episodes) + 1):
        obs, reset_info = reset_env(
            env=env,
            slow_rng=slow_rng,
            train_cfg=train_cfg,
        )

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0
        
        ep_min_soc = float("inf")
        ep_mean_soc_sum = 0.0
        ep_max_B = 0.0
        ep_mean_B_sum = 0.0
        ep_charging_slots = 0.0
        ep_outage_slots = 0.0

        ep_quality_degradation = 0.0

        ep_video_delivery_term = 0.0
        ep_battery_consume_term = 0.0
        ep_battery_charge_term = 0.0
        ep_quality_degradation_term = 0.0

        ep_scheduled_stall = 0.0
        ep_scheduled_playback = 0.0
        ep_unscheduled_stall = 0.0
        ep_unscheduled_playback = 0.0

        ep_active_action_dims = 0.0
        ep_active_action_ratio = 0.0
        ep_action_saturation_ratio = 0.0

        ep_horizon = int(env_cfg.slow_T) * int(train_cfg.eval_rounds_per_episode)
        ep_service_rate = 0.0
        ep_mean_requested_chunks = 0.0

        ep_layer_ratios = np.zeros(
            int(env_cfg.layer) + 1,
            dtype=np.float64,
        )

        for _ in range(ep_horizon):
            selected = agent.select_action(
                obs=obs,
                deterministic=True,
                update_norm=False,
            )
            ep_service_rate += float(
                selected["service_rate"]
            )

            ep_mean_requested_chunks += float(
                selected[
                    "mean_requested_chunks"
                ]
            )

            for layer_idx in range(
                1,
                int(env_cfg.layer) + 1,
            ):
                ep_layer_ratios[
                    layer_idx
                ] += float(
                    selected[
                        f"layer_{layer_idx}_ratio"
                    ]
                )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )

            is_round_boundary = bool(info.get("is_round_boundary", False))

            ep_done = bool(
                terminated
                or truncated
                or ep_steps + 1 >= ep_horizon
            )

            next_obs = next_obs_raw

            if is_round_boundary and not ep_done:
                next_obs, slow_info = apply_random_slow_action_for_current_round(
                    env=env,
                    slow_rng=slow_rng,
                    train_cfg=train_cfg,
                )
                info["next_random_slow_action"] = slow_info

            metrics = extract_info_metrics(info)

            ep_reward += float(reward)
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_min_soc = min(ep_min_soc, metrics["min_soc"])
            ep_mean_soc_sum += metrics["mean_soc"]
            ep_max_B = max(ep_max_B, metrics["max_B"])
            ep_mean_B_sum += metrics["mean_B"]
            ep_charging_slots += metrics["charging_slots"]
            ep_outage_slots += metrics["outage_slots"]

            ep_quality_degradation += (
                metrics["sum_quality_degradation"]
            )

            ep_video_delivery_term += (
                metrics["video_delivery_term"]
            )
            ep_battery_consume_term += (
                metrics["battery_consume_term"]
            )
            ep_battery_charge_term += (
                metrics["battery_charge_term"]
            )
            ep_quality_degradation_term += (
                metrics["quality_degradation_term"]
            )

            ep_scheduled_stall += (
                metrics["scheduled_stall"]
            )
            ep_scheduled_playback += (
                metrics["scheduled_playback"]
            )
            ep_unscheduled_stall += (
                metrics["unscheduled_stall"]
            )
            ep_unscheduled_playback += (
                metrics["unscheduled_playback"]
            )

            ep_active_action_dims += float(
                selected["active_action_dims"]
            )
            ep_active_action_ratio += float(
                selected["active_action_ratio"]
            )
            ep_action_saturation_ratio += float(
                selected["action_saturation_ratio"]
            )

            ep_steps += 1
            obs = next_obs

            if ep_done:
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
            f"ep={episode_idx}/{train_cfg.eval_episodes} "
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
        "checkpoint": train_cfg.checkpoint,
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
    train_cfg = get_fast_ppo_config()

    if train_cfg.mode == "train":
        train(train_cfg)
    elif train_cfg.mode == "eval":
        evaluate(train_cfg)
    else:
        raise ValueError(f"지원하지 않는 mode입니다: {train_cfg.mode}")


if __name__ == "__main__":
    main()