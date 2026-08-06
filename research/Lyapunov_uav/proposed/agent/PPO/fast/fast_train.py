from __future__ import annotations

import copy
import csv
import ctypes
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_proposed_root(start: Optional[Path] = None) -> Path:
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
        "proposed root를 찾지 못했습니다. fast_train.py의 위치를 확인하세요."
    )


PROPOSED_ROOT = _find_proposed_root()
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import EnvConfig
from env.action_types import SlowAction
from env.env import Env
from env.validators import (
    parse_slow_action,
    validate_slow_action_strict,
)

from agent.PPO.config import FastTrainConfig, get_fast_ppo_config
from agent.PPO.common import (
    FAST_OBS_KEYS,
    ensure_dir,
    infer_fast_obs_dim,
    set_seed,
    split_env_reset,
    split_env_step,
    flatten_fast_obs
)
from agent.PPO.common.utils import ScalarLogger, save_json
from agent.PPO.fast.fast_agent import (
    FastPPOAgent,
    FastPPOConfig as AgentPPOConfig,
)
from agent.PPO.fast.fast_action import FastActionCodec


# ======================================================================
# Configuration / common utilities
# ======================================================================
def build_env_config(
    train_cfg: Optional[FastTrainConfig] = None,
) -> EnvConfig:
    """
    System/environment parameter source remains proposed/config.py.

    Battery round-feasibility horizon was 5 while slow_T was 3600 in the
    original branch. The joint trainer aligns them without introducing a
    separate configuration file.
    """
    cfg = EnvConfig()
    if train_cfg is not None:
        cfg.seed = int(train_cfg.seed)
    cfg.battery.target_service_slots_per_round = int(cfg.slow_T)
    return cfg


def build_agent_ppo_config(train_cfg: FastTrainConfig) -> AgentPPOConfig:
    return AgentPPOConfig(
        rollout_steps=int(train_cfg.rollout_slots),
        update_epochs=int(train_cfg.update_epochs),
        batch_size=int(train_cfg.batch_size),
        gamma=float(train_cfg.gamma),
        gae_lambda=float(train_cfg.gae_lambda),
        actor_lr=float(train_cfg.actor_lr),
        critic_lr=float(train_cfg.critic_lr),
        max_grad_norm=float(train_cfg.max_grad_norm),
        clip_coef=float(train_cfg.clip_coef),
        value_coef=float(train_cfg.value_coef),
        categorical_entropy_coef=float(
            train_cfg.categorical_entropy_coef
        ),
        power_entropy_coef=float(train_cfg.power_entropy_coef),
        target_kl=(
            None
            if train_cfg.target_kl is None
            else float(train_cfg.target_kl)
        ),
        normalize_obs=bool(train_cfg.obs_norm),
        normalize_adv=bool(train_cfg.adv_norm),
        hidden_dims=tuple(int(x) for x in train_cfg.hidden_dims),
        init_log_std=float(train_cfg.init_log_std),
        use_value_huber_loss=bool(train_cfg.use_value_huber_loss),
        use_value_clip=bool(train_cfg.use_value_clip),
        value_clip_coef=float(train_cfg.value_clip_coef),
        fail_on_nan=bool(train_cfg.fail_on_nan),
        device=str(train_cfg.device),
    )


def _resolve_checkpoint(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    result = Path(os.path.expandvars(os.path.expanduser(path)))
    if not result.is_absolute():
        result = PROPOSED_ROOT / result
    return result.resolve()


def _trusted_checkpoint_payload(path: Path) -> Dict[str, Any]:
    """Read a checkpoint produced by this repository for metadata checks."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary.")
    return payload


def _resume_signature(
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
    ppo_cfg: AgentPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> str:
    """Hash every setting that changes the learned process or optimizer."""
    train_payload = train_cfg.to_dict()
    for mutable_key in (
        "checkpoint",
        "resume",
        "segment_id",
        "num_episodes",
        "target_total_episodes",
        "output_root",
        "run_name",
        "allow_existing_run_dir",
        "save_every_episodes",
        "save_latest_every_episode",
        "plot_every_episodes",
        "plot_smooth_window",
        "eval_episodes",
        "eval_rounds_per_episode",
    ):
        train_payload.pop(mutable_key, None)

    ppo_payload = asdict(ppo_cfg)
    # cuda and cuda:0 are equivalent for a one-GPU allocation and do not
    # change optimizer/model semantics.
    ppo_payload.pop("device", None)

    payload = {
        "schema": "fast_pretrain_resume_v2",
        "policy_type": "conditional_mixed_categorical_gaussian_v1",
        "fast_obs_keys": list(FAST_OBS_KEYS),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "train": train_payload,
        "environment": env_cfg.as_dict(),
        "ppo": ppo_payload,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_resume_checkpoint(
    checkpoint_path: Path,
    expected_signature: str,
    train_cfg: FastTrainConfig,
) -> None:
    payload = _trusted_checkpoint_payload(checkpoint_path)
    extra = payload.get("extra", {})
    actual_signature = extra.get("resume_signature")
    if actual_signature is None:
        raise RuntimeError(
            "Checkpoint predates the strict actor/critic-LR resume schema. "
            "Start the H1 run from scratch; do not resume an H0 checkpoint."
        )
    if str(actual_signature) != str(expected_signature):
        raise RuntimeError(
            "Resume signature mismatch. Model/optimizer/environment settings "
            "differ from the checkpoint; start a new run instead."
        )

    checkpoint_train_cfg = extra.get("train_config", {})
    previous_run_name = checkpoint_train_cfg.get("run_name")
    if previous_run_name != train_cfg.run_name:
        raise RuntimeError(
            "Resume run_name mismatch: "
            f"checkpoint={previous_run_name!r}, requested={train_cfg.run_name!r}."
        )


def make_run_dir(
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
) -> Path:
    if train_cfg.run_name is None:
        run_name = (
            f"{train_cfg.slow_decision_mode}_fastppo_"
            f"{train_cfg.mode}_ep{train_cfg.num_episodes}_"
            f"slowT{env_cfg.slow_T}"
        )
    else:
        run_name = str(train_cfg.run_name)

    output_root = Path(train_cfg.output_root)
    if not output_root.is_absolute():
        output_root = PROPOSED_ROOT / output_root

    run_dir = output_root / run_name
    for path in (
        run_dir,
        run_dir / "checkpoints",
        run_dir / "logs",
        run_dir / "figures",
    ):
        ensure_dir(path)
    return run_dir


def _run_dir_has_files(run_dir: Path) -> bool:
    return any(path.is_file() for path in run_dir.rglob("*"))


def _validate_run_dir_contract(
    run_dir: Path,
    train_cfg: FastTrainConfig,
) -> None:
    has_files = _run_dir_has_files(run_dir)
    if train_cfg.resume:
        if not (run_dir / "logs" / "episodes.csv").is_file():
            raise RuntimeError(
                "Resume requires the original run directory and episodes.csv. "
                f"Missing under: {run_dir}"
            )
        return

    if has_files and not bool(train_cfg.allow_existing_run_dir):
        raise RuntimeError(
            "Run directory already contains files. Use a new "
            "FAST_PPO_RUN_NAME instead of mixing experiments: "
            f"{run_dir}"
        )


def save_configs(
    run_dir: Path,
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
    ppo_cfg: AgentPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> None:
    env_payload = (
        env_cfg.as_dict() if hasattr(env_cfg, "as_dict") else asdict(env_cfg)
    )
    segment_config_dir = (
        run_dir / "configs" / f"segment_{int(train_cfg.segment_id):02d}"
    )
    ensure_dir(segment_config_dir)
    save_json(train_cfg.to_dict(), segment_config_dir / "train_config.json")
    save_json(env_payload, segment_config_dir / "env_config.json")
    save_json(asdict(ppo_cfg), segment_config_dir / "ppo_config.json")

    # Preserve the original experiment contract at the run root. Segment
    # resumes get their own immutable config snapshot above.
    root_configs = {
        "train_config.json": train_cfg.to_dict(),
        "env_config.json": env_payload,
        "ppo_config.json": asdict(ppo_cfg),
    }
    for filename, payload in root_configs.items():
        root_path = run_dir / filename
        if not root_path.exists():
            save_json(payload, root_path)
    if train_cfg.resume:
        initialization = "resumed_checkpoint"
    elif train_cfg.checkpoint is not None:
        initialization = "pretrained_checkpoint"
    else:
        initialization = "from_scratch"

    if train_cfg.slow_decision_mode == "random":
        algorithm = (
            "Fast-PPO pretraining under feasible random slow actions"
        )
        slow_solver = "random feasible baseline"
    else:
        algorithm = (
            "round-wise Slow-DPP forecast using the current frozen "
            "Fast-PPO policy -> one real Fast round -> one PPO update"
        )
        slow_solver = (
            "parallel Jacobi complete-action region-coordinate "
            "minimization with exact full-round acceptance"
        )

    run_info = {
        "proposed_root": str(PROPOSED_ROOT),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initialization": initialization,
        "phase": train_cfg.phase,
        "segment_id": int(train_cfg.segment_id),
        "resume": bool(train_cfg.resume),
        "checkpoint": (
            None
            if train_cfg.checkpoint is None
            else str(_resolve_checkpoint(train_cfg.checkpoint))
        ),
        "algorithm": algorithm,
        "slow_solver": slow_solver,
        "global_optimum_guaranteed": False,
    }
    save_json(run_info, segment_config_dir / "run_info.json")
    root_run_info = run_dir / "run_info.json"
    if not root_run_info.exists():
        save_json(run_info, root_run_info)


def get_episode_move_prob(
    train_cfg: FastTrainConfig,
    episode_idx: int,
) -> float:
    selected = float(train_cfg.mobility_curriculum[0][1])
    for start_episode, move_prob in train_cfg.mobility_curriculum:
        if int(episode_idx) < int(start_episode):
            break
        selected = float(move_prob)
    return selected


def _assert_joint_config(
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
) -> None:
    if int(train_cfg.rollout_slots) != int(env_cfg.slow_T):
        raise ValueError(
            "Round-wise Fast-PPO requires rollout_slots == env.slow_T: "
            f"{train_cfg.rollout_slots} != {env_cfg.slow_T}."
        )
    if not bool(train_cfg.freeze_obs_norm_within_round):
        raise ValueError(
            "Round-wise training requires observation normalizer fixed "
            "within every round."
        )

    if train_cfg.slow_decision_mode != "dpp":
        return

    if int(train_cfg.dpp_forecast_horizon) != int(env_cfg.slow_T):
        raise ValueError(
            "Joint mode requires dpp_forecast_horizon == env.slow_T: "
            f"{train_cfg.dpp_forecast_horizon} != {env_cfg.slow_T}."
        )


# ======================================================================
# Formulation-v6 reward/cost mapping
# ======================================================================
def formulation_reward_from_env_step(
    *,
    env_reward: float,
    info: Dict[str, Any],
    env_cfg: EnvConfig,
) -> Tuple[float, float]:
    """
    Env._compute_fast_reward()가 반환하는 env_reward는 이미 현재
    formulation의 정확한 Fast reward이다.

        R^F(t)
        = alpha_Z * sum_n Z_n(t)[d_n(t)-b_n(t)]
          - alpha_B * sum_u B_u(t)e_u(t)
          + alpha_B * sum_u B_u(t)e_u^c(t)
          - V * sum_n[P_bar*d_n(t)-q_n(t)]

    따라서 trainer에서 playback term을 다시 보정하면 안 된다.

    Forecast 경량 경로에서는 info["one_slot_dpp_cost"]를 사용하고,
    일반 경로에서는 reward_components 내부 값을 사용한다.
    두 값이 없을 경우 -env_reward를 사용한다.
    """
    del env_cfg

    formulation_reward = float(env_reward)

    reward_components = info.get("reward_components", {})
    fast_components = reward_components.get(
        "fast_reward_components",
        {},
    )

    stored_cost = info.get(
        "one_slot_dpp_cost",
        fast_components.get(
            "one_slot_dpp_cost",
            -formulation_reward,
        ),
    )
    slot_dpp_cost = float(stored_cost)

    if not np.isfinite(formulation_reward):
        raise RuntimeError(
            "Formulation reward is NaN or Inf: "
            f"{formulation_reward}"
        )
    if not np.isfinite(slot_dpp_cost):
        raise RuntimeError(
            "One-slot DPP cost is NaN or Inf: "
            f"{slot_dpp_cost}"
        )

    if not np.isclose(
        slot_dpp_cost,
        -formulation_reward,
        rtol=1e-6,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Fast reward and one-slot DPP cost disagree: "
            f"reward={formulation_reward}, "
            f"cost={slot_dpp_cost}"
        )

    return formulation_reward, slot_dpp_cost


def _hiring_cost(
    env: Env,
    slow_action: Dict[str, np.ndarray],
) -> float:
    mu = np.asarray(slow_action["uav_hiring"], dtype=np.float64)
    raw = np.asarray(env._hire_cost(), dtype=np.float64)  # existing Env API
    if raw.shape != mu.shape:
        raise ValueError(
            f"hiring cost shape mismatch: mu={mu.shape}, cost={raw.shape}"
        )
    return float(getattr(env.cfg, "hire_weight", 1.0)) * float(
        np.dot(mu, raw)
    )


# ======================================================================
# Strict feasible random action (Fast pretraining baseline)
# ======================================================================
def sample_random_slow_action(
    env: Env,
    rng: np.random.Generator,
    train_cfg: FastTrainConfig,
) -> Dict[str, np.ndarray]:
    if train_cfg.slow_decision_mode != "random":
        raise ValueError("sample_random_slow_action is for random mode only.")

    cfg = env.cfg
    m, n, u = int(cfg.num_rsu), int(cfg.num_user), int(cfg.num_uav)
    if m != u:
        raise ValueError("One-UAV-per-region requires num_rsu == num_uav.")

    region = np.asarray(env.user_region, dtype=np.int32)
    requested = np.asarray(env.requested_content, dtype=np.int32)
    cached = np.asarray(env.uav_cached_content, dtype=np.int32)

    y = np.zeros((m, n), dtype=np.int32)
    mu = np.zeros(u, dtype=np.int32)
    phi = np.zeros((u, n), dtype=np.int32)

    # RSU scheduling with hard capacity.
    for mm in range(m):
        users = np.flatnonzero(region == mm)
        sampled = users[
            rng.random(users.size) < float(train_cfg.random_rsu_user_prob)
        ]
        if sampled.size > int(cfg.rsu_capacity):
            sampled = rng.choice(
                sampled,
                size=int(cfg.rsu_capacity),
                replace=False,
            )
        y[mm, np.asarray(sampled, dtype=np.int64)] = 1

    rsu_users = y.sum(axis=0) > 0
    for uu in range(u):
        if rng.random() >= float(train_cfg.random_uav_hire_prob):
            continue
        candidates = np.flatnonzero(
            (region == uu)
            & (~rsu_users)
            & (requested == cached[uu])
        )
        if candidates.size == 0:
            continue
        selected = candidates[
            rng.random(candidates.size)
            < float(train_cfg.random_uav_user_prob)
        ]
        if selected.size == 0:
            selected = np.asarray([rng.choice(candidates)], dtype=np.int64)
        if selected.size > int(cfg.uav_user_cap):
            selected = rng.choice(
                selected,
                size=int(cfg.uav_user_cap),
                replace=False,
            )
        phi[uu, np.asarray(selected, dtype=np.int64)] = 1
        mu[uu] = 1

    action = {
        "rsu_scheduling": y,
        "uav_hiring": mu,
        "uav_scheduling": phi,
    }
    parsed = parse_slow_action(action, cfg)
    validate_slow_action_strict(
        parsed,
        cfg,
        user_region=region,
        requested_content=requested,
        uav_cached_content=cached,
        forbid_empty_hiring=True,
    )
    return action


# ======================================================================
# Slow-DPP complete-action candidate construction
# ======================================================================
@dataclass(frozen=True)
class RegionSlowCandidate:
    rsu_users: Tuple[int, ...]
    uav_users: Tuple[int, ...]


@dataclass
class SlowSelectionResult:
    action: Dict[str, np.ndarray]
    predicted_round_cost: float
    solver_mode: str
    estimator_mode: str
    global_optimum_guaranteed: bool
    coordinate_sweeps: int
    candidate_requests: int
    unique_candidates: int
    finite_candidates: int
    forecast_seconds: float
    policy_seconds: float
    env_seconds: float
    mean_gpu_batch: float
    forecast_trial_steps: int


def _region_candidates(
    env: Env,
    region_idx: int,
    train_cfg: FastTrainConfig,
) -> list[RegionSlowCandidate]:
    cfg = env.cfg
    users = tuple(
        int(x)
        for x in np.flatnonzero(
            np.asarray(env.user_region, dtype=np.int32) == int(region_idx)
        )
    )
    requested = np.asarray(env.requested_content, dtype=np.int32)
    cached = np.asarray(env.uav_cached_content, dtype=np.int32)

    result: list[RegionSlowCandidate] = []
    rsu_cap = min(int(cfg.rsu_capacity), len(users))

    for rsu_size in range(rsu_cap + 1):
        for rsu_subset in itertools.combinations(users, rsu_size):
            rsu_set = set(rsu_subset)
            eligible_uav = tuple(
                nn
                for nn in users
                if nn not in rsu_set
                and int(requested[nn]) == int(cached[region_idx])
            )
            uav_cap = min(int(cfg.uav_user_cap), len(eligible_uav))
            for uav_size in range(uav_cap + 1):
                for uav_subset in itertools.combinations(
                    eligible_uav, uav_size
                ):
                    result.append(
                        RegionSlowCandidate(
                            rsu_users=tuple(int(x) for x in rsu_subset),
                            uav_users=tuple(int(x) for x in uav_subset),
                        )
                    )
                    if len(result) > int(
                        train_cfg.dpp_max_region_candidates
                    ):
                        raise RuntimeError(
                            "Region candidate count exceeds the configured "
                            "exact-enumeration limit. Do not silently truncate "
                            "the feasible set. "
                            f"region={region_idx}, limit="
                            f"{train_cfg.dpp_max_region_candidates}."
                        )
    return result


def _zero_slow_action(env: Env) -> Dict[str, np.ndarray]:
    return {
        "rsu_scheduling": np.zeros(
            (env.num_rsu, env.num_user), dtype=np.int32
        ),
        "uav_hiring": np.zeros(env.num_uav, dtype=np.int32),
        "uav_scheduling": np.zeros(
            (env.num_uav, env.num_user), dtype=np.int32
        ),
    }


def _replace_region_candidate(
    base_action: Dict[str, np.ndarray],
    region_idx: int,
    local: RegionSlowCandidate,
) -> Dict[str, np.ndarray]:
    action = {
        key: np.asarray(value).copy()
        for key, value in base_action.items()
    }
    action["rsu_scheduling"][region_idx, :] = 0
    action["uav_scheduling"][region_idx, :] = 0
    action["uav_hiring"][region_idx] = 0

    if local.rsu_users:
        action["rsu_scheduling"][
            region_idx, np.asarray(local.rsu_users, dtype=np.int64)
        ] = 1
    if local.uav_users:
        action["uav_hiring"][region_idx] = 1
        action["uav_scheduling"][
            region_idx, np.asarray(local.uav_users, dtype=np.int64)
        ] = 1
    return action


def _validate_candidate(
    env: Env,
    action: Dict[str, np.ndarray],
    train_cfg: FastTrainConfig,
) -> SlowAction:
    parsed = parse_slow_action(action, env.cfg)
    validate_slow_action_strict(
        parsed,
        env.cfg,
        user_region=np.asarray(env.user_region, dtype=np.int32),
        requested_content=np.asarray(
            env.requested_content, dtype=np.int32
        ),
        uav_cached_content=np.asarray(
            env.uav_cached_content, dtype=np.int32
        ),
        forbid_empty_hiring=bool(train_cfg.dpp_forbid_empty_hiring),
    )
    return parsed


def _candidate_key(action: Dict[str, np.ndarray]) -> bytes:
    return b"|".join(
        np.asarray(action[key], dtype=np.int8).tobytes()
        for key in (
            "rsu_scheduling",
            "uav_hiring",
            "uav_scheduling",
        )
    )


def _env_state_digest(env: Env) -> str:
    payload = (
        int(env.t),
        int(env.episode),
        int(env.round_idx),
        int(env.round_slot),
        np.asarray(env.queue).copy(),
        np.asarray(env.user_region).copy(),
        np.asarray(env.rsu_scheduling).copy(),
        np.asarray(env.uav_hiring).copy(),
        np.asarray(env.uav_scheduling).copy(),
        np.asarray(env.requested_content).copy(),
        np.asarray(env.uav_cached_content).copy(),
        np.asarray(env.E).copy(),
        np.asarray(env.Y).copy(),
        np.asarray(env.outage).copy(),
        np.asarray(env.charging_state).copy(),
        copy.deepcopy(env.rng.bit_generator.state),
    )
    return hashlib.sha256(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _model_parameter_digest(agent: FastPPOAgent) -> str:
    hasher = hashlib.sha256()
    with torch.no_grad():
        for tensor in agent.model.state_dict().values():
            hasher.update(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            )
    return hasher.hexdigest()


def _forecast_scenario_seed(
    env: Env,
    train_cfg: FastTrainConfig,
    scenario_idx: int,
) -> int:
    return int(
        int(train_cfg.seed)
        + 1_000_003 * int(env.episode)
        + 10_007 * int(env.round_idx)
        + 101 * int(scenario_idx)
    )


def _install_mean_rsu_channel(shadow_env: Env) -> None:
    """Deterministic E[pathloss * lognormal shadowing * Rayleigh power]."""

    def mean_compute_gain(
        channel_self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        del rng
        d = channel_self.effective_distance(distance)
        pathloss = float(d ** (-float(channel_self.beta)))
        log10_factor = math.log(10.0) / 10.0
        mu_db = float(channel_self.shadowing_mu_db)
        sigma_db = float(channel_self.shadowing_sigma_db)
        shadowing_mean = math.exp(
            log10_factor * mu_db
            + 0.5 * (log10_factor * sigma_db) ** 2
        )
        # E[|CN(0,1)|^2] = 1.
        return float(pathloss * shadowing_mean)

    shadow_env.rsu_channel.compute_gain = types.MethodType(
        mean_compute_gain,
        shadow_env.rsu_channel,
    )


def _make_shadow_env(
    base_env: Env,
    action: Dict[str, np.ndarray],
    train_cfg: FastTrainConfig,
    scenario_idx: int,
) -> Env:
    shadow = copy.deepcopy(base_env)
    shadow.rng = np.random.default_rng(
        _forecast_scenario_seed(base_env, train_cfg, scenario_idx)
    )
    if bool(train_cfg.dpp_use_mean_rsu_channel):
        _install_mean_rsu_channel(shadow)

    shadow.apply_slow_action(action)
    return shadow


def _forecast_worker_loop(
    connection,
    worker_idx: int,
    obs_raw,
    mask_raw,
    action_raw,
    cost_raw,
    invalid_raw,
    batch_capacity: int,
    obs_dim: int,
    action_dim: int,
) -> None:
    """
    Shared-memory 기반 Slow-DPP shadow worker.

    대형 observation/mask/action/cost 배열은 Pipe로 전송하지 않는다.
    Pipe는 command와 completion signal에만 사용한다.
    """
    obs_matrix = np.frombuffer(
        obs_raw,
        dtype=np.float32,
        count=int(batch_capacity) * int(obs_dim),
    ).reshape(int(batch_capacity), int(obs_dim))

    mask_matrix = np.frombuffer(
        mask_raw,
        dtype=np.uint8,
        count=int(batch_capacity) * int(action_dim),
    ).reshape(int(batch_capacity), int(action_dim))

    action_matrix = np.frombuffer(
        action_raw,
        dtype=np.float32,
        count=int(batch_capacity) * int(action_dim),
    ).reshape(int(batch_capacity), int(action_dim))

    cost_vector = np.frombuffer(
        cost_raw,
        dtype=np.float64,
        count=int(batch_capacity),
    )

    invalid_vector = np.frombuffer(
        invalid_raw,
        dtype=np.uint8,
        count=int(batch_capacity),
    )

    base_env: Optional[Env] = None
    train_cfg: Optional[FastTrainConfig] = None
    codec: Optional[FastActionCodec] = None

    shadows: list[Env] = []
    current_observations: list[Dict[str, Any]] = []

    loaded_start = 0
    loaded_end = 0

    try:
        while True:
            message = connection.recv()
            if not isinstance(message, tuple) or len(message) == 0:
                raise RuntimeError(
                    f"Malformed worker command: {message!r}"
                )

            command = str(message[0])
            payload = message[1] if len(message) > 1 else None

            if command == "close":
                connection.send(
                    ("closed", int(worker_idx))
                )
                break

            if command == "set_round_context":
                base_env, train_cfg = payload
                codec = FastActionCodec(base_env.cfg)

                if int(codec.action_dim) != int(action_dim):
                    raise RuntimeError(
                        "Worker action_dim mismatch: "
                        f"codec={codec.action_dim}, "
                        f"shared={action_dim}"
                    )

                shadows = []
                current_observations = []
                loaded_start = 0
                loaded_end = 0

                connection.send(
                    ("context_ready", int(worker_idx))
                )
                continue

            if command == "load_batch":
                if (
                    base_env is None
                    or train_cfg is None
                    or codec is None
                ):
                    raise RuntimeError(
                        "load_batch received before "
                        "set_round_context."
                    )

                (
                    actions,
                    scenario_idx,
                    start,
                    end,
                ) = payload

                actions = list(actions)
                start = int(start)
                end = int(end)

                if start < 0 or end > int(batch_capacity):
                    raise ValueError(
                        "Worker shared-memory slice is invalid: "
                        f"start={start}, end={end}, "
                        f"capacity={batch_capacity}"
                    )
                if end - start != len(actions):
                    raise ValueError(
                        "Worker action/slice count mismatch: "
                        f"actions={len(actions)}, "
                        f"slice={end - start}"
                    )
                if len(actions) == 0:
                    raise ValueError(
                        "Active worker received an empty batch."
                    )

                shadows = [
                    _make_shadow_env(
                        base_env=base_env,
                        action=action,
                        train_cfg=train_cfg,
                        scenario_idx=int(scenario_idx),
                    )
                    for action in actions
                ]

                current_observations = [
                    shadow.get_fast_obs()
                    for shadow in shadows
                ]

                loaded_start = start
                loaded_end = end

                for local_idx, observation in enumerate(
                    current_observations
                ):
                    global_idx = start + local_idx

                    flattened = flatten_fast_obs(
                        observation
                    ).astype(
                        np.float32,
                        copy=False,
                    )

                    if flattened.shape != (int(obs_dim),):
                        raise ValueError(
                            "Forecast observation dimension "
                            "mismatch: "
                            f"expected={(obs_dim,)}, "
                            f"got={flattened.shape}"
                        )

                    obs_matrix[global_idx] = flattened

                    mask = (
                        codec.build_base_action_mask(
                            observation
                        )
                        .astype(
                            np.uint8,
                            copy=False,
                        )
                    )
                    if mask.shape != (int(action_dim),):
                        raise ValueError(
                            "Forecast action-mask dimension "
                            "mismatch: "
                            f"expected={(action_dim,)}, "
                            f"got={mask.shape}"
                        )

                    mask_matrix[global_idx] = mask
                    cost_vector[global_idx] = 0.0
                    invalid_vector[global_idx] = 0

                connection.send(
                    (
                        "batch_ready",
                        int(worker_idx),
                        start,
                        end,
                    )
                )
                continue

            if command == "step":
                if (
                    train_cfg is None
                    or codec is None
                    or not shadows
                ):
                    raise RuntimeError(
                        "step received before a batch was loaded."
                    )

                slot_idx, start, end = payload
                slot_idx = int(slot_idx)
                start = int(start)
                end = int(end)

                if (
                    start != loaded_start
                    or end != loaded_end
                ):
                    raise RuntimeError(
                        "Worker step slice differs from loaded "
                        "batch: "
                        f"loaded=({loaded_start},{loaded_end}), "
                        f"received=({start},{end})"
                    )

                expected_boundary = (
                    slot_idx
                    == int(
                        train_cfg.dpp_forecast_horizon
                    )
                    - 1
                )

                for local_idx, shadow in enumerate(shadows):
                    global_idx = start + local_idx
                    current_obs = current_observations[
                        local_idx
                    ]

                    env_action = codec.decode(
                        action_matrix[global_idx],
                        current_obs,
                    )

                    (
                        next_obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = split_env_step(
                        shadow.step(
                            env_action,
                            info_level="forecast",
                        )
                    )

                    actual_boundary = bool(
                        info.get(
                            "is_round_boundary",
                            False,
                        )
                    )
                    if actual_boundary != expected_boundary:
                        raise RuntimeError(
                            "Forecast boundary mismatch: "
                            f"worker={worker_idx}, "
                            f"slot={slot_idx}, "
                            f"expected={expected_boundary}, "
                            f"actual={actual_boundary}"
                        )

                    slot_cost = float(
                        info.get(
                            "one_slot_dpp_cost",
                            -float(reward),
                        )
                    )
                    if not np.isfinite(slot_cost):
                        raise RuntimeError(
                            "Forecast slot cost is nonfinite: "
                            f"worker={worker_idx}, "
                            f"candidate={global_idx}, "
                            f"slot={slot_idx}, "
                            f"cost={slot_cost}"
                        )

                    has_outage = bool(
                        info.get(
                            "has_outage",
                            np.any(
                                np.asarray(
                                    info.get("outage", []),
                                    dtype=np.int32,
                                )
                                > 0
                            ),
                        )
                    )

                    invalid = bool(
                        terminated
                        or truncated
                        or (
                            bool(
                                train_cfg
                                .dpp_reject_forecast_outage
                            )
                            and has_outage
                        )
                    )

                    current_observations[
                        local_idx
                    ] = next_obs

                    flattened = flatten_fast_obs(
                        next_obs
                    ).astype(
                        np.float32,
                        copy=False,
                    )
                    obs_matrix[global_idx] = flattened

                    mask_matrix[global_idx] = (
                        codec.build_base_action_mask(
                            next_obs
                        )
                        .astype(
                            np.uint8,
                            copy=False,
                        )
                    )

                    cost_vector[global_idx] = slot_cost
                    invalid_vector[global_idx] = int(
                        invalid
                    )

                connection.send(
                    (
                        "step_done",
                        int(worker_idx),
                        slot_idx,
                    )
                )
                continue

            raise RuntimeError(
                f"Unknown forecast command: {command!r}"
            )

    except EOFError:
        return

    except BaseException as exc:
        try:
            connection.send(
                (
                    "error",
                    int(worker_idx),
                    type(exc).__name__,
                    str(exc),
                )
            )
        except BaseException:
            pass
        raise

    finally:
        connection.close()


class _PersistentShadowProcessPool:
    """
    학습 전체에서 한 번 생성되는 shared-memory shadow pool.
    """

    def __init__(
        self,
        *,
        worker_count: int,
        batch_capacity: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        self.worker_count = max(
            1,
            int(worker_count),
        )
        self.batch_capacity = int(batch_capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        if self.batch_capacity <= 0:
            raise ValueError(
                "batch_capacity must be positive."
            )
        if self.obs_dim <= 0:
            raise ValueError(
                "obs_dim must be positive."
            )
        if self.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive."
            )

        self.context = mp.get_context("spawn")

        self._obs_raw = self.context.RawArray(
            ctypes.c_float,
            self.batch_capacity * self.obs_dim,
        )
        self._mask_raw = self.context.RawArray(
            ctypes.c_ubyte,
            self.batch_capacity * self.action_dim,
        )
        self._action_raw = self.context.RawArray(
            ctypes.c_float,
            self.batch_capacity * self.action_dim,
        )
        self._cost_raw = self.context.RawArray(
            ctypes.c_double,
            self.batch_capacity,
        )
        self._invalid_raw = self.context.RawArray(
            ctypes.c_ubyte,
            self.batch_capacity,
        )

        self.obs_matrix = np.frombuffer(
            self._obs_raw,
            dtype=np.float32,
        ).reshape(
            self.batch_capacity,
            self.obs_dim,
        )
        self.mask_matrix = np.frombuffer(
            self._mask_raw,
            dtype=np.uint8,
        ).reshape(
            self.batch_capacity,
            self.action_dim,
        )
        self.action_matrix = np.frombuffer(
            self._action_raw,
            dtype=np.float32,
        ).reshape(
            self.batch_capacity,
            self.action_dim,
        )
        self.cost_vector = np.frombuffer(
            self._cost_raw,
            dtype=np.float64,
        )
        self.invalid_vector = np.frombuffer(
            self._invalid_raw,
            dtype=np.uint8,
        )

        self.parents = []
        self.processes = []
        self._closed = False

        for worker_idx in range(self.worker_count):
            parent_conn, child_conn = (
                self.context.Pipe(
                    duplex=True
                )
            )

            process = self.context.Process(
                target=_forecast_worker_loop,
                args=(
                    child_conn,
                    int(worker_idx),
                    self._obs_raw,
                    self._mask_raw,
                    self._action_raw,
                    self._cost_raw,
                    self._invalid_raw,
                    self.batch_capacity,
                    self.obs_dim,
                    self.action_dim,
                ),
                name=f"dpp-shadow-{worker_idx}",
                daemon=True,
            )
            process.start()
            child_conn.close()

            self.parents.append(parent_conn)
            self.processes.append(process)

    @staticmethod
    def _partition_sizes(
        count: int,
        workers: int,
    ) -> list[int]:
        base, remainder = divmod(
            int(count),
            int(workers),
        )
        return [
            base + (
                1
                if worker_idx < remainder
                else 0
            )
            for worker_idx in range(int(workers))
        ]

    @staticmethod
    def _check_message(
        message,
        expected: str,
    ):
        if (
            not isinstance(message, tuple)
            or len(message) == 0
        ):
            raise RuntimeError(
                f"Malformed worker message: {message!r}"
            )

        if message[0] == "error":
            raise RuntimeError(
                "Forecast worker failed: "
                f"worker={message[1]}, "
                f"{message[2]}: {message[3]}"
            )

        if message[0] != expected:
            raise RuntimeError(
                "Unexpected worker message: "
                f"expected={expected!r}, "
                f"got={message[0]!r}, "
                f"message={message!r}"
            )

        return message

    def set_round_context(
        self,
        *,
        base_env: Env,
        train_cfg: FastTrainConfig,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "Forecast pool is already closed."
            )

        for parent in self.parents:
            parent.send(
                (
                    "set_round_context",
                    (
                        base_env,
                        train_cfg,
                    ),
                )
            )

        for parent in self.parents:
            message = parent.recv()
            self._check_message(
                message,
                "context_ready",
            )

    def rollout(
        self,
        *,
        actions: Sequence[
            Dict[str, np.ndarray]
        ],
        scenario_idx: int,
        train_cfg: FastTrainConfig,
        agent: FastPPOAgent,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
        list[int],
    ]:
        count = len(actions)

        if count == 0:
            return (
                np.zeros(0, dtype=np.float64),
                0.0,
                0.0,
                [],
            )

        if count > self.batch_capacity:
            raise ValueError(
                "Forecast batch exceeds shared-memory "
                "capacity: "
                f"count={count}, "
                f"capacity={self.batch_capacity}"
            )

        active_workers = min(
            self.worker_count,
            count,
        )

        sizes = self._partition_sizes(
            count,
            active_workers,
        )

        offsets = [0]
        for size in sizes:
            offsets.append(
                offsets[-1] + int(size)
            )

        self.cost_vector[:count] = 0.0
        self.invalid_vector[:count] = 0

        for worker_idx in range(active_workers):
            start = offsets[worker_idx]
            end = offsets[worker_idx + 1]

            self.parents[worker_idx].send(
                (
                    "load_batch",
                    (
                        list(actions[start:end]),
                        int(scenario_idx),
                        start,
                        end,
                    ),
                )
            )

        for worker_idx in range(active_workers):
            message = self.parents[
                worker_idx
            ].recv()
            self._check_message(
                message,
                "batch_ready",
            )

        accumulated = np.zeros(
            count,
            dtype=np.float64,
        )
        invalid = np.zeros(
            count,
            dtype=bool,
        )

        policy_seconds = 0.0
        env_seconds = 0.0
        gpu_batch_sizes: list[int] = []

        for slot_idx in range(
            int(
                train_cfg
                .dpp_forecast_horizon
            )
        ):
            policy_start = time.perf_counter()

            raw_actions = (
                agent
                .select_env_actions_from_matrices_batch(
                    obs_matrix=self.obs_matrix[
                        :count
                    ],
                    mask_matrix=self.mask_matrix[
                        :count
                    ],
                    deterministic=bool(
                        train_cfg
                        .dpp_deterministic_fast_forecast
                    ),
                )
            )

            np.copyto(
                self.action_matrix[:count],
                raw_actions,
                casting="no",
            )

            policy_seconds += (
                time.perf_counter()
                - policy_start
            )
            gpu_batch_sizes.append(count)

            env_start = time.perf_counter()

            for worker_idx in range(
                active_workers
            ):
                start = offsets[worker_idx]
                end = offsets[worker_idx + 1]

                self.parents[worker_idx].send(
                    (
                        "step",
                        (
                            int(slot_idx),
                            start,
                            end,
                        ),
                    )
                )

            for worker_idx in range(
                active_workers
            ):
                message = self.parents[
                    worker_idx
                ].recv()
                self._check_message(
                    message,
                    "step_done",
                )

            accumulated += self.cost_vector[
                :count
            ]
            invalid |= self.invalid_vector[
                :count
            ].astype(
                bool,
                copy=False,
            )

            env_seconds += (
                time.perf_counter()
                - env_start
            )

        accumulated[invalid] = np.inf

        return (
            accumulated,
            float(policy_seconds),
            float(env_seconds),
            gpu_batch_sizes,
        )

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        for parent, process in zip(
            self.parents,
            self.processes,
        ):
            if not process.is_alive():
                continue

            try:
                parent.send(
                    ("close", None)
                )
            except (
                BrokenPipeError,
                EOFError,
                OSError,
            ):
                pass

        for parent, process in zip(
            self.parents,
            self.processes,
        ):
            if process.is_alive():
                try:
                    if parent.poll(3.0):
                        message = parent.recv()
                        self._check_message(
                            message,
                            "closed",
                        )
                except (
                    BrokenPipeError,
                    EOFError,
                    OSError,
                    RuntimeError,
                ):
                    pass

            try:
                parent.close()
            except OSError:
                pass

        for process in self.processes:
            process.join(timeout=5.0)

            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


class SlowDPPEvaluator:
    def __init__(
        self,
        env: Env,
        agent: FastPPOAgent,
        train_cfg: FastTrainConfig,
        process_pool: _PersistentShadowProcessPool,
    ) -> None:
        self.env = env
        self.agent = agent
        self.train_cfg = train_cfg
        self.process_pool = process_pool

        self.cache: Dict[bytes, float] = {}
        self.candidate_requests = 0
        self.finite_candidates = 0
        self.policy_seconds = 0.0
        self.env_seconds = 0.0
        self.gpu_batch_sizes: list[int] = []

        self.process_pool.set_round_context(
            base_env=env,
            train_cfg=train_cfg,
        )

    def close(self) -> None:
        # Pool 소유권은 trainer에 있으므로 여기서 종료하지 않는다.
        return

    def evaluate(
        self,
        actions: Sequence[Dict[str, np.ndarray]],
    ) -> list[float]:
        self.candidate_requests += len(actions)
        scores: list[Optional[float]] = [None] * len(actions)
        missing_indices: list[int] = []
        missing_actions: list[Dict[str, np.ndarray]] = []

        for idx, action in enumerate(actions):
            _validate_candidate(self.env, action, self.train_cfg)
            key = _candidate_key(action)
            if key in self.cache:
                scores[idx] = self.cache[key]
            else:
                missing_indices.append(idx)
                missing_actions.append(action)

        batch_limit = int(self.train_cfg.dpp_candidate_batch_size)

        for start in range(0, len(missing_actions), batch_limit):
            batch_actions = missing_actions[start : start + batch_limit]
            batch_scores = self._evaluate_uncached_batch(batch_actions)

            for local_idx, score in enumerate(batch_scores):
                global_missing_idx = start + local_idx
                original_idx = missing_indices[global_missing_idx]

                scores[original_idx] = float(score)
                key = _candidate_key(batch_actions[local_idx])
                self.cache[key] = float(score)

                if np.isfinite(score):
                    self.finite_candidates += 1

        if any(score is None for score in scores):
            raise RuntimeError("Internal DPP score assignment failure.")

        return [float(score) for score in scores]

    def _evaluate_uncached_batch(
        self,
        actions: Sequence[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        count = len(actions)
        scenario_costs = np.zeros(
            (int(self.train_cfg.dpp_forecast_scenarios), count),
            dtype=np.float64,
        )

        for scenario_idx in range(int(self.train_cfg.dpp_forecast_scenarios)):
            (
                accumulated,
                policy_seconds,
                env_seconds,
                batch_sizes,
            ) = self.process_pool.rollout(
                actions=actions,
                scenario_idx=scenario_idx,
                train_cfg=self.train_cfg,
                agent=self.agent,
            )

            scenario_costs[scenario_idx] = accumulated
            self.policy_seconds += policy_seconds
            self.env_seconds += env_seconds
            self.gpu_batch_sizes.extend(batch_sizes)

        mean_fast_cost = np.mean(scenario_costs, axis=0)

        for idx, action in enumerate(actions):
            if np.isfinite(mean_fast_cost[idx]):
                mean_fast_cost[idx] += _hiring_cost(self.env, action)

        return mean_fast_cost

def select_slow_action_dpp(
    env: Env,
    agent: FastPPOAgent,
    train_cfg: FastTrainConfig,
    process_pool: _PersistentShadowProcessPool,
) -> SlowSelectionResult:
    if int(env.round_slot) != 0:
        raise RuntimeError(
            "Slow-DPP must run at round boundary, "
            f"got {env.round_slot}."
        )

    audit_runtime = bool(
        train_cfg.audit_runtime_invariants
    )

    env_digest_before = (
        _env_state_digest(env)
        if audit_runtime
        else None
    )

    model_digest_before = (
        _model_parameter_digest(agent)
        if audit_runtime
        else None
    )

    normalizer_state_before = (
        copy.deepcopy(
            agent.obs_normalizer.state_dict()
        )
        if (
            audit_runtime
            and agent.obs_normalizer is not None
        )
        else None
    )

    wall_start = time.perf_counter()

    agent.model.eval()

    evaluator = SlowDPPEvaluator(
        env=env,
        agent=agent,
        train_cfg=train_cfg,
        process_pool=process_pool,
    )

    try:
        current_action = _zero_slow_action(env)
        current_score = float("inf")

        completed_sweeps = 0

        for sweep_idx in range(
            int(train_cfg.dpp_coordinate_sweeps)
        ):
            # 현재 complete action과 모든 one-region complete-action
            # neighbor를 한 번의 evaluator 호출로 묶는다.
            #
            # 기존 코드:
            #   region별 evaluator.evaluate()를 순차 호출
            #
            # 개선 코드:
            #   한 sweep의 모든 region 후보를 모아 evaluator.evaluate()
            #   한 번으로 처리
            trial_actions: list[
                Dict[str, np.ndarray]
            ] = [current_action]

            region_blocks: list[
                Tuple[
                    int,
                    int,
                    int,
                    list[RegionSlowCandidate],
                ]
            ] = []

            for region_idx in range(
                int(env.num_rsu)
            ):
                local_candidates = (
                    _region_candidates(
                        env,
                        region_idx,
                        train_cfg,
                    )
                )

                block_start = len(trial_actions)

                trial_actions.extend(
                    _replace_region_candidate(
                        current_action,
                        region_idx,
                        local,
                    )
                    for local in local_candidates
                )

                block_end = len(trial_actions)

                region_blocks.append(
                    (
                        region_idx,
                        block_start,
                        block_end,
                        local_candidates,
                    )
                )

            trial_scores = np.asarray(
                evaluator.evaluate(
                    trial_actions
                ),
                dtype=np.float64,
            )

            current_score = float(
                trial_scores[0]
            )

            if not np.isfinite(current_score):
                raise RuntimeError(
                    "Current complete slow action "
                    "produced nonfinite cost."
                )

            # 같은 current action을 기준으로 각 region의 최선 candidate를
            # 구한다. 이 단계는 parallel/Jacobi proposal이다.
            combined_action = {
                key: np.asarray(value).copy()
                for key, value
                in current_action.items()
            }

            proposed_any = False

            # combined action이 cross-region interaction 때문에 악화될
            # 경우를 대비하여, 이미 full-round 평가된 one-region 후보 중
            # 전체 최선 후보도 저장한다.
            best_single_action = current_action
            best_single_score = current_score

            for (
                region_idx,
                block_start,
                block_end,
                local_candidates,
            ) in region_blocks:
                block_scores = trial_scores[
                    block_start:block_end
                ]

                finite = np.flatnonzero(
                    np.isfinite(block_scores)
                )

                if finite.size == 0:
                    continue

                local_idx = int(
                    finite[
                        np.argmin(
                            block_scores[finite]
                        )
                    ]
                )

                score = float(
                    block_scores[local_idx]
                )

                action_idx = (
                    block_start + local_idx
                )

                if score < best_single_score:
                    best_single_score = score
                    best_single_action = (
                        trial_actions[action_idx]
                    )

                if (
                    score
                    < current_score
                    - float(
                        train_cfg
                        .dpp_improvement_tolerance
                    )
                ):
                    combined_action = (
                        _replace_region_candidate(
                            combined_action,
                            region_idx,
                            local_candidates[
                                local_idx
                            ],
                        )
                    )
                    proposed_any = True

            if not proposed_any:
                completed_sweeps = (
                    sweep_idx + 1
                )
                break

            # 여러 region의 변경을 합친 complete action은 반드시 다시
            # full 3600-slot DPP 평가를 수행한다.
            combined_score = float(
                evaluator.evaluate(
                    [combined_action]
                )[0]
            )

            if (
                np.isfinite(combined_score)
                and combined_score
                < current_score
                - float(
                    train_cfg
                    .dpp_improvement_tolerance
                )
            ):
                current_action = combined_action
                current_score = combined_score

            elif (
                np.isfinite(best_single_score)
                and best_single_score
                < current_score
                - float(
                    train_cfg
                    .dpp_improvement_tolerance
                )
            ):
                # combined action은 악화됐지만 one-region neighbor가
                # 개선된 경우, 이미 exact full-round 평가된 최선의
                # one-region action을 채택한다.
                current_action = (
                    best_single_action
                )
                current_score = (
                    best_single_score
                )

            else:
                completed_sweeps = (
                    sweep_idx + 1
                )
                break

            completed_sweeps = (
                sweep_idx + 1
            )

    finally:
        evaluator.close()

    _validate_candidate(
        env,
        current_action,
        train_cfg,
    )

    wall_seconds = (
        time.perf_counter() - wall_start
    )

    if (
        env_digest_before is not None
        and _env_state_digest(env)
        != env_digest_before
    ):
        raise RuntimeError(
            "Slow forecast mutated "
            "the real environment."
        )

    if (
        model_digest_before is not None
        and _model_parameter_digest(agent)
        != model_digest_before
    ):
        raise RuntimeError(
            "Slow forecast mutated "
            "Fast policy parameters."
        )

    if normalizer_state_before is not None:
        after = agent.obs_normalizer.state_dict()

        if (
            pickle.dumps(after)
            != pickle.dumps(
                normalizer_state_before
            )
        ):
            raise RuntimeError(
                "Slow forecast mutated "
                "observation normalizer."
            )

    mean_batch = (
        float(
            np.mean(
                evaluator.gpu_batch_sizes
            )
        )
        if evaluator.gpu_batch_sizes
        else 0.0
    )

    return SlowSelectionResult(
        action=current_action,
        predicted_round_cost=float(
            current_score
        ),
        solver_mode=(
            "parallel_jacobi_coordinate_descent_"
            "full_action_rollout"
        ),
        estimator_mode=(
            "single_scenario_mean_rsu_channel_proxy"
            if (
                int(train_cfg.dpp_forecast_scenarios) == 1
                and bool(train_cfg.dpp_use_mean_rsu_channel)
            )
            else "sample_average_rollout"
        ),
        global_optimum_guaranteed=False,
        coordinate_sweeps=int(
            completed_sweeps
        ),
        candidate_requests=int(
            evaluator.candidate_requests
        ),
        unique_candidates=int(
            len(evaluator.cache)
        ),
        finite_candidates=int(
            evaluator.finite_candidates
        ),
        forecast_seconds=float(
            wall_seconds
        ),
        policy_seconds=float(
            evaluator.policy_seconds
        ),
        env_seconds=float(
            evaluator.env_seconds
        ),
        mean_gpu_batch=mean_batch,
        forecast_trial_steps=int(
            len(evaluator.cache)
            * int(train_cfg.dpp_forecast_scenarios)
            * int(train_cfg.dpp_forecast_horizon)
        ),
    )

# ======================================================================
# Metrics / plots
# ======================================================================
def extract_info_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    reward_components = info.get("reward_components", {})
    fast = reward_components.get("fast_reward_components", {})

    stall = np.asarray(info.get("stall", []), dtype=np.float64)
    next_e = np.asarray(info.get("next_E", []), dtype=np.float64)
    outage = np.asarray(info.get("outage", []), dtype=np.float64)
    charging = np.asarray(info.get("charging_state", []), dtype=np.float64)

    scheduled = np.zeros(stall.shape, dtype=bool)
    rsu_scheduling = np.asarray(
        info.get("rsu_scheduling", []), dtype=np.int32
    )
    uav_scheduling = np.asarray(
        info.get("uav_scheduling", []), dtype=np.int32
    )
    if stall.ndim == 1:
        if rsu_scheduling.ndim == 2 and rsu_scheduling.shape[1] == stall.size:
            scheduled |= np.any(rsu_scheduling > 0, axis=0)
        if uav_scheduling.ndim == 2 and uav_scheduling.shape[1] == stall.size:
            scheduled |= np.any(uav_scheduling > 0, axis=0)
    unscheduled = ~scheduled

    return {
        "delivery": float(fast.get("sum_delivery", 0.0)),
        "quality": float(fast.get("sum_quality", 0.0)),
        "quality_degradation": float(
            fast.get("sum_quality_degradation", 0.0)
        ),
        "consumed_soc": float(fast.get("sum_consumed_soc", 0.0)),
        "charged_soc": float(fast.get("sum_charged_soc", 0.0)),
        "stall": float(np.sum(stall)) if stall.size else 0.0,
        "scheduled_stall": (
            float(np.sum(stall[scheduled])) if stall.size else 0.0
        ),
        "unscheduled_stall": (
            float(np.sum(stall[unscheduled])) if stall.size else 0.0
        ),
        "scheduled_user_slots": float(np.sum(scheduled)),
        "unscheduled_user_slots": float(np.sum(unscheduled)),
        "min_soc": float(np.min(next_e)) if next_e.size else 0.0,
        "outage_slots": float(np.sum(outage)) if outage.size else 0.0,
        "charging_slots": (
            float(np.sum(charging)) if charging.size else 0.0
        ),
        "queue_playback_term": float(
            fast.get("queue_playback_term", 0.0)
        ),
        "video_delivery_term": float(
            fast.get("video_delivery_term", 0.0)
        ),
        "battery_consume_term": float(
            fast.get("battery_consume_term", 0.0)
        ),
        "battery_charge_term": float(
            fast.get("battery_charge_term", 0.0)
        ),
        "quality_degradation_term": float(
            fast.get("quality_degradation_term", 0.0)
        ),
    }


def _count_executed_layers(
    env_action: Dict[str, Any],
    max_layer: int,
) -> np.ndarray:
    counts = np.zeros(int(max_layer), dtype=np.int64)
    for prefix in ("rsu", "uav"):
        chunks = np.asarray(env_action[f"{prefix}_chunks"], dtype=np.int32)
        layers = np.asarray(env_action[f"{prefix}_layers"], dtype=np.int32)
        active_layers = layers[chunks > 0]
        for layer in range(1, int(max_layer) + 1):
            counts[layer - 1] += int(np.sum(active_layers == layer))
    return counts


def _finalize_round_metrics(
    totals: Dict[str, float],
    layer_counts: np.ndarray,
) -> None:
    delivery = float(totals.get("delivery", 0.0))
    totals["quality_per_chunk"] = (
        float(totals.get("quality", 0.0)) / delivery
        if delivery > 0.0
        else 0.0
    )
    totals["quality_degradation_per_chunk"] = (
        float(totals.get("quality_degradation", 0.0)) / delivery
        if delivery > 0.0
        else 0.0
    )
    scheduled_slots = float(totals.get("scheduled_user_slots", 0.0))
    unscheduled_slots = float(totals.get("unscheduled_user_slots", 0.0))
    totals["scheduled_stall_rate"] = (
        float(totals.get("scheduled_stall", 0.0)) / scheduled_slots
        if scheduled_slots > 0.0
        else 0.0
    )
    totals["unscheduled_stall_rate"] = (
        float(totals.get("unscheduled_stall", 0.0)) / unscheduled_slots
        if unscheduled_slots > 0.0
        else 0.0
    )
    if "round_fast_cost" in totals:
        fast_cost = float(totals["round_fast_cost"])
    elif "reward" in totals:
        fast_cost = -float(totals["reward"])
    else:
        fast_cost = -float(totals.get("formulation_reward", 0.0))
    totals["fast_cost"] = fast_cost
    totals["fast_cost_per_scheduled_user_slot"] = (
        fast_cost / scheduled_slots
        if scheduled_slots > 0.0
        else float("inf")
    )
    totals["delivery_per_scheduled_user_slot"] = (
        delivery / scheduled_slots
        if scheduled_slots > 0.0
        else 0.0
    )

    active_layer_actions = int(np.sum(layer_counts))
    totals["active_layer_actions"] = float(active_layer_actions)
    for layer, count in enumerate(layer_counts, start=1):
        totals[f"layer_{layer}_ratio"] = (
            float(count) / float(active_layer_actions)
            if active_layer_actions > 0
            else 0.0
        )


def _read_scalar_csv(path: Path) -> Dict[str, list[float]]:
    result: Dict[str, list[float]] = {}
    if not path.exists():
        return result
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                try:
                    result.setdefault(str(key), []).append(float(value))
                except (TypeError, ValueError):
                    pass
    return result


def save_training_plots(run_dir: Path, smooth_window: int = 5) -> None:
    """
    Save workload-aware Fast-PPO diagnostics.

    Raw episode reward is retained only as a workload-confounded reference.
    Policy selection must use deterministic checkpoint evaluation, not these
    training plots.
    """
    episodes = _read_scalar_csv(run_dir / "logs" / "episodes.csv")
    updates = _read_scalar_csv(run_dir / "logs" / "updates.csv")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    window = max(int(smooth_window), 1)

    def _series(
        data: Dict[str, list[float]],
        key: str,
    ) -> Optional[np.ndarray]:
        values = data.get(key)
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        return array if array.size > 0 else None

    def _rolling(values: np.ndarray) -> np.ndarray:
        if values.size == 0 or window <= 1:
            return values.copy()
        result = np.full(values.shape, np.nan, dtype=np.float64)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        result[window - 1 :] = np.convolve(
            values,
            kernel,
            mode="valid",
        )
        return result

    def _plot_with_rolling(
        axis: Any,
        x_values: np.ndarray,
        values: np.ndarray,
        label: str,
        *,
        color: Optional[str] = None,
    ) -> None:
        axis.plot(
            x_values,
            values,
            alpha=0.22,
            linewidth=0.8,
            color=color,
        )
        axis.plot(
            x_values,
            _rolling(values),
            linewidth=1.8,
            label=f"{label} ({window}-episode mean)",
            color=color,
        )

    episode_x = _series(episodes, "episode")
    if episode_x is not None:
        reward = _series(episodes, "episode_formulation_reward")
        active_dims = _series(episodes, "episode_active_action_dims")
        scheduled_stall_rate = _series(
            episodes,
            "episode_scheduled_stall_rate",
        )
        service_rate = _series(episodes, "episode_service_rate")
        requested_chunks = _series(
            episodes,
            "episode_requested_chunks",
        )
        quality_per_chunk = _series(
            episodes,
            "episode_quality_per_chunk",
        )
        degradation_per_chunk = _series(
            episodes,
            "episode_quality_degradation_per_chunk",
        )
        normalized_cost = _series(
            episodes,
            "episode_fast_cost_per_scheduled_user_slot",
        )

        if reward is not None and active_dims is not None:
            fast_cost = -reward
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            _plot_with_rolling(
                axes[0],
                episode_x,
                fast_cost,
                "Fast cost",
            )
            axes[0].set_title(
                "Raw Fast cost (confounded by Random Slow workload)"
            )
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Fast cost")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].scatter(
                active_dims,
                fast_cost,
                s=12,
                alpha=0.55,
            )
            if active_dims.size >= 2 and np.ptp(active_dims) > 0.0:
                slope, intercept = np.polyfit(
                    active_dims,
                    fast_cost,
                    deg=1,
                )
                order = np.argsort(active_dims)
                axes[1].plot(
                    active_dims[order],
                    intercept + slope * active_dims[order],
                    linewidth=1.5,
                    color="tab:red",
                    label="OLS trend",
                )
                axes[1].legend()
            axes[1].set_title("Workload confounding check")
            axes[1].set_xlabel("Mean active action dimensions")
            axes[1].set_ylabel("Fast cost")
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                figures_dir / "episode_workload_and_cost.png",
                dpi=200,
            )
            plt.close(fig)

        available_rate_series = [
            (
                normalized_cost,
                "Fast cost / scheduled-user-slot",
            ),
            (scheduled_stall_rate, "Scheduled stall rate"),
            (service_rate, "Service rate"),
            (requested_chunks, "Requested chunks"),
        ]
        available_rate_series = [
            item for item in available_rate_series if item[0] is not None
        ]
        if available_rate_series:
            fig, axes = plt.subplots(
                len(available_rate_series),
                1,
                figsize=(10, 2.8 * len(available_rate_series)),
                sharex=True,
            )
            axes_array = np.atleast_1d(axes)
            for axis, (values, label) in zip(
                axes_array,
                available_rate_series,
            ):
                assert values is not None
                _plot_with_rolling(
                    axis,
                    episode_x,
                    values,
                    label,
                )
                axis.set_ylabel(label)
                axis.legend()
                axis.grid(True, alpha=0.3)
            axes_array[-1].set_xlabel("Episode")
            fig.suptitle("Workload-aware Fast metrics")
            fig.tight_layout()
            fig.savefig(
                figures_dir / "episode_normalized_fast_metrics.png",
                dpi=200,
            )
            plt.close(fig)

        if (
            quality_per_chunk is not None
            and degradation_per_chunk is not None
        ):
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            _plot_with_rolling(
                axes[0],
                episode_x,
                quality_per_chunk,
                "Quality / chunk",
            )
            _plot_with_rolling(
                axes[1],
                episode_x,
                degradation_per_chunk,
                "Quality degradation / chunk",
                color="tab:red",
            )
            axes[0].set_ylabel("Quality / chunk")
            axes[1].set_ylabel("Degradation / chunk")
            axes[1].set_xlabel("Episode")
            for axis in axes:
                axis.legend()
                axis.grid(True, alpha=0.3)
            fig.suptitle("Application-level quality metrics")
            fig.tight_layout()
            fig.savefig(
                figures_dir / "episode_quality_metrics.png",
                dpi=200,
            )
            plt.close(fig)

        layer_series = [
            _series(episodes, f"episode_layer_{layer}_ratio")
            for layer in range(1, 5)
        ]
        if all(values is not None for values in layer_series):
            fig, axis = plt.subplots(figsize=(10, 4.8))
            for layer, values in enumerate(layer_series, start=1):
                assert values is not None
                axis.plot(
                    episode_x,
                    _rolling(values),
                    label=f"Layer {layer}",
                )
            axis.axhline(
                0.25,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label="Uniform marginal share",
            )
            axis.set_xlabel("Episode")
            axis.set_ylabel("Executed layer ratio")
            axis.set_title(
                "Marginal layer shares (not a conditional-policy test)"
            )
            axis.legend(ncol=3)
            axis.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                figures_dir / "episode_layer_ratios.png",
                dpi=200,
            )
            plt.close(fig)

    update_x = _series(updates, "update")
    if update_x is not None:
        kl_post = _series(updates, "approx_kl_post")
        clip_post = _series(updates, "clipfrac_post")
        explained_variance = _series(
            updates,
            "explained_variance",
        )
        value_rmse = _series(updates, "value_rmse")

        if kl_post is not None and clip_post is not None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(update_x, kl_post, linewidth=0.8)
            axes[0].axhline(
                0.03,
                color="tab:red",
                linestyle="--",
                label="screening level 0.03",
            )
            axes[0].set_ylabel("Post-update KL")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(update_x, clip_post, linewidth=0.8)
            axes[1].axhline(
                0.25,
                color="tab:orange",
                linestyle="--",
                label="screening level 0.25",
            )
            axes[1].axhline(
                0.50,
                color="tab:red",
                linestyle="--",
                label="high level 0.50",
            )
            axes[1].set_xlabel("PPO update")
            axes[1].set_ylabel("Post-update clip fraction")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.suptitle("PPO trust-region diagnostics")
            fig.tight_layout()
            fig.savefig(
                figures_dir / "ppo_trust_region.png",
                dpi=200,
            )
            plt.close(fig)

        if explained_variance is not None and value_rmse is not None:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(
                update_x,
                explained_variance,
                linewidth=0.8,
            )
            axes[0].axhline(0.0, color="black", linewidth=1.0)
            axes[0].set_ylabel("Explained variance")
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(update_x, value_rmse, linewidth=0.8)
            axes[1].set_xlabel("PPO update")
            axes[1].set_ylabel("Value RMSE")
            axes[1].grid(True, alpha=0.3)
            fig.suptitle("Critic diagnostics")
            fig.tight_layout()
            fig.savefig(
                figures_dir / "critic_diagnostics.png",
                dpi=200,
            )
            plt.close(fig)


# ======================================================================
# Agent initialization / checkpoint
# ======================================================================
def _initialize_agent(
    env: Env,
    initial_obs: Dict[str, Any],
    train_cfg: FastTrainConfig,
    ppo_cfg: AgentPPOConfig,
) -> FastPPOAgent:
    agent = FastPPOAgent(
        env_cfg=env.cfg,
        obs_dim=infer_fast_obs_dim(initial_obs),
        ppo_cfg=ppo_cfg,
    )

    if train_cfg.fail_if_cuda_unavailable and agent.device.type != "cuda":
        raise RuntimeError(
            "CUDA was required but the Fast-PPO agent is not on CUDA."
        )

    checkpoint = _resolve_checkpoint(train_cfg.checkpoint)
    loaded_checkpoint: Optional[Dict[str, Any]] = None
    expected_resume_signature = _resume_signature(
        train_cfg=train_cfg,
        env_cfg=env.cfg,
        ppo_cfg=ppo_cfg,
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
    )
    if train_cfg.legacy_transfer:
        if checkpoint is None:
            raise ValueError("legacy_transfer requires checkpoint.")
        loaded_checkpoint = agent.load_legacy_transfer(checkpoint)
    elif train_cfg.resume:
        if checkpoint is None:
            raise ValueError("resume requires checkpoint.")
        _validate_resume_checkpoint(
            checkpoint_path=checkpoint,
            expected_signature=expected_resume_signature,
            train_cfg=train_cfg,
        )
        loaded_checkpoint = agent.load(
            checkpoint, strict=True, load_optimizer=True
        )
    elif checkpoint is not None:
        # Pretrained initialization followed by online round-wise fine-tuning.
        loaded_checkpoint = agent.load(
            checkpoint,
            strict=True,
            load_optimizer=bool(train_cfg.load_optimizer_on_warm_start),
        )
    elif (
        train_cfg.slow_decision_mode == "dpp"
        and train_cfg.require_pretrained_fast_for_dpp
    ):
        raise ValueError("DPP mode requires a pretrained Fast-PPO checkpoint.")

    setattr(agent, "_loaded_checkpoint", loaded_checkpoint)
    setattr(agent, "_resume_signature", expected_resume_signature)
    return agent


def _capture_rng_state(
    env: Env,
    slow_rng: np.random.Generator,
) -> Dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "env_rng_state": copy.deepcopy(env.rng.bit_generator.state),
        "slow_rng_state": copy.deepcopy(slow_rng.bit_generator.state),
        "env_episode": int(env.episode),
    }


def _restore_resume_state(
    agent: FastPPOAgent,
    env: Env,
    slow_rng: np.random.Generator,
    train_cfg: FastTrainConfig,
) -> Tuple[int, int, int, int]:
    checkpoint = getattr(agent, "_loaded_checkpoint", None)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Resume checkpoint payload was not retained.")
    extra = checkpoint.get("extra", {})
    required = {
        "episode",
        "global_round",
        "global_real_slot",
        "update_idx",
        "rng_state",
    }
    missing = sorted(required.difference(extra))
    if missing:
        raise RuntimeError(
            "Resume checkpoint is missing required fields: "
            + ", ".join(missing)
        )
    if extra.get("slow_decision_mode") != train_cfg.slow_decision_mode:
        raise RuntimeError(
            "Resume checkpoint slow_decision_mode mismatch: "
            f"checkpoint={extra.get('slow_decision_mode')!r}, "
            f"requested={train_cfg.slow_decision_mode!r}."
        )

    rng_state = extra["rng_state"]
    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])
    torch.set_rng_state(rng_state["torch_cpu_rng_state"])
    cuda_states = rng_state.get("torch_cuda_rng_state_all", [])
    if cuda_states:
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the resume checkpoint."
            )
        torch.cuda.set_rng_state_all(cuda_states)
    env.rng.bit_generator.state = copy.deepcopy(rng_state["env_rng_state"])
    slow_rng.bit_generator.state = copy.deepcopy(
        rng_state["slow_rng_state"]
    )
    env.episode = int(rng_state.get("env_episode", extra["episode"]))

    return (
        int(extra["episode"]),
        int(extra["global_round"]),
        int(extra["global_real_slot"]),
        int(extra["update_idx"]),
    )


def _training_checkpoint_extra(
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
    ppo_cfg: AgentPPOConfig,
    env: Env,
    slow_rng: np.random.Generator,
    episode: int,
    global_round: int,
    global_real_slot: int,
    update_idx: int,
    resume_signature: str,
) -> Dict[str, Any]:
    return {
        "episode": int(episode),
        "global_round": int(global_round),
        "global_real_slot": int(global_real_slot),
        "update_idx": int(update_idx),
        "phase": train_cfg.phase,
        "segment_id": int(train_cfg.segment_id),
        "seed": int(train_cfg.seed),
        "slow_decision_mode": train_cfg.slow_decision_mode,
        "resume_signature": str(resume_signature),
        "rng_state": _capture_rng_state(env, slow_rng),
        "env_config": (
            env_cfg.as_dict()
            if hasattr(env_cfg, "as_dict")
            else asdict(env_cfg)
        ),
        "train_config": train_cfg.to_dict(),
        "ppo_config": asdict(ppo_cfg),
    }


# ======================================================================
# Round execution
# ======================================================================
def _select_and_apply_slow_action(
    env: Env,
    agent: FastPPOAgent,
    train_cfg: FastTrainConfig,
    slow_rng: np.random.Generator,
    process_pool: Optional[
        _PersistentShadowProcessPool
    ],
) -> Tuple[
    Dict[str, Any],
    Dict[str, float],
]:
    if train_cfg.slow_decision_mode == "dpp":
        if process_pool is None:
            raise RuntimeError(
                "DPP mode requires a forecast process pool."
            )

        selected = select_slow_action_dpp(
            env=env,
            agent=agent,
            train_cfg=train_cfg,
            process_pool=process_pool,
        )

        env.apply_slow_action(selected.action)

        return env.get_fast_obs(), {
            "predicted_round_cost":
                selected.predicted_round_cost,
            "solver_mode": selected.solver_mode,
            "estimator_mode": selected.estimator_mode,
            "global_optimum_guaranteed": float(
                selected.global_optimum_guaranteed
            ),
            "candidate_requests":
                float(selected.candidate_requests),
            "unique_candidates":
                float(selected.unique_candidates),
            "finite_candidates":
                float(selected.finite_candidates),
            "coordinate_sweeps":
                float(selected.coordinate_sweeps),
            "forecast_seconds":
                selected.forecast_seconds,
            "forecast_policy_seconds":
                selected.policy_seconds,
            "forecast_env_seconds":
                selected.env_seconds,
            "forecast_mean_gpu_batch":
                selected.mean_gpu_batch,
            "forecast_trial_steps": float(
                selected.forecast_trial_steps
            ),
            "forecast_trial_steps_per_second": float(
                selected.forecast_trial_steps
            ) / max(float(selected.forecast_seconds), 1e-12),
            "num_rsu_links": float(
                np.sum(
                    selected.action[
                        "rsu_scheduling"
                    ]
                )
            ),
            "num_hired_uav": float(
                np.sum(
                    selected.action[
                        "uav_hiring"
                    ]
                )
            ),
            "num_uav_links": float(
                np.sum(
                    selected.action[
                        "uav_scheduling"
                    ]
                )
            ),
        }

    action = sample_random_slow_action(
        env,
        slow_rng,
        train_cfg,
    )
    env.apply_slow_action(action)

    return env.get_fast_obs(), {
        "predicted_round_cost": float("nan"),
        "solver_mode": "random_feasible_baseline",
        "estimator_mode": "not_applicable",
        "global_optimum_guaranteed": 0.0,
        "candidate_requests": 0.0,
        "unique_candidates": 0.0,
        "finite_candidates": 0.0,
        "coordinate_sweeps": 0.0,
        "forecast_seconds": 0.0,
        "forecast_policy_seconds": 0.0,
        "forecast_env_seconds": 0.0,
        "forecast_mean_gpu_batch": 0.0,
        "forecast_trial_steps": 0.0,
        "forecast_trial_steps_per_second": 0.0,
        "num_rsu_links": float(
            np.sum(action["rsu_scheduling"])
        ),
        "num_hired_uav": float(
            np.sum(action["uav_hiring"])
        ),
        "num_uav_links": float(
            np.sum(action["uav_scheduling"])
        ),
    }


def _execute_real_round_train(
    env: Env,
    agent: FastPPOAgent,
    obs: Dict[str, Any],
    train_cfg: FastTrainConfig,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    if len(agent.buffer) != 0:
        raise RuntimeError("PPO buffer must be empty at real round start.")

    model_digest_before = (
        _model_parameter_digest(agent)
        if bool(
            train_cfg.audit_runtime_invariants
        )
        else None
    )
    raw_obs_batch: list[np.ndarray] = []

    totals = {
        "formulation_reward": 0.0,
        "round_fast_cost": 0.0,
        "delivery": 0.0,
        "quality": 0.0,
        "quality_degradation": 0.0,
        "stall": 0.0,
        "scheduled_stall": 0.0,
        "unscheduled_stall": 0.0,
        "scheduled_user_slots": 0.0,
        "unscheduled_user_slots": 0.0,
        "consumed_soc": 0.0,
        "charged_soc": 0.0,
        "outage_slots": 0.0,
        "charging_slots": 0.0,
        "queue_playback_term": 0.0,
        "video_delivery_term": 0.0,
        "battery_consume_term": 0.0,
        "battery_charge_term": 0.0,
        "quality_degradation_term": 0.0,
        "min_soc": float("inf"),
        "service_rate": 0.0,
        "requested_chunks": 0.0,
        "active_action_dims": 0.0,
        "active_action_ratio": 0.0,
        "action_saturation_ratio": 0.0,
    }

    real_start = time.perf_counter()
    agent.model.train()
    layer_counts = np.zeros(int(env.cfg.layer), dtype=np.int64)

    for slot_idx in range(int(env.slow_T)):
        # The normalizer and model are fixed for the whole round.
        selected = agent.select_action(
            obs,
            deterministic=False,
            update_norm=False,
        )
        raw_obs_batch.append(selected["raw_obs_vec"])

        next_obs, env_reward, terminated, truncated, info = split_env_step(
            env.step(selected["env_action"])
        )
        expected_boundary = slot_idx == int(env.slow_T) - 1
        if bool(info.get("is_round_boundary", False)) != expected_boundary:
            raise RuntimeError(
                f"Real round boundary mismatch at slot {slot_idx}."
            )
        if terminated or truncated:
            raise RuntimeError(
                "Environment terminated/truncated inside fixed real round."
            )

        formulation_reward, slot_cost = formulation_reward_from_env_step(
            env_reward=env_reward,
            info=info,
            env_cfg=env.cfg,
        )
        is_segment_terminal = bool(expected_boundary)

        agent.store_transition(
            obs_vec=selected["obs_vec"],
            raw_action=selected["raw_action"],
            action_mask=selected["action_mask"],
            reward=(
                formulation_reward * float(train_cfg.ppo_reward_scale)
            ),
            done=is_segment_terminal,
            value=float(selected["value"]),
            log_prob=float(selected["log_prob"]),
        )

        metrics = extract_info_metrics(info)
        layer_counts += _count_executed_layers(
            selected["env_action"], int(env.cfg.layer)
        )
        totals["formulation_reward"] += formulation_reward
        totals["round_fast_cost"] += slot_cost
        for key in (
            "delivery",
            "quality",
            "quality_degradation",
            "stall",
            "scheduled_stall",
            "unscheduled_stall",
            "scheduled_user_slots",
            "unscheduled_user_slots",
            "consumed_soc",
            "charged_soc",
            "outage_slots",
            "charging_slots",
            "queue_playback_term",
            "video_delivery_term",
            "battery_consume_term",
            "battery_charge_term",
            "quality_degradation_term",
        ):
            totals[key] += metrics[key]
        totals["min_soc"] = min(totals["min_soc"], metrics["min_soc"])
        totals["service_rate"] += float(selected.get("service_rate", 0.0))
        totals["requested_chunks"] += float(
            selected.get("mean_requested_chunks", 0.0)
        )
        totals["active_action_dims"] += float(
            selected["active_action_dims"]
        )
        totals["active_action_ratio"] += float(
            selected["active_action_ratio"]
        )
        totals["action_saturation_ratio"] += float(
            selected["action_saturation_ratio"]
        )
        obs = next_obs

    real_seconds = time.perf_counter() - real_start
    if len(agent.buffer) != int(env.slow_T) or not agent.buffer.is_full:
        raise RuntimeError(
            "One completed real round must exactly fill the PPO buffer: "
            f"len={len(agent.buffer)}, expected={env.slow_T}."
        )
    if (
        model_digest_before is not None
        and _model_parameter_digest(agent)
        != model_digest_before
    ):
        raise RuntimeError(
            "Fast policy changed inside a real round."
        )

    # Round is a finite-horizon Fast subproblem under one fixed slow action.
    agent.finish_rollout(last_obs=obs, last_done=True)
    update_start = time.perf_counter()
    update_logs = agent.update()
    update_seconds = time.perf_counter() - update_start

    # Only real observations update statistics, and only after the old-policy
    # PPO update has consumed the round buffer.
    if bool(train_cfg.freeze_obs_norm_within_round):
        agent.update_obs_normalizer(np.stack(raw_obs_batch, axis=0))

    denominator = float(max(int(env.slow_T), 1))
    for key in (
        "service_rate",
        "requested_chunks",
        "active_action_dims",
        "active_action_ratio",
        "action_saturation_ratio",
    ):
        totals[key] /= denominator

    if not np.isfinite(totals["min_soc"]):
        totals["min_soc"] = 0.0

    _finalize_round_metrics(totals, layer_counts)

    if bool(train_cfg.fail_on_outage) and totals["outage_slots"] > 0.0:
        raise RuntimeError(
            "Battery outage occurred during Fast pretraining: "
            f"outage_slots={totals['outage_slots']}."
        )
    minimum_allowed_soc = float(env.cfg.battery.e_min) - float(
        train_cfg.battery_soc_tolerance
    )
    if totals["min_soc"] < minimum_allowed_soc:
        raise RuntimeError(
            "Battery SoC fell below the configured tolerance: "
            f"min_soc={totals['min_soc']:.6f}, "
            f"allowed={minimum_allowed_soc:.6f}."
        )

    totals["hiring_cost"] = _hiring_cost(
        env,
        {
            "uav_hiring": env.uav_hiring,
            "rsu_scheduling": env.rsu_scheduling,
            "uav_scheduling": env.uav_scheduling,
        },
    )
    totals["realized_round_cost"] = (
        totals["round_fast_cost"] + totals["hiring_cost"]
    )
    totals["real_round_seconds"] = real_seconds
    totals["real_slots_per_second"] = float(env.slow_T) / max(
        real_seconds, 1e-12
    )
    totals["ppo_update_seconds"] = update_seconds
    return obs, totals, update_logs


def _execute_real_round_eval(
    env: Env,
    agent: FastPPOAgent,
    obs: Dict[str, Any],
    train_cfg: FastTrainConfig,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    totals = {
        "formulation_reward": 0.0,
        "round_fast_cost": 0.0,
        "delivery": 0.0,
        "quality": 0.0,
        "quality_degradation": 0.0,
        "stall": 0.0,
        "scheduled_stall": 0.0,
        "unscheduled_stall": 0.0,
        "scheduled_user_slots": 0.0,
        "unscheduled_user_slots": 0.0,
        "consumed_soc": 0.0,
        "charged_soc": 0.0,
        "charging_slots": 0.0,
        "outage_slots": 0.0,
        "queue_playback_term": 0.0,
        "video_delivery_term": 0.0,
        "battery_consume_term": 0.0,
        "battery_charge_term": 0.0,
        "quality_degradation_term": 0.0,
        "min_soc": float("inf"),
        "service_rate": 0.0,
        "requested_chunks": 0.0,
        "active_action_dims": 0.0,
        "active_action_ratio": 0.0,
        "action_saturation_ratio": 0.0,
    }
    agent.model.eval()
    layer_counts = np.zeros(int(env.cfg.layer), dtype=np.int64)

    for slot_idx in range(int(env.slow_T)):
        selected = agent.select_action(
            obs,
            deterministic=True,
            update_norm=False,
        )
        next_obs, env_reward, terminated, truncated, info = split_env_step(
            env.step(selected["env_action"])
        )
        expected_boundary = slot_idx == int(env.slow_T) - 1
        if bool(info.get("is_round_boundary", False)) != expected_boundary:
            raise RuntimeError("Evaluation round boundary mismatch.")
        if terminated or truncated:
            raise RuntimeError("Evaluation terminated inside a round.")

        reward, cost = formulation_reward_from_env_step(
            env_reward=env_reward,
            info=info,
            env_cfg=env.cfg,
        )
        metrics = extract_info_metrics(info)
        layer_counts += _count_executed_layers(
            selected["env_action"], int(env.cfg.layer)
        )
        totals["formulation_reward"] += reward
        totals["round_fast_cost"] += cost
        for key in (
            "delivery",
            "quality",
            "quality_degradation",
            "stall",
            "scheduled_stall",
            "unscheduled_stall",
            "scheduled_user_slots",
            "unscheduled_user_slots",
            "consumed_soc",
            "charged_soc",
            "charging_slots",
            "outage_slots",
            "queue_playback_term",
            "video_delivery_term",
            "battery_consume_term",
            "battery_charge_term",
            "quality_degradation_term",
        ):
            totals[key] += metrics[key]
        totals["min_soc"] = min(totals["min_soc"], metrics["min_soc"])
        totals["service_rate"] += float(selected.get("service_rate", 0.0))
        totals["requested_chunks"] += float(
            selected.get("mean_requested_chunks", 0.0)
        )
        totals["active_action_dims"] += float(
            selected["active_action_dims"]
        )
        totals["active_action_ratio"] += float(
            selected["active_action_ratio"]
        )
        totals["action_saturation_ratio"] += float(
            selected["action_saturation_ratio"]
        )
        obs = next_obs

    denominator = float(max(int(env.slow_T), 1))
    for key in (
        "service_rate",
        "requested_chunks",
        "active_action_dims",
        "active_action_ratio",
        "action_saturation_ratio",
    ):
        totals[key] /= denominator

    totals["hiring_cost"] = _hiring_cost(
        env,
        {
            "uav_hiring": env.uav_hiring,
            "rsu_scheduling": env.rsu_scheduling,
            "uav_scheduling": env.uav_scheduling,
        },
    )
    totals["realized_round_cost"] = (
        totals["round_fast_cost"] + totals["hiring_cost"]
    )
    if not np.isfinite(totals["min_soc"]):
        totals["min_soc"] = 0.0
    _finalize_round_metrics(totals, layer_counts)
    if bool(train_cfg.fail_on_outage) and totals["outage_slots"] > 0.0:
        raise RuntimeError(
            "Battery outage occurred during evaluation: "
            f"outage_slots={totals['outage_slots']}."
        )
    minimum_allowed_soc = float(env.cfg.battery.e_min) - float(
        train_cfg.battery_soc_tolerance
    )
    if totals["min_soc"] < minimum_allowed_soc:
        raise RuntimeError(
            "Evaluation battery SoC violated tolerance: "
            f"min_soc={totals['min_soc']:.6f}, "
            f"allowed={minimum_allowed_soc:.6f}."
        )
    return obs, totals


# ======================================================================
# Train / evaluate
# ======================================================================
def train(train_cfg: FastTrainConfig) -> None:
    set_seed(
        int(train_cfg.seed),
        deterministic=bool(train_cfg.deterministic_torch),
    )
    env_cfg = build_env_config(train_cfg)
    _assert_joint_config(train_cfg, env_cfg)
    ppo_cfg = build_agent_ppo_config(train_cfg)

    env = Env(env_cfg)
    initial_obs, reset_info = split_env_reset(env.reset())
    agent = _initialize_agent(env, initial_obs, train_cfg, ppo_cfg)

    run_dir = make_run_dir(train_cfg, env_cfg)
    _validate_run_dir_contract(run_dir, train_cfg)
    checkpoint_stem = (
        "fast_ppo_pretrain"
        if train_cfg.slow_decision_mode == "random"
        else "fast_ppo_joint"
    )
    save_configs(
        run_dir,
        train_cfg,
        env_cfg,
        ppo_cfg,
        obs_dim=agent.obs_dim,
        action_dim=agent.action_dim,
    )

    episode_logger = ScalarLogger(run_dir / "logs" / "episodes.csv")
    round_logger = ScalarLogger(run_dir / "logs" / "rounds.csv")
    update_logger = ScalarLogger(run_dir / "logs" / "updates.csv")
    slow_rng = np.random.default_rng(int(train_cfg.seed) + 10_007)
    resume_signature = str(getattr(agent, "_resume_signature"))

    completed_episode = 0
    global_real_slot = 0
    global_round = 0
    update_idx = 0
    if train_cfg.resume:
        (
            completed_episode,
            global_round,
            global_real_slot,
            update_idx,
        ) = _restore_resume_state(agent, env, slow_rng, train_cfg)

    if not train_cfg.resume and train_cfg.checkpoint is None:
        initial_extra = _training_checkpoint_extra(
            train_cfg=train_cfg,
            env_cfg=env_cfg,
            ppo_cfg=ppo_cfg,
            env=env,
            slow_rng=slow_rng,
            episode=0,
            global_round=0,
            global_real_slot=0,
            update_idx=0,
            resume_signature=resume_signature,
        )
        initial_extra["checkpoint_role"] = "initial_untrained_policy"
        initial_path = (
            run_dir
            / "checkpoints"
            / f"{checkpoint_stem}_initial.pt"
        )
        agent.save(initial_path, extra=initial_extra)
        print(f"[SAVE INITIAL] {initial_path}", flush=True)

    segment_start_episode = completed_episode + 1
    segment_end_episode = completed_episode + int(train_cfg.num_episodes)
    if segment_end_episode > int(train_cfg.target_total_episodes):
        raise RuntimeError(
            "Requested segment exceeds FAST_PPO_TARGET_TOTAL_EPISODES: "
            f"completed={completed_episode}, segment={train_cfg.num_episodes}, "
            f"target={train_cfg.target_total_episodes}."
        )

    print("=" * 100, flush=True)
    phase_label = (
        "FAST-PPO PRETRAIN"
        if train_cfg.slow_decision_mode == "random"
        else "SLOW-DPP + FAST-PPO JOINT TRAIN"
    )
    print(f"[{phase_label} START]", flush=True)
    print(
        "branch_basis   : feat/no-hrl "
        "(exact commit is recorded by the Slurm wrapper)",
        flush=True,
    )
    print(f"run_dir        : {run_dir}", flush=True)
    print(f"device         : {agent.device}", flush=True)
    print(f"slow_mode      : {train_cfg.slow_decision_mode}", flush=True)
    print(f"slow_T         : {env_cfg.slow_T}", flush=True)
    print(f"rollout_slots  : {ppo_cfg.rollout_steps}", flush=True)
    print(f"actor_lr       : {ppo_cfg.actor_lr}", flush=True)
    print(f"critic_lr      : {ppo_cfg.critic_lr}", flush=True)
    print(
        f"cat_entropy    : {ppo_cfg.categorical_entropy_coef}",
        flush=True,
    )
    print(f"target_kl      : {ppo_cfg.target_kl}", flush=True)
    print(f"checkpoint     : {_resolve_checkpoint(train_cfg.checkpoint)}", flush=True)
    print(f"resume         : {train_cfg.resume}", flush=True)
    print(
        f"episode_range  : {segment_start_episode}..{segment_end_episode}",
        flush=True,
    )
    print(f"reset_info     : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    forecast_pool: Optional[
        _PersistentShadowProcessPool
    ] = None

    if train_cfg.slow_decision_mode == "dpp":
        forecast_pool = _PersistentShadowProcessPool(
            worker_count=int(
                train_cfg.dpp_forecast_workers
            ),
            batch_capacity=int(
                train_cfg.dpp_candidate_batch_size
            ),
            obs_dim=int(agent.obs_dim),
            action_dim=int(agent.action_dim),
        )

    try:
        for episode_idx in range(
            segment_start_episode,
            segment_end_episode + 1,
        ):
            env.cfg.move_prob = get_episode_move_prob(
                train_cfg,
                episode_idx,
            )
            obs, _ = split_env_reset(env.reset())

            episode_totals = {
                "reward": 0.0,
                "realized_round_cost": 0.0,
                "hiring_cost": 0.0,
                "delivery": 0.0,
                "quality": 0.0,
                "quality_degradation": 0.0,
                "stall": 0.0,
                "scheduled_stall": 0.0,
                "unscheduled_stall": 0.0,
                "scheduled_user_slots": 0.0,
                "unscheduled_user_slots": 0.0,
                "consumed_soc": 0.0,
                "charged_soc": 0.0,
                "queue_playback_term": 0.0,
                "video_delivery_term": 0.0,
                "battery_consume_term": 0.0,
                "battery_charge_term": 0.0,
                "quality_degradation_term": 0.0,
                "outage_slots": 0.0,
                "service_rate": 0.0,
                "requested_chunks": 0.0,
                "active_action_dims": 0.0,
                "active_action_ratio": 0.0,
                "action_saturation_ratio": 0.0,
                "forecast_trial_steps": 0.0,
                "forecast_seconds": 0.0,
                "real_seconds": 0.0,
                "update_seconds": 0.0,
                "prediction_gap_sum": 0.0,
                "prediction_gap_count": 0,
                "min_soc": float("inf"),
            }
            episode_layer_counts = np.zeros(
                int(env.cfg.layer), dtype=np.float64
            )

            for round_in_episode in range(
                1,
                int(train_cfg.rounds_per_episode) + 1,
            ):
                if int(env.round_slot) != 0:
                    raise RuntimeError(
                        "Episode loop is not at a round boundary."
                    )

                # The Slow-DPP forecast uses one fixed pre-update Fast policy.
                # The selected slow action then stays fixed throughout the real
                # round, and PPO updates only after that round has completed.
                obs, slow_info = _select_and_apply_slow_action(
                    env=env,
                    agent=agent,
                    train_cfg=train_cfg,
                    slow_rng=slow_rng,
                    process_pool=forecast_pool,
                )
                policy_version = update_idx

                obs, realized, update_logs = _execute_real_round_train(
                    env=env,
                    agent=agent,
                    obs=obs,
                    train_cfg=train_cfg,
                )

                update_idx += 1
                global_round += 1
                global_real_slot += int(env_cfg.slow_T)

                predicted = float(
                    slow_info["predicted_round_cost"]
                )
                realized_cost = float(
                    realized["realized_round_cost"]
                )
                prediction_gap = (
                    realized_cost - predicted
                    if np.isfinite(predicted)
                    else float("nan")
                )

                round_row = {
                    "global_round": global_round,
                    "episode": episode_idx,
                    "segment_id": int(train_cfg.segment_id),
                    "segment_episode": (
                        episode_idx - segment_start_episode + 1
                    ),
                    "round_in_episode": round_in_episode,
                    "policy_version": policy_version,
                    "global_real_slot": global_real_slot,
                    **slow_info,
                    **realized,
                    "prediction_gap": prediction_gap,
                }
                round_logger.write(round_row)

                update_logger.write(
                    {
                        "update": update_idx,
                        "global_round": global_round,
                        "episode": episode_idx,
                        "global_real_slot": global_real_slot,
                        **update_logs,
                    }
                )

                episode_totals["reward"] += realized[
                    "formulation_reward"
                ]
                episode_totals["realized_round_cost"] += realized[
                    "realized_round_cost"
                ]
                episode_totals["hiring_cost"] += realized["hiring_cost"]
                episode_totals["delivery"] += realized["delivery"]
                episode_totals["quality"] += realized["quality"]
                episode_totals["quality_degradation"] += realized[
                    "quality_degradation"
                ]
                episode_totals["stall"] += realized["stall"]
                for key in (
                    "scheduled_stall",
                    "unscheduled_stall",
                    "scheduled_user_slots",
                    "unscheduled_user_slots",
                    "consumed_soc",
                    "charged_soc",
                    "queue_playback_term",
                    "video_delivery_term",
                    "battery_consume_term",
                    "battery_charge_term",
                    "quality_degradation_term",
                    "service_rate",
                    "requested_chunks",
                    "active_action_dims",
                    "active_action_ratio",
                    "action_saturation_ratio",
                ):
                    episode_totals[key] += realized[key]
                episode_totals["outage_slots"] += realized[
                    "outage_slots"
                ]
                episode_totals["forecast_trial_steps"] += slow_info[
                    "forecast_trial_steps"
                ]
                episode_totals["forecast_seconds"] += slow_info[
                    "forecast_seconds"
                ]
                episode_totals["real_seconds"] += realized[
                    "real_round_seconds"
                ]
                episode_totals["update_seconds"] += realized[
                    "ppo_update_seconds"
                ]
                episode_totals["min_soc"] = min(
                    episode_totals["min_soc"],
                    realized["min_soc"],
                )
                active_layers = float(realized["active_layer_actions"])
                for layer in range(1, int(env.cfg.layer) + 1):
                    episode_layer_counts[layer - 1] += (
                        active_layers * realized[f"layer_{layer}_ratio"]
                    )

                if np.isfinite(prediction_gap):
                    episode_totals["prediction_gap_sum"] += (
                        prediction_gap
                    )
                    episode_totals["prediction_gap_count"] += 1

                print(
                    (
                        "[FAST PRETRAIN ROUND] "
                        if train_cfg.slow_decision_mode == "random"
                        else "[JOINT ROUND] "
                    )
                    +
                    f"ep={episode_idx}/{segment_end_episode} "
                    f"r={round_in_episode}/"
                    f"{train_cfg.rounds_per_episode} "
                    f"policy_v={policy_version} "
                    f"pred={predicted:.4f} "
                    f"real={realized_cost:.4f} "
                    f"gap={prediction_gap:.4f} "
                    f"solver={slow_info['solver_mode']} "
                    f"candidates="
                    f"{int(slow_info['unique_candidates'])} "
                    f"forecast_s="
                    f"{slow_info['forecast_seconds']:.2f} "
                    f"real_sps="
                    f"{realized['real_slots_per_second']:.2f} "
                    f"kl_mb={update_logs['approx_kl']:.6f} "
                    f"kl_post={update_logs['approx_kl_post']:.6f} "
                    f"clip_post={update_logs['clipfrac_post']:.4f} "
                    f"ev={update_logs['explained_variance']:.4f}",
                    flush=True,
                )

            if not np.isfinite(episode_totals["min_soc"]):
                episode_totals["min_soc"] = 0.0
            _finalize_round_metrics(
                episode_totals, episode_layer_counts
            )
            round_denominator = float(
                max(int(train_cfg.rounds_per_episode), 1)
            )
            for key in (
                "service_rate",
                "requested_chunks",
                "active_action_dims",
                "active_action_ratio",
                "action_saturation_ratio",
            ):
                episode_totals[key] /= round_denominator
            episode_totals["forecast_trial_steps_per_second"] = float(
                episode_totals["forecast_trial_steps"]
            ) / max(float(episode_totals["forecast_seconds"]), 1e-12)

            prediction_gap_mean = (
                episode_totals["prediction_gap_sum"]
                / episode_totals["prediction_gap_count"]
                if episode_totals["prediction_gap_count"] > 0
                else float("nan")
            )

            episode_logger.write(
                {
                    "episode": episode_idx,
                    "segment_id": int(train_cfg.segment_id),
                    "segment_episode": (
                        episode_idx - segment_start_episode + 1
                    ),
                    "global_round": global_round,
                    "global_real_slot": global_real_slot,
                    "move_prob": float(env.cfg.move_prob),
                    "episode_formulation_reward": episode_totals[
                        "reward"
                    ],
                    "episode_realized_round_cost": episode_totals[
                        "realized_round_cost"
                    ],
                    "episode_hiring_cost": episode_totals[
                        "hiring_cost"
                    ],
                    "episode_delivery": episode_totals["delivery"],
                    "episode_quality": episode_totals["quality"],
                    "episode_quality_degradation": episode_totals[
                        "quality_degradation"
                    ],
                    "episode_stall": episode_totals["stall"],
                    "episode_scheduled_stall": episode_totals[
                        "scheduled_stall"
                    ],
                    "episode_unscheduled_stall": episode_totals[
                        "unscheduled_stall"
                    ],
                    "episode_scheduled_user_slots": episode_totals[
                        "scheduled_user_slots"
                    ],
                    "episode_unscheduled_user_slots": episode_totals[
                        "unscheduled_user_slots"
                    ],
                    "episode_scheduled_stall_rate": episode_totals[
                        "scheduled_stall_rate"
                    ],
                    "episode_unscheduled_stall_rate": episode_totals[
                        "unscheduled_stall_rate"
                    ],
                    "episode_quality_per_chunk": episode_totals[
                        "quality_per_chunk"
                    ],
                    "episode_quality_degradation_per_chunk": episode_totals[
                        "quality_degradation_per_chunk"
                    ],
                    "episode_fast_cost_per_scheduled_user_slot": (
                        -float(episode_totals["reward"])
                        / max(
                            float(
                                episode_totals[
                                    "scheduled_user_slots"
                                ]
                            ),
                            1.0,
                        )
                    ),
                    "episode_delivery_per_scheduled_user_slot": (
                        float(episode_totals["delivery"])
                        / max(
                            float(
                                episode_totals[
                                    "scheduled_user_slots"
                                ]
                            ),
                            1.0,
                        )
                    ),
                    "episode_queue_playback_term": episode_totals[
                        "queue_playback_term"
                    ],
                    "episode_video_delivery_term": episode_totals[
                        "video_delivery_term"
                    ],
                    "episode_battery_consume_term": episode_totals[
                        "battery_consume_term"
                    ],
                    "episode_battery_charge_term": episode_totals[
                        "battery_charge_term"
                    ],
                    "episode_quality_degradation_term": episode_totals[
                        "quality_degradation_term"
                    ],
                    "episode_consumed_soc": episode_totals[
                        "consumed_soc"
                    ],
                    "episode_charged_soc": episode_totals[
                        "charged_soc"
                    ],
                    "episode_outage_slots": episode_totals[
                        "outage_slots"
                    ],
                    "episode_min_soc": episode_totals["min_soc"],
                    "episode_service_rate": episode_totals[
                        "service_rate"
                    ],
                    "episode_requested_chunks": episode_totals[
                        "requested_chunks"
                    ],
                    "episode_active_action_dims": episode_totals[
                        "active_action_dims"
                    ],
                    "episode_active_action_ratio": episode_totals[
                        "active_action_ratio"
                    ],
                    "episode_action_saturation_ratio": episode_totals[
                        "action_saturation_ratio"
                    ],
                    "episode_forecast_seconds": episode_totals[
                        "forecast_seconds"
                    ],
                    "episode_real_seconds": episode_totals[
                        "real_seconds"
                    ],
                    "episode_update_seconds": episode_totals[
                        "update_seconds"
                    ],
                    "episode_prediction_gap_mean": prediction_gap_mean,
                    "episode_forecast_trial_steps": episode_totals[
                        "forecast_trial_steps"
                    ],
                    "episode_forecast_trial_steps_per_second": (
                        episode_totals[
                            "forecast_trial_steps_per_second"
                        ]
                    ),
                    **{
                        f"episode_layer_{layer}_ratio": episode_totals[
                            f"layer_{layer}_ratio"
                        ]
                        for layer in range(1, int(env.cfg.layer) + 1)
                    },
                }
            )

            print(
                "[EPISODE] "
                f"ep={episode_idx}/{segment_end_episode} "
                f"reward={episode_totals['reward']:.4f} "
                f"delivery={episode_totals['delivery']:.1f} "
                f"stall={episode_totals['stall']:.1f} "
                f"gap_mean={prediction_gap_mean:.4f} "
                f"outage_slots="
                f"{episode_totals['outage_slots']:.0f}",
                flush=True,
            )

            if (
                int(train_cfg.save_every_episodes) > 0
                and episode_idx
                % int(train_cfg.save_every_episodes)
                == 0
            ):
                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"{checkpoint_stem}_ep{episode_idx}.pt"
                )
                agent.save(
                    checkpoint_path,
                    extra=_training_checkpoint_extra(
                        train_cfg=train_cfg,
                        env_cfg=env_cfg,
                        ppo_cfg=ppo_cfg,
                        env=env,
                        slow_rng=slow_rng,
                        episode=episode_idx,
                        global_round=global_round,
                        global_real_slot=global_real_slot,
                        update_idx=update_idx,
                        resume_signature=resume_signature,
                    ),
                )
                print(f"[SAVE] {checkpoint_path}", flush=True)

            if bool(train_cfg.save_latest_every_episode):
                latest_path = (
                    run_dir
                    / "checkpoints"
                    / f"{checkpoint_stem}_latest.pt"
                )
                agent.save(
                    latest_path,
                    extra=_training_checkpoint_extra(
                        train_cfg=train_cfg,
                        env_cfg=env_cfg,
                        ppo_cfg=ppo_cfg,
                        env=env,
                        slow_rng=slow_rng,
                        episode=episode_idx,
                        global_round=global_round,
                        global_real_slot=global_real_slot,
                        update_idx=update_idx,
                        resume_signature=resume_signature,
                    ),
                )
                print(f"[SAVE LATEST] {latest_path}", flush=True)

            if (
                int(train_cfg.plot_every_episodes) > 0
                and episode_idx
                % int(train_cfg.plot_every_episodes)
                == 0
            ):
                save_training_plots(
                    run_dir,
                    smooth_window=int(
                        train_cfg.plot_smooth_window
                    ),
                )

    finally:
        if forecast_pool is not None:
            forecast_pool.close()

    if len(agent.buffer) != 0:
        raise RuntimeError("Joint training ended with a nonempty PPO buffer.")

    final_path = (
        run_dir
        / "checkpoints"
        / f"{checkpoint_stem}_final.pt"
    )
    agent.save(
        final_path,
        extra=_training_checkpoint_extra(
            train_cfg=train_cfg,
            env_cfg=env_cfg,
            ppo_cfg=ppo_cfg,
            env=env,
            slow_rng=slow_rng,
            episode=segment_end_episode,
            global_round=global_round,
            global_real_slot=global_real_slot,
            update_idx=update_idx,
            resume_signature=resume_signature,
        ),
    )
    save_json(
        {
            "final_checkpoint": str(final_path),
            "global_round": global_round,
            "global_real_slot": global_real_slot,
            "updates": update_idx,
            "segment_id": int(train_cfg.segment_id),
            "segment_start_episode": segment_start_episode,
            "segment_end_episode": segment_end_episode,
            "resumed_from": (
                str(_resolve_checkpoint(train_cfg.checkpoint))
                if train_cfg.resume
                else None
            ),
        },
        run_dir / "train_summary.json",
    )
    save_training_plots(run_dir)
    print(f"[DONE] final checkpoint: {final_path}", flush=True)


def evaluate(train_cfg: FastTrainConfig) -> None:
    set_seed(int(train_cfg.seed), deterministic=True)
    env_cfg = build_env_config(train_cfg)
    _assert_joint_config(train_cfg, env_cfg)
    env_cfg.move_prob = float(train_cfg.mobility_curriculum[-1][1])
    ppo_cfg = build_agent_ppo_config(train_cfg)

    env = Env(env_cfg)
    obs, _ = split_env_reset(env.reset())
    agent = _initialize_agent(env, obs, train_cfg, ppo_cfg)
    run_dir = make_run_dir(train_cfg, env_cfg)
    _validate_run_dir_contract(run_dir, train_cfg)
    save_configs(
        run_dir,
        train_cfg,
        env_cfg,
        ppo_cfg,
        agent.obs_dim,
        agent.action_dim,
    )

    logger = ScalarLogger(run_dir / "logs" / "eval_episodes.csv")
    slow_rng = np.random.default_rng(int(train_cfg.seed) + 20_011)
    forecast_pool: Optional[_PersistentShadowProcessPool] = None
    if train_cfg.slow_decision_mode == "dpp":
        forecast_pool = _PersistentShadowProcessPool(
            worker_count=int(train_cfg.dpp_forecast_workers),
            batch_capacity=int(train_cfg.dpp_candidate_batch_size),
            obs_dim=int(agent.obs_dim),
            action_dim=int(agent.action_dim),
        )

    try:
        for episode_idx in range(1, int(train_cfg.eval_episodes) + 1):
            obs, _ = split_env_reset(env.reset())
            totals = {
                "reward": 0.0,
                "realized_round_cost": 0.0,
                "hiring_cost": 0.0,
                "delivery": 0.0,
                "quality": 0.0,
                "quality_degradation": 0.0,
                "stall": 0.0,
                "scheduled_stall": 0.0,
                "unscheduled_stall": 0.0,
                "scheduled_user_slots": 0.0,
                "unscheduled_user_slots": 0.0,
                "consumed_soc": 0.0,
                "charged_soc": 0.0,
                "queue_playback_term": 0.0,
                "video_delivery_term": 0.0,
                "battery_consume_term": 0.0,
                "battery_charge_term": 0.0,
                "quality_degradation_term": 0.0,
                "outage_slots": 0.0,
                "min_soc": float("inf"),
                "service_rate": 0.0,
                "requested_chunks": 0.0,
                "active_action_dims": 0.0,
                "active_action_ratio": 0.0,
                "action_saturation_ratio": 0.0,
                "prediction_gap_sum": 0.0,
                "prediction_gap_count": 0,
                "forecast_seconds": 0.0,
                "forecast_trial_steps": 0.0,
            }
            layer_counts = np.zeros(int(env.cfg.layer), dtype=np.float64)

            for _ in range(int(train_cfg.eval_rounds_per_episode)):
                obs, slow_info = _select_and_apply_slow_action(
                    env,
                    agent,
                    train_cfg,
                    slow_rng,
                    process_pool=forecast_pool,
                )
                obs, realized = _execute_real_round_eval(
                    env, agent, obs, train_cfg
                )
                predicted = float(slow_info["predicted_round_cost"])
                if np.isfinite(predicted):
                    totals["prediction_gap_sum"] += (
                        realized["realized_round_cost"] - predicted
                    )
                    totals["prediction_gap_count"] += 1
                totals["reward"] += realized["formulation_reward"]
                totals["realized_round_cost"] += realized[
                    "realized_round_cost"
                ]
                totals["hiring_cost"] += realized["hiring_cost"]
                for key in (
                    "delivery",
                    "quality",
                    "quality_degradation",
                    "stall",
                    "scheduled_stall",
                    "unscheduled_stall",
                    "scheduled_user_slots",
                    "unscheduled_user_slots",
                    "consumed_soc",
                    "charged_soc",
                    "queue_playback_term",
                    "video_delivery_term",
                    "battery_consume_term",
                    "battery_charge_term",
                    "quality_degradation_term",
                    "outage_slots",
                    "service_rate",
                    "requested_chunks",
                    "active_action_dims",
                    "active_action_ratio",
                    "action_saturation_ratio",
                ):
                    totals[key] += realized[key]
                totals["min_soc"] = min(
                    totals["min_soc"], realized["min_soc"]
                )
                totals["forecast_seconds"] += slow_info["forecast_seconds"]
                totals["forecast_trial_steps"] += slow_info[
                    "forecast_trial_steps"
                ]
                active_layers = float(realized["active_layer_actions"])
                for layer in range(1, int(env.cfg.layer) + 1):
                    layer_counts[layer - 1] += (
                        active_layers * realized[f"layer_{layer}_ratio"]
                    )

            gap_mean = (
                totals["prediction_gap_sum"]
                / totals["prediction_gap_count"]
                if totals["prediction_gap_count"] > 0
                else float("nan")
            )
            if not np.isfinite(totals["min_soc"]):
                totals["min_soc"] = 0.0
            _finalize_round_metrics(totals, layer_counts)
            eval_round_denominator = float(
                max(int(train_cfg.eval_rounds_per_episode), 1)
            )
            for key in (
                "service_rate",
                "requested_chunks",
                "active_action_dims",
                "active_action_ratio",
                "action_saturation_ratio",
            ):
                totals[key] /= eval_round_denominator
            totals["forecast_trial_steps_per_second"] = float(
                totals["forecast_trial_steps"]
            ) / max(float(totals["forecast_seconds"]), 1e-12)

            logger.write(
                {
                    "episode": episode_idx,
                    **{
                        key: value
                        for key, value in totals.items()
                        if key not in {
                            "prediction_gap_sum",
                            "prediction_gap_count",
                        }
                    },
                    "prediction_gap_mean": gap_mean,
                }
            )
            print(
                "[EVAL] "
                f"ep={episode_idx}/{train_cfg.eval_episodes} "
                f"reward={totals['reward']:.4f} "
                f"delivery={totals['delivery']:.1f} "
                f"quality/chunk={totals['quality_per_chunk']:.4f} "
                f"stall={totals['stall']:.1f} "
                f"gap={gap_mean:.4f}",
                flush=True,
            )
    finally:
        if forecast_pool is not None:
            forecast_pool.close()


def main() -> None:
    train_cfg = get_fast_ppo_config()
    if train_cfg.mode == "train":
        train(train_cfg)
    elif train_cfg.mode == "eval":
        evaluate(train_cfg)
    else:
        raise ValueError(f"Unsupported mode: {train_cfg.mode}")


if __name__ == "__main__":
    main()
