from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
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
def build_env_config() -> EnvConfig:
    """
    System/environment parameter source remains proposed/config.py.

    Battery round-feasibility horizon was 5 while slow_T was 3600 in the
    original branch. The joint trainer aligns them without introducing a
    separate configuration file.
    """
    cfg = EnvConfig()
    cfg.battery.target_service_slots_per_round = int(cfg.slow_T)
    return cfg


def build_agent_ppo_config(train_cfg: FastTrainConfig) -> AgentPPOConfig:
    return AgentPPOConfig(
        rollout_steps=int(train_cfg.rollout_slots),
        update_epochs=int(train_cfg.update_epochs),
        batch_size=int(train_cfg.batch_size),
        gamma=float(train_cfg.gamma),
        gae_lambda=float(train_cfg.gae_lambda),
        lr=float(train_cfg.lr),
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


def save_configs(
    run_dir: Path,
    train_cfg: FastTrainConfig,
    env_cfg: EnvConfig,
    ppo_cfg: AgentPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> None:
    save_json(train_cfg.to_dict(), run_dir / "train_config.json")
    save_json(
        env_cfg.as_dict() if hasattr(env_cfg, "as_dict") else asdict(env_cfg),
        run_dir / "env_config.json",
    )
    save_json(asdict(ppo_cfg), run_dir / "ppo_config.json")
    save_json(
        {
            "proposed_root": str(PROPOSED_ROOT),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": (
                "pretrained Fast-PPO -> frozen round-wise Slow-DPP forecast "
                "-> real Fast round -> one PPO update"
            ),
            "slow_solver": "complete-action region-coordinate minimization",
            "global_optimum_guaranteed": False,
        },
        run_dir / "run_info.json",
    )


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
    if train_cfg.slow_decision_mode != "dpp":
        return

    if int(train_cfg.rollout_slots) != int(env_cfg.slow_T):
        raise ValueError(
            "Joint mode requires rollout_slots == env.slow_T: "
            f"{train_cfg.rollout_slots} != {env_cfg.slow_T}."
        )
    if int(train_cfg.dpp_forecast_horizon) != int(env_cfg.slow_T):
        raise ValueError(
            "Joint mode requires dpp_forecast_horizon == env.slow_T: "
            f"{train_cfg.dpp_forecast_horizon} != {env_cfg.slow_T}."
        )
    if not bool(train_cfg.freeze_obs_norm_within_round):
        raise ValueError(
            "Joint mode requires observation normalizer to remain fixed "
            "within every round."
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
    Current Env reward contains

        alpha_Z * sum Z_n(t)d_n(t)
        - alpha_B * sum B_u(t)e_u(t)
        + alpha_B * sum B_u(t)e_c(t)
        - V * degradation(t).

    Formulation v6 uses alpha_Z * Z_n(t)(d_n(t)-b_n(t)); therefore the
    action-independent one-slot term -alpha_Z * Z_n(t)b_n(t) is restored.

    Returns:
        formulation_reward, slot_dpp_cost (= -formulation_reward)
    """
    prev_z = np.asarray(info.get("prev_Z", []), dtype=np.float64)
    playback = np.asarray(info.get("playback", []), dtype=np.float64)
    if prev_z.shape != (int(env_cfg.num_user),):
        raise ValueError(f"prev_Z shape mismatch: {prev_z.shape}")
    if playback.shape != (int(env_cfg.num_user),):
        raise ValueError(f"playback shape mismatch: {playback.shape}")

    queue_playback_term = float(env_cfg.alpha_Z) * float(
        np.dot(prev_z, playback)
    )
    formulation_reward = float(env_reward) - queue_playback_term
    slot_dpp_cost = -formulation_reward

    if not np.isfinite(formulation_reward):
        raise RuntimeError("Formulation reward is NaN or Inf.")
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
    coordinate_sweeps: int
    candidate_requests: int
    unique_candidates: int
    finite_candidates: int
    forecast_seconds: float
    policy_seconds: float
    env_seconds: float
    mean_gpu_batch: float


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

def _forecast_worker_loop(connection) -> None:
    """
    Slow-DPP shadow environment 담당 persistent Worker Process.
    Pipe IPC 통신 시 Dict 객체를 전송하지 않고 순수 Float32 NumPy 배열만 송수신하여
    Pickle 직렬화 병목을 완벽히 제거합니다.
    """
    shadows: list[Env] = []
    codec: Optional[FastActionCodec] = None
    train_cfg: Optional[FastTrainConfig] = None

    try:
        while True:
            command, payload = connection.recv()

            if command == "close":
                break

            if command == "init":
                base_env, actions, cfg, scenario_idx = payload
                train_cfg = cfg
                codec = FastActionCodec(base_env.cfg)

                shadows = [
                    _make_shadow_env(
                        base_env,
                        action,
                        train_cfg,
                        int(scenario_idx),
                    )
                    for action in actions
                ]

                # Dict 관측값을 Worker 내부에서 즉시 Float32 Matrix로 변환
                obs_list = []
                mask_list = []
                for shadow in shadows:
                    dict_obs = shadow.get_fast_obs()
                    obs_list.append(flatten_fast_obs(dict_obs).astype(np.float32))
                    mask_list.append(codec.build_base_action_mask(dict_obs).astype(np.float32))

                obs_matrix = np.stack(obs_list, axis=0)
                mask_matrix = np.stack(mask_list, axis=0)

                connection.send(("ready", obs_matrix, mask_matrix))
                continue

            if command != "step":
                raise RuntimeError(f"Unknown forecast-worker command: {command}")

            if train_cfg is None or codec is None or not shadows:
                raise RuntimeError("Forecast worker received step before init.")

            raw_actions_chunk, slot_idx = payload  # (K, action_dim) float32 matrix

            count = len(shadows)
            next_obs_list = []
            next_mask_list = []
            slot_costs = np.zeros(count, dtype=np.float64)
            invalid = np.zeros(count, dtype=bool)

            # 각 shadow Env에 대해 디코딩 및 step 실행
            for idx, shadow in enumerate(shadows):
                dict_obs = shadow.get_fast_obs()
                env_action = codec.decode(raw_actions_chunk[idx], dict_obs)

                next_obs, reward, terminated, truncated, info = split_env_step(
                    shadow.step(env_action)
                )

                expected_boundary = int(slot_idx) == int(train_cfg.dpp_forecast_horizon) - 1
                actual_boundary = bool(info.get("is_round_boundary", False))
                if actual_boundary != expected_boundary:
                    raise RuntimeError(f"Forecast boundary mismatch: slot={slot_idx}")

                if terminated or truncated:
                    invalid[idx] = True

                _, slot_cost = formulation_reward_from_env_step(
                    env_reward=reward,
                    info=info,
                    env_cfg=shadow.cfg,
                )
                slot_costs[idx] = float(slot_cost)

                if bool(train_cfg.dpp_reject_forecast_outage) and np.any(
                    np.asarray(info.get("outage", []), dtype=np.int32) > 0
                ):
                    invalid[idx] = True

                # 다음 스텝용 관측값/마스크를 Worker 내에서 즉시 Vectorize
                next_dict_obs = shadow.get_fast_obs()
                next_obs_list.append(flatten_fast_obs(next_dict_obs).astype(np.float32))
                next_mask_list.append(codec.build_base_action_mask(next_dict_obs).astype(np.float32))

            next_obs_matrix = np.stack(next_obs_list, axis=0)
            next_mask_matrix = np.stack(next_mask_list, axis=0)

            # 순수 NumPy 배열만 전송 (Zero-Pickle Dict)
            connection.send(("step_result", next_obs_matrix, next_mask_matrix, slot_costs, invalid))

    except BaseException as exc:
        try:
            connection.send(("error", type(exc).__name__, str(exc)))
        except BaseException:
            pass
        raise
    finally:
        connection.close()

class _PersistentShadowProcessPool:
    def __init__(self, worker_count: int) -> None:
        self.worker_count = max(1, int(worker_count))
        self.context = mp.get_context("spawn")
        self.parents = []
        self.processes = []

        for worker_idx in range(self.worker_count):
            parent_conn, child_conn = self.context.Pipe(duplex=True)
            process = self.context.Process(
                target=_forecast_worker_loop,
                args=(child_conn,),
                name=f"dpp-shadow-{worker_idx}",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parents.append(parent_conn)
            self.processes.append(process)

    @staticmethod
    def _partition_sizes(count: int, workers: int) -> list[int]:
        base, remainder = divmod(int(count), int(workers))
        return [base + (1 if idx < remainder else 0) for idx in range(int(workers))]

    @staticmethod
    def _check_message(message, expected: str):
        if not isinstance(message, tuple) or not message:
            raise RuntimeError(f"Malformed worker message: {message!r}")
        if message[0] == "error":
            raise RuntimeError(f"Forecast worker failed: {message[1]}: {message[2]}")
        if message[0] != expected:
            raise RuntimeError(f"Unexpected worker message: expected={expected}, got={message[0]}")
        return message

    def rollout(
        self,
        *,
        base_env: Env,
        actions: Sequence[Dict[str, np.ndarray]],
        scenario_idx: int,
        train_cfg: FastTrainConfig,
        agent: FastPPOAgent,
    ) -> Tuple[np.ndarray, float, float, list[int]]:
        count = len(actions)
        if count <= 0:
            return np.zeros(0, dtype=np.float64), 0.0, 0.0, []

        active_workers = min(self.worker_count, count)
        sizes = self._partition_sizes(count, active_workers)
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)

        # 1. Worker 초기화 및 첫 관측값/마스크 Matrix 수신
        for worker_idx in range(active_workers):
            start, end = offsets[worker_idx], offsets[worker_idx + 1]
            self.parents[worker_idx].send(
                ("init", (base_env, list(actions[start:end]), train_cfg, int(scenario_idx)))
            )

        worker_obs_mats = []
        worker_mask_mats = []
        for worker_idx in range(active_workers):
            msg = self._check_message(self.parents[worker_idx].recv(), "ready")
            worker_obs_mats.append(msg[1])   # (K_w, obs_dim)
            worker_mask_mats.append(msg[2])  # (K_w, action_dim)

        obs_matrix = np.vstack(worker_obs_mats)
        mask_matrix = np.vstack(worker_mask_mats)

        accumulated = np.zeros(count, dtype=np.float64)
        invalid = np.zeros(count, dtype=bool)

        policy_seconds = 0.0
        env_seconds = 0.0
        gpu_batch_sizes: list[int] = []

        # 2. 3,600 Slot Zero-Dict High-Speed Loop
        for slot_idx in range(int(train_cfg.dpp_forecast_horizon)):
            # [GPU] Float32 Matrix 다이렉트 배치 추론 (Dict 변환 0회)
            policy_start = time.perf_counter()
            raw_actions_matrix = agent.select_env_actions_from_matrices_batch(
                obs_matrix=obs_matrix,
                mask_matrix=mask_matrix,
                deterministic=bool(train_cfg.dpp_deterministic_fast_forecast),
            )
            policy_seconds += time.perf_counter() - policy_start
            gpu_batch_sizes.append(count)

            # [CPU] Worker 분할 전송 (Float32 Array만 전달)
            env_start = time.perf_counter()
            for worker_idx in range(active_workers):
                start, end = offsets[worker_idx], offsets[worker_idx + 1]
                self.parents[worker_idx].send(
                    ("step", (raw_actions_matrix[start:end], int(slot_idx)))
                )

            next_obs_mats = []
            next_mask_mats = []

            for worker_idx in range(active_workers):
                start, end = offsets[worker_idx], offsets[worker_idx + 1]
                msg = self._check_message(self.parents[worker_idx].recv(), "step_result")

                next_obs_mats.append(msg[1])
                next_mask_mats.append(msg[2])
                accumulated[start:end] += msg[3]
                invalid[start:end] |= msg[4]

            obs_matrix = np.vstack(next_obs_mats)
            mask_matrix = np.vstack(next_mask_mats)
            env_seconds += time.perf_counter() - env_start

        accumulated[invalid] = np.inf
        return accumulated, float(policy_seconds), float(env_seconds), gpu_batch_sizes

    def close(self) -> None:
        for parent in self.parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
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
    ) -> None:
        self.env = env
        self.agent = agent
        self.train_cfg = train_cfg
        self.cache: Dict[bytes, float] = {}

        self.candidate_requests = 0
        self.finite_candidates = 0
        self.policy_seconds = 0.0
        self.env_seconds = 0.0
        self.gpu_batch_sizes: list[int] = []

        self.process_pool = _PersistentShadowProcessPool(
            int(train_cfg.dpp_forecast_workers)
        )

    def close(self) -> None:
        self.process_pool.close()

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
                base_env=self.env,
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
) -> SlowSelectionResult:
    if int(env.round_slot) != 0:
        raise RuntimeError(
            "Slow-DPP must run at round boundary, "
            f"got {env.round_slot}."
        )

    env_digest_before = _env_state_digest(env)
    model_digest_before = (
        _model_parameter_digest(agent)
    )

    normalizer_state_before = (
        copy.deepcopy(
            agent.obs_normalizer.state_dict()
        )
        if agent.obs_normalizer is not None
        else None
    )

    wall_start = time.perf_counter()

    agent.model.eval()

    evaluator = SlowDPPEvaluator(
        env,
        agent,
        train_cfg,
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
        _env_state_digest(env)
        != env_digest_before
    ):
        raise RuntimeError(
            "Slow forecast mutated "
            "the real environment."
        )

    if (
        _model_parameter_digest(agent)
        != model_digest_before
    ):
        raise RuntimeError(
            "Slow forecast mutated "
            "Fast policy parameters."
        )

    if agent.obs_normalizer is not None:
        after = (
            agent.obs_normalizer.state_dict()
        )

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
            "parallel_jacobi_complete_action_"
            "exact_acceptance"
        ),
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

    return {
        "delivery": float(fast.get("sum_delivery", 0.0)),
        "quality": float(fast.get("sum_quality", 0.0)),
        "quality_degradation": float(
            fast.get("sum_quality_degradation", 0.0)
        ),
        "consumed_soc": float(fast.get("sum_consumed_soc", 0.0)),
        "charged_soc": float(fast.get("sum_charged_soc", 0.0)),
        "stall": float(np.sum(stall)) if stall.size else 0.0,
        "min_soc": float(np.min(next_e)) if next_e.size else 0.0,
        "outage_slots": float(np.sum(outage)) if outage.size else 0.0,
        "charging_slots": (
            float(np.sum(charging)) if charging.size else 0.0
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
    del smooth_window
    data = _read_scalar_csv(run_dir / "logs" / "episodes.csv")
    if "episode" not in data:
        return

    x = np.asarray(data["episode"], dtype=np.float64)
    for key in (
        "episode_formulation_reward",
        "episode_delivery",
        "episode_stall",
        "episode_prediction_gap_mean",
    ):
        if key not in data or len(data[key]) != len(x):
            continue
        plt.figure()
        plt.plot(x, np.asarray(data[key], dtype=np.float64))
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "figures" / f"{key}.png", dpi=200)
        plt.close()


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
    if train_cfg.legacy_transfer:
        if checkpoint is None:
            raise ValueError("legacy_transfer requires checkpoint.")
        agent.load_legacy_transfer(checkpoint)
    elif train_cfg.resume:
        if checkpoint is None:
            raise ValueError("resume requires checkpoint.")
        agent.load(checkpoint, strict=True, load_optimizer=True)
    elif checkpoint is not None:
        # Pretrained initialization followed by online round-wise fine-tuning.
        agent.load(
            checkpoint,
            strict=True,
            load_optimizer=bool(train_cfg.load_optimizer_on_warm_start),
        )
    elif (
        train_cfg.slow_decision_mode == "dpp"
        and train_cfg.require_pretrained_fast_for_dpp
    ):
        raise ValueError("DPP mode requires a pretrained Fast-PPO checkpoint.")

    return agent


# ======================================================================
# Round execution
# ======================================================================
def _select_and_apply_slow_action(
    env: Env,
    agent: FastPPOAgent,
    train_cfg: FastTrainConfig,
    slow_rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    if train_cfg.slow_decision_mode == "dpp":
        selected = select_slow_action_dpp(env, agent, train_cfg)
        env.apply_slow_action(selected.action)
        return env.get_fast_obs(), {
            "predicted_round_cost": selected.predicted_round_cost,
            "candidate_requests": float(selected.candidate_requests),
            "unique_candidates": float(selected.unique_candidates),
            "finite_candidates": float(selected.finite_candidates),
            "coordinate_sweeps": float(selected.coordinate_sweeps),
            "forecast_seconds": selected.forecast_seconds,
            "forecast_policy_seconds": selected.policy_seconds,
            "forecast_env_seconds": selected.env_seconds,
            "forecast_mean_gpu_batch": selected.mean_gpu_batch,
            "num_rsu_links": float(
                np.sum(selected.action["rsu_scheduling"])
            ),
            "num_hired_uav": float(
                np.sum(selected.action["uav_hiring"])
            ),
            "num_uav_links": float(
                np.sum(selected.action["uav_scheduling"])
            ),
        }

    action = sample_random_slow_action(env, slow_rng, train_cfg)
    env.apply_slow_action(action)
    return env.get_fast_obs(), {
        "predicted_round_cost": float("nan"),
        "candidate_requests": 0.0,
        "unique_candidates": 0.0,
        "finite_candidates": 0.0,
        "coordinate_sweeps": 0.0,
        "forecast_seconds": 0.0,
        "forecast_policy_seconds": 0.0,
        "forecast_env_seconds": 0.0,
        "forecast_mean_gpu_batch": 0.0,
        "num_rsu_links": float(np.sum(action["rsu_scheduling"])),
        "num_hired_uav": float(np.sum(action["uav_hiring"])),
        "num_uav_links": float(np.sum(action["uav_scheduling"])),
    }


def _execute_real_round_train(
    env: Env,
    agent: FastPPOAgent,
    obs: Dict[str, Any],
    train_cfg: FastTrainConfig,
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
    if len(agent.buffer) != 0:
        raise RuntimeError("PPO buffer must be empty at real round start.")

    model_digest_before = _model_parameter_digest(agent)
    raw_obs_batch: list[np.ndarray] = []

    totals = {
        "formulation_reward": 0.0,
        "round_fast_cost": 0.0,
        "delivery": 0.0,
        "quality": 0.0,
        "quality_degradation": 0.0,
        "stall": 0.0,
        "consumed_soc": 0.0,
        "charged_soc": 0.0,
        "outage_slots": 0.0,
        "charging_slots": 0.0,
        "min_soc": float("inf"),
        "service_rate": 0.0,
        "requested_chunks": 0.0,
        "active_action_dims": 0.0,
        "active_action_ratio": 0.0,
        "action_saturation_ratio": 0.0,
    }

    real_start = time.perf_counter()
    agent.model.train()

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
        totals["formulation_reward"] += formulation_reward
        totals["round_fast_cost"] += slot_cost
        for key in (
            "delivery",
            "quality",
            "quality_degradation",
            "stall",
            "consumed_soc",
            "charged_soc",
            "outage_slots",
            "charging_slots",
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
    if _model_parameter_digest(agent) != model_digest_before:
        raise RuntimeError("Fast policy changed inside a real round.")

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
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    totals = {
        "formulation_reward": 0.0,
        "round_fast_cost": 0.0,
        "delivery": 0.0,
        "quality": 0.0,
        "quality_degradation": 0.0,
        "stall": 0.0,
        "outage_slots": 0.0,
        "min_soc": float("inf"),
    }
    agent.model.eval()

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
        totals["formulation_reward"] += reward
        totals["round_fast_cost"] += cost
        for key in (
            "delivery",
            "quality",
            "quality_degradation",
            "stall",
            "outage_slots",
        ):
            totals[key] += metrics[key]
        totals["min_soc"] = min(totals["min_soc"], metrics["min_soc"])
        obs = next_obs

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
    return obs, totals


# ======================================================================
# Train / evaluate
# ======================================================================
def train(train_cfg: FastTrainConfig) -> None:
    set_seed(
        int(train_cfg.seed),
        deterministic=bool(train_cfg.deterministic_torch),
    )
    env_cfg = build_env_config()
    _assert_joint_config(train_cfg, env_cfg)
    ppo_cfg = build_agent_ppo_config(train_cfg)

    env = Env(env_cfg)
    initial_obs, reset_info = split_env_reset(env.reset())
    agent = _initialize_agent(env, initial_obs, train_cfg, ppo_cfg)

    run_dir = make_run_dir(train_cfg, env_cfg)
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

    print("=" * 100, flush=True)
    print("[SLOW-DPP + FAST-PPO TRAIN START]", flush=True)
    print(f"branch_basis   : feat/fast-hrl@bf31c6f", flush=True)
    print(f"run_dir        : {run_dir}", flush=True)
    print(f"device         : {agent.device}", flush=True)
    print(f"slow_mode      : {train_cfg.slow_decision_mode}", flush=True)
    print(f"slow_T         : {env_cfg.slow_T}", flush=True)
    print(f"rollout_slots  : {ppo_cfg.rollout_steps}", flush=True)
    print(f"checkpoint     : {_resolve_checkpoint(train_cfg.checkpoint)}", flush=True)
    print(f"reset_info     : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    global_real_slot = 0
    global_round = 0
    update_idx = 0

    for episode_idx in range(1, int(train_cfg.num_episodes) + 1):
        env.cfg.move_prob = get_episode_move_prob(train_cfg, episode_idx)
        obs, _ = split_env_reset(env.reset())

        ep = {
            "reward": 0.0,
            "delivery": 0.0,
            "quality": 0.0,
            "quality_degradation": 0.0,
            "stall": 0.0,
            "outage_slots": 0.0,
            "forecast_seconds": 0.0,
            "real_seconds": 0.0,
            "update_seconds": 0.0,
            "prediction_gap_sum": 0.0,
            "prediction_gap_count": 0,
            "min_soc": float("inf"),
        }

        for round_in_episode in range(1, int(train_cfg.rounds_per_episode) + 1):
            if int(env.round_slot) != 0:
                raise RuntimeError("Episode loop is not at a round boundary.")

            obs, slow_info = _select_and_apply_slow_action(
                env,
                agent,
                train_cfg,
                slow_rng,
            )
            policy_version = update_idx

            obs, realized, update_logs = _execute_real_round_train(
                env,
                agent,
                obs,
                train_cfg,
            )
            update_idx += 1
            global_round += 1
            global_real_slot += int(env_cfg.slow_T)

            predicted = float(slow_info["predicted_round_cost"])
            realized_cost = float(realized["realized_round_cost"])
            gap = (
                realized_cost - predicted
                if np.isfinite(predicted)
                else float("nan")
            )

            round_row = {
                "global_round": global_round,
                "episode": episode_idx,
                "round_in_episode": round_in_episode,
                "policy_version": policy_version,
                "global_real_slot": global_real_slot,
                **slow_info,
                **realized,
                "prediction_gap": gap,
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

            ep["reward"] += realized["formulation_reward"]
            ep["delivery"] += realized["delivery"]
            ep["quality"] += realized["quality"]
            ep["quality_degradation"] += realized[
                "quality_degradation"
            ]
            ep["stall"] += realized["stall"]
            ep["outage_slots"] += realized["outage_slots"]
            ep["forecast_seconds"] += slow_info["forecast_seconds"]
            ep["real_seconds"] += realized["real_round_seconds"]
            ep["update_seconds"] += realized["ppo_update_seconds"]
            ep["min_soc"] = min(ep["min_soc"], realized["min_soc"])
            if np.isfinite(gap):
                ep["prediction_gap_sum"] += gap
                ep["prediction_gap_count"] += 1

            print(
                "[ROUND] "
                f"ep={episode_idx}/{train_cfg.num_episodes} "
                f"r={round_in_episode}/{train_cfg.rounds_per_episode} "
                f"policy_v={policy_version} "
                f"pred={predicted:.4f} "
                f"real={realized_cost:.4f} "
                f"gap={gap:.4f} "
                f"candidates={int(slow_info['unique_candidates'])} "
                f"forecast_s={slow_info['forecast_seconds']:.2f} "
                f"real_sps={realized['real_slots_per_second']:.2f} "
                f"kl={update_logs['approx_kl']:.6f}",
                flush=True,
            )

        if not np.isfinite(ep["min_soc"]):
            ep["min_soc"] = 0.0
        gap_mean = (
            ep["prediction_gap_sum"] / ep["prediction_gap_count"]
            if ep["prediction_gap_count"] > 0
            else float("nan")
        )
        episode_logger.write(
            {
                "episode": episode_idx,
                "global_round": global_round,
                "global_real_slot": global_real_slot,
                "move_prob": float(env.cfg.move_prob),
                "episode_formulation_reward": ep["reward"],
                "episode_delivery": ep["delivery"],
                "episode_quality": ep["quality"],
                "episode_quality_degradation": ep[
                    "quality_degradation"
                ],
                "episode_stall": ep["stall"],
                "episode_outage_slots": ep["outage_slots"],
                "episode_min_soc": ep["min_soc"],
                "episode_forecast_seconds": ep["forecast_seconds"],
                "episode_real_seconds": ep["real_seconds"],
                "episode_update_seconds": ep["update_seconds"],
                "episode_prediction_gap_mean": gap_mean,
            }
        )

        print(
            "[EPISODE] "
            f"ep={episode_idx}/{train_cfg.num_episodes} "
            f"reward={ep['reward']:.4f} "
            f"delivery={ep['delivery']:.1f} "
            f"stall={ep['stall']:.1f} "
            f"gap_mean={gap_mean:.4f} "
            f"outage_slots={ep['outage_slots']:.0f}",
            flush=True,
        )

        if (
            int(train_cfg.save_every_episodes) > 0
            and episode_idx % int(train_cfg.save_every_episodes) == 0
        ):
            checkpoint_path = (
                run_dir
                / "checkpoints"
                / f"fast_ppo_joint_ep{episode_idx}.pt"
            )
            agent.save(
                checkpoint_path,
                extra={
                    "episode": episode_idx,
                    "global_round": global_round,
                    "global_real_slot": global_real_slot,
                    "update_idx": update_idx,
                    "slow_decision_mode": train_cfg.slow_decision_mode,
                    "env_config": (
                        env_cfg.as_dict()
                        if hasattr(env_cfg, "as_dict")
                        else asdict(env_cfg)
                    ),
                    "train_config": train_cfg.to_dict(),
                    "ppo_config": asdict(ppo_cfg),
                },
            )
            print(f"[SAVE] {checkpoint_path}", flush=True)

        if (
            int(train_cfg.plot_every_episodes) > 0
            and episode_idx % int(train_cfg.plot_every_episodes) == 0
        ):
            save_training_plots(
                run_dir,
                smooth_window=int(train_cfg.plot_smooth_window),
            )

    if len(agent.buffer) != 0:
        raise RuntimeError("Joint training ended with a nonempty PPO buffer.")

    final_path = run_dir / "checkpoints" / "fast_ppo_joint_final.pt"
    agent.save(
        final_path,
        extra={
            "episode": int(train_cfg.num_episodes),
            "global_round": global_round,
            "global_real_slot": global_real_slot,
            "update_idx": update_idx,
            "slow_decision_mode": train_cfg.slow_decision_mode,
            "train_config": train_cfg.to_dict(),
            "env_config": (
                env_cfg.as_dict()
                if hasattr(env_cfg, "as_dict")
                else asdict(env_cfg)
            ),
            "ppo_config": asdict(ppo_cfg),
        },
    )
    save_json(
        {
            "final_checkpoint": str(final_path),
            "global_round": global_round,
            "global_real_slot": global_real_slot,
            "updates": update_idx,
        },
        run_dir / "train_summary.json",
    )
    save_training_plots(run_dir)
    print(f"[DONE] final checkpoint: {final_path}", flush=True)


def evaluate(train_cfg: FastTrainConfig) -> None:
    set_seed(int(train_cfg.seed), deterministic=True)
    env_cfg = build_env_config()
    _assert_joint_config(train_cfg, env_cfg)
    env_cfg.move_prob = float(train_cfg.mobility_curriculum[-1][1])
    ppo_cfg = build_agent_ppo_config(train_cfg)

    env = Env(env_cfg)
    obs, _ = split_env_reset(env.reset())
    agent = _initialize_agent(env, obs, train_cfg, ppo_cfg)
    run_dir = make_run_dir(train_cfg, env_cfg)
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

    for episode_idx in range(1, int(train_cfg.eval_episodes) + 1):
        obs, _ = split_env_reset(env.reset())
        totals = {
            "reward": 0.0,
            "delivery": 0.0,
            "quality": 0.0,
            "quality_degradation": 0.0,
            "stall": 0.0,
            "outage_slots": 0.0,
            "prediction_gap_sum": 0.0,
            "prediction_gap_count": 0,
        }

        for _ in range(int(train_cfg.eval_rounds_per_episode)):
            obs, slow_info = _select_and_apply_slow_action(
                env,
                agent,
                train_cfg,
                slow_rng,
            )
            obs, realized = _execute_real_round_eval(env, agent, obs)
            predicted = float(slow_info["predicted_round_cost"])
            if np.isfinite(predicted):
                totals["prediction_gap_sum"] += (
                    realized["realized_round_cost"] - predicted
                )
                totals["prediction_gap_count"] += 1
            totals["reward"] += realized["formulation_reward"]
            for key in (
                "delivery",
                "quality",
                "quality_degradation",
                "stall",
                "outage_slots",
            ):
                totals[key] += realized[key]

        gap_mean = (
            totals["prediction_gap_sum"]
            / totals["prediction_gap_count"]
            if totals["prediction_gap_count"] > 0
            else float("nan")
        )
        logger.write(
            {
                "episode": episode_idx,
                "reward": totals["reward"],
                "delivery": totals["delivery"],
                "quality": totals["quality"],
                "quality_degradation": totals["quality_degradation"],
                "stall": totals["stall"],
                "outage_slots": totals["outage_slots"],
                "prediction_gap_mean": gap_mean,
            }
        )
        print(
            "[EVAL] "
            f"ep={episode_idx}/{train_cfg.eval_episodes} "
            f"reward={totals['reward']:.4f} "
            f"delivery={totals['delivery']:.1f} "
            f"stall={totals['stall']:.1f} "
            f"gap={gap_mean:.4f}",
            flush=True,
        )


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
