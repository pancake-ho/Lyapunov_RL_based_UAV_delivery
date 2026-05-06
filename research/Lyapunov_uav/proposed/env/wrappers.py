from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig

from .action_types import EnvAction
from .env import Env
from .spaces import BoxSpace, DictSpace, MultiBinarySpace


def _fast_observation_space(cfg: EnvConfig) -> DictSpace:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    int_hi = np.iinfo(np.int32).max

    return DictSpace(
        spaces={
            "Q": BoxSpace(0.0, float(cfg.max_queue), (n,), np.float32),
            "Z": BoxSpace(0.0, float(cfg.max_queue), (n,), np.float32),
            "E": BoxSpace(0.0, float(cfg.battery.e_max), (u,), np.float32),
            "Y": BoxSpace(0.0, float(cfg.battery.e_max), (u,), np.float32),
            "uav_hiring": MultiBinarySpace((u,)),
            "rsu_scheduling": MultiBinarySpace((m, n)),
            "uav_scheduling": MultiBinarySpace((u, n)),
            "requested_content": BoxSpace(0, int(cfg.num_video - 1), (n,), np.int32),
            "uav_cached_content": BoxSpace(0, int(cfg.num_video - 1), (u,), np.int32),
            "outage": MultiBinarySpace((u,)),
            "charging_state": MultiBinarySpace((u,)),
            "round_idx": BoxSpace(0, int_hi, (1,), np.int32),
            "round_slot": BoxSpace(0, max(0, int(cfg.slow_T - 1)), (1,), np.int32),
            "time": BoxSpace(0, int_hi, (1,), np.int32),
        },
    )


def _slow_observation_space(cfg: EnvConfig) -> DictSpace:
    n = int(cfg.num_user)
    u = int(cfg.num_uav)
    int_hi = np.iinfo(np.int32).max

    return DictSpace(
        spaces={
            "Q": BoxSpace(0.0, float(cfg.max_queue), (n,), np.float32),
            "Z": BoxSpace(0.0, float(cfg.max_queue), (n,), np.float32),
            "E": BoxSpace(0.0, float(cfg.battery.e_max), (u,), np.float32),
            "Y": BoxSpace(0.0, float(cfg.battery.e_max), (u,), np.float32),
            "requested_content": BoxSpace(0, int(cfg.num_video - 1), (n,), np.int32),
            "uav_cached_content": BoxSpace(0, int(cfg.num_video - 1), (u,), np.int32),
            "outage": MultiBinarySpace((u,)),
            "round_idx": BoxSpace(0, int_hi, (1,), np.int32),
        },
    )


def _fast_action_space(cfg: EnvConfig) -> DictSpace:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)

    return DictSpace(
        spaces={},
        optional_spaces={
            "rsu_chunks": BoxSpace(0, int(cfg.chunk), (m, n), np.int32),
            "rsu_layers": BoxSpace(0, int(cfg.layer), (m, n), np.int32),
            "uav_chunks": BoxSpace(0, int(cfg.chunk), (u, n), np.int32),
            "uav_layers": BoxSpace(0, int(cfg.layer), (u, n), np.int32),
            "uav_power": BoxSpace(0.0, float(cfg.battery.max_tx_power), (u, n), np.float32),
        },
        allow_extra=False,
    )


def _slow_action_space(cfg: EnvConfig) -> DictSpace:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)

    return DictSpace(
        spaces={},
        optional_spaces={
            "rsu_scheduling": MultiBinarySpace((m, n)),
            "uav_hiring": MultiBinarySpace((u,)),
            "uav_scheduling": MultiBinarySpace((u, n)),
            "rsu_schedule": MultiBinarySpace((m, n)),
            "uav_schedule": MultiBinarySpace((u, n)),
        },
        allow_extra=False,
    )


class FastEnv:
    """
    Slot-level fast-timescale wrapper.

    Slow decision은 core Env.apply_slow_action()으로 별도 갱신하고,
    FastEnv.step()은 slot-level action만 전달한다.
    """
    def __init__(self, config: EnvConfig, core_env: Optional[Env] = None):
        self.cfg = config
        self.core = core_env if core_env is not None else Env(config)
        self.observation_space = _fast_observation_space(config)
        self.action_space = _fast_action_space(config)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.core.cfg.seed = int(seed)
            self.core.rng = np.random.default_rng(seed)
        obs, info = self.core.reset()
        return obs, info

    def step(self, action: EnvAction):
        return self.core.step(action)

    def apply_slow_action(self, action: EnvAction):
        if int(self.core.round_slot) != 0:
            raise RuntimeError("slow action can only be applied at a round boundary.")
        return self.core.apply_slow_action(action)

    def get_fast_obs(self) -> Dict[str, np.ndarray]:
        return self.core.get_fast_obs()


class SlowEnv:
    """
    Round-level slow-timescale wrapper.

    1단계에서는 slow decision 적용과 slow observation 계약만 제공한다.
    round rollout aggregation은 다음 단계에서 붙인다.
    """
    def __init__(self, config: EnvConfig, core_env: Optional[Env] = None):
        self.cfg = config
        self.core = core_env if core_env is not None else Env(config)
        self.observation_space = _slow_observation_space(config)
        self.action_space = _slow_action_space(config)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.core.cfg.seed = int(seed)
            self.core.rng = np.random.default_rng(seed)
        _, info = self.core.reset()
        return self.core.get_slow_obs(), info

    def step(self, action: EnvAction):
        if int(self.core.round_slot) != 0:
            raise RuntimeError("slow action can only be applied at a round boundary.")
        slow_action = self.core.apply_slow_action(action)
        info = {
            "slow_action": slow_action,
            "round_idx": int(self.core.round_idx),
            "round_slot": int(self.core.round_slot),
            "round_pending_fast_rollout": True,
            "terminated": False,
            "truncated": False,
        }
        return self.core.get_slow_obs(), 0.0, False, False, info


RoundEnv = SlowEnv
