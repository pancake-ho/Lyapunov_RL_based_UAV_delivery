from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np

try:
    from proposed.config import EnvConfig
    from proposed.env import (
        FastEnv,
        SlowEnv,
        VectorSpec,
        decode_fast_action_vector,
        decode_slow_action_vector,
        fast_action_spec,
        fast_obs_spec,
        flatten_fast_obs,
        flatten_slow_obs,
        slow_action_spec,
        slow_obs_spec,
    )
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    proposed_dir = Path(__file__).resolve().parents[3]
    proposed_dir_str = str(proposed_dir)
    if proposed_dir_str not in sys.path:
        sys.path.insert(0, proposed_dir_str)
    from config import EnvConfig
    from env import (
        FastEnv,
        SlowEnv,
        VectorSpec,
        decode_fast_action_vector,
        decode_slow_action_vector,
        fast_action_spec,
        fast_obs_spec,
        flatten_fast_obs,
        flatten_slow_obs,
        slow_action_spec,
        slow_obs_spec,
    )

PolicyLevel = Literal["slow", "fast"]


@dataclass(frozen=True)
class PPOInterfaceSpec:
    """
    PPO가 보는 1D state/action contract.
    """
    level: PolicyLevel
    obs_dim: int
    action_dim: int
    obs: Dict[str, slice]
    action: Dict[str, slice]


class PPOEnvAdapter:
    """
    SlowEnv/FastEnv dict contract와 PPOAgent vector contract를 연결한다.

    slow adapter는 round-level decision만 decode하고, fast adapter는 slot-level
    delivery/power vector만 decode한다. UAV charging은 env의 threshold 기반
    rule-based hook에 남겨둔다.
    """
    def __init__(self, env: SlowEnv | FastEnv, level: PolicyLevel):
        if level not in ("slow", "fast"):
            raise ValueError(f"level must be 'slow' or 'fast', got {level!r}")
        self.env = env
        self.level = level
        self.cfg: EnvConfig = env.cfg

        if level == "slow":
            self.observation_spec: VectorSpec = slow_obs_spec(self.cfg)
            self.action_spec: VectorSpec = slow_action_spec(self.cfg)
        else:
            self.observation_spec = fast_obs_spec(self.cfg)
            self.action_spec = fast_action_spec(self.cfg)

        self.spec = PPOInterfaceSpec(
            level=level,
            obs_dim=self.observation_spec.dim,
            action_dim=self.action_spec.dim,
            obs=dict(self.observation_spec.slices),
            action=dict(self.action_spec.slices),
        )
        self.obs_dim = self.spec.obs_dim
        self.action_dim = self.spec.action_dim

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset()
        return self.flatten_obs(obs), info

    def flatten_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.level == "slow":
            return flatten_slow_obs(obs, self.cfg)
        return flatten_fast_obs(obs, self.cfg)

    def decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        if self.level == "slow":
            return decode_slow_action_vector(action, self.cfg)
        return decode_fast_action_vector(action, self.cfg)

    def step_vector(self, action: np.ndarray):
        decoded = self.decode_action(action)
        if not self.env.action_space.contains(decoded):
            raise ValueError(f"{self.level} action vector decoded outside action_space")
        obs, reward, terminated, truncated, info = self.env.step(decoded)
        return self.flatten_obs(obs), reward, terminated, truncated, info


def make_slow_adapter(env: SlowEnv) -> PPOEnvAdapter:
    return PPOEnvAdapter(env, "slow")


def make_fast_adapter(env: FastEnv) -> PPOEnvAdapter:
    return PPOEnvAdapter(env, "fast")
