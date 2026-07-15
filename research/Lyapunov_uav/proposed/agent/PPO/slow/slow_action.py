from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from config import EnvConfig


@dataclass(frozen=True)
class SlowActionSpec:
    """
    PPO가 출력할 Slow-Timescale raw action vector의 구조를 정의함.

    현재 시나리오 기준:
        y_mn(r)    : RSU-user scheduling, shape (M, N)
        mu_u(r)    : UAV hiring, shape (U,)
        phi_un(r)  : UAV-user scheduling, shape (U, N)
    """
    num_rsu: int
    num_user: int
    num_uav: int
    
    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "SlowActionSpec":
        """
        config로부터 state 반환을 수행.
        """
        return cls(
            num_rsu=int(cfg.num_rsu),
            num_user=int(cfg.num_user),
            num_uav=int(cfg.num_uav),
        )
    
    @property
    def rsu_shape(self) -> Tuple[int, int]:
        """
        RSU와 관련된 action의 shape을 반환.
        """
        return (self.num_rsu, self.num_user)
    
    @property
    def hiring_shape(self) -> Tuple[int]:
        """
        UAV 고용 결정의 shape을 반환.
        """
        return (self.num_uav,)
    
    @property
    def uav_shape(self) -> Tuple[int, int]:
        """
        UAV와 관련된 action의 shape을 반환.
        """
        return (self.num_uav, self.num_user)
    
    @property
    def rsu_dim(self) -> int:
        return self.num_rsu * self.num_user

    @property
    def hiring_dim(self) -> int:
        return self.num_uav

    @property
    def uav_dim(self) -> int:
        return self.num_uav * self.num_user

    @property
    def action_dim(self) -> int:
        return self.rsu_dim + self.hiring_dim + self.uav_dim

    @property
    def rsu_slice(self) -> slice:
        start = self.hiring_dim
        return slice(start, start + self.rsu_dim)

    @property
    def hiring_slice(self) -> slice:
        return slice(0, self.hiring_dim)

    @property
    def uav_slice(self) -> slice:
        start = self.hiring_dim + self.rsu_dim
        return slice(start, start + self.uav_dim)
    

class SlowActionCodec:
    """
    Slow-timescale PPO action codec.

    PPO policy output은 다음과 같이 env와 호환되는 dict로 변환:
        {
            "rsu_scheduling": np.ndarray[int32], shape (M, N),
            "uav_hiring": np.ndarray[int32], shape (U,),
            "uav_scheduling": np.ndarray[int32], shape (U, N),
        }
    """
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.spec = SlowActionSpec.from_config(cfg)
    
    @property
    def action_dim(self) -> int:
        """
        spec의 action dim을 반환.
        """
        return self.spec.action_dim
    
    @property
    def hiring_shape(self) -> Tuple[int]:
        """
        spec의 hiring action dim을 반환.
        """
        return self.spec.hiring_shape
    
    def _require_binary_flat_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape != (self.action_dim,):
            raise ValueError(
                "slow action shape mismatch: "
                f"expected={(self.action_dim,)}, got={arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("slow action contains NaN or Inf.")

        is_zero = np.isclose(arr, 0.0, atol=1e-6)
        is_one = np.isclose(arr, 1.0, atol=1e-6)
        if not np.all(is_zero | is_one):
            bad = np.flatnonzero(~(is_zero | is_one))[:10]
            raise ValueError(
                "slow action must be a binary Bernoulli sample. "
                f"bad_idx={bad.tolist()}, bad_value={arr[bad].tolist()}"
            )
        return (arr >= 0.5).astype(np.int32)

    def split(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        binary = self._require_binary_flat_action(action)
        s = self.spec
        return {
            "rsu_scheduling": binary[s.rsu_slice].reshape(s.rsu_shape),
            "uav_hiring": binary[s.hiring_slice].reshape(s.hiring_shape),
            "uav_scheduling": binary[s.uav_slice].reshape(s.uav_shape),
        }
    
    def flatten(
        self,
        rsu_scheduling: np.ndarray,
        uav_hiring: np.ndarray,
        uav_scheduling: np.ndarray,
    ) -> np.ndarray:
        s = self.spec
        rsu = np.asarray(rsu_scheduling, dtype=np.float32)
        hiring = np.asarray(uav_hiring, dtype=np.float32)
        uav = np.asarray(uav_scheduling, dtype=np.float32)
        if rsu.shape != s.rsu_shape:
            raise ValueError(f"rsu_scheduling must have shape {s.rsu_shape}.")
        if hiring.shape != s.hiring_shape:
            raise ValueError(f"uav_hiring must have shape {s.hiring_shape}.")
        if uav.shape != s.uav_shape:
            raise ValueError(f"uav_scheduling must have shape {s.uav_shape}.")
        return np.concatenate(
            [hiring.reshape(-1), rsu.reshape(-1), uav.reshape(-1)], axis=0
        ).astype(np.float32)
    
    def _parse_user_region(self, obs: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(obs, Mapping):
            raise TypeError(f"slow obs must be a mapping, got {type(obs)}")
        if "user_region" not in obs:
            raise KeyError("slow obs must contain user_region for x(r).")

        region = np.asarray(obs["user_region"], dtype=np.int32).reshape(-1)
        if region.shape != (self.spec.num_user,):
            raise ValueError(
                "user_region shape mismatch: "
                f"expected={(self.spec.num_user,)}, got={region.shape}"
            )
        if np.any(region < 0) or np.any(region >= self.spec.num_rsu):
            raise ValueError(
                f"user_region must be in [0, {self.spec.num_rsu - 1}]."
            )
        return region

    @staticmethod
    def _context_get(
        context: Optional[Any], obs: Mapping[str, Any], name: str
    ) -> Any:
        if isinstance(context, Mapping) and name in context:
            return context[name]
        if context is not None and hasattr(context, name):
            return getattr(context, name)
        if name in obs:
            return obs[name]
        return None

    def _parse_content_context(
        self, obs: Mapping[str, Any], context: Optional[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        requested = self._context_get(context, obs, "requested_content")
        cached = self._context_get(context, obs, "uav_cached_content")
        if requested is None or cached is None:
            raise KeyError(
                "slow action masking requires requested_content and "
                "uav_cached_content; pass context=env."
            )

        requested_arr = np.asarray(requested, dtype=np.int32).reshape(-1)
        cached_arr = np.asarray(cached, dtype=np.int32).reshape(-1)
        if requested_arr.shape != (self.spec.num_user,):
            raise ValueError(
                "requested_content shape mismatch: "
                f"expected={(self.spec.num_user,)}, got={requested_arr.shape}"
            )
        if cached_arr.shape != (self.spec.num_uav,):
            raise ValueError(
                "uav_cached_content shape mismatch: "
                f"expected={(self.spec.num_uav,)}, got={cached_arr.shape}"
            )

        num_video = int(self.cfg.num_video)
        if np.any(requested_arr < 0) or np.any(requested_arr >= num_video):
            raise ValueError(f"requested_content must be in [0, {num_video - 1}].")
        if np.any(cached_arr < 0) or np.any(cached_arr >= num_video):
            raise ValueError(f"uav_cached_content must be in [0, {num_video - 1}].")
        return requested_arr, cached_arr

    def _region_mask_rsu(self, user_region: np.ndarray) -> np.ndarray:
        rsu_idx = np.arange(self.spec.num_rsu, dtype=np.int32)[:, None]
        return (rsu_idx == user_region[None, :]).astype(np.int32)

    def _region_mask_uav(self, user_region: np.ndarray) -> np.ndarray:
        uav_idx = np.arange(self.spec.num_uav, dtype=np.int32)[:, None]
        return (uav_idx == user_region[None, :]).astype(np.int32)

    @staticmethod
    def _cache_match_mask(
        requested_content: np.ndarray, uav_cached_content: np.ndarray
    ) -> np.ndarray:
        return (
            uav_cached_content[:, None] == requested_content[None, :]
        ).astype(np.int32)

    def build_static_action_mask(
        self,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Return the state/context feasibility mask before y and mu are sampled.

        y: only the user's current RSU region.
        mu: every UAV is a valid hiring decision.
        phi: current UAV region and cache match; hiring/residual masks are
             added conditionally by SlowActorCritic.act().
        """
        region = self._parse_user_region(obs)
        requested, cached = self._parse_content_context(obs, context)
        rsu_mask = self._region_mask_rsu(region)
        hiring_mask = np.ones(self.spec.hiring_shape, dtype=np.int32)
        uav_mask = self._region_mask_uav(region) * self._cache_match_mask(
            requested, cached
        )
        return self.flatten(rsu_mask, hiring_mask, uav_mask)

    def build_effective_action_mask(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> np.ndarray:
        """Return the exact mask for p(y)p(mu)p(phi|y,mu)."""
        parts = self.split(action)
        static = self.split(self.build_static_action_mask(obs, context=context))

        rsu = parts["rsu_scheduling"] * static["rsu_scheduling"]
        hiring = parts["uav_hiring"] * static["uav_hiring"]
        residual = 1 - (rsu.sum(axis=0) > 0).astype(np.int32)
        uav_mask = (
            static["uav_scheduling"]
            * hiring[:, None]
            * residual[None, :]
        )
        return self.flatten(
            static["rsu_scheduling"], static["uav_hiring"], uav_mask
        )

    def decode_with_info(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        raw = self._require_binary_flat_action(action)
        effective_mask = self.build_effective_action_mask(
            raw, obs, context=context
        ).astype(np.int32)
        effective = raw * effective_mask
        env_action = self.split(effective)

        rsu = env_action["rsu_scheduling"].astype(np.int32, copy=False)
        hiring = env_action["uav_hiring"].astype(np.int32, copy=False)
        uav = env_action["uav_scheduling"].astype(np.int32, copy=False)
        per_user = rsu.sum(axis=0) + uav.sum(axis=0)
        if np.any(per_user > 1):
            raise RuntimeError("slow scheduling exclusivity projection failed.")

        projected_bits = raw * (1 - effective_mask)
        static_mask = self.build_static_action_mask(obs, context=context)
        info: Dict[str, Any] = {
            "raw_rsu_links": int(self.split(raw)["rsu_scheduling"].sum()),
            "effective_rsu_links": int(rsu.sum()),
            "raw_hired_uav": int(self.split(raw)["uav_hiring"].sum()),
            "effective_hired_uav": int(hiring.sum()),
            "raw_uav_links": int(self.split(raw)["uav_scheduling"].sum()),
            "effective_uav_links": int(uav.sum()),
            "projection_count": int(projected_bits.sum()),
            "static_action_dims": int(static_mask.sum()),
            "active_action_dims": int(effective_mask.sum()),
            "active_action_ratio": float(effective_mask.mean()),
            "num_scheduled_users": int(np.sum(per_user > 0)),
            "num_residual_users": int(np.sum(per_user == 0)),
            "action_mask": effective_mask.astype(np.float32, copy=True),
        }
        return {
            "rsu_scheduling": rsu,
            "uav_hiring": hiring,
            "uav_scheduling": uav,
        }, info

    def decode(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        env_action, _ = self.decode_with_info(action, obs, context=context)
        return env_action

    def zeros_env_action(self) -> Dict[str, np.ndarray]:
        return {
            "rsu_scheduling": np.zeros(self.spec.rsu_shape, dtype=np.int32),
            "uav_hiring": np.zeros(self.spec.hiring_shape, dtype=np.int32),
            "uav_scheduling": np.zeros(self.spec.uav_shape, dtype=np.int32),
        }

    def random_binary_action(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        rsu_user_prob: float = 0.5,
        uav_hire_prob: float = 0.35,
        uav_user_prob: float = 0.4,
    ) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng()
        probs = (rsu_user_prob, uav_hire_prob, uav_user_prob)
        if any(not 0.0 <= float(p) <= 1.0 for p in probs):
            raise ValueError("all random slow-action probabilities must be in [0, 1].")
        return self.flatten(
            generator.random(self.spec.rsu_shape) < float(rsu_user_prob),
            generator.random(self.spec.hiring_shape) < float(uav_hire_prob),
            generator.random(self.spec.uav_shape) < float(uav_user_prob),
        )

    def random_env_action(
        self,
        obs: Mapping[str, Any],
        *,
        rng: Optional[np.random.Generator] = None,
        context: Optional[Any] = None,
        rsu_user_prob: float = 0.5,
        uav_hire_prob: float = 0.35,
        uav_user_prob: float = 0.4,
    ) -> Dict[str, np.ndarray]:
        raw = self.random_binary_action(
            rng=rng,
            rsu_user_prob=rsu_user_prob,
            uav_hire_prob=uav_hire_prob,
            uav_user_prob=uav_user_prob,
        )
        mask = self.build_effective_action_mask(raw, obs, context=context)
        return self.decode(raw * mask, obs, context=context)