"""One frame proposal; UAV candidates become assignments only if hired."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from config_hppo import HPPOConfig
from env.p3.types import RegionAction


def integer_action(raw, nvec, label="action") -> np.ndarray:
    arr = np.asarray(raw)
    if arr.shape != (len(nvec),) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{label}: expected numeric shape {(len(nvec),)}")
    if not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
        raise ValueError(f"{label}: entries must be finite integers")
    if np.any(arr < 0) or np.any(arr >= np.asarray(nvec)):
        raise ValueError(f"{label}: out-of-domain token")
    return arr.astype(np.int64, copy=True)


@dataclass(frozen=True)
class SchedulingProposal:
    region: int
    rsu_users: tuple[int, ...]
    uav_candidates: tuple[int, ...]

    @classmethod
    def from_tokens(cls, region, raw, members, cfg: HPPOConfig, uav_possible=True):
        tokens = integer_action(raw, cfg.frame_action_nvec, "scheduling proposal")
        eligible = set(members)
        if any(tokens[u] != 0 for u in range(cfg.num_users) if u not in eligible):
            raise ValueError("non-member scheduling token")
        rsu = tuple(int(u) for u in np.flatnonzero(tokens == 1))
        uav = tuple(int(u) for u in np.flatnonzero(tokens == 2))
        if len(rsu) > cfg.rsu_capacity or len(uav) > cfg.uav_capacity:
            raise ValueError("scheduling proposal exceeds capacity; no projection is applied")
        if uav and not uav_possible:
            raise ValueError("UAV proposal has no feasible hiring point")
        return cls(int(region), rsu, uav)

    def execute(self, hired: int, point: int) -> RegionAction:
        if hired not in (0, 1) or (hired == 0 and point != -1):
            raise ValueError("invalid completion")
        return RegionAction(self.region, hired, point, self.rsu_users,
                            self.uav_candidates if hired else ())

    def validate_completion(self, action: RegionAction) -> None:
        if action != self.execute(action.hired, action.point_index):
            raise ValueError("completion changed the proposed scheduling")


def sample_scheduling(masks, cfg: HPPOConfig, rng: np.random.Generator) -> np.ndarray:
    """Random baseline follows the same conditional support as the PPO decoder."""
    result = np.zeros(cfg.num_users, dtype=np.int64)
    counts = [0, 0, 0]
    for u, base in enumerate(masks):
        allowed = np.asarray(base, dtype=bool).copy()
        allowed[1] &= counts[1] < cfg.rsu_capacity
        allowed[2] &= counts[2] < cfg.uav_capacity
        choices = np.flatnonzero(allowed)
        if not len(choices):
            raise ValueError("empty scheduling support")
        token = int(rng.choice(choices))
        result[u] = token
        counts[token] += 1
    return result
