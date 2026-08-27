from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config_p1 import P1Config


@dataclass(frozen=True)
class RegionAction:
    """Frame-level action x_r=(mu, a, y, phi) for one region."""

    region: int
    hired: int
    point_index: int
    rsu_users: tuple[int, ...]
    uav_user: Optional[int]

    def target_x(self, cfg: P1Config) -> float:
        if self.hired == 0:
            return cfg.depot_x(self.region)
        return cfg.candidate_points(self.region)[self.point_index]


@dataclass(frozen=True)
class FastOption:
    """One exact finite slot option, including the zero-delivery option."""

    chunks: int = 0
    quality_index: int = -1
    utility: float = 0.0
    degradation: float = 0.0
    payload_bits: float = 0.0
    power_w: float = 0.0
    controllable_dpp_cost: float = 0.0


@dataclass
class P1State:
    """Physical state only: playback queue, mobility, Joule battery, UAV position."""

    queue: np.ndarray
    user_x: np.ndarray
    user_speed: np.ndarray
    battery_j: np.ndarray
    uav_x: np.ndarray

    def copy(self) -> "P1State":
        return P1State(
            queue=self.queue.copy(),
            user_x=self.user_x.copy(),
            user_speed=self.user_speed.copy(),
            battery_j=self.battery_j.copy(),
            uav_x=self.uav_x.copy(),
        )


@dataclass(frozen=True)
class FrameTrace:
    """Common-random-number channel trace used by every candidate action."""

    rsu_fading: np.ndarray
    uav_fading: np.ndarray


@dataclass
class RegionFrameResult:
    frame_dpp_cost: float
    original_cost: float
    queue_after: np.ndarray
    battery_after_j: float
    uav_x_after: float
    delivered_chunks: float
    quality_utility: float
    degradation: float
    stall_user_slots: int
    served_user_slots: int
    large_queue_violation_user_slots: int
    queue_sum: float
    energy_consumed_j: float
    energy_charged_j: float
    relocation_events: int
    battery_reserve_violations: int
    power_violations: int
    provider_violations: int
