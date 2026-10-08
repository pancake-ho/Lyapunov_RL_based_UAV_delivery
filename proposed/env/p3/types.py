from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config_p3 import P3Config


@dataclass(frozen=True)
class RegionAction:
    """Frame action x_r=(mu, a, y, phi) for one RSU region."""

    region: int
    hired: int
    point_index: int
    rsu_users: tuple[int, ...]
    uav_users: tuple[int, ...]

    def target_x(self, cfg: P3Config) -> float:
        if self.hired == 0:
            return cfg.depot_x(self.region)
        return cfg.candidate_points(self.region)[self.point_index]


@dataclass(frozen=True)
class FastOption:
    chunks: int = 0
    quality_index: int = -1
    utility: float = 0.0
    degradation: float = 0.0
    payload_bits: float = 0.0
    power_w: float = 0.0
    controllable_dpp_cost: float = 0.0


@dataclass
class P3State:
    """Physical state; Z is derived exactly as Q_tilde-Q."""

    queue: np.ndarray
    user_x: np.ndarray
    user_speed: np.ndarray
    battery_j: np.ndarray
    uav_x: np.ndarray
    last_quality_index: np.ndarray
    last_stalled: np.ndarray

    def copy(self) -> "P3State":
        return P3State(
            queue=self.queue.copy(),
            user_x=self.user_x.copy(),
            user_speed=self.user_speed.copy(),
            battery_j=self.battery_j.copy(),
            uav_x=self.uav_x.copy(),
            last_quality_index=self.last_quality_index.copy(),
            last_stalled=self.last_stalled.copy(),
        )


@dataclass(frozen=True)
class FrameTrace:
    rsu_fading: np.ndarray
    uav_fading: np.ndarray


@dataclass
class RegionFrameResult:
    frame_dpp_cost: float
    original_cost: float
    queue_after: np.ndarray
    last_quality_after: np.ndarray
    last_stalled_after: np.ndarray
    battery_after_j: float
    uav_x_after: float
    delivered_chunks: float
    payload_bits: float
    quality_utility: float
    quality_level_sum: float
    quality_histogram: np.ndarray
    degradation: float
    quality_switches: int
    quality_transitions: int
    stall_user_slots: int
    served_user_slots: int
    large_queue_violation_user_slots: int
    queue_sum: float
    queue_samples: np.ndarray
    z_samples: np.ndarray
    user_delivered_chunks: np.ndarray
    user_quality_utility: np.ndarray
    user_stall_slots: np.ndarray
    user_served_slots: np.ndarray
    user_stall_events: np.ndarray
    user_queue_sum: np.ndarray
    energy_consumed_j: float
    energy_charged_j: float
    relocation_events: int
    charging_slots: int
    charging_need_events: int
    stranded_before_charge_events: int
    return_to_charge_events: int
    precharge_depletion_events: int
    precharge_reserve_breach_events: int
    battery_reserve_violations: int
    power_violations: int
    provider_violations: int
    uav_total_power_peak_w: float
    uav_distance_sum_m: float
    uav_scheduled_user_slots: int
    rsu_scheduled_user_slots: int
    distance_opportunities: np.ndarray
    distance_served_slots: np.ndarray
    distance_stall_slots: np.ndarray
    distance_delivered_chunks: np.ndarray
    distance_quality_utility: np.ndarray
    distance_power_sum_w: np.ndarray


@dataclass(frozen=True)
class ReturnDiagnostic:
    is_return_to_charge: bool
    relocation_energy_required_j: float
    arrival_energy_j: float
    depletion_before_arrival: bool
    reserve_breach_before_charge: bool
