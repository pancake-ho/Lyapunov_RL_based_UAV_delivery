from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class P3Config:
    """Formulation-consistent P3 configuration.

    P3 has one persistent UAV per RSU region and ``2 <= J^U < J^R``.
    The defaults are reproducible smoke-test values, not calibrated paper
    parameters.  Keep this configuration separate from the legacy PPO and P1
    configurations so that existing experiments remain reproducible.
    """

    seed: int = 2026
    num_regions: int = 2
    users_per_region: int = 4
    num_frames: int = 30
    frame_slots: int = 10
    slot_duration_s: float = 1.0
    control_interval_s: float = 3.0

    region_length_m: float = 400.0
    rsu_height_m: float = 8.0
    uav_height_m: float = 80.0
    user_height_m: float = 1.5
    vehicle_speed_min_mps: float = 5.0
    vehicle_speed_max_mps: float = 20.0
    uav_max_speed_mps: float = 20.0
    candidate_offsets_m: tuple[float, ...] = (
        -100.0,
        -50.0,
        0.0,
        50.0,
        100.0,
    )

    rsu_capacity: int = 3
    uav_capacity: int = 2
    rsu_total_bandwidth_hz: float = 20e6
    uav_total_bandwidth_hz: float = 5e6
    rsu_total_power_w: float = 40.0
    uav_max_total_power_w: float = 3.0
    noise_psd_w_hz: float = 2e-20
    shannon_gap: float = 2.0
    rsu_beta0: float = 1e-7
    uav_beta0: float = 2e-9
    rsu_pathloss_exp: float = 3.0
    uav_pathloss_exp: float = 2.2

    quality_utility: tuple[float, ...] = (0.55, 0.72, 0.86, 1.0)
    chunk_size_bits: tuple[float, ...] = (0.5e6, 1e6, 2e6, 4e6)
    max_chunks_per_slot: int = 1
    playback_chunks_per_slot: float = 1.0
    initial_playback_queue: float = 3.0
    large_queue_level: float = 100.0

    alpha_z: float = 1.0
    lyapunov_v: float = 10.0
    lambda_h: float = 1.0
    hiring_cost_per_frame: float = 5.0

    battery_capacity_j: float = 548.0 * 3600.0
    initial_battery_j: float = 548.0 * 3600.0
    reserve_battery_j: float = 0.20 * 548.0 * 3600.0
    hovering_power_w: float = 597.0
    charging_power_w: float = 1000.0
    charging_efficiency: float = 1.0
    pa_efficiency: float = 1.0
    relocation_energy_j: float = 10_000.0

    rollout_scenarios: int = 2
    candidate_user_limit: int = 6
    max_rollout_actions: int = 4096
    distance_bin_edges_m: tuple[float, ...] = (
        0.0,
        50.0,
        100.0,
        150.0,
        200.0,
        math.inf,
    )

    ppo_hidden_dim: int = 128
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_ratio: float = 0.20
    ppo_value_coef: float = 0.50
    ppo_entropy_coef: float = 0.01
    ppo_max_grad_norm: float = 0.50
    ppo_target_kl: float = 0.015
    ppo_update_epochs: int = 4
    ppo_batch_size: int = 64
    ppo_reward_scale: float = 1e-3

    def __post_init__(self) -> None:
        positive_ints = {
            "num_regions": self.num_regions,
            "users_per_region": self.users_per_region,
            "num_frames": self.num_frames,
            "frame_slots": self.frame_slots,
            "rsu_capacity": self.rsu_capacity,
            "rollout_scenarios": self.rollout_scenarios,
            "candidate_user_limit": self.candidate_user_limit,
            "max_rollout_actions": self.max_rollout_actions,
            "ppo_hidden_dim": self.ppo_hidden_dim,
            "ppo_update_epochs": self.ppo_update_epochs,
            "ppo_batch_size": self.ppo_batch_size,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (2 <= self.uav_capacity < self.rsu_capacity):
            raise ValueError("P3 requires 2 <= uav_capacity < rsu_capacity")
        if self.slot_duration_s <= 0.0 or self.control_interval_s <= 0.0:
            raise ValueError("time parameters must be positive")
        if self.uav_max_speed_mps <= 0.0:
            raise ValueError("uav_max_speed_mps must be positive")
        if self.vehicle_speed_min_mps > self.vehicle_speed_max_mps:
            raise ValueError("vehicle speed range is invalid")
        if len(self.quality_utility) != len(self.chunk_size_bits):
            raise ValueError("quality utility and chunk sizes must align")
        if not self.quality_utility:
            raise ValueError("at least one quality level is required")
        if tuple(sorted(self.quality_utility)) != self.quality_utility:
            raise ValueError("quality utilities must be nondecreasing")
        if tuple(sorted(self.chunk_size_bits)) != self.chunk_size_bits:
            raise ValueError("chunk sizes must be nondecreasing")
        if self.max_chunks_per_slot <= 0:
            raise ValueError("max_chunks_per_slot must be positive")
        if not (0.0 <= self.reserve_battery_j < self.battery_capacity_j):
            raise ValueError("battery reserve is invalid")
        if not (self.reserve_battery_j <= self.initial_battery_j <= self.battery_capacity_j):
            raise ValueError("initial battery is outside the feasible range")
        if not (0.0 < self.pa_efficiency <= 1.0):
            raise ValueError("pa_efficiency must be in (0, 1]")
        if not (0.0 < self.charging_efficiency <= 1.0):
            raise ValueError("charging_efficiency must be in (0, 1]")
        edges = self.distance_bin_edges_m
        if len(edges) < 2 or edges[0] != 0.0:
            raise ValueError("distance bins must start at zero")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("distance bin edges must be strictly increasing")
        if not math.isinf(edges[-1]):
            raise ValueError("the final distance bin edge must be infinity")
        if not (0.0 < self.ppo_reward_scale):
            raise ValueError("ppo_reward_scale must be positive")
        if self.ppo_target_kl <= 0.0:
            raise ValueError("ppo_target_kl must be positive")
        if not self.candidate_offsets_m or not any(
            abs(offset) <= 1e-9 for offset in self.candidate_offsets_m
        ):
            raise ValueError("candidate offsets must include the depot offset 0")
        points = sorted(set(self.candidate_points(0)))
        if any(
            right - left > self.reachable_distance_m + 1e-9
            for left, right in zip(points, points[1:])
        ):
            raise ValueError(
                "candidate point graph must be connected under the relocation limit"
            )

    @property
    def num_users(self) -> int:
        return self.num_regions * self.users_per_region

    @property
    def num_uavs(self) -> int:
        return self.num_regions

    @property
    def num_quality_levels(self) -> int:
        return len(self.quality_utility)

    @property
    def quality_max(self) -> float:
        return float(max(self.quality_utility))

    @property
    def road_length_m(self) -> float:
        return self.num_regions * self.region_length_m

    @property
    def hover_energy_per_slot_j(self) -> float:
        return self.hovering_power_w * self.slot_duration_s

    @property
    def charge_energy_per_slot_j(self) -> float:
        return self.charging_efficiency * self.charging_power_w * self.slot_duration_s

    @property
    def reachable_distance_m(self) -> float:
        return self.uav_max_speed_mps * self.control_interval_s

    @property
    def num_distance_bins(self) -> int:
        return len(self.distance_bin_edges_m) - 1

    @property
    def uav_user_bandwidth_hz(self) -> float:
        return self.uav_total_bandwidth_hz / self.uav_capacity

    @property
    def ppo_state_dim(self) -> int:
        return 7 + 6 * self.num_users

    @property
    def ppo_action_dim(self) -> int:
        return 10

    def depot_x(self, region: int) -> float:
        self._validate_region(region)
        return (float(region) + 0.5) * self.region_length_m

    def rsu_x(self, region: int) -> float:
        return self.depot_x(region)

    def candidate_points(self, region: int) -> tuple[float, ...]:
        center = self.depot_x(region)
        left = float(region) * self.region_length_m
        right = left + self.region_length_m
        return tuple(
            min(max(center + offset, left), right)
            for offset in self.candidate_offsets_m
        )

    def _validate_region(self, region: int) -> None:
        if not 0 <= int(region) < self.num_regions:
            raise IndexError(f"invalid region index: {region}")