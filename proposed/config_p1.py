from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class P1Config:
    """
    P1 baseline: UAV 1대, UAV당 사용자 한명을 담당하는 구조
    
    This configuration is intentionally separate from ``config.EnvConfig``.
    The legacy PPO experiments therefore keep their original observation/action
    dimensions while the formulation-correct baseline can evolve independently.
    Numerical defaults are a lab-meeting smoke profile, not calibrated results.
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
    uav_capacity: int = 1
    rsu_total_bandwidth_hz: float = 20e6
    uav_total_bandwidth_hz: float = 5e6
    rsu_total_power_w: float = 40.0
    uav_max_power_w: float = 3.0
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

    rollout_scenarios: int = 4

    def __post_init__(self) -> None:
        positive_ints = {
            "num_regions": self.num_regions,
            "users_per_region": self.users_per_region,
            "num_frames": self.num_frames,
            "frame_slots": self.frame_slots,
            "rsu_capacity": self.rsu_capacity,
            "max_chunks_per_slot": self.max_chunks_per_slot,
            "rollout_scenarios": self.rollout_scenarios,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.uav_capacity != 1:
            raise ValueError("P1 requires uav_capacity == 1 (J^U = 1)")
        if self.slot_duration_s <= 0.0 or self.control_interval_s <= 0.0:
            raise ValueError("time parameters must be positive")
        if self.vehicle_speed_min_mps > self.vehicle_speed_max_mps:
            raise ValueError("vehicle speed range is invalid")
        if len(self.quality_utility) != len(self.chunk_size_bits):
            raise ValueError("quality_utility and chunk_size_bits must align")
        if not self.quality_utility:
            raise ValueError("at least one quality level is required")
        if tuple(sorted(self.quality_utility)) != self.quality_utility:
            raise ValueError("quality_utility must be nondecreasing")
        if tuple(sorted(self.chunk_size_bits)) != self.chunk_size_bits:
            raise ValueError("chunk_size_bits must be nondecreasing")
        if self.initial_battery_j > self.battery_capacity_j:
            raise ValueError("initial battery cannot exceed capacity")
        if not (0.0 <= self.reserve_battery_j < self.battery_capacity_j):
            raise ValueError("battery reserve is invalid")
        if not (0.0 < self.pa_efficiency <= 1.0):
            raise ValueError("pa_efficiency must be in (0, 1]")
        if not (0.0 < self.charging_efficiency <= 1.0):
            raise ValueError("charging_efficiency must be in (0, 1]")

    @property
    def num_users(self) -> int:
        return self.num_regions * self.users_per_region

    @property
    def num_uavs(self) -> int:
        return self.num_regions

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
        return (
            self.charging_efficiency
            * self.charging_power_w
            * self.slot_duration_s
        )

    @property
    def reachable_distance_m(self) -> float:
        return self.uav_max_speed_mps * self.control_interval_s

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
