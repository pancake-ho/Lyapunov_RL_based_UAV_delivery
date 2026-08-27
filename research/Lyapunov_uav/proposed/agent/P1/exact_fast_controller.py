from __future__ import annotations

import math

from config_p1 import P1Config
from env.p1.battery import battery_power_cap_w
from env.p1.radio import required_uav_power_w, rsu_link_capacity_bps
from env.p1.types import FastOption


class ExactFastController:
    """Exact finite-option minimizer for the slot-level P1 subproblem."""

    def __init__(self, cfg: P1Config) -> None:
        self.cfg = cfg

    def controllable_cost(
        self,
        z: float,
        chunks: int,
        quality_index: int,
    ) -> float:
        if chunks <= 0 or quality_index < 0:
            return 0.0
        degradation = (
            self.cfg.quality_max
            - self.cfg.quality_utility[quality_index]
        )
        return float(chunks) * (
            self.cfg.lyapunov_v * degradation
            - self.cfg.alpha_z * float(z)
        )

    def solve_rsu(
        self,
        z: float,
        horizontal_distance_m: float,
        fading: float,
    ) -> FastOption:
        capacity = rsu_link_capacity_bps(
            horizontal_distance_m,
            fading,
            self.cfg,
        )
        best = FastOption()
        fixed_power = self.cfg.rsu_total_power_w / self.cfg.rsu_capacity
        for quality_index, (utility, chunk_bits) in enumerate(
            zip(self.cfg.quality_utility, self.cfg.chunk_size_bits)
        ):
            feasible = min(
                self.cfg.max_chunks_per_slot,
                int(
                    math.floor(
                        capacity * self.cfg.slot_duration_s / chunk_bits
                    )
                ),
            )
            for chunks in range(1, feasible + 1):
                cost = self.controllable_cost(z, chunks, quality_index)
                if cost < best.controllable_dpp_cost - 1e-12:
                    best = FastOption(
                        chunks=chunks,
                        quality_index=quality_index,
                        utility=chunks * utility,
                        degradation=chunks * (self.cfg.quality_max - utility),
                        payload_bits=chunks * chunk_bits,
                        power_w=fixed_power,
                        controllable_dpp_cost=cost,
                    )
        return best

    def solve_uav(
        self,
        z: float,
        horizontal_distance_m: float,
        fading: float,
        battery_j: float,
        remaining_slots_including_current: int,
    ) -> FastOption:
        power_cap = battery_power_cap_w(
            battery_j,
            remaining_slots_including_current,
            self.cfg,
        )
        best = FastOption()
        for quality_index, utility in enumerate(self.cfg.quality_utility):
            for chunks in range(1, self.cfg.max_chunks_per_slot + 1):
                required = required_uav_power_w(
                    chunks,
                    quality_index,
                    horizontal_distance_m,
                    fading,
                    self.cfg,
                )
                if (
                    not math.isfinite(required)
                    or required > power_cap + 1e-12
                ):
                    continue
                cost = self.controllable_cost(z, chunks, quality_index)
                candidate = FastOption(
                    chunks=chunks,
                    quality_index=quality_index,
                    utility=chunks * utility,
                    degradation=(
                        chunks * (self.cfg.quality_max - utility)
                    ),
                    payload_bits=(
                        chunks * self.cfg.chunk_size_bits[quality_index]
                    ),
                    power_w=required,
                    controllable_dpp_cost=cost,
                )
                if cost < best.controllable_dpp_cost - 1e-12:
                    best = candidate
                elif (
                    abs(cost - best.controllable_dpp_cost) <= 1e-12
                    and required < best.power_w - 1e-12
                ):
                    best = candidate
        return best
