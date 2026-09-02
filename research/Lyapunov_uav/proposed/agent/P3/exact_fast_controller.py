from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

from config_p3 import P3Config
from env.p3.battery import battery_power_cap_w
from env.p3.radio import required_uav_power_w, rsu_link_capacity_bps
from env.p3.types import FastOption


class ExactFastController:
    """Algorithm 1: exact current-slot controller for small J^U.

    RSU users are separable.  UAV users are coupled by the total RF power cap,
    so their option product is enumerated exactly as required by Section 7.3.
    """

    def __init__(self, cfg: P3Config) -> None:
        self.cfg = cfg

    def controllable_cost(self, z: float, chunks: int, quality_index: int) -> float:
        if chunks <= 0 or quality_index < 0:
            return 0.0
        degradation = self.cfg.quality_max - self.cfg.quality_utility[quality_index]
        return float(chunks) * (
            self.cfg.lyapunov_v * degradation - self.cfg.alpha_z * float(z)
        )

    def solve_rsu(
        self,
        z: float,
        horizontal_distance_m: float,
        fading: float,
    ) -> FastOption:
        capacity = rsu_link_capacity_bps(horizontal_distance_m, fading, self.cfg)
        best = FastOption()
        fixed_power = self.cfg.rsu_total_power_w / self.cfg.rsu_capacity
        for quality_index, (utility, chunk_bits) in enumerate(
            zip(self.cfg.quality_utility, self.cfg.chunk_size_bits)
        ):
            feasible = min(
                self.cfg.max_chunks_per_slot,
                int(math.floor(capacity * self.cfg.slot_duration_s / chunk_bits)),
            )
            for chunks in range(1, feasible + 1):
                option = FastOption(
                    chunks=chunks,
                    quality_index=quality_index,
                    utility=chunks * utility,
                    degradation=chunks * (self.cfg.quality_max - utility),
                    payload_bits=chunks * chunk_bits,
                    power_w=fixed_power,
                    controllable_dpp_cost=self.controllable_cost(
                        z, chunks, quality_index
                    ),
                )
                if self._option_better(option, best):
                    best = option
        return best

    def feasible_uav_options(
        self,
        z: float,
        horizontal_distance_m: float,
        fading: float,
        individual_power_cap_w: float,
    ) -> list[FastOption]:
        options = [FastOption()]
        for quality_index, utility in enumerate(self.cfg.quality_utility):
            for chunks in range(1, self.cfg.max_chunks_per_slot + 1):
                required = required_uav_power_w(
                    chunks,
                    quality_index,
                    horizontal_distance_m,
                    fading,
                    self.cfg,
                )
                if not math.isfinite(required) or required > individual_power_cap_w + 1e-12:
                    continue
                options.append(
                    FastOption(
                        chunks=chunks,
                        quality_index=quality_index,
                        utility=chunks * utility,
                        degradation=chunks * (self.cfg.quality_max - utility),
                        payload_bits=chunks * self.cfg.chunk_size_bits[quality_index],
                        power_w=required,
                        controllable_dpp_cost=self.controllable_cost(
                            z, chunks, quality_index
                        ),
                    )
                )
        return options

    def solve_uav(
        self,
        users: Sequence[int],
        z_by_user: Mapping[int, float],
        horizontal_distance_by_user_m: Mapping[int, float],
        fading_by_user: Mapping[int, float],
        battery_j: float,
        remaining_slots_including_current: int,
        required_end_energy_j: float | None = None,
    ) -> dict[int, FastOption]:
        ordered_users = tuple(sorted(int(user) for user in users))
        if len(ordered_users) > self.cfg.uav_capacity:
            raise ValueError("scheduled UAV users exceed J^U")
        if not ordered_users:
            return {}
        total_cap = battery_power_cap_w(
            battery_j,
            remaining_slots_including_current,
            self.cfg,
            required_end_energy_j,
        )
        option_sets = [
            self.feasible_uav_options(
                z=float(z_by_user[user]),
                horizontal_distance_m=float(horizontal_distance_by_user_m[user]),
                fading=float(fading_by_user[user]),
                individual_power_cap_w=total_cap,
            )
            for user in ordered_users
        ]

        best_combo: tuple[FastOption, ...] | None = None
        best_cost = math.inf
        best_power = math.inf
        best_key: tuple | None = None
        for combo in itertools.product(*option_sets):
            total_power = sum(option.power_w for option in combo)
            if total_power > total_cap + 1e-12:
                continue
            total_cost = sum(option.controllable_dpp_cost for option in combo)
            key = tuple((option.chunks, option.quality_index) for option in combo)
            better = total_cost < best_cost - 1e-12
            tied_cost = abs(total_cost - best_cost) <= 1e-12
            if tied_cost and total_power < best_power - 1e-12:
                better = True
            if (
                tied_cost
                and abs(total_power - best_power) <= 1e-12
                and (best_key is None or key < best_key)
            ):
                better = True
            if better:
                best_combo = combo
                best_cost = float(total_cost)
                best_power = float(total_power)
                best_key = key

        if best_combo is None:
            raise RuntimeError("zero-option should always make the UAV fast problem feasible")
        return dict(zip(ordered_users, best_combo))

    @staticmethod
    def _option_better(candidate: FastOption, incumbent: FastOption) -> bool:
        if candidate.controllable_dpp_cost < incumbent.controllable_dpp_cost - 1e-12:
            return True
        if abs(candidate.controllable_dpp_cost - incumbent.controllable_dpp_cost) > 1e-12:
            return False
        if candidate.power_w < incumbent.power_w - 1e-12:
            return True
        if abs(candidate.power_w - incumbent.power_w) > 1e-12:
            return False
        return (candidate.chunks, candidate.quality_index) < (
            incumbent.chunks,
            incumbent.quality_index,
        )
