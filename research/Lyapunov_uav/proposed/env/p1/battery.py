from __future__ import annotations

from dataclasses import dataclass

from config_p1 import P1Config


@dataclass(frozen=True)
class BatteryStep:
    battery_after_j: float
    consumed_j: float
    charged_j: float


def relocation_energy_j(
    previous_x: float,
    target_x: float,
    cfg: P1Config,
) -> float:
    """Professor's first formulation uses a fixed cost for a non-zero move."""

    return cfg.relocation_energy_j if abs(target_x - previous_x) > 1e-9 else 0.0


def activation_energy_required_j(cfg: P1Config) -> float:
    """Frame-start reserve: final reserve plus all remaining hover energy."""

    return cfg.reserve_battery_j + cfg.frame_slots * cfg.hover_energy_per_slot_j


def battery_power_cap_w(
    battery_j: float,
    remaining_slots_including_current: int,
    cfg: P1Config,
) -> float:
    """Maximum RF power that preserves the per-slot remaining-hover reserve."""

    if remaining_slots_including_current <= 0:
        raise ValueError("remaining_slots_including_current must be positive")
    energy_for_comm = max(
        0.0,
        float(battery_j)
        - cfg.reserve_battery_j
        - remaining_slots_including_current * cfg.hover_energy_per_slot_j,
    )
    battery_limited = (
        cfg.pa_efficiency * energy_for_comm / cfg.slot_duration_s
    )
    return min(cfg.uav_max_power_w, battery_limited)


def apply_relocation(
    battery_j: float,
    previous_x: float,
    target_x: float,
    cfg: P1Config,
) -> BatteryStep:
    if abs(target_x - previous_x) > cfg.reachable_distance_m + 1e-9:
        raise RuntimeError("relocation reachability violation")
    energy = relocation_energy_j(previous_x, target_x, cfg)
    if battery_j + 1e-9 < energy:
        raise RuntimeError("relocation energy-causality violation")
    return BatteryStep(
        battery_after_j=float(battery_j - energy),
        consumed_j=float(energy),
        charged_j=0.0,
    )


def apply_active_slot(
    battery_j: float,
    tx_power_w: float,
    remaining_slots_including_current: int,
    cfg: P1Config,
) -> BatteryStep:
    power_cap = battery_power_cap_w(
        battery_j,
        remaining_slots_including_current,
        cfg,
    )
    if tx_power_w < -1e-12 or tx_power_w > power_cap + 1e-9:
        raise RuntimeError(
            "UAV power violates the battery-derived cap: "
            f"power={tx_power_w}, cap={power_cap}"
        )
    consumed = (
        cfg.hover_energy_per_slot_j
        + cfg.slot_duration_s * max(tx_power_w, 0.0) / cfg.pa_efficiency
    )
    battery_after = float(battery_j - consumed)
    required_after = (
        cfg.reserve_battery_j
        + (remaining_slots_including_current - 1)
        * cfg.hover_energy_per_slot_j
    )
    if battery_after + 1e-7 < required_after:
        raise RuntimeError(
            "remaining-hover reserve violation: "
            f"battery_after={battery_after}, required={required_after}"
        )
    return BatteryStep(
        battery_after_j=battery_after,
        consumed_j=float(consumed),
        charged_j=0.0,
    )


def apply_unhired_slot(
    battery_j: float,
    cfg: P1Config,
) -> BatteryStep:
    """An unhired UAV stays at its depot and passively charges."""

    accepted = max(
        0.0,
        min(cfg.charge_energy_per_slot_j, cfg.battery_capacity_j - battery_j),
    )
    return BatteryStep(
        battery_after_j=float(battery_j + accepted),
        consumed_j=0.0,
        charged_j=float(accepted),
    )
