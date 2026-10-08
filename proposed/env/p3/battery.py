from __future__ import annotations

from dataclasses import dataclass

from config_p3 import P3Config
from env.p3.types import ReturnDiagnostic


@dataclass(frozen=True)
class BatteryStep:
    battery_after_j: float
    consumed_j: float
    charged_j: float


def relocation_energy_j(previous_x: float, target_x: float, cfg: P3Config) -> float:
    """Equation (4.1): a fixed conservative cost for every non-zero move."""

    return cfg.relocation_energy_j if abs(target_x - previous_x) > 1e-9 else 0.0


def activation_energy_required_j(cfg: P3Config) -> float:
    """Equation (4.11), evaluated after frame relocation."""

    return cfg.reserve_battery_j + cfg.frame_slots * cfg.hover_energy_per_slot_j


def battery_power_cap_w(
    battery_j: float,
    remaining_slots_including_current: int,
    cfg: P3Config,
) -> float:
    """Equations (4.9)-(4.10) with ``E_th`` counted exactly once."""

    if remaining_slots_including_current <= 0:
        raise ValueError("remaining_slots_including_current must be positive")
    energy_for_comm = max(
        0.0,
        float(battery_j)
        - cfg.reserve_battery_j
        - remaining_slots_including_current * cfg.hover_energy_per_slot_j,
    )
    battery_limited = cfg.pa_efficiency * energy_for_comm / cfg.slot_duration_s
    return min(cfg.uav_max_total_power_w, battery_limited)


def diagnose_return_to_charge(
    battery_j: float,
    previous_x: float,
    depot_x: float,
    will_charge: bool,
    cfg: P3Config,
) -> ReturnDiagnostic:
    """Measure the professor-requested pre-charge depletion event.

    Under the formulation's hard reserve this rate should be zero.  It is still
    logged explicitly rather than inferred from the final SoC.  A return event
    means that an unhired UAV moves from a non-depot point to the depot.
    ``depletion_before_arrival`` checks relocation energy causality.  The
    threshold check is made on the pre-return battery, because ``E_th`` already
    includes worst-case return energy and its safety margin.
    """

    is_return = bool(will_charge and abs(previous_x - depot_x) > 1e-9)
    required = relocation_energy_j(previous_x, depot_x, cfg) if is_return else 0.0
    arrival = float(battery_j - required)
    return ReturnDiagnostic(
        is_return_to_charge=is_return,
        relocation_energy_required_j=float(required),
        arrival_energy_j=arrival,
        depletion_before_arrival=bool(is_return and arrival < -1e-9),
        reserve_breach_before_charge=bool(
            is_return and battery_j < cfg.reserve_battery_j - 1e-9
        ),
    )


def apply_relocation(
    battery_j: float,
    previous_x: float,
    target_x: float,
    cfg: P3Config,
) -> BatteryStep:
    if abs(target_x - previous_x) > cfg.reachable_distance_m + 1e-9:
        raise RuntimeError("relocation reachability violation")
    energy = relocation_energy_j(previous_x, target_x, cfg)
    if battery_j + 1e-9 < energy:
        raise RuntimeError("relocation energy-causality violation")
    return BatteryStep(float(battery_j - energy), float(energy), 0.0)


def apply_active_slot(
    battery_j: float,
    total_tx_power_w: float,
    remaining_slots_including_current: int,
    cfg: P3Config,
) -> BatteryStep:
    power_cap = battery_power_cap_w(
        battery_j,
        remaining_slots_including_current,
        cfg,
    )
    if total_tx_power_w < -1e-12 or total_tx_power_w > power_cap + 1e-9:
        raise RuntimeError(
            "UAV total power violates the effective battery cap: "
            f"power={total_tx_power_w}, cap={power_cap}"
        )
    consumed = (
        cfg.hover_energy_per_slot_j
        + cfg.slot_duration_s * max(total_tx_power_w, 0.0) / cfg.pa_efficiency
    )
    battery_after = float(battery_j - consumed)
    required_after = (
        cfg.reserve_battery_j
        + (remaining_slots_including_current - 1) * cfg.hover_energy_per_slot_j
    )
    if battery_after + 1e-7 < required_after:
        raise RuntimeError("remaining-hover reserve violation")
    return BatteryStep(battery_after, float(consumed), 0.0)


def apply_unhired_slot(battery_j: float, cfg: P3Config) -> BatteryStep:
    accepted = max(
        0.0,
        min(cfg.charge_energy_per_slot_j, cfg.battery_capacity_j - battery_j),
    )
    return BatteryStep(float(battery_j + accepted), 0.0, float(accepted))
