from __future__ import annotations

from typing import Dict, List

try:
    from proposed.config import BatteryConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import BatteryConfig

from .battery_types import CommLinkInput, UAVBatteryMode


def compute_hover_energy(
    config: BatteryConfig,
    is_hovering: bool,
) -> float:
    """
    UAV가 hovering할 때 소모하는 에너지 산식을 구현하는 함수로,
    e_hover(t) = (p_0 + p_i) * delta_t 와 같이 계산됨.
    """
    if not is_hovering:
        return 0.0
    return float(config.p_0 + config.p_i) * float(config.slot_duration)


def compute_comm_energy(
    config: BatteryConfig,
    links: List[CommLinkInput],
) -> float:
    """
    UAV가 user에게 video delivery 작업을 수행할 때 소모하는 에너지 산식을 구현하는 함수
    """
    total = 0.0

    for link in links:
        if not bool(link.scheduled):
            continue
        if int(link.delivered_chunks) <= 0:
            continue
        if float(link.payload_bits) <= 0.0: 
            continue

        tx_power = 0.0 if link.tx_power is None else max(0.0, float(link.tx_power))
        tx_time = max(0.0, float(link.tx_time))

        total += tx_power * tx_time * float(config.tx_energy_coeff)

    return float(total)


def compute_total_energy(
    hover_energy: float,
    comm_energy: float,
) -> float:
    """
    UAV가 소모하는 총 에너지 수식을 구현하는 함수로
    hovering 에너지와 comm 에너지의 합으로 정의됨
    """
    return float(hover_energy + comm_energy)


def compute_charge_energy(
    config: BatteryConfig,
    mu_active: bool,
    mode: UAVBatteryMode,
) -> float:
    """
    UAV가 충전할 때 증가하는 에너지 수식을 구현하는 함수로
    각 충전소들은 일정한 에너지 공급량을 가지고 있고, 충전량은 시간에 비례함
    """
    if not config.enable_charging:
        return 0.0
    if not mu_active:
        return 0.0
    if mode != UAVBatteryMode.CHARGE:
        return 0.0

    return float(config.charging_rate) * float(config.slot_duration)


def compute_energy_summary(
    config: BatteryConfig,
    mode: UAVBatteryMode,
    mu_active: bool,
    links: List[CommLinkInput],
    consume_hover_when_idle: bool = False,
) -> Dict[str, float]:
    """
    energy summary 구조 반환 함수로,
    hover, comm, charge, total energy로 구성됨.
    """
    if not mu_active:
        return {
            "hover_energy": 0.0,
            "comm_energy": 0.0,
            "total_energy": 0.0,
            "charge_energy": 0.0,
        }
    
    if mode == UAVBatteryMode.SERVE:
        hover_e = compute_hover_energy(config, is_hovering=True)
        comm_e = compute_comm_energy(config, links=links)
        total_e = compute_total_energy(hover_e, comm_e)
        charge_e = 0.0
    
    elif mode == UAVBatteryMode.CHARGE:
        hover_e = 0.0
        comm_e = 0.0
        total_e = 0.0
        charge_e = compute_charge_energy(config, mu_active=mu_active, mode=mode)

    elif mode == UAVBatteryMode.IDLE:
        hover_e = compute_hover_energy(config, is_hovering=consume_hover_when_idle)
        comm_e = 0.0
        total_e = compute_total_energy(hover_e, comm_e)
        charge_e = 0.0
    
    elif mode == UAVBatteryMode.OUTAGE:
        hover_e = 0.0
        comm_e = 0.0
        total_e = 0.0
        charge_e = 0.0

    else:
        raise ValueError(f"UAVBatteryMode는 {mode} 모드를 지원하지 않습니다.")

    return {
        "hover_energy": float(hover_e),
        "comm_energy": float(comm_e),
        "total_energy": float(total_e),
        "charge_energy": float(charge_e),
    }
