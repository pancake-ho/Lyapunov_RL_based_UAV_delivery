from __future__ import annotations

import math
from typing import Dict, List, Optional

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
    UAV hovering energy.

    현재 시나리오 기준:
        e_hover(t) = (P_0 + P_i) * (slot_duration)
    
    여기서 P_0는 blade profile energy,
    P_i는 induced power에 해당함.
    """
    if not is_hovering:
        return 0.0
    
    return float(config.p_0 + config.p_i) * float(config.slot_duration)


def compute_comm_energy(
    config: BatteryConfig,
    links: List[CommLinkInput],
) -> float:
    """
    UAV communication energy.

    현재 시나리오 기준에서는 p_un(t)를 직접 포함하기 때문에 기본 계산은 다음과 같음.

        e_comm(t) = sum_n (p_un(t) * slot_duration)
    """
    total = 0.0

    for link in links:
        if not bool(link.scheduled):
            continue
        if int(link.delivered_chunks) <= 0:
            continue
        if float(link.payload_bits) <= 0.0: 
            continue

        tx_time = max(0.0, float(link.tx_time))
        if tx_time <= 0.0:
            tx_time = float(config.slot_duration)
        
        tx_power = max(0.0, float(link.tx_power))
        total += tx_power * tx_time * float(config.tx_energy_coeff)
    
    return float(total)
    

def compute_charge_energy(
    config: BatteryConfig,
    mu_active: bool,
    mode: UAVBatteryMode,
) -> float:
    """
    UAV charging energy.

    현재 시나리오 기준:
        e_c(t) = I_u(t) * C_ch * (slot_duration)
    
    또한 고용되지 않은 UAV는 charging action을 수행하지 않는 것으로 처리함.
    """
    if not bool(config.enable_charging):
        return 0.0
    if not bool(config.allow_charge):
        return 0.0
    if not bool(mu_active):
        return 0.0
    if UAVBatteryMode(mode) != UAVBatteryMode.CHARGE:
        return 0.0

    return float(config.charging_rate) * float(config.slot_duration) * float(config.eta_c)


def compute_total_energy(
    hover_energy: float,
    comm_energy: float,
) -> float:
    """
    UAV가 한 slot에서 소모하는 총 energy.

    현재 시나리오 기준:
        e_u(t) = e_hover(t) + e_comm(t)
    """
    return float(hover_energy) + float(comm_energy)


def compute_energy_summary(
    config: BatteryConfig,
    mode: UAVBatteryMode,
    mu_active: bool,
    links: List[CommLinkInput],
    bandwidth: Optional[float] = None,
    consume_hover_when_idle: bool = True,
) -> Dict[str, float]:
    """
    한 slot에서 UAV의 hovering/communication/charging energy를 계산.

    현재 시나리오 기준 다음과 같이 mode별 처리를 수행함:
        SERVE:
            hovering energy + communication energy 소모
        
        IDLE:
            consume_hover_when_idle=True 이면 hovering energy 소모
        
        CHARGE:
            service와 동시에 수행하지 않으며, charging energy만 ㅈ으가
        
        OUTAGE:
            energy 변화 없음.
    """
    mode = UAVBatteryMode(mode)

    if not bool(mu_active):
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
