from __future__ import annotations

from typing import List

from proposed.config import BatteryConfig
from .battery_types import BatteryAction, UAVBatteryMode, CommLinkInput


def validate_action_mode(action: BatteryAction) -> None:
    """
    UAV의 action mode를 검증하는 함수 (충전 및 서비스 등)
    """
    if action.mode == UAVBatteryMode.CHARGE and action.links:
        raise ValueError("UAV는 CHARGE mode에서 links가 비어 있어야 합니다.")
    
    if action.mode in (UAVBatteryMode.IDLE, UAVBatteryMode.OUTAGE) and action.links:
        raise ValueError("UAV는 IDLE/OUTAGE mode에서 links가 비어 있어야 합니다.")
    
    if (not action.mu_active) and action.mode == UAVBatteryMode.SERVE:
        raise ValueError("고용되지 않은 UAV는 SERVE mode가 될 수 없습니다.")


def validate_links(links: List[CommLinkInput],) -> List[CommLinkInput]:
    """
    step 전에, 이상한 값의 입력을 걸러내는 함수
    """
    validated: List[CommLinkInput] = []

    for link in links:
        scheduled = bool(link.scheduled)
        delivered_chunks = max(0, int(link.delivered_chunks))
        payload_bits = max(0.0, float(link.payload_bits))
        channel_gain = max(0.0, float(link.channel_gain))
        noise_power = max(0.0, float(link.noise_power))

        tx_power = 0.0 if link.tx_power is None else max(0.0, float(link.tx_power))
        link_capacity_bps = max(0.0, float(link.link_capacity_bps))
        tx_time = max(0.0, float(link.tx_time))

        user_idx = int(link.user_idx)
        layer_idx = int(link.layer_idx)

        if (not scheduled) and (payload_bits > 0.0 or delivered_chunks > 0):
            raise ValueError(f"user {user_idx}: 스케줄링되지 않은 link는 양수 값의 payload/chunks를 가질 수 없습니다.")
        
        if payload_bits > 0.0 and link_capacity_bps > 0.0 and tx_time > 0.0:
            if payload_bits > link_capacity_bps * tx_time + 1e-9:
                raise ValueError(f"user {user_idx}: payload bits가 link_capacit_bps * tx_time 값을 초과합니다.")
        
        
        validated.append(
            CommLinkInput(
                scheduled=scheduled,
                delivered_chunks=delivered_chunks,
                payload_bits=payload_bits,
                channel_gain=channel_gain,
                noise_power=noise_power,
                tx_power=tx_power,
                user_idx=user_idx,
                layer_idx=layer_idx,
                link_capacity_bps=link_capacity_bps,
                tx_time=tx_time,
            )
        )

    return validated


def can_serve(
    config: BatteryConfig,
    soc: float,
) -> bool:
    """
    UAV의 Serving 여부를 검증하는 함수
    """
    return float(soc) > float(config.e_min)


def is_outage(
    soc: float,
) -> bool:
    return float(soc) <= 0.0