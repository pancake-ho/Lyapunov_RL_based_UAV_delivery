from typing import List

from config import BatteryConfig
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
        delivered_chunks = max(0, int(link.delivered_chunks))
        delivered_layers = max(0, int(link.delivered_layers))
        payload_bits = max(0.0, float(link.payload_bits))
        channel_gain = max(0.0, float(link.channel_gain))
        noise_power = max(0.0, float(link.noise_power))
        tx_power = max(0.0, float(link.tx_power))
        link_capacity_bps = max(0.0, float(link.link_capacity_bps))
        tx_time = max(0.0, float(link.tx_time))
        layer = max(0, int(link.layer))
        user_idx = int(link.user_idx)

        if payload_bits > link_capacity_bps * tx_time + 1e-9:
            raise ValueError(f"user {user_idx}: payload bits가 capacity * time 값을 초과합니다.")
        
        if (not link.scheduled) and (payload_bits > 0 or delivered_chunks > 0):
            raise ValueError("UAV가 스케줄링되지 않았는데 payload/chunks가 양수입니다.")
        
        validated.append(
            CommLinkInput(
                scheduled=bool(link.scheduled),
                delivered_chunks=delivered_chunks,
                delivered_layers=delivered_layers,
                payload_bits=payload_bits,
                channel_gain=channel_gain,
                noise_power=noise_power,
                tx_power=tx_power,
                user_idx=user_idx,
                layer=layer,
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