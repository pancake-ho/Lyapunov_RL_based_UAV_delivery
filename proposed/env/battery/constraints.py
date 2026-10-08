from __future__ import annotations

from typing import Iterable, List

try:
    from proposed.config import BatteryConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import BatteryConfig

from .battery_types import BatteryAction, UAVBatteryMode, CommLinkInput


def validate_action_mode(action: BatteryAction) -> None:
    """
    UAV battery mode와 link/mu_active 조합을 검증. (안전장치)
    """
    mode = UAVBatteryMode(action.mode)

    if not bool(action.mu_active) and mode in (UAVBatteryMode.SERVE, UAVBatteryMode.CHARGE):
        raise ValueError("고용되지 않은 UAV는 SERVE/CHARGE mode가 될 수 없습니다.")
    
    if mode == UAVBatteryMode.CHARGE and action.links:
        raise ValueError("UAV는 CHARGE mode에서 links가 비어 있어야 합니다.")
    
    if mode in (UAVBatteryMode.IDLE, UAVBatteryMode.OUTAGE) and action.links:
        raise ValueError("UAV는 IDLE/OUTAGE mode에서 links가 비어 있어야 합니다.")


def validate_links(links: List[CommLinkInput],) -> List[CommLinkInput]:
    """
    battery energy 계산 이전에 CommLinkInput 값을 검증.

    delivery_module에서 capacity clipping을 이미 수행하지만,
    안전장치 용으로 걸어두는 함수.
    """
    validated: List[CommLinkInput] = []

    for link in links:
        scheduled = bool(link.scheduled)
        delivered_chunks = max(0, int(link.delivered_chunks))
        delivered_layers = max(0, int(link.delivered_layers))
        payload_bits = max(0.0, float(link.payload_bits))
        channel_gain = max(0.0, float(link.channel_gain))
        noise_power = max(1e-30, float(link.noise_power))

        tx_power = None if link.tx_power is None else max(0.0, float(link.tx_power))
        link_capacity_bps = max(0.0, float(link.link_capacity_bps))
        tx_time = max(0.0, float(link.tx_time))

        user_idx = int(link.user_idx)
        layer_idx = int(link.layer_idx)

        if (not scheduled) and (payload_bits > 0.0 or delivered_chunks > 0):
            raise ValueError(
                f"user {user_idx}: scheduled=False link는 양수 payload/chunks를 가질 수 없습니다."
            )

        if scheduled and payload_bits > 0.0:
            if delivered_chunks <= 0:
                raise ValueError(
                    f"user {user_idx}: payload_bits는 양수인데 delivered_chunks가 0 이하입니다."
                )
            if delivered_layers <= 0:
                raise ValueError(
                    f"user {user_idx}: payload_bits는 양수인데 delivered_layers가 0 이하입니다."
                )
            if channel_gain <= 0.0:
                raise ValueError(
                    f"user {user_idx}: scheduled link의 channel_gain은 양수여야 합니다."
                )

        if payload_bits > 0.0 and link_capacity_bps > 0.0 and tx_time > 0.0:
            if payload_bits > link_capacity_bps * tx_time + 1e-6:
                raise ValueError(
                    f"user {user_idx}: payload_bits가 link_capacity_bps * tx_time을 초과합니다. "
                    f"payload_bits={payload_bits}, capacity_budget={link_capacity_bps * tx_time}"
                )

        validated.append(
            CommLinkInput(
                scheduled=scheduled,
                delivered_layers=delivered_layers,
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
    UAV의 service mode로 진입 가능한 지 확인:
    
    현재 기준:
        soc > e_min이면 service 후보로 허용하고,
        실제 energy consumption 이후 soc가 0으로 떨어지는 지는 step 결과의 outage로 기록.
    """
    return float(soc) > float(config.e_min)


def is_outage(
    soc: float,
) -> bool:
    """
    battery SoC가 완전히 고갈되었는 지 확인.
    """
    return float(soc) <= 0.0
