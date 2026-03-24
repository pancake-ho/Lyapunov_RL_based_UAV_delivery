from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import math

from config import EnvConfig
from battery import CommLinkInput
from channel import UAVChannelModel
from ..types import ParsedAction

@dataclass
class UAVDeliveryResult:
    """
    UAV Delivery 결과 클래스

    - requested_mask:
        hiring, scheduling, chunks>0, power>0, not charging 조건을 만족하는
        가장 처음의 전송 후보 집합

    - capped_mask:
        UAV별 동시 active user 수 cap 적용 후 남은 링크로,
        한 UAV가 동시에 너무 많은 user를 서비스하지 못한다는 제약을 반영한 집합

    - active_mask:
        최종 실제 전송 링크
        (동일 user에 대해 복수 UAV가 들어오면 내부적으로 1개 선택)
    """
    requested_mask: np.ndarray
    capped_mask: np.ndarray
    active_mask: np.ndarray

    delivered_chunks: np.ndarray
    delivered_bits: np.ndarray
    delivered_quality: np.ndarray

    raw_channel_gain: np.ndarray
    link_capacity_bps: np.ndarray

    tx_power: np.ndarray
    charge_mask: np.ndarray

    delivered_per_user: np.ndarray
    quality_per_user: np.ndarray

    links_uav: list[list[CommLinkInput]]


def _quality_weight(cfg: EnvConfig, layer_idx: int) -> float:
    """
    layer에 따른 quality weights를 반환하는 함수
    """
    layer = int(layer_idx)
    if layer <= 0:
        return 0.0
    if layer <= len(cfg.quality_weights):
        return float(cfg.quality_weights[layer - 1])
    return float(layer)


def _chunk_size_bits(cfg: EnvConfig, layer_idx: int) -> float:
    """
    layer 수에 따른 chunk size를 bit 단위로 계산하는 함수
    """
    layer = int(layer_idx)
    if layer <= 0:
        return 0.0
    return float(cfg.base_chunk_size_bits) * float(layer)


def _capacity_bps(
        channel: UAVChannelModel, 
        tx_power: float,
        raw_gain: float
    ) -> float:
    """
    UAV-User 간 capacity [bps] 계산 함수로,
    raw_gain은 path loss, fading 등을 포함한 순수 linear channel gain 의미하며
    송신 power도 제어할 수 있음.
    """
    gain = max(float(raw_gain), 0.0)
    p = max(float(tx_power), 0.0)
    gamma_linear = float(channel.db_to_linear(channel.gamma_db))
    snr = p * gamma_linear * gain
    return float(channel.bandwidth) * math.log2(1.0 + snr)


def _priority_score(
    cfg: EnvConfig,
    req_chunks: int,
    layer: int,
    tx_power: float,
    user_virtual_queue: float = 0.0,
    estimated_cap: float = 0.0,
    battery_penalty: float = 0.0,
) -> float:
    """
    UAV user cap 초과 시 active 후보 우선순위를 계산하는 함수로,
    현재는 임시로 "User queue + qualty gain + channel + energy"를 함께 고려하는 score 형태로 설정
    """
    return (
        2.0 * float(user_virtual_queue)
        + 1.0 * float(req_chunks) * _quality_weight(cfg, layer)
        + 1e-6 * float(estimated_cap)
        + 1e-6 * float(tx_power)
        - 1.0 * float(battery_penalty)
    )


def _select_uav_for_user(
    cfg: EnvConfig,
    user_idx: int,
    candidate_uavs: np.ndarray,
    req_chunks: np.ndarray,
    req_layers: np.ndarray,
    tx_power: np.ndarray,
    estimated_cap_bps: np.ndarray,
    battery_soc: np.ndarray,
    user_virtual_queue: np.ndarray,
) -> int:
    """
    동일한 user에 대해 복수 UAV가 경쟁할 경우 하나를 선택하게 하는 함수
    """
    best_uav = int(candidate_uavs[0])
    best_score = -np.inf

    for u in candidate_uavs:
        u = int(u)
        score = _priority_score(
            cfg=cfg,
            req_chunks=int(req_chunks[u, user_idx]),
            layer=int(req_layers[u, user_idx]),
            tx_power=float(tx_power[u, user_idx]),
            user_virtual_queue=float(user_virtual_queue[user_idx]),
            estimated_cap=float(estimated_cap_bps[u, user_idx]),
            battery_penalty=max(0.0, float(cfg.battery.e_min) - float(battery_soc[u]))
        )
        if score > best_score:
            best_score = score
            best_uav = u
    
    return best_uav
            
    
def compute_uav_delivery(
    cfg: EnvConfig,
    parsed: ParsedAction,
    uav_channel: UAVChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> UAVDeliveryResult:
    """
    UAV-User 간 delivery 동작 구현 함수
    """
    num_uav = int(cfg.num_uav)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)    
    user_cap = max(0, int(cfg.uav_user_cap))