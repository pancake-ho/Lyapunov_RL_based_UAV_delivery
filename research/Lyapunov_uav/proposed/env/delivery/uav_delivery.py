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
        hiring, scheduling, chunks>0, power>0, not charging 조건을 만족하는 링크로,
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
    RSU-User 간 capacity [bps] 계산 함수로,
    raw_gain은 path loss, fading 등을 포함한 순수 linear channel gain 의미
    """
    gain = max(float(raw_gain), 0.0)
    gamma_linear = float(channel.db_to_linear(channel.gamma_db))
    snr = gamma_linear * gain
    return float(channel.bandwidth) * math.log2(1.0 + snr)


def _priority_score(
    cfg: EnvConfig,
    req_chunks: int,
    layer: int,
    tx_power: float,
) -> float:
    """
    UAV user cap 초과 시 우선순위를 부여하는 함수로,
    QoE gain을 기준으로 판단
    """
    return (
        float(req_chunks) * _quality_weight(cfg, layer) +
        1e-6 * float(tx_power)
    )


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