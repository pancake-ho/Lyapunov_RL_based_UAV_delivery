from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import math
from config import EnvConfig
from channel import RSUChannelModel
from ..types import ParsedAction

@dataclass
class RSUDeliveryResult:
    """
    RSU Delivery 결과 클래스

    - requested_mask:
        rsu_schedule = 1 이고, rsu_chunks > 0 인 링크
    - active_mask:
        실제 이번 slot에서 유효한 RSU delivery 링크
    """
    requested_mask: np.ndarray
    active_mask: np.ndarray

    delivered_chunks: np.ndarray
    delivered_bits: np.ndarray
    delivered_quality: np.ndarray

    raw_channel_gain: np.ndarray
    link_capacity_bps: np.ndarray

    delivered_per_user: np.ndarray
    quality_per_user: np.ndarray


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


def _capacity_bps(channel: RSUChannelModel, raw_gain: float) -> float:
    """
    RSU-User 간 capacity [bps] 계산 함수로,
    raw_gain은 path loss, fading 등을 포함한 순수 linear channel gain 의미
    """
    gain = max(float(raw_gain), 0.0)
    gamma_linear = float(channel.db_to_linear(channel.gamma_db))
    snr = gamma_linear * gain
    return float(channel.bandwidth) * math.log2(1.0 + snr)


def compute_rsu_delivery(
    cfg: EnvConfig,
    parsed: ParsedAction,
    rsu_channel: RSUChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> RSUDeliveryResult:
    """
    RSU-User 간 delivery 동작 구현 함수
    """
    num_rsu = int(cfg.num_rsu)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)

    # rsu-user 간 delivery 활성화 여부
    requested_mask = (
        (parsed.rsu_scheduling == 1) &
        (parsed.rsu_chunks > 0)
    ).astype(np.int32)

    raw_channel_gain = np.zeros((num_rsu, num_user), dtype=np.float32)
    # effective_channel_gain = np.zeros((num_rsu, num_user), dtype=np.float32)
    link_capacity_bps = np.zeros((num_rsu, num_user), dtype=np.float32)

    potential_chunks = np.zeros((num_rsu, num_user), dtype=np.int32)
    potential_quality = np.zeros((num_rsu, num_user), dtype=np.float32)

    # candidate link 별 capacity 계산
    for m in range(num_rsu):
        user_idx = np.flatnonzero(requested_mask[m] == 1)
        for n in user_idx:
            distance = float(parsed.rsu_user_distance[m, n])
            layer = int(parsed.rsu_layers[m, n])
            requested_chunks = int(parsed.rsu_chunks[m, n])

            raw_gain = float(rsu_channel.compute_gain(distance=distance, rng=rng))
            cap_bps = _capacity_bps(rsu_channel, raw_gain)

            raw_channel_gain[m, n] = raw_gain
            link_capacity_bps[m, n] = cap_bps

            chunk_bits = _chunk_size_bits(cfg, layer)
            if chunk_bits <= 0.0:
                print("전송 가능한 chunk가 없습니다.")
                feasible_chunks = 0
            else:
                bit_budget = cap_bps * slot_duration
                feasible_chunks = int(np.floor(bit_budget / chunk_bits))
            
            feasible_chunks = max(0, min(requested_chunks, feasible_chunks))
            potential_chunks[m, n] = feasible_chunks
            potential_quality[m, n] = (
                float(feasible_chunks) * _quality_weight(cfg, layer)
            )
    
    # 동일 user를 여러 RSU가 동시에 스케줄링하는 경우를 방지하기 위한 안전장치
    # 이 경우, achievable quality가 가장 큰 RSU 하나만 남김
    active_mask = np.zeros((num_rsu, num_user), dtype=np.int32)
    for n in range(num_user):
        providers = np.flatnonzero(requested_mask[:, n] == 1)
        if providers.size == 0:
            print("chunk를 요청하는 user가 존재하지 않습니다.")
            continue

        scores = (
            potential_quality[providers, n] +
            1e-9 * link_capacity_bps[providers, n]
        )
        best_idx = int(np.argmax(scores))
        best_rsu = int(providers[best_idx])

        if potential_chunks[best_rsu, n] > 0:
            active_mask[best_rsu, n] = 1
    
    # 최종 delivery 확정
    delivered_chunks = np.zeros((num_rsu, num_user), dtype=np.int32)
    delivered_bits = np.zeros((num_rsu, num_user), dtype=np.float32)
    delivered_quality = np.zeros((num_rsu, num_user), dtype=np.float32)

    for m in range(num_rsu):
        user_idx = np.flatnonzero(active_mask[m] == 1)
        for n in user_idx:
            layer = int(parsed.rsu_layers[m, n])
            chunks = int(potential_chunks[m, n])

            chunk_bits = _chunk_size_bits(cfg, layer)
            quality_weight = _quality_weight(cfg, layer)

            delivered_chunks[m, n] = chunks
            delivered_bits[m, n] = float(chunks) * float(chunk_bits)
            delivered_quality[m, n] = float(chunks) * float(quality_weight)
    
    delivered_per_user = delivered_chunks.sum(axis=0).astype(np.float32)
    quality_per_user = delivered_quality.sum(axis=0).astype(np.float32)

    return RSUDeliveryResult(
        requested_mask=requested_mask,
        active_mask=active_mask,
        delivered_chunks=delivered_chunks,
        delivered_bits=delivered_bits,
        delivered_quality=delivered_quality,
        raw_channel_gain=raw_channel_gain,
        link_capacity_bps=link_capacity_bps,
        delivered_per_user=delivered_per_user,
        quality_per_user=quality_per_user,
    )