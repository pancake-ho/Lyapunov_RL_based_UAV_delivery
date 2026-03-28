from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import EnvConfig
from ..channel import RSUChannelModel
from ..action_types import ParsedAction

@dataclass
class RSUDeliveryResult:
    """
    RSU Delivery 결과 클래스

    - requested_mask:
        scheduling = 1, chunks > 0, layers > 0인 1차 delivery 후보 link

    - capped_mask:
        RSU별 동시 서비스 가능 user 수(capacity) 적용 후 남은 link

    - active_mask:
        동일 user에 대해 복수 RSU가 경쟁할 경우,
        최종적으로 실제 delivery를 수행할 link만 남긴 mask
    """
    requested_mask: np.ndarray
    capped_mask: np.ndarray
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
    layer 수에 따른 chunk size [bits] 계산하는 함수로,
    현재는 base_chunk_size_bits * k의 단순 선형 모델로 구현했음.
    """
    layer = int(layer_idx)
    if layer <= 0:
        return 0.0
    return float(cfg.base_chunk_size_bits) * float(layer)


def _safe_int_array(arr: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
    return arr.astype(np.int32, copy=False)


def _safe_float_array(arr: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape != shape:
        raise ValueError(f"{name} shape mismatch: expected {shape}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _priority_score(cfg: EnvConfig, feasible_chunks: int, layer: int, cap_bps: float) -> float:
    """
    RSU 내부 user 우선 순위 및 user-conflict score 계산 함수로,
    현재는 실제 전송 가능한 quality gain 및 tie-break용 capacity를 함께 고려함.
    """
    return (
        float(feasible_chunks) * _quality_weight(cfg, layer) + 1e-9 * float(cap_bps)
    )


def compute_rsu_delivery(
    cfg: EnvConfig,
    parsed: ParsedAction,
    rsu_channel: RSUChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> RSUDeliveryResult:
    """
    RSU-User 간 delivery 동작 구현 함수로, 다음과 같은 흐름으로 동작함.

        1) requested_mask 생성
        2) 각 candidate link의 instantaneous channel 및 feasible chunks 계산
        3) RSU capacity를 적용하여 capped_mask 생성
        4) 동일 user에 대한 복수 RSU conflict 제거 (active_mask 생성)
        5) 최종 delivery
    """
    num_rsu = int(cfg.num_rsu)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)
    rsu_capacity = max(0, int(cfg.rsu_capacity))

    rsu_scheduling = _safe_int_array(
        parsed.rsu_scheduling, (num_rsu, num_user), "parsed.rsu_scheduling"
    )
    rsu_chunks = _safe_int_array(
        parsed.rsu_chunks, (num_rsu, num_user), "parsed.rsu_chunks"
    )
    rsu_layers = _safe_int_array(
        parsed.rsu_layers, (num_rsu, num_user), "parsed.rsu_layers"
    )
    rsu_user_distance = _safe_float_array(
        parsed.rsu_user_distance, (num_rsu, num_user), "parsed.rsu_user_distance"
    )

    # 1차 delivery 후보
    requested_mask = (
        (rsu_scheduling == 1)
        & (rsu_chunks > 0)
        & (rsu_layers > 0)
    )

    raw_channel_gain = np.zeros((num_rsu, num_user), dtype=np.float32)
    link_capacity_bps = np.zeros((num_rsu, num_user), dtype=np.float32)

    potential_chunks = np.zeros((num_rsu, num_user), dtype=np.int32)
    potential_quality = np.zeros((num_rsu, num_user), dtype=np.float32)

    # candidate link 별 achievable delivery 계산
    for m in range(num_rsu):
        candidate_users = np.flatnonzero(requested_mask[m])
        for n in candidate_users:
            distance = max(float(rsu_user_distance[m, n]), float(cfg.rsu_channel.min_distance))
            layer = int(rsu_layers[m, n])
            requested_chunks = int(rsu_chunks[m, n])

            raw_gain = float(rsu_channel.compute_gain(distance=distance, rng=rng))
            cap_bps = float(rsu_channel.capacity_from_gain(raw_gain))
                            
            raw_channel_gain[m, n] = raw_gain
            link_capacity_bps[m, n] = cap_bps

            chunk_bits = _chunk_size_bits(cfg, layer)
            if chunk_bits <= 0.0:
                feasible_chunks = 0
            else:
                bit_budget = cap_bps * slot_duration
                feasible_chunks = int(np.floor(bit_budget / chunk_bits))
            
            feasible_chunks = max(0, min(requested_chunks, feasible_chunks))
            potential_chunks[m, n] = feasible_chunks
            potential_quality[m, n] = float(feasible_chunks) * _quality_weight(cfg, layer)
    
    # RSU별 동시 서비스 가능 user 수 제한 반영
    capped_mask = np.zeros((num_rsu, num_user), dtype=bool)

    for m in range(num_rsu):
        candidate_users = np.flatnonzero(requested_mask[m])

        if candidate_users.size == 0:
            continue

        if rsu_capacity <= 0:
            continue

        scores = np.array(
            [
                _priority_score(
                    cfg=cfg,
                    feasible_chunks=int(potential_chunks[m, n]),
                    layer=int(rsu_layers[m, n]),
                    cap_bps=float(link_capacity_bps[m, n]),
                )
                for n in candidate_users
            ],
            dtype=np.float64
        )

        # score가 높은 user부터 RSU capacity만큼 선택
        order = np.argsort(-scores)
        selected_users = candidate_users[order[:rsu_capacity]]

        for n in selected_users:
            if potential_chunks[m, n] > 0:
                capped_mask[m, n] = True
    
    # 동일 user를 여러 RSU가 동시에 서비스하는 경우 방지
    active_mask = np.zeros((num_rsu, num_user), dtype=bool)

    for n in range(num_user):
        providers = np.flatnonzero(capped_mask[:, n])
        if providers.size == 0:
            continue

        if providers.size == 1:
            only_rsu = int(providers[0])
            if potential_chunks[only_rsu, n] > 0:
                active_mask[only_rsu, n] = True
            continue

        scores = np.array(
            [
                _priority_score(
                    cfg=cfg,
                    feasible_chunks=int(potential_chunks[m, n]),
                    layer=int(rsu_layers[m, n]),
                    cap_bps=float(link_capacity_bps[m, n]),
                )
                for m in providers
            ],
            dtype=np.float64,
        )

        best_rsu = int(providers[int(np.argmax(scores))])
        if potential_chunks[best_rsu, n] > 0:
            active_mask[best_rsu, n] = True

    # 최종 delivery 산출
    delivered_chunks = np.zeros((num_rsu, num_user), dtype=np.int32)
    delivered_bits = np.zeros((num_rsu, num_user), dtype=np.float32)
    delivered_quality = np.zeros((num_rsu, num_user), dtype=np.float32)

    active_links = np.argwhere(active_mask)
    for m, n in active_links:
        layer = int(rsu_layers[m, n])
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
        capped_mask=capped_mask,
        active_mask=active_mask,
        delivered_chunks=delivered_chunks,
        delivered_bits=delivered_bits,
        delivered_quality=delivered_quality,
        raw_channel_gain=raw_channel_gain,
        link_capacity_bps=link_capacity_bps,
        delivered_per_user=delivered_per_user,
        quality_per_user=quality_per_user,
    )