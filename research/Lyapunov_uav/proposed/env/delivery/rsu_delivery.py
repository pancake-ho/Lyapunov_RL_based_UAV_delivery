from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig

from ..action_types import SlowAction, FastAction
from ..channel import RSUChannelModel
from ..util import _ensure_shape, _safe_get_attr


@dataclass
class RSUDeliveryResult:
    """
    RSU Delivery 결과 클래스

    - requested_mask:
        RSU-user scheduling = 1 이고, chunk/layer request 가 양수인 1차 후보 link

    - capped_mask:
        RSU별 동시 서비스 가능 user 수(capacity) 적용 후 남은 link

    - active_mask:
        동일 user에 대해 복수 RSU가 동시에 선택된 경우,
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
    layer index에 따른 quality weight를 반환하는 함수
    """
    layer = int(layer_idx)
    if layer <= 0:
        return 0.0
    if layer <= len(cfg.quality_weights):
        return float(cfg.quality_weights[layer - 1])
    return float(layer)


def _chunk_size_bits(cfg: EnvConfig, layer_idx: int) -> float:
    """
    layer 수에 따른 chunk size [bits]를 반환한다.

    현재 연구 기준:
        S(k) = config.chunk_size_bits[k - 1]

    IoTJ2025 설정:
        [2.621, 5.073, 10.658, 26.496] Kbits
    """
    layer = int(layer_idx)
    if layer <= 0:
        return 0.0

    if layer <= len(cfg.chunk_size_bits):
        return float(cfg.chunk_size_bits[layer - 1])

    raise ValueError(
        f"layer_idx={layer}에 대응하는 chunk_size_bits가 없습니다. "
        f"len(chunk_size_bits)={len(cfg.chunk_size_bits)}"
    )


def _clip_int_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    dtype: np.dtype,
    low: int,
    high: int,
    name: str,
) -> np.ndarray:
    """
    action matrix를 지정 shape/dtype으로 맞춘 뒤 [low, high] 범위로 clipping한다.
    """
    arr = _ensure_shape(value, shape, dtype, fill_value=0, strict=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=high, neginf=low)
    return np.clip(arr, low, high).astype(dtype, copy=False)


def _clip_float_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    low: float,
    high: Optional[float],
    name: str,
) -> np.ndarray:
    """
    float matrix를 지정 shape으로 맞춘 뒤 clipping한다.
    """
    arr = _ensure_shape(value, shape, np.float32, fill_value=0.0, strict=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=high if high is not None else 0.0, neginf=low)
    if high is None:
        return np.maximum(arr, low).astype(np.float32, copy=False)
    return np.clip(arr, low, high).astype(np.float32, copy=False)

def _priority_score(cfg: EnvConfig, feasible_chunks: int, layer: int, cap_bps: float, user_virtual_queue: float=0.0) -> float:
    """
    RSU capacity 초과 또는 동일 user conflict가 발생했을 때 사용하는 score 계산 함수로,
    invalid action에 대한 안전장치 목적으로 구현.
    """
    quality_gain = float(feasible_chunks) * _quality_weight(cfg, layer)
    return (
        2.0 * float(user_virtual_queue)
        + 1.0 * float(quality_gain)
        + 1e-9 * float(cap_bps)
    )


def compute_rsu_delivery(
    cfg: EnvConfig,
    slow_act: SlowAction,
    fast_act: FastAction,
    rsu_channel: RSUChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> RSUDeliveryResult:
    """
    RSU-User 간 delivery 계산 함수로,

        1) requested_mask 생성
        2) 각 candidate link의 instantaneous channel 및 feasible chunks 계산
        3) RSU capacity를 적용하여 capped_mask 생성
        4) 동일 user에 대한 복수 RSU conflict 제거 (active_mask 생성)
        5) 최종 delivery

    을 담당.
    """
    num_rsu = int(cfg.num_rsu)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)
    rsu_capacity = max(0, int(cfg.rsu_capacity))

    if slot_duration <= 0.0:
        raise ValueError(f"slot_duration은 양수여야 합니다. 현재 값: {slot_duration}")

    # slow-timescale scheduling term
    rsu_scheduling = _ensure_shape(
        _safe_get_attr(slow_act, ["rsu_scheduling"], None),
        (num_rsu, num_user),
        np.int32,
        fill_value=0,
        strict=False,
    )
    rsu_scheduling = (rsu_scheduling > 0)

    # fast-timescale delivery term
    rsu_chunks = _clip_int_matrix(
        _safe_get_attr(fast_act, ["rsu_chunks"], None),
        (num_rsu, num_user),
        np.int32,
        low=0,
        high=int(cfg.chunk),
        name="fast_act.rsu_chunks",
    )

    rsu_layers = _clip_int_matrix(
        _safe_get_attr(fast_act, ["rsu_layers"], None),
        (num_rsu, num_user),
        np.int32,
        low=0,
        high=int(cfg.layer),
        name="fast_act.rsu_layers",
    )

    rsu_user_distance = _clip_float_matrix(
        _safe_get_attr(fast_act, ["rsu_user_distance"], None),
        (num_rsu, num_user),
        low=float(cfg.rsu_channel.min_distance),
        high=None,
        name="fast_act.rsu_user_distance",
    )

    user_virtual_queue = _ensure_shape(
        _safe_get_attr(fast_act, ["user_virtual_queue"], None),
        (num_user,),
        np.float32,
        fill_value=0.0,
        strict=False,
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

    # 각 candidate link의 slot-t channel realization은 정확히 한 번만
    # 샘플링한다. 과거 구현의 불필요한 ``for m in range(num_rsu)``
    # 바깥 루프는 동일 link를 num_rsu번 다시 계산하고 RNG도 그만큼
    # 소비하여 물리 모델과 실행시간을 모두 왜곡했다.
    candidate_links = np.argwhere(requested_mask)

    for m, n in candidate_links:
        m = int(m)
        n = int(n)

        distance = max(
            float(rsu_user_distance[m, n]),
            float(cfg.rsu_channel.min_distance),
        )
        layer = int(rsu_layers[m, n])
        requested_chunks = int(rsu_chunks[m, n])

        raw_gain = float(
            rsu_channel.compute_gain(
                distance=distance,
                rng=rng,
            )
        )
        cap_bps = float(
            rsu_channel.capacity_from_gain(raw_gain)
        )

        raw_channel_gain[m, n] = raw_gain
        link_capacity_bps[m, n] = cap_bps

        chunk_bits = _chunk_size_bits(cfg, layer)

        if chunk_bits <= 0.0 or cap_bps <= 0.0:
            feasible_chunks = 0
        else:
            feasible_chunks = int(
                np.floor(
                    cap_bps
                    * slot_duration
                    / chunk_bits
                )
            )

        potential_chunks[m, n] = max(
            0,
            min(requested_chunks, feasible_chunks),
        )
    
    # RSU별 동시 서비스 가능 user 수 제한 반영
    capped_mask = np.zeros((num_rsu, num_user), dtype=bool)

    for m in range(num_rsu):
        candidate_users = np.flatnonzero(
            requested_mask[m]
        )

        if (
            candidate_users.size == 0
            or rsu_capacity <= 0
        ):
            continue

        if candidate_users.size <= rsu_capacity:
            selected_users = candidate_users

        else:
            scores = np.asarray(
                [
                    _priority_score(
                        cfg=cfg,
                        feasible_chunks=int(
                            potential_chunks[m, n]
                        ),
                        layer=int(
                            rsu_layers[m, n]
                        ),
                        cap_bps=float(
                            link_capacity_bps[m, n]
                        ),
                        user_virtual_queue=float(
                            user_virtual_queue[n]
                        ),
                    )
                    for n in candidate_users
                ],
                dtype=np.float64,
            )

            order = np.argsort(
                -scores
            )
            selected_users = candidate_users[
                order[:rsu_capacity]
            ]

        positive = (
            potential_chunks[
                m,
                selected_users,
            ]
            > 0
        )

        capped_mask[
            m,
            selected_users[positive],
        ] = True
    
    # 동일 user를 여러 RSU가 동시에 서비스하는 경우 방지 (안전장치)
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
                    user_virtual_queue=float(user_virtual_queue[n]),
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