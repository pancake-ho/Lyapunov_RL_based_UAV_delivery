from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig

from ..battery import CommLinkInput
from ..channel import UAVChannelModel
from ..action_types import SlowAction, FastAction
from ..util import _ensure_shape, _safe_get_attr


@dataclass
class UAVDeliveryResult:
    """
    UAV Delivery 결과 클래스

    - requested_mask:
        hiring, scheduling, residual-user, chunks>0, layers>0, power>0, 
        not-charging, battery-feasible, cache-feasible 조건을 모두 만족하는
        1차 전송 후보 link

    - capped_mask:
        UAV별 동시 active user 수 cap 적용 후 남은 link

    - active_mask:
        동일 user에 대해 복수 UAV가 경쟁할 경우 최종 실제 전송 link
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


def _priority_score(
    cfg: EnvConfig,
    feasible_chunks: int,
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
        + 1.0 * float(feasible_chunks) * _quality_weight(cfg, layer)
        + 1e-6 * float(estimated_cap)
        + 1e-6 * float(tx_power)
        - 1.0 * float(battery_penalty)
    )


def _normalize_battery_soc(
    battery_state,
    num_uav: int,
) -> np.ndarray:
    """
    battery state가 다양한 shape으로 들어와도 (num_uav,) 구조를 맞춰주는 함수
    """
    if battery_state is None:
        return np.zeros((num_uav,), dtype=np.float32)
    
    # case 1: battery_state 자체가 soc 속성을 가지는 경우 대비
    soc_attr = _safe_get_attr(battery_state, ["soc"], None)
    if soc_attr is not None:
        arr = np.asarray(soc_attr, dtype=np.float32)
        if arr.ndim == 0:
            return np.full((num_uav,), float(arr.item()), dtype=np.float32)
        return _ensure_shape(arr, (num_uav,), np.float32, fill_value=0.0)
    
    # case 2: array-like 직접 전달
    return _ensure_shape(battery_state, (num_uav,), np.float32, fill_value=0.0)
            
    
def compute_uav_delivery(
    cfg: EnvConfig,
    slow_act: SlowAction,
    fast_act: FastAction,
    battery_parsed,
    uav_channel: UAVChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> UAVDeliveryResult:
    """
    UAV-User delivery 동작 구현 함수
    """
    generator = rng if rng is not None else np.random.default_rng()

    num_uav = int(cfg.num_uav)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)    
    user_cap = max(0, int(cfg.uav_user_cap))

    # slow timescale
    # 고용 여부와 스케줄링 여부를 결정
    hire_mask = _ensure_shape(
        _safe_get_attr(slow_act, ["uav_hiring"], None),
        (num_uav,),
        bool,
        fill_value=False,
    )

    sched_mask = _ensure_shape(
        _safe_get_attr(slow_act, ["uav_scheduling"], None),
        (num_uav, num_user),
        bool,
        fill_value=False,
    )

    # residual user set
    # True 이면, UAV로 처리되지 못해 UAV의 후보에 포함된 것
    residual_mask = _ensure_shape(
        _safe_get_attr(fast_act, ["residual_users"], None),
        (num_user,),
        bool,
        fill_value=True,
    )

    # fast timescale
    # 청크 수 및 레이어 수, 송신 전력 제어를 결정
    req_chunks = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_chunks"], None),
        (num_uav, num_user),
        np.int32,
        fill_value=0,
    )

    req_layers = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_layers"], None),
        (num_uav, num_user),
        np.int32,
        fill_value=0,
    )

    tx_power = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_power"], None),
        (num_uav, num_user),
        np.float32,
        fill_value=0.0,
    )
    tx_power = np.clip(tx_power, 0.0, float(cfg.battery.max_tx_power))

    # 충전 여부 및 state
    charge_mask = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_charge"], None),
        (num_uav,),
        bool,
        fill_value=False,
    )

    battery_soc = _normalize_battery_soc(battery_parsed, num_uav)

    # distance 및 user side information 관련
    uav_user_distance = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_user_distance"], None),
        (num_uav, num_user),
        np.float32,
        fill_value=float(cfg.uav_channel.distance),
    )

    user_virtual_queue = _ensure_shape(
        _safe_get_attr(fast_act, ["user_virtual_queue"], None),
        (num_user,),
        np.float32,
        fill_value=0.0,
    )

    requested_content = _ensure_shape(
        _safe_get_attr(fast_act, ["requested_content"], None),
        (num_user,),
        np.int32,
        fill_value=-1,
    )

    cached_content = _ensure_shape(
        _safe_get_attr(fast_act, ["uav_cached_content"], None),
        (num_uav,),
        np.int32,
        fill_value=-1,
    )

    requested_mask = np.zeros((num_uav, num_user), dtype=bool)
    capped_mask = np.zeros((num_uav, num_user), dtype=bool)
    active_mask = np.zeros((num_uav, num_user), dtype=bool)

    delivered_chunks = np.zeros((num_uav, num_user), dtype=np.int32)
    delivered_bits = np.zeros((num_uav, num_user), dtype=np.float32)
    delivered_quality = np.zeros((num_uav, num_user), dtype=np.float32)

    raw_channel_gain = np.zeros((num_uav, num_user), dtype=np.float32)
    link_capacity_bps = np.zeros((num_uav, num_user), dtype=np.float32)

    potential_chunks = np.zeros((num_uav, num_user), dtype=np.int32)
    potential_quality = np.zeros((num_uav, num_user), dtype=np.float32)

    links_uav: list[list[CommLinkInput]] = [[] for _ in range(num_uav)]


    # requested link 생성 및 link feasiblity 계산
    for u in range(num_uav):
        if not bool(hire_mask[u]):
            continue
        if bool(charge_mask[u]):
            continue
        if float(battery_soc[u]) < float(cfg.battery.e_min):
            continue

        for n in range(num_user):
            if not bool(sched_mask[u, n]):
                continue     
            if not bool(residual_mask[n]):
                continue
            if int(req_chunks[u, n]) <= 0:
                continue
            if int(req_layers[u, n]) <= 0:
                continue
            if float(tx_power[u, n]) <= 0.0:
                continue
            
            # cache 제약: cache 정보가 둘 다 알려진 경우에만 검사
            cache_known = (int(cached_content[u]) >= 0) and (int(requested_content[n]) >= 0)
            if cache_known and int(cached_content[u]) != int(requested_content[n]):
                continue

            requested_mask[u, n] = True
            
            distance = max(
                float(uav_user_distance[u, n]),
                float(cfg.uav_channel.min_distance),
            )
            raw_gain = float(uav_channel.compute_gain(distance=distance, rng=generator))
            cap_bps = float(
                uav_channel.capacity_from_gain(
                    tx_power=float(tx_power[u, n]),
                    gain=raw_gain,
                )
            )

            raw_channel_gain[u, n] = raw_gain
            link_capacity_bps[u, n] = cap_bps

            chunk_bits = _chunk_size_bits(cfg, int(req_layers[u, n]))
            if chunk_bits <= 0.0:
                feasible = 0
            else:
                bit_budget = cap_bps * slot_duration
                feasible = int(np.floor(bit_budget / chunk_bits))
            
            feasible = max(0, min(int(req_chunks[u, n]), feasible))
            potential_chunks[u, n] = feasible
            potential_quality[u, n] = float(feasible) * _quality_weight(cfg, int(req_layers[u, n]))
    
    # UAV별 동시 서비스 가능 user cap 적용
    for u in range(num_uav):
        candidate_users = np.flatnonzero(requested_mask[u])
        if candidate_users.size == 0:
            continue
        if user_cap <= 0:
            continue

        scores = np.array(
            [
                _priority_score(
                    cfg=cfg,
                    feasible_chunks=int(potential_chunks[u, n]),
                    layer=int(req_layers[u, n]),
                    tx_power=float(tx_power[u, n]),
                    user_virtual_queue=float(user_virtual_queue[n]),
                    estimated_cap=float(link_capacity_bps[u, n]),
                    battery_penalty=max(0.0, float(cfg.battery.e_min) - float(battery_soc[u])),
                )
                for n in candidate_users
            ],
            dtype=np.float64,
        )

        order = np.argsort(-scores)
        selected_users = candidate_users[order[:user_cap]]

        for n in selected_users:
            if potential_chunks[u, n] > 0:
                capped_mask[u, n] = True
    
    # 동일 user에 대해 여러 UAV가 경쟁할 경우 대비
    for n in range(num_user):
        candidate_uavs = np.flatnonzero(capped_mask[:, n])
        if candidate_uavs.size == 0:
            continue

        if candidate_uavs.size == 1:
            u = int(candidate_uavs[0])
            if potential_chunks[u, n] > 0:
                active_mask[u, n] = True
            continue

        best_u = int(candidate_uavs[0])
        best_score = -np.inf

        for u in candidate_uavs:
            u = int(u)
            score = _priority_score(
                cfg=cfg,
                feasible_chunks=int(potential_chunks[u, n]),
                layer=int(req_layers[u, n]),
                tx_power=float(tx_power[u, n]),
                user_virtual_queue=float(user_virtual_queue[n]),
                estimated_cap=float(link_capacity_bps[u, n]),
                battery_penalty=max(0.0, float(cfg.battery.e_min) - float(battery_soc[u])),
            )
            if score > best_score:
                best_score = score
                best_u = u
        
        if potential_chunks[best_u, n] > 0:
            active_mask[best_u, n] = True
    
    # 최종 delivery 확정 및 battery 연동용 CommLinkInput 생성
    active_links = np.argwhere(active_mask)
    for u, n in active_links:
        layer = int(req_layers[u, n])
        chunks = int(potential_chunks[u, n])

        chunk_bits = _chunk_size_bits(cfg, layer)
        quality_weight = _quality_weight(cfg, layer)
        payload_bits = float(chunks) * float(chunk_bits)

        delivered_chunks[u, n] = chunks
        delivered_bits[u, n] = payload_bits
        delivered_quality[u, n] = float(chunks) * float(quality_weight)

        links_uav[u].append(
            CommLinkInput(
                scheduled=True,
                delivered_layers=layer,
                delivered_chunks=chunks,
                payload_bits=payload_bits,
                channel_gain=float(raw_channel_gain[u, n]),
                noise_power=float(uav_channel.noise_power),
                tx_power=float(tx_power[u, n]),
                user_idx=int(n),
                layer_idx=int(layer),
                link_capacity_bps=float(link_capacity_bps[u, n]),
                tx_time=int(slot_duration),
            )
        )

    delivered_per_user = delivered_chunks.sum(axis=0).astype(np.float32)
    quality_per_user = delivered_quality.sum(axis=0).astype(np.float32)

    return UAVDeliveryResult(
        requested_mask=requested_mask,
        capped_mask=capped_mask,
        active_mask=active_mask,
        delivered_chunks=delivered_chunks,
        delivered_bits=delivered_bits,
        delivered_quality=delivered_quality,
        raw_channel_gain=raw_channel_gain,
        link_capacity_bps=link_capacity_bps,
        tx_power=tx_power,
        charge_mask=charge_mask,
        delivered_per_user=delivered_per_user,
        quality_per_user=quality_per_user,
        links_uav=links_uav,
    )
