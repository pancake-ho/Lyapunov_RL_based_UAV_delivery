from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import math

from config import EnvConfig
from battery import CommLinkInput, BatteryState
from channel import UAVChannelModel
from ..types import ParsedAction
from util import _ensure_shape, _safe_get_attr

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
    battery_parsed: BatteryState,
    uav_channel: UAVChannelModel,
    rng: Optional[np.random.Generator] = None,
) -> UAVDeliveryResult:
    """
    UAV-User 간 delivery 동작 구현 함수
    """
    generator = rng if rng is not None else np.random.default_rng()

    num_uav = int(cfg.num_uav)
    num_user = int(cfg.num_user)
    slot_duration = float(cfg.battery.slot_duration)    
    user_cap = max(0, int(cfg.uav_user_cap))

    # slow timescale
    # 고용 여부와 스케줄링 여부를 결정
    hire_mask = _ensure_shape(
        _safe_get_attr(parsed, ["uav_hiring"], None),
        (num_uav,),
        bool,
        fill_value=False,
    )

    sched_mask = _ensure_shape(
        _safe_get_attr(parsed, ["uav_scheduling"], None),
        (num_uav, num_user),
        bool,
        fill_value=False,
    )

    # residual user set
    # True 이면, UAV로 처리되지 못해 UAV의 후보에 포함된 것
    residual_mask = _ensure_shape(
        _safe_get_attr(parsed, ["residual_users"], None),
        (num_user,),
        bool,
        fill_value=True,
    )

    # fast timescale
    # 청크 수 및 레이어 수, 송신 전력 제어를 결정
    req_chunks = _ensure_shape(
        _safe_get_attr(parsed, ["uav_chunks"], None),
        (num_uav, num_user),
        np.int32,
        fill_value=0,
    )

    req_layers = _ensure_shape(
        _safe_get_attr(parsed, ["uav_layers"], None),
        (num_uav, num_user),
        np.int32,
        fill_value=0,
    )

    tx_power = _ensure_shape(
        _safe_get_attr(parsed, ["uav_power"], None),
        (num_uav, num_user),
        np.float32,
        fill_value=0.0,
    )
    tx_power = np.clip(tx_power, 0.0, float(cfg.battery.max_tx_power))

    # 충전 여부 및 state
    charge_mask = _ensure_shape(
        _safe_get_attr(parsed, ["uav_charge"], None),
        (num_uav,),
        bool,
        fill_value=False,
    )

    battery_soc = _ensure_shape(
        _safe_get_attr(battery_parsed, ["soc"], None),
        (num_uav,),
        np.int32,
        fill_value=0,
    )

    # content/cache 관련
    cached_content = _ensure_shape(
        _safe_get_attr(parsed, ["uav_cached_content"], None),
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

    links_uav: list[list[CommLinkInput]] = [[] for _ in range(num_uav)]


    # requested link 생성
    for u in range(num_uav):
        # 고용되지 않은 UAV는 서비스 불가
        if not bool(hire_mask[u]):
            print("고용되지 않은 UAV는 서비스 불가합니다.")
            continue

        # charging mode이면 서비스 불가
        if bool(charge_mask[u]):
            print("충전 중인 UAV는 서비스 불가합니다.")
            continue

        # 배터리 하한 미만이면 서비스 불가
        if float(battery_soc[u]) < float(cfg.battery.e_min):
            print("UAV의 배터리가 서비스 가능한 범위보다 작으므로 서비스 불가합니다.")
            continue

        for n in range(num_user):
            # scheduling 안 되어있으면 서비스 불가
            if not bool(sched_mask[u, n]):
                print("스케줄링되지 않은 UAV는 서비스 불가합니다.")
            
            # 요청 chunk/layer/power가 0 이하이면 서비스 불가
            if int(req_chunks[u, n]) <= 0:
                print("UAV에게 요청하는 chunk 개수가 0 이하이므로 서비스 불가합니다.")
                continue
            if int(req_layers[u, n]) <= 0:
                print("UAV에게 요청하는 layer 개수가 0 이하이므로 서비스 불가합니다.")
                continue
            if float(tx_power[u, n]) <= 0.0:
                print("UAV의 tx power가 0 이하이므로 서비스 불가합니다.")
                continue
            
            # content cache 정보가 있다면, 일치 여부 확인
            cache_known = (int(cached_content[u]) >= 0) and (int(requst))