from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
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
    effective_channel_gain: np.ndarray
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