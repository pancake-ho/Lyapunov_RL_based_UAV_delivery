from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class LowObsSpec:
    """
    Low-level PPO observation vector 구조 정보
    """
    obs_dim: int
    obs: Dict[str, slice]


class LowObsAdapter:
    """
    Env state dict를 low-level PPO용 1D observation vector로 변환하는 클래스

    Low-level policy는 slot 단위 fast-timescale controller이므로,
    round-level slow decision은 관측 정보로 포함하되 직접 변경하지 않는다.

    포함 정보:
        - user queue Q, virtual queue Z
        - UAV actual SoC E, battery virtual queue Y
        - high-level slow action masks: RSU scheduling, UAV hiring, UAV scheduling
        - UAV outage/charging state
        - round slot index
        - requested/cached content id
    """
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.num_rsu = cfg.num_rsu
        self.num_user = cfg.num_user
        self.num_uav = cfg.num_uav

        self.max_queue = 