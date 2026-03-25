from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

EnvAction = Dict[str, Any]


@dataclass
class ParsedAction:
    """
    slow 및 fast timescale에서 가능한 모든 동작을 파싱하는 클래스
    """
    # slow-timescale (round-level)
    rsu_scheduling: np.ndarray      # y_mn(r), shape (M,N)
    uav_hiring: np.ndarray          # mu_m(r), shape (M,)
    uav_scheduling: np.ndarray      # phi_un(r), shape (U,N)

    # fast-timescale (slot-level)
    rsu_chunks: np.ndarray          # l_mn(t), shape (M,N)
    rsu_layers: np.ndarray          # k_mn(t), shape (M,N)

    uav_chunks: np.ndarray          # l_un(t), shape (U, N)
    uav_layers: np.ndarray          # k_un(t), shape (U, N)
    uav_power: np.ndarray           # p_un(t), shape (U, N)
    uav_charge: np.ndarray          # I_u(t), shape (U,)

    playback: np.ndarray            # b_n(t), shape (N,)
    rsu_user_distance: np.ndarray   # shape: (M, N)
    uav_user_distance: np.ndarray   # shape: (U, N)

    # residual users
    residual_users: np.ndarray


@dataclass
class StepResult:
    state: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]