from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .types import EnvAction, ParsedAction
from ..config import EnvConfig


def _as_binary_matrix(value: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    """
    binary 타입 matrix shape 검증 함수
    """
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return (arr > 0).astype(np.int32)


def _as_binary_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    """
    binary 타입 vector shape 검증 함수
    """
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
    return (arr > 0).astype(np.int32)


def _as_nonneg_int_matrix(
    value: np.ndarray,
    shape: Tuple[int, int],
    name: str,
    min_value: int = 0,
    max_value: Optional[int] = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    arr = np.maximum(arr, min_value)
    if max_value is not None:
        arr = np.minimum(arr, max_value)
    return arr


def _as_nonneg_float_matrix(
    value: np.ndarray,
    shape: Tuple[int, int],
    name: str,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    arr = np.maximum(arr, min_value)
    if max_value is not None:
        arr = np.minimum(arr, max_value)
    return arr


def _as_nonneg_float_vector(
    value: np.ndarray,
    size: int,
    name: str,
    min_value: float = 0.0,
    max_value: Optional[float] = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
    arr = np.maximum(arr, min_value)
    if max_value is not None:
        arr = np.minimum(arr, max_value)
    return arr


def _default_distance_matrix(
    rows: int,
    cols: int,
    distance: float,
    min_distance: float,
) -> np.ndarray:
    return np.full((rows, cols), max(distance, min_distance), dtype=np.float32)


def parse_action(action: EnvAction, cfg: EnvConfig) -> ParsedAction:
    """
    가능한 모든 action을 파싱하는 함수
    """
    m = cfg.num_rsu
    n = cfg.num_user
    u = cfg.num_uav

    rsu_schedule = _as_binary_matrix(
        action.get("rsu_schedule", np.zeros((m, n), dtype=np.int32)),
        (m, n),
        "rsu_schedule",
    )
    uav_hiring = _as_binary_vector(
        action.get("uav_hiring", np.zeros(u, dtype=np.int32)),
        u,
        "uav_hiring",
    )
    uav_schedule = _as_binary_matrix(
        action.get("uav_schedule", np.zeros((u, n), dtype=np.int32)),
        (u, n),
        "uav_schedule",
    )

    rsu_chunks = _as_nonneg_int_matrix(
        action.get("rsu_chunks", np.zeros((m, n), dtype=np.int32)),
        (m, n),
        "rsu_chunks",
        min_value=0,
        max_value=cfg.chunk,
    )
    rsu_layers = _as_nonneg_int_matrix(
        action.get("rsu_layers", np.ones((m, n), dtype=np.int32)),
        (m, n),
        "rsu_layers",
        min_value=1,
        max_value=cfg.layer,
    )

    uav_chunks = _as_nonneg_int_matrix(
        action.get("uav_chunks", np.zeros((u, n), dtype=np.int32)),
        (u, n),
        "uav_chunks",
        min_value=0,
        max_value=cfg.chunk,
    )
    uav_layers = _as_nonneg_int_matrix(
        action.get("uav_layers", np.ones((u, n), dtype=np.int32)),
        (u, n),
        "uav_layers",
        min_value=1,
        max_value=cfg.layer,
    )
    uav_power = _as_nonneg_float_matrix(
        action.get("uav_power", np.zeros((u, n), dtype=np.float32)),
        (u, n),
        "uav_power",
        min_value=0.0,
        max_value=cfg.battery.max_tx_power,
    )
    uav_charge = _as_binary_vector(
        action.get("uav_charge", np.zeros(u, dtype=np.int32)),
        u,
        "uav_charge",
    )

    playback = _as_nonneg_float_vector(
        action.get("playback", np.full(n, cfg.playback_rate, dtype=np.float32)),
        n,
        "playback",
        min_value=0.0,
    )

    rsu_user_distance = _as_nonneg_float_matrix(
        action.get(
            "rsu_user_distance",
            _default_distance_matrix(
                m,
                n,
                cfg.rsu_channel.distance,
                cfg.rsu_channel.min_distance,
            ),
        ),
        (m, n),
        "rsu_user_distance",
        min_value=cfg.rsu_channel.min_distance,
    )
    uav_user_distance = _as_nonneg_float_matrix(
        action.get(
            "uav_user_distance",
            _default_distance_matrix(
                u,
                n,
                cfg.uav_channel.distance,
                cfg.uav_channel.min_distance,
            ),
        ),
        (u, n),
        "uav_user_distance",
        min_value=cfg.uav_channel.min_distance,
    )

    return ParsedAction(
        rsu_schedule=rsu_schedule,
        uav_hiring=uav_hiring,
        uav_schedule=uav_schedule,
        rsu_chunks=rsu_chunks,
        rsu_layers=rsu_layers,
        uav_chunks=uav_chunks,
        uav_layers=uav_layers,
        uav_power=uav_power,
        uav_charge=uav_charge,
        playback=playback,
        rsu_user_distance=rsu_user_distance,
        uav_user_distance=uav_user_distance,
    )