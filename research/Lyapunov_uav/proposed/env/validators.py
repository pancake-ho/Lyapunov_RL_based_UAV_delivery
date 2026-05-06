from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from .action_types import EnvAction, SlowAction, FastAction
from config import EnvConfig


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
    return arr.astype(np.float32)


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
    return arr.astype(np.float32)


def _as_int_vector(
    value: np.ndarray,
    size: int,
    name: str,
    fill_value: int = -1,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
    return np.where(np.isfinite(arr), arr, fill_value).astype(np.int32)


def _default_distance_matrix(
    rows: int,
    cols: int,
    distance: float,
    min_distance: float,
) -> np.ndarray:
    return np.full((rows, cols), max(distance, min_distance), dtype=np.float32)


def parse_slow_action(action: EnvAction, cfg: EnvConfig) -> SlowAction:
    """
    round-level slow-timescale action을 파싱하는 함수
    """
    m = cfg.num_rsu
    n = cfg.num_user
    u = cfg.num_uav

    rsu_scheduling = _as_binary_matrix(
        action.get("rsu_scheduling", action.get("rsu_schedule", np.zeros((m, n), dtype=np.int32))),
        (m, n),
        "rsu_scheduling",
    )

    uav_hiring = _as_binary_vector(
        action.get("uav_hiring", np.zeros(u, dtype=np.int32)),
        u,
        "uav_hiring",
    )

    uav_scheduling = _as_binary_matrix(
        action.get("uav_schedule", action.get("uav_scheduling", np.zeros((u, n), dtype=np.int32))),
        (u, n),
        "uav_scheduling",
    )

    # 각 UAV가 고용되지 않으면 scheduling 제거
    uav_scheduling = uav_scheduling * uav_hiring[:, None]

    return SlowAction(
        rsu_scheduling=rsu_scheduling,
        uav_hiring=uav_hiring,
        uav_scheduling=uav_scheduling,
    )


def parse_fast_action(action: EnvAction, cfg: EnvConfig) -> FastAction:
    """
    slot-level fast-timescale action을 파싱하는 함수
    """
    m = cfg.num_rsu
    n = cfg.num_user
    u = cfg.num_uav

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
        min_value=0,
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
        min_value=0,
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

    residual_users = _as_binary_vector(
        action.get(
            "residual_users",
            np.ones(n, dtype=np.int32)
        ),
        n,
        "residual_users",
    )

    user_virtual_queue = _as_nonneg_float_vector(
        action.get(
            "user_virtual_queue",
            np.zeros(n, dtype=np.float32),
        ),
        n,
        "user_virtual_queue",
        min_value=0.0,
    )

    requested_content = _as_int_vector(
        action.get(
            "requested_content",
            -np.ones(n, dtype=np.int32),
        ),
        n,
        "requested_content",
    )

    uav_cached_content = _as_int_vector(
        action.get(
            "uav_cached_content",
            -np.ones(u, dtype=np.int32),
        ),
        u,
        "uav_cached_content",
    )

    # fast validator 수준에서 할 수 있는 정적 정리만 수행
    # chunks/layers/power 간 자기모순 제거
    rsu_active = ((rsu_chunks > 0) & (rsu_layers > 0)).astype(np.int32)
    rsu_chunks = rsu_chunks * rsu_active
    rsu_layers = rsu_layers * rsu_active

    uav_active = ((uav_chunks > 0) & (uav_layers > 0) & (uav_power > 0.0)).astype(np.int32)
    uav_chunks = uav_chunks * uav_active
    uav_layers = uav_layers * uav_active
    uav_power = uav_power * uav_active.astype(np.float32)

    # charging 중인 UAV는 해당 slot에서 service action 제거
    for uu in range(u):
        if uav_charge[uu] == 1:
            uav_chunks[uu, :] = 0
            uav_layers[uu, :] = 0
            uav_power[uu, :] = 0.0

    return FastAction(
        rsu_chunks=rsu_chunks,
        rsu_layers=rsu_layers,
        uav_chunks=uav_chunks,
        uav_layers=uav_layers,
        uav_power=uav_power,
        uav_charge=uav_charge,
        playback=playback,
        rsu_user_distance=rsu_user_distance,
        uav_user_distance=uav_user_distance,
        residual_users=residual_users,
        user_virtual_queue=user_virtual_queue,
        requested_content=requested_content,
        uav_cached_content=uav_cached_content,
    )
