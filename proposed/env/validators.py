from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .action_types import EnvAction, FastAction, SlowAction

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig


def _as_binary_matrix(
    value: np.ndarray,
    shape: Tuple[int, int],
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}.")
    return (arr > 0).astype(np.int32)


def _as_binary_vector(
    value: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},), got {arr.shape}."
        )
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
        raise ValueError(
            f"{name} must have shape ({size},), got {arr.shape}."
        )
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
        raise ValueError(
            f"{name} must have shape ({size},), got {arr.shape}."
        )
    return np.where(np.isfinite(arr), arr, fill_value).astype(np.int32)


def _default_distance_matrix(
    rows: int,
    cols: int,
    distance: float,
    min_distance: float,
) -> np.ndarray:
    return np.full(
        (rows, cols),
        max(distance, min_distance),
        dtype=np.float32,
    )


def parse_slow_action(action: EnvAction, cfg: EnvConfig) -> SlowAction:
    """Round-level slow action의 shape/dtype/binary gate를 파싱한다."""
    m = int(cfg.num_rsu)
    n = int(cfg.num_user)
    u = int(cfg.num_uav)

    rsu_scheduling = _as_binary_matrix(
        action.get(
            "rsu_scheduling",
            action.get(
                "rsu_schedule",
                np.zeros((m, n), dtype=np.int32),
            ),
        ),
        (m, n),
        "rsu_scheduling",
    )

    uav_hiring = _as_binary_vector(
        action.get("uav_hiring", np.zeros(u, dtype=np.int32)),
        u,
        "uav_hiring",
    )

    uav_scheduling = _as_binary_matrix(
        action.get(
            "uav_schedule",
            action.get(
                "uav_scheduling",
                np.zeros((u, n), dtype=np.int32),
            ),
        ),
        (u, n),
        "uav_scheduling",
    )

    # 최소 정적 gate. 의미 제약은 validate_slow_action_strict에서 검사한다.
    uav_scheduling = uav_scheduling * uav_hiring[:, None]

    return SlowAction(
        rsu_scheduling=rsu_scheduling,
        uav_hiring=uav_hiring,
        uav_scheduling=uav_scheduling,
    )


def validate_slow_action_strict(
    slow_action: SlowAction,
    cfg: EnvConfig,
    *,
    user_region: np.ndarray,
    requested_content: np.ndarray,
    uav_cached_content: np.ndarray,
    forbid_empty_hiring: bool = True,
) -> None:
    """
    Formulation v6의 slow feasible set을 실제 코드 action에 강제한다.

    검사 항목:
        1) coverage-region compatibility
        2) RSU/UAV capacity
        3) phi_un <= mu_u
        4) RSU 우선 residual-user 조건
        5) user별 단일 provider
        6) UAV cache compatibility
        7) empty hiring 제거(Scenario 7.1)
    """
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    n = int(cfg.num_user)

    if m != u:
        raise ValueError(
            "Current one-UAV-per-region mapping requires num_rsu == num_uav: "
            f"M={m}, U={u}."
        )

    y = np.asarray(slow_action.rsu_scheduling, dtype=np.int32)
    mu = np.asarray(slow_action.uav_hiring, dtype=np.int32)
    phi = np.asarray(slow_action.uav_scheduling, dtype=np.int32)

    if y.shape != (m, n):
        raise ValueError(
            f"rsu_scheduling shape mismatch: expected {(m, n)}, got {y.shape}."
        )
    if mu.shape != (u,):
        raise ValueError(
            f"uav_hiring shape mismatch: expected {(u,)}, got {mu.shape}."
        )
    if phi.shape != (u, n):
        raise ValueError(
            f"uav_scheduling shape mismatch: expected {(u, n)}, got {phi.shape}."
        )

    region = np.asarray(user_region, dtype=np.int32)
    requested = np.asarray(requested_content, dtype=np.int32)
    cached = np.asarray(uav_cached_content, dtype=np.int32)

    if region.shape != (n,):
        raise ValueError(
            f"user_region shape mismatch: expected {(n,)}, got {region.shape}."
        )
    if requested.shape != (n,):
        raise ValueError(
            "requested_content shape mismatch: "
            f"expected {(n,)}, got {requested.shape}."
        )
    if cached.shape != (u,):
        raise ValueError(
            "uav_cached_content shape mismatch: "
            f"expected {(u,)}, got {cached.shape}."
        )

    if np.any((region < 0) | (region >= m)):
        bad = np.flatnonzero((region < 0) | (region >= m))
        raise ValueError(
            f"Invalid user_region entries for users {bad.tolist()}."
        )

    rsu_region_mask = (
        np.arange(m, dtype=np.int32)[:, None] == region[None, :]
    )
    uav_region_mask = (
        np.arange(u, dtype=np.int32)[:, None] == region[None, :]
    )

    if np.any((y > 0) & (~rsu_region_mask)):
        bad = np.argwhere((y > 0) & (~rsu_region_mask))
        raise ValueError(
            "RSU coverage violation at links "
            f"{[tuple(map(int, x)) for x in bad.tolist()]}"
        )

    if np.any((phi > 0) & (~uav_region_mask)):
        bad = np.argwhere((phi > 0) & (~uav_region_mask))
        raise ValueError(
            "UAV coverage violation at links "
            f"{[tuple(map(int, x)) for x in bad.tolist()]}"
        )

    rsu_load = y.sum(axis=1)
    if np.any(rsu_load > int(cfg.rsu_capacity)):
        raise ValueError(
            "RSU capacity violation: "
            f"load={rsu_load.tolist()}, cap={int(cfg.rsu_capacity)}."
        )

    uav_load = phi.sum(axis=1)
    if np.any(uav_load > int(cfg.uav_user_cap)):
        raise ValueError(
            "UAV capacity violation: "
            f"load={uav_load.tolist()}, cap={int(cfg.uav_user_cap)}."
        )

    if np.any(phi > mu[:, None]):
        bad = np.argwhere(phi > mu[:, None])
        raise ValueError(
            "UAV employment gate phi_un(r) <= mu_u(r) violation at "
            f"{[tuple(map(int, x)) for x in bad.tolist()]}"
        )

    if forbid_empty_hiring:
        expected_mu = (uav_load > 0).astype(np.int32)
        if not np.array_equal(mu, expected_mu):
            raise ValueError(
                "Scenario 7.1 requires mu_u(r)=1 iff the UAV candidate set "
                "is nonempty: "
                f"mu={mu.tolist()}, expected={expected_mu.tolist()}."
            )

    rsu_provider_count = y.sum(axis=0)
    uav_provider_count = phi.sum(axis=0)

    if np.any(rsu_provider_count > 1):
        bad = np.flatnonzero(rsu_provider_count > 1)
        raise ValueError(
            f"Multiple RSU providers for users {bad.tolist()}."
        )
    if np.any(uav_provider_count > 1):
        bad = np.flatnonzero(uav_provider_count > 1)
        raise ValueError(
            f"Multiple UAV providers for users {bad.tolist()}."
        )

    provider_count = rsu_provider_count + uav_provider_count
    if np.any(provider_count > 1):
        bad = np.flatnonzero(provider_count > 1)
        raise ValueError(
            "Residual-user/single-provider constraint violated for users "
            f"{bad.tolist()}."
        )

    cache_match = cached[:, None] == requested[None, :]
    if np.any((phi > 0) & (~cache_match)):
        bad = np.argwhere((phi > 0) & (~cache_match))
        details = [
            (
                int(uu),
                int(nn),
                int(cached[uu]),
                int(requested[nn]),
            )
            for uu, nn in bad
        ]
        raise ValueError(
            "UAV cache mismatch (u, n, cached, requested): "
            f"{details}."
        )


def parse_fast_action(action: EnvAction, cfg: EnvConfig) -> FastAction:
    """Slot-level fast action을 파싱한다."""
    m = int(cfg.num_rsu)
    n = int(cfg.num_user)
    u = int(cfg.num_uav)

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
        action.get(
            "playback",
            np.full(n, cfg.playback_rate, dtype=np.float32),
        ),
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
        action.get("residual_users", np.ones(n, dtype=np.int32)),
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

    rsu_active = (
        (rsu_chunks > 0) & (rsu_layers > 0)
    ).astype(np.int32)
    rsu_chunks = rsu_chunks * rsu_active
    rsu_layers = rsu_layers * rsu_active

    uav_active = (
        (uav_chunks > 0)
        & (uav_layers > 0)
        & (uav_power > 0.0)
    ).astype(np.int32)
    uav_chunks = uav_chunks * uav_active
    uav_layers = uav_layers * uav_active
    uav_power = uav_power * uav_active.astype(np.float32)

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
