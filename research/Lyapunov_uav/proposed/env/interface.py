from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    from proposed.config import EnvConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import EnvConfig

from .action_types import EnvAction


@dataclass(frozen=True)
class VectorSpec:
    """
    PPO-compatible 1D vector contract.
    """
    dim: int
    slices: Dict[str, slice]


def _add_slice(slices: Dict[str, slice], name: str, start: int, size: int) -> int:
    slices[name] = slice(start, start + int(size))
    return start + int(size)


def slow_obs_spec(cfg: EnvConfig) -> VectorSpec:
    n = int(cfg.num_user)
    u = int(cfg.num_uav)
    slices: Dict[str, slice] = {}
    pos = 0
    pos = _add_slice(slices, "Q", pos, n)
    pos = _add_slice(slices, "Z", pos, n)
    pos = _add_slice(slices, "E", pos, u)
    pos = _add_slice(slices, "Y", pos, u)
    pos = _add_slice(slices, "requested_content", pos, n)
    pos = _add_slice(slices, "uav_cached_content", pos, u)
    pos = _add_slice(slices, "outage", pos, u)
    pos = _add_slice(slices, "round_idx", pos, 1)
    return VectorSpec(dim=pos, slices=slices)


def fast_obs_spec(cfg: EnvConfig) -> VectorSpec:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    slices: Dict[str, slice] = {}
    pos = 0
    pos = _add_slice(slices, "Q", pos, n)
    pos = _add_slice(slices, "Z", pos, n)
    pos = _add_slice(slices, "E", pos, u)
    pos = _add_slice(slices, "Y", pos, u)
    pos = _add_slice(slices, "uav_hiring", pos, u)
    pos = _add_slice(slices, "rsu_scheduling", pos, m * n)
    pos = _add_slice(slices, "uav_scheduling", pos, u * n)
    pos = _add_slice(slices, "requested_content", pos, n)
    pos = _add_slice(slices, "uav_cached_content", pos, u)
    pos = _add_slice(slices, "outage", pos, u)
    pos = _add_slice(slices, "charging_state", pos, u)
    pos = _add_slice(slices, "round_idx", pos, 1)
    pos = _add_slice(slices, "round_slot", pos, 1)
    pos = _add_slice(slices, "time", pos, 1)
    return VectorSpec(dim=pos, slices=slices)


def slow_action_spec(cfg: EnvConfig) -> VectorSpec:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    slices: Dict[str, slice] = {}
    pos = 0
    pos = _add_slice(slices, "rsu_scheduling", pos, m * n)
    pos = _add_slice(slices, "uav_hiring", pos, u)
    pos = _add_slice(slices, "uav_scheduling", pos, u * n)
    return VectorSpec(dim=pos, slices=slices)


def fast_action_spec(cfg: EnvConfig) -> VectorSpec:
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    slices: Dict[str, slice] = {}
    pos = 0
    pos = _add_slice(slices, "rsu_chunks", pos, m * n)
    pos = _add_slice(slices, "rsu_layers", pos, m * n)
    pos = _add_slice(slices, "uav_chunks", pos, u * n)
    pos = _add_slice(slices, "uav_layers", pos, u * n)
    pos = _add_slice(slices, "uav_power", pos, u * n)
    return VectorSpec(dim=pos, slices=slices)


def _flatten_key(obs: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"observation missing required key: {key}")
    return np.asarray(obs[key], dtype=np.float32).reshape(-1)


def flatten_slow_obs(obs: Dict[str, np.ndarray], cfg: EnvConfig) -> np.ndarray:
    spec = slow_obs_spec(cfg)
    out = np.zeros(spec.dim, dtype=np.float32)
    for key, slc in spec.slices.items():
        value = _flatten_key(obs, key)
        if value.size != slc.stop - slc.start:
            raise ValueError(f"{key} has flattened size {value.size}, expected {slc.stop - slc.start}")
        out[slc] = value
    return out


def flatten_fast_obs(obs: Dict[str, np.ndarray], cfg: EnvConfig) -> np.ndarray:
    spec = fast_obs_spec(cfg)
    out = np.zeros(spec.dim, dtype=np.float32)
    for key, slc in spec.slices.items():
        value = _flatten_key(obs, key)
        if value.size != slc.stop - slc.start:
            raise ValueError(f"{key} has flattened size {value.size}, expected {slc.stop - slc.start}")
        out[slc] = value
    return out


def _binary_from_tanh(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) > 0.0).astype(np.int32)


def _int_from_tanh(values: np.ndarray, max_value: int) -> np.ndarray:
    scaled = (np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0) + 1.0) * 0.5
    return np.rint(scaled * int(max_value)).astype(np.int32)


def _float_from_tanh(values: np.ndarray, max_value: float) -> np.ndarray:
    scaled = (np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0) + 1.0) * 0.5
    return (scaled * float(max_value)).astype(np.float32)


def decode_slow_action_vector(action: np.ndarray, cfg: EnvConfig) -> EnvAction:
    spec = slow_action_spec(cfg)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != spec.dim:
        raise ValueError(f"slow action vector dim mismatch: expected {spec.dim}, got {arr.size}")
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    return {
        "rsu_scheduling": _binary_from_tanh(arr[spec.slices["rsu_scheduling"]]).reshape(m, n),
        "uav_hiring": _binary_from_tanh(arr[spec.slices["uav_hiring"]]).reshape(u),
        "uav_scheduling": _binary_from_tanh(arr[spec.slices["uav_scheduling"]]).reshape(u, n),
    }


def decode_fast_action_vector(action: np.ndarray, cfg: EnvConfig) -> EnvAction:
    spec = fast_action_spec(cfg)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != spec.dim:
        raise ValueError(f"fast action vector dim mismatch: expected {spec.dim}, got {arr.size}")
    n = int(cfg.num_user)
    m = int(cfg.num_rsu)
    u = int(cfg.num_uav)
    return {
        "rsu_chunks": _int_from_tanh(arr[spec.slices["rsu_chunks"]], cfg.chunk).reshape(m, n),
        "rsu_layers": _int_from_tanh(arr[spec.slices["rsu_layers"]], cfg.layer).reshape(m, n),
        "uav_chunks": _int_from_tanh(arr[spec.slices["uav_chunks"]], cfg.chunk).reshape(u, n),
        "uav_layers": _int_from_tanh(arr[spec.slices["uav_layers"]], cfg.layer).reshape(u, n),
        "uav_power": _float_from_tanh(arr[spec.slices["uav_power"]], cfg.battery.max_tx_power).reshape(u, n),
    }
