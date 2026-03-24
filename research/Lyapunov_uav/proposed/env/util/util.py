from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


def _safe_get_attr(obj: Any, names: list[str], default: Any) -> Any:
    """
    방지를 위한 안전 getattr 함수
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _ensure_shape(
    value: Any,
    shape: tuple[int, ...],
    dtype,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    입력값을 np.ndarray 타입으로 변환하고 shape을 맞춰주는 함수
    """
    if value is None:
        return np.full(shape, fill_value, dtype=dtype)
    
    arr = np.asanyarray(value, dtype=dtype)

    if arr.shape == shape:
        return arr
    
    # scalah type이면 전체 broadcast
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=dtype)
    
    try:
        return np.broadcast_to(arr, shape).astype(dtype, copy=False)
    except Exception as e:
        raise ValueError(
            f"Expected shape {shape}, but got {arr.shape}"
        ) from e