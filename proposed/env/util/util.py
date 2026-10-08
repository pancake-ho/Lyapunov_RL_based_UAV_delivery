from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import DTypeLike, NDArray


def _safe_get_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    """
    여러 후보 attribute 이름을 순서대로 조회하고,
    처음 발견되는 값을 반환하는 함수
    """
    for name in names:
        try:
            return getattr(obj, name)
        except AttributeError:
            continue
    return default


def _ensure_shape(
    value: Any,
    shape: tuple[int, ...],
    dtype: DTypeLike,
    fill_value: float | int | bool = 0.0,
    strict: bool = False,
) -> NDArray[np.generic]:
    """
    입력값을 np.ndarray로 변환하고, 지정된 shape에 맞춰주는 함수
    """
    if not isinstance(shape, tuple):
        raise TypeError(f"'shape' mush be a tuple, but got {type(shape).__name__}.")
    
    if any((not isinstance(dim, int)) or dim < 0 for dim in shape):
        raise ValueError(f"'shape' must be a tuple of non-negative ints, but got {shape}.")

    if value is None:
        return np.full(shape, fill_value, dtype=dtype)
    
    try:
        arr = np.asarray(value, dtype=dtype)
    except Exception as e:
        raise ValueError(
            f"Failed to convert value of type {type(value).__name__} to ndarray with dtype {dtype}."
        ) from e

    if arr.shape == shape:
        return np.array(arr, dtype=dtype, copy=False)
    
    if strict:
        raise ValueError(f"Expected shape {shape}, but got {arr.shape}.")
    
    # scalah type이면 전체 shape으로 채움
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=dtype)
    
    try:
        return np.array(np.broadcast_to(arr, shape), dtype=dtype, copy=True)
    except Exception as e:
        raise ValueError(
            f"Expected shape {shape}, but got {arr.shape}: broadcasting failed"
        ) from e