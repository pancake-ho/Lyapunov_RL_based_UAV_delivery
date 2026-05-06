from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class BoxSpace:
    """
    Gymnasium Box와 비슷한 최소 contains() helper.
    """
    low: float | int
    high: float | int
    shape: tuple[int, ...]
    dtype: Any = np.float32

    def contains(self, value: Any) -> bool:
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return False
        if arr.shape != self.shape:
            return False
        expected_dtype = np.dtype(self.dtype)
        if np.issubdtype(expected_dtype, np.integer):
            if not np.issubdtype(arr.dtype, np.integer):
                return False
        elif np.issubdtype(expected_dtype, np.floating):
            if not np.issubdtype(arr.dtype, np.number):
                return False
        if not np.all(np.isfinite(arr)):
            return False
        return bool(np.all(arr >= self.low) and np.all(arr <= self.high))


@dataclass(frozen=True)
class MultiBinarySpace:
    """
    Gymnasium MultiBinary와 비슷한 최소 contains() helper.
    """
    shape: tuple[int, ...]

    def contains(self, value: Any) -> bool:
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError):
            return False
        if arr.shape != self.shape:
            return False
        if not (
            np.issubdtype(arr.dtype, np.integer)
            or np.issubdtype(arr.dtype, np.bool_)
        ):
            return False
        return bool(np.all((arr == 0) | (arr == 1)))


@dataclass(frozen=True)
class DictSpace:
    """
    Dict 형태 observation/action 검증용 최소 helper.

    optional_spaces는 parser가 default를 제공하는 action key처럼 생략 가능한
    항목을 표현한다.
    """
    spaces: Mapping[str, Any]
    optional_spaces: Optional[Mapping[str, Any]] = None
    allow_extra: bool = False

    def contains(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False

        optional: Dict[str, Any] = dict(self.optional_spaces or {})

        for key, space in self.spaces.items():
            if key not in value:
                return False
            if not space.contains(value[key]):
                return False

        for key, space in optional.items():
            if key in value and not space.contains(value[key]):
                return False

        if self.allow_extra:
            return True

        allowed_keys = set(self.spaces.keys()) | set(optional.keys())
        return set(value.keys()).issubset(allowed_keys)
