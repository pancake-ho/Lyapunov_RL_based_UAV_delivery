from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray | list | tuple | float | int | bool


def _to_1d_float_array(x: Any) -> np.ndarray:
    """
    Scalar/list/np.ndarray type의 Input data를 1D float32 array type으로 변환.
    
    Dict type의 경우는 flatten_obs에서 recursive 처리.
    """
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).reshape(-1)

    if isinstance(x, (float, int, bool, np.number)):
        return np.asarray([x], dtype=np.float32)

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32).reshape(-1)

    raise TypeError(f"지원하지 않는 observation value type: {type(x)}")


def flatten_obs(
    obs: Dict[str, Any] | np.ndarray | Sequence[Any],
    sort_keys: bool = True,
    exclude_keys: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    Env observation을 1D float32 vector로 Flatten.

    현재 시나리오 기준:
        env reset/step이 dict obs를 반환하는 구조를 고려함.
    """
    exclude = set(exclude_keys or [])

    if isinstance(obs, dict):
        keys = sorted(obs.keys()) if sort_keys else list(obs.keys())
        arrays: List[np.ndarray] = []

        for key in keys:
            if key in exclude:
                continue

            value = obs[key]

            if isinstance(value, dict):
                arrays.append(flatten_obs(value, sort_keys=sort_keys, exclude_keys=exclude))
            else:
                arrays.append(_to_1d_float_array(value))

        if len(arrays) == 0:
            return np.zeros((0,), dtype=np.float32)

        return np.concatenate(arrays, axis=0).astype(np.float32)

    return _to_1d_float_array(obs)


def flatten_obs_with_keys(
    obs: Dict[str, Any],
    keys: Sequence[str],
    missing: str = "zeros",
) -> np.ndarray:
    """
    explicit key 순서를 사용하여 observation을 flatten.

    key 순서를 고정할 목적으로 사용하고, PPO input shape이 key 정렬 변화에 흔들리지 않게
    하려면, fast/slow 쪽에서 explicit key order를 사용하는 것이 가장 안전함.

    현재 시나리오 기준:
        missing:
            "zeros": 없는 key는 길이 1짜리 zero로 대체
            "skip": 없는 key 무시
            "raise": 없는 key이면 KeyError
    """
    if not isinstance(obs, dict):
        raise TypeError(f"flatten_obs_with_keys() 함수는 dict type-obs를 기대합니다. 현재 type: {type(obs)}")

    arrays: List[np.ndarray] = []

    for key in keys:
        if key not in obs:
            if missing == "zeros":
                arrays.append(np.zeros((1,), dtype=np.float32))
                continue
            if missing == "skip":
                continue
            if missing == "raise":
                raise KeyError(f"Observation key '{key}' 를 찾을 수 없습니다.")
            raise ValueError(f"지원하지 않는 missing mode: {missing}")

        value = obs[key]
        if isinstance(value, dict):
            arrays.append(flatten_obs(value, sort_keys=True))
        else:
            arrays.append(_to_1d_float_array(value))

    if len(arrays) == 0:
        return np.zeros((0,), dtype=np.float32)

    return np.concatenate(arrays, axis=0).astype(np.float32)


def infer_flat_dim(
    obs: Dict[str, Any] | np.ndarray | Sequence[Any],
    keys: Optional[Sequence[str]] = None,
) -> int:
    """
    flatten_obs_with_keys() 함수를 사용하는 경우, flatten shape 확인용
    """
    if keys is None:
        return int(flatten_obs(obs).shape[0])

    if not isinstance(obs, dict):
        raise TypeError("obs가 dict type인 경우에만 keys가 사용될 수 있습니다.")

    return int(flatten_obs_with_keys(obs, keys=keys).shape[0])


def split_env_reset(reset_result: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Gymnasium / Gym 스타일 reset() 함수 호환 처리용.

    Gymnasium 스타일:
        obs, info = env.reset()

    구 gym 스타일:
        obs = env.reset()
    """
    if isinstance(reset_result, tuple):
        if len(reset_result) != 2:
            raise ValueError(
                "env.reset() tuple 결과는 (obs, info) shape을 가져야 합니다."
                f"got length {len(reset_result)}"
            )
        obs, info = reset_result
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise TypeError(f"reset info는 dict type을 가져야 합니다. 현재 type: {type(info)}")
        return obs, info

    return reset_result, {}


def split_env_step(step_result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """
    Gymnasium / Gym 스타일 step() 함수 호환 처리용.

    Gymnasium 스타일:
        next_obs, reward, terminated, truncated, info = env.step(action)

    구 gym 스타일:
        next_obs, reward, done, info = env.step(action)

    반환은 항상:
        next_obs, reward, terminated, truncated, info
    """
    if not isinstance(step_result, tuple):
        raise TypeError(f"env.step()는 tuple type을 반환해야 합니다. 현재 type: {type(step_result)}")

    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise TypeError(f"step info는 dict type을 가져야 합니다. 현재 type: {type(info)}")

        return next_obs, float(reward), bool(terminated), bool(truncated), info

    if len(step_result) == 4:
        next_obs, reward, done, info = step_result
        if info is None:
            info = {}
        if not isinstance(info, dict):
            raise TypeError(f"step info는 dict type을 가져야 합니다. 현재 type: {type(info)}")

        return next_obs, float(reward), bool(done), False, info

    raise ValueError(
        "env.step() 결과는 length 4 또는 5입니다. "
        f"got length {len(step_result)}"
    )


def is_round_boundary(info: Dict[str, Any]) -> bool:
    """
    Env info에서 round boundary 여부를 읽음.

    현재 시나리오 기준:
        env.py에서 아래 key 중 하나를 넣어두면 자동 인식함.
            - is_round_boundary
            - round_boundary
            - boundary

    현재 fast PPO만 학습할 때는 boundary에서도 slot transition으로 처리 가능하지만,
    slow PPO/HRL 통합 단계에서는 이 함수로 round-level update 타이밍을 잡음.
    """
    if not isinstance(info, dict):
        return False

    for key in ("is_round_boundary", "round_boundary", "boundary"):
        if key in info:
            return bool(info[key])

    return False


def get_info_scalar(
    info: Dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    """
    info를 scalah(float) type으로 반환하는 함수
    """
    value = info.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_fixed_slow_decision_from_info(info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Fast PPO 단독 학습 시, info에 들어있는 slow decision을 추출할 때 사용함.

    현재 시나리오 기준:
        env 구현에 따라 key 이름이 다를 경우를 대비하여 후보 key를 넓게 설정함.

        반환 key:
            rsu_scheduling
            uav_hiring
            uav_scheduling

        없는 경우는 빈 array로 설정.
    """
    rsu_candidates = ("rsu_scheduling", "mu", "mu_mn", "rsu_schedule")
    hire_candidates = ("uav_hiring", "y", "y_u", "hire")
    uav_candidates = ("uav_scheduling", "phi", "phi_un", "uav_schedule")

    def _find(candidates: Sequence[str]) -> np.ndarray:
        for candidate in candidates:
            if candidate in info:
                return _to_1d_float_array(info[candidate])
        return np.zeros((0,), dtype=np.float32)

    return {
        "rsu_scheduling": _find(rsu_candidates),
        "uav_hiring": _find(hire_candidates),
        "uav_scheduling": _find(uav_candidates),
    }


def merge_obs_with_slow_decision(
    obs: Dict[str, Any],
    slow_decision: Dict[str, Any],
    key_name: str = "slow_decision",
) -> Dict[str, Any]:
    """
    fast obs에 round-level fixed slow decision을 condition으로 붙임.

    현재 시나리오 기준:
        Fast policy 기준:
            slow decision은 fast action이 아니라 condition으로 간주.
    """
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict to merge slow decision, got {type(obs)}")

    merged = dict(obs)
    merged[key_name] = slow_decision
    return merged


def remove_keys_from_obs(
    obs: Dict[str, Any],
    keys_to_remove: Iterable[str],
) -> Dict[str, Any]:
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict, got {type(obs)}")

    remove_set = set(keys_to_remove)
    return {k: v for k, v in obs.items() if k not in remove_set}