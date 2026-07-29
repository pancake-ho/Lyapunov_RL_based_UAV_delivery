from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ArrayLike = np.ndarray | list | tuple | float | int | bool

# 현재 fast-timescale policy state 정의:
# s_L(t) = [Z(t), B(t), region(t), speed(t),
#           serving distance(t), connection(r)]
#
# Instantaneous fading/CSI는 입력하지 않는다.  Distance는 mobility state이며,
# round-fixed association의 slot별 capacity 변화를 학습하기 위해 필요하다.
FAST_OBS_KEYS: tuple[str, ...] = (
    "Z",
    "B",
    "user_region",
    "user_speed_kmh",
    "rsu_serving_distance",
    "uav_serving_distance",
    "rsu_connection",
    "uav_connection",
)

# 현재 slow-timescale policy state 정의:
# s_H(r) = [Z(r), B(r), user_region(r)]
SLOW_OBS_KEYS: tuple[str, ...] = (
    "Z",
    "B",
    "user_region",
)

def _to_1d_float_array(x: Any) -> np.ndarray:
    """
    Scalar/list/np.ndarray type의 Input data를 1D float32 array type으로 변환하는 함수.
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
    일반 dict observation flatten 수행 함수.
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
    missing: str = "raise",
) -> np.ndarray:
    """
    explicit key 순서를 사용하여 observation을 flatten하는 함수.

    현재 PPO에서는 key 순서가 바뀌면 network input 의미가 바뀌므로,
    반드시 explicit key order를 사용하는 것이 안전하다.
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

def flatten_fast_obs(
    obs: Dict[str, Any],
) -> np.ndarray:
    """
    Fast-timescale PPO observation flatten.

    확정 state:
        Z
        B
        user_region
        user_speed_kmh
        rsu_serving_distance
        uav_serving_distance
        rsu_connection
        uav_connection

    Instantaneous fading/CSI는 policy input에 넣지 않는다.
    Slow scheduling/hiring raw action도 별도로 중복 입력하지 않는다.
    """
    return flatten_obs_with_keys(
        obs=obs,
        keys=FAST_OBS_KEYS,
        missing="raise",
    )

def flatten_slow_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Slow-timescale PPO용 observation flatten.

    현재 확정 state:
        Z
        B
        user_region
    """
    return flatten_obs_with_keys(
        obs=obs,
        keys=SLOW_OBS_KEYS,
        missing="raise",
    )


def infer_flat_dim(
    obs: Dict[str, Any] | np.ndarray | Sequence[Any],
    keys: Optional[Sequence[str]] = None,
) -> int:
    """
    일반 flatten dimension 추론 함수.
    """
    if keys is None:
        return int(flatten_obs(obs).shape[0])

    if not isinstance(obs, dict):
        raise TypeError("obs가 dict type인 경우에만 keys를 사용할 수 있습니다.")

    return int(flatten_obs_with_keys(obs, keys=keys).shape[0])


def infer_fast_obs_dim(obs: Dict[str, Any]) -> int:
    """
    Fast-timescale PPO용 obs_dim 추론 함수.
    """
    return int(flatten_fast_obs(obs).shape[0])


def infer_slow_obs_dim(obs: Dict[str, Any]) -> int:
    """
    Slow-timescale PPO용 obs_dim 추론 함수.
    """
    return int(flatten_slow_obs(obs).shape[0])


def split_env_reset(reset_result: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Gymnasium / Gym 스타일 reset() 호환 처리.
    """
    if isinstance(reset_result, tuple):
        if len(reset_result) != 2:
            raise ValueError(
                "env.reset() tuple 결과는 (obs, info) shape이어야 합니다. "
                f"got length={len(reset_result)}"
            )

        obs, info = reset_result

        if info is None:
            info = {}

        if not isinstance(info, dict):
            raise TypeError(f"reset info는 dict type이어야 합니다. 현재 type={type(info)}")

        return obs, info

    return reset_result, {}


def split_env_step(step_result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """
    Gymnasium / Gym 스타일 step() 호환 처리.
    """
    if not isinstance(step_result, tuple):
        raise TypeError(f"env.step()은 tuple을 반환해야 합니다. 현재 type={type(step_result)}")

    if len(step_result) == 5:
        next_obs, reward, terminated, truncated, info = step_result

        if info is None:
            info = {}

        if not isinstance(info, dict):
            raise TypeError(f"step info는 dict type이어야 합니다. 현재 type={type(info)}")

        return next_obs, float(reward), bool(terminated), bool(truncated), info

    if len(step_result) == 4:
        next_obs, reward, done, info = step_result

        if info is None:
            info = {}

        if not isinstance(info, dict):
            raise TypeError(f"step info는 dict type이어야 합니다. 현재 type={type(info)}")

        return next_obs, float(reward), bool(done), False, info

    raise ValueError(
        "env.step() 결과는 length 4 또는 5여야 합니다. "
        f"got length={len(step_result)}"
    )


def is_round_boundary(info: Dict[str, Any]) -> bool:
    """
    Env info에서 round boundary 여부를 읽는다.
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
    info dict에서 scalar 값을 안전하게 읽는다.
    """
    value = info.get(key, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def remove_keys_from_obs(
    obs: Dict[str, Any],
    keys_to_remove: Iterable[str],
) -> Dict[str, Any]:
    """
    observation에서 특정 key들을 제거한다.
    """
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict, got {type(obs)}")

    remove_set = set(keys_to_remove)
    return {k: v for k, v in obs.items() if k not in remove_set}