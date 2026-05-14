from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np

from config import EnvConfig, ChannelConfig, BatteryConfig
from env import Env


# =============================================================================
# Pretty logger
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BLUE = "\033[94m"


def supports_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


USE_COLOR = supports_color()


def c(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{RESET}"


def hr(char: str = "=", width: int = 110) -> None:
    print(char * width)


def title(text: str) -> None:
    hr("=")
    print(c(f"[ {text} ]", BOLD + CYAN))
    hr("=")


def section(text: str) -> None:
    print()
    hr("-")
    print(c(text, BOLD + BLUE))
    hr("-")


def ok(text: str) -> None:
    print(c(f"  [OK] {text}", GREEN))


def warn(text: str) -> None:
    print(c(f"  [WARN] {text}", YELLOW))


def fail(text: str) -> None:
    print(c(f"  [FAIL] {text}", RED))


def kv(key: str, value: Any, indent: int = 4) -> None:
    print(" " * indent + f"{key}: {value}")


def fmt_arr(arr: Any, precision: int = 4) -> str:
    np_arr = np.asarray(arr)
    return np.array2string(
        np_arr,
        precision=precision,
        suppress_small=True,
        separator=", ",
    )


# =============================================================================
# Test assertion helpers
# =============================================================================

class TestFailure(AssertionError):
    pass


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


def assert_true(condition: bool, message: str) -> None:
    if not bool(condition):
        raise TestFailure(message)


def assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise TestFailure(f"{name}: actual={actual}, expected={expected}")


def assert_finite_scalar(name: str, value: Any) -> None:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise TestFailure(f"{name}: scalar type expected, got {type(value)}")
    if not np.isfinite(float(value)):
        raise TestFailure(f"{name}: value is not finite: {value}")


def assert_array_shape(name: str, value: Any, expected_shape: Tuple[int, ...]) -> None:
    arr = np.asarray(value)
    if arr.shape != expected_shape:
        raise TestFailure(
            f"{name}: shape mismatch, actual={arr.shape}, expected={expected_shape}"
        )


def assert_array_finite(name: str, value: Any) -> None:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.number):
        if not np.all(np.isfinite(arr)):
            raise TestFailure(f"{name}: contains NaN or Inf.\nvalue={arr}")


def assert_array_equal(name: str, actual: Any, expected: Any) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)

    if actual_arr.shape != expected_arr.shape:
        raise TestFailure(
            f"{name}: shape mismatch, actual={actual_arr.shape}, expected={expected_arr.shape}"
        )

    if not np.array_equal(actual_arr, expected_arr):
        raise TestFailure(
            f"{name}: array mismatch\n"
            f"actual=\n{actual_arr}\n"
            f"expected=\n{expected_arr}"
        )


def assert_array_close(
    name: str,
    actual: Any,
    expected: Any,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)

    if actual_arr.shape != expected_arr.shape:
        raise TestFailure(
            f"{name}: shape mismatch, actual={actual_arr.shape}, expected={expected_arr.shape}"
        )

    if not np.allclose(actual_arr, expected_arr, atol=atol, rtol=rtol):
        raise TestFailure(
            f"{name}: array not close\n"
            f"actual=\n{actual_arr}\n"
            f"expected=\n{expected_arr}\n"
            f"diff=\n{actual_arr - expected_arr}"
        )


def assert_binary_array(name: str, value: Any) -> None:
    arr = np.asarray(value)
    unique = np.unique(arr)
    valid = np.all(np.isin(unique, [0, 1, False, True]))
    if not valid:
        raise TestFailure(f"{name}: expected binary array, unique={unique}")


def assert_raises(
    name: str,
    expected_exception: Type[BaseException],
    fn: Callable[[], Any],
) -> None:
    try:
        fn()
    except expected_exception:
        ok(f"{name}: expected {expected_exception.__name__} raised")
        return
    except Exception as exc:
        raise TestFailure(
            f"{name}: expected {expected_exception.__name__}, "
            f"but got {type(exc).__name__}: {exc}"
        ) from exc

    raise TestFailure(f"{name}: expected {expected_exception.__name__}, but no exception raised")


# =============================================================================
# Config and action builders
# =============================================================================

def build_test_config() -> EnvConfig:
    """
    env.py runtime 검증용 compact config.

    목적:
        - num_rsu == num_uav 제약 만족
        - slow_T를 작게 두어 round boundary를 빠르게 검증
        - base_chunk_size_bits를 작게 두어 delivery path가 열리게 함
        - energy 소모량을 작게 두어 여러 step 동안 battery가 안정적으로 변하게 함
    """
    return EnvConfig(
        num_user=3,
        num_rsu=3,
        num_uav=3,
        slow_T=3,
        num_video=3,
        layer=3,
        chunk=3,
        rsu_capacity=2,
        uav_user_cap=2,
        init_queue=5.0,
        playback_rate=1.0,
        max_queue=20.0,
        base_chunk_size_bits=100.0,
        quality_weights=(1.0, 2.0, 3.0),
        uav_hiring_cost=1.0,
        seed=2026,
        rsu_channel=ChannelConfig(
            distance=5.0,
            min_distance=1.0,
            bandwidth=1e6,
            gamma_db=30.0,
            inr_db=0.0,
            sigma_db=0.0,
            mu_db=0.0,
            beta=2.0,
            seed=2026,
        ),
        uav_channel=ChannelConfig(
            distance=5.0,
            min_distance=1.0,
            bandwidth=1e6,
            altitude=10.0,
            beta_zero=100.0,
            noise_power=1.0,
            capacity_gap=1.0,
            seed=2026,
        ),
        battery=BatteryConfig(
            e_max=100,
            e_init=100,
            e_min=10.0,
            p_0=1.0,
            p_i=1.0,
            tx_energy_coeff=1.0,
            charging_rate=10.0,
            eta_c=1.0,
            enable_charging=True,
            allow_charge=True,
            slot_duration=1.0,
            target_service_slots_per_round=3,
            battery_capacity_energy=3000.0,
            max_tx_power=10.0,
        ),
    )


def make_empty_slow_action(cfg: EnvConfig) -> Dict[str, np.ndarray]:
    return {
        "rsu_scheduling": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "uav_hiring": np.zeros(cfg.num_uav, dtype=np.int32),
        "uav_scheduling": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
    }


def make_slow_action(
    cfg: EnvConfig,
    rsu_idx: int | None = None,
    uav_idx: int | None = None,
    user_idx: int | None = None,
    enable_rsu: bool = True,
    enable_uav: bool = True,
) -> Dict[str, np.ndarray]:
    action = make_empty_slow_action(cfg)

    if user_idx is None:
        return action

    if enable_rsu and rsu_idx is not None:
        action["rsu_scheduling"][rsu_idx, user_idx] = 1

    if enable_uav and uav_idx is not None:
        action["uav_hiring"][uav_idx] = 1
        action["uav_scheduling"][uav_idx, user_idx] = 1

    return action


def make_empty_fast_action(cfg: EnvConfig) -> Dict[str, np.ndarray]:
    return {
        "rsu_chunks": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "rsu_layers": np.zeros((cfg.num_rsu, cfg.num_user), dtype=np.int32),
        "uav_chunks": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
        "uav_layers": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.int32),
        "uav_power": np.zeros((cfg.num_uav, cfg.num_user), dtype=np.float32),
        "rsu_user_distance": np.full(
            (cfg.num_rsu, cfg.num_user),
            5.0,
            dtype=np.float32,
        ),
        "uav_user_distance": np.full(
            (cfg.num_uav, cfg.num_user),
            5.0,
            dtype=np.float32,
        ),
    }


def make_fast_action(
    cfg: EnvConfig,
    rsu_idx: int = 0,
    uav_idx: int = 0,
    user_idx: int = 0,
    chunks: int = 1,
    layer: int = 1,
    power: float = 10.0,
    enable_rsu: bool = True,
    enable_uav: bool = True,
) -> Dict[str, np.ndarray]:
    action = make_empty_fast_action(cfg)

    if enable_rsu:
        action["rsu_chunks"][rsu_idx, user_idx] = int(chunks)
        action["rsu_layers"][rsu_idx, user_idx] = int(layer)

    if enable_uav:
        action["uav_chunks"][uav_idx, user_idx] = int(chunks)
        action["uav_layers"][uav_idx, user_idx] = int(layer)
        action["uav_power"][uav_idx, user_idx] = float(power)

    return action


# =============================================================================
# Environment invariant checks
# =============================================================================

def expected_fast_obs_shapes(cfg: EnvConfig) -> Dict[str, Tuple[int, ...]]:
    return {
        "Z": (cfg.num_user,),
        "Y": (cfg.num_uav,),
        "uav_scheduling": (cfg.num_uav, cfg.num_user),
        "rsu_user_distance": (cfg.num_rsu, cfg.num_user),
        "uav_user_distance": (cfg.num_uav, cfg.num_user),
    }


def expected_slow_obs_shapes(cfg: EnvConfig) -> Dict[str, Tuple[int, ...]]:
    return {
        "Z": (cfg.num_user,),
        "Y": (cfg.num_uav,),
        "rsu_user_distance": (cfg.num_rsu, cfg.num_user),
        "uav_user_distance": (cfg.num_uav, cfg.num_user),
    }


def assert_obs_dict(
    obs: Dict[str, np.ndarray],
    expected_shapes: Dict[str, Tuple[int, ...]],
    name: str,
) -> None:
    if not isinstance(obs, dict):
        raise TestFailure(f"{name}: obs must be dict, got {type(obs)}")

    for key, shape in expected_shapes.items():
        if key not in obs:
            raise TestFailure(f"{name}: obs missing required key: {key}")
        assert_array_shape(f"{name}.obs[{key}]", obs[key], shape)
        assert_array_finite(f"{name}.obs[{key}]", obs[key])


def assert_fast_obs_dict(obs: Dict[str, np.ndarray], cfg: EnvConfig, name: str) -> None:
    assert_obs_dict(obs, expected_fast_obs_shapes(cfg), name)
    assert_binary_array(f"{name}.obs[uav_scheduling]", obs["uav_scheduling"])


def assert_slow_obs_dict(obs: Dict[str, np.ndarray], cfg: EnvConfig, name: str) -> None:
    """
    slow-timescale obs 검증.

    현재 formulation 기준:
        s_H(r) = [Z(r), B(r), x(r)]

    현재 코드에서는 B(r)를 아직 Y key로 반환하고 있으므로,
    slow obs key는 ["Z", "Y", "rsu_user_distance", "uav_user_distance"] 기준으로 검사한다.
    """
    assert_obs_dict(obs, expected_slow_obs_shapes(cfg), name)


def assert_env_state(env: Env, cfg: EnvConfig, name: str) -> None:
    assert_array_shape(f"{name}.queue", env.queue, (cfg.num_user,))
    assert_array_shape(f"{name}.Z", env.Z, (cfg.num_user,))
    assert_array_shape(f"{name}.E", env.E, (cfg.num_uav,))
    assert_array_shape(f"{name}.Y", env.Y, (cfg.num_uav,))
    assert_array_shape(f"{name}.rsu_scheduling", env.rsu_scheduling, (cfg.num_rsu, cfg.num_user))
    assert_array_shape(f"{name}.uav_hiring", env.uav_hiring, (cfg.num_uav,))
    assert_array_shape(f"{name}.uav_scheduling", env.uav_scheduling, (cfg.num_uav, cfg.num_user))
    assert_array_shape(f"{name}.outage", env.outage, (cfg.num_uav,))
    assert_array_shape(f"{name}.charging_state", env.charging_state, (cfg.num_uav,))

    for key, value in {
        "queue": env.queue,
        "Z": env.Z,
        "E": env.E,
        "Y": env.Y,
        "rsu_scheduling": env.rsu_scheduling,
        "uav_hiring": env.uav_hiring,
        "uav_scheduling": env.uav_scheduling,
        "outage": env.outage,
        "charging_state": env.charging_state,
    }.items():
        assert_array_finite(f"{name}.{key}", value)

    assert_binary_array(f"{name}.rsu_scheduling", env.rsu_scheduling)
    assert_binary_array(f"{name}.uav_hiring", env.uav_hiring)
    assert_binary_array(f"{name}.uav_scheduling", env.uav_scheduling)
    assert_binary_array(f"{name}.outage", env.outage)
    assert_binary_array(f"{name}.charging_state", env.charging_state)

    assert_true(np.all(env.queue >= -1e-6), f"{name}: queue has negative values")
    assert_true(np.all(env.E >= -1e-6), f"{name}: E has negative values")
    assert_true(np.all(env.E <= cfg.battery.e_max + 1e-6), f"{name}: E exceeds e_max")
    assert_true(np.all(env.Y >= -1e-6), f"{name}: Y has negative values")
    assert_true(np.all(env.Y <= cfg.battery.e_max + 1e-6), f"{name}: Y exceeds e_max")

    expected_z = np.clip(float(cfg.max_queue) - env.queue, 0.0, float(cfg.max_queue))
    assert_array_close(f"{name}: Z = max_queue - Q", env.Z, expected_z)

    expected_y = np.clip(float(cfg.battery.e_max) - env.E, 0.0, float(cfg.battery.e_max))
    assert_array_close(f"{name}: Y = e_max - E", env.Y, expected_y, atol=1e-4)


def assert_step_result(
    env: Env,
    next_obs: Dict[str, np.ndarray],
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    cfg: EnvConfig,
    step_name: str,
) -> None:
    assert_fast_obs_dict(next_obs, cfg, name=f"{step_name}.next_obs")
    assert_env_state(env, cfg, name=f"{step_name}.env_state")

    assert_finite_scalar(f"{step_name}.reward", reward)

    if not isinstance(terminated, bool):
        raise TestFailure(f"{step_name}: terminated must be bool, got {type(terminated)}")
    if not isinstance(truncated, bool):
        raise TestFailure(f"{step_name}: truncated must be bool, got {type(truncated)}")
    if not isinstance(info, dict):
        raise TestFailure(f"{step_name}: info must be dict, got {type(info)}")

    required_info_keys = [
        "prev_time",
        "next_time",
        "prev_round_slot",
        "next_round_slot",
        "active_round_idx",
        "next_round_idx",
        "is_round_boundary",
        "terminated",
        "truncated",
        "prev_Q",
        "next_Q",
        "prev_Z",
        "next_Z",
        "playback",
        "consumed",
        "stall",
        "prev_E",
        "next_E",
        "prev_Y",
        "next_Y",
        "battery_step_info",
        "delivered_rsu_per_user",
        "delivered_uav_per_user",
        "delivered_total_per_user",
        "quality_rsu_per_user",
        "quality_uav_per_user",
        "quality_total_per_user",
        "reward_components",
        "rsu_result",
        "uav_result",
    ]

    for key in required_info_keys:
        if key not in info:
            raise TestFailure(f"{step_name}: info missing key: {key}")

    assert_equal(f"{step_name}: info.next_time == env.t", int(info["next_time"]), int(env.t))
    assert_equal(
        f"{step_name}: info.next_round_idx == env.round_idx",
        int(info["next_round_idx"]),
        int(env.round_idx),
    )
    assert_equal(
        f"{step_name}: info.next_round_slot == env.round_slot",
        int(info["next_round_slot"]),
        int(env.round_slot),
    )

    vector_shapes = {
        "prev_Q": (cfg.num_user,),
        "next_Q": (cfg.num_user,),
        "prev_Z": (cfg.num_user,),
        "next_Z": (cfg.num_user,),
        "playback": (cfg.num_user,),
        "consumed": (cfg.num_user,),
        "stall": (cfg.num_user,),
        "prev_E": (cfg.num_uav,),
        "next_E": (cfg.num_uav,),
        "prev_Y": (cfg.num_uav,),
        "next_Y": (cfg.num_uav,),
        "delivered_rsu_per_user": (cfg.num_user,),
        "delivered_uav_per_user": (cfg.num_user,),
        "delivered_total_per_user": (cfg.num_user,),
        "quality_rsu_per_user": (cfg.num_user,),
        "quality_uav_per_user": (cfg.num_user,),
        "quality_total_per_user": (cfg.num_user,),
    }

    for key, shape in vector_shapes.items():
        assert_array_shape(f"{step_name}.info[{key}]", info[key], shape)
        assert_array_finite(f"{step_name}.info[{key}]", info[key])

    battery_info = info["battery_step_info"]
    if not isinstance(battery_info, list):
        raise TestFailure(f"{step_name}: battery_step_info must be list")
    if len(battery_info) != cfg.num_uav:
        raise TestFailure(
            f"{step_name}: battery_step_info length mismatch, "
            f"actual={len(battery_info)}, expected={cfg.num_uav}"
        )

    for u, binfo in enumerate(battery_info):
        if not isinstance(binfo, dict):
            raise TestFailure(f"{step_name}: battery_step_info[{u}] must be dict")
        for key in [
            "hover_energy",
            "comm_energy",
            "total_consumed",
            "charged_energy",
            "consumed_soc",
            "charged_soc",
            "soc_before",
            "soc_after",
            "virtual_before",
            "virtual_after",
            "outage",
        ]:
            if key not in binfo:
                raise TestFailure(f"{step_name}: battery_step_info[{u}] missing key: {key}")

    rc = info["reward_components"]
    if not isinstance(rc, dict):
        raise TestFailure(f"{step_name}: reward_components must be dict")

    for key in ["fast_reward", "slow_reward", "fast_reward_components", "slow_reward_components"]:
        if key not in rc:
            raise TestFailure(f"{step_name}: reward_components missing key: {key}")

    assert_finite_scalar(f"{step_name}.reward_components.fast_reward", rc["fast_reward"])
    assert_finite_scalar(f"{step_name}.reward_components.slow_reward", rc["slow_reward"])

    # Delivery decomposition check
    delivered_rsu = np.asarray(info["delivered_rsu_per_user"], dtype=np.float32)
    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)
    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_array_close(
        f"{step_name}: delivered_total = delivered_rsu + delivered_uav",
        delivered_total,
        delivered_rsu + delivered_uav,
    )

    # Quality decomposition check
    quality_rsu = np.asarray(info["quality_rsu_per_user"], dtype=np.float32)
    quality_uav = np.asarray(info["quality_uav_per_user"], dtype=np.float32)
    quality_total = np.asarray(info["quality_total_per_user"], dtype=np.float32)
    assert_array_close(
        f"{step_name}: quality_total = quality_rsu + quality_uav",
        quality_total,
        quality_rsu + quality_uav,
    )

    # Queue update check
    prev_q = np.asarray(info["prev_Q"], dtype=np.float32)
    next_q = np.asarray(info["next_Q"], dtype=np.float32)
    playback = np.asarray(info["playback"], dtype=np.float32)
    expected_next_q = np.maximum(prev_q - playback, 0.0) + delivered_total
    assert_array_close(
        f"{step_name}: Q(t+1) = max(Q(t)-b,0)+d(t)",
        next_q,
        expected_next_q,
        atol=1e-4,
    )
    assert_array_close(f"{step_name}: env.queue == info.next_Q", env.queue, next_q)

    # Virtual queue check
    expected_next_z = np.clip(float(cfg.max_queue) - next_q, 0.0, float(cfg.max_queue))
    assert_array_close(
        f"{step_name}: Z(t+1) = max_queue - Q(t+1)",
        info["next_Z"],
        expected_next_z,
        atol=1e-4,
    )

    # Battery E/Y check
    assert_array_close(f"{step_name}: env.E == info.next_E", env.E, info["next_E"], atol=1e-4)
    assert_array_close(f"{step_name}: env.Y == info.next_Y", env.Y, info["next_Y"], atol=1e-4)

    # Reward consistency check
    assert_reward_consistency(info=info, reward=reward, cfg=cfg, step_name=step_name)


def assert_reward_consistency(
    info: Dict[str, Any],
    reward: float,
    cfg: EnvConfig,
    step_name: str,
) -> None:
    rc = info["reward_components"]
    frc = rc["fast_reward_components"]
    src = rc["slow_reward_components"]

    required_fast_keys = [
        "theta_z",
        "prev_Z",
        "delivered_total_per_user",
        "prev_Y",
        "consumed_soc_per_uav",
        "charged_soc_per_uav",
        "quality_total_per_user",
        "video_delivery_term",
        "battery_consume_term",
        "battery_charge_term",
        "quality_term",
        "fast_reward",
    ]

    for key in required_fast_keys:
        if key not in frc:
            raise TestFailure(f"{step_name}: fast_reward_components missing key: {key}")

    theta_z = np.asarray(frc["theta_z"], dtype=np.float32)
    next_z = np.asarray(frc["next_Z"], dtype=np.float32)
    delivered = np.asarray(frc["delivered_total_per_user"], dtype=np.float32)
    next_y = np.asarray(frc["next_Y"], dtype=np.float32)
    consumed_soc = np.asarray(frc["consumed_soc_per_uav"], dtype=np.float32)
    charged_soc = np.asarray(frc["charged_soc_per_uav"], dtype=np.float32)
    quality = np.asarray(frc["quality_total_per_user"], dtype=np.float32)

    V = float(getattr(cfg.reward, "V", 1.0))

    expected_video_term = float(np.sum((next_z - theta_z) * delivered))
    expected_battery_consume_term = -float(np.sum(next_y * consumed_soc))
    expected_battery_charge_term = float(np.sum(next_y * charged_soc))
    expected_quality_term = V * float(np.sum(quality))

    assert_array_close(
        f"{step_name}: video_delivery_term",
        np.array([frc["video_delivery_term"]]),
        np.array([expected_video_term]),
        atol=1e-4,
    )
    assert_array_close(
        f"{step_name}: battery_consume_term",
        np.array([frc["battery_consume_term"]]),
        np.array([expected_battery_consume_term]),
        atol=1e-4,
    )
    assert_array_close(
        f"{step_name}: battery_charge_term",
        np.array([frc["battery_charge_term"]]),
        np.array([expected_battery_charge_term]),
        atol=1e-4,
    )
    assert_array_close(
        f"{step_name}: quality_term",
        np.array([frc["quality_term"]]),
        np.array([expected_quality_term]),
        atol=1e-4,
    )

    expected_fast_reward = (
        expected_video_term
        + expected_battery_consume_term
        + expected_battery_charge_term
        + expected_quality_term
    )

    assert_array_close(
        f"{step_name}: fast_reward = sum(fast terms)",
        np.array([frc["fast_reward"]]),
        np.array([expected_fast_reward]),
        atol=1e-4,
    )

    assert_array_close(
        f"{step_name}: returned reward == fast_reward",
        np.array([reward]),
        np.array([rc["fast_reward"]]),
        atol=1e-4,
    )

    if not bool(info["is_round_boundary"]):
        assert_array_close(
            f"{step_name}: slow_reward must be zero before boundary",
            np.array([rc["slow_reward"]]),
            np.array([0.0]),
            atol=1e-6,
        )
        if "slow_reward" in src:
            assert_array_close(
                f"{step_name}: slow_components.slow_reward must be zero before boundary",
                np.array([src["slow_reward"]]),
                np.array([0.0]),
                atol=1e-6,
            )
    else:
        if "round_fast_reward_sum" not in src:
            raise TestFailure(f"{step_name}: boundary slow components missing round_fast_reward_sum")
        if "hire_cost" not in src:
            raise TestFailure(f"{step_name}: boundary slow components missing hire_cost")

        expected_slow_reward = float(src["round_fast_reward_sum"]) + float(src["hire_cost"])
        assert_array_close(
            f"{step_name}: slow_reward = round_fast_reward_sum + hire_cost",
            np.array([rc["slow_reward"]]),
            np.array([expected_slow_reward]),
            atol=1e-4,
        )


def print_step_summary(env: Env, reward: float, info: Dict[str, Any], label: str) -> None:
    print(c(label, BOLD))
    kv("time_slot", f"{info.get('prev_time')} -> {info.get('next_time')}")
    kv("round_idx", f"{info.get('active_round_idx')} -> {info.get('next_round_idx')}")
    kv("round_slot", f"{info.get('prev_round_slot')} -> {info.get('next_round_slot')}")
    kv("is_round_boundary", info.get("is_round_boundary"))
    kv("reward", f"{float(reward):.6f}")
    kv("next_Q", fmt_arr(env.queue))
    kv("next_Z", fmt_arr(env.Z))
    kv("next_E", fmt_arr(env.E))
    kv("next_B", fmt_arr(env.Y))
    kv("rsu_chunk", fmt_arr(info.get("delivered_rsu_per_user")))
    kv("uav_chunk", fmt_arr(info.get("delivered_uav_per_user")))
    kv("chunk_total", fmt_arr(info.get("delivered_total_per_user")))
    kv("quality_total", fmt_arr(info.get("quality_total_per_user")))
    kv("charging", fmt_arr(env.charging_state))

    rc = info.get("reward_components", {})
    if isinstance(rc, dict):
        frc = rc.get("fast_reward_components", {})
        src = rc.get("slow_reward_components", {})
        if isinstance(frc, dict):
            kv(
                "fast_terms",
                {
                    "vid_term": round(float(frc.get("video_delivery_term", 0.0)), 2),
                    "battery_consume": round(float(frc.get("battery_consume_term", 0.0)), 2),
                    "battery_charge": round(float(frc.get("battery_charge_term", 0.0)), 2),
                    "quality": round(float(frc.get("quality_term", 0.0)), 2),
                    "slot_fast_reward": round(float(frc.get("fast_reward", 0.0)), 2),
                },
            )
        if isinstance(src, dict):
            kv(
                "slow_terms",
                {
                    "is_round_boundary": src.get("is_round_boundary"),
                    "slow_reward": round(float(src.get("slow_reward", 0.0)), 2),
                    "hire_cost": round(float(src.get("hire_cost", 0.0)), 2),
                },
            )
    print("\n")


# =============================================================================
# Scenario setup helpers
# =============================================================================

def prepare_content_match(env: Env, user_idx: int, uav_idx: int, content_id: int) -> None:
    env.requested_content[user_idx] = int(content_id)
    env.uav_cached_content[uav_idx] = int(content_id)


def prepare_content_mismatch(env: Env, user_idx: int, uav_idx: int) -> None:
    env.requested_content[user_idx] = 0
    env.uav_cached_content[uav_idx] = 1


def set_uav_soc(env: Env, cfg: EnvConfig, uav_idx: int, soc: float) -> None:
    soc_clipped = float(np.clip(float(soc), 0.0, float(cfg.battery.e_max)))
    env.batteries[uav_idx].soc = soc_clipped
    env.batteries[uav_idx].virtual_q = float(cfg.battery.e_max) - soc_clipped


def assert_slow_decision_equal(
    env: Env,
    slow: Dict[str, np.ndarray],
    name: str,
) -> None:
    assert_array_equal(f"{name}.rsu_scheduling", env.rsu_scheduling, slow["rsu_scheduling"])
    assert_array_equal(f"{name}.uav_hiring", env.uav_hiring, slow["uav_hiring"])
    assert_array_equal(f"{name}.uav_scheduling", env.uav_scheduling, slow["uav_scheduling"])


# =============================================================================
# Individual tests
# =============================================================================

def test_01_reset_and_initial_obs() -> None:
    section("TEST 01 | reset 및 초기 fast/slow observation/state 검증")

    cfg = build_test_config()
    env = Env(cfg)

    fast_obs, reset_info = env.reset()
    slow_obs = env.get_slow_obs()

    kv("reset_info", reset_info)
    kv("fast_obs_keys", list(fast_obs.keys()))
    kv("slow_obs_keys", list(slow_obs.keys()))

    assert_fast_obs_dict(fast_obs, cfg, "reset_fast")
    assert_slow_obs_dict(slow_obs, cfg, "reset_slow")
    assert_env_state(env, cfg, "after_reset")

    assert_equal("reset t", env.t, 0)
    assert_equal("reset round_idx", env.round_idx, 0)
    assert_equal("reset round_slot", env.round_slot, 0)
    assert_array_close("round_start_E == E after reset", env.round_start_E, env.E)

    assert_array_close("slow_obs.Z == env.Z", slow_obs["Z"], env.Z)
    assert_array_close("slow_obs.Y == env.Y", slow_obs["Y"], env.Y)
    assert_array_close(
        "slow_obs.rsu_user_distance == env.rsu_user_distance",
        slow_obs["rsu_user_distance"],
        env.rsu_user_distance,
    )
    assert_array_close(
        "slow_obs.uav_user_distance == env.uav_user_distance",
        slow_obs["uav_user_distance"],
        env.uav_user_distance,
    )

    ok("reset 및 fast/slow observation/state 정상")


def test_02_no_slow_action_blocks_delivery() -> None:
    section("TEST 02 | slow action 미적용 상태에서 fast action delivery masking 검증")

    cfg = build_test_config()
    env = Env(cfg)
    fast_obs, _ = env.reset()
    assert_fast_obs_dict(fast_obs, cfg, "reset")

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    fast = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    next_obs, reward, terminated, truncated, info = env.step(fast)

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "no_slow_step")
    print_step_summary(env, reward, info, "[no slow action step]")

    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_array_close(
        "no slow action: delivered_total must be zero",
        delivered_total,
        np.zeros(cfg.num_user, dtype=np.float32),
    )

    ok("slow decision이 없으면 fast delivery action이 masking됨")


def test_03_basic_service_flow_and_queue_battery() -> None:
    section("TEST 03 | 기본 RSU/UAV service, queue update, battery update 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    env.apply_slow_action(slow)
    assert_slow_decision_equal(env, slow, "slow_applied")

    fast = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    prev_e = env.E.copy()

    next_obs, reward, terminated, truncated, info = env.step(fast)

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "basic_service_step")
    print_step_summary(env, reward, info, "[basic service step]")

    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_true(float(delivered_total[0]) > 0.0, "user 0 should receive positive delivery")

    assert_true(
        float(env.E[0]) < float(prev_e[0]),
        f"hired serving UAV 0 should consume battery: prev={prev_e[0]}, now={env.E[0]}",
    )

    assert_true(not terminated, "basic service step should not terminate")
    assert_true(not truncated, "basic service step should not truncate")

    ok("기본 service/queue/battery 흐름 정상")


def test_04_content_mismatch_blocks_uav_delivery() -> None:
    section("TEST 04 | UAV cache content mismatch 시 UAV delivery 차단 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_mismatch(env, user_idx=0, uav_idx=0)

    slow = make_slow_action(
        cfg,
        rsu_idx=None,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )
    env.apply_slow_action(slow)

    fast = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )

    next_obs, reward, terminated, truncated, info = env.step(fast)

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "content_mismatch_step")
    print_step_summary(env, reward, info, "[content mismatch step]")

    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)
    assert_array_close(
        "content mismatch: UAV delivered_uav must be zero",
        delivered_uav,
        np.zeros(cfg.num_user, dtype=np.float32),
        atol=1e-6,
    )

    ok("UAV cached content와 user requested content가 다르면 UAV delivery가 차단됨")


def test_05_charging_blocks_service_and_updates_battery() -> None:
    section("TEST 05 | low SoC에서 rule-based charging, service 차단, battery update 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow = make_slow_action(
        cfg,
        rsu_idx=None,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )
    env.apply_slow_action(slow)

    low_soc = float(cfg.battery.e_min)
    set_uav_soc(env, cfg, uav_idx=0, soc=low_soc)

    prev_e = env.E.copy()
    prev_y = env.Y.copy()

    fast = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )

    next_obs, reward, terminated, truncated, info = env.step(fast)

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "charging_step")
    print_step_summary(env, reward, info, "[charging step]")

    assert_equal("charging_state[0]", int(env.charging_state[0]), 1)

    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)
    assert_array_close(
        "charging: UAV delivery must be blocked",
        delivered_uav,
        np.zeros(cfg.num_user, dtype=np.float32),
        atol=1e-6,
    )

    binfo0 = info["battery_step_info"][0]
    assert_true(
        float(binfo0["charged_soc"]) > 0.0,
        f"charging UAV should have positive charged_soc, got {binfo0['charged_soc']}",
    )
    assert_true(
        float(env.E[0]) >= float(prev_e[0]) - 1e-6,
        f"charging should not decrease UAV 0 SoC: prev={prev_e[0]}, now={env.E[0]}",
    )
    assert_true(
        float(env.Y[0]) <= float(prev_y[0]) + 1e-6,
        f"charging should not increase UAV 0 virtual queue: prev={prev_y[0]}, now={env.Y[0]}",
    )

    ok("low SoC에서 charging이 service를 차단하고 battery를 갱신함")


def test_06_round_transition_and_accumulator() -> None:
    section("TEST 06 | round boundary, round_idx/round_slot, accumulator reset 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow0 = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    fast0 = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)

    env.apply_slow_action(slow0)

    assert_equal("after slow0 t", env.t, 0)
    assert_equal("after slow0 round_idx", env.round_idx, 0)
    assert_equal("after slow0 round_slot", env.round_slot, 0)
    assert_array_close("after slow0 round_start_E == E", env.round_start_E, env.E)

    for local_step in range(1, cfg.slow_T + 1):
        next_obs, reward, terminated, truncated, info = env.step(fast0)
        label = f"round0_step{local_step}"

        assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, label)
        print_step_summary(env, reward, info, f"[{label}]")

        if local_step < cfg.slow_T:
            assert_true(
                not bool(info["is_round_boundary"]),
                f"{label}: boundary should be False",
            )
            assert_equal(f"{label}: round_idx", env.round_idx, 0)
            assert_equal(f"{label}: round_slot", env.round_slot, local_step)
            assert_true(
                abs(float(env.round_fast_reward_sum)) > 0.0,
                f"{label}: round_fast_reward_sum should be accumulating",
            )
        else:
            assert_true(
                bool(info["is_round_boundary"]),
                f"{label}: boundary should be True",
            )
            assert_equal(f"{label}: round_idx", env.round_idx, 1)
            assert_equal(f"{label}: round_slot", env.round_slot, 0)
            assert_array_close(
                f"{label}: round_start_E reset to current E",
                env.round_start_E,
                env.E,
                atol=1e-4,
            )
            assert_array_close(
                f"{label}: round_fast_reward_sum reset after boundary",
                np.array([env.round_fast_reward_sum]),
                np.array([0.0]),
                atol=1e-6,
            )

        assert_slow_decision_equal(env, slow0, f"{label}: slow decision persistence")

    ok("round boundary 및 accumulator reset 정상")


def test_07_slow_decision_persistence_until_reapply() -> None:
    section("TEST 07 | slow decision이 다음 apply_slow_action 전까지 유지되는지 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)
    prepare_content_match(env, user_idx=1, uav_idx=1, content_id=1)

    slow0 = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    slow1 = make_slow_action(cfg, rsu_idx=1, uav_idx=1, user_idx=1)

    env.apply_slow_action(slow0)
    assert_slow_decision_equal(env, slow0, "round0_slow0_applied")

    fast0 = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)

    for i in range(cfg.slow_T):
        next_obs, reward, terminated, truncated, info = env.step(fast0)
        assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, f"slow_persist_step{i}")
        assert_slow_decision_equal(env, slow0, f"slow must persist after step {i}")

    assert_equal("after boundary round_idx", env.round_idx, 1)
    assert_equal("after boundary round_slot", env.round_slot, 0)
    assert_slow_decision_equal(env, slow0, "slow still persists after boundary before reapply")

    env.apply_slow_action(slow1)
    assert_slow_decision_equal(env, slow1, "slow changed only after reapply")

    ok("slow decision은 round 동안 유지되고, 새 apply_slow_action에서만 변경됨")


def test_08_validator_shape_errors_and_clipping_behavior() -> None:
    section("TEST 08 | validator shape error 및 clipping behavior 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    bad_slow = make_empty_slow_action(cfg)
    bad_slow["uav_hiring"] = np.zeros((cfg.num_uav, 1), dtype=np.int32)

    assert_raises(
        "bad slow action shape should raise ValueError",
        ValueError,
        lambda: env.apply_slow_action(bad_slow),
    )

    bad_fast = make_empty_fast_action(cfg)
    bad_fast["uav_power"] = np.zeros((cfg.num_uav + 1, cfg.num_user), dtype=np.float32)

    assert_raises(
        "bad fast action shape should raise ValueError",
        ValueError,
        lambda: env.step(bad_fast),
    )

    # 현재 validators.py 구현은 음수/초과값을 reject하지 않고 min/max로 clipping한다.
    # 따라서 이 테스트는 "에러가 나지 않고 finite state를 유지하는지"와
    # "초과 power가 max_tx_power 이하로 처리되는지"를 간접 검증한다.
    env = Env(cfg)
    env.reset()
    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    env.apply_slow_action(slow)

    clipped_fast = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    clipped_fast["rsu_chunks"][0, 0] = cfg.chunk + 100
    clipped_fast["rsu_layers"][0, 0] = cfg.layer + 100
    clipped_fast["uav_chunks"][0, 0] = cfg.chunk + 100
    clipped_fast["uav_layers"][0, 0] = cfg.layer + 100
    clipped_fast["uav_power"][0, 0] = cfg.battery.max_tx_power + 1000.0
    clipped_fast["playback"] = -np.ones(cfg.num_user, dtype=np.float32)

    next_obs, reward, terminated, truncated, info = env.step(clipped_fast)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "clipped_fast_step")

    assert_true(np.all(np.asarray(info["playback"]) >= 0.0), "playback should be clipped to nonnegative")
    assert_true(np.all(env.E >= 0.0), "E should remain nonnegative after clipped action")
    assert_true(np.all(env.E <= cfg.battery.e_max + 1e-6), "E should remain within e_max after clipped action")

    ok("validator shape error 및 clipping behavior 정상")


def test_09_rsu_only_and_uav_only_paths() -> None:
    section("TEST 09 | RSU-only / UAV-only delivery path 분리 검증")

    cfg = build_test_config()

    # RSU-only
    env = Env(cfg)
    env.reset()

    slow_rsu = make_slow_action(
        cfg,
        rsu_idx=0,
        uav_idx=None,
        user_idx=0,
        enable_rsu=True,
        enable_uav=False,
    )
    env.apply_slow_action(slow_rsu)

    fast_rsu = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=True,
        enable_uav=True,
    )

    next_obs, reward, terminated, truncated, info = env.step(fast_rsu)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "rsu_only_step")
    print_step_summary(env, reward, info, "[RSU-only step]")

    delivered_rsu = np.asarray(info["delivered_rsu_per_user"], dtype=np.float32)
    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)

    assert_true(float(delivered_rsu[0]) > 0.0, "RSU-only: RSU delivery should be positive")
    assert_array_close(
        "RSU-only: UAV delivery should be zero",
        delivered_uav,
        np.zeros(cfg.num_user, dtype=np.float32),
    )

    # UAV-only
    env = Env(cfg)
    env.reset()
    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow_uav = make_slow_action(
        cfg,
        rsu_idx=None,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )
    env.apply_slow_action(slow_uav)

    fast_uav = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=True,
        enable_uav=True,
    )

    next_obs, reward, terminated, truncated, info = env.step(fast_uav)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "uav_only_step")
    print_step_summary(env, reward, info, "[UAV-only step]")

    delivered_rsu = np.asarray(info["delivered_rsu_per_user"], dtype=np.float32)
    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)

    assert_array_close(
        "UAV-only: RSU delivery should be zero",
        delivered_rsu,
        np.zeros(cfg.num_user, dtype=np.float32),
    )
    assert_true(float(delivered_uav[0]) > 0.0, "UAV-only: UAV delivery should be positive")

    ok("RSU-only / UAV-only delivery path 분리 정상")


def test_10_multi_step_stability_smoke() -> None:
    section("TEST 10 | 여러 step 반복 시 finite/state invariant 유지 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)
    prepare_content_match(env, user_idx=1, uav_idx=1, content_id=1)

    slow = make_empty_slow_action(cfg)
    slow["rsu_scheduling"][0, 0] = 1
    slow["rsu_scheduling"][1, 1] = 1
    slow["uav_hiring"][0] = 1
    slow["uav_hiring"][1] = 1
    slow["uav_scheduling"][0, 0] = 1
    slow["uav_scheduling"][1, 1] = 1

    env.apply_slow_action(slow)

    for step_idx in range(12):
        fast = make_empty_fast_action(cfg)

        fast["rsu_chunks"][0, 0] = 1
        fast["rsu_layers"][0, 0] = 1
        fast["uav_chunks"][0, 0] = 1
        fast["uav_layers"][0, 0] = 1
        fast["uav_power"][0, 0] = 5.0

        fast["rsu_chunks"][1, 1] = 1
        fast["rsu_layers"][1, 1] = 2
        fast["uav_chunks"][1, 1] = 1
        fast["uav_layers"][1, 1] = 2
        fast["uav_power"][1, 1] = 5.0

        next_obs, reward, terminated, truncated, info = env.step(fast)
        assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, f"multi_step_{step_idx}")

        if step_idx in {0, 1, 2, 5, 11}:
            print_step_summary(env, reward, info, f"[multi step {step_idx}]")

        assert_true(not truncated, f"multi_step_{step_idx}: should not truncate in compact smoke test")

        if bool(info["is_round_boundary"]):
            # 다음 round slow decision을 다시 넣는 HRL 흐름을 명시적으로 모사
            env.apply_slow_action(slow)

    ok("multi-step stability smoke 정상")


# =============================================================================
# Test runner
# =============================================================================

def run_test(name: str, fn: Callable[[], None]) -> TestResult:
    try:
        fn()
        return TestResult(name=name, passed=True, message="passed")
    except Exception as exc:
        fail(f"{name} failed: {exc}")
        traceback.print_exc()
        return TestResult(name=name, passed=False, message=str(exc))


def main() -> int:
    title("ENV COMPREHENSIVE TEST | feat/hrl")

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("test_01_reset_and_initial_obs", test_01_reset_and_initial_obs),
        ("test_02_no_slow_action_blocks_delivery", test_02_no_slow_action_blocks_delivery),
        ("test_03_basic_service_flow_and_queue_battery", test_03_basic_service_flow_and_queue_battery),
        ("test_04_content_mismatch_blocks_uav_delivery", test_04_content_mismatch_blocks_uav_delivery),
        ("test_05_charging_blocks_service_and_updates_battery", test_05_charging_blocks_service_and_updates_battery),
        ("test_06_round_transition_and_accumulator", test_06_round_transition_and_accumulator),
        ("test_07_slow_decision_persistence_until_reapply", test_07_slow_decision_persistence_until_reapply),
        ("test_08_validator_shape_errors_and_clipping_behavior", test_08_validator_shape_errors_and_clipping_behavior),
        ("test_09_rsu_only_and_uav_only_paths", test_09_rsu_only_and_uav_only_paths),
        ("test_10_multi_step_stability_smoke", test_10_multi_step_stability_smoke),
    ]

    results: List[TestResult] = []

    for name, fn in tests:
        results.append(run_test(name, fn))

    print()
    hr("=")
    print(c("[ SUMMARY ]", BOLD + CYAN))
    hr("=")

    passed = 0
    failed = 0

    for result in results:
        if result.passed:
            passed += 1
            print(c(f"[PASS] {result.name}", GREEN))
        else:
            failed += 1
            print(c(f"[FAIL] {result.name}: {result.message}", RED))

    print()
    kv("passed", passed, indent=2)
    kv("failed", failed, indent=2)
    kv("total", len(results), indent=2)

    if failed == 0:
        print()
        ok("All env tests passed.")
        return 0

    print()
    fail("Some env tests failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())