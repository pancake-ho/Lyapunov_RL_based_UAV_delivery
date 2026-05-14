from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

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


def hr(char: str = "=", width: int = 100) -> None:
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
            f"{name}: shape mismatch, actual={actual_arr.shape}, "
            f"expected={expected_arr.shape}"
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
            f"{name}: shape mismatch, actual={actual_arr.shape}, "
            f"expected={expected_arr.shape}"
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
    rsu_idx: int,
    uav_idx: int,
    user_idx: int,
) -> Dict[str, np.ndarray]:
    action = make_empty_slow_action(cfg)
    action["rsu_scheduling"][rsu_idx, user_idx] = 1
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
    rsu_idx: int,
    uav_idx: int,
    user_idx: int,
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

def expected_obs_shapes(cfg: EnvConfig) -> Dict[str, Tuple[int, ...]]:
    return {
        "Z": (cfg.num_user,),
        "Y": (cfg.num_uav,),
        "uav_scheduling": (cfg.num_uav, cfg.num_user),
        "rsu_user_distance": (cfg.num_rsu, cfg.num_user),
        "uav_user_distance": (cfg.num_uav, cfg.num_user),
    }


def assert_obs_dict(obs: Dict[str, np.ndarray], cfg: EnvConfig, name: str) -> None:
    """
    현재 env.py의 get_fast_obs() 기준 observation 검증.

    interface.py를 삭제한 구조이므로 flatten vector 검사는 수행하지 않는다.
    """
    if not isinstance(obs, dict):
        raise TestFailure(f"{name}: obs must be dict, got {type(obs)}")

    shapes = expected_obs_shapes(cfg)

    for key, shape in shapes.items():
        if key not in obs:
            raise TestFailure(f"{name}: obs missing required key: {key}")
        assert_array_shape(f"{name}.obs[{key}]", obs[key], shape)
        assert_array_finite(f"{name}.obs[{key}]", obs[key])

    assert_binary_array(f"{name}.obs[uav_scheduling]", obs["uav_scheduling"])


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

    assert_array_finite(f"{name}.queue", env.queue)
    assert_array_finite(f"{name}.Z", env.Z)
    assert_array_finite(f"{name}.E", env.E)
    assert_array_finite(f"{name}.Y", env.Y)

    assert_binary_array(f"{name}.rsu_scheduling", env.rsu_scheduling)
    assert_binary_array(f"{name}.uav_hiring", env.uav_hiring)
    assert_binary_array(f"{name}.uav_scheduling", env.uav_scheduling)
    assert_binary_array(f"{name}.outage", env.outage)
    assert_binary_array(f"{name}.charging_state", env.charging_state)

    assert_true(np.all(env.queue >= -1e-6), f"{name}: queue has negative values")
    assert_true(np.all(env.queue <= cfg.max_queue + cfg.chunk * cfg.num_rsu + cfg.chunk * cfg.num_uav + 1e-6),
                f"{name}: queue is unexpectedly large")
    assert_true(np.all(env.E >= -1e-6), f"{name}: E has negative values")
    assert_true(np.all(env.E <= cfg.battery.e_max + 1e-6), f"{name}: E exceeds e_max")
    assert_true(np.all(env.Y >= -1e-6), f"{name}: Y has negative values")

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
    assert_obs_dict(next_obs, cfg, name=f"{step_name}.next_obs")
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

    assert_array_shape(f"{step_name}.info.prev_Q", info["prev_Q"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.next_Q", info["next_Q"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.prev_Z", info["prev_Z"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.next_Z", info["next_Z"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.playback", info["playback"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.consumed", info["consumed"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.stall", info["stall"], (cfg.num_user,))
    assert_array_shape(f"{step_name}.info.prev_E", info["prev_E"], (cfg.num_uav,))
    assert_array_shape(f"{step_name}.info.next_E", info["next_E"], (cfg.num_uav,))
    assert_array_shape(f"{step_name}.info.prev_Y", info["prev_Y"], (cfg.num_uav,))
    assert_array_shape(f"{step_name}.info.next_Y", info["next_Y"], (cfg.num_uav,))

    for key in [
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
        "delivered_rsu_per_user",
        "delivered_uav_per_user",
        "delivered_total_per_user",
        "quality_rsu_per_user",
        "quality_uav_per_user",
        "quality_total_per_user",
    ]:
        assert_array_finite(f"{step_name}.info[{key}]", info[key])

    battery_info = info["battery_step_info"]
    if not isinstance(battery_info, list):
        raise TestFailure(f"{step_name}: battery_step_info must be list")
    if len(battery_info) != cfg.num_uav:
        raise TestFailure(
            f"{step_name}: battery_step_info length mismatch, "
            f"actual={len(battery_info)}, expected={cfg.num_uav}"
        )

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


def print_step_summary(env: Env, reward: float, info: Dict[str, Any], label: str) -> None:
    print(c(label, BOLD))
    kv("time", f"{info.get('prev_time')} -> {info.get('next_time')}")
    kv("round", f"{info.get('active_round_idx')} -> {info.get('next_round_idx')}")
    kv("round_slot", f"{info.get('prev_round_slot')} -> {info.get('next_round_slot')}")
    kv("is_round_boundary", info.get("is_round_boundary"))
    kv("reward", f"{float(reward):.6f}")
    kv("Q", fmt_arr(env.queue))
    kv("Z", fmt_arr(env.Z))
    kv("E", fmt_arr(env.E))
    kv("Y", fmt_arr(env.Y))
    kv("delivered_rsu", fmt_arr(info.get("delivered_rsu_per_user")))
    kv("delivered_uav", fmt_arr(info.get("delivered_uav_per_user")))
    kv("delivered_total", fmt_arr(info.get("delivered_total_per_user")))
    kv("quality_total", fmt_arr(info.get("quality_total_per_user")))
    kv("outage", fmt_arr(env.outage))
    kv("charging_state", fmt_arr(env.charging_state))

    rc = info.get("reward_components", {})
    if isinstance(rc, dict):
        frc = rc.get("fast_reward_components", {})
        src = rc.get("slow_reward_components", {})
        if isinstance(frc, dict):
            kv(
                "fast_terms",
                {
                    "video": round(float(frc.get("video_delivery_term", 0.0)), 6),
                    "battery_consume": round(float(frc.get("battery_consume_term", 0.0)), 6),
                    "battery_charge": round(float(frc.get("battery_charge_term", 0.0)), 6),
                    "quality": round(float(frc.get("quality_term", 0.0)), 6),
                },
            )
        if isinstance(src, dict):
            kv(
                "slow_terms",
                {
                    "is_round_boundary": src.get("is_round_boundary"),
                    "slow_reward": round(float(src.get("slow_reward", 0.0)), 6),
                    "hire_cost": round(float(src.get("hire_cost", 0.0)), 6),
                },
            )


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


# =============================================================================
# Individual tests
# =============================================================================

def test_01_reset_and_initial_obs() -> None:
    section("TEST 01 | reset 및 초기 observation/state 검증")

    cfg = build_test_config()
    env = Env(cfg)

    obs, reset_info = env.reset()

    kv("reset_info", reset_info)
    kv("obs_keys", list(obs.keys()))

    assert_obs_dict(obs, cfg, "reset")
    assert_env_state(env, cfg, "after_reset")

    assert_equal("reset t", env.t, 0)
    assert_equal("reset round_idx", env.round_idx, 0)
    assert_equal("reset round_slot", env.round_slot, 0)
    assert_array_close("round_start_E == E after reset", env.round_start_E, env.E)

    ok("reset observation/state 정상")


def test_02_no_slow_action_blocks_delivery() -> None:
    section("TEST 02 | slow action 미적용 상태에서 fast action이 delivery로 반영되지 않는지 검증")

    cfg = build_test_config()
    env = Env(cfg)
    obs, _ = env.reset()
    assert_obs_dict(obs, cfg, "reset")

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

    assert_array_equal("slow.rsu_scheduling applied", env.rsu_scheduling, slow["rsu_scheduling"])
    assert_array_equal("slow.uav_hiring applied", env.uav_hiring, slow["uav_hiring"])
    assert_array_equal("slow.uav_scheduling applied", env.uav_scheduling, slow["uav_scheduling"])

    fast = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)

    prev_e0 = env.E.copy()

    next_obs, reward, terminated, truncated, info = env.step(fast)

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "basic_service_step")
    print_step_summary(env, reward, info, "[basic service step]")

    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_true(float(delivered_total[0]) > 0.0, "user 0 should receive positive delivery")

    assert_true(
        float(env.E[0]) < float(prev_e0[0]),
        f"hired serving UAV 0 should consume battery: prev={prev_e0[0]}, now={env.E[0]}",
    )

    assert_true(not terminated, "basic service step should not terminate")
    assert_true(not truncated, "basic service step should not truncate")

    ok("기본 service/queue/battery 흐름 정상")


def test_04_round_transition_and_accumulator() -> None:
    section("TEST 04 | round boundary, round_idx/round_slot, accumulator reset 검증")

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
            assert_equal(f"{label}: t", env.t, cfg.slow_T)
            assert_equal(f"{label}: round_idx", env.round_idx, 1)
            assert_equal(f"{label}: round_slot", env.round_slot, 0)

            # boundary 이후 _start_new_round()가 호출되면 round_start_E는 현재 E와 같아야 함
            assert_array_close(
                f"{label}: boundary round_start_E == E",
                env.round_start_E,
                env.E,
                atol=1e-4,
            )

            # boundary 이후 accumulator reset 확인
            assert_true(
                abs(float(env.round_fast_reward_sum)) <= 1e-6,
                f"{label}: round_fast_reward_sum should reset, got {env.round_fast_reward_sum}",
            )
            assert_true(
                abs(float(env.round_quality_sum)) <= 1e-6,
                f"{label}: round_quality_sum should reset, got {env.round_quality_sum}",
            )
            assert_true(
                abs(float(env.round_delivery_sum)) <= 1e-6,
                f"{label}: round_delivery_sum should reset, got {env.round_delivery_sum}",
            )

        assert_array_equal(
            f"{label}: slow rsu_scheduling 유지",
            env.rsu_scheduling,
            slow0["rsu_scheduling"],
        )
        assert_array_equal(
            f"{label}: slow uav_hiring 유지",
            env.uav_hiring,
            slow0["uav_hiring"],
        )
        assert_array_equal(
            f"{label}: slow uav_scheduling 유지",
            env.uav_scheduling,
            slow0["uav_scheduling"],
        )

        assert_true(not terminated, f"{label}: should not terminate")
        assert_true(not truncated, f"{label}: should not truncate")

    ok("round boundary 및 accumulator reset 정상")


def test_05_slow_action_reapply_next_round() -> None:
    section("TEST 05 | 새 round에서 slow action 재적용 및 decision 교체 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)
    prepare_content_match(env, user_idx=1, uav_idx=1, content_id=1)

    slow0 = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    fast0 = make_fast_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)

    env.apply_slow_action(slow0)

    for _ in range(cfg.slow_T):
        env.step(fast0)

    assert_equal("after round0 boundary round_idx", env.round_idx, 1)
    assert_equal("after round0 boundary round_slot", env.round_slot, 0)

    slow1 = make_slow_action(cfg, rsu_idx=1, uav_idx=1, user_idx=1)
    fast1 = make_fast_action(cfg, rsu_idx=1, uav_idx=1, user_idx=1)

    env.apply_slow_action(slow1)

    assert_equal("after slow1 apply round_idx", env.round_idx, 1)
    assert_equal("after slow1 apply round_slot", env.round_slot, 0)
    assert_array_equal("slow1.rsu_scheduling applied", env.rsu_scheduling, slow1["rsu_scheduling"])
    assert_array_equal("slow1.uav_hiring applied", env.uav_hiring, slow1["uav_hiring"])
    assert_array_equal("slow1.uav_scheduling applied", env.uav_scheduling, slow1["uav_scheduling"])
    assert_array_close("slow1 round_start_E == E", env.round_start_E, env.E)

    next_obs, reward, terminated, truncated, info = env.step(fast1)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "round1_first_step")
    print_step_summary(env, reward, info, "[round1 first step]")

    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_true(float(delivered_total[1]) > 0.0, "round1 user 1 should receive delivery")
    assert_equal("round1 first step round_idx", env.round_idx, 1)
    assert_equal("round1 first step round_slot", env.round_slot, 1)

    ok("round 전이 이후 slow action 교체 정상")


def test_06_cache_mismatch_blocks_uav_delivery() -> None:
    section("TEST 06 | UAV cache mismatch 시 UAV delivery 차단 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_mismatch(env, user_idx=0, uav_idx=0)

    # RSU는 꺼두고 UAV만 시도해서 cache mismatch 영향만 확인
    slow = make_empty_slow_action(cfg)
    slow["uav_hiring"][0] = 1
    slow["uav_scheduling"][0, 0] = 1

    fast = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )

    env.apply_slow_action(slow)

    next_obs, reward, terminated, truncated, info = env.step(fast)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "cache_mismatch_step")
    print_step_summary(env, reward, info, "[cache mismatch step]")

    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)
    assert_array_close(
        "cache mismatch: delivered_uav must be zero",
        delivered_uav,
        np.zeros(cfg.num_user, dtype=np.float32),
    )

    ok("cache mismatch에서 UAV delivery 차단 정상")


def test_07_low_battery_rule_based_charging() -> None:
    section("TEST 07 | low battery에서 rule-based charging hook 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    prepare_content_match(env, user_idx=0, uav_idx=0, content_id=0)

    slow = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    fast = make_fast_action(
        cfg,
        rsu_idx=0,
        uav_idx=0,
        user_idx=0,
        enable_rsu=False,
        enable_uav=True,
    )

    env.apply_slow_action(slow)

    # e_min 이하이면 _rule_based_uav_charge()가 charging을 켜야 함
    set_uav_soc(env, cfg, uav_idx=0, soc=cfg.battery.e_min)

    e_before = env.E.copy()

    next_obs, reward, terminated, truncated, info = env.step(fast)
    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "low_battery_charge_step")
    print_step_summary(env, reward, info, "[low battery charge step]")

    uav_charge_effective = np.asarray(info.get("uav_charge_effective"), dtype=np.int32)
    assert_array_shape("uav_charge_effective", uav_charge_effective, (cfg.num_uav,))
    assert_equal("uav_charge_effective[0]", int(uav_charge_effective[0]), 1)

    delivered_uav = np.asarray(info["delivered_uav_per_user"], dtype=np.float32)
    assert_array_close(
        "charging UAV should not deliver",
        delivered_uav,
        np.zeros(cfg.num_user, dtype=np.float32),
    )

    assert_true(
        float(env.E[0]) >= float(e_before[0]),
        f"charging should not decrease E[0]: before={e_before[0]}, after={env.E[0]}",
    )

    ok("low battery charging hook 정상")


def test_08_empty_fast_action_is_safe() -> None:
    section("TEST 08 | empty fast action이 안전하게 처리되는지 검증")

    cfg = build_test_config()
    env = Env(cfg)
    env.reset()

    slow = make_slow_action(cfg, rsu_idx=0, uav_idx=0, user_idx=0)
    env.apply_slow_action(slow)

    # 완전히 빈 action dict를 넣어도 parser default로 안전하게 처리되어야 함
    next_obs, reward, terminated, truncated, info = env.step({})

    assert_step_result(env, next_obs, reward, terminated, truncated, info, cfg, "empty_action_step")
    print_step_summary(env, reward, info, "[empty fast action step]")

    delivered_total = np.asarray(info["delivered_total_per_user"], dtype=np.float32)
    assert_array_close(
        "empty fast action: delivered_total must be zero",
        delivered_total,
        np.zeros(cfg.num_user, dtype=np.float32),
    )

    ok("empty fast action 처리 정상")


# =============================================================================
# Test runner
# =============================================================================

def run_test(name: str, fn: Callable[[], None]) -> TestResult:
    try:
        fn()
        return TestResult(name=name, passed=True, message="passed")
    except Exception as exc:
        fail(f"{name} failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return TestResult(name=name, passed=False, message=f"{type(exc).__name__}: {exc}")


def main() -> None:
    title("ENV.PY COMPREHENSIVE RUNTIME TEST")

    print("Purpose:")
    print("  - import/reset/apply_slow_action/step runtime 검증")
    print("  - queue update, virtual queue, battery E/Y 정합성 검증")
    print("  - round boundary 및 slow decision 유지/교체 검증")
    print("  - cache mismatch, low battery charging, empty action edge case 검증")
    print()

    cfg_preview = build_test_config()
    print("Config preview:")
    kv("num_user / num_rsu / num_uav", f"{cfg_preview.num_user} / {cfg_preview.num_rsu} / {cfg_preview.num_uav}")
    kv("slow_T", cfg_preview.slow_T)
    kv("layer / chunk", f"{cfg_preview.layer} / {cfg_preview.chunk}")
    kv("max_queue", cfg_preview.max_queue)
    kv("battery e_init/e_min/e_max", f"{cfg_preview.battery.e_init} / {cfg_preview.battery.e_min} / {cfg_preview.battery.e_max}")
    kv("base_chunk_size_bits", cfg_preview.base_chunk_size_bits)

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("TEST 01 reset_and_initial_obs", test_01_reset_and_initial_obs),
        ("TEST 02 no_slow_action_blocks_delivery", test_02_no_slow_action_blocks_delivery),
        ("TEST 03 basic_service_flow_and_queue_battery", test_03_basic_service_flow_and_queue_battery),
        ("TEST 04 round_transition_and_accumulator", test_04_round_transition_and_accumulator),
        ("TEST 05 slow_action_reapply_next_round", test_05_slow_action_reapply_next_round),
        ("TEST 06 cache_mismatch_blocks_uav_delivery", test_06_cache_mismatch_blocks_uav_delivery),
        ("TEST 07 low_battery_rule_based_charging", test_07_low_battery_rule_based_charging),
        ("TEST 08 empty_fast_action_is_safe", test_08_empty_fast_action_is_safe),
    ]

    results: List[TestResult] = []

    for test_name, test_fn in tests:
        result = run_test(test_name, test_fn)
        results.append(result)

    title("TEST SUMMARY")

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for result in results:
        status = c("PASS", GREEN) if result.passed else c("FAIL", RED)
        print(f"{status} | {result.name}")
        if not result.passed:
            print(f"       {result.message}")

    print()
    print(f"Total: {passed}/{total} passed")

    if passed != total:
        hr("=")
        fail("ENV TEST FAILED")
        print("하나 이상의 테스트가 실패했으므로 위 traceback과 해당 section 로그를 확인해야 한다.")
        hr("=")
        raise SystemExit(1)

    hr("=")
    print(c("[ALL TESTS PASSED] env.py runtime/invariant checks completed successfully", BOLD + GREEN))
    hr("=")


if __name__ == "__main__":
    main()