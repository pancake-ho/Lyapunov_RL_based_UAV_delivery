from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import EnvConfig  # noqa: E402
from env.env import Env  # noqa: E402
from agent.PPO.slow.slow_config import (  # noqa: E402
    SlowDPPConfig,
)
from agent.PPO.slow.slow_dpp_controller import (  # noqa: E402
    SlowDPPController,
)


class ZeroFastAgent:
    """Torch-free deterministic fast agent for the controller smoke test."""

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.cfg = env_cfg

    def select_action(
        self,
        _obs: Dict[str, Any],
        deterministic: bool,
        update_norm: bool,
    ) -> Dict[str, Any]:
        if not deterministic:
            raise AssertionError(
                "Forecast fast action must be deterministic."
            )
        if update_norm:
            raise AssertionError(
                "Forecast must not update the observation normalizer."
            )

        m = int(self.cfg.num_rsu)
        n = int(self.cfg.num_user)
        u = int(self.cfg.num_uav)
        return {
            "env_action": {
                "rsu_chunks": np.zeros(
                    (m, n),
                    dtype=np.int32,
                ),
                "rsu_layers": np.zeros(
                    (m, n),
                    dtype=np.int32,
                ),
                "uav_chunks": np.zeros(
                    (u, n),
                    dtype=np.int32,
                ),
                "uav_layers": np.zeros(
                    (u, n),
                    dtype=np.int32,
                ),
                "uav_power": np.zeros(
                    (u, n),
                    dtype=np.float32,
                ),
                "playback": np.ones(
                    n,
                    dtype=np.float32,
                ),
            },
            "active_action_dims": 0,
            "active_action_ratio": 0.0,
            "service_rate": 0.0,
            "mean_requested_chunks": 0.0,
            "action_saturation_ratio": 0.0,
            **{
                f"layer_{layer}_ratio": 0.0
                for layer in range(
                    1,
                    int(self.cfg.layer) + 1,
                )
            },
        }


def _small_env_config() -> EnvConfig:
    base = EnvConfig()
    return replace(
        base,
        num_user=4,
        num_rsu=2,
        num_uav=2,
        slow_T=2,
        episode_slots=2,
        mobility_mode="fsmc",
        move_prob=0.0,
        seed=2026,
    )


def _small_dpp_config() -> SlowDPPConfig:
    return SlowDPPConfig(
        device="cpu",
        num_episodes=1,
        rounds_per_episode=1,
        forecast_scenarios=1,
        max_exact_region_candidates=128,
        max_coordinate_sweeps=4,
    )


def _prepare_known_state(env: Env) -> None:
    env.user_region = np.asarray(
        [0, 0, 1, 1],
        dtype=np.int32,
    )
    env.requested_content = np.asarray(
        [0, 1, 0, 1],
        dtype=np.int32,
    )
    env.uav_cached_content = np.asarray(
        [0, 0],
        dtype=np.int32,
    )


def _assert_feasible(
    env: Env,
    action: Dict[str, np.ndarray],
) -> None:
    rsu = np.asarray(
        action["rsu_scheduling"],
        dtype=np.int32,
    )
    hiring = np.asarray(
        action["uav_hiring"],
        dtype=np.int32,
    )
    uav = np.asarray(
        action["uav_scheduling"],
        dtype=np.int32,
    )

    provider_count = rsu.sum(axis=0) + uav.sum(axis=0)
    if np.any(provider_count > 1):
        raise AssertionError(
            "RSU/UAV exclusivity was violated."
        )
    if np.any(uav > hiring[:, None]):
        raise AssertionError(
            "UAV scheduling without employment was selected."
        )

    region = np.asarray(env.user_region, dtype=np.int32)
    requested = np.asarray(
        env.requested_content,
        dtype=np.int32,
    )
    cached = np.asarray(
        env.uav_cached_content,
        dtype=np.int32,
    )
    for user_idx in range(int(env.num_user)):
        if rsu[:, user_idx].sum() > 0:
            provider = int(
                np.flatnonzero(rsu[:, user_idx])[0]
            )
            if provider != int(region[user_idx]):
                raise AssertionError(
                    "Cross-region RSU link was selected."
                )
        if uav[:, user_idx].sum() > 0:
            provider = int(
                np.flatnonzero(uav[:, user_idx])[0]
            )
            if provider != int(region[user_idx]):
                raise AssertionError(
                    "Cross-region UAV link was selected."
                )
            if int(cached[provider]) != int(
                requested[user_idx]
            ):
                raise AssertionError(
                    "Cache-incompatible UAV link was selected."
                )


def run_contract_smoke_test() -> None:
    env_cfg = _small_env_config()
    dpp_cfg = _small_dpp_config()
    env = Env(env_cfg)
    env.reset()
    _prepare_known_state(env)

    controller = SlowDPPController(
        env_cfg=env_cfg,
        dpp_cfg=dpp_cfg,
    )
    fast_agent = ZeroFastAgent(env_cfg)

    base = controller._initial_rsu_first_action(env)
    region_zero_candidates = list(
        controller._iter_region_candidates(
            env=env,
            base_action=base,
            region_idx=0,
        )
    )
    expected_count = controller._region_assignment_count(
        env,
        0,
    )
    if len(region_zero_candidates) != expected_count:
        raise AssertionError(
            "Exact region candidate enumeration is incomplete."
        )
    if expected_count != 10:
        raise AssertionError(
            f"Unexpected known-state candidate count: {expected_count}"
        )

    has_idle_hire_candidate = any(
        int(candidate["uav_hiring"][0]) == 1
        and int(
            candidate["uav_scheduling"][0].sum()
        )
        == 0
        for candidate in region_zero_candidates
    )
    if not has_idle_hire_candidate:
        raise AssertionError(
            "mu was derived from phi instead of being jointly enumerated."
        )

    selected = controller.select_action(
        env=env,
        fast_agent=fast_agent,
    )
    action = selected["env_action"]
    _assert_feasible(env, action)

    predicted_cost = float(
        selected["predicted_round_dpp_cost"]
    )
    if not math.isfinite(predicted_cost):
        raise AssertionError(
            "Selected forecast DPP cost is not finite."
        )
    if int(action["uav_hiring"].sum()) != 0:
        raise AssertionError(
            "Zero-service smoke test should not hire a UAV."
        )
    if int(action["rsu_scheduling"].sum()) != int(
        env_cfg.num_user
    ):
        raise AssertionError(
            "Exact DPP ties must follow RSU priority."
        )

    applied = env.apply_slow_action(action)
    fixed_rsu = applied.rsu_scheduling.copy()
    fixed_hiring = applied.uav_hiring.copy()
    fixed_uav = applied.uav_scheduling.copy()

    boundary_info: Dict[str, Any] | None = None
    for slot in range(int(env_cfg.slow_T)):
        selected_fast = fast_agent.select_action(
            env.get_fast_obs(),
            deterministic=True,
            update_norm=False,
        )
        _, _, terminated, truncated, info = env.step(
            selected_fast["env_action"]
        )

        if terminated:
            raise AssertionError(
                "Environment terminated unexpectedly."
            )
        if slot + 1 < int(env_cfg.slow_T):
            if not np.array_equal(
                env.rsu_scheduling,
                fixed_rsu,
            ):
                raise AssertionError(
                    "RSU slow action changed inside a round."
                )
            if not np.array_equal(
                env.uav_hiring,
                fixed_hiring,
            ):
                raise AssertionError(
                    "UAV employment changed inside a round."
                )
            if not np.array_equal(
                env.uav_scheduling,
                fixed_uav,
            ):
                raise AssertionError(
                    "UAV scheduling changed inside a round."
                )

        if bool(info.get("is_round_boundary", False)):
            boundary_info = dict(info)
            break
        if truncated:
            raise AssertionError(
                "Episode truncated before the round boundary."
            )

    if boundary_info is None:
        raise AssertionError(
            "No complete slow round was produced."
        )

    reward_components = dict(
        boundary_info.get("reward_components", {})
    )
    slow_reward = float(
        reward_components.get("slow_reward", np.nan)
    )
    slow_components = dict(
        reward_components.get(
            "slow_reward_components",
            {},
        )
    )
    round_dpp_cost = float(
        slow_components.get("round_dpp_cost", np.nan)
    )
    if not bool(
        slow_components.get("is_round_boundary", False)
    ):
        raise AssertionError(
            "Slow reward was not emitted at the boundary."
        )
    if not np.isclose(
        slow_reward,
        -round_dpp_cost,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise AssertionError(
            "R_H(r)=-J^S(r) identity failed."
        )

    invalid = {
        "rsu_scheduling": np.zeros(
            (2, 4),
            dtype=np.int32,
        ),
        "uav_hiring": np.asarray(
            [1, 0],
            dtype=np.int32,
        ),
        "uav_scheduling": np.zeros(
            (2, 4),
            dtype=np.int32,
        ),
    }
    invalid["uav_scheduling"][0, 1] = 1
    try:
        controller._validate_action(env, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Cache-incompatible UAV action was accepted."
        )

    print(
        "[PASS] slow DPP contract: exact joint candidate enumeration, "
        "explicit employment, region/cache/exclusivity feasibility, "
        "fixed round action, forecast boundary, and R_H=-J^S"
    )


if __name__ == "__main__":
    run_contract_smoke_test()
