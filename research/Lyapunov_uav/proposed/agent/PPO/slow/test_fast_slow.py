from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import EnvConfig  # noqa: E402
from env.env import Env  # noqa: E402
from agent.PPO.common import (  # noqa: E402
    infer_fast_obs_dim,
    infer_slow_obs_dim,
    split_env_reset,
    split_env_step,
)
from agent.PPO.fast.fast_agent import (  # noqa: E402
    FastPPOAgent,
    FastPPOConfig,
)
from agent.PPO.slow.slow_agent import (  # noqa: E402
    SlowPPOAgent,
    SlowPPOConfig,
)


def run_contract_smoke_test() -> None:
    """
    Exercise one complete high-level transition with a short test horizon.

    The production scenario keeps slow_T=3600.  This smoke test uses slow_T=8
    only to test the boundary contract quickly; it does not change or validate
    the research timescale itself.
    """
    env_cfg = replace(
        EnvConfig(),
        slow_T=8,
        episode_slots=8,
        move_prob=0.0,
        seed=9876,
    )
    env = Env(env_cfg)
    fast_obs, _ = split_env_reset(env.reset())
    slow_obs = env.get_slow_obs()

    slow_agent = SlowPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_slow_obs_dim(slow_obs),
        ppo_cfg=SlowPPOConfig(
            rollout_rounds=8,
            batch_size=4,
            normalize_obs=False,
            device="cpu",
            rsu_init_logit=0.0,
            hiring_init_logit=0.0,
            uav_init_logit=0.0,
        ),
    )
    fast_agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_fast_obs_dim(fast_obs),
        ppo_cfg=FastPPOConfig(
            rollout_steps=8,
            batch_size=4,
            normalize_obs=False,
            device="cpu",
        ),
    )

    selected_slow = slow_agent.select_action(
        slow_obs,
        context=env,
        deterministic=True,
        update_norm=False,
    )
    if int(selected_slow["action_info"]["projection_count"]) != 0:
        raise AssertionError("slow action required post-sampling projection.")

    binary = np.asarray(
        selected_slow["binary_action"], dtype=np.float32
    )
    mask = np.asarray(
        selected_slow["action_mask"], dtype=np.float32
    )
    if np.any(binary * (1.0 - mask) != 0.0):
        raise AssertionError("slow action has non-zero bits outside its mask.")

    env_action = selected_slow["env_action"]
    rsu = np.asarray(env_action["rsu_scheduling"], dtype=np.int32)
    hiring = np.asarray(env_action["uav_hiring"], dtype=np.int32)
    uav = np.asarray(env_action["uav_scheduling"], dtype=np.int32)
    if np.any(uav > hiring[:, None]):
        raise AssertionError("phi_un=1 while mu_u=0.")
    if np.any(rsu.sum(axis=0) + uav.sum(axis=0) > 1):
        raise AssertionError("per-user slow scheduling is not exclusive.")

    env.apply_slow_action(env_action)
    fixed_rsu = env.rsu_scheduling.copy()
    fixed_hiring = env.uav_hiring.copy()
    fixed_uav = env.uav_scheduling.copy()

    boundary_info = None
    for slot in range(int(env_cfg.slow_T)):
        selected_fast = fast_agent.select_action(
            fast_obs,
            deterministic=True,
            update_norm=False,
        )
        fast_obs, _, terminated, truncated, info = split_env_step(
            env.step(selected_fast["env_action"])
        )

        if slot + 1 < int(env_cfg.slow_T):
            if not np.array_equal(env.rsu_scheduling, fixed_rsu):
                raise AssertionError("RSU slow action changed within a round.")
            if not np.array_equal(env.uav_hiring, fixed_hiring):
                raise AssertionError(
                    "UAV hiring action changed within a round."
                )
            if not np.array_equal(env.uav_scheduling, fixed_uav):
                raise AssertionError("UAV slow action changed within a round.")

        if bool(info.get("is_round_boundary", False)):
            boundary_info = info
        if terminated:
            raise AssertionError("unexpected termination in smoke test.")
        if truncated and slot + 1 < int(env_cfg.slow_T):
            raise AssertionError("episode truncated before round boundary.")

    if boundary_info is None:
        raise AssertionError("no slow round boundary was produced.")

    reward_components = boundary_info.get("reward_components", {})
    slow_components = reward_components.get(
        "slow_reward_components", {}
    )
    if not bool(slow_components.get("is_round_boundary", False)):
        raise AssertionError("slow reward missing at round boundary.")

    observed = float(reward_components.get("slow_reward", np.nan))
    expected = float(slow_components["hire_cost"]) + float(
        slow_components["round_fast_reward_sum"]
    )
    tolerance = 1e-3 + 1e-6 * abs(expected)
    if not np.isfinite(observed) or abs(observed - expected) > tolerance:
        raise AssertionError(
            "slow reward identity failed: "
            f"observed={observed}, expected={expected}, "
            f"tolerance={tolerance}"
        )

    required_info = {
        "transmitted_rsu_per_user",
        "transmitted_uav_per_user",
        "battery_step_info",
        "prev_connection_state",
        "next_connection_state",
    }
    missing = sorted(required_info - set(boundary_info))
    if missing:
        raise AssertionError(
            f"boundary info is missing required metrics: {missing}"
        )

    print(
        "[PASS] fast/slow contract smoke test: "
        "mask, exclusivity, fixed round action, and reward identity"
    )


if __name__ == "__main__":
    run_contract_smoke_test()