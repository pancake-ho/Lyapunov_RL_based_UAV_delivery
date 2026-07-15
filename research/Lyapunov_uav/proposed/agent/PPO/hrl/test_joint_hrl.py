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


def _assert_finite(logs: dict[str, float], name: str) -> None:
    bad = {
        key: value
        for key, value in logs.items()
        if isinstance(value, (int, float)) and not np.isfinite(float(value))
    }
    if bad:
        raise AssertionError(f"{name} update produced non-finite logs: {bad}")


def run_joint_contract_test() -> None:
    """
    Run two short complete slow transitions and update both PPO levels.

    ``slow_T=8`` is test-only.  Production remains ``slow_T=3600``.
    """
    env_cfg = replace(
        EnvConfig(),
        slow_T=8,
        episode_slots=16,
        move_prob=0.0,
        seed=9917,
    )
    env = Env(env_cfg)
    fast_obs, _ = split_env_reset(env.reset())
    slow_obs = env.get_slow_obs()
    fast_agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_fast_obs_dim(fast_obs),
        ppo_cfg=FastPPOConfig(
            rollout_steps=16,
            update_epochs=1,
            batch_size=8,
            normalize_obs=False,
            normalize_adv=True,
            device="cpu",
        ),
    )
    slow_agent = SlowPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_slow_obs_dim(slow_obs),
        ppo_cfg=SlowPPOConfig(
            rollout_rounds=2,
            update_epochs=1,
            batch_size=1,
            normalize_obs=False,
            normalize_adv=True,
            reward_scale=1e-6,
            rsu_init_logit=0.0,
            hiring_init_logit=0.0,
            uav_init_logit=0.0,
            device="cpu",
        ),
    )

    for round_index in range(2):
        slow_obs = env.get_slow_obs()
        slow_selected = slow_agent.select_action(
            slow_obs,
            context=env,
            deterministic=False,
            update_norm=False,
        )
        if int(slow_selected["action_info"]["projection_count"]) != 0:
            raise AssertionError("slow action required post-hoc projection.")
        env.apply_slow_action(slow_selected["env_action"])
        fast_obs = env.get_fast_obs()
        boundary_info = None

        for _ in range(int(env_cfg.slow_T)):
            fast_selected = fast_agent.select_action(
                fast_obs,
                deterministic=False,
                update_norm=False,
            )
            next_obs, reward, terminated, truncated, info = split_env_step(
                env.step(fast_selected["env_action"])
            )
            boundary = bool(info.get("is_round_boundary", False))
            fast_agent.store_transition(
                obs_vec=fast_selected["obs_vec"],
                raw_action=fast_selected["raw_action"],
                action_mask=fast_selected["action_mask"],
                reward=float(reward) * 1e-4,
                done=boundary,
                value=float(fast_selected["value"]),
                log_prob=float(fast_selected["log_prob"]),
            )
            fast_obs = next_obs
            if boundary:
                boundary_info = info
                break
            if terminated or truncated:
                raise AssertionError("episode ended before a slow boundary.")

        if boundary_info is None:
            raise AssertionError("missing slow boundary info.")
        components = boundary_info["reward_components"]
        slow_components = components["slow_reward_components"]
        observed = float(components["slow_reward"])
        expected = float(slow_components["hire_cost"]) + float(
            slow_components["round_fast_reward_sum"]
        )
        tolerance = 1e-3 + 1e-6 * abs(expected)
        if abs(observed - expected) > tolerance:
            raise AssertionError(
                "slow reward identity failed: "
                f"observed={observed}, expected={expected}."
            )

        episode_done = bool(round_index == 1)
        next_slow_obs = env.get_slow_obs()
        slow_agent.store_transition(
            obs_vec=slow_selected["obs_vec"],
            binary_action=slow_selected["binary_action"],
            action_mask=slow_selected["action_mask"],
            reward=observed,
            done=episode_done,
            value=float(slow_selected["value"]),
            log_prob=float(slow_selected["log_prob"]),
        )

    if not fast_agent.buffer.is_full or not slow_agent.buffer.is_full:
        raise AssertionError("joint rollouts did not align at the boundary.")
    fast_agent.finish_rollout(fast_obs, last_done=True)
    slow_agent.finish_rollout(next_slow_obs, last_done=True)
    _assert_finite(fast_agent.update(), "fast")
    _assert_finite(slow_agent.update(), "slow")
    if len(fast_agent.buffer) != 0 or len(slow_agent.buffer) != 0:
        raise AssertionError("PPO buffers were not reset after updates.")
    print(
        "[PASS] joint HRL contract: complete round reward, aligned on-policy "
        "buffers, and finite fast/slow PPO updates"
    )


if __name__ == "__main__":
    run_joint_contract_test()