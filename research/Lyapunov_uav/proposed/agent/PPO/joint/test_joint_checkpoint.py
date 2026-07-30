from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from env.env import Env
from agent.PPO.common import (
    set_seed,
    split_env_reset,
    split_env_step,
)
from agent.PPO.slow.slow_dpp_controller import (
    SlowDPPController,
)
from agent.PPO.joint.joint_config import (
    get_slow_joint_train_config,
)
from agent.PPO.joint.joint_train import (
    FAST_INITIALIZATION,
    _apply_selected_slow_action,
    _build_fast_agent,
    _extract_slow_reward,
    _make_env_config,
    _prime_fresh_obs_normalizer,
    _resolve_ppo_reward_scale,
    _select_slow_action,
)


def run_fresh_joint_smoke_test() -> None:
    run_cfg = replace(
        get_slow_joint_train_config(),
        device="cpu",
        num_episodes=1,
        rounds_per_episode=1,
        final_slow_T=2,
        final_target_service_slots_per_round=2,
        forecast_scenarios=1,
        max_coordinate_sweeps=4,
        fast_rollout_steps=2,
        fast_batch_size=2,
        fast_update_epochs=1,
        fast_target_kl=None,
        resume_checkpoint=None,
    )
    if hasattr(run_cfg, "fast_checkpoint"):
        raise RuntimeError(
            "Fresh joint config must not expose a Fast-only checkpoint."
        )

    set_seed(
        int(run_cfg.seed),
        deterministic=bool(
            run_cfg.deterministic_torch
        ),
    )
    env_cfg = _make_env_config(
        run_cfg=run_cfg,
        checkpoint_extra=None,
    )
    probe_env = Env(env_cfg)
    sample_obs, _ = split_env_reset(
        probe_env.reset()
    )
    fast_agent = _build_fast_agent(
        run_cfg=run_cfg,
        env_cfg=env_cfg,
        sample_fast_obs=sample_obs,
    )

    if fast_agent.device.type != "cpu":
        raise RuntimeError(
            "Fresh joint smoke test expected a CPU agent."
        )
    if fast_agent.optimizer.state:
        raise RuntimeError(
            "Fresh Adam optimizer already contains state."
        )
    if len(fast_agent.buffer) != 0:
        raise RuntimeError(
            "Fresh PPO rollout buffer is not empty."
        )
    if fast_agent.obs_normalizer is not None:
        initial_count = float(
            fast_agent.obs_normalizer.rms.count
        )
        if not np.isclose(
            initial_count,
            1e-4,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Fresh observation normalizer already contains samples."
            )

    reward_scale = _resolve_ppo_reward_scale(
        run_cfg=run_cfg,
        checkpoint_extra=None,
    )
    controller = SlowDPPController(
        env_cfg=env_cfg,
        dpp_cfg=run_cfg,
    )
    env = Env(env_cfg)
    split_env_reset(env.reset())

    before = [
        parameter.detach().clone()
        for parameter in fast_agent.model.parameters()
    ]

    _prime_fresh_obs_normalizer(
        fast_agent=fast_agent,
        obs=env.get_fast_obs(),
    )
    if (
        fast_agent.obs_normalizer is not None
        and float(
            fast_agent.obs_normalizer.rms.count
        ) <= initial_count
    ):
        raise RuntimeError(
            "Fresh observation normalizer was not primed by the first "
            "real joint state."
        )

    selected_slow = _select_slow_action(
        controller=controller,
        env=env,
        fast_agent=fast_agent,
    )
    if selected_slow["action_info"]["controller"] != (
        "round_dpp_coordinate_descent"
    ):
        raise RuntimeError(
            "Slow action was not produced by the DPP controller."
        )
    _apply_selected_slow_action(
        env=env,
        selected=selected_slow,
    )

    boundary_info = None
    last_obs = env.get_fast_obs()
    for slot_idx in range(int(env_cfg.slow_T)):
        selected_fast = fast_agent.select_action(
            env.get_fast_obs(),
            deterministic=False,
            update_norm=True,
        )
        (
            last_obs,
            raw_reward,
            terminated,
            truncated,
            info,
        ) = split_env_step(
            env.step(
                selected_fast["env_action"]
            )
        )
        done = bool(
            terminated
            or truncated
            or slot_idx + 1 == int(env_cfg.slow_T)
        )
        fast_agent.store_transition(
            obs_vec=selected_fast["obs_vec"],
            raw_action=selected_fast["raw_action"],
            action_mask=selected_fast["action_mask"],
            reward=(
                float(raw_reward)
                * float(reward_scale)
            ),
            done=done,
            value=float(selected_fast["value"]),
            log_prob=float(
                selected_fast["log_prob"]
            ),
        )
        if bool(
            info.get("is_round_boundary", False)
        ):
            boundary_info = info
            break

    if boundary_info is None:
        raise RuntimeError(
            "Fresh smoke run did not reach a round boundary."
        )
    slow_reward, slow_components = _extract_slow_reward(
        boundary_info
    )
    round_cost = float(
        slow_components["round_dpp_cost"]
    )
    if not np.isclose(
        slow_reward,
        -round_cost,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Fresh smoke R_H(r)=-J^S(r) check failed."
        )

    fast_agent.finish_rollout(
        last_obs=last_obs,
        last_done=True,
    )
    update_logs = fast_agent.update()

    changed = any(
        not torch.equal(
            old,
            new.detach(),
        )
        for old, new in zip(
            before,
            fast_agent.model.parameters(),
        )
    )
    if not changed:
        raise RuntimeError(
            "Fresh Fast PPO parameters did not change."
        )
    if int(
        update_logs["completed_minibatches"]
    ) <= 0:
        raise RuntimeError(
            "Fresh Fast PPO update completed no minibatches."
        )
    if not all(
        np.isfinite(float(value))
        for value in update_logs.values()
    ):
        raise RuntimeError(
            "Fresh Fast PPO update produced NaN or Inf."
        )
    if len(fast_agent.buffer) != 0:
        raise RuntimeError(
            "Fast PPO buffer was not reset after update."
        )

    print(
        "[PASS] fresh joint smoke: "
        f"fast_initialization={FAST_INITIALIZATION}, "
        "no Fast-only checkpoint, Slow DPP selected (y, mu, phi), "
        "one complete round was realized, Fast PPO parameters updated, "
        "and R_H(r)=-J^S(r) held."
    )


if __name__ == "__main__":
    run_fresh_joint_smoke_test()