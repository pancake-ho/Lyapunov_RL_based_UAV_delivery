from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from env.env import Env
from agent.PPO.common import (
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
    _apply_selected_slow_action,
    _build_fast_agent,
    _extract_slow_reward,
    _load_trusted_checkpoint,
    _make_env_config,
    _resolve_path,
    _resolve_ppo_reward_scale,
    _select_slow_action,
)


def run_checkpoint_joint_smoke_test() -> None:
    run_cfg = replace(
        get_slow_joint_train_config(),
        num_episodes=1,
        rounds_per_episode=1,
        final_slow_T=2,
        final_target_service_slots_per_round=2,
        forecast_scenarios=1,
        max_coordinate_sweeps=4,
        fast_rollout_steps=2,
        fast_batch_size=2,
        fast_update_epochs=1,
        load_pretrained_optimizer=False,
    )
    checkpoint_path = _resolve_path(
        run_cfg.fast_checkpoint
    )
    checkpoint = _load_trusted_checkpoint(
        checkpoint_path
    )
    extra = dict(checkpoint.get("extra", {}))
    env_cfg = _make_env_config(
        run_cfg=run_cfg,
        checkpoint_extra=extra,
    )

    probe_env = Env(env_cfg)
    sample_obs, _ = split_env_reset(
        probe_env.reset()
    )
    fast_agent = _build_fast_agent(
        run_cfg=run_cfg,
        env_cfg=env_cfg,
        sample_fast_obs=sample_obs,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        load_optimizer=False,
    )
    if fast_agent.device.type != "cuda":
        raise RuntimeError(
            "Checkpoint joint smoke test requires CUDA."
        )

    reward_scale = _resolve_ppo_reward_scale(
        run_cfg=run_cfg,
        checkpoint_extra=extra,
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
    completed_rounds = 0
    last_obs = env.get_fast_obs()

    for round_idx in range(
        int(run_cfg.rounds_per_episode)
    ):
        selected_slow = _select_slow_action(
            controller=controller,
            env=env,
            fast_agent=fast_agent,
        )
        _apply_selected_slow_action(
            env=env,
            selected=selected_slow,
        )

        boundary_info = None
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
                or (
                    round_idx + 1
                    == int(run_cfg.rounds_per_episode)
                    and slot_idx + 1
                    == int(env_cfg.slow_T)
                )
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
                "Smoke run did not reach a round boundary."
            )
        slow_reward, slow_components = (
            _extract_slow_reward(boundary_info)
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
                "Smoke R_H(r)=-J^S(r) check failed."
            )
        completed_rounds += 1

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
            "Fast PPO parameters did not change in the smoke update."
        )
    if int(
        update_logs["completed_minibatches"]
    ) <= 0:
        raise RuntimeError(
            "Fast PPO smoke update completed no minibatches."
        )
    if not all(
        np.isfinite(float(value))
        for value in update_logs.values()
    ):
        raise RuntimeError(
            "Fast PPO smoke update produced NaN or Inf."
        )
    if completed_rounds != int(
        run_cfg.rounds_per_episode
    ):
        raise RuntimeError(
            "Smoke run completed the wrong number of rounds."
        )
    if len(fast_agent.buffer) != 0:
        raise RuntimeError(
            "Fast PPO buffer was not reset after update."
        )

    print(
        "[PASS] checkpoint joint smoke: Slow DPP selected complete "
        "(y, mu, phi) actions, realized full rounds, preserved "
        "R_H=-J^S, stored mixed-action transitions, and updated Fast PPO."
    )


if __name__ == "__main__":
    run_checkpoint_joint_smoke_test()
