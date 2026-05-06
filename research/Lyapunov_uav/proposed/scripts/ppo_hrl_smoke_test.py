from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _add_import_paths() -> None:
    script_path = Path(__file__).resolve()
    proposed_dir = script_path.parents[1]
    lyapunov_dir = proposed_dir.parent
    for path in (lyapunov_dir, proposed_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _check_forward(agent, obs: np.ndarray, action_dim: int) -> None:
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    mean, log_std, value = agent.model.forward(obs_t)
    assert tuple(mean.shape) == (action_dim,)
    assert tuple(log_std.shape) == (action_dim,)
    assert value.ndim == 0

    batch_obs = obs_t.unsqueeze(0)
    batch_mean, batch_log_std, batch_value = agent.model.forward(batch_obs)
    assert tuple(batch_mean.shape) == (1, action_dim)
    assert tuple(batch_log_std.shape) == (1, action_dim)
    assert tuple(batch_value.shape) == (1,)


def _check_action_path(agent, obs: np.ndarray, action_dim: int) -> np.ndarray:
    action, log_prob, value = agent.act(obs)
    assert action.shape == (action_dim,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.isfinite(log_prob)
    assert np.isfinite(value)

    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
    new_log_prob, values, entropy = agent.evaluate_actions(obs_t, action_t)
    assert tuple(new_log_prob.shape) == (1,)
    assert tuple(values.shape) == (1,)
    assert tuple(entropy.shape) == (1,)
    return action


def main() -> None:
    _add_import_paths()

    from proposed.agent.PPO.common.hrl_adapter import make_fast_adapter, make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_buffer import RolloutBuffer
    from proposed.agent.PPO.ppo_config import PPOConfig
    from proposed.config import EnvConfig
    from proposed.env import Env, FastEnv, SlowEnv

    env_cfg = EnvConfig()
    core = Env(env_cfg)
    slow_env = SlowEnv(env_cfg, core_env=core)
    fast_env = FastEnv(env_cfg, core_env=core)

    slow_adapter = make_slow_adapter(slow_env)
    fast_adapter = make_fast_adapter(fast_env)

    ppo_cfg = PPOConfig(
        hidden_dim=32,
        batch_size=2,
        update_epochs=1,
        rollout_steps=2,
        device="cpu",
    )
    slow_agent = PPOAgent(slow_adapter.obs_dim, slow_adapter.action_dim, ppo_cfg)
    fast_agent = PPOAgent(fast_adapter.obs_dim, fast_adapter.action_dim, ppo_cfg)

    slow_obs, slow_info = slow_adapter.reset()
    assert slow_adapter.env.observation_space.contains(core.get_slow_obs())
    _check_forward(slow_agent, slow_obs, slow_adapter.action_dim)
    slow_action = _check_action_path(slow_agent, slow_obs, slow_adapter.action_dim)
    next_slow_obs, slow_reward, slow_terminated, slow_truncated, slow_step_info = slow_adapter.step_vector(slow_action)
    assert next_slow_obs.shape == (slow_adapter.obs_dim,)
    assert slow_reward == 0.0
    assert slow_terminated is False
    assert slow_truncated is False
    assert slow_step_info["round_pending_fast_rollout"] is True

    fast_obs = fast_adapter.flatten_obs(core.get_fast_obs())
    assert fast_adapter.env.observation_space.contains(core.get_fast_obs())
    _check_forward(fast_agent, fast_obs, fast_adapter.action_dim)
    _check_action_path(fast_agent, fast_obs, fast_adapter.action_dim)

    rollout = RolloutBuffer()
    obs = fast_obs
    done = 0.0
    last_info = {}
    for _ in range(2):
        action, log_prob, value = fast_agent.act(obs)
        next_obs, reward, terminated, truncated, info = fast_adapter.step_vector(action)
        assert next_obs.shape == (fast_adapter.obs_dim,)
        assert "dpp_terms" in info
        assert info["dpp_terms"] is info["reward_components"]
        assert info["charge_action_provided"] is False
        done = float(terminated or truncated)
        rollout.add(obs, action, reward, done, value, log_prob)
        obs = next_obs
        last_info = info

    last_value = fast_agent.value(obs)
    rollout.compute_return_and_advantages(
        last_value=last_value,
        last_done=done,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )
    batch = rollout.get_tensors(torch.device("cpu"))
    assert tuple(batch["obs"].shape) == (2, fast_adapter.obs_dim)
    assert tuple(batch["actions"].shape) == (2, fast_adapter.action_dim)

    stats = fast_agent.update(rollout)
    for value in stats.values():
        assert np.isfinite(value)

    assert "uav_charge_effective" in last_info
    print("ppo hrl smoke test passed")


if __name__ == "__main__":
    main()
