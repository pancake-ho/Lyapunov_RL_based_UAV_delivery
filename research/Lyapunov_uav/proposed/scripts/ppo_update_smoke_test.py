from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_log(record: Dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True))


def _assert_obs_action(
    *,
    obs: np.ndarray,
    action: np.ndarray,
    obs_dim: int,
    action_dim: int,
) -> None:
    assert obs.shape == (obs_dim,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert action.shape == (action_dim,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()


def _assert_policy_shapes(agent, obs: np.ndarray, action: np.ndarray) -> Tuple[float, float, float]:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    action_t = torch.as_tensor(action, dtype=torch.float32, device=agent.device).unsqueeze(0)

    mean, log_std, value = agent.model.forward(obs_t)
    assert tuple(mean.shape) == tuple(action_t.shape)
    assert tuple(log_std.shape) == tuple(action_t.shape)
    assert tuple(value.shape) == (1,)

    log_prob, values, entropy = agent.evaluate_actions(obs_t, action_t)
    assert tuple(log_prob.shape) == (1,)
    assert tuple(values.shape) == (1,)
    assert tuple(entropy.shape) == (1,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(entropy).all()
    return float(log_prob.item()), float(values.item()), float(entropy.item())


def _assert_batch_shapes(batch: Dict[str, torch.Tensor], rollout_steps: int, obs_dim: int, action_dim: int) -> None:
    assert tuple(batch["obs"].shape) == (rollout_steps, obs_dim)
    assert tuple(batch["actions"].shape) == (rollout_steps, action_dim)
    assert tuple(batch["log_probs"].shape) == (rollout_steps,)
    assert tuple(batch["returns"].shape) == (rollout_steps,)
    assert tuple(batch["advantages"].shape) == (rollout_steps,)
    assert tuple(batch["values"].shape) == (rollout_steps,)
    for tensor in batch.values():
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def _make_ppo_cfg(*, device: str, hidden_dim: int, rollout_steps: int):
    from proposed.agent.PPO.ppo_config import PPOConfig

    return PPOConfig(
        hidden_dim=hidden_dim,
        batch_size=rollout_steps,
        update_epochs=1,
        rollout_steps=rollout_steps,
        device=device,
    )


def _run_slow_update_smoke(*, cfg, steps: int, device: str, hidden_dim: int) -> Dict[str, Any]:
    from proposed.agent.PPO.common.hrl_adapter import make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_buffer import RolloutBuffer
    from proposed.env import SlowEnv

    slow_env = SlowEnv(cfg)
    adapter = make_slow_adapter(slow_env)
    ppo_cfg = _make_ppo_cfg(device=device, hidden_dim=hidden_dim, rollout_steps=steps)
    agent = PPOAgent(adapter.obs_dim, adapter.action_dim, ppo_cfg)
    rollout = RolloutBuffer()

    obs, reset_info = adapter.reset()
    assert slow_env.observation_space.contains(slow_env.core.get_slow_obs())

    last_done = 0.0
    first_eval = None
    for step_idx in range(steps):
        action, log_prob, value = agent.act(obs)
        _assert_obs_action(obs=obs, action=action, obs_dim=adapter.obs_dim, action_dim=adapter.action_dim)
        eval_log_prob, eval_value, entropy = _assert_policy_shapes(agent, obs, action)
        if first_eval is None:
            first_eval = {
                "eval_log_prob": eval_log_prob,
                "eval_value": eval_value,
                "entropy": entropy,
            }

        next_obs, reward, terminated, truncated, info = adapter.step_vector(action)
        assert next_obs.shape == (adapter.obs_dim,)
        assert next_obs.dtype == np.float32
        assert info["round_pending_fast_rollout"] is True
        done = float(terminated or truncated)
        rollout.add(obs, action, reward, done, value, log_prob)
        obs = next_obs
        last_done = done

    last_value = agent.value(obs)
    rollout.compute_return_and_advantages(
        last_value=last_value,
        last_done=last_done,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )
    batch = rollout.get_tensors(agent.device)
    _assert_batch_shapes(batch, steps, adapter.obs_dim, adapter.action_dim)
    stats = agent.update(rollout)
    assert all(np.isfinite(value) for value in stats.values())

    return {
        "level": "slow",
        "obs_dim": adapter.obs_dim,
        "action_dim": adapter.action_dim,
        "steps": steps,
        "reset_round_slot": int(reset_info["round_slot"]),
        "last_done": float(last_done),
        "advantage_shape": list(np.asarray(rollout.advantages).shape),
        "return_shape": list(np.asarray(rollout.returns).shape),
        "batch_obs_shape": list(batch["obs"].shape),
        "batch_action_shape": list(batch["actions"].shape),
        "first_eval": first_eval,
        "update_stats": stats,
    }


def _run_fast_update_smoke(*, cfg, steps: int, device: str, hidden_dim: int) -> Dict[str, Any]:
    from proposed.agent.PPO.common.hrl_adapter import make_fast_adapter, make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_buffer import RolloutBuffer
    from proposed.env import Env, FastEnv, SlowEnv

    core = Env(cfg)
    slow_env = SlowEnv(cfg, core_env=core)
    fast_env = FastEnv(cfg, core_env=core)
    slow_adapter = make_slow_adapter(slow_env)
    fast_adapter = make_fast_adapter(fast_env)
    ppo_cfg = _make_ppo_cfg(device=device, hidden_dim=hidden_dim, rollout_steps=steps)
    slow_agent = PPOAgent(slow_adapter.obs_dim, slow_adapter.action_dim, ppo_cfg)
    fast_agent = PPOAgent(fast_adapter.obs_dim, fast_adapter.action_dim, ppo_cfg)
    rollout = RolloutBuffer()

    slow_obs, reset_info = slow_adapter.reset()
    slow_action, _, _ = slow_agent.act(slow_obs)
    _, slow_reward, slow_terminated, slow_truncated, slow_info = slow_adapter.step_vector(slow_action)
    assert slow_reward == 0.0
    assert slow_terminated is False
    assert slow_truncated is False
    assert slow_info["round_pending_fast_rollout"] is True

    obs = fast_adapter.flatten_obs(core.get_fast_obs())
    assert fast_env.observation_space.contains(core.get_fast_obs())

    last_done = 0.0
    first_eval = None
    last_info: Dict[str, Any] = {}
    for step_idx in range(steps):
        action, log_prob, value = fast_agent.act(obs)
        _assert_obs_action(obs=obs, action=action, obs_dim=fast_adapter.obs_dim, action_dim=fast_adapter.action_dim)
        eval_log_prob, eval_value, entropy = _assert_policy_shapes(fast_agent, obs, action)
        if first_eval is None:
            first_eval = {
                "eval_log_prob": eval_log_prob,
                "eval_value": eval_value,
                "entropy": entropy,
            }

        next_obs, reward, terminated, truncated, info = fast_adapter.step_vector(action)
        assert next_obs.shape == (fast_adapter.obs_dim,)
        assert next_obs.dtype == np.float32
        assert "dpp_terms" in info
        assert info["dpp_terms"] is info["reward_components"]
        assert info["charge_action_provided"] is False
        done = float(terminated or truncated)
        rollout.add(obs, action, reward, done, value, log_prob)
        obs = next_obs
        last_done = done
        last_info = info

    last_value = fast_agent.value(obs)
    rollout.compute_return_and_advantages(
        last_value=last_value,
        last_done=last_done,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )
    batch = rollout.get_tensors(fast_agent.device)
    _assert_batch_shapes(batch, steps, fast_adapter.obs_dim, fast_adapter.action_dim)
    stats = fast_agent.update(rollout)
    assert all(np.isfinite(value) for value in stats.values())

    dpp_terms = last_info.get("dpp_terms", {})
    return {
        "level": "fast",
        "obs_dim": fast_adapter.obs_dim,
        "action_dim": fast_adapter.action_dim,
        "steps": steps,
        "reset_round_slot": int(reset_info["round_slot"]),
        "last_done": float(last_done),
        "advantage_shape": list(np.asarray(rollout.advantages).shape),
        "return_shape": list(np.asarray(rollout.returns).shape),
        "batch_obs_shape": list(batch["obs"].shape),
        "batch_action_shape": list(batch["actions"].shape),
        "first_eval": first_eval,
        "last_dpp_reward": float(dpp_terms.get("total_dpp_reward", 0.0)),
        "last_terminated": bool(last_info.get("terminated", False)),
        "last_truncated": bool(last_info.get("truncated", False)),
        "charge_action_provided": bool(last_info.get("charge_action_provided", True)),
        "update_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PPO update for slow and fast HRL smoke paths.")
    parser.add_argument("--slow-steps", type=int, default=4)
    parser.add_argument("--fast-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but is not available.")
    if args.slow_steps <= 1 or args.fast_steps <= 1:
        raise ValueError("slow-steps and fast-steps must be greater than 1 for PPO update smoke.")

    _add_import_paths()
    from proposed.config import EnvConfig

    _set_seed(args.seed)
    cfg = EnvConfig()
    cfg.seed = int(args.seed)

    slow_result = _run_slow_update_smoke(
        cfg=cfg,
        steps=int(args.slow_steps),
        device=str(args.device),
        hidden_dim=int(args.hidden_dim),
    )
    _json_log({"event": "ppo_update_smoke", **slow_result})

    _set_seed(args.seed)
    cfg = EnvConfig()
    cfg.seed = int(args.seed)
    fast_result = _run_fast_update_smoke(
        cfg=cfg,
        steps=int(args.fast_steps),
        device=str(args.device),
        hidden_dim=int(args.hidden_dim),
    )
    _json_log({"event": "ppo_update_smoke", **fast_result})


if __name__ == "__main__":
    main()
