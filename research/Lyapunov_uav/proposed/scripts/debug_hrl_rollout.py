from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

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


def _array_summary(value: Any) -> Dict[str, float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "sum": 0.0}
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _slow_action_summary(action: Dict[str, Any]) -> Dict[str, int]:
    return {
        "rsu_scheduled_links": int(np.asarray(action["rsu_scheduling"]).sum()),
        "uav_hired": int(np.asarray(action["uav_hiring"]).sum()),
        "uav_scheduled_links": int(np.asarray(action["uav_scheduling"]).sum()),
    }


def _fast_action_summary(action: Dict[str, Any]) -> Dict[str, float]:
    rsu_chunks = np.asarray(action["rsu_chunks"], dtype=np.float32)
    uav_chunks = np.asarray(action["uav_chunks"], dtype=np.float32)
    uav_power = np.asarray(action["uav_power"], dtype=np.float32)
    return {
        "rsu_active_links": int((rsu_chunks > 0.0).sum()),
        "rsu_chunks_sum": float(rsu_chunks.sum()),
        "uav_active_links": int((uav_chunks > 0.0).sum()),
        "uav_chunks_sum": float(uav_chunks.sum()),
        "uav_power_sum": float(uav_power.sum()),
        "uav_power_max": float(uav_power.max()) if uav_power.size else 0.0,
    }


def _dpp_summary(info: Dict[str, Any]) -> Dict[str, float]:
    terms = info.get("dpp_terms", {}) or {}
    keys = (
        "video_delivery_pressure",
        "quality_reward",
        "battery_service_pressure",
        "charging_effect",
        "total_dpp_reward",
    )
    return {key: float(terms.get(key, 0.0)) for key in keys}


def _policy_probe(agent, obs: np.ndarray, action_dim: int):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    mean, log_std, value = agent.model.forward(obs_t)
    assert tuple(mean.shape) == (action_dim,)
    assert tuple(log_std.shape) == (action_dim,)
    assert value.ndim == 0

    action, log_prob, sampled_value = agent.act(obs)
    assert action.shape == (action_dim,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.isfinite(log_prob)
    assert np.isfinite(sampled_value)
    return action, log_prob, sampled_value


def _log(record: Dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True))


def _run_slow_rollout(*, cfg, steps: int, hidden_dim: int):
    from proposed.agent.PPO.common.hrl_adapter import make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_config import PPOConfig
    from proposed.env import SlowEnv

    slow_env = SlowEnv(cfg)
    slow_adapter = make_slow_adapter(slow_env)
    slow_agent = PPOAgent(
        slow_adapter.obs_dim,
        slow_adapter.action_dim,
        PPOConfig(hidden_dim=hidden_dim, device="cpu"),
    )

    obs, reset_info = slow_adapter.reset()
    _log(
        {
            "event": "slow_reset",
            "obs_dim": slow_adapter.obs_dim,
            "action_dim": slow_adapter.action_dim,
            "reset_info": reset_info,
            "queue": _array_summary(slow_env.core.queue),
            "virtual_queue": _array_summary(slow_env.core.Z),
            "uav_soc": _array_summary(slow_env.core.E),
            "uav_virtual_queue": _array_summary(slow_env.core.Y),
        }
    )

    for step_idx in range(steps):
        action_vec, log_prob, value = _policy_probe(slow_agent, obs, slow_adapter.action_dim)
        decoded_action = slow_adapter.decode_action(action_vec)
        next_obs, reward, terminated, truncated, info = slow_adapter.step_vector(action_vec)
        assert next_obs.shape == (slow_adapter.obs_dim,)
        _log(
            {
                "event": "slow_step",
                "step": step_idx,
                "slow_action_summary": _slow_action_summary(decoded_action),
                "reward": float(reward),
                "queue": _array_summary(slow_env.core.queue),
                "virtual_queue": _array_summary(slow_env.core.Z),
                "uav_soc": _array_summary(slow_env.core.E),
                "uav_virtual_queue": _array_summary(slow_env.core.Y),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "log_prob": float(log_prob),
                "value": float(value),
                "round_idx": int(info["round_idx"]),
                "round_slot": int(info["round_slot"]),
            }
        )
        obs = next_obs


def _run_hrl_fast_rollout(*, cfg, steps: int, hidden_dim: int):
    from proposed.agent.PPO.common.hrl_adapter import make_fast_adapter, make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_config import PPOConfig
    from proposed.env import Env, FastEnv, SlowEnv

    core = Env(cfg)
    slow_env = SlowEnv(cfg, core_env=core)
    fast_env = FastEnv(cfg, core_env=core)
    slow_adapter = make_slow_adapter(slow_env)
    fast_adapter = make_fast_adapter(fast_env)
    ppo_cfg = PPOConfig(hidden_dim=hidden_dim, device="cpu")
    slow_agent = PPOAgent(slow_adapter.obs_dim, slow_adapter.action_dim, ppo_cfg)
    fast_agent = PPOAgent(fast_adapter.obs_dim, fast_adapter.action_dim, ppo_cfg)

    slow_obs, reset_info = slow_adapter.reset()
    fast_obs = fast_adapter.flatten_obs(core.get_fast_obs())
    _log(
        {
            "event": "hrl_reset",
            "slow_obs_dim": slow_adapter.obs_dim,
            "slow_action_dim": slow_adapter.action_dim,
            "fast_obs_dim": fast_adapter.obs_dim,
            "fast_action_dim": fast_adapter.action_dim,
            "reset_info": reset_info,
            "queue": _array_summary(core.queue),
            "virtual_queue": _array_summary(core.Z),
            "uav_soc": _array_summary(core.E),
            "uav_virtual_queue": _array_summary(core.Y),
        }
    )

    active_slow_summary: Dict[str, int] = {
        "rsu_scheduled_links": 0,
        "uav_hired": 0,
        "uav_scheduled_links": 0,
    }

    for step_idx in range(steps):
        if int(core.round_slot) == 0:
            slow_obs = slow_adapter.flatten_obs(core.get_slow_obs())
            slow_action_vec, slow_log_prob, slow_value = _policy_probe(
                slow_agent,
                slow_obs,
                slow_adapter.action_dim,
            )
            decoded_slow_action = slow_adapter.decode_action(slow_action_vec)
            next_slow_obs, slow_reward, slow_terminated, slow_truncated, slow_info = slow_adapter.step_vector(
                slow_action_vec
            )
            active_slow_summary = _slow_action_summary(decoded_slow_action)
            assert next_slow_obs.shape == (slow_adapter.obs_dim,)
            _log(
                {
                    "event": "slow_decision",
                    "step": step_idx,
                    "slow_action_summary": active_slow_summary,
                    "reward": float(slow_reward),
                    "queue": _array_summary(core.queue),
                    "virtual_queue": _array_summary(core.Z),
                    "uav_soc": _array_summary(core.E),
                    "uav_virtual_queue": _array_summary(core.Y),
                    "terminated": bool(slow_terminated),
                    "truncated": bool(slow_truncated),
                    "log_prob": float(slow_log_prob),
                    "value": float(slow_value),
                    "round_idx": int(slow_info["round_idx"]),
                    "round_slot": int(slow_info["round_slot"]),
                }
            )

        fast_obs = fast_adapter.flatten_obs(core.get_fast_obs())
        fast_action_vec, fast_log_prob, fast_value = _policy_probe(
            fast_agent,
            fast_obs,
            fast_adapter.action_dim,
        )
        decoded_fast_action = fast_adapter.decode_action(fast_action_vec)
        next_fast_obs, reward, terminated, truncated, info = fast_adapter.step_vector(fast_action_vec)
        assert next_fast_obs.shape == (fast_adapter.obs_dim,)
        assert "dpp_terms" in info
        assert info["dpp_terms"] is info["reward_components"]
        assert info["charge_action_provided"] is False

        _log(
            {
                "event": "fast_step",
                "step": step_idx,
                "slow_action_summary": active_slow_summary,
                "fast_action_summary": _fast_action_summary(decoded_fast_action),
                "reward": float(reward),
                "dpp_terms": _dpp_summary(info),
                "queue": _array_summary(info["next_Q"]),
                "virtual_queue": _array_summary(info["next_Z"]),
                "uav_soc": _array_summary(info["next_E"]),
                "uav_virtual_queue": _array_summary(info["next_Y"]),
                "uav_charge_effective_sum": int(np.asarray(info["uav_charge_effective"]).sum()),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "log_prob": float(fast_log_prob),
                "value": float(fast_value),
                "round_idx": int(info["next_round_idx"]),
                "round_slot": int(info["next_round_slot"]),
            }
        )
        fast_obs = next_fast_obs


def main() -> None:
    parser = argparse.ArgumentParser(description="Short reproducible HRL PPO debug rollout.")
    parser.add_argument("--slow-steps", type=int, default=5)
    parser.add_argument("--fast-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=32)
    args = parser.parse_args()

    if args.slow_steps <= 0 or args.fast_steps <= 0:
        raise ValueError("slow-steps and fast-steps must be positive.")

    _add_import_paths()
    from proposed.config import EnvConfig

    _set_seed(args.seed)
    cfg = EnvConfig()
    cfg.seed = int(args.seed)

    _run_slow_rollout(cfg=cfg, steps=int(args.slow_steps), hidden_dim=int(args.hidden_dim))
    _run_hrl_fast_rollout(cfg=cfg, steps=int(args.fast_steps), hidden_dim=int(args.hidden_dim))


if __name__ == "__main__":
    main()
