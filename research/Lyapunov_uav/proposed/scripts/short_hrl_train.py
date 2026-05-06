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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested but is not available.")
    return device


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


def _reward_metadata(cfg) -> Dict[str, Any]:
    return {
        "reward_preset": str(cfg.reward.preset_name),
        "reward_coefficients": cfg.reward_coefficients(),
    }


def _json_line(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


def _make_ppo_cfg(*, device: str, hidden_dim: int, rollout_steps: int):
    from proposed.agent.PPO.ppo_config import PPOConfig

    return PPOConfig(
        hidden_dim=hidden_dim,
        batch_size=rollout_steps,
        update_epochs=1,
        rollout_steps=rollout_steps,
        device=device,
    )


def _save_checkpoint(path: Path, *, fast_agent, slow_agent, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fast_model": fast_agent.model.state_dict(),
            "fast_optimizer": fast_agent.optimizer.state_dict(),
            "slow_model": slow_agent.model.state_dict(),
            "slow_optimizer": slow_agent.optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def run_short_train(args: argparse.Namespace) -> Path:
    from proposed.agent.PPO.common.hrl_adapter import make_fast_adapter, make_slow_adapter
    from proposed.agent.PPO.ppo_agent import PPOAgent
    from proposed.agent.PPO.ppo_buffer import RolloutBuffer
    from proposed.config import EnvConfig
    from proposed.env import Env, FastEnv, SlowEnv

    device = _resolve_device(str(args.device))
    _set_seed(int(args.seed))

    log_dir = Path(args.log_dir).expanduser().resolve()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{args.run_name}.jsonl"
    if log_path.exists() and not args.append_log:
        log_path.unlink()

    cfg = EnvConfig()
    cfg.seed = int(args.seed)
    cfg.set_reward_preset(str(args.reward_preset))

    core = Env(cfg)
    slow_env = SlowEnv(cfg, core_env=core)
    fast_env = FastEnv(cfg, core_env=core)
    slow_adapter = make_slow_adapter(slow_env)
    fast_adapter = make_fast_adapter(fast_env)

    rollout_steps = max(2, int(args.rollout_steps))
    ppo_cfg = _make_ppo_cfg(device=device, hidden_dim=int(args.hidden_dim), rollout_steps=rollout_steps)
    slow_agent = PPOAgent(slow_adapter.obs_dim, slow_adapter.action_dim, ppo_cfg)
    fast_agent = PPOAgent(fast_adapter.obs_dim, fast_adapter.action_dim, ppo_cfg)

    slow_obs, reset_info = slow_adapter.reset()
    fast_obs = fast_adapter.flatten_obs(core.get_fast_obs())
    assert fast_obs.dtype == np.float32

    _json_line(
        log_path,
        {
            "event": "short_train_start",
            "run_name": str(args.run_name),
            "seed": int(args.seed),
            "device": device,
            "max_steps": int(args.max_steps),
            "max_updates": int(args.max_updates),
            "rollout_steps": rollout_steps,
            "slow_obs_dim": slow_adapter.obs_dim,
            "slow_action_dim": slow_adapter.action_dim,
            "fast_obs_dim": fast_adapter.obs_dim,
            "fast_action_dim": fast_adapter.action_dim,
            "reset_info": reset_info,
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            **_reward_metadata(cfg),
        },
    )

    global_step = 0
    update_idx = 0
    last_slow_reward = 0.0
    active_slow_summary = {
        "rsu_scheduled_links": 0,
        "uav_hired": 0,
        "uav_scheduled_links": 0,
    }

    while global_step < int(args.max_steps) and update_idx < int(args.max_updates):
        rollout = RolloutBuffer()
        rollout_fast_reward = 0.0
        last_info: Dict[str, Any] = {}
        rollout_start_step = global_step

        for _ in range(rollout_steps):
            if global_step >= int(args.max_steps):
                break

            if int(core.round_slot) == 0:
                slow_obs = slow_adapter.flatten_obs(core.get_slow_obs())
                slow_action, _, _ = slow_agent.act(slow_obs)
                decoded_slow = slow_adapter.decode_action(slow_action)
                _, slow_reward, slow_terminated, slow_truncated, slow_info = slow_adapter.step_vector(slow_action)
                assert slow_terminated is False
                assert slow_truncated is False
                last_slow_reward = float(slow_reward)
                active_slow_summary = {
                    "rsu_scheduled_links": int(np.asarray(decoded_slow["rsu_scheduling"]).sum()),
                    "uav_hired": int(np.asarray(decoded_slow["uav_hiring"]).sum()),
                    "uav_scheduled_links": int(np.asarray(decoded_slow["uav_scheduling"]).sum()),
                }
                assert slow_info["round_pending_fast_rollout"] is True

            fast_obs = fast_adapter.flatten_obs(core.get_fast_obs())
            action, log_prob, value = fast_agent.act(fast_obs)
            assert fast_obs.shape == (fast_adapter.obs_dim,)
            assert fast_obs.dtype == np.float32
            assert action.shape == (fast_adapter.action_dim,)
            assert action.dtype == np.float32

            next_obs, reward, terminated, truncated, info = fast_adapter.step_vector(action)
            assert "dpp_terms" in info
            assert info["dpp_terms"] is info["reward_components"]
            done = float(terminated or truncated)
            rollout.add(fast_obs, action, reward, done, value, log_prob)

            rollout_fast_reward += float(reward)
            fast_obs = next_obs
            last_info = info
            global_step += 1

        if len(rollout) == 0:
            break

        last_value = fast_agent.value(fast_obs)
        last_done = float(bool(last_info.get("terminated", False) or last_info.get("truncated", False)))
        rollout.compute_return_and_advantages(
            last_value=last_value,
            last_done=last_done,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )
        stats = fast_agent.update(rollout)
        update_idx += 1

        total_reward = float(last_slow_reward + rollout_fast_reward)
        record = {
            "event": "short_train_update",
            "run_name": str(args.run_name),
            "step": int(global_step),
            "update": int(update_idx),
            "rollout_start_step": int(rollout_start_step),
            "rollout_len": int(len(rollout)),
            "slow_reward": float(last_slow_reward),
            "fast_reward": float(rollout_fast_reward),
            "total_reward": total_reward,
            "dpp_terms": _dpp_summary(last_info),
            "queue": _array_summary(last_info.get("next_Q", core.queue)),
            "virtual_queue": _array_summary(last_info.get("next_Z", core.Z)),
            "uav_soc": _array_summary(last_info.get("next_E", core.E)),
            "uav_virtual_queue": _array_summary(last_info.get("next_Y", core.Y)),
            "slow_action_summary": active_slow_summary,
            "terminated": bool(last_info.get("terminated", False)),
            "truncated": bool(last_info.get("truncated", False)),
            "actor_loss": float(stats["policy_loss"]),
            "critic_loss": float(stats["value_loss"]),
            "entropy": float(stats["entropy"]),
            "approx_kl": float(stats["approx_kl_div"]),
            "clip_frac": float(stats["clip_frac"]),
            **_reward_metadata(cfg),
        }
        _json_line(log_path, record)

        if bool(args.save_checkpoint) and (
            update_idx == 1 or update_idx % max(1, int(args.checkpoint_interval)) == 0
        ):
            ckpt_path = checkpoint_dir / f"{args.run_name}_update{update_idx:04d}.pt"
            _save_checkpoint(
                ckpt_path,
                fast_agent=fast_agent,
                slow_agent=slow_agent,
                meta={
                    "run_name": str(args.run_name),
                    "step": int(global_step),
                    "update": int(update_idx),
                    "seed": int(args.seed),
                    "device": device,
                    "log_path": str(log_path),
                    **_reward_metadata(cfg),
                },
            )
            _json_line(
                log_path,
                {
                    "event": "checkpoint_saved",
                    "run_name": str(args.run_name),
                    "step": int(global_step),
                    "update": int(update_idx),
                    "checkpoint_path": str(ckpt_path),
                },
            )

    _json_line(
        log_path,
        {
            "event": "short_train_done",
            "run_name": str(args.run_name),
            "step": int(global_step),
            "updates": int(update_idx),
        },
    )
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Short HRL PPO train run for pre-Seraph validation.")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--log-dir", type=str, default="Lyapunov_uav/proposed/outputs/logs")
    parser.add_argument("--checkpoint-dir", type=str, default="Lyapunov_uav/proposed/outputs/checkpoints")
    parser.add_argument("--run-name", type=str, default="short_hrl_train")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--reward-preset", type=str, default="balanced")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive.")
    if args.max_updates <= 0:
        raise ValueError("max-updates must be positive.")
    if args.rollout_steps <= 1:
        raise ValueError("rollout-steps must be greater than 1.")

    _add_import_paths()
    log_path = run_short_train(args)
    print(json.dumps({"event": "short_train_log_path", "log_path": str(log_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
