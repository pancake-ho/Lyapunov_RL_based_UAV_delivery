from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import EnvConfig  # noqa: E402
from env.env import Env  # noqa: E402
from agent.PPO.common import (  # noqa: E402
    ScalarLogger,
    infer_fast_obs_dim,
    infer_slow_obs_dim,
    set_seed,
    split_env_reset,
    split_env_step,
)
from agent.PPO.common.utils import save_json  # noqa: E402
from agent.PPO.fast.fast_agent import (  # noqa: E402
    FastPPOAgent,
    FastPPOConfig,
)
from agent.PPO.slow.slow_agent import SlowPPOAgent  # noqa: E402
from agent.PPO.slow.slow_config import (  # noqa: E402
    SlowTrainConfig,
    get_slow_train_config,
)
from agent.PPO.slow.slow_metrics import HRLMetrics  # noqa: E402


def _resolve_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = PROPOSED_ROOT / result
    return result.resolve()


def _make_env_config(
    *, seed: int, move_prob: float, rounds_per_episode: int
) -> EnvConfig:
    base = EnvConfig()
    return replace(
        base,
        seed=int(seed),
        move_prob=float(move_prob),
        episode_slots=int(base.slow_T) * int(rounds_per_episode),
    )


def _load_trusted_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    """
    Read metadata from a checkpoint produced by this repository.

    weights_only=False is intentional only for the user's own trusted
    checkpoints because extra may contain NumPy arrays.
    """
    try:
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch before weights_only
        checkpoint = torch.load(path, map_location="cpu")
    return dict(checkpoint.get("extra", {}))


def _build_frozen_fast_agent(
    train_cfg: SlowTrainConfig,
    env_cfg: EnvConfig,
    sample_fast_obs: Dict[str, Any],
) -> FastPPOAgent:
    checkpoint_path = _resolve_path(train_cfg.fast_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"fast checkpoint does not exist: {checkpoint_path}"
        )

    extra = _load_trusted_checkpoint_metadata(checkpoint_path)
    saved_config = dict(extra.get("fast_ppo_config", {}))
    allowed = {field.name for field in fields(FastPPOConfig)}
    config_values = {
        key: value for key, value in saved_config.items() if key in allowed
    }
    config_values["device"] = str(train_cfg.device)
    config_values.setdefault(
        "hidden_dims",
        tuple(int(x) for x in train_cfg.fast_hidden_dims_fallback),
    )
    config_values.setdefault(
        "init_log_std", float(train_cfg.fast_init_log_std_fallback)
    )
    fast_ppo_config = FastPPOConfig(**config_values)

    obs_dim = infer_fast_obs_dim(sample_fast_obs)
    saved_obs_dim = int(extra.get("obs_dim", obs_dim))
    if saved_obs_dim != obs_dim:
        raise ValueError(
            "fast checkpoint/environment observation mismatch: "
            f"checkpoint={saved_obs_dim}, environment={obs_dim}"
        )

    agent = FastPPOAgent(
        env_cfg=env_cfg, obs_dim=obs_dim, ppo_cfg=fast_ppo_config
    )
    checkpoint = agent.load(
        checkpoint_path, strict=True, load_optimizer=False
    )
    loaded_extra = checkpoint.get("extra", {})
    if agent.obs_normalizer is not None and "obs_normalizer" not in loaded_extra:
        raise RuntimeError(
            "fast checkpoint has no observation-normalizer state. "
            "A normalized fast policy cannot be evaluated safely without it."
        )
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    return agent


def _build_slow_agent(
    train_cfg: SlowTrainConfig,
    env_cfg: EnvConfig,
    sample_slow_obs: Dict[str, Any],
) -> Tuple[SlowPPOAgent, int, int, int]:
    agent = SlowPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_slow_obs_dim(sample_slow_obs),
        ppo_cfg=train_cfg.make_slow_ppo_config(),
    )
    start_episode = 0
    global_round = 0
    update_index = 0
    if train_cfg.resume_slow or train_cfg.mode == "eval":
        if train_cfg.slow_checkpoint is None:
            raise ValueError("slow_checkpoint is required.")
        checkpoint = agent.load(
            _resolve_path(train_cfg.slow_checkpoint),
            strict=True,
            load_optimizer=bool(train_cfg.resume_slow),
        )
        extra = checkpoint.get("extra", {})
        start_episode = int(extra.get("episode", 0))
        global_round = int(extra.get("global_round", 0))
        update_index = int(extra.get("update", 0))
    return agent, start_episode, global_round, update_index


def _select_random_slow_action(
    slow_agent: SlowPPOAgent,
    env: Env,
    slow_obs: Dict[str, Any],
    rng: np.random.Generator,
    train_cfg: SlowTrainConfig,
) -> Dict[str, Any]:
    raw = slow_agent.codec.random_binary_action(
        rng=rng,
        rsu_user_prob=float(train_cfg.baseline_rsu_user_prob),
        uav_hire_prob=float(train_cfg.baseline_uav_hire_prob),
        uav_user_prob=float(train_cfg.baseline_uav_user_prob),
    )
    mask = slow_agent.codec.build_effective_action_mask(
        raw, slow_obs, context=env
    )
    action = raw * mask
    env_action, action_info = slow_agent.codec.decode_with_info(
        action, slow_obs, context=env
    )
    if int(action_info["projection_count"]) != 0:
        raise RuntimeError("random masked slow baseline required projection.")
    return {
        "binary_action": action.astype(np.float32),
        "raw_action": action.astype(np.float32),
        "action_mask": mask.astype(np.float32),
        "env_action": env_action,
        "action_info": action_info,
    }


def _run_fast_round(
    env: Env,
    fast_agent: FastPPOAgent,
    *,
    deterministic: bool,
) -> Tuple[HRLMetrics, Dict[str, Any], bool, bool]:
    fast_obs = env.get_fast_obs()
    metrics = HRLMetrics(
        fast_layer_ratios=np.zeros(int(env.cfg.layer), dtype=np.float64)
    )
    boundary_info: Optional[Dict[str, Any]] = None
    terminated = False
    truncated = False

    for _ in range(int(env.slow_T)):
        fast_selected = fast_agent.select_action(
            fast_obs,
            deterministic=bool(deterministic),
            update_norm=False,
        )
        next_obs, _, terminated, truncated, info = split_env_step(
            env.step(fast_selected["env_action"])
        )
        metrics.add_slot(
            info, fast_selected, num_layers=int(env.cfg.layer)
        )
        fast_obs = next_obs
        if bool(info.get("is_round_boundary", False)):
            boundary_info = info
            break
        if terminated or truncated:
            raise RuntimeError(
                "episode ended before a complete slow round; "
                "episode_slots must be a multiple of slow_T."
            )

    if boundary_info is None:
        raise RuntimeError("fast loop did not reach a slow round boundary.")
    if metrics.slots != int(env.slow_T):
        raise RuntimeError(
            f"incomplete round: expected={env.slow_T}, got={metrics.slots}"
        )
    return metrics, boundary_info, bool(terminated), bool(truncated)


def _extract_slow_reward(
    boundary_info: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    reward_components = boundary_info.get("reward_components", {})
    slow_reward = float(reward_components.get("slow_reward", 0.0))
    slow_components = dict(
        reward_components.get("slow_reward_components", {})
    )
    if not bool(slow_components.get("is_round_boundary", False)):
        raise RuntimeError("boundary info has no completed slow reward.")
    component_reward = float(slow_components.get("slow_reward", slow_reward))
    if not np.isclose(slow_reward, component_reward, rtol=1e-6, atol=1e-3):
        raise RuntimeError("slow reward fields disagree at round boundary.")
    return slow_reward, slow_components


def evaluate_paired(
    train_cfg: SlowTrainConfig,
    fast_agent: FastPPOAgent,
    slow_agent: SlowPPOAgent,
    *,
    random_slow: bool,
) -> Dict[str, float]:
    aggregate = HRLMetrics(
        fast_layer_ratios=np.zeros(4, dtype=np.float64)
    )
    for seed in train_cfg.eval_seeds:
        env_cfg = _make_env_config(
            seed=int(seed),
            move_prob=float(train_cfg.eval_move_prob),
            rounds_per_episode=int(train_cfg.eval_rounds_per_seed),
        )
        env = Env(env_cfg)
        split_env_reset(env.reset())
        rng = np.random.default_rng(int(seed) + 100_000)
        seed_metrics = HRLMetrics(
            fast_layer_ratios=np.zeros(int(env_cfg.layer), dtype=np.float64)
        )

        for _ in range(int(train_cfg.eval_rounds_per_seed)):
            slow_obs = env.get_slow_obs()
            if random_slow:
                slow_selected = _select_random_slow_action(
                    slow_agent, env, slow_obs, rng, train_cfg
                )
            else:
                slow_selected = slow_agent.select_action(
                    slow_obs,
                    context=env,
                    deterministic=True,
                    update_norm=False,
                )
            env.apply_slow_action(slow_selected["env_action"])
            round_metrics, info, terminated, truncated = _run_fast_round(
                env,
                fast_agent,
                deterministic=bool(train_cfg.fast_deterministic_eval),
            )
            slow_reward, slow_components = _extract_slow_reward(info)
            round_metrics.add_round(
                slow_reward, slow_components, slow_selected
            )
            seed_metrics.merge(round_metrics)
            if terminated or truncated:
                break
        aggregate.merge(seed_metrics)
    return aggregate.summary()


def _checkpoint_is_eligible(
    candidate: Dict[str, float],
    baseline: Dict[str, float],
    train_cfg: SlowTrainConfig,
    env_cfg: EnvConfig,
) -> Tuple[bool, Dict[str, float]]:
    min_reward = float(baseline["slow_reward_per_round"]) + (
        abs(float(baseline["slow_reward_per_round"]))
        * float(train_cfg.min_reward_improvement_fraction)
    )
    min_delivery = (
        float(baseline["delivery_per_slot"])
        * float(train_cfg.min_delivery_fraction_of_baseline)
    )
    max_degradation = (
        float(baseline["quality_degradation_per_chunk"])
        * float(train_cfg.max_degradation_ratio_to_baseline)
    )
    max_scheduled_stall = max(
        float(baseline["scheduled_stall_rate"])
        * float(train_cfg.max_scheduled_stall_ratio_to_baseline),
        float(baseline["scheduled_stall_rate"])
        + float(train_cfg.scheduled_stall_absolute_tolerance),
    )
    soc_floor = float(env_cfg.battery.e_min) - 0.1
    checks = {
        "reward_ok": float(candidate["slow_reward_per_round"] >= min_reward),
        "delivery_ok": float(candidate["delivery_per_slot"] >= min_delivery),
        "degradation_ok": float(
            baseline["quality_degradation_per_chunk"] <= 0.0
            or candidate["quality_degradation_per_chunk"] <= max_degradation
        ),
        "scheduled_stall_ok": float(
            candidate["scheduled_stall_rate"] <= max_scheduled_stall
        ),
        "outage_ok": float(candidate["outage_slots"] <= 0.0),
        "soc_ok": float(candidate["min_soc"] >= soc_floor),
        "min_delivery_per_slot": float(min_delivery),
        "max_degradation_per_chunk": float(max_degradation),
        "max_scheduled_stall_rate": float(max_scheduled_stall),
        "soc_floor": float(soc_floor),
        "min_slow_reward_per_round": float(min_reward),
    }
    eligible = all(
        bool(checks[name])
        for name in (
            "reward_ok", "delivery_ok", "degradation_ok", "scheduled_stall_ok",
            "outage_ok", "soc_ok",
        )
    )
    return eligible, checks


def _save_slow_checkpoint(
    slow_agent: SlowPPOAgent,
    path: Path,
    *,
    episode: int,
    global_slot: int,
    global_round: int,
    update_index: int,
    fast_checkpoint: str,
) -> None:
    slow_agent.save(
        path,
        extra={
            "episode": int(episode),
            "global_slot": int(global_slot),
            "global_round": int(global_round),
            "update": int(update_index),
            "fast_checkpoint": str(fast_checkpoint),
            "fast_policy_frozen": True,
        },
    )


def train(train_cfg: SlowTrainConfig) -> None:
    set_seed(train_cfg.seed, deterministic=train_cfg.deterministic_torch)
    env_cfg = _make_env_config(
        seed=train_cfg.seed,
        move_prob=train_cfg.train_move_prob,
        rounds_per_episode=train_cfg.rounds_per_episode,
    )
    probe_env = Env(env_cfg)
    sample_fast_obs, _ = split_env_reset(probe_env.reset())
    sample_slow_obs = probe_env.get_slow_obs()

    fast_agent = _build_frozen_fast_agent(
        train_cfg, env_cfg, sample_fast_obs
    )
    slow_agent, start_episode, global_round, update_index = _build_slow_agent(
        train_cfg, env_cfg, sample_slow_obs
    )
    env = Env(env_cfg)
    global_slot = global_round * int(env_cfg.slow_T)

    run_dir = _resolve_path(Path(train_cfg.output_root) / train_cfg.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(train_cfg.to_dict(), run_dir / "train_config.json")
    save_json(env_cfg.as_dict(), run_dir / "env_config.json")

    episode_logger = ScalarLogger(run_dir / "episode_metrics.csv")
    round_logger = ScalarLogger(run_dir / "round_metrics.csv")
    update_logger = ScalarLogger(run_dir / "update_metrics.csv")
    eval_logger = ScalarLogger(run_dir / "evaluation_metrics.csv")

    baseline = evaluate_paired(
        train_cfg, fast_agent, slow_agent, random_slow=True
    )
    save_json(baseline, run_dir / "paired_random_slow_baseline.json")
    best_eligible_reward = -np.inf

    print("=" * 88)
    print("[SLOW HRL TRAIN START]")
    print(f"run_dir        : {run_dir}")
    print(f"fast_checkpoint: {_resolve_path(train_cfg.fast_checkpoint)}")
    print(f"fast_frozen    : True")
    print(f"fast_det_train : {train_cfg.fast_deterministic_train}")
    print(f"slow_obs_dim   : {slow_agent.obs_dim}")
    print(f"slow_action_dim: {slow_agent.action_dim}")
    print(f"rollout_rounds : {train_cfg.rollout_rounds}")
    print(f"move_prob      : {train_cfg.train_move_prob}")
    print("=" * 88)

    last_next_slow_obs = sample_slow_obs
    last_done = True
    for episode in range(start_episode + 1, int(train_cfg.num_episodes) + 1):
        split_env_reset(env.reset())
        slow_obs = env.get_slow_obs()
        episode_metrics = HRLMetrics(
            fast_layer_ratios=np.zeros(int(env_cfg.layer), dtype=np.float64)
        )

        for local_round in range(int(train_cfg.rounds_per_episode)):
            slow_selected = slow_agent.select_action(
                slow_obs,
                context=env,
                deterministic=False,
                update_norm=True,
            )
            env.apply_slow_action(slow_selected["env_action"])
            round_metrics, info, terminated, truncated = _run_fast_round(
                env,
                fast_agent,
                deterministic=bool(train_cfg.fast_deterministic_train),
            )
            global_slot += round_metrics.slots
            global_round += 1
            slow_reward, slow_components = _extract_slow_reward(info)
            round_metrics.add_round(
                slow_reward, slow_components, slow_selected
            )
            episode_metrics.merge(round_metrics)

            done = bool(
                terminated
                or truncated
                or local_round + 1 >= int(train_cfg.rounds_per_episode)
            )
            next_slow_obs = env.get_slow_obs()
            slow_agent.store_transition(
                obs_vec=slow_selected["obs_vec"],
                binary_action=slow_selected["binary_action"],
                action_mask=slow_selected["action_mask"],
                reward=slow_reward,
                done=done,
                value=float(slow_selected["value"]),
                log_prob=float(slow_selected["log_prob"]),
            )

            round_logger.write({
                "episode": episode,
                "round_in_episode": local_round + 1,
                "global_round": global_round,
                "global_slot": global_slot,
                **round_metrics.summary(),
            })
            last_next_slow_obs = next_slow_obs
            last_done = done

            if slow_agent.buffer.is_full:
                slow_agent.finish_rollout(next_slow_obs, last_done=done)
                update_logs = slow_agent.update()
                update_index += 1
                update_logger.write({
                    "update": update_index,
                    "episode": episode,
                    "global_round": global_round,
                    "global_slot": global_slot,
                    **update_logs,
                })
                print(
                    "[SLOW UPDATE] "
                    f"update={update_index} round={global_round} "
                    f"loss={update_logs['policy_loss']:.6f} "
                    f"value={update_logs['value_loss']:.6f} "
                    f"entropy={update_logs['entropy']:.6f} "
                    f"kl={update_logs['approx_kl']:.6f} "
                    f"ev={update_logs['explained_variance']:.4f}"
                )
            if done:
                break
            slow_obs = next_slow_obs

        episode_summary = episode_metrics.summary()
        episode_logger.write({
            "episode": episode,
            "global_round": global_round,
            "global_slot": global_slot,
            **episode_summary,
        })
        print(
            "[SLOW EPISODE] "
            f"ep={episode}/{train_cfg.num_episodes} "
            f"reward/round={episode_summary['slow_reward_per_round']:.3f} "
            f"delivery/slot={episode_summary['delivery_per_slot']:.4f} "
            f"degradation/chunk="
            f"{episode_summary['quality_degradation_per_chunk']:.4f} "
            f"scheduled_stall={episode_summary['scheduled_stall_rate']:.6f} "
            f"hired/round={episode_summary['hired_uav_per_round']:.3f}"
        )

        if episode % int(train_cfg.save_every_episodes) == 0:
            path = checkpoint_dir / f"slow_ppo_ep{episode}.pt"
            _save_slow_checkpoint(
                slow_agent, path, episode=episode,
                global_slot=global_slot, global_round=global_round,
                update_index=update_index,
                fast_checkpoint=str(_resolve_path(train_cfg.fast_checkpoint)),
            )
            _save_slow_checkpoint(
                slow_agent, checkpoint_dir / "slow_ppo_latest.pt",
                episode=episode, global_slot=global_slot,
                global_round=global_round, update_index=update_index,
                fast_checkpoint=str(_resolve_path(train_cfg.fast_checkpoint)),
            )

        if episode % int(train_cfg.evaluate_every_episodes) == 0:
            candidate = evaluate_paired(
                train_cfg, fast_agent, slow_agent, random_slow=False
            )
            eligible, checks = _checkpoint_is_eligible(
                candidate, baseline, train_cfg, env_cfg
            )
            eval_logger.write({
                "episode": episode,
                "global_round": global_round,
                "eligible": float(eligible),
                **candidate,
                **checks,
            })
            if (
                eligible
                and candidate["slow_reward_per_round"] > best_eligible_reward
            ):
                best_eligible_reward = candidate["slow_reward_per_round"]
                _save_slow_checkpoint(
                    slow_agent,
                    checkpoint_dir / "slow_ppo_best_eligible.pt",
                    episode=episode, global_slot=global_slot,
                    global_round=global_round, update_index=update_index,
                    fast_checkpoint=str(_resolve_path(train_cfg.fast_checkpoint)),
                )
            print(
                "[PAIRED EVAL] "
                f"ep={episode} eligible={int(eligible)} "
                f"reward/round={candidate['slow_reward_per_round']:.3f} "
                f"delivery/slot={candidate['delivery_per_slot']:.4f} "
                f"degradation/chunk="
                f"{candidate['quality_degradation_per_chunk']:.4f}"
            )

    if len(slow_agent.buffer) > 0:
        slow_agent.finish_rollout(
            last_next_slow_obs, last_done=last_done
        )
        update_logs = slow_agent.update()
        update_index += 1
        update_logger.write({
            "update": update_index,
            "episode": int(train_cfg.num_episodes),
            "global_round": global_round,
            "global_slot": global_slot,
            **update_logs,
        })

    final_path = checkpoint_dir / "slow_ppo_final.pt"
    _save_slow_checkpoint(
        slow_agent, final_path, episode=int(train_cfg.num_episodes),
        global_slot=global_slot, global_round=global_round,
        update_index=update_index,
        fast_checkpoint=str(_resolve_path(train_cfg.fast_checkpoint)),
    )
    final_eval = evaluate_paired(
        train_cfg, fast_agent, slow_agent, random_slow=False
    )
    save_json(final_eval, run_dir / "final_paired_evaluation.json")
    print(f"[SLOW HRL TRAIN DONE] final_checkpoint={final_path}")


def evaluate(train_cfg: SlowTrainConfig) -> None:
    set_seed(train_cfg.seed, deterministic=train_cfg.deterministic_torch)
    env_cfg = _make_env_config(
        seed=train_cfg.seed,
        move_prob=train_cfg.eval_move_prob,
        rounds_per_episode=train_cfg.eval_rounds_per_seed,
    )
    env = Env(env_cfg)
    sample_fast_obs, _ = split_env_reset(env.reset())
    fast_agent = _build_frozen_fast_agent(
        train_cfg, env_cfg, sample_fast_obs
    )
    slow_agent, _, _, _ = _build_slow_agent(
        train_cfg, env_cfg, env.get_slow_obs()
    )
    baseline = evaluate_paired(
        train_cfg, fast_agent, slow_agent, random_slow=True
    )
    candidate = evaluate_paired(
        train_cfg, fast_agent, slow_agent, random_slow=False
    )
    eligible, checks = _checkpoint_is_eligible(
        candidate, baseline, train_cfg, env_cfg
    )
    result = {
        "eligible": bool(eligible),
        "baseline": baseline,
        "slow_policy": candidate,
        "eligibility_checks": checks,
    }
    output_dir = _resolve_path(Path(train_cfg.output_root) / train_cfg.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result, output_dir / "standalone_paired_evaluation.json")
    print(result)


def main() -> None:
    train_cfg = get_slow_train_config()
    if train_cfg.mode == "train":
        train(train_cfg)
    else:
        evaluate(train_cfg)


if __name__ == "__main__":
    main()