from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

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
from agent.PPO.hrl.hrl_config import (  # noqa: E402
    JointHRLConfig,
    get_joint_hrl_config,
)
from agent.PPO.slow.slow_agent import SlowPPOAgent  # noqa: E402
from agent.PPO.slow.slow_metrics import HRLMetrics  # noqa: E402
from agent.PPO.slow.slow_train import (  # noqa: E402
    _checkpoint_is_eligible,
    _extract_slow_reward,
    _load_trusted_checkpoint_metadata,
    _make_env_config,
    _resolve_path,
    evaluate_paired,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_joint_fast_agent(
    cfg: JointHRLConfig,
    env_cfg: EnvConfig,
    sample_fast_obs: Mapping[str, Any],
) -> Tuple[FastPPOAgent, Path, str]:
    checkpoint_path = _resolve_path(cfg.initial_fast_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"initial fast checkpoint does not exist: {checkpoint_path}"
        )

    extra = _load_trusted_checkpoint_metadata(checkpoint_path)
    policy_type = str(extra.get("policy_type", ""))
    if policy_type != "conditional_mixed_categorical_gaussian_v1":
        raise ValueError(
            "joint HRL requires the mixed categorical/Gaussian fast policy; "
            f"checkpoint policy_type={policy_type!r}."
        )

    saved_config = dict(extra.get("fast_ppo_config", {}))
    allowed = {field.name for field in fields(FastPPOConfig)}
    values = {key: value for key, value in saved_config.items() if key in allowed}
    values.update({
        "rollout_steps": int(env_cfg.slow_T) * int(cfg.fast_rollout_rounds),
        "update_epochs": int(cfg.fast_update_epochs),
        "batch_size": int(cfg.fast_batch_size),
        "gamma": float(cfg.fast_gamma),
        "gae_lambda": float(cfg.fast_gae_lambda),
        "lr": float(cfg.fast_lr),
        "max_grad_norm": float(cfg.fast_max_grad_norm),
        "clip_coef": float(cfg.fast_clip_coef),
        "value_coef": float(cfg.fast_value_coef),
        "categorical_entropy_coef": float(
            cfg.fast_categorical_entropy_coef
        ),
        "power_entropy_coef": float(cfg.fast_power_entropy_coef),
        "normalize_obs": bool(cfg.obs_norm),
        "normalize_adv": bool(cfg.adv_norm),
        "use_value_huber_loss": bool(cfg.use_value_huber_loss),
        "use_value_clip": bool(cfg.use_value_clip),
        "value_clip_coef": float(cfg.fast_value_clip_coef),
        "fail_on_nan": bool(cfg.fail_on_nan),
        "target_kl": (
            None if cfg.fast_target_kl is None else float(cfg.fast_target_kl)
        ),
        "device": str(cfg.device),
    })
    values.setdefault(
        "hidden_dims", tuple(int(x) for x in cfg.fast_hidden_dims_fallback)
    )
    values.setdefault("init_log_std", float(cfg.fast_init_log_std_fallback))
    fast_ppo_config = FastPPOConfig(**values)

    obs_dim = infer_fast_obs_dim(dict(sample_fast_obs))
    saved_obs_dim = int(extra.get("obs_dim", obs_dim))
    if saved_obs_dim != obs_dim:
        raise ValueError(
            "fast checkpoint/environment observation mismatch: "
            f"checkpoint={saved_obs_dim}, environment={obs_dim}."
        )

    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=fast_ppo_config,
    )
    loaded = agent.load(checkpoint_path, strict=True, load_optimizer=False)
    loaded_extra = loaded.get("extra", {})
    if agent.obs_normalizer is not None and "obs_normalizer" not in loaded_extra:
        raise RuntimeError(
            "normalized fast checkpoint has no observation-normalizer state."
        )
    return agent, checkpoint_path, _sha256(checkpoint_path)


def _build_slow_agent(
    cfg: JointHRLConfig,
    env_cfg: EnvConfig,
    sample_slow_obs: Mapping[str, Any],
) -> SlowPPOAgent:
    agent = SlowPPOAgent(
        env_cfg=env_cfg,
        obs_dim=infer_slow_obs_dim(dict(sample_slow_obs)),
        ppo_cfg=cfg.make_slow_ppo_config(),
    )
    if cfg.initial_slow_checkpoint is not None:
        checkpoint_path = _resolve_path(cfg.initial_slow_checkpoint)
        loaded = agent.load(
            checkpoint_path, strict=True, load_optimizer=False
        )
        loaded_extra = loaded.get("extra", {})
        if agent.obs_normalizer is not None and "obs_normalizer" not in loaded_extra:
            raise RuntimeError(
                "normalized slow checkpoint has no observation-normalizer state."
            )
    return agent


def _set_fast_trainable(agent: FastPPOAgent, trainable: bool) -> None:
    for parameter in agent.model.parameters():
        parameter.requires_grad_(bool(trainable))
    agent.model.train(bool(trainable))


def _training_state(
    *,
    episode: int,
    global_slot: int,
    global_round: int,
    fast_update: int,
    slow_update: int,
    best_eligible_reward: float,
    best_manifest: Optional[str],
    env: Env,
) -> Dict[str, Any]:
    return {
        "episode": int(episode),
        "global_slot": int(global_slot),
        "global_round": int(global_round),
        "fast_update": int(fast_update),
        "slow_update": int(slow_update),
        "best_eligible_reward": float(best_eligible_reward),
        "best_manifest": best_manifest,
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "env_rng_state": env.rng.bit_generator.state,
    }


def _save_pair(
    *,
    fast_agent: FastPPOAgent,
    slow_agent: SlowPPOAgent,
    checkpoint_dir: Path,
    tag: str,
    initial_fast_checkpoint: Path,
    initial_fast_sha256: str,
    training_state: Optional[Dict[str, Any]],
    resumable: bool,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pair_id = (
        f"{tag}-"
        f"{0 if training_state is None else training_state['global_round']}-"
        f"{0 if training_state is None else training_state['fast_update']}-"
        f"{0 if training_state is None else training_state['slow_update']}"
    )
    fast_path = checkpoint_dir / f"fast_joint_{tag}.pt"
    slow_path = checkpoint_dir / f"slow_joint_{tag}.pt"
    fast_tmp = checkpoint_dir / f"fast_joint_{tag}.tmp.pt"
    slow_tmp = checkpoint_dir / f"slow_joint_{tag}.tmp.pt"

    common: Dict[str, Any] = {
        "joint_pair_id": pair_id,
        "joint_hrl": True,
        "resumable": bool(resumable),
        "initial_fast_checkpoint": str(initial_fast_checkpoint),
        "initial_fast_sha256": str(initial_fast_sha256),
    }
    if training_state is not None:
        common["joint_training_state"] = training_state

    try:
        fast_agent.save(fast_tmp, extra={**common, "hierarchy_level": "fast"})
        slow_agent.save(slow_tmp, extra={**common, "hierarchy_level": "slow"})
        fast_tmp.replace(fast_path)
        slow_tmp.replace(slow_path)
    finally:
        fast_tmp.unlink(missing_ok=True)
        slow_tmp.unlink(missing_ok=True)

    manifest = {
        "pair_id": pair_id,
        "tag": tag,
        "resumable": bool(resumable),
        "fast_checkpoint": str(fast_path),
        "slow_checkpoint": str(slow_path),
        "initial_fast_checkpoint": str(initial_fast_checkpoint),
        "initial_fast_sha256": str(initial_fast_sha256),
    }
    if training_state is not None:
        manifest.update({
            "episode": int(training_state["episode"]),
            "global_slot": int(training_state["global_slot"]),
            "global_round": int(training_state["global_round"]),
            "fast_update": int(training_state["fast_update"]),
            "slow_update": int(training_state["slow_update"]),
            "best_eligible_reward": float(
                training_state["best_eligible_reward"]
            ),
            "best_manifest": training_state["best_manifest"],
        })
    manifest_path = checkpoint_dir / f"joint_{tag}_manifest.json"
    save_json(manifest, manifest_path)
    return manifest_path


def _load_manifest(path: str | Path) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = _resolve_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"resume manifest does not exist: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest_path, dict(data)


def _restore_pair_for_resume(
    *,
    cfg: JointHRLConfig,
    fast_agent: FastPPOAgent,
    slow_agent: SlowPPOAgent,
    expected_initial_fast_sha256: str,
) -> Tuple[int, int, int, int, int, float, Optional[str], Dict[str, Any]]:
    if cfg.resume_manifest is None:
        return 0, 0, 0, 0, 0, -np.inf, None, {}
    _, manifest = _load_manifest(cfg.resume_manifest)
    if not bool(manifest.get("resumable", False)):
        raise ValueError("the selected joint checkpoint is evaluation-only.")

    fast_loaded = fast_agent.load(
        _resolve_path(manifest["fast_checkpoint"]),
        strict=True,
        load_optimizer=True,
    )
    slow_loaded = slow_agent.load(
        _resolve_path(manifest["slow_checkpoint"]),
        strict=True,
        load_optimizer=True,
    )
    fast_extra = fast_loaded.get("extra", {})
    slow_extra = slow_loaded.get("extra", {})
    pair_id = str(manifest["pair_id"])
    if str(fast_extra.get("joint_pair_id")) != pair_id:
        raise RuntimeError("fast checkpoint does not match the resume manifest.")
    if str(slow_extra.get("joint_pair_id")) != pair_id:
        raise RuntimeError("slow checkpoint does not match the resume manifest.")
    if str(fast_extra.get("initial_fast_sha256")) != str(
        expected_initial_fast_sha256
    ):
        raise RuntimeError(
            "resume checkpoint was initialized from a different fast policy."
        )
    state = dict(fast_extra.get("joint_training_state", {}))
    slow_state = dict(slow_extra.get("joint_training_state", {}))
    if not state:
        raise RuntimeError("resume checkpoint has no joint training state.")
    for key in (
        "episode",
        "global_slot",
        "global_round",
        "fast_update",
        "slow_update",
    ):
        if state.get(key) != slow_state.get(key):
            raise RuntimeError(
                f"fast/slow resume training state mismatch for {key}."
            )

    return (
        int(state["episode"]),
        int(state["global_slot"]),
        int(state["global_round"]),
        int(state["fast_update"]),
        int(state["slow_update"]),
        float(state.get("best_eligible_reward", -np.inf)),
        state.get("best_manifest"),
        state,
    )


def _restore_rng_state(state: Mapping[str, Any], env: Env) -> None:
    if not state:
        return
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_states = state.get("torch_cuda_rng_state_all", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    env.rng.bit_generator.state = state["env_rng_state"]


def _assert_empty_resume_buffers(
    cfg: JointHRLConfig,
    fast_agent: FastPPOAgent,
    slow_agent: SlowPPOAgent,
    global_round: int,
) -> None:
    if len(fast_agent.buffer) != 0 or len(slow_agent.buffer) != 0:
        raise RuntimeError("resumable checkpoint requires empty PPO buffers.")
    if global_round % int(cfg.slow_rollout_rounds) != 0:
        raise RuntimeError("resume point is not a slow rollout boundary.")
    if global_round > int(cfg.fast_freeze_rounds):
        trained_rounds = global_round - int(cfg.fast_freeze_rounds)
        if trained_rounds % int(cfg.fast_rollout_rounds) != 0:
            raise RuntimeError("resume point is not a fast rollout boundary.")


def _evaluation_view(
    cfg: JointHRLConfig,
    *,
    seeds: Tuple[int, ...],
    move_prob: float,
) -> JointHRLConfig:
    return replace(
        cfg,
        selection_eval_seeds=tuple(int(x) for x in seeds),
        selection_move_prob=float(move_prob),
        resume_manifest=None,
    )


def _paired_matrix(
    cfg: JointHRLConfig,
    *,
    learned_fast: FastPPOAgent,
    initial_fast: FastPPOAgent,
    learned_slow: SlowPPOAgent,
) -> Dict[str, Dict[str, float]]:
    return {
        "learned_slow_learned_fast": evaluate_paired(
            cfg, learned_fast, learned_slow, random_slow=False
        ),
        "random_slow_learned_fast": evaluate_paired(
            cfg, learned_fast, learned_slow, random_slow=True
        ),
        "learned_slow_initial_fast": evaluate_paired(
            cfg, initial_fast, learned_slow, random_slow=False
        ),
        "random_slow_initial_fast": evaluate_paired(
            cfg, initial_fast, learned_slow, random_slow=True
        ),
    }


def _load_pair_for_evaluation(
    manifest_path: Path,
    fast_agent: FastPPOAgent,
    slow_agent: SlowPPOAgent,
) -> None:
    _, manifest = _load_manifest(manifest_path)
    fast_loaded = fast_agent.load(
        _resolve_path(manifest["fast_checkpoint"]),
        strict=True,
        load_optimizer=False,
    )
    slow_loaded = slow_agent.load(
        _resolve_path(manifest["slow_checkpoint"]),
        strict=True,
        load_optimizer=False,
    )
    pair_id = str(manifest["pair_id"])
    if str(fast_loaded.get("extra", {}).get("joint_pair_id")) != pair_id:
        raise RuntimeError("evaluation fast checkpoint pair mismatch.")
    if str(slow_loaded.get("extra", {}).get("joint_pair_id")) != pair_id:
        raise RuntimeError("evaluation slow checkpoint pair mismatch.")
    fast_agent.model.eval()
    slow_agent.model.eval()


def train(cfg: JointHRLConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.deterministic_torch)
    if set(cfg.selection_eval_seeds) & set(cfg.final_test_seeds):
        raise ValueError(
            "selection_eval_seeds and final_test_seeds must be disjoint."
        )
    env_cfg = _make_env_config(
        seed=int(cfg.seed),
        move_prob=float(cfg.train_move_prob),
        rounds_per_episode=int(cfg.rounds_per_episode),
    )
    cfg.validate_environment(env_cfg)

    probe_env = Env(env_cfg)
    sample_fast_obs, _ = split_env_reset(probe_env.reset())
    sample_slow_obs = probe_env.get_slow_obs()
    fast_agent, initial_fast_path, initial_fast_sha = _build_joint_fast_agent(
        cfg, env_cfg, sample_fast_obs
    )
    slow_agent = _build_slow_agent(cfg, env_cfg, sample_slow_obs)

    (
        start_episode,
        global_slot,
        global_round,
        fast_update,
        slow_update,
        best_eligible_reward,
        best_manifest_raw,
        resume_state,
    ) = _restore_pair_for_resume(
        cfg=cfg,
        fast_agent=fast_agent,
        slow_agent=slow_agent,
        expected_initial_fast_sha256=initial_fast_sha,
    )
    best_manifest = (
        None if best_manifest_raw is None else str(best_manifest_raw)
    )
    _assert_empty_resume_buffers(
        cfg, fast_agent, slow_agent, global_round
    )

    run_dir = _resolve_path(Path(cfg.output_root) / cfg.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg.to_dict(), run_dir / "joint_train_config.json")
    save_json(env_cfg.as_dict(), run_dir / "env_config.json")
    save_json(
        {
            "path": str(initial_fast_path),
            "sha256": initial_fast_sha,
        },
        run_dir / "initial_fast_checkpoint.json",
    )

    episode_logger = ScalarLogger(run_dir / "episode_metrics.csv")
    round_logger = ScalarLogger(run_dir / "round_metrics.csv")
    fast_update_logger = ScalarLogger(run_dir / "fast_update_metrics.csv")
    slow_update_logger = ScalarLogger(run_dir / "slow_update_metrics.csv")
    eval_logger = ScalarLogger(run_dir / "selection_evaluation.csv")

    env = Env(env_cfg)
    _restore_rng_state(resume_state, env)
    _set_fast_trainable(
        fast_agent, global_round >= int(cfg.fast_freeze_rounds)
    )
    slow_agent.model.train()

    print("=" * 96)
    print("[JOINT SLOW/FAST HRL TRAIN START]")
    print(f"run_dir             : {run_dir}")
    print(f"initial_fast        : {initial_fast_path}")
    print(f"initial_fast_sha256 : {initial_fast_sha}")
    print(f"slow_T              : {env_cfg.slow_T}")
    print(f"fast_rollout_slots  : {fast_agent.ppo_cfg.rollout_steps}")
    print(f"slow_rollout_rounds : {slow_agent.ppo_cfg.rollout_rounds}")
    print(f"fast_freeze_rounds  : {cfg.fast_freeze_rounds}")
    print(f"start_episode       : {start_episode}")
    print("=" * 96)

    for episode in range(start_episode + 1, int(cfg.num_episodes) + 1):
        split_env_reset(env.reset())
        episode_metrics = HRLMetrics(
            fast_layer_ratios=np.zeros(
                int(env_cfg.layer), dtype=np.float64
            )
        )

        for local_round in range(int(cfg.rounds_per_episode)):
            if int(env.round_slot) != 0:
                raise RuntimeError("slow action requested away from a boundary.")

            fast_trainable = global_round >= int(cfg.fast_freeze_rounds)
            _set_fast_trainable(fast_agent, fast_trainable)
            slow_obs = env.get_slow_obs()
            slow_selected = slow_agent.select_action(
                slow_obs,
                context=env,
                deterministic=False,
                update_norm=True,
            )
            env.apply_slow_action(slow_selected["env_action"])
            fast_obs = env.get_fast_obs()
            round_metrics = HRLMetrics(
                fast_layer_ratios=np.zeros(
                    int(env_cfg.layer), dtype=np.float64
                )
            )
            boundary_info: Optional[Dict[str, Any]] = None
            terminated = False
            truncated = False

            for _ in range(int(env_cfg.slow_T)):
                fast_selected = fast_agent.select_action(
                    fast_obs,
                    deterministic=(
                        bool(cfg.fast_deterministic_while_frozen)
                        if not fast_trainable
                        else False
                    ),
                    update_norm=bool(fast_trainable),
                )
                next_fast_obs, reward, terminated, truncated, info = (
                    split_env_step(env.step(fast_selected["env_action"]))
                )
                is_round_boundary = bool(
                    info.get("is_round_boundary", False)
                )
                if fast_trainable:
                    # Every slow boundary is an option terminal for fast GAE.
                    # Thus the fast value never bootstraps through a not-yet
                    # selected next slow action.
                    fast_agent.store_transition(
                        obs_vec=fast_selected["obs_vec"],
                        raw_action=fast_selected["raw_action"],
                        action_mask=fast_selected["action_mask"],
                        reward=float(reward) * float(cfg.fast_reward_scale),
                        done=is_round_boundary,
                        value=float(fast_selected["value"]),
                        log_prob=float(fast_selected["log_prob"]),
                    )
                round_metrics.add_slot(
                    info,
                    fast_selected,
                    num_layers=int(env_cfg.layer),
                )
                fast_obs = next_fast_obs
                if is_round_boundary:
                    boundary_info = info
                    break
                if terminated or truncated:
                    raise RuntimeError(
                        "episode ended before a complete slow transition."
                    )

            if boundary_info is None:
                raise RuntimeError("fast loop did not reach a slow boundary.")
            if round_metrics.slots != int(env_cfg.slow_T):
                raise RuntimeError(
                    "incomplete slow transition: "
                    f"expected={env_cfg.slow_T}, got={round_metrics.slots}."
                )

            slow_reward, slow_components = _extract_slow_reward(
                boundary_info
            )
            round_metrics.add_round(
                slow_reward, slow_components, slow_selected
            )
            episode_metrics.merge(round_metrics)
            global_slot += int(round_metrics.slots)
            global_round += 1

            expected_episode_done = (
                local_round + 1 >= int(cfg.rounds_per_episode)
            )
            episode_done = bool(terminated or truncated)
            if episode_done != expected_episode_done:
                raise RuntimeError(
                    "environment episode boundary is not aligned with "
                    "rounds_per_episode."
                )
            next_slow_obs = env.get_slow_obs()
            slow_agent.store_transition(
                obs_vec=slow_selected["obs_vec"],
                binary_action=slow_selected["binary_action"],
                action_mask=slow_selected["action_mask"],
                reward=slow_reward,
                done=episode_done,
                value=float(slow_selected["value"]),
                log_prob=float(slow_selected["log_prob"]),
            )

            if fast_trainable and fast_agent.buffer.is_full:
                fast_agent.finish_rollout(fast_obs, last_done=True)
                logs = fast_agent.update()
                fast_update += 1
                fast_update_logger.write({
                    "update": fast_update,
                    "episode": episode,
                    "global_round": global_round,
                    "global_slot": global_slot,
                    **logs,
                })
                print(
                    "[FAST JOINT UPDATE] "
                    f"update={fast_update} round={global_round} "
                    f"kl={logs['approx_kl']:.6f} "
                    f"cat_entropy={logs['categorical_entropy']:.6f} "
                    f"ev={logs['explained_variance']:.4f}"
                )

            if slow_agent.buffer.is_full:
                slow_agent.finish_rollout(
                    next_slow_obs, last_done=episode_done
                )
                logs = slow_agent.update()
                slow_update += 1
                slow_update_logger.write({
                    "update": slow_update,
                    "episode": episode,
                    "global_round": global_round,
                    "global_slot": global_slot,
                    **logs,
                })
                print(
                    "[SLOW JOINT UPDATE] "
                    f"update={slow_update} round={global_round} "
                    f"kl={logs['approx_kl']:.6f} "
                    f"entropy={logs['entropy']:.6f} "
                    f"ev={logs['explained_variance']:.4f}"
                )

            round_logger.write({
                "episode": episode,
                "round_in_episode": local_round + 1,
                "global_round": global_round,
                "global_slot": global_slot,
                "fast_trainable": float(fast_trainable),
                "fast_update": fast_update,
                "slow_update": slow_update,
                **round_metrics.summary(),
            })

        episode_logger.write({
            "episode": episode,
            "global_round": global_round,
            "global_slot": global_slot,
            "fast_update": fast_update,
            "slow_update": slow_update,
            **episode_metrics.summary(),
        })
        print(
            "[JOINT EPISODE] "
            f"ep={episode}/{cfg.num_episodes} "
            f"reward/round="
            f"{episode_metrics.summary()['slow_reward_per_round']:.3f} "
            f"delivery/slot="
            f"{episode_metrics.summary()['delivery_per_slot']:.4f}"
        )

        if episode % int(cfg.evaluate_every_episodes) == 0:
            selection_cfg = _evaluation_view(
                cfg,
                seeds=tuple(cfg.selection_eval_seeds),
                move_prob=float(cfg.selection_move_prob),
            )
            candidate = evaluate_paired(
                selection_cfg,
                fast_agent,
                slow_agent,
                random_slow=False,
            )
            baseline = evaluate_paired(
                selection_cfg,
                fast_agent,
                slow_agent,
                random_slow=True,
            )
            eligible, checks = _checkpoint_is_eligible(
                candidate, baseline, selection_cfg, env_cfg
            )
            eval_logger.write({
                "episode": episode,
                "global_round": global_round,
                "global_slot": global_slot,
                "eligible": float(eligible),
                **{f"candidate_{k}": v for k, v in candidate.items()},
                **{f"baseline_{k}": v for k, v in baseline.items()},
                **checks,
            })
            if (
                eligible
                and candidate["slow_reward_per_round"]
                > best_eligible_reward
            ):
                best_eligible_reward = float(
                    candidate["slow_reward_per_round"]
                )
                best_manifest_path = _save_pair(
                    fast_agent=fast_agent,
                    slow_agent=slow_agent,
                    checkpoint_dir=checkpoint_dir,
                    tag="best_eligible",
                    initial_fast_checkpoint=initial_fast_path,
                    initial_fast_sha256=initial_fast_sha,
                    training_state=None,
                    resumable=False,
                )
                best_manifest = str(best_manifest_path)
            print(
                "[PAIRED SELECTION EVAL] "
                f"ep={episode} eligible={int(eligible)} "
                f"candidate_reward/round="
                f"{candidate['slow_reward_per_round']:.3f} "
                f"baseline_reward/round="
                f"{baseline['slow_reward_per_round']:.3f}"
            )

        if episode % int(cfg.save_every_episodes) == 0:
            _assert_empty_resume_buffers(
                cfg, fast_agent, slow_agent, global_round
            )
            state = _training_state(
                episode=episode,
                global_slot=global_slot,
                global_round=global_round,
                fast_update=fast_update,
                slow_update=slow_update,
                best_eligible_reward=best_eligible_reward,
                best_manifest=best_manifest,
                env=env,
            )
            _save_pair(
                fast_agent=fast_agent,
                slow_agent=slow_agent,
                checkpoint_dir=checkpoint_dir,
                tag=f"resume_ep{episode}",
                initial_fast_checkpoint=initial_fast_path,
                initial_fast_sha256=initial_fast_sha,
                training_state=state,
                resumable=True,
            )
            _save_pair(
                fast_agent=fast_agent,
                slow_agent=slow_agent,
                checkpoint_dir=checkpoint_dir,
                tag="latest",
                initial_fast_checkpoint=initial_fast_path,
                initial_fast_sha256=initial_fast_sha,
                training_state=state,
                resumable=True,
            )

    _assert_empty_resume_buffers(cfg, fast_agent, slow_agent, global_round)
    final_state = _training_state(
        episode=int(cfg.num_episodes),
        global_slot=global_slot,
        global_round=global_round,
        fast_update=fast_update,
        slow_update=slow_update,
        best_eligible_reward=best_eligible_reward,
        best_manifest=best_manifest,
        env=env,
    )
    final_manifest = _save_pair(
        fast_agent=fast_agent,
        slow_agent=slow_agent,
        checkpoint_dir=checkpoint_dir,
        tag="final",
        initial_fast_checkpoint=initial_fast_path,
        initial_fast_sha256=initial_fast_sha,
        training_state=final_state,
        resumable=True,
    )

    selected_manifest = (
        Path(best_manifest) if best_manifest is not None else final_manifest
    )
    _load_pair_for_evaluation(
        selected_manifest, fast_agent, slow_agent
    )
    initial_fast_agent, _, _ = _build_joint_fast_agent(
        cfg, env_cfg, sample_fast_obs
    )
    initial_fast_agent.model.eval()

    static_cfg = _evaluation_view(
        cfg,
        seeds=tuple(cfg.final_test_seeds),
        move_prob=0.0,
    )
    weak_cfg = _evaluation_view(
        cfg,
        seeds=tuple(cfg.final_test_seeds),
        move_prob=float(cfg.weak_mobility_move_prob),
    )
    static_matrix = _paired_matrix(
        static_cfg,
        learned_fast=fast_agent,
        initial_fast=initial_fast_agent,
        learned_slow=slow_agent,
    )
    weak_matrix = _paired_matrix(
        weak_cfg,
        learned_fast=fast_agent,
        initial_fast=initial_fast_agent,
        learned_slow=slow_agent,
    )
    heldout_eligible, heldout_checks = _checkpoint_is_eligible(
        static_matrix["learned_slow_learned_fast"],
        static_matrix["random_slow_learned_fast"],
        static_cfg,
        env_cfg,
    )
    result = {
        "selected_manifest": str(selected_manifest),
        "selection_found_eligible_checkpoint": bool(
            best_manifest is not None
        ),
        "heldout_static_eligible": bool(heldout_eligible),
        "heldout_static_checks": heldout_checks,
        "static": static_matrix,
        "weak_mobility": weak_matrix,
    }
    save_json(result, run_dir / "heldout_paired_ablation.json")
    print(
        "[JOINT HRL TRAIN DONE] "
        f"final_manifest={final_manifest} "
        f"selected_manifest={selected_manifest} "
        f"heldout_static_eligible={int(heldout_eligible)}"
    )


def main() -> None:
    train(get_joint_hrl_config())


if __name__ == "__main__":
    main()