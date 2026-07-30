from __future__ import annotations

import copy
import csv
import random
import signal
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import torch


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))

from config import (  # noqa: E402
    BatteryConfig,
    ChannelConfig,
    EnvConfig,
)
from env.env import Env  # noqa: E402
from agent.PPO.common import (  # noqa: E402
    ScalarLogger,
    infer_fast_obs_dim,
    set_seed,
    split_env_reset,
    split_env_step,
)
from agent.PPO.common.utils import save_json  # noqa: E402
from agent.PPO.fast.fast_agent import (  # noqa: E402
    FastPPOAgent,
    FastPPOConfig,
)
from agent.PPO.slow.slow_dpp_controller import (  # noqa: E402
    SlowDPPController,
)
from agent.PPO.joint.joint_config import (  # noqa: E402
    SlowJointTrainConfig,
    get_slow_joint_train_config,
)
from agent.PPO.slow.slow_metrics import (  # noqa: E402
    SlowDPPMetrics,
)


DataclassT = TypeVar("DataclassT")
JOINT_TRAINER_KIND = "slow_dpp_fast_ppo_scratch_v2"
FAST_INITIALIZATION = "random_from_scratch"
STOP_REQUESTED = False


def _request_stop(
    signum: int,
    _frame: Any,
) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        "[JOINT SIGNAL] "
        f"received signal={signum}; "
        "will save at the next completed round boundary",
        flush=True,
    )


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)


def _resolve_path(path: str | Path) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = PROPOSED_ROOT / result
    return result.resolve()


def _load_trusted_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Load a checkpoint created by this repository.

    weights_only=False is required because exact-resume metadata contains
    NumPy arrays, RNG state, environment state, and metric dataclasses.
    Never use this loader for an untrusted external checkpoint.
    """
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Fast checkpoint must contain a dictionary."
        )
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint has no model_state_dict."
        )
    return checkpoint


def _filtered_dataclass(
    cls: Type[DataclassT],
    values: Mapping[str, Any],
) -> DataclassT:
    allowed = {field.name for field in fields(cls)}
    filtered = {
        key: value
        for key, value in values.items()
        if key in allowed
    }
    return cls(**filtered)


def _make_env_config(
    run_cfg: SlowJointTrainConfig,
    checkpoint_extra: Optional[Mapping[str, Any]] = None,
) -> EnvConfig:
    """
    Build the final joint environment without using Fast-only metadata.

    A fresh run starts from the repository's current EnvConfig defaults and
    applies the explicit final joint-timescale overrides. A resume run
    reconstructs the exact saved environment and validates it against the
    immutable joint contract.
    """
    expected_episode_slots = (
        int(run_cfg.final_slow_T)
        * int(run_cfg.rounds_per_episode)
    )

    if checkpoint_extra is None:
        base = EnvConfig()
        battery = replace(
            base.battery,
            target_service_slots_per_round=int(
                run_cfg.final_target_service_slots_per_round
            ),
        )
        env_cfg = replace(
            base,
            seed=int(run_cfg.seed),
            slow_T=int(run_cfg.final_slow_T),
            episode_slots=expected_episode_slots,
            mobility_mode=str(
                run_cfg.final_mobility_mode
            ),
            move_prob=float(
                run_cfg.final_move_prob
            ),
            uav_hiring_cost=float(
                run_cfg.final_uav_hiring_cost
            ),
            battery=battery,
        )
    else:
        if checkpoint_extra.get(
            "trainer_kind"
        ) != JOINT_TRAINER_KIND:
            raise RuntimeError(
                "Resume checkpoint trainer_kind mismatch."
            )
        if checkpoint_extra.get(
            "fast_initialization"
        ) != FAST_INITIALIZATION:
            raise RuntimeError(
                "Resume checkpoint was not initialized by the "
                "from-scratch joint trainer."
            )

        saved = checkpoint_extra.get("env_config")
        if not isinstance(saved, Mapping):
            raise RuntimeError(
                "Joint resume checkpoint has no env_config metadata."
            )
        values = dict(saved)
        for key in ("rsu_channel", "uav_channel"):
            nested = values.get(key)
            if isinstance(nested, Mapping):
                values[key] = _filtered_dataclass(
                    ChannelConfig,
                    nested,
                )
        battery_value = values.get("battery")
        if isinstance(battery_value, Mapping):
            battery_value = _filtered_dataclass(
                BatteryConfig,
                battery_value,
            )
        if not isinstance(battery_value, BatteryConfig):
            raise RuntimeError(
                "Joint resume env_config has invalid battery metadata."
            )
        values["battery"] = battery_value
        allowed = {field.name for field in fields(EnvConfig)}
        values = {
            key: value
            for key, value in values.items()
            if key in allowed
        }
        env_cfg = EnvConfig(**values)

    contract = {
        "seed": int(run_cfg.seed),
        "slow_T": int(run_cfg.final_slow_T),
        "episode_slots": expected_episode_slots,
        "mobility_mode": str(
            run_cfg.final_mobility_mode
        ),
        "move_prob": float(
            run_cfg.final_move_prob
        ),
        "uav_hiring_cost": float(
            run_cfg.final_uav_hiring_cost
        ),
        "target_service_slots_per_round": int(
            run_cfg.final_target_service_slots_per_round
        ),
    }
    actual = {
        "seed": int(env_cfg.seed),
        "slow_T": int(env_cfg.slow_T),
        "episode_slots": int(
            env_cfg.episode_slots or 0
        ),
        "mobility_mode": str(env_cfg.mobility_mode),
        "move_prob": float(env_cfg.move_prob),
        "uav_hiring_cost": float(
            env_cfg.uav_hiring_cost
        ),
        "target_service_slots_per_round": int(
            env_cfg.battery.target_service_slots_per_round
        ),
    }
    if actual != contract:
        raise RuntimeError(
            "Joint environment contract mismatch: "
            f"expected={contract}, actual={actual}"
        )

    expected_slots = (
        int(env_cfg.slow_T)
        * int(run_cfg.rounds_per_episode)
    )
    if int(env_cfg.episode_slots or 0) != expected_slots:
        raise RuntimeError(
            "Joint episode is not an integer number of slow rounds."
        )
    return env_cfg


def _build_fast_agent(
    run_cfg: SlowJointTrainConfig,
    env_cfg: EnvConfig,
    sample_fast_obs: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
    checkpoint: Optional[Mapping[str, Any]] = None,
) -> FastPPOAgent:
    """
    Build a randomly initialized Fast PPO agent.

    When checkpoint_path/checkpoint are supplied, the random construction is
    immediately overwritten by an exact same-trainer resume state, including
    optimizer and observation-normalizer state. Fast-only transfer is not
    supported.
    """
    fast_ppo_config = FastPPOConfig(
        rollout_steps=int(run_cfg.fast_rollout_steps),
        update_epochs=int(run_cfg.fast_update_epochs),
        batch_size=int(run_cfg.fast_batch_size),
        gamma=float(run_cfg.fast_gamma),
        gae_lambda=float(run_cfg.fast_gae_lambda),
        lr=float(run_cfg.fast_lr),
        max_grad_norm=float(run_cfg.fast_max_grad_norm),
        clip_coef=float(run_cfg.fast_clip_coef),
        value_coef=float(run_cfg.fast_value_coef),
        categorical_entropy_coef=float(
            run_cfg.fast_categorical_entropy_coef
        ),
        power_entropy_coef=float(
            run_cfg.fast_power_entropy_coef
        ),
        normalize_obs=bool(run_cfg.fast_normalize_obs),
        normalize_adv=bool(run_cfg.fast_normalize_adv),
        hidden_dims=tuple(
            int(value)
            for value in run_cfg.fast_hidden_dims
        ),
        init_log_std=float(run_cfg.fast_init_log_std),
        use_value_huber_loss=bool(
            run_cfg.fast_use_value_huber_loss
        ),
        use_value_clip=bool(
            run_cfg.fast_use_value_clip
        ),
        value_clip_coef=float(
            run_cfg.fast_value_clip_coef
        ),
        fail_on_nan=bool(run_cfg.fast_fail_on_nan),
        target_kl=(
            None
            if run_cfg.fast_target_kl is None
            else float(run_cfg.fast_target_kl)
        ),
        device=str(run_cfg.device),
    )

    obs_dim = infer_fast_obs_dim(sample_fast_obs)
    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=fast_ppo_config,
    )

    if (checkpoint_path is None) != (checkpoint is None):
        raise ValueError(
            "checkpoint_path and checkpoint must be provided together."
        )
    if checkpoint_path is not None and checkpoint is not None:
        extra = dict(checkpoint.get("extra", {}))
        if extra.get("trainer_kind") != JOINT_TRAINER_KIND:
            raise RuntimeError(
                "Only an exact checkpoint from this joint trainer can "
                "be resumed."
            )
        if extra.get(
            "fast_initialization"
        ) != FAST_INITIALIZATION:
            raise RuntimeError(
                "Resume checkpoint did not originate from random "
                "joint initialization."
            )
        if "optimizer_state_dict" not in checkpoint:
            raise RuntimeError(
                "Exact joint resume checkpoint has no optimizer state."
            )
        policy_type = str(extra.get("policy_type", ""))
        if policy_type != (
            "conditional_mixed_categorical_gaussian_v1"
        ):
            raise RuntimeError(
                "Resume checkpoint has an incompatible Fast policy type: "
                f"{policy_type!r}"
            )
        saved_obs_dim = int(extra.get("obs_dim", -1))
        if saved_obs_dim != obs_dim:
            raise ValueError(
                "Resume checkpoint/environment observation mismatch: "
                f"checkpoint={saved_obs_dim}, environment={obs_dim}"
            )

        saved_fast_cfg = extra.get("fast_ppo_config")
        if not isinstance(saved_fast_cfg, Mapping):
            raise RuntimeError(
                "Resume checkpoint has no Fast PPO config metadata."
            )
        expected_fast_cfg = {
            key: value
            for key, value in vars(fast_ppo_config).items()
            if key != "device"
        }
        actual_fast_cfg = {
            key: saved_fast_cfg.get(key)
            for key in expected_fast_cfg
        }
        if actual_fast_cfg != expected_fast_cfg:
            raise RuntimeError(
                "Fast PPO config differs from the exact-resume "
                "checkpoint. Use the original joint config."
            )

        loaded = agent.load(
            checkpoint_path,
            strict=True,
            load_optimizer=True,
        )
        loaded_extra = dict(loaded.get("extra", {}))
        if (
            agent.obs_normalizer is not None
            and "obs_normalizer" not in loaded_extra
        ):
            raise RuntimeError(
                "Exact resume checkpoint has no observation-normalizer "
                "state."
            )

    if int(agent.ppo_cfg.rollout_steps) != int(env_cfg.slow_T):
        raise ValueError(
            "PPO rollout_steps must equal slow_T so every Fast update uses "
            "exactly one completed slow round: "
            f"rollout_steps={agent.ppo_cfg.rollout_steps}, "
            f"slow_T={env_cfg.slow_T}"
        )

    agent.model.train()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(True)
    return agent


def _resolve_ppo_reward_scale(
    run_cfg: SlowJointTrainConfig,
    checkpoint_extra: Optional[Mapping[str, Any]] = None,
) -> float:
    value = float(run_cfg.ppo_reward_scale)
    if checkpoint_extra is not None:
        saved = checkpoint_extra.get("ppo_reward_scale")
        if saved is None:
            raise RuntimeError(
                "Exact resume checkpoint has no PPO reward scale."
            )
        if not np.isclose(
            value,
            float(saved),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                "Configured PPO reward scale differs from the exact-resume "
                f"checkpoint: configured={value}, saved={saved}"
            )

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Invalid ppo_reward_scale: {value}"
        )
    return value


def _prime_fresh_obs_normalizer(
    fast_agent: FastPPOAgent,
    obs: Mapping[str, Any],
) -> None:
    """
    Add the first observable joint state to a fresh normalizer.

    Slow candidate forecasts use update_norm=False so they cannot contaminate
    training statistics. Without this one real-state update, the first Slow
    decision would evaluate an untrained network on completely unnormalized
    queue, battery, and distance inputs.
    """
    normalizer = fast_agent.obs_normalizer
    if normalizer is None:
        return
    count = float(normalizer.rms.count)
    if not np.isclose(
        count,
        1e-4,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Fresh observation normalizer was unexpectedly pre-populated: "
            f"count={count}"
        )
    fast_agent.obs_to_vec(
        dict(obs),
        update_norm=True,
    )


def _select_slow_action(
    controller: SlowDPPController,
    env: Env,
    fast_agent: FastPPOAgent,
) -> Dict[str, Any]:
    was_training = bool(fast_agent.model.training)
    fast_agent.model.eval()
    try:
        selected = controller.select_action(
            env=env,
            fast_agent=fast_agent,
        )
    finally:
        fast_agent.model.train(was_training)
    return selected


def _apply_selected_slow_action(
    env: Env,
    selected: Mapping[str, Any],
) -> None:
    if int(env.round_slot) != 0:
        raise RuntimeError(
            "Slow action can be applied only at a round boundary."
        )
    env.apply_slow_action(selected["env_action"])


def _extract_slow_reward(
    boundary_info: Mapping[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    reward_components = dict(
        boundary_info.get("reward_components", {})
    )
    slow_reward = float(
        reward_components.get("slow_reward", np.nan)
    )
    slow_components = dict(
        reward_components.get(
            "slow_reward_components",
            {},
        )
    )

    if not bool(
        slow_components.get("is_round_boundary", False)
    ):
        raise RuntimeError(
            "Round boundary has no completed slow DPP value."
        )
    round_dpp_cost = float(
        slow_components.get(
            "round_dpp_cost",
            -slow_reward,
        )
    )
    component_reward = float(
        slow_components.get("slow_reward", slow_reward)
    )
    if not (
        np.isfinite(slow_reward)
        and np.isfinite(round_dpp_cost)
    ):
        raise RuntimeError(
            "Realized slow reward/DPP cost is NaN or Inf."
        )
    if not np.isclose(
        slow_reward,
        component_reward,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Slow reward fields disagree."
        )
    if not np.isclose(
        slow_reward,
        -round_dpp_cost,
        rtol=1e-6,
        atol=1e-3,
    ):
        raise RuntimeError(
            "Realized R_H(r)=-J^S(r) identity is violated."
        )
    return slow_reward, slow_components


def _new_metrics(env_cfg: EnvConfig) -> SlowDPPMetrics:
    return SlowDPPMetrics(
        fast_layer_ratios=np.zeros(
            int(env_cfg.layer),
            dtype=np.float64,
        )
    )


def _capture_buffer(agent: FastPPOAgent) -> Dict[str, Any]:
    buffer = agent.buffer
    size = int(len(buffer))
    return {
        "capacity": int(buffer.capacity),
        "cnt": int(buffer.cnt),
        "full": bool(buffer.full),
        "advantages_ready": bool(
            buffer.advantages_ready
        ),
        "obs": buffer.obs[:size].copy(),
        "actions": buffer.actions[:size].copy(),
        "rewards": buffer.rewards[:size].copy(),
        "dones": buffer.dones[:size].copy(),
        "values": buffer.values[:size].copy(),
        "log_probs": buffer.log_probs[:size].copy(),
        "action_masks": buffer.action_masks[:size].copy(),
        "advantages": buffer.advantages[:size].copy(),
        "returns": buffer.returns[:size].copy(),
    }


def _restore_buffer(
    agent: FastPPOAgent,
    state: Mapping[str, Any],
) -> None:
    buffer = agent.buffer
    if int(state.get("capacity", -1)) != int(
        buffer.capacity
    ):
        raise ValueError(
            "Saved PPO buffer capacity does not match the current agent."
        )

    size = int(np.asarray(state["rewards"]).shape[0])
    if size < 0 or size > int(buffer.capacity):
        raise ValueError(
            f"Invalid saved PPO buffer size: {size}"
        )

    buffer.reset()
    if size > 0:
        buffer.obs[:size] = np.asarray(
            state["obs"],
            dtype=np.float32,
        )
        buffer.actions[:size] = np.asarray(
            state["actions"],
            dtype=np.float32,
        )
        buffer.rewards[:size] = np.asarray(
            state["rewards"],
            dtype=np.float32,
        )
        buffer.dones[:size] = np.asarray(
            state["dones"],
            dtype=np.float32,
        )
        buffer.values[:size] = np.asarray(
            state["values"],
            dtype=np.float32,
        )
        buffer.log_probs[:size] = np.asarray(
            state["log_probs"],
            dtype=np.float32,
        )
        buffer.action_masks[:size] = np.asarray(
            state["action_masks"],
            dtype=np.float32,
        )
        buffer.advantages[:size] = np.asarray(
            state["advantages"],
            dtype=np.float32,
        )
        buffer.returns[:size] = np.asarray(
            state["returns"],
            dtype=np.float32,
        )

    buffer.cnt = int(state.get("cnt", size))
    buffer.full = bool(state.get("full", False))
    buffer.advantages_ready = bool(
        state.get("advantages_ready", False)
    )

    expected_size = (
        int(buffer.capacity)
        if buffer.full
        else int(buffer.cnt)
    )
    if expected_size != size:
        raise ValueError(
            "Saved PPO buffer counters disagree with saved arrays."
        )


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _last_csv_int(
    path: Path,
    field: str,
) -> Optional[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    last: Optional[int] = None
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            value = row.get(field)
            if value not in (None, ""):
                last = int(float(value))
    return last


def _save_joint_checkpoint(
    *,
    path: Path,
    agent: FastPPOAgent,
    run_cfg: SlowJointTrainConfig,
    env_cfg: EnvConfig,
    env: Env,
    run_dir: Path,
    ppo_reward_scale: float,
    current_episode: int,
    rounds_completed_in_episode: int,
    global_round: int,
    global_slot: int,
    update_idx: int,
    pending_slow_selected: Optional[Dict[str, Any]],
    episode_metrics: SlowDPPMetrics,
    aggregate_metrics: SlowDPPMetrics,
) -> None:
    if int(env.round_slot) != 0:
        raise RuntimeError(
            "Exact-resume checkpoint must be saved at a round boundary."
        )
    agent.save(
        path,
        extra={
            "trainer_kind": JOINT_TRAINER_KIND,
            "fast_initialization": FAST_INITIALIZATION,
            "run_dir": str(run_dir),
            "current_episode": int(current_episode),
            "rounds_completed_in_episode": int(
                rounds_completed_in_episode
            ),
            "global_round": int(global_round),
            "global_slot": int(global_slot),
            "update_idx": int(update_idx),
            "env_config": env_cfg.as_dict(),
            "joint_train_config": run_cfg.to_dict(),
            "ppo_reward_scale": float(
                ppo_reward_scale
            ),
            "env_state": copy.deepcopy(env.__dict__),
            "ppo_buffer_state": _capture_buffer(agent),
            "pending_slow_selected": copy.deepcopy(
                pending_slow_selected
            ),
            "episode_metrics": copy.deepcopy(
                episode_metrics
            ),
            "aggregate_metrics": copy.deepcopy(
                aggregate_metrics
            ),
            "rng_state": _capture_rng_state(),
        },
    )


def _write_update_log(
    logger: ScalarLogger,
    *,
    update_idx: int,
    global_slot: int,
    current_episode: int,
    rounds_completed_in_episode: int,
    update_logs: Mapping[str, float],
) -> None:
    logger.write(
        {
            "update": int(update_idx),
            "global_slot": int(global_slot),
            "episode": int(current_episode),
            "round_in_episode": int(
                rounds_completed_in_episode
            ),
            **dict(update_logs),
        }
    )
    print(
        "[JOINT FAST UPDATE] "
        f"update={update_idx} "
        f"slot={global_slot} "
        f"episode={current_episode} "
        f"round={rounds_completed_in_episode} "
        f"policy_loss={update_logs['policy_loss']:.6f} "
        f"value_loss={update_logs['value_loss']:.6f} "
        f"cat_entropy={update_logs['categorical_entropy']:.6f} "
        f"power_entropy={update_logs['power_entropy']:.6f} "
        f"kl={update_logs['approx_kl']:.6f} "
        f"clipfrac={update_logs['clipfrac']:.4f} "
        f"ev={update_logs['explained_variance']:.4f} "
        f"minibatches="
        f"{int(update_logs['completed_minibatches'])}",
        flush=True,
    )


def run(run_cfg: SlowJointTrainConfig) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    _install_signal_handlers()

    set_seed(
        int(run_cfg.seed),
        deterministic=bool(
            run_cfg.deterministic_torch
        ),
    )

    is_resume = run_cfg.resume_checkpoint is not None
    checkpoint_path: Optional[Path] = None
    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_extra: Optional[Dict[str, Any]] = None
    if is_resume:
        checkpoint_path = _resolve_path(
            str(run_cfg.resume_checkpoint)
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Joint resume checkpoint does not exist: "
                f"{checkpoint_path}"
            )
        checkpoint = _load_trusted_checkpoint(
            checkpoint_path
        )
        checkpoint_extra = dict(
            checkpoint.get("extra", {})
        )
        if checkpoint_extra.get(
            "trainer_kind"
        ) != JOINT_TRAINER_KIND:
            raise RuntimeError(
                "resume_checkpoint was not produced by the current "
                "from-scratch joint trainer."
            )
        if checkpoint_extra.get(
            "fast_initialization"
        ) != FAST_INITIALIZATION:
            raise RuntimeError(
                "resume_checkpoint did not originate from a random "
                "Fast PPO joint initialization."
            )

    env_cfg = _make_env_config(
        run_cfg=run_cfg,
        checkpoint_extra=checkpoint_extra,
    )
    probe_env = Env(env_cfg)
    sample_fast_obs, _ = split_env_reset(
        probe_env.reset()
    )

    fast_agent = _build_fast_agent(
        run_cfg=run_cfg,
        env_cfg=env_cfg,
        sample_fast_obs=sample_fast_obs,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
    )
    if (
        str(run_cfg.device).startswith("cuda")
        and fast_agent.device.type != "cuda"
    ):
        raise RuntimeError(
            "CUDA joint training was requested, but FastPPOAgent "
            f"resolved device={fast_agent.device}."
        )
    ppo_reward_scale = _resolve_ppo_reward_scale(
        run_cfg=run_cfg,
        checkpoint_extra=checkpoint_extra,
    )

    controller = SlowDPPController(
        env_cfg=env_cfg,
        dpp_cfg=run_cfg,
    )
    env = Env(env_cfg)

    run_dir = _resolve_path(
        Path(run_cfg.output_root) / run_cfg.run_name
    )
    if is_resume:
        if checkpoint_extra is None:
            raise RuntimeError(
                "Internal error: resume metadata is unavailable."
            )
        saved_run_dir_value = checkpoint_extra.get(
            "run_dir"
        )
        if not isinstance(saved_run_dir_value, str):
            raise RuntimeError(
                "Exact resume checkpoint has no run_dir metadata."
            )
        saved_run_dir = _resolve_path(
            saved_run_dir_value
        )
        if saved_run_dir != run_dir:
            raise RuntimeError(
                "Exact resume requires the original run directory: "
                f"configured={run_dir}, saved={saved_run_dir}"
            )
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    episode_csv = log_dir / "episodes.csv"
    round_csv = log_dir / "rounds.csv"
    update_csv = log_dir / "updates.csv"

    if not is_resume:
        existing_outputs = [
            path
            for path in (
                episode_csv,
                round_csv,
                update_csv,
            )
            if path.is_file() and path.stat().st_size > 0
        ]
        if existing_outputs:
            raise FileExistsError(
                "Fresh joint run refuses to append to existing logs. "
                "Change run_name or set resume_checkpoint: "
                + ", ".join(
                    str(path)
                    for path in existing_outputs
                )
            )

    save_json(
        run_cfg.to_dict(),
        run_dir / "joint_train_config.json",
    )
    save_json(
        env_cfg.as_dict(),
        run_dir / "env_config.json",
    )
    save_json(
        {
            "trainer_kind": JOINT_TRAINER_KIND,
            "fast_initialization": FAST_INITIALIZATION,
            "loaded_resume_checkpoint": (
                None
                if checkpoint_path is None
                else str(checkpoint_path)
            ),
            "fast_only_checkpoint_used": False,
            "slow_policy": "none",
            "slow_controller": (
                "round_dpp_coordinate_minimization"
            ),
            "fast_policy": "online_mixed_action_ppo",
            "ppo_reward_scale": float(
                ppo_reward_scale
            ),
        },
        run_dir / "run_info.json",
    )

    episode_logger = ScalarLogger(episode_csv)
    round_logger = ScalarLogger(round_csv)
    update_logger = ScalarLogger(update_csv)

    if is_resume:
        if checkpoint_extra is None:
            raise RuntimeError(
                "Internal error: resume metadata is unavailable."
            )
        required = (
            "env_state",
            "ppo_buffer_state",
            "episode_metrics",
            "aggregate_metrics",
            "rng_state",
            "current_episode",
            "rounds_completed_in_episode",
            "global_round",
            "global_slot",
            "update_idx",
        )
        missing = [
            key
            for key in required
            if key not in checkpoint_extra
        ]
        if missing:
            raise RuntimeError(
                "Joint resume checkpoint is incomplete: "
                + ", ".join(missing)
            )

        env.__dict__.clear()
        env.__dict__.update(
            copy.deepcopy(
                checkpoint_extra["env_state"]
            )
        )
        _restore_buffer(
            fast_agent,
            checkpoint_extra["ppo_buffer_state"],
        )

        current_episode = int(
            checkpoint_extra["current_episode"]
        )
        rounds_completed = int(
            checkpoint_extra[
                "rounds_completed_in_episode"
            ]
        )
        global_round = int(
            checkpoint_extra["global_round"]
        )
        global_slot = int(
            checkpoint_extra["global_slot"]
        )
        update_idx = int(
            checkpoint_extra["update_idx"]
        )
        pending_slow_selected = copy.deepcopy(
            checkpoint_extra.get(
                "pending_slow_selected"
            )
        )
        episode_metrics = copy.deepcopy(
            checkpoint_extra["episode_metrics"]
        )
        aggregate = copy.deepcopy(
            checkpoint_extra["aggregate_metrics"]
        )

        if not isinstance(
            episode_metrics,
            SlowDPPMetrics,
        ) or not isinstance(
            aggregate,
            SlowDPPMetrics,
        ):
            raise TypeError(
                "Saved metric state has an unexpected type."
            )
        if int(env.round_slot) != 0:
            raise RuntimeError(
                "Saved environment is not at a round boundary."
            )
        if (
            rounds_completed
            < int(run_cfg.rounds_per_episode)
            and pending_slow_selected is None
        ):
            raise RuntimeError(
                "Resume checkpoint has no pending slow action."
            )
        if _last_csv_int(
            round_csv,
            "global_round",
        ) not in (None, global_round):
            raise RuntimeError(
                "rounds.csv is ahead of or behind the resume checkpoint. "
                "Use the matching run directory/checkpoint."
            )
        if _last_csv_int(
            update_csv,
            "update",
        ) not in (None, update_idx):
            raise RuntimeError(
                "updates.csv is ahead of or behind the resume checkpoint."
            )

        _restore_rng_state(
            checkpoint_extra["rng_state"]
        )

        if rounds_completed == int(
            run_cfg.rounds_per_episode
        ):
            if current_episode >= int(
                run_cfg.num_episodes
            ):
                raise RuntimeError(
                    "The resume checkpoint already completed the "
                    "configured num_episodes. Increase num_episodes "
                    "to continue training."
                )
            split_env_reset(env.reset())
            current_episode += 1
            rounds_completed = 0
            episode_metrics = _new_metrics(env_cfg)
            pending_slow_selected = _select_slow_action(
                controller=controller,
                env=env,
                fast_agent=fast_agent,
            )
            _apply_selected_slow_action(
                env=env,
                selected=pending_slow_selected,
            )
    else:
        split_env_reset(env.reset())
        current_episode = 1
        rounds_completed = 0
        global_round = 0
        global_slot = 0
        update_idx = 0
        episode_metrics = _new_metrics(env_cfg)
        aggregate = _new_metrics(env_cfg)
        _prime_fresh_obs_normalizer(
            fast_agent=fast_agent,
            obs=env.get_fast_obs(),
        )
        pending_slow_selected = _select_slow_action(
            controller=controller,
            env=env,
            fast_agent=fast_agent,
        )
        _apply_selected_slow_action(
            env=env,
            selected=pending_slow_selected,
        )

    print("=" * 100, flush=True)
    print("[SLOW DPP + FAST PPO JOINT TRAIN START]", flush=True)
    print(f"run_dir              : {run_dir}", flush=True)
    print(f"fast_initialization  : {FAST_INITIALIZATION}", flush=True)
    print(f"fast_only_checkpoint : not used", flush=True)
    print(f"resume_checkpoint    : {checkpoint_path}", flush=True)
    print(f"resume                : {is_resume}", flush=True)
    print(f"device                : {fast_agent.device}", flush=True)
    print(f"slow_policy           : none", flush=True)
    print(f"slow_controller       : DPP coordinate minimization", flush=True)
    print(f"fast_policy           : trainable mixed-action PPO", flush=True)
    print(f"slow_T                : {env_cfg.slow_T}", flush=True)
    print(f"episode_slots         : {env_cfg.episode_slots}", flush=True)
    print(f"rounds_per_episode    : {run_cfg.rounds_per_episode}", flush=True)
    print(f"forecast_scenarios    : {run_cfg.forecast_scenarios}", flush=True)
    print(f"rollout_steps         : {fast_agent.ppo_cfg.rollout_steps}", flush=True)
    print(f"ppo_reward_scale      : {ppo_reward_scale}", flush=True)
    print(f"start_episode         : {current_episode}", flush=True)
    print(f"start_round           : {rounds_completed}", flush=True)
    print("=" * 100, flush=True)

    while current_episode <= int(run_cfg.num_episodes):
        if rounds_completed >= int(
            run_cfg.rounds_per_episode
        ):
            raise RuntimeError(
                "Trainer entered an already completed episode."
            )
        if pending_slow_selected is None:
            raise RuntimeError(
                "No slow action is prepared for the current round."
            )

        selected_slow = pending_slow_selected
        pending_slow_selected = None
        round_metrics = _new_metrics(env_cfg)
        boundary_info: Optional[Dict[str, Any]] = None
        episode_done = False
        next_obs: Optional[Dict[str, Any]] = None

        for slot_in_round in range(
            int(env_cfg.slow_T)
        ):
            fast_agent.model.train()
            fast_selected = fast_agent.select_action(
                env.get_fast_obs(),
                deterministic=False,
                update_norm=True,
            )
            (
                next_obs,
                raw_reward,
                terminated,
                truncated,
                info,
            ) = split_env_step(
                env.step(
                    fast_selected["env_action"]
                )
            )

            is_last_configured_round = (
                rounds_completed + 1
                == int(run_cfg.rounds_per_episode)
            )
            is_last_slot = (
                slot_in_round + 1
                == int(env_cfg.slow_T)
            )
            episode_done = bool(
                terminated
                or truncated
                or (
                    is_last_configured_round
                    and is_last_slot
                )
            )

            fast_agent.store_transition(
                obs_vec=fast_selected["obs_vec"],
                raw_action=fast_selected["raw_action"],
                action_mask=fast_selected["action_mask"],
                reward=(
                    float(raw_reward)
                    * float(ppo_reward_scale)
                ),
                done=episode_done,
                value=float(fast_selected["value"]),
                log_prob=float(
                    fast_selected["log_prob"]
                ),
            )
            round_metrics.add_slot(
                info=info,
                fast_selected=fast_selected,
                num_layers=int(env_cfg.layer),
            )
            global_slot += 1

            if bool(
                info.get("is_round_boundary", False)
            ):
                boundary_info = dict(info)
                break
            if terminated or truncated:
                raise RuntimeError(
                    "Episode ended before a complete slow round."
                )

        if boundary_info is None or next_obs is None:
            raise RuntimeError(
                "Joint Fast loop did not reach a slow-round boundary."
            )
        if int(round_metrics.slots) != int(
            env_cfg.slow_T
        ):
            raise RuntimeError(
                "Incomplete realized slow round: "
                f"expected={env_cfg.slow_T}, "
                f"got={round_metrics.slots}"
            )

        rounds_completed += 1
        global_round += 1

        slow_reward, slow_components = _extract_slow_reward(
            boundary_info
        )
        realized_cost = -float(slow_reward)
        predicted_cost = float(
            selected_slow[
                "predicted_round_dpp_cost"
            ]
        )
        prediction_std = float(
            selected_slow[
                "predicted_round_dpp_cost_std"
            ]
        )
        round_metrics.add_round(
            slow_reward=slow_reward,
            slow_components=slow_components,
            slow_selected=selected_slow,
            predicted_cost=predicted_cost,
            realized_cost=realized_cost,
        )
        episode_metrics.merge(round_metrics)
        aggregate.merge(round_metrics)

        round_logger.write(
            {
                "episode": int(current_episode),
                "round_in_episode": int(
                    rounds_completed
                ),
                "global_round": int(global_round),
                "global_slot": int(global_slot),
                "ppo_update_before_round": int(
                    update_idx
                ),
                "predicted_round_dpp_cost": (
                    predicted_cost
                ),
                "predicted_round_dpp_cost_std": (
                    prediction_std
                ),
                "realized_round_dpp_cost": (
                    realized_cost
                ),
                "round_dpp_prediction_error": (
                    realized_cost - predicted_cost
                ),
                "round_dpp_prediction_abs_error": abs(
                    realized_cost - predicted_cost
                ),
                **selected_slow["action_info"],
                **round_metrics.summary(),
            }
        )

        if (
            global_round
            % int(run_cfg.log_every_rounds)
            == 0
        ):
            print(
                "[JOINT ROUND] "
                f"ep={current_episode}/{run_cfg.num_episodes} "
                "round="
                f"{rounds_completed}/"
                f"{run_cfg.rounds_per_episode} "
                f"global_round={global_round} "
                f"pred={predicted_cost:.3f} "
                f"real={realized_cost:.3f} "
                "hired="
                f"{selected_slow['action_info']['effective_hired_uav']} "
                "unique_candidates="
                f"{selected_slow['action_info']['unique_candidates']} "
                f"buffer={len(fast_agent.buffer)}/"
                f"{fast_agent.buffer.capacity}",
                flush=True,
            )

        update_logs: Optional[Dict[str, float]] = None

        if episode_done:
            if not bool(boundary_info.get("truncated", False)):
                raise RuntimeError(
                    "Configured episode ended without Env truncation."
                )
            if rounds_completed != int(
                run_cfg.rounds_per_episode
            ):
                raise RuntimeError(
                    "Episode ended with the wrong round count."
                )

            is_final_configured_episode = (
                current_episode
                == int(run_cfg.num_episodes)
            )
            if (
                fast_agent.buffer.is_full
                or (
                    is_final_configured_episode
                    and len(fast_agent.buffer) > 0
                )
            ):
                fast_agent.finish_rollout(
                    last_obs=next_obs,
                    last_done=True,
                )
                update_logs = fast_agent.update()
                update_idx += 1
                _write_update_log(
                    update_logger,
                    update_idx=update_idx,
                    global_slot=global_slot,
                    current_episode=current_episode,
                    rounds_completed_in_episode=(
                        rounds_completed
                    ),
                    update_logs=update_logs,
                )
        else:
            if int(env.round_slot) != 0:
                raise RuntimeError(
                    "Next slow decision is not at a round boundary."
                )

            if fast_agent.buffer.is_full:
                # Bootstrap the completed old-policy rollout with a
                # provisional next slow action selected by that same policy.
                provisional = _select_slow_action(
                    controller=controller,
                    env=env,
                    fast_agent=fast_agent,
                )
                _apply_selected_slow_action(
                    env=env,
                    selected=provisional,
                )
                bootstrap_obs = env.get_fast_obs()

                fast_agent.finish_rollout(
                    last_obs=bootstrap_obs,
                    last_done=False,
                )
                update_logs = fast_agent.update()
                update_idx += 1
                _write_update_log(
                    update_logger,
                    update_idx=update_idx,
                    global_slot=global_slot,
                    current_episode=current_episode,
                    rounds_completed_in_episode=(
                        rounds_completed
                    ),
                    update_logs=update_logs,
                )

                # The next realized round must use the newly updated Fast
                # policy, so recompute and overwrite the provisional action.
                pending_slow_selected = _select_slow_action(
                    controller=controller,
                    env=env,
                    fast_agent=fast_agent,
                )
                _apply_selected_slow_action(
                    env=env,
                    selected=pending_slow_selected,
                )
            else:
                pending_slow_selected = _select_slow_action(
                    controller=controller,
                    env=env,
                    fast_agent=fast_agent,
                )
                _apply_selected_slow_action(
                    env=env,
                    selected=pending_slow_selected,
                )

        if (
            update_logs is not None
            and update_idx
            % int(run_cfg.save_every_updates)
            == 0
            and not episode_done
        ):
            checkpoint_out = (
                checkpoint_dir
                / (
                    f"joint_ep{current_episode}"
                    f"_round{rounds_completed}"
                    f"_update{update_idx}.pt"
                )
            )
            _save_joint_checkpoint(
                path=checkpoint_out,
                agent=fast_agent,
                run_cfg=run_cfg,
                env_cfg=env_cfg,
                env=env,
                run_dir=run_dir,
                ppo_reward_scale=ppo_reward_scale,
                current_episode=current_episode,
                rounds_completed_in_episode=(
                    rounds_completed
                ),
                global_round=global_round,
                global_slot=global_slot,
                update_idx=update_idx,
                pending_slow_selected=(
                    pending_slow_selected
                ),
                episode_metrics=episode_metrics,
                aggregate_metrics=aggregate,
            )
            print(
                f"[JOINT SAVE] {checkpoint_out}",
                flush=True,
            )

        if episode_done:
            episode_summary = episode_metrics.summary()
            episode_logger.write(
                {
                    "episode": int(current_episode),
                    "global_round": int(global_round),
                    "global_slot": int(global_slot),
                    "ppo_updates": int(update_idx),
                    **episode_summary,
                }
            )

            if (
                current_episode
                % int(run_cfg.log_every_episodes)
                == 0
            ):
                print(
                    "[JOINT EPISODE] "
                    f"ep={current_episode}/"
                    f"{run_cfg.num_episodes} "
                    "dpp_cost/round="
                    f"{episode_summary['round_dpp_cost_mean']:.3f} "
                    "delivery/slot="
                    f"{episode_summary['delivery_per_slot']:.4f} "
                    "degradation/chunk="
                    f"{episode_summary['quality_degradation_per_chunk']:.4f} "
                    "scheduled_stall="
                    f"{episode_summary['scheduled_stall_rate']:.6f} "
                    "hired/round="
                    f"{episode_summary['hired_uav_per_round']:.3f} "
                    "forecast_mae="
                    f"{episode_summary['prediction_mae']:.3f}",
                    flush=True,
                )

            completed_episode = int(current_episode)
            if completed_episode >= int(
                run_cfg.num_episodes
            ):
                break

            split_env_reset(env.reset())
            current_episode += 1
            rounds_completed = 0
            episode_metrics = _new_metrics(env_cfg)
            pending_slow_selected = _select_slow_action(
                controller=controller,
                env=env,
                fast_agent=fast_agent,
            )
            _apply_selected_slow_action(
                env=env,
                selected=pending_slow_selected,
            )

            episode_checkpoint = (
                checkpoint_dir
                / (
                    f"joint_after_ep{completed_episode}"
                    f"_update{update_idx}.pt"
                )
            )
            _save_joint_checkpoint(
                path=episode_checkpoint,
                agent=fast_agent,
                run_cfg=run_cfg,
                env_cfg=env_cfg,
                env=env,
                run_dir=run_dir,
                ppo_reward_scale=ppo_reward_scale,
                current_episode=current_episode,
                rounds_completed_in_episode=0,
                global_round=global_round,
                global_slot=global_slot,
                update_idx=update_idx,
                pending_slow_selected=(
                    pending_slow_selected
                ),
                episode_metrics=episode_metrics,
                aggregate_metrics=aggregate,
            )
            print(
                f"[JOINT SAVE] {episode_checkpoint}",
                flush=True,
            )

        if STOP_REQUESTED:
            signal_checkpoint = (
                checkpoint_dir
                / (
                    f"joint_signal_ep{current_episode}"
                    f"_round{rounds_completed}"
                    f"_update{update_idx}.pt"
                )
            )
            _save_joint_checkpoint(
                path=signal_checkpoint,
                agent=fast_agent,
                run_cfg=run_cfg,
                env_cfg=env_cfg,
                env=env,
                run_dir=run_dir,
                ppo_reward_scale=ppo_reward_scale,
                current_episode=current_episode,
                rounds_completed_in_episode=(
                    rounds_completed
                ),
                global_round=global_round,
                global_slot=global_slot,
                update_idx=update_idx,
                pending_slow_selected=(
                    pending_slow_selected
                ),
                episode_metrics=episode_metrics,
                aggregate_metrics=aggregate,
            )
            save_json(
                {
                    "status": "stopped_after_signal",
                    "signal_checkpoint": str(
                        signal_checkpoint
                    ),
                    "current_episode": int(
                        current_episode
                    ),
                    "rounds_completed_in_episode": int(
                        rounds_completed
                    ),
                    "global_round": int(global_round),
                    "global_slot": int(global_slot),
                    "ppo_updates": int(update_idx),
                },
                run_dir / "interrupted_summary.json",
            )
            print(
                "[JOINT STOPPED SAFELY] "
                f"resume_checkpoint={signal_checkpoint}",
                flush=True,
            )
            return

    if len(fast_agent.buffer) > 0:
        final_obs = env.get_fast_obs()
        fast_agent.finish_rollout(
            last_obs=final_obs,
            last_done=True,
        )
        final_update_logs = fast_agent.update()
        update_idx += 1
        _write_update_log(
            update_logger,
            update_idx=update_idx,
            global_slot=global_slot,
            current_episode=current_episode,
            rounds_completed_in_episode=(
                rounds_completed
            ),
            update_logs=final_update_logs,
        )

    final_checkpoint = (
        checkpoint_dir / "joint_fast_ppo_final.pt"
    )
    _save_joint_checkpoint(
        path=final_checkpoint,
        agent=fast_agent,
        run_cfg=run_cfg,
        env_cfg=env_cfg,
        env=env,
        run_dir=run_dir,
        ppo_reward_scale=ppo_reward_scale,
        current_episode=current_episode,
        rounds_completed_in_episode=(
            rounds_completed
        ),
        global_round=global_round,
        global_slot=global_slot,
        update_idx=update_idx,
        pending_slow_selected=None,
        episode_metrics=episode_metrics,
        aggregate_metrics=aggregate,
    )

    final_summary = aggregate.summary()
    save_json(
        final_summary,
        run_dir / "final_summary.json",
    )
    save_json(
        {
            "trainer_kind": JOINT_TRAINER_KIND,
            "episodes": int(run_cfg.num_episodes),
            "global_round": int(global_round),
            "global_slot": int(global_slot),
            "ppo_updates": int(update_idx),
            "fast_initialization": FAST_INITIALIZATION,
            "fast_only_checkpoint_used": False,
            "final_checkpoint": str(
                final_checkpoint
            ),
        },
        run_dir / "train_summary.json",
    )

    print("=" * 100, flush=True)
    print("[SLOW DPP + FAST PPO JOINT TRAIN DONE]", flush=True)
    print(f"final checkpoint: {final_checkpoint}", flush=True)
    print(final_summary, flush=True)
    print("=" * 100, flush=True)


def main() -> None:
    run(get_slow_joint_train_config())


if __name__ == "__main__":
    main()