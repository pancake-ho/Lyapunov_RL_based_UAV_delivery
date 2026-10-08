from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SlowJointTrainConfig:
    """
    Slow-DPP + Fast-PPO from-scratch joint-training settings.

    A fresh run initializes the Fast actor, critic, optimizer, observation
    normalizer, and rollout buffer from scratch. The Fast-only experiment is
    not used as a source checkpoint. The only supported load path is an exact
    resume checkpoint produced by this joint trainer.
    """

    # ------------------------------------------------------------------
    # 1) Reproducibility / device
    # ------------------------------------------------------------------
    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"

    # ------------------------------------------------------------------
    # 2) Output / exact resume
    # ------------------------------------------------------------------
    output_root: str = "joint"
    run_name: str = (
        "slow_dpp_fast_ppo_scratch_long_mp_seed2026"
    )

    # Set this only to resume a checkpoint produced by this trainer.
    # A fresh run must leave it as None.
    resume_checkpoint: Optional[str] = None

    # ------------------------------------------------------------------
    # 3) Final joint-training horizon and environment contract
    # ------------------------------------------------------------------
    # 100 episodes x 10 rounds x 3,600 slots:
    #   - 1,000 realized Slow rounds
    #   - 3,600,000 actual Fast slots
    #   - 1,000 Fast-PPO updates
    #
    # The current trainer additionally performs provisional Slow selections
    # for PPO bootstrap. With 10 rounds/episode, this yields approximately
    # 1,900 Slow-DPP selections over the complete run.
    num_episodes: int = 100
    rounds_per_episode: int = 10

    # Final formulation: 1 second/slot and 3,600 slots/Slow round.
    final_slow_T: int = 3_600
    final_mobility_mode: str = "fsmc"
    final_move_prob: float = 1e-4
    final_uav_hiring_cost: float = 5_000.0
    final_target_service_slots_per_round: int = 3_600

    # ------------------------------------------------------------------
    # 4) Non-learning Slow-DPP controller
    # ------------------------------------------------------------------
    # K=1 is the training-time compute compromise. All candidates use common
    # random numbers. Use K>=4 only for final frozen-policy evaluation.
    forecast_scenarios: int = 1
    forecast_seed_offset: int = 20_000_000
    forecast_fast_deterministic: bool = False

    # Exact candidates within the same coordinate are independent once the
    # base action is fixed. Evaluate them in lock-step so policy inference is
    # a GPU batch instead of repeated batch-1 calls. Environment steps use
    # independent trial objects and can therefore use the allocated CPUs.
    forecast_candidate_batch_size: int = 64
    forecast_env_workers: int = 16

    # Region-local candidates are exhaustively enumerated. Exceeding this
    # guard raises an error; it never prunes candidates to top-k.
    max_exact_region_candidates: int = 8_192

    # Whole-system coordinate minimization over complete region candidates.
    # The loop exits early as soon as a full sweep makes no change.
    max_coordinate_sweeps: int = 10
    dpp_tie_atol: float = 1e-3
    dpp_tie_rtol: float = 1e-9

    # ------------------------------------------------------------------
    # 5) Fast-PPO rollout and optimization
    # ------------------------------------------------------------------
    # One complete Slow round is one Fast rollout and one PPO update.
    # 3,600 / 450 = 8 minibatches/epoch; 4 epochs = at most 32 optimizer
    # steps per completed Slow round.
    fast_rollout_steps: int = 3_600
    fast_batch_size: int = 450
    fast_update_epochs: int = 4

    fast_gamma: float = 0.99
    fast_gae_lambda: float = 0.95

    # From-scratch learning rate follows the verified Fast-only setting.
    # target_kl protects a long joint run from rare oversized PPO updates.
    fast_lr: float = 6e-5
    fast_clip_coef: float = 0.15
    fast_target_kl: Optional[float] = 0.03
    fast_max_grad_norm: float = 0.5

    fast_value_coef: float = 0.5
    fast_categorical_entropy_coef: float = 1e-4
    fast_power_entropy_coef: float = 1e-5

    fast_normalize_obs: bool = True
    fast_normalize_adv: bool = True

    fast_hidden_dims: Tuple[int, ...] = (256, 256)
    fast_init_log_std: float = -1.0

    fast_use_value_huber_loss: bool = True
    fast_use_value_clip: bool = True
    fast_value_clip_coef: float = 0.5
    fast_fail_on_nan: bool = True

    # Env returns raw DPP reward. Only the PPO buffer uses this scale.
    ppo_reward_scale: float = 1e-4

    # ------------------------------------------------------------------
    # 6) Logging / exact-resume checkpoints
    # ------------------------------------------------------------------
    # Ten updates equal one configured episode. The trainer also saves at
    # every episode boundary, while a Slurm signal triggers a boundary-safe
    # checkpoint independently of this interval.
    save_every_updates: int = 10
    log_every_rounds: int = 1
    log_every_episodes: int = 1

    def __post_init__(self) -> None:
        for name in (
            "num_episodes",
            "rounds_per_episode",
            "forecast_scenarios",
            "forecast_candidate_batch_size",
            "forecast_env_workers",
            "max_exact_region_candidates",
            "max_coordinate_sweeps",
            "final_slow_T",
            "final_target_service_slots_per_round",
            "fast_rollout_steps",
            "fast_batch_size",
            "fast_update_epochs",
            "save_every_updates",
            "log_every_rounds",
            "log_every_episodes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")

        if not str(self.output_root).strip():
            raise ValueError("output_root must not be empty.")
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty.")
        if not str(self.device).strip():
            raise ValueError("device must not be empty.")

        if int(self.forecast_seed_offset) < 0:
            raise ValueError(
                "forecast_seed_offset must be nonnegative."
            )
        for name in ("dpp_tie_atol", "dpp_tie_rtol"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative."
                )

        if not 0.0 <= float(self.final_move_prob) <= 1.0:
            raise ValueError(
                "final_move_prob must be in [0, 1]."
            )
        if self.final_mobility_mode != "fsmc":
            raise ValueError(
                "The final formulation requires "
                "final_mobility_mode='fsmc'."
            )
        if (
            not math.isfinite(float(self.final_uav_hiring_cost))
            or float(self.final_uav_hiring_cost) < 0.0
        ):
            raise ValueError(
                "final_uav_hiring_cost must be finite and nonnegative."
            )
        if int(self.final_target_service_slots_per_round) != int(
            self.final_slow_T
        ):
            raise ValueError(
                "final_target_service_slots_per_round must equal "
                "final_slow_T for the current full-round service target."
            )

        if int(self.fast_rollout_steps) != int(self.final_slow_T):
            raise ValueError(
                "fast_rollout_steps must equal final_slow_T so one update "
                "uses exactly one completed Slow round."
            )
        if int(self.fast_batch_size) > int(self.fast_rollout_steps):
            raise ValueError(
                "fast_batch_size must not exceed fast_rollout_steps."
            )
        if int(self.fast_rollout_steps) % int(self.fast_batch_size) != 0:
            raise ValueError(
                "fast_rollout_steps must be divisible by fast_batch_size."
            )

        for name in (
            "fast_lr",
            "fast_max_grad_norm",
            "fast_value_clip_coef",
            "ppo_reward_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive."
                )

        for name in (
            "fast_value_coef",
            "fast_categorical_entropy_coef",
            "fast_power_entropy_coef",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and nonnegative."
                )

        if not 0.0 < float(self.fast_gamma) <= 1.0:
            raise ValueError("fast_gamma must be in (0, 1].")
        if not 0.0 < float(self.fast_gae_lambda) <= 1.0:
            raise ValueError(
                "fast_gae_lambda must be in (0, 1]."
            )
        if not 0.0 < float(self.fast_clip_coef) < 1.0:
            raise ValueError(
                "fast_clip_coef must be in (0, 1)."
            )
        if (
            self.fast_target_kl is not None
            and (
                not math.isfinite(float(self.fast_target_kl))
                or float(self.fast_target_kl) <= 0.0
            )
        ):
            raise ValueError(
                "fast_target_kl must be None or finite and positive."
            )

        if not self.fast_hidden_dims:
            raise ValueError(
                "fast_hidden_dims must not be empty."
            )
        if any(
            int(value) <= 0
            for value in self.fast_hidden_dims
        ):
            raise ValueError(
                "fast_hidden_dims entries must be positive."
            )
        if not math.isfinite(float(self.fast_init_log_std)):
            raise ValueError(
                "fast_init_log_std must be finite."
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_slow_joint_train_config() -> SlowJointTrainConfig:
    cfg = SlowJointTrainConfig()
    replacements: Dict[str, object] = {}
    resume_checkpoint = os.environ.get(
        "JOINT_RESUME_CHECKPOINT"
    )
    if resume_checkpoint:
        replacements["resume_checkpoint"] = resume_checkpoint

    batch_size = os.environ.get(
        "JOINT_FORECAST_CANDIDATE_BATCH_SIZE"
    )
    if batch_size:
        replacements["forecast_candidate_batch_size"] = int(
            batch_size
        )

    env_workers = os.environ.get(
        "JOINT_FORECAST_ENV_WORKERS"
    )
    if env_workers:
        replacements["forecast_env_workers"] = int(
            env_workers
        )

    return (
        replace(cfg, **replacements)
        if replacements
        else cfg
    )