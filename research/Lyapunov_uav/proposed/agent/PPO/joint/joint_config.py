from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from dataclasses import replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SlowJointTrainConfig:
    """
    Slow DPP + Fast PPO online joint-training settings.

    The slow side is a non-learning DPP controller. Only the Fast PPO
    parameters are updated. A converged Fast-only checkpoint is used as the
    initialization point.
    """

    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"

    output_root: str = "joint"
    run_name: str = "slow_dpp_fast_ppo_joint_seed2026"

    # Fast-only pretraining checkpoint. Network shape, observation normalizer,
    # entropy coefficients, gamma/GAE, and PPO reward scale are restored from
    # this trusted checkpoint. Final joint environment/timescale values below
    # intentionally override the pretraining environment.
    fast_checkpoint: str = (
        "fast/"
        "fast_mixed_seed2026_continuous_mobility_slot1s_noklstop/"
        "checkpoints/fast_ppo_final.pt"
    )

    # Set this only when resuming a checkpoint produced by slow_joint_train.py.
    # The saved environment, PPO buffer, RNG, optimizer, episode, and pending
    # slow action are restored.
    resume_checkpoint: Optional[str] = None
    load_pretrained_optimizer: bool = False

    # Final formulation: 1 s/slot, 3,600 slots/round, 10 rounds/episode.
    num_episodes: int = 10
    rounds_per_episode: int = 10
    final_slow_T: int = 3_600
    final_mobility_mode: str = "fsmc"
    final_move_prob: float = 1e-4
    final_uav_hiring_cost: float = 5_000.0
    final_target_service_slots_per_round: int = 3_600

    # The source checkpoint used for this run was trained with the following
    # environment. These values are guards, not final formulation values.
    expected_source_slow_T: int = 2
    expected_source_mobility_mode: str = "continuous"

    # Expected round-DPP estimator used during training. K=1 keeps exact
    # candidate enumeration but uses one common-random-number forecast sample.
    # Use K>=4 for final evaluation after training, not for the first full run.
    forecast_scenarios: int = 1
    forecast_seed_offset: int = 20_000_000
    forecast_fast_deterministic: bool = False

    # Region-local candidates are exhaustively enumerated. Exceeding this
    # fail-fast guard raises an error; it never prunes to top-k.
    max_exact_region_candidates: int = 8192

    # Whole-system coordinate minimization over complete region candidates.
    max_coordinate_sweeps: int = 10
    dpp_tie_atol: float = 1e-3
    dpp_tie_rtol: float = 1e-9

    # One Fast-PPO update is performed after each complete slow round.
    fast_rollout_steps: int = 3_600
    fast_batch_size: int = 450
    fast_update_epochs: int = 4
    fast_lr: float = 3e-5
    fast_clip_coef: float = 0.15
    fast_target_kl: Optional[float] = 0.03
    fast_max_grad_norm: float = 0.5

    # Save an exact-resume joint checkpoint after this many PPO updates and at
    # every episode boundary.
    save_every_updates: int = 5
    log_every_rounds: int = 1
    log_every_episodes: int = 1

    # If None, restore the value from the Fast-only checkpoint's train_config.
    # Do not silently change reward scale between pretraining and joint train.
    ppo_reward_scale: Optional[float] = None

    # Used only for old checkpoints that lack Fast PPO metadata.
    fast_hidden_dims_fallback: Tuple[int, ...] = (256, 256)
    fast_init_log_std_fallback: float = -1.0

    def __post_init__(self) -> None:
        for name in (
            "num_episodes",
            "rounds_per_episode",
            "forecast_scenarios",
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

        if int(self.forecast_seed_offset) < 0:
            raise ValueError(
                "forecast_seed_offset must be nonnegative."
            )
        for name in ("dpp_tie_atol", "dpp_tie_rtol"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if not 0.0 <= float(self.final_move_prob) <= 1.0:
            raise ValueError(
                "final_move_prob must be in [0, 1]."
            )
        if self.final_mobility_mode != "fsmc":
            raise ValueError(
                "The final formulation requires final_mobility_mode='fsmc'."
            )
        if float(self.final_uav_hiring_cost) < 0.0:
            raise ValueError(
                "final_uav_hiring_cost must be nonnegative."
            )
        if int(self.fast_rollout_steps) != int(self.final_slow_T):
            raise ValueError(
                "fast_rollout_steps must equal final_slow_T so one update "
                "uses exactly one completed slow round."
            )
        if int(self.fast_rollout_steps) % int(self.fast_batch_size) != 0:
            raise ValueError(
                "fast_rollout_steps must be divisible by fast_batch_size."
            )
        if float(self.fast_lr) <= 0.0:
            raise ValueError("fast_lr must be positive.")
        if not 0.0 < float(self.fast_clip_coef) < 1.0:
            raise ValueError("fast_clip_coef must be in (0, 1).")
        if (
            self.fast_target_kl is not None
            and float(self.fast_target_kl) <= 0.0
        ):
            raise ValueError(
                "fast_target_kl must be None or positive."
            )
        if float(self.fast_max_grad_norm) <= 0.0:
            raise ValueError(
                "fast_max_grad_norm must be positive."
            )
        if not self.fast_checkpoint:
            raise ValueError(
                "fast_checkpoint must not be empty."
            )
        if (
            self.ppo_reward_scale is not None
            and float(self.ppo_reward_scale) <= 0.0
        ):
            raise ValueError(
                "ppo_reward_scale must be None or positive."
            )
        if not self.fast_hidden_dims_fallback:
            raise ValueError(
                "fast_hidden_dims_fallback must not be empty."
            )
        if any(
            int(value) <= 0
            for value in self.fast_hidden_dims_fallback
        ):
            raise ValueError(
                "fast_hidden_dims_fallback entries must be positive."
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_slow_joint_train_config() -> SlowJointTrainConfig:
    cfg = SlowJointTrainConfig()
    fast_checkpoint = os.environ.get(
        "JOINT_FAST_CHECKPOINT"
    )
    resume_checkpoint = os.environ.get(
        "JOINT_RESUME_CHECKPOINT"
    )
    if fast_checkpoint:
        cfg = replace(
            cfg,
            fast_checkpoint=fast_checkpoint,
        )
    if resume_checkpoint:
        cfg = replace(
            cfg,
            resume_checkpoint=resume_checkpoint,
        )
    return cfg
