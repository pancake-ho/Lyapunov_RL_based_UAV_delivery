from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from .slow_agent import SlowPPOConfig


@dataclass(frozen=True)
class SlowTrainConfig:
    """Execution config for frozen-fast / trainable-slow HRL."""

    mode: str = "train"  # train | eval
    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"

    output_root: str = "slow"
    run_name: str = "slow_static_frozen_fast_seed2026"

    # Trusted checkpoint produced by this repository.
    fast_checkpoint: str = (
        "fast/fast_final_static_cat5e4_seed2026/"
        "checkpoints/fast_ppo_ep200.pt"
    )
    slow_checkpoint: Optional[str] = None
    resume_slow: bool = False

    # A slow episode contains 10 one-hour rounds = 36,000 fast slots.
    num_episodes: int = 1000
    rounds_per_episode: int = 10
    train_move_prob: float = 0.0

    # Freeze the converged fast policy while learning the slow MDP.
    fast_deterministic_train: bool = True

    save_every_episodes: int = 10
    evaluate_every_episodes: int = 25

    # Paired deterministic evaluation.
    eval_seeds: Tuple[int, ...] = (2026, 2027, 2028)
    eval_rounds_per_seed: int = 5
    eval_move_prob: float = 0.0
    fast_deterministic_eval: bool = True

    # Baseline used to prevent reward-only no-service checkpoint selection.
    baseline_rsu_user_prob: float = 0.50
    baseline_uav_hire_prob: float = 0.70
    baseline_uav_user_prob: float = 0.80
    min_reward_improvement_fraction: float = 0.02
    min_delivery_fraction_of_baseline: float = 0.90
    max_degradation_ratio_to_baseline: float = 1.05
    max_scheduled_stall_ratio_to_baseline: float = 1.20
    scheduled_stall_absolute_tolerance: float = 1e-4

    # Slow PPO: 128 round transitions = 460,800 fast slots per update.
    rollout_rounds: int = 128
    update_epochs: int = 4
    batch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-5
    clip_coef: float = 0.15
    target_kl: Optional[float] = 0.02
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    max_grad_norm: float = 0.5
    reward_scale: float = 1e-6

    hidden_dims: Tuple[int, ...] = (256, 256)
    # Match the existing random-slow baseline at initialization.
    rsu_init_logit: float = 0.0
    hiring_init_logit: float = 0.0
    uav_init_logit: float = 0.0
    min_logit: float = -20.0
    max_logit: float = 20.0

    obs_norm: bool = True
    adv_norm: bool = True
    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    value_clip_coef: float = 1.0
    fail_on_nan: bool = True

    # Components copied only when old fast checkpoint metadata is incomplete.
    fast_hidden_dims_fallback: Tuple[int, ...] = (256, 256)
    fast_init_log_std_fallback: float = -1.0

    def __post_init__(self) -> None:
        if self.mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'.")
        for name in (
            "num_episodes",
            "rounds_per_episode",
            "save_every_episodes",
            "evaluate_every_episodes",
            "eval_rounds_per_seed",
            "rollout_rounds",
            "update_epochs",
            "batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.eval_seeds:
            raise ValueError("eval_seeds must not be empty.")
        for name in ("train_move_prob", "eval_move_prob"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        for name in (
            "baseline_rsu_user_prob",
            "baseline_uav_hire_prob",
            "baseline_uav_user_prob",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not 0.0 < float(self.min_delivery_fraction_of_baseline) <= 1.0:
            raise ValueError("min_delivery_fraction_of_baseline must be in (0, 1].")
        if float(self.min_reward_improvement_fraction) < 0.0:
            raise ValueError("min_reward_improvement_fraction must be >= 0.")
        if float(self.max_degradation_ratio_to_baseline) < 1.0:
            raise ValueError("max_degradation_ratio_to_baseline must be >= 1.")
        if float(self.max_scheduled_stall_ratio_to_baseline) < 1.0:
            raise ValueError("max_scheduled_stall_ratio_to_baseline must be >= 1.")
        if float(self.scheduled_stall_absolute_tolerance) < 0.0:
            raise ValueError("scheduled_stall_absolute_tolerance must be >= 0.")
        if self.resume_slow and self.slow_checkpoint is None:
            raise ValueError("resume_slow=True requires slow_checkpoint.")
        if self.mode == "eval" and self.slow_checkpoint is None:
            raise ValueError("eval mode requires slow_checkpoint.")

        # Reuse SlowPPOConfig validation for all PPO-specific fields.
        self.make_slow_ppo_config()

    def make_slow_ppo_config(self) -> SlowPPOConfig:
        return SlowPPOConfig(
            rollout_rounds=int(self.rollout_rounds),
            update_epochs=int(self.update_epochs),
            batch_size=int(self.batch_size),
            gamma=float(self.gamma),
            gae_lambda=float(self.gae_lambda),
            lr=float(self.lr),
            max_grad_norm=float(self.max_grad_norm),
            clip_coef=float(self.clip_coef),
            value_coef=float(self.value_coef),
            entropy_coef=float(self.entropy_coef),
            target_kl=(
                None if self.target_kl is None else float(self.target_kl)
            ),
            normalize_obs=bool(self.obs_norm),
            normalize_adv=bool(self.adv_norm),
            reward_scale=float(self.reward_scale),
            hidden_dims=tuple(int(x) for x in self.hidden_dims),
            rsu_init_logit=float(self.rsu_init_logit),
            hiring_init_logit=float(self.hiring_init_logit),
            uav_init_logit=float(self.uav_init_logit),
            min_logit=float(self.min_logit),
            max_logit=float(self.max_logit),
            use_value_huber_loss=bool(self.use_value_huber_loss),
            use_value_clip=bool(self.use_value_clip),
            value_clip_coef=float(self.value_clip_coef),
            fail_on_nan=bool(self.fail_on_nan),
            device=str(self.device),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_slow_train_config() -> SlowTrainConfig:
    return SlowTrainConfig()