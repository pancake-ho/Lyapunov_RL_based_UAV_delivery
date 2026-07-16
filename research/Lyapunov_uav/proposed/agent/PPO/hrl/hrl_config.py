from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

from config import EnvConfig
from agent.PPO.slow.slow_agent import SlowPPOConfig


@dataclass(frozen=True)
class JointHRLConfig:
    # 1. Util
    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"

    output_root: str = "hrl"
    run_name: str = "joint_mobility_p1e4_seed2026"

    # fast-/slow-timescale checkpoint 초기화할 경우 대비
    initial_fast_checkpoint: str = (
        "fast/fast_final_static_cat5e4_seed2026/"
        "checkpoints/fast_ppo_ep40.pt"
    )
    initial_slow_checkpoint: Optional[str] = None
    resume_manifest: Optional[str] = None

    # 640eps + 10rounds + 3600slots (= 총 23,040,000번의 fast transitions)
    # 6400번의 slow transitions + 50번의 slow PPO 업데이트 진행
    num_episodes: int = 640
    rounds_per_episode: int = 10
    train_move_prob: float = 1e-4

    fast_freeze_rounds: int = 0
    fast_deterministic_while_frozen: bool = False

    # joint fast fine-tuning
    fast_rollout_rounds: int = 4
    fast_update_epochs: int = 2
    fast_batch_size: int = 900
    fast_gamma: float = 0.99
    fast_gae_lambda: float = 0.95
    fast_lr: float = 1e-5
    fast_clip_coef: float = 0.10
    fast_target_kl: Optional[float] = 0.01
    fast_value_coef: float = 0.5
    fast_categorical_entropy_coef: float = 5e-4
    fast_power_entropy_coef: float = 1e-4
    fast_max_grad_norm: float = 0.5
    fast_reward_scale: float = 1e-4
    fast_hidden_dims_fallback: Tuple[int, ...] = (256, 256)
    fast_init_log_std_fallback: float = -1.0

    # joint slow fine-tuning
    slow_rollout_rounds: int = 128
    slow_update_epochs: int = 4
    slow_batch_size: int = 32
    slow_gamma: float = 0.99
    slow_gae_lambda: float = 0.95
    slow_lr: float = 3e-5
    slow_clip_coef: float = 0.15
    slow_target_kl: Optional[float] = 0.02
    slow_value_coef: float = 0.5
    slow_entropy_coef: float = 1e-3
    slow_max_grad_norm: float = 0.5
    slow_reward_scale: float = 1e-6

    slow_hidden_dims: Tuple[int, ...] = (256, 256)
    rsu_init_logit: float = 0.0
    hiring_init_logit: float = 0.0
    uav_init_logit: float = 0.0
    min_logit: float = -20.0
    max_logit: float = 20.0

    obs_norm: bool = True
    adv_norm: bool = True
    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    fast_value_clip_coef: float = 0.5
    slow_value_clip_coef: float = 1.0
    fail_on_nan: bool = True

    # eval
    evaluate_every_episodes: int = 64
    save_every_episodes: int = 64

    selection_eval_seeds: Tuple[int, ...] = (2026, 2027, 2028)
    final_test_seeds: Tuple[int, ...] = (3031, 3032, 3033, 3034, 3035)
    eval_rounds_per_seed: int = 5
    
    selection_move_prob: float = 1e-4
    weak_mobility_move_prob: float = 1e-4
    fast_deterministic_eval: bool = True

    # baseline 대비
    baseline_rsu_user_prob: float = 0.50
    baseline_uav_hire_prob: float = 0.70
    baseline_uav_user_prob: float = 0.80

    # 안전장치
    min_reward_improvement_fraction: float = 0.02
    min_delivery_fraction_of_baseline: float = 0.90
    max_degradation_ratio_to_baseline: float = 1.05
    max_scheduled_stall_ratio_to_baseline: float = 1.20
    scheduled_stall_absolute_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        for name in (
            "num_episodes",
            "rounds_per_episode",
            "fast_rollout_rounds",
            "fast_update_epochs",
            "fast_batch_size",
            "slow_rollout_rounds",
            "slow_update_epochs",
            "slow_batch_size",
            "evaluate_every_episodes",
            "save_every_episodes",
            "eval_rounds_per_seed",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if int(self.fast_freeze_rounds) < 0:
            raise ValueError("fast_freeze_rounds must be non-negative.")
        if int(self.fast_freeze_rounds) % int(self.fast_rollout_rounds) != 0:
            raise ValueError(
                "fast_freeze_rounds must be divisible by fast_rollout_rounds."
            )
        if int(self.fast_freeze_rounds) % int(self.slow_rollout_rounds) != 0:
            raise ValueError(
                "fast_freeze_rounds must end after a complete slow PPO "
                "rollout."
            )
        for name in (
            "fast_gamma",
            "fast_gae_lambda",
        ):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        for name in (
            "fast_lr",
            "fast_max_grad_norm",
            "fast_reward_scale",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < float(self.fast_clip_coef) < 1.0:
            raise ValueError("fast_clip_coef must be in (0, 1).")
        if self.fast_target_kl is not None and float(self.fast_target_kl) <= 0.0:
            raise ValueError("fast_target_kl must be positive or None.")
        if min(
            float(self.fast_value_coef),
            float(self.fast_categorical_entropy_coef),
            float(self.fast_power_entropy_coef),
        ) < 0.0:
            raise ValueError("fast loss coefficients must be non-negative.")
        for name in (
            "train_move_prob",
            "selection_move_prob",
            "weak_mobility_move_prob",
            "baseline_rsu_user_prob",
            "baseline_uav_hire_prob",
            "baseline_uav_user_prob",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if not self.selection_eval_seeds or not self.final_test_seeds:
            raise ValueError("selection and final-test seeds must be non-empty.")
        if not 0.0 < float(self.min_delivery_fraction_of_baseline) <= 1.0:
            raise ValueError("min_delivery_fraction_of_baseline must be in (0, 1].")
        if float(self.min_reward_improvement_fraction) < 0.0:
            raise ValueError("min_reward_improvement_fraction must be non-negative.")
        if float(self.max_degradation_ratio_to_baseline) < 1.0:
            raise ValueError("max_degradation_ratio_to_baseline must be >= 1.")
        if float(self.max_scheduled_stall_ratio_to_baseline) < 1.0:
            raise ValueError("max_scheduled_stall_ratio_to_baseline must be >= 1.")
        self.make_slow_ppo_config()
    
    @property
    def eval_seeds(self) -> Tuple[int, ...]:
        """Compatibility view used by slow_train.evaluate_paired()."""
        return tuple(int(x) for x in self.selection_eval_seeds)

    @property
    def eval_move_prob(self) -> float:
        """Compatibility view used by slow_train.evaluate_paired()."""
        return float(self.selection_move_prob)

    def validate_environment(self, env_cfg: EnvConfig) -> None:
        fast_rollout_slots = int(env_cfg.slow_T) * int(self.fast_rollout_rounds)
        if fast_rollout_slots % int(self.fast_batch_size) != 0:
            raise ValueError(
                "slow_T * fast_rollout_rounds must be divisible by "
                "fast_batch_size; got "
                f"{env_cfg.slow_T} * {self.fast_rollout_rounds} and "
                f"batch={self.fast_batch_size}."
            )
        total_rounds = int(self.num_episodes) * int(self.rounds_per_episode)
        if int(self.fast_freeze_rounds) >= total_rounds:
            raise ValueError(
                "fast_freeze_rounds must be smaller than total training rounds."
            )
        if total_rounds % int(self.slow_rollout_rounds) != 0:
            raise ValueError(
                "num_episodes * rounds_per_episode must end at a complete "
                "slow rollout."
            )
        if (
            total_rounds - int(self.fast_freeze_rounds)
        ) % int(self.fast_rollout_rounds) != 0:
            raise ValueError("training must end at a complete fast rollout.")
        save_rounds = int(self.save_every_episodes) * int(
            self.rounds_per_episode
        )
        if save_rounds % int(self.slow_rollout_rounds) != 0:
            raise ValueError(
                "save_every_episodes must land on an empty slow buffer."
            )
        if save_rounds % int(self.fast_rollout_rounds) != 0:
            raise ValueError(
                "save_every_episodes must land on an empty fast buffer."
            )

    def make_slow_ppo_config(self) -> SlowPPOConfig:
        return SlowPPOConfig(
            rollout_rounds=int(self.slow_rollout_rounds),
            update_epochs=int(self.slow_update_epochs),
            batch_size=int(self.slow_batch_size),
            gamma=float(self.slow_gamma),
            gae_lambda=float(self.slow_gae_lambda),
            lr=float(self.slow_lr),
            max_grad_norm=float(self.slow_max_grad_norm),
            clip_coef=float(self.slow_clip_coef),
            value_coef=float(self.slow_value_coef),
            entropy_coef=float(self.slow_entropy_coef),
            target_kl=(
                None
                if self.slow_target_kl is None
                else float(self.slow_target_kl)
            ),
            normalize_obs=bool(self.obs_norm),
            normalize_adv=bool(self.adv_norm),
            reward_scale=float(self.slow_reward_scale),
            hidden_dims=tuple(int(x) for x in self.slow_hidden_dims),
            rsu_init_logit=float(self.rsu_init_logit),
            hiring_init_logit=float(self.hiring_init_logit),
            uav_init_logit=float(self.uav_init_logit),
            min_logit=float(self.min_logit),
            max_logit=float(self.max_logit),
            use_value_huber_loss=bool(self.use_value_huber_loss),
            use_value_clip=bool(self.use_value_clip),
            value_clip_coef=float(self.slow_value_clip_coef),
            fail_on_nan=bool(self.fail_on_nan),
            device=str(self.device),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_joint_hrl_config() -> JointHRLConfig:
    return JointHRLConfig()