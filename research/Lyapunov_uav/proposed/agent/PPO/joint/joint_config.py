from __future__ import annotations

from dataclasses import asdict, dataclass
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

    # Fast-only pretraining checkpoint. The checkpoint metadata is the source
    # of truth for EnvConfig, FastPPOConfig, and PPO reward scaling.
    fast_checkpoint: str = (
        "fast/"
        "fast_mixed_seed2026_continuous_mobility_slot1s_noklstop/"
        "checkpoints/fast_ppo_final.pt"
    )

    # Set this only when resuming a checkpoint produced by slow_joint_train.py.
    # The saved environment, PPO buffer, RNG, optimizer, episode, and pending
    # slow action are restored.
    resume_checkpoint: Optional[str] = None
    load_pretrained_optimizer: bool = True

    # Joint fine-tuning horizon. With the current checkpoint slow_T=2,
    # one episode remains 18,000 rounds x 2 slots = 36,000 slots, matching
    # Fast-only pretraining.
    num_episodes: int = 10
    rounds_per_episode: int = 18_000

    # Expected round-DPP estimator used during training. K=1 keeps exact
    # candidate enumeration but uses one common-random-number forecast sample.
    # Use K>=4 for final evaluation after training, not for the first full run.
    forecast_scenarios: int = 1
    forecast_seed_offset: int = 20_000_000

    # Region-local candidates are exhaustively enumerated. Exceeding this
    # fail-fast guard raises an error; it never prunes to top-k.
    max_exact_region_candidates: int = 8192

    # Whole-system coordinate minimization over complete region candidates.
    max_coordinate_sweeps: int = 10
    dpp_tie_tolerance: float = 1e-6

    # Save an exact-resume joint checkpoint after this many PPO updates and at
    # every episode boundary.
    save_every_updates: int = 5
    log_every_rounds: int = 100
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
        if float(self.dpp_tie_tolerance) < 0.0:
            raise ValueError(
                "dpp_tie_tolerance must be nonnegative."
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
    return SlowJointTrainConfig()