from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SlowDPPConfig:
    """
    fixed slow-timescale DPP controller를 위한 실행 config class
    """
    seed: int = 2026
    deterministic_torch: bool = True
    device: str = "cuda"

    output_root: str = "slow"
    run_name: str = "slow_dpp_frozen_fast_seed2026"

    # Trusted checkpoint produced by this repository.
    fast_checkpoint: str = (
        "fast/fast_final_static_cat5e4_seed2026/"
        "checkpoints/fast_ppo_ep200.pt"
    )

    # A slow episode contains 10 one-hour rounds = 36,000 fast slots.
    num_episodes: int = 100
    rounds_per_episode: int = 10
    move_prob: float = 0.0005

    # Fixed algorithm: region-wise coordinate minimization. Each candidate
    # is evaluated by a complete T-slot rollout of the frozen fast PPO.
    max_coordinate_sweeps: int = 20
    forecast_scenarios: int = 1
    forecast_seed_offset: int = 10_000_000
    max_region_candidates: int = 4096

    log_every_episodes: int = 1

    # Used only if an older fast checkpoint lacks complete metadata.
    fast_hidden_dims_fallback: Tuple[int, ...] = (256, 256)
    fast_init_log_std_fallback: float = -1.0

    def __post_init__(self) -> None:
        for name in (
            "num_episodes",
            "rounds_per_episode",
            "max_coordinate_sweeps",
            "forecast_scenarios",
            "max_region_candidates",
            "log_every_episodes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")

        if not 0.0 <= float(self.move_prob) <= 1.0:
            raise ValueError("move_prob must be in [0, 1].")

        if int(self.forecast_seed_offset) < 0:
            raise ValueError("forecast_seed_offset must be nonnegative.")

        if not self.fast_checkpoint:
            raise ValueError("fast_checkpoint must not be empty.")

        if not self.fast_hidden_dims_fallback:
            raise ValueError("fast_hidden_dims_fallback must not be empty.")

        if any(int(x) <= 0 for x in self.fast_hidden_dims_fallback):
            raise ValueError(
                "fast_hidden_dims_fallback entries must be positive."
            )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def get_slow_dpp_config() -> SlowDPPConfig:
    return SlowDPPConfig()