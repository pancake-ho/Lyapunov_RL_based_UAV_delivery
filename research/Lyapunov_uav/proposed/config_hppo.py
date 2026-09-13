from __future__ import annotations

"""Hierarchical-PPO configuration on top of the P3 system model.

``HPPOConfig`` *inherits every physical parameter and helper of*
:class:`config_p3.P3Config` (road/region geometry, fixed reserved resource
blocks, Shannon-gap radio model, quality ladder, playback queue, large-Q^e
virtual queue, persistent UAV battery, hiring cost, DPP weights).  Only the
learning-algorithm fields that ``uav_hierarchical_ppo`` introduced are added
here, so the P3 model and the HRL solver are configured in one object and a
checkpoint stores the full configuration.

Nothing in this file redefines a P3 physical quantity.
"""

import argparse
import math
from dataclasses import dataclass, fields, replace

from config_p3 import P3Config


REWARD_MODES = ("dpp", "objective_lagrangian", "objective_only")


@dataclass(frozen=True)
class HPPOConfig(P3Config):
    # ---------------- hierarchical-PPO action encoding ----------------
    # Slot PPO chooses a discrete UAV power level per UAV user:
    #   p_n = level / (uav_power_levels - 1) * P^U_max  (level = 0 ... levels-1)
    uav_power_levels: int = 7
    # Scheduling contains one provider candidate per global user.
    # Hiring/location are selected by independent fast-policy rollouts.
    completion_seed_offset: int = 40_000_061

    # ---------------- reward ----------------
    # "dpp"                 : reward = -(DPP cost) * ppo_reward_scale.
    #                         Slot reward = -J_F(t) (Eq. 6.12), frame reward
    #                         = -(V*lambda_H*C^H + sum_t J_F(t)) (Eq. 8.4).
    #                         This is exactly the frame reward used by the
    #                         P3 slow PPO in run/p3_train_ppo.py.
    # "objective_lagrangian": reward = -(original objective) - dual * Z-cost
    #                         (uav_hierarchical_ppo default).
    # "objective_only"      : reward = -(original objective) only (ablation).
    reward_mode: str = "dpp"
    constraint_reward_scale: float = 1.0
    dual_init: float = 1.0
    dual_lr: float = 0.05
    dual_max: float = 100.0
    z_target_normalized: float = 0.35
    stall_training_penalty: float = 0.0
    # Reward scale for the objective_* modes (ppo_reward_scale is used for "dpp").
    objective_reward_scale: float = 1.0
    # Operating condition 0 <= Q <= Q^e of the large-Q^e derivation, enforced by
    # the environment exactly as ExactFastController does (delivery is capped so
    # that Q(t+1) <= Q^e). Set False to let Q exceed Q^e and only record it.
    enforce_queue_admissibility: bool = True

    # ---------------- PPO (shared by both levels) ----------------
    hidden_dims: tuple[int, ...] = (256, 256)
    ppo_minibatch_size: int = 256
    ppo_adv_eps: float = 1e-8
    slot_update_every_frames: int = 8
    frame_update_every_episodes: int = 2

    # ---------------- episodes / run ----------------
    train_episodes: int = 100
    eval_episodes: int = 5
    save_every_episodes: int = 25
    deterministic_eval: bool = True
    device: str = "cpu"
    # Set explicitly to use disjoint evaluation episodes; default preserves Claude.
    episode_offset: int = 0
    torch_num_threads: int = 1

    # ---------------- logging ----------------
    write_jsonl_trace: bool = True
    write_human_debug_log: bool = True
    log_hidden_csi: bool = True
    log_observation_vectors: bool = False
    console_log_every_slots: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("train_episodes", "eval_episodes", "save_every_episodes", "torch_num_threads"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.episode_offset < 0 or self.completion_seed_offset < 0:
            raise ValueError("episode/seed offsets must be nonnegative")
        if not (0 < self.ppo_gamma <= 1 and 0 <= self.ppo_gae_lambda <= 1):
            raise ValueError("invalid gamma or GAE lambda")
        if self.ppo_learning_rate <= 0 or self.ppo_adv_eps <= 0:
            raise ValueError("learning rate and advantage epsilon must be positive")
        if self.uav_power_levels < 2:
            raise ValueError("uav_power_levels must be >= 2")
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if self.ppo_minibatch_size <= 0:
            raise ValueError("ppo_minibatch_size must be positive")
        if self.slot_update_every_frames <= 0 or self.frame_update_every_episodes <= 0:
            raise ValueError("PPO update intervals must be positive")
        if self.dual_lr < 0.0 or self.dual_max < 0.0 or self.dual_init < 0.0:
            raise ValueError("dual parameters must be non-negative")
        if not self.hidden_dims or any(int(h) <= 0 for h in self.hidden_dims):
            raise ValueError("hidden_dims must be positive integers")

    # ------------------------------------------------------------------
    # Observation / action geometry shared by env and agents
    # ------------------------------------------------------------------
    @property
    def num_candidate_points(self) -> int:
        return len(self.candidate_offsets_m)

    @property
    def frame_action_nvec(self) -> tuple[int, ...]:
        """0: unserved, 1: RSU, 2: UAV candidate conditional on hiring."""
        return (3,) * self.num_users

    @property
    def slot_action_nvec(self) -> tuple[int, ...]:
        """Per-region slot action: [l_n(Lmax+1)*N, k_n(K)*N, plevel_n(levels)*N]."""
        return (
            (self.max_chunks_per_slot + 1,) * self.num_users
            + (self.num_quality_levels,) * self.num_users
            + (self.uav_power_levels,) * self.num_users
        )

    @property
    def frame_obs_user_dim(self) -> int:
        # [present, Q/Qe, Z/Qe, rel_x, speed, nominal RSU link score,
        #  nominal UAV link score at every candidate point]
        # identical to agent/P3/features.py (no instantaneous CSI).
        return 6 + self.num_candidate_points

    @property
    def frame_obs_global_dim(self) -> int:
        # [region id, battery SoC, UAV rel-x, |N_m|/N, mean Q/Qe, max Z/Qe,
        #  battery below activation flag, frame progress]
        return 8

    @property
    def frame_obs_dim(self) -> int:
        return self.frame_obs_global_dim + self.frame_obs_user_dim * self.num_users

    @property
    def slot_obs_user_dim(self) -> int:
        # [present, provider one-hot(3: none/RSU/UAV), Q/Qe, Z/Qe, rel_x, speed,
        #  RSU horizontal distance, UAV horizontal distance,
        #  last quality index / K]
        return 1 + 3 + 7

    @property
    def slot_obs_global_dim(self) -> int:
        # [region id, hired, battery SoC, P_eff/P_max, UAV rel-x,
        #  slot progress, remaining slots / T, |N_m|/N]
        return 8

    @property
    def slot_obs_dim(self) -> int:
        return self.slot_obs_global_dim + self.slot_obs_user_dim * self.num_users

    def power_level_to_w(self, level: int) -> float:
        return float(level) / float(self.uav_power_levels - 1) * self.uav_max_total_power_w


# ----------------------------------------------------------------------
# argparse layer (uav_hierarchical_ppo style) that overrides dataclass fields
# ----------------------------------------------------------------------
def _parse_tuple_float(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_tuple_int(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _parse_bool(text: str) -> bool:
    value = str(text).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical PPO (frame PPO + slot PPO) on the P3 system model. "
            "Every HPPOConfig/P3Config field can be overridden as --<field-name>."
        )
    )
    parser.add_argument("--mode", choices=("train", "eval", "random"), default="train")
    parser.add_argument("--output-dir", type=str, default="outputs/hppo")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--frame-checkpoint", type=str, default="")
    parser.add_argument("--slot-checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    defaults = HPPOConfig()
    for field in fields(HPPOConfig):
        name = f"--{field.name.replace('_', '-')}"
        value = getattr(defaults, field.name)
        if isinstance(value, bool):
            parser.add_argument(name, type=_parse_bool, default=None)
        elif isinstance(value, int):
            parser.add_argument(name, type=int, default=None)
        elif isinstance(value, float):
            parser.add_argument(name, type=float, default=None)
        elif isinstance(value, tuple):
            if value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                parser.add_argument(name, type=_parse_tuple_int, default=None)
            else:
                parser.add_argument(name, type=_parse_tuple_float, default=None)
        else:
            parser.add_argument(name, type=str, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> HPPOConfig:
    overrides = {}
    for field in fields(HPPOConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            if field.name == "distance_bin_edges_m":
                if not value:
                    raise ValueError("distance_bin_edges_m cannot be empty")
                value = tuple(value[:-1]) + (math.inf,) if not math.isinf(value[-1]) else tuple(value)
            overrides[field.name] = value
    return replace(HPPOConfig(), **overrides)


def parse_config(argv=None) -> tuple[argparse.Namespace, HPPOConfig]:
    args = build_parser().parse_args(argv)
    return args, config_from_args(args)
