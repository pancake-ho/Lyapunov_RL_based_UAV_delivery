from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config_p3 import P3Config
from env.p3.battery import activation_energy_required_j
from env.p3.radio import required_uav_power_w, rsu_link_capacity_bps
from env.p3.types import P3State, RegionAction


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def build_state_features(
    state: P3State,
    region: int,
    region_users: Sequence[int],
    cfg: P3Config,
) -> np.ndarray:
    """Fixed-size region state with padded variable-size user features."""

    cfg._validate_region(region)
    users = tuple(sorted(int(user) for user in region_users))
    queue_values = [float(state.queue[user]) for user in users]
    z_values = [cfg.large_queue_level - value for value in queue_values]
    center = cfg.rsu_x(region)
    global_features = [
        region / max(cfg.num_regions - 1, 1),
        float(state.battery_j[region]) / cfg.battery_capacity_j,
        (float(state.uav_x[region]) - center) / cfg.region_length_m,
        len(users) / cfg.num_users,
        _mean(queue_values) / cfg.large_queue_level,
        (max(z_values) if z_values else 0.0) / cfg.large_queue_level,
        float(
            state.battery_j[region]
            < activation_energy_required_j(cfg)
        ),
    ]

    user_features = np.zeros(
        (cfg.num_users, cfg.ppo_user_feature_dim), dtype=np.float32
    )
    speed_span = max(cfg.vehicle_speed_max_mps - cfg.vehicle_speed_min_mps, 1e-9)
    for user in users:
        q = float(state.queue[user])
        z = cfg.large_queue_level - q
        relative_x = (float(state.user_x[user]) - center) / cfg.region_length_m
        speed = (float(state.user_speed[user]) - cfg.vehicle_speed_min_mps) / speed_span
        rsu_capacity = rsu_link_capacity_bps(
            abs(float(state.user_x[user]) - center), 1.0, cfg
        )
        rsu_score = np.log1p(
            rsu_capacity * cfg.slot_duration_s / cfg.chunk_size_bits[0]
        ) / np.log1p(100.0)
        uav_scores = []
        nominal_user_cap = cfg.uav_max_total_power_w / cfg.uav_capacity
        for point in cfg.candidate_points(region):
            required = required_uav_power_w(
                1,
                0,
                abs(point - float(state.user_x[user])),
                1.0,
                cfg,
            )
            ratio = nominal_user_cap / max(required, 1e-12)
            uav_scores.append(np.log1p(ratio) / np.log1p(1e6))
        user_features[user] = np.asarray(
            (
                1.0,
                np.clip(q / cfg.large_queue_level, 0.0, 2.0),
                np.clip(z / cfg.large_queue_level, -1.0, 1.0),
                np.clip(relative_x, -2.0, 2.0),
                np.clip(speed, 0.0, 1.0),
                np.clip(rsu_score, 0.0, 2.0),
                *(np.clip(score, 0.0, 2.0) for score in uav_scores),
            ),
            dtype=np.float32,
        )
    result = np.concatenate(
        [np.asarray(global_features, dtype=np.float32), user_features.reshape(-1)]
    )
    if result.shape != (cfg.ppo_state_dim,):
        raise RuntimeError(f"unexpected PPO state shape: {result.shape}")
    return result


def build_candidate_signatures(
    actions: Sequence[RegionAction],
    cfg: P3Config,
) -> np.ndarray:
    """Encode feasible joint actions for sequential hard masking.

    PPO is factorized as hire, hovering point, and one provider token per
    user. Enumeration is retained only as the exact feasibility oracle.
    """

    if not actions:
        raise ValueError("at least one feasible action is required")
    signatures = np.zeros(
        (len(actions), cfg.ppo_signature_dim), dtype=np.int64
    )
    for row, action in enumerate(actions):
        signatures[row, 0] = int(action.hired)
        signatures[row, 1] = 0 if action.hired == 0 else action.point_index + 1
        signatures[row, 2 + np.asarray(action.rsu_users, dtype=np.int64)] = 1
        signatures[row, 2 + np.asarray(action.uav_users, dtype=np.int64)] = 2
    if np.unique(signatures, axis=0).shape[0] != len(actions):
        raise RuntimeError("feasible action enumeration contains duplicate signatures")
    return signatures
