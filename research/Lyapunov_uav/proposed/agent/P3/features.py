from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config_p3 import P3Config
from env.p3.battery import activation_energy_required_j
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

    user_features = np.zeros((cfg.num_users, 6), dtype=np.float32)
    speed_span = max(cfg.vehicle_speed_max_mps - cfg.vehicle_speed_min_mps, 1e-9)
    for row, user in enumerate(users[: cfg.num_users]):
        q = float(state.queue[user])
        z = cfg.large_queue_level - q
        relative_x = (float(state.user_x[user]) - center) / cfg.region_length_m
        speed = (float(state.user_speed[user]) - cfg.vehicle_speed_min_mps) / speed_span
        rsu_distance = abs(float(state.user_x[user]) - center) / cfg.region_length_m
        user_features[row] = (
            1.0,
            np.clip(q / cfg.large_queue_level, 0.0, 2.0),
            np.clip(z / cfg.large_queue_level, -1.0, 1.0),
            np.clip(relative_x, -2.0, 2.0),
            np.clip(speed, 0.0, 1.0),
            np.clip(rsu_distance, 0.0, 2.0),
        )
    result = np.concatenate(
        [np.asarray(global_features, dtype=np.float32), user_features.reshape(-1)]
    )
    if result.shape != (cfg.ppo_state_dim,):
        raise RuntimeError(f"unexpected PPO state shape: {result.shape}")
    return result


def build_action_features(
    state: P3State,
    action: RegionAction,
    cfg: P3Config,
) -> np.ndarray:
    """Candidate-conditioned features used by the masked categorical actor."""

    center = cfg.rsu_x(action.region)
    target = action.target_x(cfg)
    rsu_z = [
        cfg.large_queue_level - float(state.queue[user]) for user in action.rsu_users
    ]
    uav_z = [
        cfg.large_queue_level - float(state.queue[user]) for user in action.uav_users
    ]
    rsu_distance = [
        abs(center - float(state.user_x[user])) / cfg.region_length_m
        for user in action.rsu_users
    ]
    uav_distance = [
        abs(target - float(state.user_x[user])) / cfg.region_length_m
        for user in action.uav_users
    ]
    return np.asarray(
        [
            float(action.hired),
            (target - center) / cfg.region_length_m,
            abs(target - float(state.uav_x[action.region]))
            / max(cfg.reachable_distance_m, 1e-9),
            len(action.rsu_users) / cfg.rsu_capacity,
            len(action.uav_users) / cfg.uav_capacity,
            _mean(rsu_z) / cfg.large_queue_level,
            _mean(uav_z) / cfg.large_queue_level,
            _mean(rsu_distance),
            _mean(uav_distance),
            float(action.hired == 0 and abs(float(state.uav_x[action.region]) - center) > 1e-9),
        ],
        dtype=np.float32,
    )


def build_candidate_feature_matrix(
    state: P3State,
    actions: Sequence[RegionAction],
    cfg: P3Config,
) -> np.ndarray:
    if not actions:
        raise ValueError("at least one feasible action is required")
    matrix = np.stack([build_action_features(state, action, cfg) for action in actions])
    if matrix.shape[1] != cfg.ppo_action_dim:
        raise RuntimeError(f"unexpected PPO action feature shape: {matrix.shape}")
    return matrix.astype(np.float32, copy=False)
