from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from config_p3 import P3Config
from env.p3.battery import activation_energy_required_j, relocation_energy_j
from env.p3.types import P3State, RegionAction


def powerset_at_most(values: Sequence[int], capacity: int) -> Iterable[tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in values))
    for size in range(min(int(capacity), len(ordered)) + 1):
        yield from combinations(ordered, size)


def region_membership(state: P3State, cfg: P3Config) -> np.ndarray:
    region = np.floor(state.user_x / cfg.region_length_m).astype(np.int32)
    return np.clip(region, 0, cfg.num_regions - 1)


def initialize_state(cfg: P3Config) -> P3State:
    rng = np.random.default_rng(cfg.seed)
    user_x = np.empty(cfg.num_users, dtype=np.float64)
    for region in range(cfg.num_regions):
        start = region * cfg.region_length_m
        indices = slice(region * cfg.users_per_region, (region + 1) * cfg.users_per_region)
        user_x[indices] = rng.uniform(
            start + 0.05 * cfg.region_length_m,
            start + 0.95 * cfg.region_length_m,
            size=cfg.users_per_region,
        )
    return P3State(
        queue=np.full(cfg.num_users, cfg.initial_playback_queue, dtype=np.float64),
        user_x=user_x,
        user_speed=rng.uniform(
            cfg.vehicle_speed_min_mps,
            cfg.vehicle_speed_max_mps,
            size=cfg.num_users,
        ),
        battery_j=np.full(cfg.num_uavs, cfg.initial_battery_j, dtype=np.float64),
        uav_x=np.asarray(
            [cfg.depot_x(region) for region in range(cfg.num_regions)],
            dtype=np.float64,
        ),
        last_quality_index=np.full(cfg.num_users, -1, dtype=np.int32),
    )


def validate_state(state: P3State, cfg: P3Config) -> None:
    user_shape = (cfg.num_users,)
    uav_shape = (cfg.num_uavs,)
    for name in ("queue", "user_x", "user_speed", "last_quality_index"):
        value = np.asarray(getattr(state, name))
        if value.shape != user_shape:
            raise ValueError(f"{name} shape must be {user_shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("battery_j", "uav_x"):
        value = np.asarray(getattr(state, name))
        if value.shape != uav_shape:
            raise ValueError(f"{name} shape must be {uav_shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(state.queue < -1e-9):
        raise ValueError("physical playback queue cannot be negative")
    if np.any(state.battery_j < -1e-9):
        raise ValueError("physical battery cannot be negative")
    if np.any(state.battery_j > cfg.battery_capacity_j + 1e-9):
        raise ValueError("battery exceeds capacity")
    if np.any(state.last_quality_index < -1) or np.any(
        state.last_quality_index >= cfg.num_quality_levels
    ):
        raise ValueError("last_quality_index is outside {-1, ..., K-1}")


def shortlist_region_users(
    state: P3State,
    region: int,
    region_users: Sequence[int],
    cfg: P3Config,
) -> tuple[int, ...]:
    """Section 8.4 urgency-channel shortlist for scalable action generation."""

    users = tuple(sorted(int(user) for user in region_users))
    if len(users) <= cfg.candidate_user_limit:
        return users
    center = cfg.rsu_x(region)
    scored = []
    for user in users:
        z = cfg.large_queue_level - float(state.queue[user])
        distance = abs(float(state.user_x[user]) - center) / cfg.region_length_m
        score = z / cfg.large_queue_level + distance
        scored.append((score, -user, user))
    scored.sort(reverse=True)
    return tuple(sorted(item[2] for item in scored[: cfg.candidate_user_limit]))


def enumerate_region_actions(
    state: P3State,
    region: int,
    region_users: Sequence[int],
    cfg: P3Config,
    candidate_users: Sequence[int] | None = None,
) -> list[RegionAction]:
    """Enumerate P3 hiring/location/RSU-subset/UAV-subset actions."""

    cfg._validate_region(region)
    all_users = tuple(sorted(int(user) for user in region_users))
    candidates = tuple(
        sorted(int(user) for user in (candidate_users if candidate_users is not None else all_users))
    )
    if not set(candidates).issubset(set(all_users)):
        raise ValueError("candidate users must belong to the region")
    previous_x = float(state.uav_x[region])
    battery = float(state.battery_j[region])
    actions: list[RegionAction] = []

    depot = cfg.depot_x(region)
    if abs(depot - previous_x) <= cfg.reachable_distance_m + 1e-9:
        move_energy = relocation_energy_j(previous_x, depot, cfg)
        if battery + 1e-9 >= move_energy:
            for rsu_users in powerset_at_most(candidates, cfg.rsu_capacity):
                actions.append(RegionAction(region, 0, -1, rsu_users, tuple()))

    for point_index, point_x in enumerate(cfg.candidate_points(region)):
        if abs(point_x - previous_x) > cfg.reachable_distance_m + 1e-9:
            continue
        move_energy = relocation_energy_j(previous_x, point_x, cfg)
        remaining_battery = battery - move_energy
        if remaining_battery + 1e-9 < activation_energy_required_j(cfg):
            continue
        # The formulation permits a hired UAV to serve from zero to J^U users.
        for uav_users in powerset_at_most(candidates, cfg.uav_capacity):
            residual = tuple(user for user in candidates if user not in set(uav_users))
            for rsu_users in powerset_at_most(residual, cfg.rsu_capacity):
                actions.append(
                    RegionAction(region, 1, point_index, rsu_users, uav_users)
                )

    if not actions:
        raise RuntimeError(
            "no feasible P3 frame action: "
            f"region={region}, battery_j={battery}, previous_x={previous_x}"
        )
    return actions


def validate_region_action(
    action: RegionAction,
    region_users: Sequence[int],
    cfg: P3Config,
) -> None:
    users = set(int(user) for user in region_users)
    rsu = tuple(int(user) for user in action.rsu_users)
    uav = tuple(int(user) for user in action.uav_users)
    if action.hired not in (0, 1):
        raise ValueError("mu must be binary")
    if len(rsu) > cfg.rsu_capacity or len(set(rsu)) != len(rsu):
        raise ValueError("RSU capacity or duplicate assignment violation")
    if len(uav) > cfg.uav_capacity or len(set(uav)) != len(uav):
        raise ValueError("UAV capacity or duplicate assignment violation")
    if not set(rsu).issubset(users) or not set(uav).issubset(users):
        raise ValueError("cross-region association")
    if set(rsu).intersection(uav):
        raise ValueError("single-provider constraint violation")
    if action.hired == 0:
        if uav or action.point_index != -1:
            raise ValueError("unhired UAV must be at depot with no UAV users")
        return
    if not 0 <= action.point_index < len(cfg.candidate_offsets_m):
        raise ValueError("invalid hovering point")
