from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from config_p1 import P1Config
from env.p1.battery import (
    activation_energy_required_j,
    relocation_energy_j,
)
from env.p1.types import P1State, RegionAction


def powerset_at_most(
    values: Sequence[int],
    capacity: int,
) -> Iterable[tuple[int, ...]]:
    ordered = tuple(sorted(int(value) for value in values))
    for size in range(min(int(capacity), len(ordered)) + 1):
        yield from combinations(ordered, size)


def region_membership(state: P1State, cfg: P1Config) -> np.ndarray:
    region = np.floor(state.user_x / cfg.region_length_m).astype(np.int32)
    return np.clip(region, 0, cfg.num_regions - 1)


def initialize_state(cfg: P1Config) -> P1State:
    rng = np.random.default_rng(cfg.seed)
    user_x = np.empty(cfg.num_users, dtype=np.float64)
    for region in range(cfg.num_regions):
        start = region * cfg.region_length_m
        indices = slice(
            region * cfg.users_per_region,
            (region + 1) * cfg.users_per_region,
        )
        user_x[indices] = rng.uniform(
            start + 0.05 * cfg.region_length_m,
            start + 0.95 * cfg.region_length_m,
            size=cfg.users_per_region,
        )
    speed = rng.uniform(
        cfg.vehicle_speed_min_mps,
        cfg.vehicle_speed_max_mps,
        size=cfg.num_users,
    )
    return P1State(
        queue=np.full(
            cfg.num_users,
            cfg.initial_playback_queue,
            dtype=np.float64,
        ),
        user_x=user_x,
        user_speed=speed,
        battery_j=np.full(
            cfg.num_uavs,
            cfg.initial_battery_j,
            dtype=np.float64,
        ),
        uav_x=np.asarray(
            [cfg.depot_x(region) for region in range(cfg.num_regions)],
            dtype=np.float64,
        ),
    )


def validate_state(state: P1State, cfg: P1Config) -> None:
    expected_user = (cfg.num_users,)
    expected_uav = (cfg.num_uavs,)
    for name in ("queue", "user_x", "user_speed"):
        value = np.asarray(getattr(state, name))
        if value.shape != expected_user:
            raise ValueError(f"{name} shape must be {expected_user}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("battery_j", "uav_x"):
        value = np.asarray(getattr(state, name))
        if value.shape != expected_uav:
            raise ValueError(f"{name} shape must be {expected_uav}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(state.queue < -1e-9):
        raise ValueError("physical playback queue cannot be negative")
    if np.any(state.battery_j < -1e-9):
        raise ValueError("physical battery cannot be negative")
    if np.any(state.battery_j > cfg.battery_capacity_j + 1e-9):
        raise ValueError("physical battery exceeds capacity")


def enumerate_region_actions(
    state: P1State,
    region: int,
    region_users: Sequence[int],
    cfg: P1Config,
) -> list[RegionAction]:
    """Enumerate the full finite P1 frame-action set for one region."""

    cfg._validate_region(region)
    users = tuple(sorted(int(user) for user in region_users))
    previous_x = float(state.uav_x[region])
    battery = float(state.battery_j[region])
    actions: list[RegionAction] = []

    depot = cfg.depot_x(region)
    if abs(depot - previous_x) <= cfg.reachable_distance_m + 1e-9:
        move_energy = relocation_energy_j(previous_x, depot, cfg)
        if battery + 1e-9 >= move_energy:
            for rsu_users in powerset_at_most(users, cfg.rsu_capacity):
                actions.append(
                    RegionAction(region, 0, -1, rsu_users, None)
                )

    for point_index, point_x in enumerate(cfg.candidate_points(region)):
        if abs(point_x - previous_x) > cfg.reachable_distance_m + 1e-9:
            continue
        move_energy = relocation_energy_j(previous_x, point_x, cfg)
        remaining_battery = battery - move_energy
        if remaining_battery + 1e-9 < activation_energy_required_j(cfg):
            continue
        for uav_user in users:
            rsu_candidates = tuple(user for user in users if user != uav_user)
            for rsu_users in powerset_at_most(
                rsu_candidates,
                cfg.rsu_capacity,
            ):
                actions.append(
                    RegionAction(
                        region=region,
                        hired=1,
                        point_index=point_index,
                        rsu_users=rsu_users,
                        uav_user=uav_user,
                    )
                )

    if not actions:
        raise RuntimeError(
            "no feasible P1 frame action: "
            f"region={region}, battery_j={battery}, previous_x={previous_x}"
        )
    return actions


def validate_region_action(
    action: RegionAction,
    region_users: Sequence[int],
    cfg: P1Config,
) -> None:
    users = set(int(user) for user in region_users)
    rsu = tuple(int(user) for user in action.rsu_users)
    if action.hired not in (0, 1):
        raise ValueError("mu must be binary")
    if len(rsu) > cfg.rsu_capacity or len(set(rsu)) != len(rsu):
        raise ValueError("RSU capacity or duplicate assignment violation")
    if not set(rsu).issubset(users):
        raise ValueError("RSU cross-region association")
    if action.hired == 0:
        if action.uav_user is not None or action.point_index != -1:
            raise ValueError("unhired UAV must be at depot with no UAV user")
        return
    if action.uav_user is None or action.uav_user not in users:
        raise ValueError("hired P1 UAV must serve exactly one region user")
    if not 0 <= action.point_index < len(cfg.candidate_offsets_m):
        raise ValueError("invalid hovering point")
    if action.uav_user in set(rsu):
        raise ValueError("single-provider constraint violation")
