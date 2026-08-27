from __future__ import annotations

from typing import Sequence

import numpy as np

from agent.P1.exact_fast_controller import ExactFastController
from config_p1 import P1Config
from env.p1.battery import (
    activation_energy_required_j,
    apply_active_slot,
    apply_relocation,
    apply_unhired_slot,
    battery_power_cap_w,
)
from env.p1.topology import validate_region_action
from env.p1.types import (
    FastOption,
    FrameTrace,
    P1State,
    RegionAction,
    RegionFrameResult,
)


def generate_frame_trace(cfg: P1Config, seed: int) -> FrameTrace:
    """Generate a reproducible fading trace for rollout or realization."""

    rng = np.random.default_rng(int(seed))
    rsu = np.clip(
        rng.exponential(
            scale=1.0,
            size=(cfg.frame_slots, cfg.num_regions, cfg.num_users),
        ),
        0.05,
        10.0,
    )
    uav = np.clip(
        rng.exponential(
            scale=1.0,
            size=(
                cfg.frame_slots,
                cfg.num_regions,
                len(cfg.candidate_offsets_m),
                cfg.num_users,
            ),
        ),
        0.05,
        10.0,
    )
    return FrameTrace(rsu_fading=rsu, uav_fading=uav)


def validate_frame_trace(trace: FrameTrace, cfg: P1Config) -> None:
    expected_rsu = (cfg.frame_slots, cfg.num_regions, cfg.num_users)
    expected_uav = (
        cfg.frame_slots,
        cfg.num_regions,
        len(cfg.candidate_offsets_m),
        cfg.num_users,
    )
    if np.asarray(trace.rsu_fading).shape != expected_rsu:
        raise ValueError(
            f"rsu_fading shape must be {expected_rsu}, "
            f"got {np.asarray(trace.rsu_fading).shape}"
        )
    if np.asarray(trace.uav_fading).shape != expected_uav:
        raise ValueError(
            f"uav_fading shape must be {expected_uav}, "
            f"got {np.asarray(trace.uav_fading).shape}"
        )


def simulate_region_frame(
    state: P1State,
    action: RegionAction,
    region_users: Sequence[int],
    trace: FrameTrace,
    cfg: P1Config,
    fast_controller: ExactFastController | None = None,
) -> RegionFrameResult:
    """Evaluate one frame action using exact slot-level minimization.

    Association and hovering point are fixed for the whole frame.  Mobility and
    fading change every slot.  The simulation mutates neither ``state`` nor
    ``trace``, which is required for fair common-random-number rollout.
    """

    validate_region_action(action, region_users, cfg)
    validate_frame_trace(trace, cfg)
    controller = fast_controller or ExactFastController(cfg)
    users = tuple(sorted(int(user) for user in region_users))
    queue = state.queue.copy()
    battery = float(state.battery_j[action.region])
    previous_x = float(state.uav_x[action.region])
    target_x = action.target_x(cfg)

    relocation = apply_relocation(
        battery,
        previous_x,
        target_x,
        cfg,
    )
    battery = relocation.battery_after_j
    relocation_events = int(relocation.consumed_j > 0.0)
    if action.hired and battery + 1e-9 < activation_energy_required_j(cfg):
        raise RuntimeError("frame activation reserve violation")

    dpp_sum = 0.0
    original_degradation = 0.0
    delivered_sum = 0.0
    utility_sum = 0.0
    stall_slots = 0
    served_user_slots = 0
    q_violation_slots = 0
    queue_sum = 0.0
    consumed_energy = relocation.consumed_j
    charged_energy = 0.0
    reserve_violations = 0
    power_violations = 0
    provider_violations = 0

    for slot in range(cfg.frame_slots):
        remaining = cfg.frame_slots - slot
        user_position = np.mod(
            state.user_x
            + state.user_speed * cfg.slot_duration_s * slot,
            cfg.road_length_m,
        )
        options: dict[int, FastOption] = {}
        provider_count = {user: 0 for user in users}

        for user in action.rsu_users:
            horizontal = abs(
                cfg.rsu_x(action.region) - float(user_position[user])
            )
            options[user] = controller.solve_rsu(
                z=cfg.large_queue_level - float(queue[user]),
                horizontal_distance_m=horizontal,
                fading=float(trace.rsu_fading[slot, action.region, user]),
            )
            provider_count[user] += 1

        uav_power = 0.0
        if action.hired and action.uav_user is not None:
            user = int(action.uav_user)
            horizontal = abs(target_x - float(user_position[user]))
            option = controller.solve_uav(
                z=cfg.large_queue_level - float(queue[user]),
                horizontal_distance_m=horizontal,
                fading=float(
                    trace.uav_fading[
                        slot,
                        action.region,
                        action.point_index,
                        user,
                    ]
                ),
                battery_j=battery,
                remaining_slots_including_current=remaining,
            )
            options[user] = option
            provider_count[user] += 1
            uav_power = option.power_w
            effective_cap = battery_power_cap_w(battery, remaining, cfg)
            power_violations += int(uav_power > effective_cap + 1e-9)

        provider_violations += sum(
            int(count > 1) for count in provider_count.values()
        )

        for user in users:
            q_before = float(queue[user])
            z_before = cfg.large_queue_level - q_before
            actual_departure = min(
                q_before,
                cfg.playback_chunks_per_slot,
            )
            option = options.get(user, FastOption())
            delivered = float(option.chunks)
            degradation = float(option.degradation)

            dpp_sum += (
                cfg.alpha_z
                * z_before
                * (actual_departure - delivered)
                + cfg.lyapunov_v * degradation
            )
            original_degradation += degradation
            delivered_sum += delivered
            utility_sum += option.utility
            stall_slots += int(q_before < cfg.playback_chunks_per_slot)
            served_user_slots += int(delivered > 0.0)
            q_violation_slots += int(q_before > cfg.large_queue_level)
            queue_sum += q_before

            # Exact physical update.  large_queue_level is not a clipping bound.
            queue[user] = max(
                q_before - cfg.playback_chunks_per_slot,
                0.0,
            ) + delivered

        if action.hired:
            battery_step = apply_active_slot(
                battery,
                uav_power,
                remaining,
                cfg,
            )
            battery = battery_step.battery_after_j
            consumed_energy += battery_step.consumed_j
            required_after = (
                cfg.reserve_battery_j
                + (remaining - 1) * cfg.hover_energy_per_slot_j
            )
            reserve_violations += int(battery + 1e-7 < required_after)
        else:
            battery_step = apply_unhired_slot(battery, cfg)
            battery = battery_step.battery_after_j
            charged_energy += battery_step.charged_j

    hiring_penalty_dpp = (
        cfg.lyapunov_v
        * cfg.lambda_h
        * cfg.hiring_cost_per_frame
        * action.hired
    )
    original_hiring_cost = (
        cfg.lambda_h * cfg.hiring_cost_per_frame * action.hired
    )
    local_indices = np.asarray(users, dtype=np.int64)
    return RegionFrameResult(
        frame_dpp_cost=float(dpp_sum + hiring_penalty_dpp),
        original_cost=float(original_degradation + original_hiring_cost),
        queue_after=queue[local_indices].copy(),
        battery_after_j=float(battery),
        uav_x_after=float(target_x),
        delivered_chunks=float(delivered_sum),
        quality_utility=float(utility_sum),
        degradation=float(original_degradation),
        stall_user_slots=int(stall_slots),
        served_user_slots=int(served_user_slots),
        large_queue_violation_user_slots=int(q_violation_slots),
        queue_sum=float(queue_sum),
        energy_consumed_j=float(consumed_energy),
        energy_charged_j=float(charged_energy),
        relocation_events=int(relocation_events),
        battery_reserve_violations=int(reserve_violations),
        power_violations=int(power_violations),
        provider_violations=int(provider_violations),
    )


def apply_region_result(
    state: P1State,
    region: int,
    region_users: Sequence[int],
    result: RegionFrameResult,
) -> None:
    """Commit one realized regional result to the global physical state."""

    users = np.asarray(tuple(int(user) for user in region_users), dtype=np.int64)
    if result.queue_after.shape != users.shape:
        raise ValueError(
            "queue result shape does not match region users: "
            f"{result.queue_after.shape} != {users.shape}"
        )
    if users.size:
        state.queue[users] = result.queue_after
    state.battery_j[region] = result.battery_after_j
    state.uav_x[region] = result.uav_x_after


def advance_mobility_one_frame(state: P1State, cfg: P1Config) -> None:
    state.user_x = np.mod(
        state.user_x
        + state.user_speed * cfg.slot_duration_s * cfg.frame_slots,
        cfg.road_length_m,
    )
