from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agent.P3.exact_fast_controller import ExactFastController
from config_p3 import P3Config
from env.p3.battery import (
    activation_energy_required_j,
    apply_active_slot,
    apply_relocation,
    apply_unhired_slot,
    battery_power_cap_w,
    diagnose_return_to_charge,
)
from env.p3.topology import validate_region_action
from env.p3.types import FastOption, FrameTrace, P3State, RegionAction, RegionFrameResult


def generate_frame_trace(cfg: P3Config, seed: int) -> FrameTrace:
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


def validate_frame_trace(trace: FrameTrace, cfg: P3Config) -> None:
    expected_rsu = (cfg.frame_slots, cfg.num_regions, cfg.num_users)
    expected_uav = (
        cfg.frame_slots,
        cfg.num_regions,
        len(cfg.candidate_offsets_m),
        cfg.num_users,
    )
    if np.asarray(trace.rsu_fading).shape != expected_rsu:
        raise ValueError(f"rsu_fading must have shape {expected_rsu}")
    if np.asarray(trace.uav_fading).shape != expected_uav:
        raise ValueError(f"uav_fading must have shape {expected_uav}")


def distance_bin_index(distance_m: float, cfg: P3Config) -> int:
    edges = np.asarray(cfg.distance_bin_edges_m, dtype=np.float64)
    index = int(np.searchsorted(edges, float(distance_m), side="right") - 1)
    return int(np.clip(index, 0, cfg.num_distance_bins - 1))


def update_playback_queue(
    queue_before: float,
    delivered_chunks: float,
    cfg: P3Config,
) -> tuple[float, float, float, float]:
    """Equations (3.9), (3.10), and (3.13), with an exact identity check.

    Returns ``(queue_after, z_before, z_after, actual_departure)``.
    ``Z`` is deliberately derived from the physical queue instead of stored as
    an independently drifting state variable.
    """

    q_before = float(queue_before)
    delivered = float(delivered_chunks)
    if q_before < -1e-9 or delivered < -1e-9:
        raise ValueError("queue and delivered chunks must be non-negative")
    actual_departure = min(q_before, cfg.playback_chunks_per_slot)
    queue_after = q_before - actual_departure + delivered
    z_before = cfg.large_queue_level - q_before
    z_after = cfg.large_queue_level - queue_after
    recurrence = z_before + actual_departure - delivered
    if not np.isclose(z_after, recurrence, rtol=0.0, atol=1e-9):
        raise RuntimeError("virtual/physical queue identity violation")
    return (
        float(queue_after),
        float(z_before),
        float(z_after),
        float(actual_departure),
    )


def simulate_region_frame(
    state: P3State,
    action: RegionAction,
    region_users: Sequence[int],
    trace: FrameTrace,
    cfg: P3Config,
    fast_controller: ExactFastController | None = None,
) -> RegionFrameResult:
    """Run one P3 frame without mutating state or the common trace."""

    validate_region_action(action, region_users, cfg)
    validate_frame_trace(trace, cfg)
    controller = fast_controller or ExactFastController(cfg)
    users = tuple(sorted(int(user) for user in region_users))
    queue = state.queue.copy()
    last_quality = state.last_quality_index.copy()
    last_stalled = state.last_stalled.copy()
    battery = float(state.battery_j[action.region])
    previous_x = float(state.uav_x[action.region])
    target_x = action.target_x(cfg)

    return_diag = diagnose_return_to_charge(
        battery_j=battery,
        previous_x=previous_x,
        depot_x=cfg.depot_x(action.region),
        will_charge=action.hired == 0,
        cfg=cfg,
    )
    charging_needed = battery < activation_energy_required_j(cfg) - 1e-9
    stranded_before_charge = return_diag.depletion_before_arrival
    relocation = apply_relocation(battery, previous_x, target_x, cfg)
    battery = relocation.battery_after_j
    if action.hired and battery + 1e-9 < activation_energy_required_j(cfg):
        raise RuntimeError("frame activation reserve violation")

    dpp_sum = 0.0
    degradation_sum = 0.0
    delivered_sum = 0.0
    payload_bits_sum = 0.0
    utility_sum = 0.0
    quality_level_sum = 0.0
    quality_hist = np.zeros(cfg.num_quality_levels, dtype=np.float64)
    quality_switches = 0
    quality_transitions = 0
    stall_slots = 0
    served_slots = 0
    q_violation_slots = 0
    queue_sum = 0.0
    queue_samples: list[float] = []
    z_samples: list[float] = []
    user_index = {user: index for index, user in enumerate(users)}
    user_delivered = np.zeros(len(users), dtype=np.float64)
    user_utility = np.zeros(len(users), dtype=np.float64)
    user_stall_slots = np.zeros(len(users), dtype=np.int64)
    user_served_slots = np.zeros(len(users), dtype=np.int64)
    user_stall_events = np.zeros(len(users), dtype=np.int64)
    user_queue_sum = np.zeros(len(users), dtype=np.float64)
    consumed_energy = relocation.consumed_j
    charged_energy = 0.0
    reserve_violations = 0
    power_violations = 0
    provider_violations = 0
    peak_uav_power = 0.0
    uav_distance_sum = 0.0
    uav_scheduled_slots = 0
    rsu_scheduled_slots = 0

    distance_opportunities = np.zeros(cfg.num_distance_bins, dtype=np.float64)
    distance_served = np.zeros_like(distance_opportunities)
    distance_stall = np.zeros_like(distance_opportunities)
    distance_delivered = np.zeros_like(distance_opportunities)
    distance_quality = np.zeros_like(distance_opportunities)
    distance_power = np.zeros_like(distance_opportunities)

    for slot in range(cfg.frame_slots):
        remaining = cfg.frame_slots - slot
        user_position = np.mod(
            state.user_x + state.user_speed * cfg.slot_duration_s * slot,
            cfg.road_length_m,
        )
        options: dict[int, FastOption] = {}
        provider_count = {user: 0 for user in users}

        for user in action.rsu_users:
            horizontal = abs(cfg.rsu_x(action.region) - float(user_position[user]))
            options[user] = controller.solve_rsu(
                z=cfg.large_queue_level - float(queue[user]),
                horizontal_distance_m=horizontal,
                fading=float(trace.rsu_fading[slot, action.region, user]),
            )
            provider_count[user] += 1
        rsu_scheduled_slots += len(action.rsu_users)

        uav_distances: dict[int, float] = {}
        if action.hired:
            z_by_user = {
                user: cfg.large_queue_level - float(queue[user])
                for user in action.uav_users
            }
            uav_distances = {
                user: abs(target_x - float(user_position[user]))
                for user in action.uav_users
            }
            uav_distance_sum += sum(uav_distances.values())
            uav_scheduled_slots += len(uav_distances)
            fading_by_user = {
                user: float(
                    trace.uav_fading[
                        slot,
                        action.region,
                        action.point_index,
                        user,
                    ]
                )
                for user in action.uav_users
            }
            uav_options = controller.solve_uav(
                users=action.uav_users,
                z_by_user=z_by_user,
                horizontal_distance_by_user_m=uav_distances,
                fading_by_user=fading_by_user,
                battery_j=battery,
                remaining_slots_including_current=remaining,
            )
            options.update(uav_options)
            for user in action.uav_users:
                provider_count[user] += 1

        total_uav_power = sum(
            options.get(user, FastOption()).power_w for user in action.uav_users
        )
        peak_uav_power = max(peak_uav_power, total_uav_power)
        effective_cap = battery_power_cap_w(
            battery,
            remaining,
            cfg,
        )
        power_violations += int(total_uav_power > effective_cap + 1e-9)
        provider_violations += sum(int(count > 1) for count in provider_count.values())

        for user in users:
            q_before = float(queue[user])
            z_before = cfg.large_queue_level - q_before
            option = options.get(user, FastOption())
            delivered = float(option.chunks)
            degradation = float(option.degradation)
            stalled = int(q_before < cfg.playback_chunks_per_slot)
            q_after, z_before, _, actual_departure = update_playback_queue(
                q_before,
                delivered,
                cfg,
            )

            dpp_sum += (
                cfg.alpha_z * z_before * (actual_departure - delivered)
                + cfg.lyapunov_v * degradation
            )
            degradation_sum += degradation
            delivered_sum += delivered
            payload_bits_sum += option.payload_bits
            utility_sum += option.utility
            stall_slots += stalled
            served_slots += int(delivered > 0.0)
            q_violation_slots += int(q_before > cfg.large_queue_level)
            queue_sum += q_before
            queue_samples.append(q_before)
            z_samples.append(z_before)
            local_index = user_index[user]
            user_delivered[local_index] += delivered
            user_utility[local_index] += option.utility
            user_stall_slots[local_index] += stalled
            user_served_slots[local_index] += int(delivered > 0.0)
            user_stall_events[local_index] += int(stalled and not last_stalled[user])
            user_queue_sum[local_index] += q_before
            last_stalled[user] = bool(stalled)

            if option.chunks > 0 and option.quality_index >= 0:
                quality_hist[option.quality_index] += option.chunks
                quality_level_sum += option.chunks * (option.quality_index + 1)
                if last_quality[user] >= 0:
                    quality_transitions += 1
                    quality_switches += int(last_quality[user] != option.quality_index)
                last_quality[user] = option.quality_index

            if user in uav_distances:
                bin_index = distance_bin_index(uav_distances[user], cfg)
                distance_opportunities[bin_index] += 1.0
                distance_stall[bin_index] += stalled
                if delivered > 0.0:
                    distance_served[bin_index] += 1.0
                    distance_delivered[bin_index] += delivered
                    distance_quality[bin_index] += option.utility
                    distance_power[bin_index] += option.power_w

            queue[user] = q_after

        if action.hired:
            battery_step = apply_active_slot(
                battery,
                total_uav_power,
                remaining,
                cfg,
            )
            battery = battery_step.battery_after_j
            consumed_energy += battery_step.consumed_j
            required_after = cfg.reserve_battery_j + (
                remaining - 1
            ) * cfg.hover_energy_per_slot_j
            reserve_violations += int(battery + 1e-7 < required_after)
        else:
            battery_step = apply_unhired_slot(battery, cfg)
            battery = battery_step.battery_after_j
            charged_energy += battery_step.charged_j

    hiring_cost = cfg.lambda_h * cfg.hiring_cost_per_frame * action.hired
    local = np.asarray(users, dtype=np.int64)
    return RegionFrameResult(
        frame_dpp_cost=float(dpp_sum + cfg.lyapunov_v * hiring_cost),
        original_cost=float(degradation_sum + hiring_cost),
        queue_after=queue[local].copy(),
        last_quality_after=last_quality[local].copy(),
        last_stalled_after=last_stalled[local].copy(),
        battery_after_j=float(battery),
        uav_x_after=float(target_x),
        delivered_chunks=float(delivered_sum),
        payload_bits=float(payload_bits_sum),
        quality_utility=float(utility_sum),
        quality_level_sum=float(quality_level_sum),
        quality_histogram=quality_hist,
        degradation=float(degradation_sum),
        quality_switches=int(quality_switches),
        quality_transitions=int(quality_transitions),
        stall_user_slots=int(stall_slots),
        served_user_slots=int(served_slots),
        large_queue_violation_user_slots=int(q_violation_slots),
        queue_sum=float(queue_sum),
        queue_samples=np.asarray(queue_samples, dtype=np.float64),
        z_samples=np.asarray(z_samples, dtype=np.float64),
        user_delivered_chunks=user_delivered,
        user_quality_utility=user_utility,
        user_stall_slots=user_stall_slots,
        user_served_slots=user_served_slots,
        user_stall_events=user_stall_events,
        user_queue_sum=user_queue_sum,
        energy_consumed_j=float(consumed_energy),
        energy_charged_j=float(charged_energy),
        relocation_events=int(relocation.consumed_j > 0.0),
        charging_slots=int((action.hired == 0) * cfg.frame_slots),
        charging_need_events=int(charging_needed),
        stranded_before_charge_events=int(stranded_before_charge),
        return_to_charge_events=int(return_diag.is_return_to_charge),
        precharge_depletion_events=int(return_diag.depletion_before_arrival),
        precharge_reserve_breach_events=int(return_diag.reserve_breach_before_charge),
        battery_reserve_violations=int(reserve_violations),
        power_violations=int(power_violations),
        provider_violations=int(provider_violations),
        uav_total_power_peak_w=float(peak_uav_power),
        uav_distance_sum_m=float(uav_distance_sum),
        uav_scheduled_user_slots=int(uav_scheduled_slots),
        rsu_scheduled_user_slots=int(rsu_scheduled_slots),
        distance_opportunities=distance_opportunities,
        distance_served_slots=distance_served,
        distance_stall_slots=distance_stall,
        distance_delivered_chunks=distance_delivered,
        distance_quality_utility=distance_quality,
        distance_power_sum_w=distance_power,
    )


def apply_region_result(
    state: P3State,
    region: int,
    region_users: Sequence[int],
    result: RegionFrameResult,
) -> None:
    users = np.asarray(tuple(int(user) for user in region_users), dtype=np.int64)
    if result.queue_after.shape != users.shape:
        raise ValueError("queue result shape does not match region users")
    if result.last_quality_after.shape != users.shape:
        raise ValueError("quality result shape does not match region users")
    if result.last_stalled_after.shape != users.shape:
        raise ValueError("stall-state result shape does not match region users")
    if users.size:
        state.queue[users] = result.queue_after
        state.last_quality_index[users] = result.last_quality_after
        state.last_stalled[users] = result.last_stalled_after
    state.battery_j[region] = result.battery_after_j
    state.uav_x[region] = result.uav_x_after


def advance_mobility_one_frame(state: P3State, cfg: P3Config) -> None:
    state.user_x = np.mod(
        state.user_x
        + state.user_speed * cfg.slot_duration_s * cfg.frame_slots,
        cfg.road_length_m,
    )
