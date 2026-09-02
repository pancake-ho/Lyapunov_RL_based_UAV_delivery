from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from agent.P3.features import build_candidate_feature_matrix, build_state_features
from agent.P3.slow_rollout_controller import SlowRolloutController
from config_p3 import P3Config
from env.p3.environment import (
    advance_mobility_one_frame,
    apply_region_result,
    generate_frame_trace,
    simulate_region_frame,
)
from env.p3.topology import initialize_state, region_membership, validate_state
from env.p3.types import RegionAction

if TYPE_CHECKING:
    from agent.P3.ppo_agent import PPOAgent


@dataclass
class PolicyRunResult:
    summary: dict
    frame_rows: list[dict]
    distance_rows: list[dict]
    point_rows: list[dict]
    quality_rows: list[dict]


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def action_record(action: RegionAction, cfg: P3Config) -> dict:
    return {
        "region": action.region,
        "mu": action.hired,
        "point_index": action.point_index,
        "target_x": action.target_x(cfg),
        "rsu_users": list(action.rsu_users),
        "uav_users": list(action.uav_users),
    }


def quality_percentile(histogram: np.ndarray, cfg: P3Config, quantile: float) -> float:
    total = float(np.sum(histogram))
    if total <= 0.0:
        return 0.0
    threshold = quantile * total
    cumulative = 0.0
    for index, count in enumerate(histogram):
        cumulative += float(count)
        if cumulative >= threshold:
            return float(cfg.quality_utility[index])
    return float(cfg.quality_utility[-1])


def run_policy(
    cfg: P3Config,
    policy: str,
    output_dir: Path,
    ppo_agent: "PPOAgent | None" = None,
) -> PolicyRunResult:
    if policy == "ppo" and ppo_agent is None:
        raise ValueError("policy='ppo' requires a loaded PPOAgent")
    state = initialize_state(cfg)
    controller = SlowRolloutController(cfg)
    frame_rows: list[dict] = []
    total_float = {
        key: 0.0
        for key in (
            "delivered_chunks",
            "quality_utility",
            "quality_level_sum",
            "degradation",
            "queue_sum",
            "energy_consumed_j",
            "energy_charged_j",
            "original_cost",
            "dpp_cost",
            "uav_distance_sum_m",
        )
    }
    total_int = {
        key: 0
        for key in (
            "quality_switches",
            "quality_transitions",
            "stall_user_slots",
            "served_user_slots",
            "large_queue_violation_user_slots",
            "hired_uav_frames",
            "relocation_events",
            "charging_slots",
            "charging_need_events",
            "stranded_before_charge_events",
            "return_to_charge_events",
            "precharge_depletion_events",
            "precharge_reserve_breach_events",
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
            "uav_scheduled_user_slots",
        )
    }
    quality_hist = np.zeros(cfg.num_quality_levels, dtype=np.float64)
    distance_opportunities = np.zeros(cfg.num_distance_bins, dtype=np.float64)
    distance_served = np.zeros_like(distance_opportunities)
    distance_stall = np.zeros_like(distance_opportunities)
    distance_delivered = np.zeros_like(distance_opportunities)
    distance_quality = np.zeros_like(distance_opportunities)
    distance_power = np.zeros_like(distance_opportunities)
    point_counts = np.zeros(len(cfg.candidate_offsets_m), dtype=np.int64)
    peak_uav_power = 0.0
    started = time.perf_counter()

    for frame in range(cfg.num_frames):
        membership = region_membership(state, cfg)
        actions: list[RegionAction] = []
        predicted_scores: list[float] = []
        enumerated_counts: list[int] = []
        evaluated_counts: list[int] = []
        selection_started = time.perf_counter()

        for region in range(cfg.num_regions):
            users = np.flatnonzero(membership == region).tolist()
            if policy == "ppo":
                candidates = controller.candidate_actions(
                    state,
                    region,
                    users,
                    policy="ppo",
                    frame=frame,
                )
                state_features = build_state_features(state, region, users, cfg)
                candidate_features = build_candidate_feature_matrix(
                    state, candidates.actions, cfg
                )
                choice = ppo_agent.select(
                    state_features,
                    candidate_features,
                    deterministic=True,
                )
                actions.append(candidates.actions[choice.action_index])
                predicted_scores.append(math.nan)
                enumerated_counts.append(candidates.enumerated_count)
                evaluated_counts.append(candidates.evaluated_count)
            else:
                selection = controller.select(
                    state,
                    region,
                    users,
                    frame,
                    policy,
                )
                actions.append(selection.action)
                predicted_scores.append(selection.estimated_dpp_cost)
                enumerated_counts.append(selection.enumerated_count)
                evaluated_counts.append(selection.evaluated_count)
        selection_seconds = time.perf_counter() - selection_started

        realized_trace = generate_frame_trace(cfg, controller.realized_seed(frame))
        frame_float = {key: 0.0 for key in total_float}
        frame_int = {key: 0 for key in total_int}
        frame_quality_hist = np.zeros_like(quality_hist)
        frame_peak_power = 0.0

        for region, action in enumerate(actions):
            users = np.flatnonzero(membership == region).tolist()
            result = simulate_region_frame(
                state,
                action,
                users,
                realized_trace,
                cfg,
                controller.fast_controller,
            )
            apply_region_result(state, region, users, result)
            frame_float["delivered_chunks"] += result.delivered_chunks
            frame_float["quality_utility"] += result.quality_utility
            frame_float["quality_level_sum"] += result.quality_level_sum
            frame_float["degradation"] += result.degradation
            frame_float["queue_sum"] += result.queue_sum
            frame_float["energy_consumed_j"] += result.energy_consumed_j
            frame_float["energy_charged_j"] += result.energy_charged_j
            frame_float["original_cost"] += result.original_cost
            frame_float["dpp_cost"] += result.frame_dpp_cost
            frame_float["uav_distance_sum_m"] += result.uav_distance_sum_m
            frame_int["quality_switches"] += result.quality_switches
            frame_int["quality_transitions"] += result.quality_transitions
            frame_int["stall_user_slots"] += result.stall_user_slots
            frame_int["served_user_slots"] += result.served_user_slots
            frame_int["large_queue_violation_user_slots"] += (
                result.large_queue_violation_user_slots
            )
            frame_int["hired_uav_frames"] += int(action.hired)
            frame_int["relocation_events"] += result.relocation_events
            frame_int["charging_slots"] += result.charging_slots
            frame_int["charging_need_events"] += result.charging_need_events
            frame_int["stranded_before_charge_events"] += (
                result.stranded_before_charge_events
            )
            frame_int["return_to_charge_events"] += result.return_to_charge_events
            frame_int["precharge_depletion_events"] += result.precharge_depletion_events
            frame_int["precharge_reserve_breach_events"] += (
                result.precharge_reserve_breach_events
            )
            frame_int["battery_reserve_violations"] += result.battery_reserve_violations
            frame_int["power_violations"] += result.power_violations
            frame_int["provider_violations"] += result.provider_violations
            frame_int["uav_scheduled_user_slots"] += result.uav_scheduled_user_slots
            frame_quality_hist += result.quality_histogram
            distance_opportunities += result.distance_opportunities
            distance_served += result.distance_served_slots
            distance_stall += result.distance_stall_slots
            distance_delivered += result.distance_delivered_chunks
            distance_quality += result.distance_quality_utility
            distance_power += result.distance_power_sum_w
            frame_peak_power = max(frame_peak_power, result.uav_total_power_peak_w)
            if action.hired:
                point_counts[action.point_index] += 1

        advance_mobility_one_frame(state, cfg)
        validate_state(state, cfg)
        user_slots = cfg.num_users * cfg.frame_slots
        frame_rows.append(
            {
                "seed": cfg.seed,
                "policy": policy,
                "frame": frame,
                "actions_json": json.dumps(
                    [action_record(action, cfg) for action in actions],
                    separators=(",", ":"),
                ),
                "hired_uavs": frame_int["hired_uav_frames"],
                "stall_ratio": safe_ratio(frame_int["stall_user_slots"], user_slots),
                "served_user_ratio": safe_ratio(frame_int["served_user_slots"], user_slots),
                "delivered_chunks": frame_float["delivered_chunks"],
                "average_quality_utility": safe_ratio(
                    frame_float["quality_utility"], frame_float["delivered_chunks"]
                ),
                "average_quality_level": safe_ratio(
                    frame_float["quality_level_sum"], frame_float["delivered_chunks"]
                ),
                "quality_p05_utility": quality_percentile(frame_quality_hist, cfg, 0.05),
                "quality_switch_rate": safe_ratio(
                    frame_int["quality_switches"], frame_int["quality_transitions"]
                ),
                "degradation_per_chunk": safe_ratio(
                    frame_float["degradation"], frame_float["delivered_chunks"]
                ),
                "mean_queue": safe_ratio(frame_float["queue_sum"], user_slots),
                "max_queue_end": float(np.max(state.queue)),
                "large_queue_violation_rate": safe_ratio(
                    frame_int["large_queue_violation_user_slots"], user_slots
                ),
                "min_battery_soc": float(np.min(state.battery_j) / cfg.battery_capacity_j),
                "mean_battery_soc": float(np.mean(state.battery_j) / cfg.battery_capacity_j),
                "charging_fraction": safe_ratio(
                    frame_int["charging_slots"], cfg.num_uavs * cfg.frame_slots
                ),
                "return_to_charge_events": frame_int["return_to_charge_events"],
                "charging_need_events": frame_int["charging_need_events"],
                "stranded_before_charge_rate": safe_ratio(
                    frame_int["stranded_before_charge_events"],
                    frame_int["charging_need_events"],
                ),
                "precharge_depletion_rate": safe_ratio(
                    frame_int["precharge_depletion_events"],
                    frame_int["return_to_charge_events"],
                ),
                "precharge_reserve_breach_rate": safe_ratio(
                    frame_int["precharge_reserve_breach_events"],
                    frame_int["return_to_charge_events"],
                ),
                "mean_uav_user_distance_m": safe_ratio(
                    frame_float["uav_distance_sum_m"],
                    frame_int["uav_scheduled_user_slots"],
                ),
                "peak_uav_total_power_w": frame_peak_power,
                "energy_consumed_j": frame_float["energy_consumed_j"],
                "energy_charged_j": frame_float["energy_charged_j"],
                "original_cost": frame_float["original_cost"],
                "dpp_cost": frame_float["dpp_cost"],
                "rollout_predicted_dpp": (
                    float(sum(predicted_scores))
                    if all(math.isfinite(value) for value in predicted_scores)
                    else math.nan
                ),
                "enumerated_actions": int(sum(enumerated_counts)),
                "evaluated_actions": int(sum(evaluated_counts)),
                "selection_seconds": float(selection_seconds),
                "battery_reserve_violations": frame_int["battery_reserve_violations"],
                "power_violations": frame_int["power_violations"],
                "provider_violations": frame_int["provider_violations"],
            }
        )
        for key in total_float:
            total_float[key] += frame_float[key]
        for key in total_int:
            total_int[key] += frame_int[key]
        quality_hist += frame_quality_hist
        peak_uav_power = max(peak_uav_power, frame_peak_power)

    runtime = time.perf_counter() - started
    total_user_slots = cfg.num_frames * cfg.num_users * cfg.frame_slots
    total_uav_frames = cfg.num_frames * cfg.num_regions
    total_uav_slots = total_uav_frames * cfg.frame_slots
    summary = {
        "seed": cfg.seed,
        "policy": policy,
        "frames": cfg.num_frames,
        "user_slots": total_user_slots,
        "stall_ratio": safe_ratio(total_int["stall_user_slots"], total_user_slots),
        "served_user_ratio": safe_ratio(total_int["served_user_slots"], total_user_slots),
        "hire_rate": safe_ratio(total_int["hired_uav_frames"], total_uav_frames),
        "charging_fraction": safe_ratio(total_int["charging_slots"], total_uav_slots),
        "delivered_chunks_per_user_slot": safe_ratio(
            total_float["delivered_chunks"], total_user_slots
        ),
        "average_quality_utility": safe_ratio(
            total_float["quality_utility"], total_float["delivered_chunks"]
        ),
        "average_quality_level": safe_ratio(
            total_float["quality_level_sum"], total_float["delivered_chunks"]
        ),
        "quality_p05_utility": quality_percentile(quality_hist, cfg, 0.05),
        "quality_switch_rate": safe_ratio(
            total_int["quality_switches"], total_int["quality_transitions"]
        ),
        "degradation_per_chunk": safe_ratio(
            total_float["degradation"], total_float["delivered_chunks"]
        ),
        "mean_queue": safe_ratio(total_float["queue_sum"], total_user_slots),
        "large_queue_violation_rate": safe_ratio(
            total_int["large_queue_violation_user_slots"], total_user_slots
        ),
        "original_cost_per_user_slot": safe_ratio(
            total_float["original_cost"], total_user_slots
        ),
        "dpp_cost_per_user_slot": safe_ratio(
            total_float["dpp_cost"], total_user_slots
        ),
        "energy_per_delivered_chunk_j": safe_ratio(
            total_float["energy_consumed_j"], total_float["delivered_chunks"]
        ),
        "min_final_battery_soc": float(np.min(state.battery_j) / cfg.battery_capacity_j),
        "return_to_charge_events": total_int["return_to_charge_events"],
        "charging_need_events": total_int["charging_need_events"],
        "stranded_before_charge_events": total_int["stranded_before_charge_events"],
        "stranded_before_charge_rate": safe_ratio(
            total_int["stranded_before_charge_events"],
            total_int["charging_need_events"],
        ),
        "precharge_depletion_events": total_int["precharge_depletion_events"],
        "precharge_depletion_rate": safe_ratio(
            total_int["precharge_depletion_events"], total_int["return_to_charge_events"]
        ),
        "precharge_reserve_breach_events": total_int[
            "precharge_reserve_breach_events"
        ],
        "precharge_reserve_breach_rate": safe_ratio(
            total_int["precharge_reserve_breach_events"],
            total_int["return_to_charge_events"],
        ),
        "mean_uav_user_distance_m": safe_ratio(
            total_float["uav_distance_sum_m"], total_int["uav_scheduled_user_slots"]
        ),
        "peak_uav_total_power_w": peak_uav_power,
        "battery_reserve_violations": total_int["battery_reserve_violations"],
        "power_violations": total_int["power_violations"],
        "provider_violations": total_int["provider_violations"],
        "runtime_seconds": float(runtime),
    }

    distance_rows = []
    for index in range(cfg.num_distance_bins):
        left = cfg.distance_bin_edges_m[index]
        right = cfg.distance_bin_edges_m[index + 1]
        distance_rows.append(
            {
                "seed": cfg.seed,
                "policy": policy,
                "bin_index": index,
                "distance_left_m": left,
                "distance_right_m": right,
                "opportunities": distance_opportunities[index],
                "served_slots": distance_served[index],
                "stall_slots": distance_stall[index],
                "delivered_chunks": distance_delivered[index],
                "quality_utility_sum": distance_quality[index],
                "power_sum_w": distance_power[index],
                "uav_service_ratio": safe_ratio(
                    distance_served[index], distance_opportunities[index]
                ),
                "stall_ratio": safe_ratio(
                    distance_stall[index], distance_opportunities[index]
                ),
                "average_quality_utility": safe_ratio(
                    distance_quality[index], distance_delivered[index]
                ),
                "average_required_power_w": safe_ratio(
                    distance_power[index], distance_served[index]
                ),
            }
        )
    point_rows = [
        {
            "seed": cfg.seed,
            "policy": policy,
            "point_index": index,
            "offset_m": cfg.candidate_offsets_m[index],
            "selection_count": int(point_counts[index]),
            "hired_uav_frames": total_int["hired_uav_frames"],
            "selection_rate_given_hired": safe_ratio(
                point_counts[index], total_int["hired_uav_frames"]
            ),
        }
        for index in range(len(cfg.candidate_offsets_m))
    ]
    quality_rows = [
        {
            "seed": cfg.seed,
            "policy": policy,
            "quality_index": index,
            "quality_level": index + 1,
            "quality_utility": cfg.quality_utility[index],
            "chunk_size_bits": cfg.chunk_size_bits[index],
            "delivered_chunks": quality_hist[index],
            "delivered_chunk_share": safe_ratio(
                quality_hist[index], float(np.sum(quality_hist))
            ),
        }
        for index in range(cfg.num_quality_levels)
    ]
    write_csv(output_dir / f"frames_{policy}_seed{cfg.seed}.csv", frame_rows)
    return PolicyRunResult(summary, frame_rows, distance_rows, point_rows, quality_rows)
