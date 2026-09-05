from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from agent.P3.features import build_candidate_signatures, build_state_features
from agent.P3.ppo_agent import PPOAgent, PPOTransition, finish_trajectory
from agent.P3.slow_rollout_controller import SlowRolloutController
from config_p3 import P3Config
from env.p3.environment import (
    advance_mobility_one_frame,
    apply_region_result,
    generate_frame_trace,
    simulate_region_frame,
)
from env.p3.topology import initialize_state, region_membership, validate_state
from run.p3_common import array_max_or_current, run_policy, safe_ratio, write_csv


ProgressCallback = Callable[[int, dict], None]


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one validation seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("validation seeds must be unique")
    return seeds


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def json_safe(value):
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write_csv(temporary, rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class EventLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.events_path = output_dir / "training_events.jsonl"
        self.status_path = output_dir / "status.json"

    def emit(self, payload: dict, echo_json: bool = False) -> dict:
        event = json_safe(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )
        with self.events_path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(
                json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            )
            stream.flush()
        atomic_write_json(self.status_path, event)
        if echo_json:
            print(
                json.dumps(event, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        return event


def quality_percentile(
    histogram: np.ndarray,
    cfg: P3Config,
    quantile: float,
) -> float:
    total = float(histogram.sum())
    if total <= 0.0:
        return 0.0
    threshold = total * quantile
    cumulative = 0.0
    for index, count in enumerate(histogram):
        cumulative += float(count)
        if cumulative >= threshold:
            return float(cfg.quality_utility[index])
    return float(cfg.quality_utility[-1])


def build_episode_summary(
    totals: dict[str, float],
    quality_histogram: np.ndarray,
    min_battery_soc: float,
    peak_uav_total_power_w: float,
    processed_frames: int,
    cfg: P3Config,
) -> dict:
    total_user_slots = processed_frames * cfg.num_users * cfg.frame_slots
    total_uav_frames = processed_frames * cfg.num_regions
    total_uav_slots = total_uav_frames * cfg.frame_slots
    return {
        "processed_frames": int(processed_frames),
        "scaled_return": float(totals["scaled_reward"]),
        "dpp_cost_per_user_slot": safe_ratio(totals["dpp_cost"], total_user_slots),
        "original_cost_per_user_slot": safe_ratio(
            totals["original_cost"], total_user_slots
        ),
        "stall_ratio": safe_ratio(totals["stall_user_slots"], total_user_slots),
        "served_user_ratio": safe_ratio(totals["served_user_slots"], total_user_slots),
        "delivered_chunks_per_user_slot": safe_ratio(
            totals["delivered_chunks"], total_user_slots
        ),
        "average_quality_utility": safe_ratio(
            totals["quality_utility"], totals["delivered_chunks"]
        ),
        "quality_p05_utility": quality_percentile(quality_histogram, cfg, 0.05),
        "quality_switch_rate": safe_ratio(
            totals["quality_switches"], totals["quality_transitions"]
        ),
        "degradation_per_chunk": safe_ratio(
            totals["degradation"], totals["delivered_chunks"]
        ),
        "mean_queue": safe_ratio(totals["queue_sum"], total_user_slots),
        "mean_z": safe_ratio(totals["z_sum"], total_user_slots),
        "max_queue": float(totals["max_queue"]),
        "large_queue_violation_rate": safe_ratio(
            totals["large_queue_violation_user_slots"], total_user_slots
        ),
        "hire_rate": safe_ratio(totals["hired_uav_frames"], total_uav_frames),
        "mean_hire_probability": safe_ratio(
            totals["hire_probability_sum"], totals["policy_decisions"]
        ),
        "charging_fraction": safe_ratio(totals["charging_slots"], total_uav_slots),
        "return_to_charge_events": int(totals["return_to_charge_events"]),
        "charging_need_events": int(totals["charging_need_events"]),
        "stranded_before_charge_events": int(
            totals["stranded_before_charge_events"]
        ),
        "stranded_before_charge_rate": safe_ratio(
            totals["stranded_before_charge_events"],
            totals["charging_need_events"],
        ),
        "precharge_depletion_events": int(totals["precharge_depletion_events"]),
        "precharge_depletion_rate": safe_ratio(
            totals["precharge_depletion_events"],
            totals["return_to_charge_events"],
        ),
        "precharge_reserve_breach_events": int(
            totals["precharge_reserve_breach_events"]
        ),
        "precharge_reserve_breach_rate": safe_ratio(
            totals["precharge_reserve_breach_events"],
            totals["return_to_charge_events"],
        ),
        "min_battery_soc": float(min_battery_soc),
        "energy_per_delivered_chunk_j": safe_ratio(
            totals["energy_consumed_j"], totals["delivered_chunks"]
        ),
        "mean_uav_user_distance_m": safe_ratio(
            totals["uav_distance_sum_m"], totals["uav_scheduled_user_slots"]
        ),
        "peak_uav_total_power_w": float(peak_uav_total_power_w),
        "battery_reserve_violations": int(totals["battery_reserve_violations"]),
        "power_violations": int(totals["power_violations"]),
        "provider_violations": int(totals["provider_violations"]),
    }


def train_episode(
    agent: PPOAgent,
    cfg: P3Config,
    progress_interval_frames: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict, list[PPOTransition]]:
    state = initialize_state(cfg)
    controller = SlowRolloutController(cfg)
    trajectories: dict[int, list[PPOTransition]] = {
        region: [] for region in range(cfg.num_regions)
    }
    totals = {
        key: 0.0
        for key in (
            "scaled_reward",
            "dpp_cost",
            "original_cost",
            "stall_user_slots",
            "served_user_slots",
            "delivered_chunks",
            "quality_utility",
            "degradation",
            "quality_switches",
            "quality_transitions",
            "queue_sum",
            "z_sum",
            "max_queue",
            "large_queue_violation_user_slots",
            "hired_uav_frames",
            "charging_slots",
            "charging_need_events",
            "stranded_before_charge_events",
            "return_to_charge_events",
            "precharge_depletion_events",
            "precharge_reserve_breach_events",
            "energy_consumed_j",
            "uav_distance_sum_m",
            "uav_scheduled_user_slots",
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
            "hire_probability_sum",
            "policy_decisions",
        )
    }
    quality_histogram = np.zeros(cfg.num_quality_levels, dtype=np.float64)
    min_battery_soc = float(np.min(state.battery_j) / cfg.battery_capacity_j)
    peak_uav_total_power_w = 0.0
    totals["max_queue"] = array_max_or_current(0.0, state.queue)

    for frame in range(cfg.num_frames):
        membership = region_membership(state, cfg)
        pending: list[dict] = []
        actions = []
        for region in range(cfg.num_regions):
            users = np.flatnonzero(membership == region).tolist()
            candidates = controller.candidate_actions(
                state,
                region,
                users,
                policy="ppo",
                frame=frame,
            )
            state_features = build_state_features(state, region, users, cfg)
            candidate_signatures = build_candidate_signatures(
                candidates.actions, cfg
            )
            choice = agent.select(
                state_features,
                candidate_signatures,
                deterministic=False,
            )
            action = candidates.actions[choice.action_index]
            actions.append(action)
            pending.append(
                {
                    "region": region,
                    "users": users,
                    "state_features": state_features,
                    "candidate_signatures": candidate_signatures,
                    "choice": choice,
                }
            )

        realized_trace = generate_frame_trace(cfg, controller.realized_seed(frame))
        frame_results = []
        for item, action in zip(pending, actions):
            result = simulate_region_frame(
                state,
                action,
                item["users"],
                realized_trace,
                cfg,
                controller.fast_controller,
            )
            frame_results.append(result)
            apply_region_result(state, item["region"], item["users"], result)
            totals["dpp_cost"] += result.frame_dpp_cost
            totals["original_cost"] += result.original_cost
            totals["stall_user_slots"] += result.stall_user_slots
            totals["served_user_slots"] += result.served_user_slots
            totals["delivered_chunks"] += result.delivered_chunks
            totals["quality_utility"] += result.quality_utility
            totals["degradation"] += result.degradation
            totals["quality_switches"] += result.quality_switches
            totals["quality_transitions"] += result.quality_transitions
            totals["queue_sum"] += result.queue_sum
            totals["z_sum"] += float(np.sum(result.z_samples))
            totals["max_queue"] = array_max_or_current(
                totals["max_queue"], result.queue_samples
            )
            totals["large_queue_violation_user_slots"] += (
                result.large_queue_violation_user_slots
            )
            totals["hired_uav_frames"] += action.hired
            totals["charging_slots"] += result.charging_slots
            totals["charging_need_events"] += result.charging_need_events
            totals["stranded_before_charge_events"] += (
                result.stranded_before_charge_events
            )
            totals["return_to_charge_events"] += result.return_to_charge_events
            totals["precharge_depletion_events"] += (
                result.precharge_depletion_events
            )
            totals["precharge_reserve_breach_events"] += (
                result.precharge_reserve_breach_events
            )
            totals["energy_consumed_j"] += result.energy_consumed_j
            totals["uav_distance_sum_m"] += result.uav_distance_sum_m
            totals["uav_scheduled_user_slots"] += result.uav_scheduled_user_slots
            totals["battery_reserve_violations"] += (
                result.battery_reserve_violations
            )
            totals["power_violations"] += result.power_violations
            totals["provider_violations"] += result.provider_violations
            quality_histogram += result.quality_histogram
            peak_uav_total_power_w = max(
                peak_uav_total_power_w,
                float(result.uav_total_power_peak_w),
            )

        advance_mobility_one_frame(state, cfg)
        validate_state(state, cfg)
        totals["max_queue"] = array_max_or_current(
            totals["max_queue"], state.queue
        )
        min_battery_soc = min(
            min_battery_soc,
            float(np.min(state.battery_j) / cfg.battery_capacity_j),
        )
        done = frame == cfg.num_frames - 1
        next_membership = region_membership(state, cfg)
        for item, result in zip(pending, frame_results):
            region = int(item["region"])
            next_users = np.flatnonzero(next_membership == region).tolist()
            next_features = build_state_features(state, region, next_users, cfg)
            next_value = 0.0 if done else agent.value(next_features)
            reward = -result.frame_dpp_cost * cfg.ppo_reward_scale
            totals["scaled_reward"] += reward
            choice = item["choice"]
            totals["hire_probability_sum"] += choice.hire_probability
            totals["policy_decisions"] += 1.0
            trajectories[region].append(
                PPOTransition(
                    state_features=item["state_features"],
                    candidate_signatures=item["candidate_signatures"],
                    action_index=choice.action_index,
                    old_log_prob=choice.log_prob,
                    old_value=choice.value,
                    reward=float(reward),
                    next_value=float(next_value),
                    done=done,
                )
            )

        processed_frames = frame + 1
        should_report = (
            progress_callback is not None
            and progress_interval_frames > 0
            and (
                processed_frames == 1
                or processed_frames % progress_interval_frames == 0
                or processed_frames == cfg.num_frames
            )
        )
        if should_report:
            progress_callback(
                processed_frames,
                build_episode_summary(
                    totals,
                    quality_histogram,
                    min_battery_soc,
                    peak_uav_total_power_w,
                    processed_frames,
                    cfg,
                ),
            )

    transitions: list[PPOTransition] = []
    for region_trajectory in trajectories.values():
        finish_trajectory(region_trajectory, cfg)
        transitions.extend(region_trajectory)
    return (
        build_episode_summary(
            totals,
            quality_histogram,
            min_battery_soc,
            peak_uav_total_power_w,
            cfg.num_frames,
            cfg,
        ),
        transitions,
    )


VALIDATION_MEAN_METRICS = {
    "validation_dpp_per_user_slot": "dpp_cost_per_user_slot",
    "validation_original_cost_per_user_slot": "original_cost_per_user_slot",
    "validation_stall_ratio": "stall_ratio",
    "validation_served_user_ratio": "served_user_ratio",
    "validation_delivered_chunks_per_user_slot": "delivered_chunks_per_user_slot",
    "validation_average_quality_utility": "average_quality_utility",
    "validation_quality_p05_utility": "quality_p05_utility",
    "validation_quality_switch_rate": "quality_switch_rate",
    "validation_hire_rate": "hire_rate",
    "validation_mean_z": "mean_z",
    "validation_p95_z": "p95_z",
    "validation_max_queue": "max_queue",
    "validation_large_queue_violation_rate": "large_queue_violation_rate",
    "validation_steady_state_stall_ratio": "steady_state_stall_ratio",
    "validation_worst_user_stall_ratio": "worst_user_stall_ratio",
    "validation_worst_user_quality_utility": "worst_user_quality_utility",
    "validation_jain_service_fairness": "jain_service_fairness",
    "validation_rsu_capacity_utilization": "rsu_capacity_utilization",
    "validation_uav_capacity_utilization_given_hire": "uav_capacity_utilization_given_hire",
    "validation_min_final_battery_soc": "min_final_battery_soc",
    "validation_stranded_before_charge_rate": "stranded_before_charge_rate",
    "validation_precharge_depletion_rate": "precharge_depletion_rate",
    "validation_precharge_reserve_breach_rate": "precharge_reserve_breach_rate",
    "validation_mean_uav_user_distance_m": "mean_uav_user_distance_m",
}


def empty_validation_summary() -> dict:
    result = {key: math.nan for key in VALIDATION_MEAN_METRICS}
    result.update(
        {
            "validation_dpp_std": math.nan,
            "validation_battery_reserve_violations": math.nan,
            "validation_power_violations": math.nan,
            "validation_provider_violations": math.nan,
            "validation_rsu_only_stall_ratio": math.nan,
            "validation_stall_improvement_vs_rsu": math.nan,
            "validation_action_collapse": math.nan,
        }
    )
    return result


def validate_agent(
    agent: PPOAgent,
    agent_cfg: P3Config,
    validation_seeds: tuple[int, ...],
    episode: int,
    output_dir: Path,
    logger: EventLogger,
    baseline_cache: dict[int, dict],
) -> dict:
    summaries: list[dict] = []
    validation_dir = output_dir / "validation" / f"episode_{episode + 1:04d}"
    for index, seed in enumerate(validation_seeds, start=1):
        print(
            f"[VAL-START] ep={episode + 1:04d} "
            f"seed={seed} ({index}/{len(validation_seeds)})",
            flush=True,
        )
        started = time.perf_counter()
        validation_cfg = replace(agent_cfg, seed=seed)
        if seed not in baseline_cache:
            baseline_cache[seed] = run_policy(
                validation_cfg,
                "rsu_only",
                validation_dir,
                write_outputs=False,
            ).summary
        summary = run_policy(
            validation_cfg,
            "ppo",
            validation_dir,
            ppo_agent=agent,
        ).summary
        summaries.append(summary)
        elapsed = time.perf_counter() - started
        print(
            f"[VAL-END] ep={episode + 1:04d} seed={seed} "
            f"dpp={summary['dpp_cost_per_user_slot']:.6f} "
            f"stall={summary['stall_ratio']:.4f} "
            f"delivery={summary['delivered_chunks_per_user_slot']:.4f} "
            f"quality={summary['average_quality_utility']:.4f} "
            f"hire={summary['hire_rate']:.4f} "
            f"soc={summary['min_final_battery_soc']:.4f} "
            f"elapsed={format_duration(elapsed)}",
            flush=True,
        )
        logger.emit(
            {
                "event": "validation_seed_end",
                "episode": episode,
                "validation_seed": seed,
                "validation_index": index,
                "validation_count": len(validation_seeds),
                "elapsed_seconds": elapsed,
                **{
                    key: summary[key]
                    for key in (
                        "dpp_cost_per_user_slot",
                        "stall_ratio",
                        "served_user_ratio",
                        "delivered_chunks_per_user_slot",
                        "average_quality_utility",
                        "quality_p05_utility",
                        "hire_rate",
                        "min_final_battery_soc",
                        "stranded_before_charge_rate",
                        "precharge_depletion_rate",
                        "precharge_reserve_breach_rate",
                    )
                },
            }
        )

    aggregate = {
        output_key: float(np.mean([row[source_key] for row in summaries]))
        for output_key, source_key in VALIDATION_MEAN_METRICS.items()
    }
    aggregate["validation_dpp_std"] = float(
        np.std([row["dpp_cost_per_user_slot"] for row in summaries])
    )
    rsu_stall = float(
        np.mean([baseline_cache[seed]["stall_ratio"] for seed in validation_seeds])
    )
    aggregate["validation_rsu_only_stall_ratio"] = rsu_stall
    aggregate["validation_stall_improvement_vs_rsu"] = (
        rsu_stall - aggregate["validation_stall_ratio"]
    )
    hire_rate = aggregate["validation_hire_rate"]
    aggregate["validation_action_collapse"] = int(
        not 0.01 <= hire_rate <= 0.99
    )
    for key in (
        "battery_reserve_violations",
        "power_violations",
        "provider_violations",
    ):
        aggregate[f"validation_{key}"] = int(sum(row[key] for row in summaries))
    return aggregate


def rolling_mean(rows: list[dict], metric: str, window: int) -> float:
    values = [
        float(row[metric])
        for row in rows[-window:]
        if math.isfinite(float(row[metric]))
    ]
    return float(np.mean(values)) if values else math.nan


def finite_xy(rows: list[dict], metric: str) -> tuple[list[int], list[float]]:
    points = []
    for row in rows:
        value = float(row.get(metric, math.nan))
        if math.isfinite(value):
            points.append((int(row["episode"]), value))
    return [point[0] for point in points], [point[1] for point in points]


def plot_training_curve(rows: list[dict], output_path: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    purple = "#7030A0"
    orange = "#ED7D31"
    blue = "#4472C4"
    green = "#70AD47"
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    def draw(
        axis,
        metric: str,
        title: str,
        ylabel: str,
        *,
        label: str | None = None,
        color: str = purple,
        marker: str | None = None,
    ) -> None:
        x, y = finite_xy(rows, metric)
        if x:
            axis.plot(
                x,
                y,
                color=color,
                linewidth=1.4,
                marker=marker,
                markersize=4,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        if label is not None:
            axis.legend(fontsize=8)

    draw(
        axes[0, 0],
        "dpp_cost_per_user_slot",
        "Train / validation DPP",
        "cost per user-slot",
        label="train",
    )
    x, y = finite_xy(rows, "validation_dpp_per_user_slot")
    if x:
        axes[0, 0].plot(
            x,
            y,
            color=orange,
            marker="o",
            linewidth=1.4,
            markersize=5,
            label="validation mean",
        )
        axes[0, 0].legend(fontsize=8)
    draw(axes[0, 1], "stall_ratio", "Training stall ratio", "fraction")
    draw(
        axes[0, 2],
        "delivered_chunks_per_user_slot",
        "Delivered chunks",
        "chunks per user-slot",
    )
    draw(
        axes[0, 3],
        "original_cost_per_user_slot",
        "Original objective",
        "cost per user-slot",
    )
    draw(
        axes[1, 0],
        "average_quality_utility",
        "Delivered quality",
        "utility",
        label="mean",
    )
    x, y = finite_xy(rows, "quality_p05_utility")
    if x:
        axes[1, 0].plot(x, y, color=orange, linewidth=1.2, label="p05")
        axes[1, 0].legend(fontsize=8)
    draw(axes[1, 1], "hire_rate", "UAV hire rate", "fraction")
    draw(axes[1, 2], "mean_queue", "Playback queue", "mean chunks")
    draw(
        axes[1, 3],
        "mean_uav_user_distance_m",
        "UAV-user distance",
        "metres",
    )
    draw(axes[2, 0], "min_battery_soc", "Minimum battery SoC", "fraction")
    draw(
        axes[2, 1],
        "stranded_before_charge_rate",
        "Charging-risk rates",
        "fraction",
        label="stranded / need",
    )
    for metric, label, color in (
        ("precharge_depletion_rate", "depletion / return", orange),
        ("precharge_reserve_breach_rate", "reserve breach / return", blue),
    ):
        x, y = finite_xy(rows, metric)
        if x:
            axes[2, 1].plot(x, y, color=color, linewidth=1.2, label=label)
    axes[2, 1].legend(fontsize=8)
    draw(
        axes[2, 2],
        "normalized_entropy",
        "Normalized policy entropy",
        "fraction of max entropy",
    )
    draw(axes[2, 3], "approx_kl", "PPO approximate KL", "KL")
    target_x, targets = finite_xy(rows, "target_kl")
    if target_x:
        axes[2, 3].plot(
            target_x,
            targets,
            color=green,
            linestyle="--",
            linewidth=1.0,
            label="target",
        )
        axes[2, 3].legend(fontsize=8)

    fig.suptitle("P3 Upper-level PPO Training Diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    fig.savefig(temporary, dpi=180)
    plt.close(fig)
    os.replace(temporary, output_path)


def check_hard_constraints(summary: dict) -> int:
    return sum(
        int(summary[key])
        for key in (
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        )
    )


def check_finite_metrics(summary: dict, context: str) -> None:
    invalid = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    }
    if invalid:
        raise FloatingPointError(f"non-finite metrics in {context}: {invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train upper-level PPO with exact P3 fast solver"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--validation-seeds",
        type=parse_seed_list,
        default=parse_seed_list("92026,92027,92028"),
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help="backward-compatible single-seed override",
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping; recommended for final learning curves",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--progress-interval-frames", type=int, default=20)
    parser.add_argument("--plot-interval", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--fail-on-hard-violation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/p3_ppo"))
    args = parser.parse_args()

    if args.episodes <= 0 or args.frames <= 0:
        parser.error("episodes and frames must be positive")
    if args.validation_interval <= 0:
        parser.error("validation-interval must be positive")
    if args.early_stopping_patience < 0:
        parser.error("early-stopping-patience cannot be negative")
    if args.min_delta < 0.0:
        parser.error("min-delta cannot be negative")
    for name in (
        "checkpoint_interval",
        "progress_interval_frames",
        "plot_interval",
        "rolling_window",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")

    validation_seeds = (
        (int(args.validation_seed),)
        if args.validation_seed is not None
        else tuple(args.validation_seeds)
    )
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(output_dir)

    agent_cfg = P3Config(seed=args.seed, num_frames=args.frames)
    overrides = {}
    if args.learning_rate is not None:
        overrides["ppo_learning_rate"] = args.learning_rate
    if args.entropy_coef is not None:
        overrides["ppo_entropy_coef"] = args.entropy_coef
    if args.target_kl is not None:
        overrides["ppo_target_kl"] = args.target_kl
    agent_cfg = replace(agent_cfg, **overrides)
    agent = PPOAgent(agent_cfg, device=args.device)

    run_config = {
        "episodes": args.episodes,
        "frames": args.frames,
        "seed": args.seed,
        "validation_seeds": validation_seeds,
        "validation_interval": args.validation_interval,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "checkpoint_interval": args.checkpoint_interval,
        "progress_interval_frames": args.progress_interval_frames,
        "plot_interval": args.plot_interval,
        "rolling_window": args.rolling_window,
        "device": str(agent.device),
        "output": str(output_dir),
        "config": asdict(agent_cfg),
    }
    atomic_write_json(output_dir / "run_config.json", run_config)
    agent.save(output_dir / "initial.pt", metadata={"episode": -1, **run_config})

    print(
        f"[RUN] episodes={args.episodes} frames={args.frames} seed={args.seed} "
        f"validation_seeds={','.join(map(str, validation_seeds))} "
        f"validation_interval={args.validation_interval} "
        f"patience={args.early_stopping_patience} "
        f"lr={agent_cfg.ppo_learning_rate:g} "
        f"target_kl={agent_cfg.ppo_target_kl:g} device={agent.device}",
        flush=True,
    )
    logger.emit({"event": "run_start", **run_config}, echo_json=True)

    rows: list[dict] = []
    best_validation_dpp = math.inf
    best_selected_validation_dpp = math.inf
    best_validation_stall = math.inf
    best_validation_original_cost = math.inf
    best_validation_deployable = False
    best_validation_episode = -1
    validations_without_improvement = 0
    started = time.perf_counter()
    termination_reason = "completed"
    validation_baseline_cache: dict[int, dict] = {}

    try:
        for episode in range(args.episodes):
            env_cfg = replace(agent_cfg, seed=args.seed + episode * 1009)
            episode_started = time.perf_counter()
            print(
                f"[EP-START] ep={episode + 1:04d}/{args.episodes:04d} "
                f"environment_seed={env_cfg.seed}",
                flush=True,
            )

            def report_frame(processed_frames: int, partial: dict) -> None:
                elapsed = time.perf_counter() - episode_started
                seconds_per_frame = elapsed / processed_frames
                eta_episode = seconds_per_frame * (args.frames - processed_frames)
                logger.emit(
                    {
                        "event": "frame_progress",
                        "episode": episode,
                        "episodes": args.episodes,
                        "environment_seed": env_cfg.seed,
                        "frame": processed_frames,
                        "frames": args.frames,
                        "episode_elapsed_seconds": elapsed,
                        "episode_eta_seconds": eta_episode,
                        **partial,
                    }
                )
                print(
                    f"[TRAIN] ep={episode + 1:04d}/{args.episodes:04d} "
                    f"frame={processed_frames:04d}/{args.frames:04d} "
                    f"dpp={partial['dpp_cost_per_user_slot']:.6f} "
                    f"stall={partial['stall_ratio']:.4f} "
                    f"delivery={partial['delivered_chunks_per_user_slot']:.4f} "
                    f"quality={partial['average_quality_utility']:.4f} "
                    f"hire={partial['hire_rate']:.4f} "
                    f"soc={partial['min_battery_soc']:.4f} "
                    f"charge-risk={partial['precharge_reserve_breach_rate']:.4f} "
                    f"elapsed={format_duration(elapsed)} "
                    f"eta-ep={format_duration(eta_episode)}",
                    flush=True,
                )

            episode_summary, transitions = train_episode(
                agent,
                env_cfg,
                progress_interval_frames=args.progress_interval_frames,
                progress_callback=report_frame,
            )
            check_finite_metrics(episode_summary, f"training episode {episode}")
            hard_violations = check_hard_constraints(episode_summary)
            if hard_violations and args.fail_on_hard_violation:
                raise RuntimeError(
                    f"hard constraint violation in episode {episode}: "
                    f"battery={episode_summary['battery_reserve_violations']}, "
                    f"power={episode_summary['power_violations']}, "
                    f"provider={episode_summary['provider_violations']}"
                )

            update_started = time.perf_counter()
            update_summary = agent.update(transitions)
            check_finite_metrics(update_summary, f"PPO update {episode}")
            update_seconds = time.perf_counter() - update_started
            row = {
                "episode": episode,
                "environment_seed": env_cfg.seed,
                **episode_summary,
                **update_summary,
                "target_kl": agent_cfg.ppo_target_kl,
                **empty_validation_summary(),
            }

            should_validate = (
                (episode + 1) % args.validation_interval == 0
                or episode == args.episodes - 1
            )
            improved = False
            if should_validate:
                validation = validate_agent(
                    agent,
                    agent_cfg,
                    validation_seeds,
                    episode,
                    output_dir,
                    logger,
                    validation_baseline_cache,
                )
                check_finite_metrics(validation, f"validation episode {episode}")
                row.update(validation)
                validation_dpp = float(row["validation_dpp_per_user_slot"])
                dpp_improved = validation_dpp < best_validation_dpp - args.min_delta
                if dpp_improved:
                    best_validation_dpp = validation_dpp
                    agent.save(
                        output_dir / "best_dpp.pt",
                        metadata={
                            "episode": episode,
                            "selector": "minimum_validation_dpp",
                            "validation_seeds": validation_seeds,
                            "validation_summary": validation,
                        },
                    )
                validation_stall = float(row["validation_stall_ratio"])
                validation_original = float(
                    row["validation_original_cost_per_user_slot"]
                )
                validation_deployable = (
                    int(row["validation_action_collapse"]) == 0
                    and float(row["validation_stall_improvement_vs_rsu"]) >= 0.0
                    and float(row["validation_large_queue_violation_rate"]) == 0.0
                    and int(row["validation_battery_reserve_violations"]) == 0
                    and int(row["validation_power_violations"]) == 0
                    and int(row["validation_provider_violations"]) == 0
                )
                if validation_deployable:
                    improved = (
                        not best_validation_deployable
                        or validation_original
                        < best_validation_original_cost - args.min_delta
                        or (
                            abs(
                                validation_original
                                - best_validation_original_cost
                            )
                            <= args.min_delta
                            and validation_dpp < best_selected_validation_dpp
                        )
                    )
                else:
                    improved = (
                        not best_validation_deployable
                        and (
                            validation_stall
                            < best_validation_stall - args.min_delta
                            or (
                                abs(validation_stall - best_validation_stall)
                                <= args.min_delta
                                and validation_original
                                < best_validation_original_cost
                            )
                        )
                    )
                if improved:
                    best_validation_stall = validation_stall
                    best_validation_original_cost = validation_original
                    best_selected_validation_dpp = validation_dpp
                    best_validation_deployable = bool(validation_deployable)
                    best_validation_episode = episode
                    validations_without_improvement = 0
                    agent.save(
                        output_dir / "best.pt",
                        metadata={
                            "episode": episode,
                            "selector": (
                                "minimum_original_cost_with_stability_guardrails"
                                if validation_deployable
                                else "provisional_minimum_validation_stall"
                            ),
                            "deployable": bool(validation_deployable),
                            "validation_seeds": validation_seeds,
                            "validation_dpp_per_user_slot": validation_dpp,
                            "validation_dpp_std": row["validation_dpp_std"],
                            "validation_summary": validation,
                        },
                    )
                else:
                    validations_without_improvement += 1
                print(
                    f"[VAL] ep={episode + 1:04d} "
                    f"dpp={validation_dpp:.6f}±{row['validation_dpp_std']:.6f} "
                    f"stall={validation_stall:.4f} "
                    f"vs-rsu={row['validation_stall_improvement_vs_rsu']:+.4f} "
                    f"hire={row['validation_hire_rate']:.4f} "
                    f"collapse={int(row['validation_action_collapse'])} "
                    f"deployable={int(validation_deployable)} "
                    f"best-stall={best_validation_stall:.4f}@"
                    f"{best_validation_episode + 1:04d} "
                    f"bad-validations={validations_without_improvement}/"
                    f"{args.early_stopping_patience} improved={int(improved)}",
                    flush=True,
                )

            candidate_rows = [*rows, row]
            row["rolling_dpp_mean"] = rolling_mean(
                candidate_rows, "dpp_cost_per_user_slot", args.rolling_window
            )
            row["rolling_stall_mean"] = rolling_mean(
                candidate_rows, "stall_ratio", args.rolling_window
            )
            row["rolling_delivery_mean"] = rolling_mean(
                candidate_rows,
                "delivered_chunks_per_user_slot",
                args.rolling_window,
            )
            row["rolling_quality_mean"] = rolling_mean(
                candidate_rows, "average_quality_utility", args.rolling_window
            )
            row["rolling_hire_mean"] = rolling_mean(
                candidate_rows, "hire_rate", args.rolling_window
            )
            episode_seconds = time.perf_counter() - episode_started
            elapsed_seconds = time.perf_counter() - started
            average_episode_seconds = elapsed_seconds / (episode + 1)
            remaining_episodes = args.episodes - episode - 1
            row["update_seconds"] = update_seconds
            row["episode_seconds"] = episode_seconds
            row["elapsed_seconds"] = elapsed_seconds
            row["eta_seconds"] = average_episode_seconds * remaining_episodes
            row["best_validation_dpp_per_user_slot"] = (
                best_validation_dpp
                if math.isfinite(best_validation_dpp)
                else math.nan
            )
            row["best_validation_episode"] = best_validation_episode
            row["best_validation_stall_ratio"] = (
                best_validation_stall
                if math.isfinite(best_validation_stall)
                else math.nan
            )
            row["validations_without_improvement"] = validations_without_improvement
            rows.append(row)

            checkpoint_metadata = {
                "episode": episode,
                "completed_episodes": episode + 1,
                "best_validation_episode": best_validation_episode,
                "best_validation_dpp_per_user_slot": (
                    best_validation_dpp
                    if math.isfinite(best_validation_dpp)
                    else None
                ),
                "selected_validation_dpp_per_user_slot": (
                    best_selected_validation_dpp
                    if math.isfinite(best_selected_validation_dpp)
                    else None
                ),
                "best_validation_stall_ratio": (
                    best_validation_stall
                    if math.isfinite(best_validation_stall)
                    else None
                ),
                "best_validation_original_cost_per_user_slot": (
                    best_validation_original_cost
                    if math.isfinite(best_validation_original_cost)
                    else None
                ),
                "best_validation_deployable": best_validation_deployable,
                "validations_without_improvement": validations_without_improvement,
                "elapsed_seconds": elapsed_seconds,
            }
            agent.save(output_dir / "latest.pt", metadata=checkpoint_metadata)
            if (episode + 1) % args.checkpoint_interval == 0:
                agent.save(
                    output_dir
                    / "checkpoints"
                    / f"episode_{episode + 1:04d}.pt",
                    metadata=checkpoint_metadata,
                )

            atomic_write_csv(output_dir / "training_curve.csv", rows)
            if (
                (episode + 1) % args.plot_interval == 0
                or should_validate
                or episode == args.episodes - 1
            ):
                plot_training_curve(rows, output_dir / "training_curve.png")

            print(
                f"[EP-END] ep={episode + 1:04d}/{args.episodes:04d} "
                f"dpp={row['dpp_cost_per_user_slot']:.6f} "
                f"roll-dpp={row['rolling_dpp_mean']:.6f} "
                f"stall={row['stall_ratio']:.4f} "
                f"delivery={row['delivered_chunks_per_user_slot']:.4f} "
                f"quality={row['average_quality_utility']:.4f} "
                f"q05={row['quality_p05_utility']:.4f} "
                f"hire={row['hire_rate']:.4f} "
                f"entropy={row['normalized_entropy']:.4f} "
                f"kl={row['approx_kl']:.5f} "
                f"epoch-time={format_duration(episode_seconds)} "
                f"eta-run={format_duration(row['eta_seconds'])}",
                flush=True,
            )
            logger.emit({"event": "episode_end", **row}, echo_json=True)

            should_stop = (
                should_validate
                and args.early_stopping_patience > 0
                and validations_without_improvement
                >= args.early_stopping_patience
            )
            if should_stop:
                termination_reason = "early_stopping"
                print(
                    f"[EARLY-STOP] no validation improvement larger than "
                    f"{args.min_delta:g} for "
                    f"{validations_without_improvement} validations; "
                    f"best episode={best_validation_episode + 1}, "
                    f"best stall={best_validation_stall:.6f}",
                    flush=True,
                )
                break
    except BaseException as error:
        termination_reason = "failed"
        logger.emit(
            {
                "event": "run_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_episodes": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
            },
            echo_json=True,
        )
        raise
    finally:
        if rows:
            atomic_write_csv(output_dir / "training_curve.csv", rows)
            plot_training_curve(rows, output_dir / "training_curve.png")

    total_seconds = time.perf_counter() - started
    final_summary = {
        "termination_reason": termination_reason,
        "requested_episodes": args.episodes,
        "completed_episodes": len(rows),
        "best_validation_episode": best_validation_episode,
        "best_validation_dpp_per_user_slot": (
            best_validation_dpp if math.isfinite(best_validation_dpp) else None
        ),
        "selected_validation_dpp_per_user_slot": (
            best_selected_validation_dpp
            if math.isfinite(best_selected_validation_dpp)
            else None
        ),
        "best_validation_stall_ratio": (
            best_validation_stall
            if math.isfinite(best_validation_stall)
            else None
        ),
        "best_validation_original_cost_per_user_slot": (
            best_validation_original_cost
            if math.isfinite(best_validation_original_cost)
            else None
        ),
        "best_validation_deployable": best_validation_deployable,
        "validation_seeds": validation_seeds,
        "runtime_seconds": total_seconds,
        "output_dir": str(output_dir),
    }
    atomic_write_json(output_dir / "training_summary.json", final_summary)
    logger.emit({"event": "run_end", **final_summary}, echo_json=True)
    print(
        f"[DONE] reason={termination_reason} completed={len(rows)}/{args.episodes} "
        f"best-stall={best_validation_stall:.6f}@"
        f"{best_validation_episode + 1:04d} "
        f"best-dpp={best_validation_dpp:.6f} "
        f"runtime={format_duration(total_seconds)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
