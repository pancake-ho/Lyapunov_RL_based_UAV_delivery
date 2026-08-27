from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from agent.P1.slow_rollout_controller import (
    SUPPORTED_POLICIES,
    SlowRolloutController,
)
from config_p1 import P1Config
from env.p1.environment import (
    advance_mobility_one_frame,
    apply_region_result,
    generate_frame_trace,
    simulate_region_frame,
)
from env.p1.topology import (
    initialize_state,
    region_membership,
    validate_state,
)
from env.p1.types import RegionAction


POLICY_LABELS = {
    "dpp": "Proposed\n(Adaptive Hire & Placement)",
    "always_hire": "Always Hire\n(Adaptive Placement)",
    "fixed_rsu": "Always Hire\n(Fixed at RSU)",
    "rsu_only": "RSU Only\n(No UAV)",
}

PLOT_POLICY_ORDER = (
    "dpp",
    "always_hire",
    "fixed_rsu",
    "rsu_only",
)

POLICY_COLORS = {
    "dpp": "#4472C4",
    "always_hire": "#ED7D31",
    "fixed_rsu": "#70AD47",
    "rsu_only": "#A5A5A5",
}


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def action_record(action: RegionAction, cfg: P1Config) -> dict:
    return {
        "region": action.region,
        "mu": action.hired,
        "point_index": action.point_index,
        "target_x": action.target_x(cfg),
        "rsu_users": list(action.rsu_users),
        "uav_user": action.uav_user,
    }


def run_policy(
    cfg: P1Config,
    policy: str,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    state = initialize_state(cfg)
    controller = SlowRolloutController(cfg)
    rows: list[dict] = []
    totals: dict[str, float] = {
        "delivered_chunks": 0.0,
        "quality_utility": 0.0,
        "degradation": 0.0,
        "stall_user_slots": 0.0,
        "served_user_slots": 0.0,
        "large_queue_violation_user_slots": 0.0,
        "queue_sum": 0.0,
        "energy_consumed_j": 0.0,
        "energy_charged_j": 0.0,
        "hired_uav_frames": 0.0,
        "relocation_events": 0.0,
        "battery_reserve_violations": 0.0,
        "power_violations": 0.0,
        "provider_violations": 0.0,
        "original_cost": 0.0,
        "dpp_cost": 0.0,
    }
    started = time.perf_counter()

    for frame in range(cfg.num_frames):
        membership = region_membership(state, cfg)
        actions: list[RegionAction] = []
        predicted_scores: list[float] = []
        candidate_counts: list[int] = []
        selection_started = time.perf_counter()
        for region in range(cfg.num_regions):
            users = np.flatnonzero(membership == region).tolist()
            selection = controller.select(
                state,
                region,
                users,
                frame,
                policy,
            )
            actions.append(selection.action)
            predicted_scores.append(selection.estimated_dpp_cost)
            candidate_counts.append(selection.candidate_count)
        selection_seconds = time.perf_counter() - selection_started

        # A separate realization prevents rollout look-ahead leakage.
        realized_trace = generate_frame_trace(
            cfg,
            controller.realized_seed(frame),
        )
        frame_float = {
            key: 0.0
            for key in (
                "delivered_chunks",
                "quality_utility",
                "degradation",
                "queue_sum",
                "energy_consumed_j",
                "energy_charged_j",
                "original_cost",
                "dpp_cost",
            )
        }
        frame_int = {
            key: 0
            for key in (
                "stall_user_slots",
                "served_user_slots",
                "large_queue_violation_user_slots",
                "hired_uav_frames",
                "relocation_events",
                "battery_reserve_violations",
                "power_violations",
                "provider_violations",
            )
        }

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
            frame_float["degradation"] += result.degradation
            frame_float["queue_sum"] += result.queue_sum
            frame_float["energy_consumed_j"] += result.energy_consumed_j
            frame_float["energy_charged_j"] += result.energy_charged_j
            frame_float["original_cost"] += result.original_cost
            frame_float["dpp_cost"] += result.frame_dpp_cost
            frame_int["stall_user_slots"] += result.stall_user_slots
            frame_int["served_user_slots"] += result.served_user_slots
            frame_int["large_queue_violation_user_slots"] += (
                result.large_queue_violation_user_slots
            )
            frame_int["hired_uav_frames"] += int(action.hired)
            frame_int["relocation_events"] += result.relocation_events
            frame_int["battery_reserve_violations"] += (
                result.battery_reserve_violations
            )
            frame_int["power_violations"] += result.power_violations
            frame_int["provider_violations"] += result.provider_violations

        advance_mobility_one_frame(state, cfg)
        validate_state(state, cfg)
        user_slots = cfg.num_users * cfg.frame_slots
        rows.append(
            {
                "seed": cfg.seed,
                "policy": policy,
                "frame": frame,
                "actions_json": json.dumps(
                    [action_record(action, cfg) for action in actions],
                    separators=(",", ":"),
                ),
                "hired_uavs": frame_int["hired_uav_frames"],
                "stall_ratio": frame_int["stall_user_slots"] / user_slots,
                "served_user_ratio": (
                    frame_int["served_user_slots"] / user_slots
                ),
                "delivered_chunks": frame_float["delivered_chunks"],
                "degradation_per_chunk": (
                    frame_float["degradation"]
                    / frame_float["delivered_chunks"]
                    if frame_float["delivered_chunks"] > 0.0
                    else 0.0
                ),
                "mean_queue": frame_float["queue_sum"] / user_slots,
                "max_queue_end": float(np.max(state.queue)),
                "large_queue_violation_rate": (
                    frame_int["large_queue_violation_user_slots"]
                    / user_slots
                ),
                "min_battery_soc": float(
                    np.min(state.battery_j) / cfg.battery_capacity_j
                ),
                "mean_battery_soc": float(
                    np.mean(state.battery_j) / cfg.battery_capacity_j
                ),
                "energy_consumed_j": frame_float["energy_consumed_j"],
                "energy_charged_j": frame_float["energy_charged_j"],
                "original_cost": frame_float["original_cost"],
                "dpp_cost": frame_float["dpp_cost"],
                "rollout_predicted_dpp": float(sum(predicted_scores)),
                "candidate_count": int(sum(candidate_counts)),
                "selection_seconds": float(selection_seconds),
                "battery_reserve_violations": frame_int[
                    "battery_reserve_violations"
                ],
                "power_violations": frame_int["power_violations"],
                "provider_violations": frame_int["provider_violations"],
            }
        )
        for key, value in frame_float.items():
            totals[key] += value
        for key, value in frame_int.items():
            totals[key] += value

    runtime = time.perf_counter() - started
    total_user_slots = cfg.num_frames * cfg.num_users * cfg.frame_slots
    total_uav_frames = cfg.num_frames * cfg.num_regions
    summary = {
        "seed": cfg.seed,
        "policy": policy,
        "frames": cfg.num_frames,
        "user_slots": total_user_slots,
        "stall_ratio": totals["stall_user_slots"] / total_user_slots,
        "served_user_ratio": totals["served_user_slots"] / total_user_slots,
        "hire_rate": totals["hired_uav_frames"] / total_uav_frames,
        "delivered_chunks_per_user_slot": (
            totals["delivered_chunks"] / total_user_slots
        ),
        "degradation_per_chunk": (
            totals["degradation"] / totals["delivered_chunks"]
            if totals["delivered_chunks"] > 0.0
            else 0.0
        ),
        "mean_queue": totals["queue_sum"] / total_user_slots,
        "large_queue_violation_rate": (
            totals["large_queue_violation_user_slots"] / total_user_slots
        ),
        "original_cost_per_user_slot": (
            totals["original_cost"] / total_user_slots
        ),
        "energy_per_delivered_chunk_j": (
            totals["energy_consumed_j"] / totals["delivered_chunks"]
            if totals["delivered_chunks"] > 0.0
            else 0.0
        ),
        "min_final_battery_soc": float(
            np.min(state.battery_j) / cfg.battery_capacity_j
        ),
        "battery_reserve_violations": int(
            totals["battery_reserve_violations"]
        ),
        "power_violations": int(totals["power_violations"]),
        "provider_violations": int(totals["provider_violations"]),
        "relocation_events": int(totals["relocation_events"]),
        "runtime_seconds": float(runtime),
    }
    write_csv(
        output_dir / f"frames_{policy}_seed{cfg.seed}.csv",
        rows,
    )
    return summary, rows


def mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size <= 1:
        return mean, 0.0
    ci95 = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(array.size)
    return mean, ci95


def aggregate_summaries(summaries: Sequence[dict]) -> list[dict]:
    metrics = (
        "stall_ratio",
        "served_user_ratio",
        "hire_rate",
        "delivered_chunks_per_user_slot",
        "degradation_per_chunk",
        "mean_queue",
        "large_queue_violation_rate",
        "original_cost_per_user_slot",
        "energy_per_delivered_chunk_j",
        "min_final_battery_soc",
        "runtime_seconds",
    )
    aggregate: list[dict] = []
    for policy in sorted({str(row["policy"]) for row in summaries}):
        rows = [row for row in summaries if row["policy"] == policy]
        item: dict[str, object] = {"policy": policy, "num_seeds": len(rows)}
        for metric in metrics:
            mean, ci95 = mean_ci95([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci95
        for violation in (
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        ):
            item[f"{violation}_total"] = sum(
                int(row[violation]) for row in rows
            )
        aggregate.append(item)
    return aggregate


def create_plot(
    cfg: P1Config,
    aggregate: Sequence[dict],
    frame_rows: dict[tuple[str, int], list[dict]],
    first_seed: int,
    output_path: Path,
) -> None:
    import os
    import tempfile

    matplotlib_cache = Path(tempfile.gettempdir()) / "p1-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib.pyplot as plt

    # Proposed method가 먼저 나오도록 plot 순서 고정
    aggregate_by_policy = {
        str(row["policy"]): row
        for row in aggregate
    }

    policy_ids = [
        policy
        for policy in PLOT_POLICY_ORDER
        if policy in aggregate_by_policy
    ]

    plot_rows = [
        aggregate_by_policy[policy]
        for policy in policy_ids
    ]

    display_labels = [
        POLICY_LABELS[policy]
        for policy in policy_ids
    ]

    colors = [
        POLICY_COLORS[policy]
        for policy in policy_ids
    ]

    x = np.arange(len(policy_ids))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 8),
        constrained_layout=True,
    )

    # Stall ratio
    axes[0, 0].bar(
        x,
        [float(row["stall_ratio_mean"]) for row in plot_rows],
        yerr=[
            float(row["stall_ratio_ci95"])
            for row in plot_rows
        ],
        color=colors,
        capsize=4,
    )
    axes[0, 0].set_title("Stall Ratio (Mean ± 95% CI)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(display_labels, fontsize=8)
    axes[0, 0].set_ylabel("Stalled User-Slots / Total User-Slots")

    # UAV hire rate
    axes[0, 1].bar(
        x,
        [float(row["hire_rate_mean"]) for row in plot_rows],
        yerr=[
            float(row["hire_rate_ci95"])
            for row in plot_rows
        ],
        color=colors,
        capsize=4,
    )
    axes[0, 1].set_title("UAV Hire Rate (Mean ± 95% CI)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(display_labels, fontsize=8)
    axes[0, 1].set_ylim(0.0, 1.05)

    # Seed별 trajectory
    for policy in policy_ids:
        color = POLICY_COLORS[policy]
        rows = frame_rows.get((policy, first_seed), [])
        label = POLICY_LABELS[policy].replace("\n", " ")

        axes[1, 0].plot(
            [int(row["frame"]) for row in rows],
            [float(row["min_battery_soc"]) for row in rows],
            label=label,
            color=color,
        )

        axes[1, 1].plot(
            [int(row["frame"]) for row in rows],
            [float(row["mean_queue"]) for row in rows],
            label=label,
            color=color,
        )

    # Battery reserve
    reserve_soc = (
        cfg.reserve_battery_j
        / cfg.battery_capacity_j
    )

    axes[1, 0].axhline(
        reserve_soc,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Battery Reserve",
    )
    axes[1, 0].set_title(
        f"Minimum UAV SoC (Example Seed: {first_seed})"
    )
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("SoC Fraction")
    axes[1, 0].legend(fontsize=7)

    # Playback queue
    axes[1, 1].set_title(
        f"Mean Playback Queue (Example Seed: {first_seed})"
    )
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Queue Length (Chunks)")
    axes[1, 1].legend(fontsize=7)

    fig.suptitle(
        "P1 Policy Comparison: "
        "Stall–Hiring–Battery Trade-off"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=180,
    )
    plt.close(fig)


def parse_csv_list(text: str, cast) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1 exact-fast modular lab-meeting smoke baseline",
    )
    parser.add_argument(
        "--policies",
        default=",".join(SUPPORTED_POLICIES),
    )
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--output", default="outputs/p1_modular")
    args = parser.parse_args()

    policies = parse_csv_list(args.policies, str)
    seeds = parse_csv_list(args.seeds, int)
    if not policies or not seeds:
        parser.error("at least one policy and seed are required")
    unknown = sorted(set(policies) - set(SUPPORTED_POLICIES))
    if unknown:
        parser.error(f"unknown policies: {unknown}")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = P1Config(
        num_frames=int(args.frames),
        rollout_scenarios=int(args.rollouts),
    )
    summaries: list[dict] = []
    frame_rows: dict[tuple[str, int], list[dict]] = {}
    for seed in seeds:
        for policy in policies:
            cfg = replace(base_cfg, seed=int(seed))
            summary, rows = run_policy(cfg, policy, output_dir)
            summaries.append(summary)
            frame_rows[(policy, seed)] = rows
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))

    aggregate = aggregate_summaries(summaries)
    write_csv(output_dir / "seed_summaries.csv", summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    with (output_dir / "experiment.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "warning": "Uncalibrated smoke profile; not a paper result.",
                "config": asdict(base_cfg),
                "summaries": summaries,
                "aggregate": aggregate,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    create_plot(
        base_cfg,
        aggregate,
        frame_rows,
        first_seed=seeds[0],
        output_path=output_dir / "p1_labmeeting.png",
    )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
