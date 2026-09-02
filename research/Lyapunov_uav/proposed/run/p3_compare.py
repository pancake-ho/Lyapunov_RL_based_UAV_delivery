from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from agent.P3.slow_rollout_controller import SUPPORTED_RULE_POLICIES
from config_p3 import P3Config
from run.p3_common import PolicyRunResult, run_policy, safe_ratio, write_csv


POLICY_LABELS = {
    "dpp": "Structured rollout",
    "ppo": "PPO + exact fast",
    "always_hire": "Always hire",
    "fixed_rsu": "Fixed at RSU",
    "nearest_hotspot": "Nearest hotspot",
    "load_threshold": "Load threshold",
    "rsu_only": "RSU only",
    "random": "Random feasible",
}
POLICY_COLORS = {
    "dpp": "#4472C4",
    "ppo": "#7030A0",
    "always_hire": "#ED7D31",
    "fixed_rsu": "#70AD47",
    "nearest_hotspot": "#5B9BD5",
    "load_threshold": "#FFC000",
    "rsu_only": "#A5A5A5",
    "random": "#C55A11",
}


def parse_csv_list(text: str, cast) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size <= 1:
        return mean, 0.0
    return mean, 1.96 * float(np.std(array, ddof=1)) / math.sqrt(array.size)


def aggregate_summaries(summaries: Sequence[dict], policies: Sequence[str]) -> list[dict]:
    metrics = (
        "stall_ratio",
        "served_user_ratio",
        "hire_rate",
        "charging_fraction",
        "delivered_chunks_per_user_slot",
        "average_quality_utility",
        "average_quality_level",
        "quality_p05_utility",
        "quality_switch_rate",
        "degradation_per_chunk",
        "mean_queue",
        "large_queue_violation_rate",
        "original_cost_per_user_slot",
        "dpp_cost_per_user_slot",
        "energy_per_delivered_chunk_j",
        "min_final_battery_soc",
        "precharge_depletion_rate",
        "precharge_reserve_breach_rate",
        "stranded_before_charge_rate",
        "mean_uav_user_distance_m",
        "peak_uav_total_power_w",
        "runtime_seconds",
    )
    result: list[dict] = []
    for policy in policies:
        rows = [row for row in summaries if row["policy"] == policy]
        item: dict[str, float | int | str] = {
            "policy": policy,
            "num_seeds": len(rows),
        }
        for metric in metrics:
            mean, ci = mean_ci95([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci
        for name in (
            "return_to_charge_events",
            "charging_need_events",
            "stranded_before_charge_events",
            "precharge_depletion_events",
            "precharge_reserve_breach_events",
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        ):
            item[f"{name}_total"] = sum(int(row[name]) for row in rows)
        # Event-weighted rates are the authoritative charging-risk statistics.
        returns = int(item["return_to_charge_events_total"])
        item["precharge_depletion_rate_pooled"] = safe_ratio(
            int(item["precharge_depletion_events_total"]), returns
        )
        item["precharge_reserve_breach_rate_pooled"] = safe_ratio(
            int(item["precharge_reserve_breach_events_total"]), returns
        )
        item["stranded_before_charge_rate_pooled"] = safe_ratio(
            int(item["stranded_before_charge_events_total"]),
            int(item["charging_need_events_total"]),
        )
        result.append(item)
    return result


def aggregate_distance(rows: Sequence[dict], policies: Sequence[str]) -> list[dict]:
    output = []
    bin_indices = sorted({int(row["bin_index"]) for row in rows})
    for policy in policies:
        for bin_index in bin_indices:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and int(row["bin_index"]) == bin_index
            ]
            opportunities = sum(float(row["opportunities"]) for row in selected)
            served = sum(float(row["served_slots"]) for row in selected)
            stalled = sum(float(row["stall_slots"]) for row in selected)
            delivered = sum(float(row["delivered_chunks"]) for row in selected)
            utility = sum(float(row["quality_utility_sum"]) for row in selected)
            power = sum(float(row["power_sum_w"]) for row in selected)
            example = selected[0]
            output.append(
                {
                    "policy": policy,
                    "bin_index": bin_index,
                    "distance_left_m": example["distance_left_m"],
                    "distance_right_m": example["distance_right_m"],
                    "opportunities": opportunities,
                    "uav_service_ratio": safe_ratio(served, opportunities),
                    "stall_ratio": safe_ratio(stalled, opportunities),
                    "average_quality_utility": safe_ratio(utility, delivered),
                    "average_required_power_w": safe_ratio(power, served),
                }
            )
    return output


def aggregate_points(rows: Sequence[dict], policies: Sequence[str]) -> list[dict]:
    output = []
    point_indices = sorted({int(row["point_index"]) for row in rows})
    for policy in policies:
        for point_index in point_indices:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and int(row["point_index"]) == point_index
            ]
            count = sum(int(row["selection_count"]) for row in selected)
            hired = sum(int(row["hired_uav_frames"]) for row in selected)
            output.append(
                {
                    "policy": policy,
                    "point_index": point_index,
                    "offset_m": selected[0]["offset_m"],
                    "selection_count": count,
                    "selection_rate_given_hired": safe_ratio(count, hired),
                }
            )
    return output


def aggregate_quality(rows: Sequence[dict], policies: Sequence[str]) -> list[dict]:
    output = []
    quality_indices = sorted({int(row["quality_index"]) for row in rows})
    for policy in policies:
        selected_policy = [row for row in rows if row["policy"] == policy]
        total = sum(float(row["delivered_chunks"]) for row in selected_policy)
        for quality_index in quality_indices:
            selected = [
                row for row in selected_policy if int(row["quality_index"]) == quality_index
            ]
            chunks = sum(float(row["delivered_chunks"]) for row in selected)
            output.append(
                {
                    "policy": policy,
                    "quality_index": quality_index,
                    "quality_level": selected[0]["quality_level"],
                    "quality_utility": selected[0]["quality_utility"],
                    "chunk_size_bits": selected[0]["chunk_size_bits"],
                    "delivered_chunks": chunks,
                    "delivered_chunk_share": safe_ratio(chunks, total),
                }
            )
    return output


def _setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    return plt


def plot_overview(aggregate: Sequence[dict], output_path: Path) -> None:
    plt = _setup_matplotlib()
    policies = [str(row["policy"]) for row in aggregate]
    labels = [POLICY_LABELS.get(policy, policy) for policy in policies]
    colors = [POLICY_COLORS.get(policy, "#4472C4") for policy in policies]
    x = np.arange(len(policies))
    panels = (
        ("stall_ratio", "Stall ratio", "lower is better"),
        ("average_quality_utility", "Average delivered quality utility", "higher is better"),
        ("quality_p05_utility", "5th-percentile quality utility", "higher is better"),
        ("hire_rate", "UAV hire rate", "hired UAV-frames / total"),
        ("charging_fraction", "Charging-mode fraction", "unhired UAV-slots / total"),
        ("stranded_before_charge_rate", "Depletion before charger", "stranded / charging-need events"),
        ("min_final_battery_soc", "Minimum final battery SoC", "fraction"),
        ("mean_uav_user_distance_m", "Mean UAV-user distance", "metres"),
        ("original_cost_per_user_slot", "Original objective cost", "per user-slot"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), constrained_layout=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        means = [float(row[f"{metric}_mean"]) for row in aggregate]
        errors = [float(row[f"{metric}_ci95"]) for row in aggregate]
        axis.bar(x, means, yerr=errors, color=colors, capsize=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
        if metric == "stranded_before_charge_rate":
            for index, row in enumerate(aggregate):
                axis.text(
                    index,
                    means[index] + errors[index] + 0.005,
                    f"needs={int(row['charging_need_events_total'])}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            axis.set_ylim(bottom=0.0)
    fig.suptitle("P3 Formulation Metrics (mean +/- 95% CI)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_location_distance(
    point_rows: Sequence[dict],
    distance_rows: Sequence[dict],
    policies: Sequence[str],
    output_path: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    offsets = sorted({float(row["offset_m"]) for row in point_rows})
    width = 0.8 / max(len(policies), 1)
    x = np.arange(len(offsets))
    for policy_index, policy in enumerate(policies):
        selected = [row for row in point_rows if row["policy"] == policy]
        selected.sort(key=lambda row: float(row["offset_m"]))
        axes[0, 0].bar(
            x + (policy_index - (len(policies) - 1) / 2) * width,
            [float(row["selection_rate_given_hired"]) for row in selected],
            width=width,
            label=POLICY_LABELS.get(policy, policy),
            color=POLICY_COLORS.get(policy),
        )
    axes[0, 0].set_title("Hover-point selection rate given hire")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"{offset:g} m" for offset in offsets])
    axes[0, 0].set_ylabel("selection fraction")
    axes[0, 0].legend(fontsize=7)

    distance_metrics = (
        ("uav_service_ratio", "UAV service ratio by distance", "served / scheduled user-slots"),
        ("average_quality_utility", "Delivered quality by distance", "average utility"),
        ("average_required_power_w", "Required RF power by distance", "W per served user-slot"),
    )
    for axis, (metric, title, ylabel) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]), distance_metrics
    ):
        for policy in policies:
            selected = [row for row in distance_rows if row["policy"] == policy]
            selected.sort(key=lambda row: int(row["bin_index"]))
            if not any(float(row["opportunities"]) > 0.0 for row in selected):
                continue
            labels = []
            for row in selected:
                left = float(row["distance_left_m"])
                right = float(row["distance_right_m"])
                right_text = "inf" if math.isinf(right) else f"{right:g}"
                labels.append(f"{left:g}-{right_text}")
            values = [
                float(row[metric]) if float(row["opportunities"]) > 0.0 else math.nan
                for row in selected
            ]
            axis.plot(
                np.arange(len(selected)),
                values,
                marker="o",
                label=POLICY_LABELS.get(policy, policy),
                color=POLICY_COLORS.get(policy),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("horizontal distance bin (m)")
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axis.legend(fontsize=7)
    fig.suptitle("UAV Position and Distance Effects")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories(
    frame_rows: dict[tuple[str, int], list[dict]],
    policies: Sequence[str],
    first_seed: int,
    output_path: Path,
) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    metrics = (
        ("min_battery_soc", "Minimum UAV SoC", "SoC fraction"),
        ("average_quality_utility", "Delivered quality", "average utility"),
        ("charging_fraction", "Charging cycle", "charging-mode fraction"),
    )
    for axis, (metric, title, ylabel) in zip(axes, metrics):
        for policy in policies:
            rows = frame_rows[(policy, first_seed)]
            axis.plot(
                [int(row["frame"]) for row in rows],
                [float(row[metric]) for row in rows],
                label=POLICY_LABELS.get(policy, policy),
                color=POLICY_COLORS.get(policy),
            )
        axis.set_title(title)
        axis.set_xlabel("frame")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=7)
    fig.suptitle(f"Battery-Quality-Charging Trajectories (seed {first_seed})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 policy comparison and requested metrics")
    parser.add_argument(
        "--policies",
        default="dpp,rsu_only,always_hire,fixed_rsu,nearest_hotspot",
    )
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/p3_compare"))
    args = parser.parse_args()

    policies = parse_csv_list(args.policies, str)
    seeds = parse_csv_list(args.seeds, int)
    allowed = set(SUPPORTED_RULE_POLICIES) | {"ppo"}
    unknown = sorted(set(policies) - allowed)
    if unknown:
        parser.error(f"unknown policies: {unknown}")
    if "ppo" in policies and args.ppo_checkpoint is None:
        parser.error("--ppo-checkpoint is required when policy list includes ppo")
    if not policies or not seeds:
        parser.error("at least one policy and seed are required")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = P3Config(
        num_frames=int(args.frames),
        rollout_scenarios=int(args.rollouts),
    )
    summaries: list[dict] = []
    all_distance: list[dict] = []
    all_points: list[dict] = []
    all_quality: list[dict] = []
    frame_map: dict[tuple[str, int], list[dict]] = {}

    for seed in seeds:
        cfg = replace(base_cfg, seed=int(seed))
        ppo_agent = None
        if "ppo" in policies:
            from agent.P3.ppo_agent import PPOAgent

            ppo_agent = PPOAgent(cfg, device=args.device)
            ppo_agent.load(args.ppo_checkpoint)
        for policy in policies:
            result: PolicyRunResult = run_policy(
                cfg,
                policy,
                output_dir,
                ppo_agent=ppo_agent,
            )
            summaries.append(result.summary)
            all_distance.extend(result.distance_rows)
            all_points.extend(result.point_rows)
            all_quality.extend(result.quality_rows)
            frame_map[(policy, seed)] = result.frame_rows
            print(json.dumps(result.summary, ensure_ascii=False, sort_keys=True))

    aggregate = aggregate_summaries(summaries, policies)
    distance_aggregate = aggregate_distance(all_distance, policies)
    point_aggregate = aggregate_points(all_points, policies)
    quality_aggregate = aggregate_quality(all_quality, policies)
    write_csv(output_dir / "seed_summaries.csv", summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "distance_by_policy.csv", distance_aggregate)
    write_csv(output_dir / "hover_point_by_policy.csv", point_aggregate)
    write_csv(output_dir / "quality_distribution.csv", quality_aggregate)
    plot_overview(aggregate, output_dir / "p3_overview.png")
    plot_location_distance(
        point_aggregate,
        distance_aggregate,
        policies,
        output_dir / "p3_location_distance.png",
    )
    plot_trajectories(
        frame_map,
        policies,
        seeds[0],
        output_dir / "p3_battery_quality_trajectory.png",
    )
    with (output_dir / "experiment.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "warning": "Smoke defaults are not calibrated paper results.",
                "formulation": "P3: one persistent multi-user UAV per RSU region",
                "config": asdict(base_cfg),
                "aggregate": aggregate,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
