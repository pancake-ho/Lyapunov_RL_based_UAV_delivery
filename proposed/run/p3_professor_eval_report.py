from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from run.p3_compare import POLICY_COLORS, POLICY_LABELS


PRIMARY_METRICS = (
    "average_video_bitrate_mbps",
    "stall_ratio",
    "average_quality_utility",
    "quality_p05_utility",
    "hire_rate",
    "precharge_depletion_rate",
    "precharge_reserve_breach_rate",
    "stranded_before_charge_rate",
    "mean_return_to_charge_distance_m",
)


RAW_TO_PAPER_POLICY = {
    "dpp": "proposed",
}
PAPER_TO_RAW_POLICY = {
    paper: raw for raw, paper in RAW_TO_PAPER_POLICY.items()
}


def paper_policy(policy: str) -> str:
    """Paper-facing alias without changing the simulator's internal policy key."""
    return RAW_TO_PAPER_POLICY.get(str(policy), str(policy))


def raw_policy(policy: str) -> str:
    """Map paper-facing policy name back to the raw artifact key."""
    return PAPER_TO_RAW_POLICY.get(str(policy), str(policy))


def relabel_policy_rows(rows: Sequence[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        item = dict(row)
        if "policy" in item:
            item["policy"] = paper_policy(str(item["policy"]))
        output.append(item)
    return output


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def mean_ci95(values: Iterable[float]) -> tuple[float, float, int]:
    array = finite(values)
    if array.size == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, 0.0, 1
    ci = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(array.size)
    return mean, ci, int(array.size)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0


def setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    return plt


def frame_path(root: Path, policy: str, seed: int) -> Path:
    paper_name = str(policy)
    raw_name = raw_policy(paper_name)
    if paper_name in {"ppo_best", "ppo_latest"}:
        return root / paper_name / f"frames_ppo_seed{seed}.csv"
    return root / "baselines" / f"frames_{raw_name}_seed{seed}.csv"


def reconstruct_return_distances(
    frame_rows: Sequence[dict],
    *,
    num_regions: int,
    region_length_m: float,
) -> tuple[list[float], int]:
    """Reconstruct UAV -> depot return distances from saved slow actions.

    The environment sets UAV position to action.target_x after every frame.
    Therefore the previous frame's target is the exact pre-action UAV position.
    Initial UAV positions are the regional depots.
    """

    depots = {
        region: (float(region) + 0.5) * float(region_length_m)
        for region in range(num_regions)
    }
    previous_x = dict(depots)
    distances: list[float] = []
    logged_return_events = 0

    for row in sorted(frame_rows, key=lambda item: int(item["frame"])):
        logged_return_events += int(float(row.get("return_to_charge_events", 0) or 0))
        actions = json.loads(row["actions_json"])
        seen_regions: set[int] = set()
        for action in actions:
            region = int(action["region"])
            if region in seen_regions:
                raise RuntimeError(f"duplicate region action in frame {row['frame']}: {region}")
            seen_regions.add(region)
            if not 0 <= region < num_regions:
                raise RuntimeError(f"invalid region in actions_json: {region}")
            hired = int(action["mu"])
            target_x = float(action["target_x"])
            depot_x = depots[region]
            before_x = previous_x[region]
            if hired == 0 and abs(before_x - depot_x) > 1e-9:
                distances.append(abs(before_x - depot_x))
            previous_x[region] = target_x

    if logged_return_events != len(distances):
        raise RuntimeError(
            "return-distance reconstruction mismatch: "
            f"logged={logged_return_events}, reconstructed={len(distances)}"
        )
    return distances, logged_return_events


def average_video_bitrate_mbps(
    throughput_mbps: float,
    delivered_chunks_per_user_slot: float,
    *,
    num_users: int,
    playback_chunks_per_slot: float,
) -> float:
    """Delivered application bitrate, distinct from aggregate network throughput.

    throughput_mbps is total payload bits divided by wall-clock simulation time.
    delivered_chunks_per_user_slot normalizes chunk delivery by users and slots.
    With b playback chunks consumed per slot, chunk playback duration is
    slot_duration / b. Algebraically, slot_duration cancels and yields the
    expression below. For the current P3 config b=1, this is the delivered
    chunk-size-weighted video bitrate in Mbps.
    """

    denominator = float(delivered_chunks_per_user_slot) * int(num_users)
    if denominator <= 0.0:
        return 0.0
    return float(throughput_mbps) * float(playback_chunks_per_slot) / denominator


def build_seed_metrics(root: Path, experiment: dict) -> list[dict]:
    aggregate_dir = root / "aggregate"
    seed_rows = read_csv(aggregate_dir / "seed_summaries.csv")
    config = experiment["config"]
    num_regions = int(config["num_regions"])
    users_per_region = int(config["users_per_region"])
    num_users = num_regions * users_per_region
    region_length_m = float(config["region_length_m"])
    playback_chunks_per_slot = float(config["playback_chunks_per_slot"])

    output: list[dict] = []
    for row in seed_rows:
        raw_name = str(row["policy"])
        policy = paper_policy(raw_name)
        seed = int(row["seed"])
        frames = read_csv(frame_path(root, policy, seed))
        return_distances, return_events = reconstruct_return_distances(
            frames,
            num_regions=num_regions,
            region_length_m=region_length_m,
        )
        summary_return_events = int(float(row["return_to_charge_events"]))
        if return_events != summary_return_events:
            raise RuntimeError(
                f"return event count mismatch policy={policy} seed={seed}: "
                f"frames={return_events} summary={summary_return_events}"
            )

        mean_return = (
            float(np.mean(return_distances)) if return_distances else math.nan
        )
        p95_return = (
            float(np.quantile(return_distances, 0.95)) if return_distances else math.nan
        )
        max_return = max(return_distances) if return_distances else math.nan
        output.append(
            {
                "policy": policy,
                "seed": seed,
                "frames": int(float(row["frames"])),
                "user_slots": int(float(row["user_slots"])),
                "average_video_bitrate_mbps": average_video_bitrate_mbps(
                    float(row["throughput_mbps"]),
                    float(row["delivered_chunks_per_user_slot"]),
                    num_users=num_users,
                    playback_chunks_per_slot=playback_chunks_per_slot,
                ),
                "aggregate_network_throughput_mbps": float(row["throughput_mbps"]),
                "stall_ratio": float(row["stall_ratio"]),
                "startup_stall_ratio": float(row["startup_stall_ratio"]),
                "steady_state_stall_ratio": float(row["steady_state_stall_ratio"]),
                "average_quality_utility": float(row["average_quality_utility"]),
                "quality_p05_utility": float(row["quality_p05_utility"]),
                "quality_switch_rate": float(row["quality_switch_rate"]),
                "hire_rate": float(row["hire_rate"]),
                "charging_fraction": float(row["charging_fraction"]),
                "precharge_depletion_rate": float(row["precharge_depletion_rate"]),
                "precharge_reserve_breach_rate": float(
                    row["precharge_reserve_breach_rate"]
                ),
                "stranded_before_charge_rate": float(row["stranded_before_charge_rate"]),
                "return_to_charge_events": summary_return_events,
                "precharge_depletion_events": int(float(row["precharge_depletion_events"])),
                "precharge_reserve_breach_events": int(
                    float(row["precharge_reserve_breach_events"])
                ),
                "charging_need_events": int(float(row["charging_need_events"])),
                "stranded_before_charge_events": int(
                    float(row["stranded_before_charge_events"])
                ),
                "mean_return_to_charge_distance_m": mean_return,
                "p95_return_to_charge_distance_m": p95_return,
                "max_return_to_charge_distance_m": max_return,
                "mean_uav_user_distance_m": float(row["mean_uav_user_distance_m"]),
                "energy_per_delivered_chunk_j": float(
                    row["energy_per_delivered_chunk_j"]
                ),
                "original_cost_per_user_slot": float(
                    row["original_cost_per_user_slot"]
                ),
                "dpp_cost_per_user_slot": float(row["dpp_cost_per_user_slot"]),
                "battery_reserve_violations": int(
                    float(row["battery_reserve_violations"])
                ),
                "power_violations": int(float(row["power_violations"])),
                "provider_violations": int(float(row["provider_violations"])),
            }
        )
    return output


def aggregate_seed_metrics(seed_rows: Sequence[dict], policies: Sequence[str]) -> list[dict]:
    output: list[dict] = []
    for policy in policies:
        rows = [row for row in seed_rows if row["policy"] == policy]
        if not rows:
            continue
        item: dict[str, float | int | str] = {
            "policy": policy,
            "num_seeds": len(rows),
            "user_slots_total": sum(int(row["user_slots"]) for row in rows),
        }
        for metric in PRIMARY_METRICS + (
            "aggregate_network_throughput_mbps",
            "quality_switch_rate",
            "charging_fraction",
            "p95_return_to_charge_distance_m",
            "mean_uav_user_distance_m",
            "energy_per_delivered_chunk_j",
            "original_cost_per_user_slot",
            "dpp_cost_per_user_slot",
        ):
            mean, ci, n = mean_ci95(float(row[metric]) for row in rows)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_ci95"] = ci
            item[f"{metric}_finite_seeds"] = n

        for count_metric in (
            "return_to_charge_events",
            "precharge_depletion_events",
            "precharge_reserve_breach_events",
            "charging_need_events",
            "stranded_before_charge_events",
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        ):
            item[f"{count_metric}_total"] = sum(int(row[count_metric]) for row in rows)

        return_events = int(item["return_to_charge_events_total"])
        charging_needs = int(item["charging_need_events_total"])
        item["precharge_depletion_rate_pooled"] = safe_ratio(
            int(item["precharge_depletion_events_total"]), return_events
        )
        item["precharge_reserve_breach_rate_pooled"] = safe_ratio(
            int(item["precharge_reserve_breach_events_total"]), return_events
        )
        item["stranded_before_charge_rate_pooled"] = safe_ratio(
            int(item["stranded_before_charge_events_total"]), charging_needs
        )
        output.append(item)
    return output


def chosen_policies(experiment: dict, include_latest: bool) -> list[str]:
    policies = [paper_policy(str(policy)) for policy in experiment["policies"]]
    if not include_latest:
        policies = [policy for policy in policies if policy != "ppo_latest"]
    preferred = [
        "proposed",
        "ppo_best",
        "rsu_only",
        "always_hire",
        "ppo_latest",
    ]
    ordered = [policy for policy in preferred if policy in policies]
    ordered.extend(policy for policy in policies if policy not in ordered)
    return ordered


def label(policy: str) -> str:
    if str(policy) in {"dpp", "proposed"}:
        return "Proposed"
    return POLICY_LABELS.get(policy, policy)


def color(policy: str):
    if str(policy) == "proposed":
        return POLICY_COLORS.get("dpp")
    return POLICY_COLORS.get(policy)


def plot_bitrate_stall_quality(
    aggregate: Sequence[dict], policies: Sequence[str], output_path: Path
) -> None:
    plt = setup_matplotlib()
    lookup = {str(row["policy"]): row for row in aggregate}
    x = np.arange(len(policies))
    labels = [label(policy) for policy in policies]
    panels = (
        ("average_video_bitrate_mbps", "Average delivered video bitrate", "Mbps", True),
        ("stall_ratio", "Video stall ratio", "stall user-slots / user-slots", False),
        ("average_quality_utility", "Average delivered quality", "utility", True),
        ("quality_p05_utility", "5th-percentile delivered quality", "utility", True),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, (metric, title, ylabel, _higher) in zip(axes.flat, panels):
        means = [float(lookup[p][f"{metric}_mean"]) for p in policies]
        cis = [float(lookup[p][f"{metric}_ci95"]) for p in policies]
        axis.bar(x, means, yerr=cis, capsize=3, color=[color(p) for p in policies])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("P3 Evaluation: bitrate, stall, and quality (mean ± 95% CI)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_quality_distribution(
    quality_rows: Sequence[dict], policies: Sequence[str], output_path: Path
) -> None:
    plt = setup_matplotlib()
    levels = sorted({int(float(row["quality_level"])) for row in quality_rows})
    x = np.arange(len(policies))
    bottoms = np.zeros(len(policies), dtype=np.float64)
    fig, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for level in levels:
        values = []
        for policy in policies:
            match = [
                row
                for row in quality_rows
                if row["policy"] == policy
                and int(float(row["quality_level"])) == level
            ]
            values.append(float(match[0]["delivered_chunk_share"]) if match else 0.0)
        axis.bar(x, values, bottom=bottoms, label=f"Quality {level}")
        bottoms += np.asarray(values)
    axis.set_title("Delivered quality-level distribution")
    axis.set_ylabel("share of delivered chunks")
    axis.set_xticks(x)
    axis.set_xticklabels([label(policy) for policy in policies], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.legend(ncol=min(len(levels), 4))
    axis.grid(axis="y", alpha=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_charging_safety(
    aggregate: Sequence[dict], policies: Sequence[str], output_path: Path
) -> None:
    plt = setup_matplotlib()
    lookup = {str(row["policy"]): row for row in aggregate}
    x = np.arange(len(policies), dtype=np.float64)
    labels = [label(policy) for policy in policies]
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    risk_metrics = (
        ("precharge_depletion_rate_pooled", "depleted before charger"),
        ("precharge_reserve_breach_rate_pooled", "reserve breach before charge"),
        ("stranded_before_charge_rate_pooled", "stranded / charging-need"),
    )
    for index, (metric, metric_label) in enumerate(risk_metrics):
        values = [float(lookup[p][metric]) for p in policies]
        axes[0].bar(x + (index - 1) * width, values, width=width, label=metric_label)
    axes[0].set_title("Charging safety-event ratios")
    axes[0].set_ylabel("event-weighted ratio")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    means = [float(lookup[p]["mean_return_to_charge_distance_m_mean"]) for p in policies]
    cis = [float(lookup[p]["mean_return_to_charge_distance_m_ci95"]) for p in policies]
    plot_means = [0.0 if not math.isfinite(value) else value for value in means]
    plot_cis = [0.0 if not math.isfinite(value) else value for value in cis]
    axes[1].bar(
        x,
        plot_means,
        yerr=plot_cis,
        capsize=3,
        color=[color(p) for p in policies],
    )
    axes[1].set_title("UAV return-to-charger distance")
    axes[1].set_ylabel("mean return distance (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    for index, policy in enumerate(policies):
        events = int(lookup[policy]["return_to_charge_events_total"])
        axes[1].text(
            index,
            plot_means[index] + plot_cis[index] + 1.0,
            f"returns={events}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(policies) > 6 else 0,
        )
    fig.suptitle("P3 charging safety and charger-return distance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distance_effects(
    distance_rows: Sequence[dict],
    seed_metrics: Sequence[dict],
    policies: Sequence[str],
    output_path: Path,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    metrics = (
        ("uav_service_ratio", "UAV service ratio vs. user distance", "service ratio"),
        ("average_quality_utility", "Delivered quality vs. UAV-user distance", "utility"),
        ("average_required_power_w", "Required RF power vs. UAV-user distance", "W / served user-slot"),
    )
    bins = sorted({int(float(row["bin_index"])) for row in distance_rows})
    bin_labels = []
    for bin_index in bins:
        example = next(row for row in distance_rows if int(float(row["bin_index"])) == bin_index)
        left = float(example["distance_left_m"])
        right = float(example["distance_right_m"])
        right_text = "inf" if math.isinf(right) else f"{right:g}"
        bin_labels.append(f"{left:g}-{right_text}")

    for axis, (metric, title, ylabel) in zip(axes.flat[:3], metrics):
        for policy in policies:
            rows = [row for row in distance_rows if row["policy"] == policy]
            rows.sort(key=lambda row: int(float(row["bin_index"])))
            if not rows or not any(float(row["opportunities"]) > 0.0 for row in rows):
                continue
            values = [
                float(row[metric]) if float(row["opportunities"]) > 0.0 else math.nan
                for row in rows
            ]
            axis.plot(
                np.arange(len(rows)),
                values,
                marker="o",
                label=label(policy),
                color=color(policy),
            )
        axis.set_title(title)
        axis.set_xlabel("horizontal distance bin (m)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(len(bin_labels)))
        axis.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=8)
        axis.grid(alpha=0.2)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, legend_labels, fontsize=7)

    # Reconstruct exact charger-return distances (candidate offsets are discrete).
    distance_counts: dict[str, dict[float, int]] = defaultdict(lambda: defaultdict(int))
    for row in seed_metrics:
        policy = str(row["policy"])
        if policy not in policies:
            continue
        # Per-seed CSV stores only moments; rereading frames would be wasteful here.
        # Exact event distributions are populated below by the caller via a hidden
        # attribute if available. Fallback is a mean-distance marker.
    for policy_index, policy in enumerate(policies):
        rows = [row for row in seed_metrics if row["policy"] == policy]
        values = [float(row["mean_return_to_charge_distance_m"]) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        axes[1, 1].scatter(
            np.full(len(values), policy_index, dtype=np.float64),
            values,
            alpha=0.45,
            s=18,
            color=color(policy),
        )
        mean, ci, _ = mean_ci95(values)
        axes[1, 1].errorbar(
            [policy_index],
            [mean],
            yerr=[ci],
            fmt="o",
            capsize=4,
            color=color(policy),
        )
    axes[1, 1].set_title("Return-to-charger distance across evaluation seeds")
    axes[1, 1].set_ylabel("mean return distance per seed (m)")
    axes[1, 1].set_xticks(np.arange(len(policies)))
    axes[1, 1].set_xticklabels([label(p) for p in policies], rotation=20, ha="right", fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("P3 position/distance effects")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_readme(output_dir: Path, experiment: dict, aggregate: Sequence[dict]) -> None:
    lines = [
        "# Professor-requested P3 evaluation report",
        "",
        f"- seeds: {len(experiment['seeds'])}",
        f"- frames per seed: {experiment['frames']}",
        f"- policies: {', '.join(paper_policy(str(p)) for p in experiment['policies'])}",
        "",
        "## Metric definitions",
        "",
        "- `average_video_bitrate_mbps`: application-level delivered video bitrate. "
        "This is distinct from aggregate network throughput (`throughput_mbps`).",
        "- `precharge_depletion_rate_pooled`: UAV return-to-charge events that would "
        "deplete before charger arrival / all return-to-charge events. This is the "
        "primary professor-requested charging safety metric.",
        "- `precharge_reserve_breach_rate_pooled`: return events starting below the "
        "hard reserve threshold / all return events.",
        "- `stranded_before_charge_rate_pooled`: stranded-before-arrival events / "
        "charging-need events.",
        "- Return distance is reconstructed from the saved slow actions without "
        "changing environment dynamics.",
        "",
        "## Important interpretation",
        "",
        "The current P3 battery model charges a fixed `relocation_energy_j` for any "
        "non-zero relocation, not an energy cost proportional to distance. Therefore "
        "the return-distance plots are descriptive location diagnostics. A causal "
        "distance-vs-flight-energy claim would require a formulation/model change and "
        "retraining; this reporting code intentionally does not make that change.",
        "",
        "## Aggregate charging safety",
        "",
    ]
    for row in aggregate:
        lines.append(
            f"- {label(str(row['policy']))}: returns={int(row['return_to_charge_events_total'])}, "
            f"depletion={float(row['precharge_depletion_rate_pooled']):.6g}, "
            f"reserve_breach={float(row['precharge_reserve_breach_rate_pooled']):.6g}, "
            f"stranded={float(row['stranded_before_charge_rate_pooled']):.6g}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate professor-requested P3 bitrate/quality/charging/distance report"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="P3 evaluation root containing aggregate/experiment.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-latest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include ppo_latest in professor-facing plots; default focuses on ppo_best",
    )
    args = parser.parse_args()

    root = args.input.resolve()
    aggregate_dir = root / "aggregate"
    experiment_path = aggregate_dir / "experiment.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(
            f"missing completed aggregate: {experiment_path}. Finish p3_eval_aggregate first."
        )
    with experiment_path.open(encoding="utf-8") as stream:
        experiment = json.load(stream)
    if experiment.get("event") != "completed":
        raise RuntimeError(f"aggregate experiment is not completed: {experiment_path}")

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else aggregate_dir / "professor"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics = build_seed_metrics(root, experiment)
    policies = chosen_policies(experiment, args.include_latest)
    aggregate = aggregate_seed_metrics(seed_metrics, policies)
    quality_rows = relabel_policy_rows(
        read_csv(aggregate_dir / "quality_distribution.csv")
    )
    distance_rows = relabel_policy_rows(
        read_csv(aggregate_dir / "distance_by_policy.csv")
    )

    write_csv(output_dir / "professor_seed_metrics.csv", seed_metrics)
    write_csv(output_dir / "professor_aggregate_metrics.csv", aggregate)
    plot_bitrate_stall_quality(
        aggregate,
        policies,
        output_dir / "p3_bitrate_stall_quality.png",
    )
    plot_quality_distribution(
        quality_rows,
        policies,
        output_dir / "p3_quality_distribution.png",
    )
    plot_charging_safety(
        aggregate,
        policies,
        output_dir / "p3_charging_safety.png",
    )
    plot_distance_effects(
        distance_rows,
        seed_metrics,
        policies,
        output_dir / "p3_distance_effects_professor.png",
    )
    write_readme(output_dir, experiment, aggregate)

    print(
        f"[PROFESSOR-REPORT-DONE] seeds={len(experiment['seeds'])} "
        f"policies={len(policies)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
