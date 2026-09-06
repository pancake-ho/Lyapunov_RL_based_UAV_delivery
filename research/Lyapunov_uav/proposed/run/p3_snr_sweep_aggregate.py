from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from run.p3_snr_sweep import (
    DEFAULT_POLICIES,
    DEFAULT_SEEDS,
    DEFAULT_SNRS_DB,
    build_tasks,
    parse_list,
    result_path,
)


POLICY_LABELS = {
    "proposed": "Proposed",
    "slow_ppo": "Slow-PPO",
    "rsu_only": "RSU Only",
    "always_hire": "Always Hire",
}
POLICY_MARKERS = {
    "proposed": "o",
    "slow_ppo": "s",
    "rsu_only": "^",
    "always_hire": "x",
}
POLICY_LINESTYLES = {
    "proposed": "-",
    "slow_ppo": "--",
    "rsu_only": ":",
    "always_hire": "-.",
}
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t95(n: int) -> float:
    if n <= 1:
        return 0.0
    return _T95.get(n - 1, 1.96)


def mean_ci95(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, 0
    mean = float(np.mean(array))
    if array.size == 1:
        return mean, 0.0, 1
    se = float(np.std(array, ddof=1)) / math.sqrt(array.size)
    return mean, t95(int(array.size)) * se, int(array.size)


def atomic_write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )
    return plt


def load_rows(
    root: Path,
    snrs: Sequence[float],
    seeds: Sequence[int],
    policies: Sequence[str],
) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    missing: list[Path] = []
    for snr_db, policy, seed in build_tasks(snrs, seeds, policies):
        path = result_path(root, snr_db, policy, seed)
        if not path.is_file():
            missing.append(path)
            continue
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("event") != "completed":
            missing.append(path)
            continue
        rows.append(
            {
                "snr_db": float(payload["snr_db"]),
                "policy": str(payload["policy"]),
                "seed": int(payload["seed"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                **{
                    key: float(value)
                    for key, value in payload["metrics"].items()
                },
            }
        )
    return rows, missing


def aggregate_rows(
    rows: Sequence[dict],
    snrs: Sequence[float],
    policies: Sequence[str],
) -> list[dict]:
    metrics = (
        "average_video_bitrate_mbps",
        "stall_ratio",
        "average_quality_utility",
        "average_quality_level",
        "quality_p05_utility",
        "hire_rate",
        "original_cost_per_user_slot",
        "dpp_cost_per_user_slot",
        "delivered_chunks_per_user_slot",
        "aggregate_network_throughput_mbps",
        "mean_uav_user_distance_m",
        "energy_per_delivered_chunk_j",
        "max_queue",
        "large_queue_violation_rate",
    )
    output: list[dict] = []
    for policy in policies:
        for snr_db in snrs:
            selected = [
                row
                for row in rows
                if row["policy"] == policy
                and abs(float(row["snr_db"]) - float(snr_db)) <= 1e-9
            ]
            if not selected:
                continue
            item: dict[str, float | int | str] = {
                "policy": policy,
                "snr_db": float(snr_db),
                "num_seeds": len(selected),
            }
            for metric in metrics:
                mean, ci, n = mean_ci95(float(row[metric]) for row in selected)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_ci95"] = ci
                item[f"{metric}_finite_seeds"] = n
            for violation in (
                "battery_reserve_violations",
                "power_violations",
                "provider_violations",
            ):
                item[f"{violation}_total"] = sum(
                    int(row[violation]) for row in selected
                )
            output.append(item)
    return output


def _series(aggregate: Sequence[dict], policy: str, metric: str):
    selected = [row for row in aggregate if row["policy"] == policy]
    selected.sort(key=lambda row: float(row["snr_db"]))
    x = np.asarray(
        [float(row["snr_db"]) for row in selected],
        dtype=np.float64,
    )
    y = np.asarray(
        [float(row[f"{metric}_mean"]) for row in selected],
        dtype=np.float64,
    )
    ci = np.asarray(
        [float(row[f"{metric}_ci95"]) for row in selected],
        dtype=np.float64,
    )
    return x, y, ci


def plot_metric(
    aggregate: Sequence[dict],
    policies: Sequence[str],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    multiplier: float = 1.0,
    ylim: tuple[float, float] | None = None,
    note: str | None = None,
) -> None:
    plt = setup_matplotlib()
    fig, axis = plt.subplots(figsize=(6.7, 5.1), constrained_layout=True)

    for policy in policies:
        x, y, ci = _series(aggregate, policy, metric)
        axis.errorbar(
            x,
            y * multiplier,
            yerr=ci * multiplier,
            marker=POLICY_MARKERS.get(policy, "o"),
            linestyle=POLICY_LINESTYLES.get(policy, "-"),
            linewidth=1.8,
            markersize=6,
            capsize=3,
            label=POLICY_LABELS.get(policy, policy),
        )

    axis.set_title(title)
    axis.set_xlabel("Nominal SNR (dB)")
    axis.set_ylabel(ylabel)
    axis.set_xticks([20, 25, 30, 35, 40])
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.18)
    axis.legend(ncol=2, loc="best")
    if note:
        axis.text(
            0.5,
            -0.20,
            note,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_main_panel(
    aggregate: Sequence[dict],
    policies: Sequence[str],
    output: Path,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.6), constrained_layout=True)
    panels = (
        ("stall_ratio", "Stall Rate (%)", "Stall Rate", 100.0),
        (
            "average_video_bitrate_mbps",
            "Average Video Bitrate (Mbps)",
            "Average Video Bitrate",
            1.0,
        ),
        (
            "average_quality_utility",
            "Average Quality Utility u(q)",
            "Quality Utility",
            1.0,
        ),
        ("hire_rate", "UAV Hiring Rate (%)", "UAV Hiring Rate", 100.0),
    )

    for axis, (metric, ylabel, title, multiplier) in zip(axes.flat, panels):
        for policy in policies:
            x, y, ci = _series(aggregate, policy, metric)
            axis.errorbar(
                x,
                y * multiplier,
                yerr=ci * multiplier,
                marker=POLICY_MARKERS.get(policy, "o"),
                linestyle=POLICY_LINESTYLES.get(policy, "-"),
                linewidth=1.8,
                markersize=5.5,
                capsize=2.5,
                label=POLICY_LABELS.get(policy, policy),
            )
        axis.set_title(title)
        axis.set_xlabel("Nominal SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([20, 25, 30, 35, 40])
        axis.grid(alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    out: Path,
    snrs: Sequence[float],
    seeds: Sequence[int],
    policies: Sequence[str],
) -> None:
    lines = [
        "# P3 SNR sweep",
        "",
        f"- SNR points: {', '.join(f'{value:g}' for value in snrs)} dB",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- policies: {', '.join(POLICY_LABELS.get(p, p) for p in policies)}",
        "",
        "## SNR definition",
        "",
        "P3 does not contain one global fixed transmit-SNR variable.",
        "Instantaneous SNR is derived from transmit power, path loss/distance,",
        "fading, bandwidth, Shannon gap, and noise PSD.",
        "",
        "This experiment therefore defines 30 dB as the nominal operating point",
        "of the original trained channel and shifts all instantaneous link SNRs",
        "by changing the common noise PSD:",
        "",
        "- 20 dB -> baseline -10 dB",
        "- 25 dB -> baseline -5 dB",
        "- 30 dB -> exactly the original trained channel",
        "- 35 dB -> baseline +5 dB",
        "- 40 dB -> baseline +10 dB",
        "",
        "Geometry, fading, transmit-power decisions and battery constraints remain active.",
        "For this reason the x-axis is `Nominal SNR (dB)`, not a fixed per-link `Transmit SNR`.",
        "",
        "## Quality utility",
        "",
        "- Q1: u=0.55, 0.5 Mbit/chunk",
        "- Q2: u=0.72, 1 Mbit/chunk",
        "- Q3: u=0.86, 2 Mbit/chunk",
        "- Q4: u=1.00, 4 Mbit/chunk",
        "",
        "`Slow-PPO` means the validation-selected `best.pt` checkpoint.",
        "95% confidence intervals are seed-level Student-t intervals.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate and plot P3 SNR sweep")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--snrs-db",
        default=":".join(str(int(value)) for value in DEFAULT_SNRS_DB),
    )
    parser.add_argument("--seeds", default=":".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--policies", default=":".join(DEFAULT_POLICIES))
    parser.add_argument("--if-complete", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    snrs = parse_list(args.snrs_db, float)
    seeds = parse_list(args.seeds, int)
    policies = parse_list(args.policies, str)

    root = args.input.resolve()
    out = args.output.resolve() if args.output is not None else root / "aggregate"
    rows, missing = load_rows(root, snrs, seeds, policies)

    if missing:
        if args.if_complete:
            print(
                f"[SNR-AGGREGATE-DEFER] completed={len(rows)} "
                f"expected={len(rows)+len(missing)} missing={len(missing)}",
                flush=True,
            )
            return
        preview = "\n".join(str(path) for path in missing[:10])
        raise RuntimeError(
            f"SNR sweep incomplete; missing {len(missing)} runs:\n{preview}"
        )

    aggregate = aggregate_rows(rows, snrs, policies)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(out / "snr_seed_metrics.csv", rows)
    atomic_write_csv(out / "snr_aggregate_metrics.csv", aggregate)

    plot_main_panel(aggregate, policies, out / "p3_snr_sweep_main.png")
    plot_metric(
        aggregate,
        policies,
        metric="stall_ratio",
        ylabel="Stall Rate (%)",
        title="Stall Rate vs. Nominal SNR",
        output=out / "01_stall_vs_snr.png",
        multiplier=100.0,
    )
    plot_metric(
        aggregate,
        policies,
        metric="average_video_bitrate_mbps",
        ylabel="Average Video Bitrate (Mbps)",
        title="Average Video Bitrate vs. Nominal SNR",
        output=out / "02_bitrate_vs_snr.png",
    )
    plot_metric(
        aggregate,
        policies,
        metric="average_quality_utility",
        ylabel="Average Quality Utility u(q)",
        title="Quality Utility vs. Nominal SNR",
        output=out / "03_quality_utility_vs_snr.png",
        ylim=(0.5, 1.02),
        note="Q1: u=0.55 | Q2: u=0.72 | Q3: u=0.86 | Q4: u=1.00",
    )
    plot_metric(
        aggregate,
        policies,
        metric="average_quality_level",
        ylabel="Average Quality Level",
        title="Average Quality Level vs. Nominal SNR",
        output=out / "04_quality_level_vs_snr.png",
        ylim=(1.0, 4.05),
        note="Q1/Q2/Q3/Q4 chunk sizes: 0.5 / 1 / 2 / 4 Mbit",
    )
    plot_metric(
        aggregate,
        policies,
        metric="hire_rate",
        ylabel="UAV Hiring Rate (%)",
        title="UAV Hiring Rate vs. Nominal SNR",
        output=out / "05_hiring_vs_snr.png",
        multiplier=100.0,
        ylim=(0.0, 100.0),
    )
    plot_metric(
        aggregate,
        policies,
        metric="original_cost_per_user_slot",
        ylabel="Original Objective Cost / User-Slot",
        title="Original Cost vs. Nominal SNR",
        output=out / "06_original_cost_vs_snr.png",
    )
    write_readme(out, snrs, seeds, policies)

    print(
        f"[SNR-AGGREGATE-DONE] runs={len(rows)} "
        f"snrs={len(snrs)} seeds={len(seeds)} policies={len(policies)} "
        f"output={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
