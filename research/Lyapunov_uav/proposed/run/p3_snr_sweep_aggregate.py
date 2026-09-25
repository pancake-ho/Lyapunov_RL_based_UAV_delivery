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
    "proposed": "Proposed-NoRL",
    "slow_ppo": "Proposed-RL",
    "rsu_only": "RSU Only",
    "always_hire": "Always Hire",
}

# Paper-style: compact markers + clearly distinguishable line styles.
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

DEFAULT_SHOW_CI = False

_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def setup_matplotlib():
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))

    import matplotlib.pyplot as plt

    # Keep the style close to a conventional IEEE/Elsevier paper figure:
    # compact serif text, thin axes, no decorative grid, small markers.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
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

    for snr_db, policy, seed in build_tasks(
        snrs,
        seeds,
        policies,
    ):
        path = result_path(
            root,
            snr_db,
            policy,
            seed,
        )

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
                "runtime_seconds": float(
                    payload["runtime_seconds"]
                ),
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
                and abs(
                    float(row["snr_db"]) - float(snr_db)
                ) <= 1e-9
            ]

            if not selected:
                continue

            item: dict[str, float | int | str] = {
                "policy": policy,
                "snr_db": float(snr_db),
                "num_seeds": len(selected),
            }

            for metric in metrics:
                mean, ci, n = mean_ci95(
                    float(row[metric])
                    for row in selected
                )
                item[f"{metric}_mean"] = mean
                item[f"{metric}_ci95"] = ci
                item[f"{metric}_finite_seeds"] = n

            for violation in (
                "battery_reserve_violations",
                "power_violations",
                "provider_violations",
            ):
                item[f"{violation}_total"] = sum(
                    int(row[violation])
                    for row in selected
                )

            output.append(item)

    return output


def _series(
    aggregate: Sequence[dict],
    policy: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in aggregate
        if str(row["policy"]) == str(policy)
    ]

    selected.sort(
        key=lambda row: float(row["snr_db"])
    )

    x = np.asarray(
        [
            float(row["snr_db"])
            for row in selected
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            float(row[f"{metric}_mean"])
            for row in selected
        ],
        dtype=np.float64,
    )

    ci = np.asarray(
        [
            float(row[f"{metric}_ci95"])
            for row in selected
        ],
        dtype=np.float64,
    )

    return x, y, ci


def configure_axis(
    axis,
    snrs: Sequence[float],
    *,
    xlabel: str = "Nominal SNR (dB)",
) -> None:
    snr_values = np.asarray(
        tuple(float(value) for value in snrs),
        dtype=np.float64,
    )

    axis.set_xticks(snr_values)
    axis.set_xticklabels(
        [
            f"{int(value)}"
            if abs(value - round(value)) <= 1e-9
            else f"{value:g}"
            for value in snr_values
        ]
    )

    if snr_values.size > 0:
        left = float(np.min(snr_values))
        right = float(np.max(snr_values))
        margin = max(
            0.65,
            0.035 * max(right - left, 1.0),
        )
        axis.set_xlim(
            left - margin,
            right + margin,
        )

    axis.set_xlabel(xlabel)

    # The reference paper style does not use a strong background grid.
    axis.grid(False)

    axis.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
    )


def draw_paper_lines(
    axis,
    aggregate: Sequence[dict],
    policies: Sequence[str],
    *,
    metric: str,
    multiplier: float,
    show_ci: bool,
) -> None:
    for policy in policies:
        x, y, ci = _series(
            aggregate,
            policy,
            metric,
        )

        # Important: points stay exactly at the evaluated SNR values.
        # No visualization-only horizontal dodge is applied.
        axis.plot(
            x,
            y * multiplier,
            marker=POLICY_MARKERS.get(
                policy,
                "o",
            ),
            linestyle=POLICY_LINESTYLES.get(
                policy,
                "-",
            ),
            linewidth=1.25,
            markersize=4.2,
            markeredgewidth=0.8,
            label=POLICY_LABELS.get(
                policy,
                policy,
            ),
            zorder=3,
        )

        # The paper-style default intentionally omits error bars.
        # CI values are still preserved in snr_aggregate_metrics.csv.
        if show_ci:
            axis.errorbar(
                x,
                y * multiplier,
                yerr=ci * multiplier,
                fmt="none",
                elinewidth=0.75,
                capsize=2.0,
                capthick=0.75,
                alpha=0.55,
                zorder=2,
            )


def plot_metric(
    aggregate: Sequence[dict],
    policies: Sequence[str],
    snrs: Sequence[float],
    *,
    metric: str,
    ylabel: str,
    output: Path,
    multiplier: float = 1.0,
    ylim: tuple[float, float] | None = None,
    note: str | None = None,
    show_ci: bool = DEFAULT_SHOW_CI,
) -> None:
    plt = setup_matplotlib()

    fig, axis = plt.subplots(
        figsize=(4.3, 3.35),
        constrained_layout=True,
    )

    draw_paper_lines(
        axis,
        aggregate,
        policies,
        metric=metric,
        multiplier=multiplier,
        show_ci=show_ci,
    )

    configure_axis(
        axis,
        snrs,
    )

    axis.set_ylabel(ylabel)

    if ylim is not None:
        axis.set_ylim(*ylim)

    # Reference-like: compact legend above the axes, no large subplot title.
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        columnspacing=1.1,
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    if note:
        axis.text(
            0.5,
            -0.28,
            note,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_paper_panel(
    aggregate: Sequence[dict],
    policies: Sequence[str],
    snrs: Sequence[float],
    output: Path,
    *,
    show_ci: bool = DEFAULT_SHOW_CI,
) -> None:
    """Create one compact multi-panel figure close to the reference paper."""
    plt = setup_matplotlib()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.0),
        constrained_layout=True,
    )

    panels = (
        (
            "stall_ratio",
            "Stall Rate (%)",
            100.0,
            None,
        ),
        (
            "average_video_bitrate_mbps",
            "Average Video Bitrate (Mbps)",
            1.0,
            None,
        ),
        (
            "average_quality_utility",
            "Quality Utility $u(q)$",
            1.0,
            (0.5, 1.02),
        ),
        (
            "average_quality_level",
            "Average Quality Level",
            1.0,
            (1.0, 4.05),
        ),
        (
            "hire_rate",
            "UAV Hiring Rate (%)",
            100.0,
            (0.0, 100.0),
        ),
        (
            "original_cost_per_user_slot",
            "Quality Degradation + Hiring Cost / User-Slot",
            1.0,
            None,
        ),
    )

    for axis, (
        metric,
        ylabel,
        multiplier,
        ylim,
    ) in zip(axes.flat, panels):
        draw_paper_lines(
            axis,
            aggregate,
            policies,
            metric=metric,
            multiplier=multiplier,
            show_ci=show_ci,
        )

        configure_axis(
            axis,
            snrs,
            xlabel="Nominal SNR (dB)",
        )

        axis.set_ylabel(ylabel)

        if ylim is not None:
            axis.set_ylim(*ylim)

    # One shared legend, as in the reference multi-panel figure.
    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=min(4, len(labels)),
        columnspacing=1.35,
        handlelength=2.1,
        handletextpad=0.45,
        frameon=False,
    )

    fig.suptitle(
        "[Nominal SNR]",
        y=1.055,
        fontsize=12.5,
        fontweight="semibold",
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def write_readme(
    out: Path,
    snrs: Sequence[float],
    seeds: Sequence[int],
    policies: Sequence[str],
    *,
    show_ci: bool,
) -> None:
    lines = [
        "# P3 SNR sweep",
        "",
        f"- SNR points: {', '.join(f'{value:g}' for value in snrs)} dB",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        (
            "- policies: "
            + ", ".join(
                POLICY_LABELS.get(policy, policy)
                for policy in policies
            )
        ),
        "",
        "## Plot convention",
        "",
        "- Mean values at 20/25/30/35/40 dB are connected with paper-style lines.",
        "- Markers are placed exactly at the evaluated SNR values; no horizontal dodge is used.",
        "- Individual plots omit large titles and place a compact legend above the axis.",
        "- The multi-panel figure uses one shared legend and a group heading `[Nominal SNR]`.",
        (
            "- Error bars are shown."
            if show_ci
            else "- Error bars are hidden by default for paper-style readability."
        ),
        "- Student-t 95% CI values remain available in `snr_aggregate_metrics.csv`.",
        "",
        "## SNR definition",
        "",
        "P3 does not contain one global fixed transmit-SNR variable.",
        "Instantaneous SNR is derived from transmit power, path loss/distance,",
        "fading, bandwidth, Shannon gap, and noise PSD.",
        "",
        "The experiment defines 30 dB as the nominal operating point",
        "of the original trained channel and shifts all instantaneous link SNRs",
        "through the common noise PSD:",
        "",
        "- 20 dB -> baseline -10 dB",
        "- 25 dB -> baseline -5 dB",
        "- 30 dB -> original trained/evaluated channel",
        "- 35 dB -> baseline +5 dB",
        "- 40 dB -> baseline +10 dB",
        "",
        "Geometry, fading, power decisions and battery constraints remain active.",
        "",
        "## Quality utility",
        "",
        "- Q1: u=0.55, 0.5 Mbit/chunk",
        "- Q2: u=0.72, 1 Mbit/chunk",
        "- Q3: u=0.86, 2 Mbit/chunk",
        "- Q4: u=1.00, 4 Mbit/chunk",
        "",
        "## Policy display names",
        "",
        "- `Proposed-NoRL` = raw policy key `proposed`: slow-timescale structured rollout / DPP, no RL.",
        "- `Proposed-RL` = raw policy key `slow_ppo`: slow-timescale PPO using the validation-selected `best.pt` checkpoint.",
        "",
        "## Degradation + hiring objective",
        "",
        "`original_cost_per_user_slot` is kept as the internal CSV key for backward compatibility.",
        "Its plotted meaning is made explicit as:",
        "",
        "`quality degradation + weighted UAV hiring cost`, normalized by user-slots.",
    ]

    (out / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and render paper-style P3 SNR sweep figures"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--snrs-db",
        default=":".join(
            str(int(value))
            for value in DEFAULT_SNRS_DB
        ),
    )

    parser.add_argument(
        "--seeds",
        default=":".join(
            map(str, DEFAULT_SEEDS)
        ),
    )

    parser.add_argument(
        "--policies",
        default=":".join(
            DEFAULT_POLICIES
        ),
    )

    parser.add_argument(
        "--show-ci",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_CI,
        help=(
            "optionally overlay thin 95% Student-t CI bars; "
            "default is no CI bars to match the reference paper style"
        ),
    )

    parser.add_argument(
        "--if-complete",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    snrs = parse_list(
        args.snrs_db,
        float,
    )

    seeds = parse_list(
        args.seeds,
        int,
    )

    policies = parse_list(
        args.policies,
        str,
    )

    root = args.input.resolve()

    out = (
        args.output.resolve()
        if args.output is not None
        else root / "aggregate"
    )

    rows, missing = load_rows(
        root,
        snrs,
        seeds,
        policies,
    )

    if missing:
        if args.if_complete:
            print(
                f"[SNR-AGGREGATE-DEFER] "
                f"completed={len(rows)} "
                f"expected={len(rows) + len(missing)} "
                f"missing={len(missing)}",
                flush=True,
            )
            return

        preview = "\n".join(
            str(path)
            for path in missing[:10]
        )

        raise RuntimeError(
            f"SNR sweep incomplete; "
            f"missing {len(missing)} runs:\n"
            f"{preview}"
        )

    aggregate = aggregate_rows(
        rows,
        snrs,
        policies,
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_csv(
        out / "snr_seed_metrics.csv",
        rows,
    )

    atomic_write_csv(
        out / "snr_aggregate_metrics.csv",
        aggregate,
    )

    # Main paper-like 2x3 multi-panel figure.
    plot_paper_panel(
        aggregate,
        policies,
        snrs,
        out / "p3_snr_sweep_paper.png",
        show_ci=args.show_ci,
    )

    # Keep the old main filename for compatibility, but render it with the
    # exact same paper-style panel.
    plot_paper_panel(
        aggregate,
        policies,
        snrs,
        out / "p3_snr_sweep_main.png",
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="stall_ratio",
        ylabel="Stall Rate (%)",
        output=out / "01_stall_vs_snr.png",
        multiplier=100.0,
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="average_video_bitrate_mbps",
        ylabel="Average Video Bitrate (Mbps)",
        output=out / "02_bitrate_vs_snr.png",
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="average_quality_utility",
        ylabel="Quality Utility $u(q)$",
        output=out / "03_quality_utility_vs_snr.png",
        ylim=(0.5, 1.02),
        note=(
            "Q1: 0.55 | Q2: 0.72 | "
            "Q3: 0.86 | Q4: 1.00"
        ),
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="average_quality_level",
        ylabel="Average Quality Level",
        output=out / "04_quality_level_vs_snr.png",
        ylim=(1.0, 4.05),
        note=(
            "Q1/Q2/Q3/Q4 chunk sizes: "
            "0.5 / 1 / 2 / 4 Mbit"
        ),
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="hire_rate",
        ylabel="UAV Hiring Rate (%)",
        output=out / "05_hiring_vs_snr.png",
        multiplier=100.0,
        ylim=(0.0, 100.0),
        show_ci=args.show_ci,
    )

    plot_metric(
        aggregate,
        policies,
        snrs,
        metric="original_cost_per_user_slot",
        ylabel="Quality Degradation + Hiring Cost / User-Slot",
        output=out / "06_degradation_hiring_cost_vs_snr.png",
        show_ci=args.show_ci,
    )

    write_readme(
        out,
        snrs,
        seeds,
        policies,
        show_ci=args.show_ci,
    )

    print(
        f"[SNR-AGGREGATE-DONE] "
        f"runs={len(rows)} "
        f"snrs={len(snrs)} "
        f"seeds={len(seeds)} "
        f"policies={len(policies)} "
        f"style=paper-lines "
        f"show_ci={args.show_ci} "
        f"output={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
