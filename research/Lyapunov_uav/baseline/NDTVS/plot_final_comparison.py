#!/usr/bin/env python3
"""
Paper-style final comparison plots for Track-A evaluation.

This script reads the three paired final evaluation results:

    <final_root>/ndtvs/evaluation.json
    <final_root>/hppo_rsu/evaluation.json
    <final_root>/proposed/evaluation.json

and produces paper-style 1x3 line figures similar to conventional
networking-paper performance plots.

IMPORTANT
---------
The x-axis is the paired held-out test scenario index.

It is NOT a controlled physical parameter such as SNR, V, bandwidth,
or hiring cost. Therefore these plots show scenario-wise paired
performance, not a parameter sweep.

Outputs
-------
<out>/qoe_per_scenario.png
<out>/qoe_per_scenario.pdf

<out>/service_per_scenario.png
<out>/service_per_scenario.pdf

<out>/resource_per_scenario.png
<out>/resource_per_scenario.pdf

<out>/paper_plot_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ALGORITHMS = (
    "proposed",
    "hppo_rsu",
    "ndtvs",
)

DISPLAY_NAMES = {
    "proposed": "Proposed",
    "hppo_rsu": "HPPO-RSU",
    "ndtvs": "NDTVS",
}


# Print-friendly combination of marker and line style.
# Colors are secondary; the figures remain distinguishable in grayscale.
STYLE = {
    "proposed": {
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.6,
        "markersize": 4.5,
    },
    "hppo_rsu": {
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.5,
        "markersize": 4.2,
    },
    "ndtvs": {
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.5,
        "markersize": 4.5,
    },
}


def configure_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(
            Path(tempfile.gettempdir())
            / "trackA_paper_final_plot"
        ),
    )

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )

    return plt


def load_evaluation(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"evaluation.json not found: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if "per_episode" not in data:
        raise ValueError(
            f"'per_episode' missing: {path}"
        )

    if not data["per_episode"]:
        raise ValueError(
            f"No evaluation episodes: {path}"
        )

    return data


def load_all(final_root: Path):
    data = {}

    for algorithm in ALGORITHMS:
        path = (
            final_root
            / algorithm
            / "evaluation.json"
        )

        result = load_evaluation(path)

        if result["algorithm"] != algorithm:
            raise ValueError(
                f"Algorithm mismatch: "
                f"{path} says {result['algorithm']}"
            )

        data[algorithm] = result

    # ---------------------------------------------------------
    # Paired-test consistency checks
    # ---------------------------------------------------------
    reference = data["proposed"]

    ref_ids = [
        row["episode"]
        for row in reference["per_episode"]
    ]

    for algorithm in ALGORITHMS[1:]:
        current = data[algorithm]

        if current["offset"] != reference["offset"]:
            raise ValueError(
                f"Offset mismatch: {algorithm}"
            )

        if (
            current["scenario_seed"]
            != reference["scenario_seed"]
        ):
            raise ValueError(
                f"Scenario seed mismatch: {algorithm}"
            )

        current_ids = [
            row["episode"]
            for row in current["per_episode"]
        ]

        if current_ids != ref_ids:
            raise ValueError(
                f"Episode identity mismatch: {algorithm}"
            )

    return data


def series(data, algorithm, metric):
    rows = data[algorithm]["per_episode"]

    missing = [
        i
        for i, row in enumerate(rows)
        if metric not in row
    ]

    if missing:
        raise KeyError(
            f"{metric} missing from {algorithm} "
            f"episode rows {missing[:5]}"
        )

    return np.asarray(
        [
            float(row[metric])
            for row in rows
        ],
        dtype=float,
    )


def scenario_axis(data):
    count = len(
        data["proposed"]["per_episode"]
    )

    return np.arange(
        1,
        count + 1,
        dtype=int,
    )


def plot_algorithm_lines(
    ax,
    x,
    data,
    metric,
    ylabel,
    ylim=None,
    show_legend=False,
):
    for algorithm in ALGORITHMS:
        y = series(
            data,
            algorithm,
            metric,
        )

        ax.plot(
            x,
            y,
            label=DISPLAY_NAMES[algorithm],
            **STYLE[algorithm],
        )

    ax.set_xlabel(
        "Paired Test Scenario"
    )

    ax.set_ylabel(ylabel)

    ax.set_xticks(x)

    if len(x) > 10:
        # Avoid overcrowded labels for 20 scenarios.
        visible = np.arange(
            1,
            len(x) + 1,
            2,
        )

        ax.set_xticks(visible)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.35,
    )

    if show_legend:
        ax.legend(
            loc="best",
            frameon=True,
        )


def save_figure(
    fig,
    out_dir,
    stem,
    dpi,
):
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    return png, pdf


def plot_qoe_figure(
    data,
    out_dir,
    dpi,
):
    """
    Closest analogue to the reference paper:

      Buffering Rate     -> Stall Ratio
      Average Quality    -> Average Quality Utility
      Fluctuation Rate   -> Switch Magnitude
    """

    plt = configure_matplotlib()

    x = scenario_axis(data)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.15),
        layout="constrained",
    )

    plot_algorithm_lines(
        axes[0],
        x,
        data,
        "stall_ratio",
        "Stall Ratio",
        ylim=(0.0, None),
        show_legend=True,
    )

    plot_algorithm_lines(
        axes[1],
        x,
        data,
        "average_quality_utility",
        "Average Quality Utility",
        ylim=(0.0, None),
    )

    plot_algorithm_lines(
        axes[2],
        x,
        data,
        "switch_magnitude_per_user_slot",
        "Quality Switching / User-slot",
        ylim=(0.0, None),
    )

    # No large suptitle:
    # networking papers normally put explanation in the caption.
    png, pdf = save_figure(
        fig,
        out_dir,
        "qoe_per_scenario",
        dpi,
    )

    plt.close(fig)

    return png, pdf


def plot_service_figure(
    data,
    out_dir,
    dpi,
):
    plt = configure_matplotlib()

    x = scenario_axis(data)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.15),
        layout="constrained",
    )

    plot_algorithm_lines(
        axes[0],
        x,
        data,
        "qoe_surrogate_per_user_slot",
        "QoE / User-slot",
        show_legend=True,
    )

    plot_algorithm_lines(
        axes[1],
        x,
        data,
        "delivered_chunks_per_user_slot",
        "Delivered Chunks / User-slot",
        ylim=(0.0, None),
    )

    plot_algorithm_lines(
        axes[2],
        x,
        data,
        "request_failure_ratio",
        "Request Failure Ratio",
        ylim=(0.0, None),
    )

    png, pdf = save_figure(
        fig,
        out_dir,
        "service_per_scenario",
        dpi,
    )

    plt.close(fig)

    return png, pdf


def plot_resource_figure(
    data,
    out_dir,
    dpi,
):
    plt = configure_matplotlib()

    x = scenario_axis(data)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.2, 3.15),
        layout="constrained",
    )

    plot_algorithm_lines(
        axes[0],
        x,
        data,
        "dpp_cost_per_user_slot",
        "DPP Cost / User-slot",
        show_legend=True,
    )

    plot_algorithm_lines(
        axes[1],
        x,
        data,
        "hire_rate",
        "UAV Hiring Rate",
        ylim=(0.0, 1.0),
    )

    plot_algorithm_lines(
        axes[2],
        x,
        data,
        "energy_consumed_j",
        "UAV Energy Consumption [J]",
        ylim=(0.0, None),
    )

    png, pdf = save_figure(
        fig,
        out_dir,
        "resource_per_scenario",
        dpi,
    )

    plt.close(fig)

    return png, pdf


def save_manifest(
    final_root,
    output_dir,
    data,
):
    reference = data["proposed"]

    manifest = {
        "final_root": str(
            final_root.resolve()
        ),
        "output_dir": str(
            output_dir.resolve()
        ),
        "algorithms": list(ALGORITHMS),
        "display_names": DISPLAY_NAMES,
        "scenario_seed": (
            reference["scenario_seed"]
        ),
        "offset": reference["offset"],
        "paired_test_episodes": len(
            reference["per_episode"]
        ),
        "x_axis": (
            "Paired held-out test scenario index"
        ),
        "warning": (
            "The x-axis is not a controlled parameter sweep. "
            "Lines connect paired held-out scenarios only."
        ),
        "figures": {
            "qoe_per_scenario": [
                "stall_ratio",
                "average_quality_utility",
                "switch_magnitude_per_user_slot",
            ],
            "service_per_scenario": [
                "qoe_surrogate_per_user_slot",
                "delivered_chunks_per_user_slot",
                "request_failure_ratio",
            ],
            "resource_per_scenario": [
                "dpp_cost_per_user_slot",
                "hire_rate",
                "energy_consumed_j",
            ],
        },
    }

    (
        output_dir
        / "paper_plot_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "final_root",
        type=Path,
        help=(
            "Final evaluation directory containing "
            "ndtvs/, hppo_rsu/, proposed/"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: <final_root>/comparison/paper_plots"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if not 100 <= args.dpi <= 600:
        parser.error(
            "--dpi must be between 100 and 600"
        )

    final_root = (
        args.final_root.resolve()
    )

    if not final_root.is_dir():
        parser.error(
            f"Not a directory: {final_root}"
        )

    output_dir = (
        args.out.resolve()
        if args.out is not None
        else (
            final_root
            / "comparison"
            / "paper_plots"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_all(final_root)

    qoe = plot_qoe_figure(
        data,
        output_dir,
        args.dpi,
    )

    service = plot_service_figure(
        data,
        output_dir,
        args.dpi,
    )

    resource = plot_resource_figure(
        data,
        output_dir,
        args.dpi,
    )

    save_manifest(
        final_root,
        output_dir,
        data,
    )

    print(
        "[OK] Paper-style figures generated"
    )

    for path in (
        *qoe,
        *service,
        *resource,
        output_dir
        / "paper_plot_manifest.json",
    ):
        print(" -", path)


if __name__ == "__main__":
    main()