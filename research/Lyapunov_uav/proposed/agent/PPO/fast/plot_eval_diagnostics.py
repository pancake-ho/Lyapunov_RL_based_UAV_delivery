from __future__ import annotations

import argparse
import csv
import math

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEEDS = (
    2026,
    2027,
    2028,
    2029,
    2030,
)

TERM_METRICS = (
    (
        "queue_playback_term",
        "Queue / Playback",
    ),
    (
        "video_delivery_term",
        "Video Delivery",
    ),
    (
        "quality_degradation_term",
        "Quality Degradation",
    ),
    (
        "battery_consume_term",
        "Battery Consumption",
    ),
    (
        "battery_charge_term",
        "Battery Charging",
    ),
)

PERFORMANCE_METRICS = (
    (
        "realized_round_cost",
        "Realized DPP Cost",
    ),
    (
        "scheduled_stall_rate",
        "Scheduled Stall Rate",
    ),
    (
        "delivery_per_scheduled_user_slot",
        "Delivery / Scheduled User-Slot",
    ),
    (
        "quality_per_chunk",
        "Quality / Chunk",
    ),
    (
        "quality_degradation_per_chunk",
        "Quality Degradation / Chunk",
    ),
    (
        "service_rate",
        "Service Rate",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--random-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--rsu-only-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--full-dpp-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--slow-t",
        type=int,
        default=3600,
    )

    return parser.parse_args()


def read_rows(
    path: Path,
) -> List[
    Dict[str, str]
]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def as_float(
    row: Dict[str, str],
    key: str,
) -> float:
    value = float(
        row[key]
    )

    if not math.isfinite(
        value
    ):
        return float(
            "nan"
        )

    return value


def t95(
    n: int,
) -> float:
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }

    if n < 2:
        return float(
            "nan"
        )

    return table.get(
        n,
        1.96,
    )


def mean_ci(
    values: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "mean_ci expects [seed, point] array."
        )

    num_points = int(
        values.shape[1]
    )

    mean = np.full(
        num_points,
        np.nan,
        dtype=np.float64,
    )

    half = np.full(
        num_points,
        np.nan,
        dtype=np.float64,
    )

    for idx in range(
        num_points
    ):
        column = values[
            :,
            idx,
        ]

        finite = column[
            np.isfinite(
                column
            )
        ]

        n = int(
            finite.size
        )

        if n == 0:
            continue

        mean[idx] = float(
            np.mean(
                finite
            )
        )

        if n == 1:
            half[idx] = 0.0
            continue

        std = float(
            np.std(
                finite,
                ddof=1,
            )
        )

        half[idx] = (
            t95(n)
            * std
            / math.sqrt(
                n
            )
        )

    return mean, half


def load_method(
    root: Path,
    seeds: Sequence[int],
) -> Dict[
    int,
    List[
        Dict[str, str]
    ]
]:
    result = {}

    for seed in seeds:
        path = (
            root
            / f"seed{seed}"
            / "logs"
            / "eval_rounds.csv"
        )

        rows = read_rows(
            path
        )

        if not rows:
            raise RuntimeError(
                f"Empty evaluation CSV: {path}"
            )

        result[
            int(seed)
        ] = rows

    return result


def verify_matching_grid(
    datasets: Dict[
        str,
        Dict[
            int,
            List[
                Dict[str, str]
            ]
        ]
    ],
    seeds: Sequence[int],
) -> None:
    names = list(
        datasets.keys()
    )

    reference_name = names[0]

    for seed in seeds:
        reference = datasets[
            reference_name
        ][seed]

        for name in names[1:]:
            candidate = datasets[
                name
            ][seed]

            if len(
                candidate
            ) != len(
                reference
            ):
                raise RuntimeError(
                    f"Row count mismatch: "
                    f"seed={seed}, "
                    f"{reference_name}="
                    f"{len(reference)}, "
                    f"{name}={len(candidate)}"
                )

            for idx, (
                lhs,
                rhs,
            ) in enumerate(
                zip(
                    reference,
                    candidate,
                ),
                start=1,
            ):
                for key in (
                    "episode",
                    "round_in_episode",
                    "global_eval_round",
                ):
                    if (
                        lhs[key]
                        != rhs[key]
                    ):
                        raise RuntimeError(
                            "Evaluation grid mismatch: "
                            f"seed={seed}, row={idx}, "
                            f"key={key}, "
                            f"{reference_name}={lhs[key]}, "
                            f"{name}={rhs[key]}"
                        )

                for key in (
                    "exogenous_start_digest",
                    "exogenous_end_digest",
                ):
                    lhs_value = lhs.get(
                        key
                    )

                    rhs_value = rhs.get(
                        key
                    )

                    if (
                        lhs_value is None
                        or rhs_value is None
                        or lhs_value == ""
                        or rhs_value == ""
                    ):
                        raise RuntimeError(
                            "Missing exogenous digest: "
                            f"seed={seed}, "
                            f"row={idx}, "
                            f"key={key}"
                        )

                    if (
                        lhs_value
                        != rhs_value
                    ):
                        raise RuntimeError(
                            "Exogenous trace mismatch: "
                            f"seed={seed}, "
                            f"row={idx}, "
                            f"key={key}, "
                            f"{reference_name} vs {name}"
                        )

def metric_array(
    method_data: Dict[
        int,
        List[
            Dict[str, str]
        ]
    ],
    seeds: Sequence[int],
    key: str,
) -> np.ndarray:
    return np.asarray(
        [
            [
                as_float(
                    row,
                    key,
                )
                for row in method_data[
                    int(seed)
                ]
            ]
            for seed in seeds
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = tuple(
        int(seed)
        for seed in args.seeds
    )

    slow_t = int(
        args.slow_t
    )

    if slow_t <= 0:
        raise ValueError(
            "slow_t must be positive."
        )

    datasets = {
        "Random":
            load_method(
                args.random_root,
                seeds,
            ),

        "RSU-only DPP":
            load_method(
                args.rsu_only_root,
                seeds,
            ),

        "Full DPP":
            load_method(
                args.full_dpp_root,
                seeds,
            ),
    }

    verify_matching_grid(
        datasets,
        seeds,
    )

    reference_rows = datasets[
        "Random"
    ][
        seeds[0]
    ]

    x = np.asarray(
        [
            int(
                row[
                    "global_eval_round"
                ]
            )
            * slow_t
            for row in reference_rows
        ],
        dtype=np.int64,
    )

    # ==============================================================
    # Figure 1: weighted reward-term magnitude
    # ==============================================================

    fig, axes = plt.subplots(
        len(
            TERM_METRICS
        ),
        1,
        figsize=(
            11,
            3.0
            * len(
                TERM_METRICS
            ),
        ),
        sharex=True,
    )

    axes = np.atleast_1d(
        axes
    )

    for axis, (
        key,
        label,
    ) in zip(
        axes,
        TERM_METRICS,
    ):
        for method, data in datasets.items():
            values = np.abs(
                metric_array(
                    data,
                    seeds,
                    key,
                )
            ) / float(
                slow_t
            )

            mean, half = mean_ci(
                values
            )

            axis.plot(
                x,
                mean,
                label=method,
                linewidth=1.7,
            )

            axis.fill_between(
                x,
                mean - half,
                mean + half,
                alpha=0.16,
            )

        axis.set_ylabel(
            label
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    axes[-1].set_xlabel(
        "Global Step"
    )

    axes[0].legend()

    fig.suptitle(
        "Weighted Reward-Term Magnitude "
        f"({len(seeds)}-seed mean ± 95% CI)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "weighted_reward_terms_vs_global_step.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ==============================================================
    # Figure 2: final performance
    # ==============================================================

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(
            12,
            10,
        ),
        sharex=True,
    )

    axes_flat = axes.reshape(
        -1
    )

    for axis, (
        key,
        label,
    ) in zip(
        axes_flat,
        PERFORMANCE_METRICS,
    ):
        for method, data in datasets.items():
            values = metric_array(
                data,
                seeds,
                key,
            )

            mean, half = mean_ci(
                values
            )

            axis.plot(
                x,
                mean,
                label=method,
                linewidth=1.7,
            )

            axis.fill_between(
                x,
                mean - half,
                mean + half,
                alpha=0.16,
            )

        axis.set_ylabel(
            label
        )

        axis.grid(
            True,
            alpha=0.3,
        )

    for axis in axes[-1]:
        axis.set_xlabel(
            "Global Step"
        )

    axes_flat[0].legend()

    fig.suptitle(
        "Evaluation Performance "
        "(5-seed mean ± 95% CI)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "performance_vs_global_step.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ==============================================================
    # Figure 3: reward-term composition
    # ==============================================================

    method_names = list(
        datasets.keys()
    )

    composition = []

    for method in method_names:
        data = datasets[
            method
        ]

        magnitudes = []

        for key, _ in TERM_METRICS:
            values = np.abs(
                metric_array(
                    data,
                    seeds,
                    key,
                )
            )

            seed_means = np.nanmean(
                values,
                axis=1,
            )

            magnitudes.append(
                float(
                    np.nanmean(
                        seed_means
                    )
                )
            )

        magnitudes_arr = np.asarray(
            magnitudes,
            dtype=np.float64,
        )

        denominator = max(
            float(
                np.sum(
                    magnitudes_arr
                )
            ),
            1e-12,
        )

        composition.append(
            magnitudes_arr
            / denominator
        )

    composition_arr = np.asarray(
        composition
    )

    x_index = np.arange(
        len(
            method_names
        )
    )

    fig, axis = plt.subplots(
        figsize=(
            10,
            5.5,
        )
    )

    bottom = np.zeros(
        len(
            method_names
        ),
        dtype=np.float64,
    )

    for term_idx, (
        _,
        label,
    ) in enumerate(
        TERM_METRICS
    ):
        values = composition_arr[
            :,
            term_idx,
        ]

        axis.bar(
            x_index,
            values,
            bottom=bottom,
            label=label,
        )

        bottom += values

    axis.set_xticks(
        x_index,
        method_names,
    )

    axis.set_ylabel(
        "Weighted Magnitude Share"
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "reward_term_composition.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ==============================================================
    # Figure 4: DPP forecast error
    # ==============================================================

    fig, axis = plt.subplots(
        figsize=(
            10,
            5,
        )
    )

    for method in (
        "RSU-only DPP",
        "Full DPP",
    ):
        values = metric_array(
            datasets[
                method
            ],
            seeds,
            "prediction_abs_relative_error",
        )

        mean, half = mean_ci(
            100.0
            * values
        )

        axis.plot(
            x,
            mean,
            label=method,
            linewidth=1.7,
        )

        axis.fill_between(
            x,
            mean - half,
            mean + half,
            alpha=0.16,
        )

    axis.set_xlabel(
        "Global Step"
    )

    axis.set_ylabel(
        "Prediction Absolute Relative Error [%]"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "forecast_error_vs_global_step.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    print(
        "[PLOT DONE]",
        output_dir,
    )


if __name__ == "__main__":
    main()