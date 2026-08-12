from __future__ import annotations

import argparse
import csv

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slot-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=500,
    )

    return parser.parse_args()


def read_numeric_csv(
    path: Path,
) -> dict[
    str,
    np.ndarray
]:
    columns: dict[
        str,
        list[float]
    ] = {}

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row in reader:
            for key, value in row.items():
                try:
                    columns.setdefault(
                        key,
                        [],
                    ).append(
                        float(
                            value
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    return {
        key: np.asarray(
            values,
            dtype=np.float64,
        )
        for key, values in columns.items()
    }


def rolling(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        window <= 1
        or values.size < window
    ):
        return values.copy()

    kernel = np.ones(
        window,
        dtype=np.float64,
    ) / float(
        window
    )

    result = np.full(
        values.shape,
        np.nan,
        dtype=np.float64,
    )

    result[
        window - 1 :
    ] = np.convolve(
        values,
        kernel,
        mode="valid",
    )

    return result


def main() -> None:
    args = parse_args()

    data = read_numeric_csv(
        args.slot_csv
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    window = max(
        int(
            args.smooth_window
        ),
        1,
    )

    x = data[
        "global_step"
    ]

    # --------------------------------------------------------------
    # Reward terms
    # --------------------------------------------------------------

    reward_series = (
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

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(
            11,
            14,
        ),
        sharex=True,
    )

    for axis, (
        key,
        label,
    ) in zip(
        axes,
        reward_series,
    ):
        values = np.abs(
            data[key]
        )

        axis.plot(
            x,
            rolling(
                values,
                window,
            ),
            linewidth=1.2,
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

    fig.suptitle(
        f"Slot-level Weighted Reward Terms "
        f"({window}-slot moving mean)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "slot_reward_terms.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Queue
    # --------------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    for key, label in (
        (
            "queue_next_mean",
            "Mean Queue",
        ),
        (
            "queue_next_max",
            "Max Queue",
        ),
        (
            "virtual_queue_next_mean",
            "Mean Virtual Queue",
        ),
    ):
        axis.plot(
            x,
            rolling(
                data[key],
                window,
            ),
            label=label,
        )

    axis.set_xlabel(
        "Global Step"
    )

    axis.set_ylabel(
        "Queue"
    )

    axis.legend()
    axis.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "slot_queue.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Tx power
    # --------------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(
            11,
            5,
        )
    )

    for key, label in (
        (
            "uav_tx_power_attempt_sum_w",
            "Attempted UAV Tx Power",
        ),
        (
            "uav_tx_power_active_sum_w",
            "Delivered-link UAV Tx Power",
        ),
    ):
        axis.plot(
            x,
            rolling(
                data[key],
                window,
            ),
            label=label,
        )

    axis.set_xlabel(
        "Global Step"
    )

    axis.set_ylabel(
        "Power [W]"
    )

    axis.legend()
    axis.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "slot_uav_tx_power.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Data rate / throughput
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11,
            8,
        ),
        sharex=True,
    )

    axes[0].plot(
        x,
        rolling(
            data[
                "rsu_active_capacity_bps"
            ]
            / 1e6,
            window,
        ),
        label="RSU Capacity",
    )

    axes[0].plot(
        x,
        rolling(
            data[
                "uav_active_capacity_bps"
            ]
            / 1e6,
            window,
        ),
        label="UAV Capacity",
    )

    axes[0].set_ylabel(
        "Capacity [Mbps]"
    )

    axes[0].legend()
    axes[0].grid(
        True,
        alpha=0.3,
    )

    for key, label in (
        (
            "rsu_throughput_bps",
            "RSU Throughput",
        ),
        (
            "uav_throughput_bps",
            "UAV Throughput",
        ),
        (
            "total_throughput_bps",
            "Total Throughput",
        ),
    ):
        axes[1].plot(
            x,
            rolling(
                data[key]
                / 1e6,
                window,
            ),
            label=label,
        )

    axes[1].set_xlabel(
        "Global Step"
    )

    axes[1].set_ylabel(
        "Throughput [Mbps]"
    )

    axes[1].legend()
    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "slot_rate_and_throughput.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------------
    # Energy / SoC
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            11,
            8,
        ),
        sharex=True,
    )

    for key, label in (
        (
            "uav_hover_energy_j",
            "Hover Energy",
        ),
        (
            "uav_comm_energy_j",
            "Communication Energy",
        ),
        (
            "uav_consumed_energy_j",
            "Total Consumed Energy",
        ),
    ):
        axes[0].plot(
            x,
            rolling(
                data[key],
                window,
            ),
            label=label,
        )

    axes[0].set_ylabel(
        "Energy [J/slot]"
    )

    axes[0].legend()
    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        x,
        rolling(
            data[
                "soc_next_mean"
            ],
            window,
        ),
        label="Mean SoC",
    )

    axes[1].plot(
        x,
        rolling(
            data[
                "soc_next_min"
            ],
            window,
        ),
        label="Minimum SoC",
    )

    axes[1].set_xlabel(
        "Global Step"
    )

    axes[1].set_ylabel(
        "SoC"
    )

    axes[1].legend()
    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "slot_energy_and_soc.png",
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