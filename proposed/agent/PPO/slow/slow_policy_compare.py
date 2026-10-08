from __future__ import annotations

import argparse
import csv
import json
import math
import statistics

from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_SEEDS = (
    2026,
    2027,
    2028,
)

PRIMARY_METRIC = (
    "realized_round_cost"
)

METRICS = (
    "realized_round_cost",
    "fast_cost",
    "hiring_cost",

    "stall",
    "scheduled_stall",
    "unscheduled_stall",

    "scheduled_user_slots",
    "unscheduled_user_slots",

    "scheduled_stall_rate",
    "unscheduled_stall_rate",

    "delivery",
    "delivery_per_scheduled_user_slot",

    "quality_per_chunk",
    "quality_degradation_per_chunk",

    "service_rate",
    "requested_chunks",

    "num_rsu_links",
    "num_hired_uav",
    "num_uav_links",

    "consumed_soc",
    "charged_soc",
    "charging_slots",
    "outage_slots",
    "min_soc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dpp-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--random-root",
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
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--rounds-per-episode",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--minimum-allowed-soc",
        type=float,
        default=19.95,
    )

    return parser.parse_args()


def read_csv(
    path: Path,
) -> List[Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    rows: List[
        Dict[str, float]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle
        )

        for row_idx, row in enumerate(
            reader,
            start=2,
        ):
            converted: Dict[
                str,
                float,
            ] = {}

            for key, value in row.items():
                if key is None:
                    continue

                try:
                    converted[
                        str(key)
                    ] = float(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    # String diagnostics such as solver_mode
                    # are irrelevant to numerical comparison.
                    continue

            rows.append(
                converted
            )

    if not rows:
        raise RuntimeError(
            f"Empty CSV: {path}"
        )

    return rows


def mean(
    values: Sequence[float],
) -> float:
    return statistics.fmean(
        float(value)
        for value in values
    )


def sample_std(
    values: Sequence[float],
) -> float:
    if len(values) <= 1:
        return 0.0

    return statistics.stdev(
        float(value)
        for value in values
    )


def t_critical_95(
    sample_size: int,
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

    if sample_size < 2:
        return float(
            "inf"
        )

    return table.get(
        sample_size,
        1.96,
    )


def mean_ci95(
    values: Sequence[float],
) -> tuple[
    float,
    float,
    float,
]:
    center = mean(
        values
    )

    if len(values) < 2:
        return (
            center,
            float("-inf"),
            float("inf"),
        )

    standard_error = (
        sample_std(values)
        / math.sqrt(
            len(values)
        )
    )

    margin = (
        t_critical_95(
            len(values)
        )
        * standard_error
    )

    return (
        center,
        center - margin,
        center + margin,
    )


def validate_rows(
    rows: Sequence[
        Dict[str, float]
    ],
    *,
    expected_rows: int,
    minimum_allowed_soc: float,
    label: str,
) -> None:
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"{label}: "
            f"expected {expected_rows} rows, "
            f"got {len(rows)}"
        )

    keys = [
        (
            int(row["episode"]),
            int(
                row[
                    "round_in_episode"
                ]
            ),
        )
        for row in rows
    ]

    if len(set(keys)) != len(keys):
        raise RuntimeError(
            f"{label}: duplicate episode/round keys"
        )

    for idx, row in enumerate(
        rows,
        start=1,
    ):
        for metric in METRICS:
            value = float(
                row[metric]
            )

            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    f"{label}: non-finite "
                    f"{metric} at row {idx}"
                )

        accounting_error = abs(
            float(
                row[
                    "realized_round_cost"
                ]
            )
            - (
                float(
                    row["fast_cost"]
                )
                + float(
                    row[
                        "hiring_cost"
                    ]
                )
            )
        )

        if accounting_error > 1e-5:
            raise RuntimeError(
                f"{label}: cost accounting "
                f"mismatch at row {idx}: "
                f"{accounting_error}"
            )

        expected_user_slots = (
            float(
                row[
                    "num_rsu_links"
                ]
            )
            + float(
                row[
                    "num_uav_links"
                ]
            )
        ) * 3600.0

        slot_error = abs(
            float(
                row[
                    "scheduled_user_slots"
                ]
            )
            - expected_user_slots
        )

        if slot_error > 1e-5:
            raise RuntimeError(
                f"{label}: scheduled slot "
                f"accounting mismatch "
                f"at row {idx}"
            )

        if (
            float(
                row[
                    "outage_slots"
                ]
            )
            != 0.0
        ):
            raise RuntimeError(
                f"{label}: outage "
                f"at row {idx}"
            )

        if (
            float(
                row["min_soc"]
            )
            < float(
                minimum_allowed_soc
            )
        ):
            raise RuntimeError(
                f"{label}: min SoC violation "
                f"at row {idx}: "
                f"{row['min_soc']}"
            )


def write_csv(
    path: Path,
    rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    str(key)
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def main() -> None:
    args = parse_args()

    dpp_root = (
        args.dpp_root
        .expanduser()
        .resolve()
    )

    random_root = (
        args.random_root
        .expanduser()
        .resolve()
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

    expected_rows = (
        int(args.episodes)
        * int(
            args.rounds_per_episode
        )
    )

    seed_rows: List[
        Dict[str, object]
    ] = []

    seed_deltas: Dict[
        str,
        List[float],
    ] = {
        metric: []
        for metric in METRICS
    }

    for seed in args.seeds:
        dpp_path = (
            dpp_root
            / f"seed{seed}"
            / "logs"
            / "eval_rounds.csv"
        )

        random_path = (
            random_root
            / f"seed{seed}"
            / "logs"
            / "eval_rounds.csv"
        )

        dpp_rows = read_csv(
            dpp_path
        )

        random_rows = read_csv(
            random_path
        )

        validate_rows(
            dpp_rows,
            expected_rows=expected_rows,
            minimum_allowed_soc=float(
                args.minimum_allowed_soc
            ),
            label=(
                f"DPP seed={seed}"
            ),
        )

        validate_rows(
            random_rows,
            expected_rows=expected_rows,
            minimum_allowed_soc=float(
                args.minimum_allowed_soc
            ),
            label=(
                f"Random seed={seed}"
            ),
        )

        dpp_keys = [
            (
                int(row["episode"]),
                int(
                    row[
                        "round_in_episode"
                    ]
                ),
            )
            for row in dpp_rows
        ]

        random_keys = [
            (
                int(row["episode"]),
                int(
                    row[
                        "round_in_episode"
                    ]
                ),
            )
            for row in random_rows
        ]

        if dpp_keys != random_keys:
            raise RuntimeError(
                f"seed={seed}: "
                "episode/round grids differ."
            )

        row: Dict[
            str,
            object,
        ] = {
            "seed": int(seed),
            "rounds": expected_rows,
        }

        for metric in METRICS:
            dpp_mean = mean(
                [
                    float(
                        item[metric]
                    )
                    for item
                    in dpp_rows
                ]
            )

            random_mean = mean(
                [
                    float(
                        item[metric]
                    )
                    for item
                    in random_rows
                ]
            )

            delta = (
                dpp_mean
                - random_mean
            )

            relative_delta = (
                delta
                / abs(
                    random_mean
                )
                if abs(
                    random_mean
                )
                > 1e-12
                else float("nan")
            )

            row[
                f"dpp_{metric}"
            ] = dpp_mean

            row[
                f"random_{metric}"
            ] = random_mean

            row[
                f"delta_{metric}"
            ] = delta

            row[
                f"relative_delta_{metric}"
            ] = relative_delta

            seed_deltas[
                metric
            ].append(
                delta
            )

        seed_rows.append(
            row
        )

    aggregate_rows: List[
        Dict[str, object]
    ] = []

    for metric in METRICS:
        deltas = seed_deltas[
            metric
        ]

        (
            center,
            low,
            high,
        ) = mean_ci95(
            deltas
        )

        aggregate_rows.append(
            {
                "metric": metric,
                "n_seeds": len(
                    deltas
                ),
                "paired_delta_mean":
                    center,
                "paired_delta_ci95_low":
                    low,
                "paired_delta_ci95_high":
                    high,
            }
        )

    primary = next(
        row
        for row in aggregate_rows
        if row["metric"]
        == PRIMARY_METRIC
    )

    primary_pass = (
        float(
            primary[
                "paired_delta_ci95_high"
            ]
        )
        < 0.0
    )

    decision = {
        "primary_metric":
            PRIMARY_METRIC,

        "pairing_unit":
            "seed_mean",

        "seeds": [
            int(seed)
            for seed
            in args.seeds
        ],

        "rounds_per_seed":
            expected_rows,

        "dpp_root":
            str(dpp_root),

        "random_root":
            str(random_root),

        "dpp_improves_primary_metric":
            bool(
                primary_pass
            ),

        "primary_delta_mean":
            float(
                primary[
                    "paired_delta_mean"
                ]
            ),

        "primary_delta_ci95_low":
            float(
                primary[
                    "paired_delta_ci95_low"
                ]
            ),

        "primary_delta_ci95_high":
            float(
                primary[
                    "paired_delta_ci95_high"
                ]
            ),
    }

    write_csv(
        output_dir
        / "seed_summary.csv",
        seed_rows,
    )

    write_csv(
        output_dir
        / "paired_deltas.csv",
        aggregate_rows,
    )

    (
        output_dir
        / "decision.json"
    ).write_text(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            decision,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()