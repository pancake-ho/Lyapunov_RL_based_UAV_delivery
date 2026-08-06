from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_EPISODES: Tuple[int, ...] = (
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
)
DEFAULT_SEEDS: Tuple[int, ...] = (2026, 2027, 2028)

SUMMARY_METRICS: Tuple[str, ...] = (
    "fast_cost_per_scheduled_user_slot",
    "scheduled_stall_rate",
    "quality_per_chunk",
    "quality_degradation_per_chunk",
    "delivery_per_scheduled_user_slot",
    "service_rate",
    "requested_chunks",
    "outage_slots",
    "min_soc",
    "scheduled_user_slots",
    "unscheduled_user_slots",
    "hiring_cost",
)
WORKLOAD_KEYS: Tuple[str, ...] = (
    "scheduled_user_slots",
    "unscheduled_user_slots",
    "hiring_cost",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    episode: int
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Fast-PPO checkpoints with common seeds and select a "
            "checkpoint only when it beats the initial policy under paired "
            "deterministic evaluation."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing *_initial.pt and *_epN.pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for evaluation runs and aggregate reports.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPISODES),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--eval-rounds-per-episode",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minimum-allowed-soc",
        type=float,
        default=19.95,
        help=(
            "Feasibility threshold. The current e_min=20 and configured "
            "0.05 tolerance imply 19.95."
        ),
    )
    parser.add_argument(
        "--workload-relative-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum aggregate relative difference from the initial-policy "
            "run for each workload key and seed. Random Slow actions are "
            "state-dependent, so exact row equality is not required."
        ),
    )
    parser.add_argument(
        "--reuse-completed",
        action="store_true",
        help="Reuse an existing eval_episodes.csv instead of rerunning it.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    return parser.parse_args()


def _single_match(
    checkpoint_dir: Path,
    patterns: Sequence[str],
    description: str,
) -> Path:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(checkpoint_dir.glob(pattern))
    unique = sorted({path.resolve() for path in matches if path.is_file()})
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one {description}; found {len(unique)}: "
            f"{[str(path) for path in unique]}"
        )
    return unique[0]


def discover_checkpoints(
    checkpoint_dir: Path,
    episodes: Sequence[int],
) -> List[CheckpointSpec]:
    directory = checkpoint_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")

    initial = _single_match(
        directory,
        ("*_initial.pt",),
        "initial checkpoint",
    )
    checkpoints = [
        CheckpointSpec(
            label="initial",
            episode=0,
            path=initial,
        )
    ]

    for episode in episodes:
        episode_value = int(episode)
        if episode_value <= 0:
            raise ValueError("Checkpoint episodes must be positive.")
        path = _single_match(
            directory,
            (
                f"*_ep{episode_value}.pt",
                f"*ep{episode_value:03d}.pt",
            ),
            f"episode-{episode_value} checkpoint",
        )
        checkpoints.append(
            CheckpointSpec(
                label=f"ep{episode_value}",
                episode=episode_value,
                path=path,
            )
        )
    return checkpoints


def _read_float_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for row_index, row in enumerate(reader, start=2):
            converted: Dict[str, float] = {}
            for key, value in row.items():
                if key is None or value is None:
                    raise ValueError(
                        f"Malformed CSV field at {path}:{row_index}."
                    )
                try:
                    converted[str(key)] = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Non-numeric value at {path}:{row_index}, "
                        f"column={key!r}: {value!r}"
                    ) from exc
            rows.append(converted)
    if not rows:
        raise RuntimeError(f"Evaluation CSV is empty: {path}")
    return rows


def _validate_eval_rows(
    rows: Sequence[Mapping[str, float]],
    expected_episodes: int,
) -> None:
    if len(rows) != int(expected_episodes):
        raise RuntimeError(
            "Evaluation episode count mismatch: "
            f"expected={expected_episodes}, actual={len(rows)}"
        )
    expected_indices = list(range(1, int(expected_episodes) + 1))
    actual_indices = [int(row["episode"]) for row in rows]
    if actual_indices != expected_indices:
        raise RuntimeError(
            "Evaluation episode indices are not contiguous: "
            f"{actual_indices}"
        )

    required = set(SUMMARY_METRICS) | {"reward"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(
            "Evaluation CSV is missing required metrics: "
            f"{sorted(missing)}"
        )

    for row_index, row in enumerate(rows, start=1):
        for key in required:
            if not math.isfinite(float(row[key])):
                raise RuntimeError(
                    f"Non-finite evaluation metric: row={row_index}, "
                    f"key={key}, value={row[key]}"
                )


def _eval_run_dir(
    output_dir: Path,
    checkpoint: CheckpointSpec,
    seed: int,
) -> Path:
    return (
        output_dir
        / "runs"
        / checkpoint.label
        / f"seed{int(seed)}"
    )


def run_one_evaluation(
    *,
    project_root: Path,
    python_bin: Path,
    output_dir: Path,
    checkpoint: CheckpointSpec,
    seed: int,
    eval_episodes: int,
    eval_rounds_per_episode: int,
    reuse_completed: bool,
) -> List[Dict[str, float]]:
    run_dir = _eval_run_dir(output_dir, checkpoint, seed)
    csv_path = run_dir / "logs" / "eval_episodes.csv"
    if reuse_completed and csv_path.is_file():
        rows = _read_float_csv(csv_path)
        _validate_eval_rows(rows, eval_episodes)
        return rows
    if run_dir.exists():
        raise RuntimeError(
            f"Evaluation run directory already exists: {run_dir}. "
            "Use a new --output-dir or --reuse-completed."
        )

    console_dir = output_dir / "console"
    console_dir.mkdir(parents=True, exist_ok=True)
    console_path = (
        console_dir / f"{checkpoint.label}_seed{int(seed)}.log"
    )

    environment = os.environ.copy()
    for key in (
        "FAST_PRETRAIN_RESUME_CHECKPOINT",
        "JOINT_RESUME_CHECKPOINT",
        "JOINT_FAST_CHECKPOINT",
        "FAST_PPO_RESUME",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "FAST_PPO_PHASE": "eval_pretrain",
            "FAST_PPO_CHECKPOINT": str(checkpoint.path),
            "FAST_PPO_SEED": str(int(seed)),
            "FAST_PPO_EVAL_EPISODES": str(int(eval_episodes)),
            "FAST_PPO_EVAL_ROUNDS_PER_EPISODE": str(
                int(eval_rounds_per_episode)
            ),
            "FAST_PPO_OUTPUT_ROOT": str(
                (output_dir / "runs").resolve()
            ),
            "FAST_PPO_RUN_NAME": (
                f"{checkpoint.label}/seed{int(seed)}"
            ),
            "FAST_PPO_DEVICE": environment.get(
                "FAST_PPO_DEVICE",
                "cuda",
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )

    command = [
        str(python_bin),
        "-u",
        "-m",
        "agent.PPO.fast.fast_train",
    ]
    with console_path.open("w", encoding="utf-8") as console:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=console,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for {checkpoint.label}, seed={seed}; "
            f"exit={completed.returncode}, log={console_path}"
        )
    rows = _read_float_csv(csv_path)
    _validate_eval_rows(rows, eval_episodes)
    return rows


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return statistics.fmean(data) if data else float("nan")


def _sample_std(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return statistics.stdev(data) if len(data) > 1 else 0.0


def _t_critical_95(sample_size: int) -> float:
    # Two-sided 95% Student-t critical values for common small samples.
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
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if sample_size < 2:
        return float("inf")
    if sample_size in table:
        return table[sample_size]
    return 1.96


def _mean_ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    mean_value = _mean(values)
    if len(values) < 2:
        return mean_value, float("-inf"), float("inf")
    standard_error = _sample_std(values) / math.sqrt(len(values))
    margin = _t_critical_95(len(values)) * standard_error
    return mean_value, mean_value - margin, mean_value + margin


def _workload_matches(
    reference: Sequence[Mapping[str, float]],
    candidate: Sequence[Mapping[str, float]],
    *,
    relative_tolerance: float,
    absolute_tolerance: float = 1e-9,
) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if len(reference) != len(candidate):
        return False, ["row-count mismatch"]
    for key in WORKLOAD_KEYS:
        reference_total = sum(float(row[key]) for row in reference)
        candidate_total = sum(float(row[key]) for row in candidate)
        difference = candidate_total - reference_total
        denominator = max(abs(reference_total), absolute_tolerance)
        relative_difference = abs(difference) / denominator
        if (
            abs(difference) > float(absolute_tolerance)
            and relative_difference > float(relative_tolerance)
        ):
            issues.append(
                f"key={key}, reference={reference_total}, "
                f"candidate={candidate_total}, "
                f"relative_delta={relative_difference}"
            )
    return not issues, issues


def aggregate_results(
    *,
    checkpoints: Sequence[CheckpointSpec],
    seeds: Sequence[int],
    results: Mapping[Tuple[str, int], Sequence[Mapping[str, float]]],
    minimum_allowed_soc: float,
    workload_relative_tolerance: float,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    Dict[str, object],
]:
    initial = next(
        checkpoint for checkpoint in checkpoints
        if checkpoint.episode == 0
    )
    summary_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []

    flattened: Dict[str, List[Mapping[str, float]]] = {}
    workload_status: Dict[str, bool] = {}
    workload_issues: Dict[str, List[str]] = {}
    workload_issue_counts: Dict[str, int] = {}

    for checkpoint in checkpoints:
        checkpoint_rows: List[Mapping[str, float]] = []
        matched = True
        issues: List[str] = []
        for seed in seeds:
            current = list(results[(checkpoint.label, int(seed))])
            checkpoint_rows.extend(current)
            if checkpoint.episode != 0:
                reference = list(results[(initial.label, int(seed))])
                seed_matched, seed_issues = _workload_matches(
                    reference,
                    current,
                    relative_tolerance=float(
                        workload_relative_tolerance
                    ),
                )
                matched = matched and seed_matched
                issues.extend(
                    f"seed={seed}: {issue}" for issue in seed_issues
                )
        flattened[checkpoint.label] = checkpoint_rows
        workload_status[checkpoint.label] = matched
        workload_issue_counts[checkpoint.label] = len(issues)
        workload_issues[checkpoint.label] = issues[:20]

    for checkpoint in checkpoints:
        rows = flattened[checkpoint.label]
        outage_total = sum(float(row["outage_slots"]) for row in rows)
        min_soc = min(float(row["min_soc"]) for row in rows)
        feasible = (
            outage_total == 0.0
            and min_soc >= float(minimum_allowed_soc)
        )
        summary: Dict[str, object] = {
            "label": checkpoint.label,
            "episode": checkpoint.episode,
            "checkpoint": str(checkpoint.path),
            "n": len(rows),
            "workload_matched": workload_status[checkpoint.label],
            "workload_issue_count": workload_issue_counts[
                checkpoint.label
            ],
            "feasible": feasible,
            "outage_total": outage_total,
            "min_soc_min": min_soc,
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _sample_std(values)
        summary_rows.append(summary)

        if checkpoint.episode == 0:
            continue
        delta: Dict[str, object] = {
            "label": checkpoint.label,
            "episode": checkpoint.episode,
            "checkpoint": str(checkpoint.path),
            "n_pairs": len(seeds),
            "pairing_unit": "seed_mean",
            "workload_matched": workload_status[checkpoint.label],
            "feasible": feasible,
        }
        for metric in (
            "fast_cost_per_scheduled_user_slot",
            "scheduled_stall_rate",
            "quality_per_chunk",
            "quality_degradation_per_chunk",
            "delivery_per_scheduled_user_slot",
            "service_rate",
        ):
            # Seeds, not episodes nested within a seed, are the independent
            # experimental units. Using all episode rows as independent pairs
            # would understate uncertainty (pseudo-replication).
            differences = []
            for seed in seeds:
                current_seed_rows = results[
                    (checkpoint.label, int(seed))
                ]
                initial_seed_rows = results[
                    (initial.label, int(seed))
                ]
                differences.append(
                    _mean(
                        float(row[metric])
                        for row in current_seed_rows
                    )
                    - _mean(
                        float(row[metric])
                        for row in initial_seed_rows
                    )
                )
            mean_delta, lower, upper = _mean_ci95(differences)
            delta[f"{metric}_delta_mean"] = mean_delta
            delta[f"{metric}_delta_ci95_low"] = lower
            delta[f"{metric}_delta_ci95_high"] = upper
        delta_rows.append(delta)

    feasible_candidates = [
        row for row in summary_rows
        if int(row["episode"]) > 0
        and bool(row["feasible"])
        and bool(row["workload_matched"])
    ]
    best_mean = (
        min(
            feasible_candidates,
            key=lambda row: float(
                row[
                    "fast_cost_per_scheduled_user_slot_mean"
                ]
            ),
        )
        if feasible_candidates
        else None
    )

    qualified_labels = set()
    for row in delta_rows:
        if (
            bool(row["feasible"])
            and bool(row["workload_matched"])
            and float(
                row[
                    "fast_cost_per_scheduled_user_slot_delta_ci95_high"
                ]
            )
            < 0.0
        ):
            qualified_labels.add(str(row["label"]))
    qualified = [
        row for row in feasible_candidates
        if str(row["label"]) in qualified_labels
    ]
    selected = (
        min(
            qualified,
            key=lambda row: float(
                row[
                    "fast_cost_per_scheduled_user_slot_mean"
                ]
            ),
        )
        if qualified
        else None
    )

    decision = {
        "selection_rule": (
            "Feasible and workload-matched checkpoint with the lowest mean "
            "Fast cost per scheduled-user-slot, provided its paired 95% CI "
            "across seed-level means against the initial policy is entirely "
            "below zero."
        ),
        "selected_checkpoint": (
            str(selected["checkpoint"]) if selected is not None else None
        ),
        "selected_label": (
            str(selected["label"]) if selected is not None else None
        ),
        "best_mean_checkpoint": (
            str(best_mean["checkpoint"]) if best_mean is not None else None
        ),
        "best_mean_label": (
            str(best_mean["label"]) if best_mean is not None else None
        ),
        "qualified_labels": sorted(qualified_labels),
        "slow_dpp_gate_passed": selected is not None,
        "minimum_allowed_soc": float(minimum_allowed_soc),
        "workload_relative_tolerance": float(
            workload_relative_tolerance
        ),
        "workload_issues": workload_issues,
    }
    return summary_rows, delta_rows, decision


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.6g}"
    return str(value)


def _write_markdown_report(
    path: Path,
    summary_rows: Sequence[Mapping[str, object]],
    delta_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
) -> None:
    lines = [
        "# Fast-PPO checkpoint sweep",
        "",
        "## Decision",
        "",
        (
            "- Slow DPP gate: **PASS**"
            if bool(decision["slow_dpp_gate_passed"])
            else "- Slow DPP gate: **FAIL**"
        ),
        f"- Selected: `{decision['selected_checkpoint']}`",
        f"- Best mean only: `{decision['best_mean_checkpoint']}`",
        "",
        "A checkpoint is selected only when it is feasible, the workload is "
        "within tolerance, and the seed-level paired 95% confidence interval "
        "for Fast cost per scheduled-user-slot is entirely below zero.",
        "",
        "## Aggregate metrics",
        "",
        "| checkpoint | feasible | workload | cost/scheduled slot | "
        "scheduled stall | quality/chunk | degradation/chunk | service |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {feasible} | {workload} | {cost} | {stall} | "
            "{quality} | {degradation} | {service} |".format(
                label=row["label"],
                feasible=row["feasible"],
                workload=row["workload_matched"],
                cost=_format_number(
                    row[
                        "fast_cost_per_scheduled_user_slot_mean"
                    ]
                ),
                stall=_format_number(
                    row["scheduled_stall_rate_mean"]
                ),
                quality=_format_number(
                    row["quality_per_chunk_mean"]
                ),
                degradation=_format_number(
                    row[
                        "quality_degradation_per_chunk_mean"
                    ]
                ),
                service=_format_number(row["service_rate_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired change versus initial",
            "",
            "| checkpoint | cost delta | 95% CI low | 95% CI high | "
            "stall delta | quality delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| {label} | {delta} | {low} | {high} | {stall} | "
            "{quality} |".format(
                label=row["label"],
                delta=_format_number(
                    row[
                        "fast_cost_per_scheduled_user_slot_delta_mean"
                    ]
                ),
                low=_format_number(
                    row[
                        "fast_cost_per_scheduled_user_slot_delta_ci95_low"
                    ]
                ),
                high=_format_number(
                    row[
                        "fast_cost_per_scheduled_user_slot_delta_ci95_high"
                    ]
                ),
                stall=_format_number(
                    row["scheduled_stall_rate_delta_mean"]
                ),
                quality=_format_number(
                    row["quality_per_chunk_delta_mean"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Rule",
            "",
            str(decision["selection_rule"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.eval_episodes <= 0 or args.eval_rounds_per_episode <= 0:
        raise ValueError("Evaluation horizons must be positive.")
    if not 0.0 <= args.workload_relative_tolerance <= 1.0:
        raise ValueError(
            "workload-relative-tolerance must be in [0, 1]."
        )
    if len(args.seeds) < 2:
        raise ValueError(
            "At least two evaluation seeds are required for a confidence "
            "interval. Three or more are recommended."
        )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Evaluation seeds must be unique.")

    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(
        args.checkpoint_dir,
        args.episodes,
    )

    results: Dict[
        Tuple[str, int],
        Sequence[Mapping[str, float]],
    ] = {}
    for checkpoint in checkpoints:
        for seed in args.seeds:
            print(
                "[SWEEP] "
                f"checkpoint={checkpoint.label} seed={seed}",
                flush=True,
            )
            rows = run_one_evaluation(
                project_root=project_root,
                python_bin=args.python.expanduser().resolve(),
                output_dir=output_dir,
                checkpoint=checkpoint,
                seed=int(seed),
                eval_episodes=int(args.eval_episodes),
                eval_rounds_per_episode=int(
                    args.eval_rounds_per_episode
                ),
                reuse_completed=bool(args.reuse_completed),
            )
            results[(checkpoint.label, int(seed))] = rows

    summary_rows, delta_rows, decision = aggregate_results(
        checkpoints=checkpoints,
        seeds=args.seeds,
        results=results,
        minimum_allowed_soc=float(args.minimum_allowed_soc),
        workload_relative_tolerance=float(
            args.workload_relative_tolerance
        ),
    )
    _write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_deltas.csv", delta_rows)
    (output_dir / "selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_report(
        output_dir / "CHECKPOINT_SWEEP_REPORT.md",
        summary_rows,
        delta_rows,
        decision,
    )

    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    if not bool(decision["slow_dpp_gate_passed"]):
        print(
            "[GATE FAIL] No checkpoint has proven improvement over the "
            "initial policy. Do not connect H1 to Slow DPP yet.",
            flush=True,
        )
    else:
        print(
            "[GATE PASS] Use only selected_checkpoint with its saved "
            "observation normalizer for the frozen-Fast Slow-DPP test.",
            flush=True,
        )


if __name__ == "__main__":
    main()
