from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from run.p3_common import write_csv
from run.p3_compare import (
    POLICY_LABELS,
    aggregate_distance,
    aggregate_points,
    aggregate_quality,
    aggregate_summaries,
    plot_location_distance,
    plot_overview,
    plot_trajectories,
)
from run.p3_eval_shard import (
    DEFAULT_BASELINE_POLICIES,
    EvalTask,
    artifact_paths,
    parse_list,
)


DEFAULT_POLICY_ORDER = (
    "dpp",
    "ppo_best",
    "ppo_latest",
    "rsu_only",
    "always_hire",
    "fixed_rsu",
    "nearest_hotspot",
    "load_threshold",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def relabel(rows: list[dict], policy: str) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item["policy"] = policy
        output.append(item)
    return output


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_tasks(
    seeds: tuple[int, ...],
    baseline_policies: tuple[str, ...],
) -> tuple[EvalTask, ...]:
    tasks = [
        EvalTask(group="baselines", policy=policy, seed=seed)
        for policy in baseline_policies
        for seed in seeds
    ]
    tasks.extend(
        EvalTask(group=group, policy="ppo", seed=seed)
        for group in ("ppo_best", "ppo_latest")
        for seed in seeds
    )
    return tuple(tasks)


def completed_payloads(
    root: Path,
    tasks: tuple[EvalTask, ...],
) -> tuple[list[dict], list[EvalTask]]:
    payloads = []
    incomplete = []
    for task in tasks:
        paths = artifact_paths(root / task.group, task)
        required = ("frames", "distance", "points", "quality", "users", "summary")
        if not all(paths[name].is_file() for name in required):
            incomplete.append(task)
            continue
        try:
            with paths["summary"].open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            incomplete.append(task)
            continue
        if payload.get("event") != "completed":
            incomplete.append(task)
            continue
        payloads.append(payload)
    return payloads, incomplete


def aggregate_marker_matches(
    path: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...],
    payloads: list[dict],
) -> bool:
    try:
        with path.open(encoding="utf-8") as stream:
            marker = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    fingerprints = [payload.get("fingerprint", {}) for payload in payloads]
    source_set = {item.get("source_sha256") for item in fingerprints}
    frames_set = {int(item.get("frames", -1)) for item in fingerprints}
    rollouts_set = {int(item.get("rollouts", -1)) for item in fingerprints}
    checkpoint_sha256 = {
        group: next(
            (
                item.get("checkpoint_sha256")
                for item in fingerprints
                if item.get("group") == group
            ),
            None,
        )
        for group in ("ppo_best", "ppo_latest")
    }
    return bool(
        marker.get("event") == "completed"
        and marker.get("seeds") == list(seeds)
        and marker.get("policies") == list(policies)
        and len(source_set) == 1
        and marker.get("source_sha256") == next(iter(source_set))
        and len(frames_set) == 1
        and marker.get("frames") == next(iter(frames_set))
        and len(rollouts_set) == 1
        and marker.get("rollouts") == next(iter(rollouts_set))
        and marker.get("checkpoint_sha256") == checkpoint_sha256
    )


def load_completed_task(
    root: Path,
    task: EvalTask,
) -> tuple[
    dict,
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    dict,
]:
    group_dir = root / task.group
    paths = artifact_paths(group_dir, task)
    required = ("frames", "distance", "points", "quality", "users", "summary")
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete task group={task.group} policy={task.policy} "
            f"seed={task.seed}; missing={missing}"
        )
    with paths["summary"].open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("event") != "completed":
        raise RuntimeError(f"task summary is not completed: {paths['summary']}")
    fingerprint = payload.get("fingerprint", {})
    if (
        fingerprint.get("group") != task.group
        or fingerprint.get("policy") != task.policy
        or int(fingerprint.get("seed", -1)) != task.seed
    ):
        raise RuntimeError(f"task fingerprint mismatch: {paths['summary']}")

    display_policy = task.group if task.policy == "ppo" else task.policy
    summary = dict(payload["summary"])
    summary["policy"] = display_policy
    summary["seed"] = task.seed
    return (
        summary,
        relabel(read_csv(paths["frames"]), display_policy),
        relabel(read_csv(paths["distance"]), display_policy),
        relabel(read_csv(paths["points"]), display_policy),
        relabel(read_csv(paths["quality"]), display_policy),
        relabel(read_csv(paths["users"]), display_policy),
        payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate completed P3 evaluation shards"
    )
    parser.add_argument(
        "--seeds",
        default=":".join(map(str, range(3026, 3036))),
        help="comma- or colon-separated evaluation seeds",
    )
    parser.add_argument(
        "--baseline-policies",
        default=":".join(DEFAULT_BASELINE_POLICIES),
    )
    parser.add_argument("--input", type=Path, default=Path("outputs/p3_eval_final"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--if-complete",
        action="store_true",
        help="return successfully without aggregation while shards are incomplete",
    )
    args = parser.parse_args()

    try:
        seeds = parse_list(args.seeds, int)
        baseline_policies = parse_list(args.baseline_policies, str)
    except ValueError as error:
        parser.error(str(error))
    input_root = args.input.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else input_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    available = set(baseline_policies) | {"ppo_best", "ppo_latest"}
    policies = tuple(policy for policy in DEFAULT_POLICY_ORDER if policy in available)
    policies += tuple(sorted(available - set(policies)))
    tasks = expected_tasks(seeds, baseline_policies)
    if args.if_complete:
        payloads, incomplete = completed_payloads(input_root, tasks)
        if incomplete:
            print(
                f"[AGGREGATE-DEFER] completed={len(payloads)}/{len(tasks)} "
                f"remaining={len(incomplete)}",
                flush=True,
            )
            return
        marker = output_dir / "experiment.json"
        if aggregate_marker_matches(marker, seeds, policies, payloads):
            print(f"[AGGREGATE-SKIP] matching result exists: {marker}", flush=True)
            return

    summaries: list[dict] = []
    distance_rows: list[dict] = []
    point_rows: list[dict] = []
    quality_rows: list[dict] = []
    user_rows: list[dict] = []
    frame_map: dict[tuple[str, int], list[dict]] = {}
    runtime_rows: list[dict] = []
    fingerprints: list[dict] = []
    configs: list[dict] = []

    for task in tasks:
        summary, frames, distance, points, quality, users, payload = load_completed_task(
            input_root,
            task,
        )
        display_policy = str(summary["policy"])
        summaries.append(summary)
        distance_rows.extend(distance)
        point_rows.extend(points)
        quality_rows.extend(quality)
        user_rows.extend(users)
        frame_map[(display_policy, task.seed)] = frames
        runtime_rows.append(
            {
                "group": task.group,
                "policy": display_policy,
                "seed": task.seed,
                "runtime_seconds": payload["runtime_seconds"],
                "selection_workers": payload.get("selection_workers", 1),
                "slurm_job_id": payload.get("slurm_job_id"),
                "slurm_array_task_id": payload.get("slurm_array_task_id"),
                "host": payload.get("host"),
            }
        )
        fingerprints.append(payload["fingerprint"])
        configs.append(payload["config"])

    frames_set = {int(item["frames"]) for item in fingerprints}
    rollouts_set = {int(item["rollouts"]) for item in fingerprints}
    source_set = {item.get("source_sha256") for item in fingerprints}
    if len(frames_set) != 1 or len(rollouts_set) != 1 or len(source_set) != 1:
        raise RuntimeError(
            "mixed experiment inputs: "
            f"frames={frames_set}, rollouts={rollouts_set}, sources={source_set}"
        )

    expected_pairs = {(policy, seed) for policy in policies for seed in seeds}
    actual_pairs = {(str(row["policy"]), int(row["seed"])) for row in summaries}
    if actual_pairs != expected_pairs:
        missing = sorted(expected_pairs - actual_pairs)
        extra = sorted(actual_pairs - expected_pairs)
        raise RuntimeError(f"incomplete result matrix: missing={missing}, extra={extra}")

    aggregate = aggregate_summaries(summaries, policies)
    distance_aggregate = aggregate_distance(distance_rows, policies)
    point_aggregate = aggregate_points(point_rows, policies)
    quality_aggregate = aggregate_quality(quality_rows, policies)
    checkpoint_sha256 = {
        group: next(
            (
                item.get("checkpoint_sha256")
                for item in fingerprints
                if item.get("group") == group
            ),
            None,
        )
        for group in ("ppo_best", "ppo_latest")
    }

    write_csv(output_dir / "seed_summaries.csv", summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "distance_by_policy.csv", distance_aggregate)
    write_csv(output_dir / "hover_point_by_policy.csv", point_aggregate)
    write_csv(output_dir / "quality_distribution.csv", quality_aggregate)
    write_csv(output_dir / "per_user_metrics.csv", user_rows)
    write_csv(output_dir / "task_runtimes.csv", runtime_rows)
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
    atomic_write_json(
        output_dir / "experiment.json",
        {
            "event": "completed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "formulation": "P3: one persistent multi-user UAV per RSU region",
            "seeds": list(seeds),
            "frames": next(iter(frames_set)),
            "rollouts": next(iter(rollouts_set)),
            "source_sha256": next(iter(source_set)),
            "checkpoint_sha256": checkpoint_sha256,
            "selection_workers": sorted(
                {int(row["selection_workers"]) for row in runtime_rows}
            ),
            "policies": list(policies),
            "labels": {policy: POLICY_LABELS.get(policy, policy) for policy in policies},
            "input_root": str(input_root),
            "config": configs[0],
            "aggregate": aggregate,
        },
    )
    print(
        f"[AGGREGATE-DONE] tasks={len(tasks)} policies={len(policies)} "
        f"seeds={len(seeds)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
