from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from agent.P3.slow_rollout_controller import SUPPORTED_RULE_POLICIES
from config_p3 import P3Config
from run.p3_common import PolicyRunResult, run_policy


SCHEMA_VERSION = 2
DEFAULT_SEEDS = tuple(range(3026, 3036))
# Expensive policies first keeps the tail of a bounded Slurm array short.
DEFAULT_BASELINE_POLICIES = (
    "dpp",
    "load_threshold",
    "always_hire",
    "fixed_rsu",
    "nearest_hotspot",
    "rsu_only",
)
PPO_GROUPS = ("ppo_best", "ppo_latest")
SOURCE_FILES = (
    "config_p3.py",
    "agent/P3/exact_fast_controller.py",
    "agent/P3/features.py",
    "agent/P3/ppo_agent.py",
    "agent/P3/slow_rollout_controller.py",
    "env/p3/battery.py",
    "env/p3/environment.py",
    "env/p3/topology.py",
    "env/p3/types.py",
    "run/p3_common.py",
    "run/p3_eval_shard.py",
)


@dataclass(frozen=True)
class EvalTask:
    group: str
    policy: str
    seed: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(text: str, cast) -> tuple:
    normalized = text.replace(":", ",")
    values = tuple(cast(item.strip()) for item in normalized.split(",") if item.strip())
    if not values:
        raise ValueError("at least one value is required")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate values are not allowed: {values}")
    return values


def build_eval_tasks(
    seeds: Sequence[int],
    baseline_policies: Sequence[str],
) -> tuple[EvalTask, ...]:
    seeds = tuple(int(seed) for seed in seeds)
    policies = tuple(str(policy) for policy in baseline_policies)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    if len(set(policies)) != len(policies):
        raise ValueError("baseline policies must be unique")
    unknown = sorted(set(policies) - set(SUPPORTED_RULE_POLICIES))
    if unknown:
        raise ValueError(f"unknown baseline policies: {unknown}")
    if "ppo" in policies:
        raise ValueError("PPO checkpoints are added separately; remove ppo from baselines")

    tasks = [
        EvalTask(group="baselines", policy=policy, seed=seed)
        for policy in policies
        for seed in seeds
    ]
    tasks.extend(
        EvalTask(group=group, policy="ppo", seed=seed)
        for group in PPO_GROUPS
        for seed in seeds
    )
    return tuple(tasks)


def assigned_task_indices(
    task_count: int,
    worker_index: int,
    worker_count: int,
) -> tuple[int, ...]:
    """Return a deterministic, balanced strided assignment for one worker."""

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError(
            f"worker_index must be in [0, {worker_count - 1}], got {worker_index}"
        )
    return tuple(range(worker_index, task_count, worker_count))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(project_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = project_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source file: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(json_safe(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
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


def artifact_paths(group_dir: Path, task: EvalTask) -> dict[str, Path]:
    suffix = f"{task.policy}_seed{task.seed}"
    return {
        "frames": group_dir / f"frames_{suffix}.csv",
        "distance": group_dir / f"distance_{suffix}.csv",
        "points": group_dir / f"points_{suffix}.csv",
        "quality": group_dir / f"quality_{suffix}.csv",
        "summary": group_dir / "summaries" / f"summary_{suffix}.json",
        "status": group_dir / "status" / f"status_{suffix}.json",
    }


def task_fingerprint(
    task: EvalTask,
    frames: int,
    rollouts: int,
    checkpoint_sha256: str | None,
    source_digest: str | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "group": task.group,
        "policy": task.policy,
        "seed": task.seed,
        "frames": int(frames),
        "rollouts": int(rollouts),
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_digest,
    }


def is_complete(paths: dict[str, Path], fingerprint: dict) -> bool:
    required = ("frames", "distance", "points", "quality", "summary")
    if not all(paths[name].is_file() and paths[name].stat().st_size > 0 for name in required):
        return False
    try:
        with paths["summary"].open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("event") == "completed" and payload.get("fingerprint") == fingerprint


def write_result_artifacts(
    result: PolicyRunResult,
    paths: dict[str, Path],
) -> None:
    atomic_write_csv(paths["frames"], result.frame_rows)
    atomic_write_csv(paths["distance"], result.distance_rows)
    atomic_write_csv(paths["points"], result.point_rows)
    atomic_write_csv(paths["quality"], result.quality_rows)


def checkpoint_for_task(
    task: EvalTask,
    best_checkpoint: Path | None,
    latest_checkpoint: Path | None,
) -> Path | None:
    if task.group == "ppo_best":
        return best_checkpoint
    if task.group == "ppo_latest":
        return latest_checkpoint
    return None


def run_task(args: argparse.Namespace, task: EvalTask) -> int:
    checkpoint = checkpoint_for_task(
        task,
        args.best_checkpoint,
        args.latest_checkpoint,
    )
    checkpoint_digest = None
    if task.policy == "ppo":
        if checkpoint is None:
            raise ValueError(f"{task.group} requires its PPO checkpoint")
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing PPO checkpoint: {checkpoint}")
        checkpoint_digest = sha256_file(checkpoint)

    group_dir = args.output.resolve() / task.group
    group_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(group_dir, task)
    project_dir = Path(__file__).resolve().parents[1]
    fingerprint = task_fingerprint(
        task,
        args.frames,
        args.rollouts,
        checkpoint_digest,
        source_sha256(project_dir),
    )
    if args.resume and is_complete(paths, fingerprint):
        print(
            f"[EVAL-SKIP] group={task.group} policy={task.policy} seed={task.seed} "
            f"reason=matching-completed-artifacts",
            flush=True,
        )
        return 0

    started = time.perf_counter()
    status_base = {
        "fingerprint": fingerprint,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "selection_workers": args.selection_workers,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": os.uname().nodename,
    }
    atomic_write_json(
        paths["status"],
        {"event": "started", "timestamp_utc": utc_now(), **status_base},
    )
    print(
        f"[EVAL-START] group={task.group} policy={task.policy} seed={task.seed} "
        f"frames={args.frames} rollouts={args.rollouts} "
        f"selection_workers={args.selection_workers} device={args.device}",
        flush=True,
    )

    def on_progress(processed_frames: int, frame_row: dict) -> None:
        elapsed = time.perf_counter() - started
        seconds_per_frame = elapsed / max(processed_frames, 1)
        eta = seconds_per_frame * (args.frames - processed_frames)
        payload = {
            "event": "progress",
            "timestamp_utc": utc_now(),
            **status_base,
            "processed_frames": processed_frames,
            "total_frames": args.frames,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "last_frame_selection_seconds": frame_row["selection_seconds"],
            "last_frame_enumerated_actions": frame_row["enumerated_actions"],
            "last_frame_evaluated_actions": frame_row["evaluated_actions"],
        }
        atomic_write_json(paths["status"], payload)
        print(
            f"[EVAL] group={task.group} policy={task.policy} seed={task.seed} "
            f"frame={processed_frames:04d}/{args.frames:04d} "
            f"selection={float(frame_row['selection_seconds']):.3f}s "
            f"evaluated={int(frame_row['evaluated_actions'])} "
            f"elapsed={elapsed / 60.0:.1f}m eta={eta / 60.0:.1f}m",
            flush=True,
        )

    def terminate(signum, _frame) -> None:
        raise RuntimeError(f"received termination signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        cfg = P3Config(
            seed=task.seed,
            num_frames=args.frames,
            rollout_scenarios=args.rollouts,
        )
        ppo_agent = None
        if task.policy == "ppo":
            from agent.P3.ppo_agent import PPOAgent

            ppo_agent = PPOAgent(cfg, device=args.device)
            ppo_agent.load(checkpoint)
        result = run_policy(
            cfg,
            task.policy,
            group_dir,
            ppo_agent=ppo_agent,
            rollout_workers=args.selection_workers,
            progress_interval_frames=args.progress_interval,
            progress_callback=on_progress,
            write_outputs=False,
        )
        hard_violations = sum(
            int(result.summary[name])
            for name in (
                "battery_reserve_violations",
                "power_violations",
                "provider_violations",
            )
        )
        if hard_violations and args.fail_on_hard_violation:
            raise RuntimeError(
                "hard constraint violation: "
                f"battery={result.summary['battery_reserve_violations']} "
                f"power={result.summary['power_violations']} "
                f"provider={result.summary['provider_violations']}"
            )
        write_result_artifacts(result, paths)
        elapsed = time.perf_counter() - started
        completion = {
            "event": "completed",
            "timestamp_utc": utc_now(),
            **status_base,
            "runtime_seconds": elapsed,
            "fingerprint": fingerprint,
            "config": asdict(cfg),
            "summary": result.summary,
            "artifacts": {
                name: str(path)
                for name, path in paths.items()
                if name not in ("summary", "status")
            },
        }
        atomic_write_json(paths["summary"], completion)
        atomic_write_json(paths["status"], completion)
        print(
            f"[EVAL-DONE] group={task.group} policy={task.policy} seed={task.seed} "
            f"runtime={elapsed / 60.0:.1f}m "
            f"dpp={float(result.summary['dpp_cost_per_user_slot']):.6f} "
            f"stall={float(result.summary['stall_ratio']):.4f}",
            flush=True,
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            paths["status"],
            {
                "event": "failed",
                "timestamp_utc": utc_now(),
                **status_base,
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one restartable P3 evaluation task from a Slurm array"
    )
    parser.add_argument(
        "--seeds",
        default=":".join(map(str, DEFAULT_SEEDS)),
        help="comma- or colon-separated evaluation seeds",
    )
    parser.add_argument(
        "--baseline-policies",
        default=":".join(DEFAULT_BASELINE_POLICIES),
        help="comma- or colon-separated non-PPO policies",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task-index", type=int)
    task_group.add_argument(
        "--worker-index",
        type=int,
        help="run this worker's strided share of the complete task matrix",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="number of long-lived Slurm workers sharing the task matrix",
    )
    parser.add_argument("--print-task-count", action="store_true")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--best-checkpoint", type=Path)
    parser.add_argument("--latest-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--selection-workers",
        type=int,
        default=4,
        help="CPU processes used inside one slow-action rollout task",
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-hard-violation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/p3_eval_final"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        seeds = parse_list(args.seeds, int)
        baseline_policies = parse_list(args.baseline_policies, str)
        tasks = build_eval_tasks(seeds, baseline_policies)
    except ValueError as error:
        parser.error(str(error))
    if args.print_task_count:
        print(len(tasks))
        return
    if (
        args.frames <= 0
        or args.rollouts <= 0
        or args.progress_interval <= 0
        or args.selection_workers <= 0
    ):
        parser.error(
            "frames, rollouts, progress-interval, and selection-workers must be positive"
        )
    if args.worker_index is not None:
        try:
            task_indices = assigned_task_indices(
                len(tasks),
                args.worker_index,
                args.worker_count,
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        task_index = 0 if args.task_index is None else args.task_index
        if not 0 <= task_index < len(tasks):
            parser.error(f"task-index must be in [0, {len(tasks) - 1}]")
        task_indices = (task_index,)

    for position, task_index in enumerate(task_indices, start=1):
        task = tasks[task_index]
        print(
            f"[WORKER-TASK] worker={args.worker_index} "
            f"position={position}/{len(task_indices)} task_index={task_index} "
            f"group={task.group} policy={task.policy} seed={task.seed}",
            flush=True,
        )
        exit_code = run_task(args, task)
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()