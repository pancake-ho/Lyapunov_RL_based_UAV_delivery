from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from config_p3 import P3Config
from run.p3_common import run_policy


SCHEMA_VERSION = 1
DEFAULT_SNRS_DB = (20.0, 25.0, 30.0, 35.0, 40.0)
DEFAULT_SEEDS = (120026, 120027, 120028, 120029, 120030)
DEFAULT_POLICIES = ("proposed", "always_hire", "slow_ppo", "rsu_only")
RAW_POLICY = {
    "proposed": "dpp",
    "always_hire": "always_hire",
    "slow_ppo": "ppo",
    "rsu_only": "rsu_only",
}
SOURCE_FILES = (
    "config_p3.py",
    "agent/P3/exact_fast_controller.py",
    "agent/P3/features.py",
    "agent/P3/ppo_agent.py",
    "agent/P3/slow_rollout_controller.py",
    "env/p3/battery.py",
    "env/p3/environment.py",
    "env/p3/radio.py",
    "env/p3/topology.py",
    "env/p3/types.py",
    "run/p3_common.py",
    "run/p3_snr_sweep.py",
)


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


def snr_token(snr_db: float) -> str:
    value = float(snr_db)
    if abs(value - round(value)) <= 1e-9:
        return str(int(round(value)))
    return str(value).replace("-", "m").replace(".", "p")


def shifted_noise_psd_w_hz(
    base_noise_psd_w_hz: float,
    target_snr_db: float,
    baseline_snr_db: float,
) -> float:
    """Uniformly shift instantaneous link SNR through the common noise floor.

    The original P3 model has no single fixed transmit-SNR variable. Link SNR
    remains a derived value of transmit power, geometry/path loss, fading,
    bandwidth, Shannon gap and noise PSD.  We therefore keep the original
    radio equations and apply:
        N0(target) = N0(base) * 10^((baseline-target)/10).

    Setting target==baseline reproduces the original trained channel exactly.
    """
    if base_noise_psd_w_hz <= 0.0:
        raise ValueError("base noise PSD must be positive")
    scale = 10.0 ** ((float(baseline_snr_db) - float(target_snr_db)) / 10.0)
    return float(base_noise_psd_w_hz * scale)


def checkpoint_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def config_from_checkpoint(path: Path) -> P3Config:
    payload = checkpoint_payload(path)
    raw = dict(payload.get("config") or {})
    if not raw:
        raise RuntimeError(f"checkpoint has no saved config: {path}")
    names = {field.name for field in fields(P3Config)}
    raw = {key: value for key, value in raw.items() if key in names}
    for name in (
        "candidate_offsets_m",
        "quality_utility",
        "chunk_size_bits",
        "distance_bin_edges_m",
    ):
        if name in raw and isinstance(raw[name], list):
            raw[name] = tuple(raw[name])
    return P3Config(**raw)


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
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                json_safe(payload),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def average_video_bitrate_mbps(summary: dict, cfg: P3Config) -> float:
    denominator = float(summary["delivered_chunks_per_user_slot"]) * cfg.num_users
    if denominator <= 0.0:
        return 0.0
    return (
        float(summary["throughput_mbps"])
        * cfg.playback_chunks_per_slot
        / denominator
    )


def build_tasks(
    snrs_db: Sequence[float],
    seeds: Sequence[int],
    policies: Sequence[str],
) -> tuple[tuple[float, str, int], ...]:
    unknown = sorted(set(policies) - set(RAW_POLICY))
    if unknown:
        raise ValueError(f"unknown sweep policies: {unknown}")
    order = ("proposed", "always_hire", "slow_ppo", "rsu_only")
    requested = [policy for policy in order if policy in set(policies)]
    tasks: list[tuple[float, str, int]] = []
    for policy in requested:
        for snr_db in snrs_db:
            for seed in seeds:
                tasks.append((float(snr_db), policy, int(seed)))
    return tuple(tasks)


def assigned_indices(
    task_count: int,
    worker_index: int,
    worker_count: int,
) -> tuple[int, ...]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index is outside worker_count")
    return tuple(range(worker_index, task_count, worker_count))


def result_path(root: Path, snr_db: float, policy: str, seed: int) -> Path:
    return (
        root
        / "runs"
        / f"snr_{snr_token(snr_db)}"
        / policy
        / f"seed_{seed}.json"
    )


def task_fingerprint(
    *,
    snr_db: float,
    baseline_snr_db: float,
    seed: int,
    policy: str,
    frames: int,
    rollouts: int,
    checkpoint_sha: str,
    source_sha: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "snr_db": float(snr_db),
        "baseline_snr_db": float(baseline_snr_db),
        "seed": int(seed),
        "policy": str(policy),
        "frames": int(frames),
        "rollouts": int(rollouts),
        "checkpoint_sha256": checkpoint_sha,
        "source_sha256": source_sha,
    }


def is_complete(path: Path, fingerprint: dict) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("event") == "completed" and payload.get("fingerprint") == fingerprint


def run_one(
    args: argparse.Namespace,
    base_cfg: P3Config,
    checkpoint_sha: str,
    source_sha: str,
    snr_db: float,
    policy: str,
    seed: int,
) -> None:
    output = result_path(args.output.resolve(), snr_db, policy, seed)
    fingerprint = task_fingerprint(
        snr_db=snr_db,
        baseline_snr_db=args.baseline_snr_db,
        seed=seed,
        policy=policy,
        frames=args.frames,
        rollouts=args.rollouts,
        checkpoint_sha=checkpoint_sha,
        source_sha=source_sha,
    )
    if args.resume and is_complete(output, fingerprint):
        print(
            f"[SNR-SKIP] snr={snr_db:g} policy={policy} seed={seed} "
            "reason=matching-completed-result",
            flush=True,
        )
        return

    noise_psd = shifted_noise_psd_w_hz(
        base_cfg.noise_psd_w_hz,
        target_snr_db=snr_db,
        baseline_snr_db=args.baseline_snr_db,
    )
    cfg = replace(
        base_cfg,
        seed=int(seed),
        num_frames=int(args.frames),
        rollout_scenarios=int(args.rollouts),
        noise_psd_w_hz=float(noise_psd),
    )

    raw_policy = RAW_POLICY[policy]
    ppo_agent = None
    if policy == "slow_ppo":
        from agent.P3.ppo_agent import PPOAgent

        ppo_agent = PPOAgent(cfg, device=args.device)
        ppo_agent.load(args.best_checkpoint)

    started = time.perf_counter()
    print(
        f"[SNR-START] snr={snr_db:g}dB policy={policy} seed={seed} "
        f"frames={args.frames} rollouts={args.rollouts} "
        f"noise_psd={noise_psd:.6g} workers={args.selection_workers}",
        flush=True,
    )

    def progress(processed_frames: int, frame_row: dict) -> None:
        elapsed = time.perf_counter() - started
        eta = elapsed / max(processed_frames, 1) * (args.frames - processed_frames)
        print(
            f"[SNR-EVAL] snr={snr_db:g} policy={policy} seed={seed} "
            f"frame={processed_frames:04d}/{args.frames:04d} "
            f"selection={float(frame_row['selection_seconds']):.3f}s "
            f"evaluated={int(frame_row['evaluated_actions'])} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )

    result = run_policy(
        cfg,
        raw_policy,
        output.parent,
        ppo_agent=ppo_agent,
        rollout_workers=args.selection_workers,
        progress_interval_frames=args.progress_interval,
        progress_callback=progress,
        write_outputs=False,
        ppo_deterministic=True,
    )

    hard = {
        name: int(result.summary[name])
        for name in (
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        )
    }
    if any(hard.values()):
        raise RuntimeError(f"hard constraint violation at snr={snr_db}: {hard}")

    elapsed = time.perf_counter() - started
    summary = result.summary
    payload = {
        "event": "completed",
        "timestamp_utc": utc_now(),
        "fingerprint": fingerprint,
        "snr_db": float(snr_db),
        "baseline_snr_db": float(args.baseline_snr_db),
        "snr_shift_db": float(snr_db - args.baseline_snr_db),
        "base_noise_psd_w_hz": float(base_cfg.noise_psd_w_hz),
        "effective_noise_psd_w_hz": float(noise_psd),
        "policy": policy,
        "raw_policy": raw_policy,
        "seed": int(seed),
        "runtime_seconds": float(elapsed),
        "metrics": {
            "average_video_bitrate_mbps": average_video_bitrate_mbps(summary, cfg),
            "stall_ratio": float(summary["stall_ratio"]),
            "average_quality_utility": float(summary["average_quality_utility"]),
            "average_quality_level": float(summary["average_quality_level"]),
            "quality_p05_utility": float(summary["quality_p05_utility"]),
            "hire_rate": float(summary["hire_rate"]),
            "original_cost_per_user_slot": float(summary["original_cost_per_user_slot"]),
            "dpp_cost_per_user_slot": float(summary["dpp_cost_per_user_slot"]),
            "delivered_chunks_per_user_slot": float(summary["delivered_chunks_per_user_slot"]),
            "aggregate_network_throughput_mbps": float(summary["throughput_mbps"]),
            "mean_uav_user_distance_m": float(summary["mean_uav_user_distance_m"]),
            "energy_per_delivered_chunk_j": float(summary["energy_per_delivered_chunk_j"]),
            "max_queue": float(summary["max_queue"]),
            "large_queue_violation_rate": float(summary["large_queue_violation_rate"]),
            **hard,
        },
        "config": asdict(cfg),
    }
    atomic_write_json(output, payload)
    print(
        f"[SNR-DONE] snr={snr_db:g} policy={policy} seed={seed} "
        f"runtime={elapsed/60:.1f}m "
        f"bitrate={payload['metrics']['average_video_bitrate_mbps']:.4f}Mbps "
        f"stall={payload['metrics']['stall_ratio']:.6g} "
        f"quality={payload['metrics']['average_quality_utility']:.4f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3 nominal-SNR sensitivity sweep using the Stage-2 checkpoint"
    )
    parser.add_argument(
        "--snrs-db",
        default=":".join(str(int(value)) for value in DEFAULT_SNRS_DB),
    )
    parser.add_argument("--baseline-snr-db", type=float, default=30.0)
    parser.add_argument("--seeds", default=":".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--policies", default=":".join(DEFAULT_POLICIES))
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--selection-workers", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--best-checkpoint", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--print-task-count", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.frames <= 0 or args.rollouts <= 0 or args.selection_workers <= 0:
        parser.error("frames, rollouts, and selection-workers must be positive")
    if not args.best_checkpoint.is_file():
        parser.error(f"missing best checkpoint: {args.best_checkpoint}")

    try:
        snrs = parse_list(args.snrs_db, float)
        seeds = parse_list(args.seeds, int)
        policies = parse_list(args.policies, str)
        tasks = build_tasks(snrs, seeds, policies)
        indices = assigned_indices(len(tasks), args.worker_index, args.worker_count)
    except ValueError as error:
        parser.error(str(error))

    if args.print_task_count:
        print(len(tasks))
        return

    base_cfg = config_from_checkpoint(args.best_checkpoint)
    checkpoint_sha = sha256_file(args.best_checkpoint)
    project_dir = Path(__file__).resolve().parents[1]
    source_sha = source_sha256(project_dir)

    print(
        f"[SNR-SWEEP] tasks={len(tasks)} worker={args.worker_index}/{args.worker_count} "
        f"assigned={len(indices)} snrs={snrs} seeds={seeds} policies={policies}",
        flush=True,
    )
    print(
        f"[SNR-DEFINITION] baseline={args.baseline_snr_db:g}dB uses original "
        f"N0={base_cfg.noise_psd_w_hz:.6g}; target shifts all instantaneous "
        "link SNRs by target-baseline dB while preserving power, geometry, "
        "path loss and fading.",
        flush=True,
    )

    def terminate(signum, _frame) -> None:
        raise RuntimeError(f"received termination signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        for position, index in enumerate(indices, start=1):
            snr_db, policy, seed = tasks[index]
            print(
                f"[SNR-TASK] position={position}/{len(indices)} index={index} "
                f"snr={snr_db:g} policy={policy} seed={seed}",
                flush=True,
            )
            run_one(
                args,
                base_cfg,
                checkpoint_sha,
                source_sha,
                snr_db,
                policy,
                seed,
            )
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
