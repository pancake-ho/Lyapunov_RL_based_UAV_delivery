from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.P3.ppo_agent import PPOAgent
from config_p3 import P3Config
from run.p3_common import run_policy
from run.p3_train_ppo import parse_seed_list


_TUPLE_CONFIG_FIELDS = {
    "candidate_offsets_m",
    "quality_utility",
    "chunk_size_bits",
    "distance_bin_edges_m",
}


def json_safe(value: Any) -> Any:
    """Convert a payload into strict-JSON-safe Python objects.

    Non-finite floating-point values are converted to ``None`` only for
    serialization. Runtime metrics are checked separately and still fail the
    acceptance gate when they contain NaN/Inf.
    """
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    safe_payload = json_safe(payload)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                safe_payload,
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


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def config_from_checkpoint(
    checkpoint_path: Path,
    frames: int,
    device: str,
) -> P3Config:
    """Reconstruct the evaluation config from the training checkpoint.

    The old acceptance code silently recreated ``P3Config`` from defaults,
    which can evaluate a checkpoint under a different environment after config
    changes. The checkpoint already stores the exact training config, so that
    config is authoritative. Only the evaluation horizon is overridden.
    """
    checkpoint = _torch_load(checkpoint_path, device)
    raw = checkpoint.get("config")
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"checkpoint has no usable config dictionary: {checkpoint_path}"
        )

    allowed = {field.name for field in fields(P3Config)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RuntimeError(f"checkpoint config contains unknown fields: {unknown}")

    normalized = dict(raw)
    for name in _TUPLE_CONFIG_FIELDS:
        if name in normalized and not isinstance(normalized[name], tuple):
            normalized[name] = tuple(normalized[name])

    cfg = P3Config(**normalized)
    return replace(cfg, num_frames=int(frames))


def nonfinite_metric_paths(rows: list[dict], label: str) -> list[str]:
    invalid: list[str] = []
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float, np.integer, np.floating)):
                if not math.isfinite(float(value)):
                    invalid.append(f"{label}[{row_index}].{key}={value!r}")
    return invalid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast acceptance gate for Changed_Form P3 slow PPO"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("93026,93027,93028"),
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--min-hire-rate", type=float, default=0.01)
    parser.add_argument("--max-hire-rate", type=float, default=0.99)
    parser.add_argument("--min-stall-improvement", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.frames <= 0:
        parser.error("frames must be positive")
    if not 0.0 <= args.min_hire_rate < args.max_hire_rate <= 1.0:
        parser.error("hire-rate gate must satisfy 0 <= min < max <= 1")

    checkpoint_path = args.checkpoint.resolve()
    base_cfg = config_from_checkpoint(
        checkpoint_path=checkpoint_path,
        frames=args.frames,
        device=args.device,
    )
    agent = PPOAgent(base_cfg, device=args.device)
    metadata = agent.load(checkpoint_path)

    ppo_rows: list[dict] = []
    rsu_rows: list[dict] = []
    for seed in args.seeds:
        cfg = replace(base_cfg, seed=int(seed))
        ppo_rows.append(
            run_policy(
                cfg,
                "ppo",
                args.output.parent,
                ppo_agent=agent,
                write_outputs=False,
                ppo_deterministic=True,
            ).summary
        )
        rsu_rows.append(
            run_policy(
                cfg,
                "rsu_only",
                args.output.parent,
                write_outputs=False,
            ).summary
        )

    invalid_metrics = [
        *nonfinite_metric_paths(ppo_rows, "ppo"),
        *nonfinite_metric_paths(rsu_rows, "rsu_only"),
    ]

    def mean(rows: list[dict], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    ppo_stall = mean(ppo_rows, "stall_ratio")
    rsu_stall = mean(rsu_rows, "stall_ratio")
    hire_rate = mean(ppo_rows, "hire_rate")
    hard_violations = sum(
        int(row[name])
        for row in ppo_rows
        for name in (
            "battery_reserve_violations",
            "power_violations",
            "provider_violations",
        )
    )

    checks = {
        "finite_metrics": len(invalid_metrics) == 0,
        "hard_constraints": hard_violations == 0,
        "noncollapsed_hiring": (
            args.min_hire_rate <= hire_rate <= args.max_hire_rate
        ),
        "beats_rsu_only_stall": (
            rsu_stall - ppo_stall >= args.min_stall_improvement
        ),
        "large_buffer_assumption": all(
            float(row["large_queue_violation_rate"]) == 0.0 for row in ppo_rows
        ),
        "queue_can_prefetch": (
            base_cfg.max_chunks_per_slot > base_cfg.playback_chunks_per_slot
        ),
    }

    payload = {
        "event": "passed" if all(checks.values()) else "failed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "seeds": list(args.seeds),
        "config": asdict(base_cfg),
        "criteria": {
            "min_hire_rate": args.min_hire_rate,
            "max_hire_rate": args.max_hire_rate,
            "min_stall_improvement": args.min_stall_improvement,
        },
        "metrics": {
            "ppo_stall_ratio": ppo_stall,
            "rsu_only_stall_ratio": rsu_stall,
            "stall_improvement": rsu_stall - ppo_stall,
            "ppo_hire_rate": hire_rate,
            "ppo_mean_z": mean(ppo_rows, "mean_z"),
            "ppo_p95_z": mean(ppo_rows, "p95_z"),
            "ppo_max_queue": max(float(row["max_queue"]) for row in ppo_rows),
            "ppo_large_queue_violation_rate": mean(
                ppo_rows, "large_queue_violation_rate"
            ),
            "ppo_worst_user_stall": mean(
                ppo_rows, "worst_user_stall_ratio"
            ),
            "ppo_jain_service_fairness": mean(
                ppo_rows, "jain_service_fairness"
            ),
            "hard_violations": hard_violations,
        },
        "invalid_metric_paths": invalid_metrics,
        "checks": checks,
        "ppo_seed_summaries": ppo_rows,
        "rsu_only_seed_summaries": rsu_rows,
    }

    atomic_write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    if payload["event"] != "passed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
