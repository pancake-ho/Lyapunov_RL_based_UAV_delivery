from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from agent.P3.ppo_agent import PPOAgent
from config_p3 import P3Config
from run.p3_common import run_policy
from run.p3_train_ppo import parse_seed_list


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast acceptance gate for Changed_Form slow PPO"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=parse_seed_list, default=parse_seed_list("93026,93027,93028")
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

    base_cfg = P3Config(num_frames=args.frames)
    agent = PPOAgent(base_cfg, device=args.device)
    metadata = agent.load(args.checkpoint.resolve())
    ppo_rows = []
    rsu_rows = []
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
                cfg, "rsu_only", args.output.parent, write_outputs=False
            ).summary
        )

    mean = lambda rows, key: float(np.mean([float(row[key]) for row in rows]))
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
        "finite_metrics": all(
            math.isfinite(float(row[key]))
            for row in ppo_rows
            for key in (
                "stall_ratio",
                "hire_rate",
                "mean_queue",
                "p95_queue",
                "mean_z",
                "p95_z",
                "jain_service_fairness",
            )
        ),
        "hard_constraints": hard_violations == 0,
        "noncollapsed_hiring": args.min_hire_rate <= hire_rate <= args.max_hire_rate,
        "beats_rsu_only_stall": (
            rsu_stall - ppo_stall >= args.min_stall_improvement
        ),
        "large_buffer_assumption": all(
            float(row["large_queue_violation_rate"]) == 0.0 for row in ppo_rows
        ),
        "queue_can_prefetch": base_cfg.max_chunks_per_slot > base_cfg.playback_chunks_per_slot,
    }
    payload = {
        "event": "passed" if all(checks.values()) else "failed",
        "checkpoint": str(args.checkpoint.resolve()),
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
            "ppo_worst_user_stall": mean(ppo_rows, "worst_user_stall_ratio"),
            "ppo_jain_service_fairness": mean(ppo_rows, "jain_service_fairness"),
            "hard_violations": hard_violations,
        },
        "checks": checks,
        "ppo_seed_summaries": ppo_rows,
        "rsu_only_seed_summaries": rsu_rows,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    if payload["event"] != "passed":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
