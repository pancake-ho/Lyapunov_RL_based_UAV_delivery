"""Single-method checkpoint evaluation and evaluation resume."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from baseline.NDTVS.common.config import read_config, HPPOConfig
from baseline.NDTVS.common.io import atomic, write_rows, load_bundle, jsonable
from baseline.NDTVS.common.runtime import require_device, Budget
from baseline.NDTVS.models.policy import make_agents
from baseline.NDTVS.training.rollout import episode, hrl
from baseline.NDTVS.common.checkpoint import verify_checkpoint_source, source_hashes, restore_agents, evaluation_resume_spec
from baseline.NDTVS.rewards.qoe import QOE_WEIGHTS, reward_spec


def run_eval(args):
    require_device(args.device)
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise FileExistsError("Evaluation output must be empty")
    if args.algorithm == "proposed":
        cfg = replace(read_config(args.config), device=args.device)
        agents = make_agents(cfg, "proposed")
        extras = [a.load(p) for a, p in zip(agents, (args.frame_checkpoint, args.slot_checkpoint))]
        if not extras[0].get("pair_id") or extras[0]["pair_id"] != extras[1].get("pair_id"):
            raise ValueError("Proposed checkpoints must be a matched pair")
        provenance = {"pair": extras, "files": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                      for p in (args.frame_checkpoint, args.slot_checkpoint)}}
    else:
        saved = load_bundle(args.checkpoint)
        migration = verify_checkpoint_source(saved, args.algorithm)
        cfg = HPPOConfig(**{k: tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
                           if isinstance(v, list) else v for k, v in saved["spec"]["config"].items()})
        cfg = replace(cfg, device=args.device)
        agents = make_agents(cfg, args.algorithm)
        restore_agents(agents, saved["policies"])
        provenance = {"source_verification": migration, "reward_definition": saved["spec"].get("qoe_definition"),
                      "trained_episodes": saved["next_episode"], "file": str(args.checkpoint),
                      "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    cfg = replace(cfg, seed=args.scenario_seed)
    torch.set_num_threads(cfg.torch_num_threads)
    hrl.seed_all(args.scenario_seed)
    if args.offset < 2_000_000:
        raise ValueError("Final evaluation uses offset >= 2000000, disjoint from training/validation")
    spec = {"algorithm": args.algorithm, "config": asdict(cfg), "offset": args.offset,
            "scenario_seed": args.scenario_seed, "target_episodes": args.episodes,
            "provenance": provenance, "source_sha256": source_hashes(), "qoe_weights": QOE_WEIGHTS, "qoe_definition": reward_spec()}
    rows, durations = [], []
    if args.resume:
        previous = json.loads((args.out / "evaluation_partial.json").read_text())
        if evaluation_resume_spec(previous["spec"], jsonable(spec)) != jsonable(spec):
            raise ValueError("Evaluation resume settings/checkpoints differ")
        rows, durations = previous["rows"], previous["durations"]
    budget = Budget(args.walltime_seconds, args.reserve_seconds)
    try:
        for i in range(len(rows), args.episodes):
            if budget.expired(max(durations, default=0) * 1.5):
                break
            began = time.monotonic()
            rows.append(episode(cfg, args.algorithm, agents, args.offset + i, cfg.dual_init, False,
                                args.out / f"ep_{i:04d}", args.trace))
            durations.append(time.monotonic() - began)
            atomic(args.out / "evaluation_partial.json", {"spec": spec, "rows": rows, "durations": durations})
        atomic(args.out / "evaluation_partial.json", {"spec": spec, "rows": rows, "durations": durations})
    finally:
        budget.close()
    if len(rows) != args.episodes:
        print("Evaluation paused; resubmit with --resume", flush=True)
        return 75
    atomic(args.out / "evaluation.json", {**spec, "per_episode": rows})
    write_rows(args.out / "evaluation.csv", rows)
    print(json.dumps({k: float(np.mean([r[k] for r in rows])) for k in
                     ("paper_qoe_per_user_slot", "stall_ratio", "average_quality_utility", "hire_rate")}, indent=2))
    return 0
