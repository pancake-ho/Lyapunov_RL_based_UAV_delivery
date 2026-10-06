"""Training, validation selection and resumable checkpoints."""
from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
import numpy as np
import torch
from baseline.NDTVS.common.config import read_config, VERSION, BASE_COMMIT
from baseline.NDTVS.common.io import atomic, write_rows, load_bundle, jsonable
from baseline.NDTVS.common.runtime import require_device, rng_state, restore_rng, Budget
from baseline.NDTVS.models.policy import make_agents
from baseline.NDTVS.training.rollout import episode, hrl
from baseline.NDTVS.common.checkpoint import source_hashes, policy_state, restore_agents, resume_spec
from baseline.NDTVS.rewards.qoe import QOE_WEIGHTS, reward_spec


def run_train(args):
    require_device(args.device)
    root = args.out.resolve()
    cfg = replace(read_config(args.config), device=args.device)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if not 0 < args.episodes < 1_000_000:
        raise ValueError("training episodes must be in [1, 999999]")
    if cfg.episode_offset != 0:
        raise ValueError("Training uses episode IDs starting at zero; use a source config with episode_offset=0")
    if args.algorithm == "hppo_rsu" and cfg.reward_mode != "dpp":
        raise ValueError("The requested proposed ablation uses the current DPP objective")
    spec = {"config": jsonable(asdict(cfg)), "algorithm": args.algorithm,
            "episodes": args.episodes, "val_every": args.val_every, "val_episodes": args.val_episodes,
            "source_sha256": source_hashes(), "version": VERSION, "base_commit": BASE_COMMIT,
            "qoe_weights": QOE_WEIGHTS, "qoe_definition": reward_spec(),
            "learning_settings": ({"hidden_dims": [128, 64], "actor_lr": 1e-5, "critic_lr": 1e-4,
                                   "gamma": 0.95, "gae_lambda": 0.95, "minibatch_size": 64,
                                   "update_every_episodes": 1} if args.algorithm == "ndtvs"
                                  else "identical to proposed config")}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty output directory; use --resume or another --out")
    root.mkdir(parents=True, exist_ok=True)
    hrl.seed_all(cfg.seed)
    torch.set_num_threads(cfg.torch_num_threads)
    agents = make_agents(cfg, args.algorithm)
    spec["model_parameters"] = [sum(p.numel() for p in a.net.parameters()) for a in agents]
    rows, vals, start, dual, best = [], [], 0, cfg.dual_init, -math.inf
    durations = []
    if args.resume:
        saved = load_bundle(root / "latest.pt")
        if jsonable(resume_spec(saved, args.algorithm)) != jsonable(spec):
            raise ValueError("Resume configuration/source/budget changed; restore the original settings")
        restore_agents(agents, saved["policies"])
        restore_rng(saved["rng"])
        rows, vals, start, dual, best = (saved[k] for k in ("rows", "validation", "next_episode", "dual", "best"))
        durations = saved["durations"]
    atomic(root / "experiment.json", spec)
    attempt = root / "segments" / str(time.time_ns())
    attempt.mkdir(parents=True)
    budget = Budget(args.walltime_seconds, args.reserve_seconds)

    def save(next_episode):
        payload = {"spec": spec, "policies": policy_state(agents), "rng": rng_state(),
                   "next_episode": next_episode, "rows": rows, "validation": vals,
                   "dual": dual, "best": best, "durations": durations}
        atomic(root / "latest.pt", payload, binary=True)
        write_rows(root / "training.csv", rows)
        atomic(root / "validation.json", vals)
        return payload

    def validate(completed):
        nonlocal best
        if not completed or any(v["trained_episodes"] == completed for v in vals):
            return
        if completed % args.val_every and completed != args.episodes:
            return
        if budget.expired(max(durations[-10:], default=0) * (args.val_episodes + 1)):
            return  # Resume retries this validation before advancing training.
        state = rng_state()
        try:
            scores = [episode(cfg, args.algorithm, agents, 1_000_000 + i, dual, False,
                              attempt / f"val_{completed:06d}_{i:03d}", False)
                      for i in range(args.val_episodes)]
        finally:
            restore_rng(state)
        key = "paper_qoe_per_user_slot" if args.algorithm == "ndtvs" else "dpp_cost_per_user_slot"
        score = float(np.mean([r[key] for r in scores])) * (1 if args.algorithm == "ndtvs" else -1)
        vals.append({"trained_episodes": completed, "selection_score": score,
                     "selection_metric": key, "per_episode": scores})
        improved = score > best
        if improved:
            best = score
        payload = save(completed)
        if improved:
            atomic(root / "best.pt", payload, binary=True)

    try:
        if not args.resume:
            save(0)  # Even an immediate hard kill has a restart point.
        completed = start
        if args.resume:
            # Recover a crash between committing validation and exporting best.pt.
            if vals and vals[-1]["trained_episodes"] == start and vals[-1]["selection_score"] == best:
                atomic(root / "best.pt", saved, binary=True)
            validate(start)
        for ep in range(start, args.episodes):
            if budget.expired(max(durations[-10:], default=0) * 1.5):
                break
            began = time.monotonic()
            trace = args.trace_every > 0 and ((ep + 1) % args.trace_every == 0 or ep == 0)
            directory = attempt / f"train_{ep:06d}"
            row = episode(cfg, args.algorithm, agents, ep, dual, True, directory, trace)
            if args.algorithm == "ndtvs":
                row.update(agents[0].update())
            elif (ep + 1) % cfg.frame_update_every_episodes == 0 or ep + 1 == args.episodes:
                row.update(agents[0].update())
            row["trace_dir"] = str(directory.relative_to(root)) if trace else ""
            rows.append(row)
            completed = ep + 1
            durations.append(time.monotonic() - began)
            save(completed)  # Commit before potentially expensive validation.
            validate(completed)
            print(f"{args.algorithm} {completed}/{args.episodes} "
                  f"QoE={row['paper_qoe_per_user_slot']:.5f} stall={row['stall_ratio']:.4f} "
                  f"{durations[-1]:.2f}s", flush=True)
            if args.max_new_episodes and completed - start >= args.max_new_episodes:
                budget.reason = "max-new-episodes"
                break
        done = completed == args.episodes
        atomic(root / "status.json", {"status": "complete" if done else "paused",
               "completed_episodes": completed, "target_episodes": args.episodes,
               "reason": budget.reason, "convergence": "not certified",
               "validation_points": len(vals), "checkpoint": "latest.pt",
               "final_validation_pending": done and not any(v["trained_episodes"] == completed for v in vals)})
        return 0 if done else 75
    finally:
        budget.close()
