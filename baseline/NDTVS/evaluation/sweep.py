"""Paired smoke, nominal evaluation and resumable SNR orchestration."""
from __future__ import annotations

import os
import platform
import time
import uuid
import numpy as np
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import baseline.NDTVS.training.rollout as rollout
from baseline.NDTVS.common.checkpoint import paired_resume_header, EVALUATOR_SOURCE_FILES
from baseline.NDTVS.common.paths import HERE
from baseline.NDTVS.evaluation.scenario import PairingObserver, radio_sanity
from baseline.NDTVS.evaluation.policies import load_policies, policy_digest
from baseline.NDTVS.evaluation.checks import (OFFSETS, METRICS, require, read_json,
    sha256, algorithm, check_rows, verify_saved)


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import baseline.NDTVS.api as c
    torch = c.torch
    c.require_device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    configs, policies, provenance = load_policies(args, c)
    names = list(policies)
    nominal = configs["proposed"].noise_psd_w_hz
    levels = list(OFFSETS) if args.mode == "sweep" else [0]
    ids = list(range(args.offset, args.offset + (1 if args.mode == "smoke" else args.episodes)))
    source = c.source_hashes()
    evaluator = {name: sha256(HERE / name) for name in EVALUATOR_SOURCE_FILES}
    selection = c.jsonable({"source_sha256": source, "checkpoints": provenance,
                            "configs": {k: asdict(v) for k, v in configs.items()},
                            "qoe_definition": c.reward_spec(), "device": args.device})
    if args.mode == "sweep":
        gate = read_json(args.out / "smoke/verification.json")
        current_header = {"selection": selection, "evaluators": evaluator}
        previous_header = {"selection": gate["selection"], "evaluators": gate["evaluators"]}
        require(gate["passed"] and paired_resume_header(previous_header, current_header) == current_header,
                "Run matching smoke first")
        verify_saved(args.out / "smoke")
        smoke_ids = read_json(args.out / "smoke/state.json")["spec"]["scenario_ids"]
        require(not set(smoke_ids).intersection(ids), "Use test episodes disjoint from smoke")
    root = args.out / args.mode
    spec = {"mode": args.mode, "selection": selection, "policy_order": names,
            "scenario_seed": args.scenario_seed, "scenario_ids": ids,
            "snr_offsets_db": levels, "evaluators": evaluator,
            "metric_window": "all slots, including initial buffer; no warmup exclusion",
            "radio_sanity": radio_sanity(configs["proposed"])}
    state_file = root / "state.json"
    if state_file.exists():
        require(args.resume, "Output exists; use --resume or a new --out")
        state = read_json(state_file)
        require(paired_resume_header(state["spec"], spec) == spec,
                "Resume settings/checkpoints/sources differ")
        if state["spec"] != spec:
            state["spec"] = spec
            c.atomic(state_file, state)
    else:
        require(not root.exists() or not any(root.iterdir()), "Nonempty output has no resumable state")
        state = {"spec": spec, "cells": {}}
        c.atomic(state_file, state)
    before = {name: policy_digest(policies[name]) for name in names}
    fingerprints, durations = {}, []
    for key, cell in state["cells"].items():
        delta, name = key.split("/")
        require(int(delta) in levels and name in names, "Unexpected saved cell")
        require(cell["policy_sha256"] == before[name], "Saved policy differs")
        require(len(cell["rows"]) == len(cell["durations"]), "Partial row/duration mismatch")
        check_rows(cell["rows"], configs[name], name, ids, fingerprints)
        durations.extend(cell["durations"])
    c.atomic(root / "attempts" / (uuid.uuid4().hex + ".json"),
             {"python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "job_id": os.environ.get("SLURM_JOB_ID")})
    budget = c.Budget(args.walltime_seconds, args.reserve_seconds)
    try:
        for delta in levels:
            for name in names:
                key = f"{delta}/{name}"
                cell = state["cells"].setdefault(key, {"rows": [], "durations": [], "policy_sha256": before[name]})
                cfg = replace(configs[name], device=args.device, seed=args.scenario_seed,
                              noise_psd_w_hz=nominal * 10**(-delta/10))
                for i in range(len(cell["rows"]), len(ids)):
                    if budget.expired(1.5 * max(durations, default=0)):
                        print("PAUSED: resubmit identical command with --resume", flush=True)
                        return 75
                    c.hrl.seed_all(args.scenario_seed)
                    directory = root / "episodes" / f"snr{delta:+d}_{name}_{ids[i]}_{uuid.uuid4().hex[:8]}"
                    trace = args.trace or i == 0
                    began = time.monotonic()
                    observed_p = type("ObservedProposed", (PairingObserver, rollout.P3HierarchicalEnv), {})
                    observed_r = type("ObservedRSU", (PairingObserver, rollout.RSUEnv), {})
                    with patch.object(rollout, "P3HierarchicalEnv", observed_p), patch.object(rollout, "RSUEnv", observed_r):
                        row = c.episode(cfg, algorithm(name), policies[name], ids[i], cfg.dual_init, False, directory, trace)
                    check_rows([row], cfg, name, [ids[i]], fingerprints)
                    require(policy_digest(policies[name]) == before[name], "Policy changed during evaluation")
                    if trace:
                        from baseline.NDTVS.analysis.cli import audit
                        require(audit(directory) == 0, "Trace physics/QoE audit failed")
                        row["audit_file"] = str((directory / "audit.json").relative_to(root))
                        row["audit_sha256"] = sha256(directory / "audit.json")
                    cell["rows"].append(row)
                    elapsed = time.monotonic() - began
                    cell["durations"].append(elapsed)
                    durations.append(elapsed)
                    c.atomic(state_file, state)
                    print(f"{name} SNR={delta:+d} {i+1}/{len(ids)} QoE={row[METRICS[0]]:.6f}", flush=True)
    finally:
        budget.close()
    require(c.source_hashes() == source, "Sources changed during evaluation")
    require({k: sha256(HERE / k) for k in evaluator} == evaluator,
            "Evaluator/auditor changed during evaluation")
    for item in provenance.values():
        for path, value in item["files"].items():
            require(sha256(path) == value, "Checkpoint changed during evaluation")
    c.atomic(root / "verification.json", {"passed": True, "selection": selection,
             "evaluators": evaluator, "state_sha256": sha256(state_file),
             "scenario_sha256": fingerprints,
             "policy_unchanged": {k: before[k] == policy_digest(policies[k]) for k in names}})
    verify_saved(root)
    return 0
