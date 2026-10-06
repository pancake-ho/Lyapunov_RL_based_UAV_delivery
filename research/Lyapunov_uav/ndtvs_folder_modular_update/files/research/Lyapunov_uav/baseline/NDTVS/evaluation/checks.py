"""Evaluation schemas, pairing and committed-output verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from baseline.NDTVS.analysis.compare import comparable_config
ALGORITHMS = ("proposed", "hppo_rsu", "ndtvs")
OFFSETS = (-10, -5, 0, 5, 10)
METRICS = (
    "paper_qoe_per_user_slot", "stall_ratio", "average_quality_utility",
    "paper_qv_per_user_slot", "rebuffer_s_per_user", "stall_time_ratio",
    "average_received_psnr_db", "request_failure_ratio",
    "delivered_chunks_per_user_slot", "dpp_cost_per_user_slot",
    "original_cost_per_user_slot", "hire_rate", "energy_consumed_j",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def physical(cfg):
    # Match existing comparison, retaining the DPP reward scale explicitly.
    from baseline.NDTVS.analysis.cli import comparable_config
    return {**comparable_config(cfg), "ppo_reward_scale": cfg["ppo_reward_scale"]}


def algorithm(name):
    return "proposed" if name.startswith("proposed") else name


def validate_row(row, cfg, name, episode):
    require(row["episode"] == episode, "Unexpected episode identity")
    require(row["frames"] == cfg.num_frames and
            row["observed_slots"] == cfg.num_frames * cfg.frame_slots, "Incomplete episode")
    require(np.isfinite([v for v in row.values() if isinstance(v, (int, float))]).all(),
            "Nonfinite evaluation summary")
    require(all(k in row for k in METRICS), "Required metric missing")
    require(row["reserve_violations"] == row["power_violations"] == 0, "Physical invariant failed")
    require(row["q_gt_qe_rate"] == 0, "Queue admissibility violation")
    if algorithm(name) != "proposed":
        require(all(row[k] == 0 for k in ("hire_rate", "hiring_cost_total", "energy_consumed_j")),
                "RSU-only invariant failed")


def check_rows(rows, cfg, name, ids, fingerprints):
    require(len(rows) <= len(ids), "Too many saved rows")
    for row, expected in zip(rows, ids):
        validate_row(row, cfg, name, expected)
        require(fingerprints.setdefault(expected, row["scenario_sha256"]) == row["scenario_sha256"],
                "Actual mobility/fading pairing mismatch")


def verify_saved(root):
    state, gate = read_json(root / "state.json"), read_json(root / "verification.json")
    spec = state["spec"]
    require(gate["passed"] and gate["state_sha256"] == sha256(root / "state.json"), "Incomplete/modified result")
    require(gate["selection"] == spec["selection"] and gate["evaluators"] == spec["evaluators"], "Provenance mismatch")
    names, ids, levels = spec["policy_order"], spec["scenario_ids"], spec["snr_offsets_db"]
    require(gate["policy_unchanged"] == {k: True for k in names}, "Policy changed/missing")
    require(set(state["cells"]) == {f"{d}/{p}" for d in levels for p in names}, "Incomplete grid")
    fingerprints = {}
    from types import SimpleNamespace
    for key, cell in state["cells"].items():
        name = key.split("/")[1]
        require(len(cell["rows"]) == len(cell["durations"]) == len(ids), "Incomplete cell")
        cfg = SimpleNamespace(**spec["selection"]["configs"][name])
        check_rows(cell["rows"], cfg, name, ids, fingerprints)
        for row in cell["rows"]:
            if "audit_file" in row:
                path = (root / row["audit_file"]).resolve()
                require(root.resolve() in path.parents, "Invalid audit path")
                require(sha256(path) == row["audit_sha256"] and read_json(path)["valid"], "Failed/modified audit")
        require("audit_file" in cell["rows"][0], "First scenario must have a trace audit")
    require({str(k): v for k, v in fingerprints.items()} == gate["scenario_sha256"], "Fingerprint mismatch")
    return state
