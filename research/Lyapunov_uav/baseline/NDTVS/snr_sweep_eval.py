"""Fixed-checkpoint SNR robustness evaluation; audited against feat/hrl 6201893.

Install beside ndtvs_common.py. Existing source/checkpoints are read-only.
Exit 75 means paused: resubmit the SAME command with --resume.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ALGORITHMS = ("proposed", "hppo_rsu", "ndtvs")
OFFSETS = (-10, -5, 0, 5, 10)
EXPECTED_EPISODES = {"ndtvs": 175, "hppo_rsu": 800, "proposed": 124}
AUDITED_COMMIT = "6201893088287c1274b9c2a4251ec2e68f2378fa"
AUDITED_SOURCE_DIGEST = "445b344c85f2aa4fa98a3d58de6862b201f506927d8100b2565a9e83f66b6d0d"
AUDITED_ANALYSIS_DIGEST = "1ef94919ee7471f24ea0601bdaf0ddb69f8b147eed1eb3557064380ab1f97324"
BASE_NOISE = 2e-20
METRICS = (
    "qoe_surrogate_per_user_slot", "stall_ratio", "average_quality_utility",
    "switch_magnitude_per_user_slot", "request_failure_ratio",
    "delivered_chunks_per_user_slot", "dpp_cost_per_user_slot",
    "original_cost_per_user_slot", "hire_rate", "energy_consumed_j",
)
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "baseline/NDTVS/runs/common_gpu"
TRAIN = ROOT / "proposed/outputs/hppo/hrl_revision_train_seed2026_job142434"


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
    from ndtvs_analysis import comparable_config
    return {**comparable_config(cfg), "ppo_reward_scale": cfg["ppo_reward_scale"]}


def radio_sanity(cfg):
    from env.p3.radio import capacity_bps, rsu_link_capacity_bps
    from hppo.env import uav_link_capacity_bps
    rows = []
    for delta in OFFSETS:
        noise = BASE_NOISE * 10.0 ** (-delta / 10.0)
        effective = replace(cfg, noise_psd_w_hz=noise)
        for bw in (cfg.rsu_total_bandwidth_hz / cfg.rsu_capacity,
                   cfg.uav_user_bandwidth_hz):
            require(effective.shannon_gap * noise * bw > 1e-30,
                    "Noise floor clamp would invalidate the exact SNR offset")
            snr = 1e-12 / (effective.shannon_gap * noise * bw)
            base_snr = 1e-12 / (cfg.shannon_gap * BASE_NOISE * bw)
            require(math.isclose(10 * math.log10(snr / base_snr), delta, abs_tol=1e-12),
                    "SNR shift check failed")
            require(math.isfinite(capacity_bps(bw, 1, 1e-12, effective)), "Bad capacity")
        rows.append({"snr_offset_db": delta, "noise_psd_w_hz": noise,
                     "rsu_bps": rsu_link_capacity_bps(150, 1, effective),
                     "uav_bps": uav_link_capacity_bps(150, 1, 1, effective)[0]})
    require(rows[2]["noise_psd_w_hz"] == cfg.noise_psd_w_hz == BASE_NOISE,
            "Nominal checkpoint noise is not 2e-20 W/Hz")
    for key in ("rsu_bps", "uav_bps"):
        require(all(a[key] < b[key] for a, b in zip(rows, rows[1:])),
                "Fixed-link capacity is not strictly increasing")
    return rows


class PairingObserver:
    """Observe actual exogenous arrays; never change transitions or actions."""
    def add_arrays(self, tag, *arrays):
        self._scenario.update(tag.encode())
        for values in arrays:
            a = np.ascontiguousarray(values, dtype="<f8")
            require(np.isfinite(a).all(), "Nonfinite exogenous state")
            self._scenario.update(str(a.shape).encode())
            self._scenario.update(a.tobytes())

    def reset(self, episode=0, seed=None):
        super().reset(episode, seed)
        self._scenario = hashlib.sha256(f"{self.cfg.seed}:{episode}".encode())
        self._observed_slots = 0
        self.add_arrays("initial", self.state.user_x, self.state.user_speed)

    def prepare_frame(self):
        obs = super().prepare_frame()
        # Hash every potential RSU/UAV link, including unselected UAV points.
        self.add_arrays(f"frame:{self.frame}", self.trace.rsu_fading, self.trace.uav_fading)
        return obs

    def step_slot(self, actions):
        result = super().step_slot(actions)
        self.add_arrays(f"slot:{self._observed_slots}", self.state.user_x, self.state.user_speed)
        self._observed_slots += 1
        for rg in result.info["regions"].values():
            values = [rg[k] for k in ("total_requested_power_w", "total_executed_power_w",
                      "p_eff_w", "battery_before_j", "battery_after_j",
                      "hover_energy_j", "communication_energy_j")]
            require(np.isfinite(values).all(), "Nonfinite physical metric")
            require(rg["reserve_ok"], "Reserve violation")
            require(-1e-9 <= rg["total_executed_power_w"] <= rg["p_eff_w"] + 1e-9,
                    "Executed power violates feasibility")
        return result

    def episode_summary(self):
        row = super().episode_summary()
        row.update(scenario_sha256=self._scenario.hexdigest(),
                   observed_slots=self._observed_slots)
        return row


def policy_digest(agents):
    h = hashlib.sha256()
    for agent in agents:
        h.update(str(agent.update_count).encode())
        h.update(str(agent.buffer_size()).encode())
        for name, tensor in sorted(agent.net.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_policies(args, c):
    policies, provenance, configs = {}, {}, {}
    for name, path in (("ndtvs", args.ndtvs_checkpoint), ("hppo_rsu", args.rsu_checkpoint)):
        saved = c.load_bundle(path)
        require(saved["spec"]["algorithm"] == name, "Checkpoint algorithm mismatch")
        require(saved["spec"]["source_sha256"] == c.source_hashes(),
                "Checkpoint algorithm/source mismatch; restore its exact sources")
        require(saved["next_episode"] == EXPECTED_EPISODES[name],
                f"Expected {name} validation-best ep{EXPECTED_EPISODES[name]}")
        d = saved["spec"]["config"]
        cfg = c.HPPOConfig(**{k: tuple(math.inf if x is None and k == "distance_bin_edges_m"
                                    else x for x in v) if isinstance(v, list) else v
                             for k, v in d.items()})
        configs[name] = cfg
        policies[name] = c.make_agents(replace(cfg, device="cuda"), name)
        c.restore_agents(policies[name], saved["policies"])
        provenance[name] = {"selected_training_episode": saved["next_episode"],
                            "files": {str(path.resolve()): sha256(path)}}
    cfg = c.read_config(args.config)
    configs["proposed"] = cfg
    agents = c.make_agents(replace(cfg, device="cuda"), "proposed")
    paths = (args.frame_checkpoint, args.slot_checkpoint)
    # Validate the ORIGINAL physical/control config before any SNR replacement.
    extras = [agent.load(path) for agent, path in zip(agents, paths)]
    require(extras[0].get("pair_id") and extras[0] == extras[1],
            "Proposed checkpoint pair metadata mismatch")
    require(extras[0].get("episode") == 123, "Expected Proposed ep124 (zero-based extra.episode=123)")
    policies["proposed"] = agents
    provenance["proposed"] = {"selected_training_episode": 124, "pair": extras,
                              "files": {str(p.resolve()): sha256(p) for p in paths}}
    reference = None
    for name in ALGORITHMS:
        cfg = configs[name]
        require(cfg.seed == 2026 and cfg.episode_offset == 0,
                f"Unexpected training seed/offset in {name}; do not silently reinterpret")
        require(cfg.reward_mode == "dpp" and cfg.noise_psd_w_hz == BASE_NOISE,
                f"Unexpected nominal configuration in {name}")
        require(cfg.mask_queue_actions and cfg.enforce_queue_admissibility
                and cfg.delivery_mode == "all_or_nothing", "Atomic queue-masked delivery required")
        current = physical(c.jsonable(asdict(cfg)))
        require(reference is None or current == reference, "Algorithm physical/control configs differ")
        reference = current
        for agent in policies[name]:
            agent.net.eval()
    return configs, policies, provenance


def validate_row(row, cfg, algorithm, episode):
    require(row["episode"] == episode, "Unexpected episode identity")
    require(row["frames"] == cfg.num_frames and
            row["observed_slots"] == cfg.num_frames * cfg.frame_slots, "Incomplete episode")
    numeric = [v for v in row.values() if isinstance(v, (int, float))]
    require(np.isfinite(numeric).all(), "Nonfinite evaluation summary")
    require(all(k in row for k in METRICS), "Required metric missing")
    require(row["reserve_violations"] == row["power_violations"] == 0, "Physical invariant failed")
    if algorithm != "proposed":
        require(all(row[k] == 0 for k in ("hire_rate", "hiring_cost_total", "energy_consumed_j")),
                "No-UAV invariant failed")


def reference_evaluations(root):
    found = {}
    for path in root.rglob("evaluation.json"):
        d = read_json(path)
        name = d.get("algorithm")
        if name in ALGORITHMS:
            require(name not in found, f"Ambiguous final evaluations for {name} under {root}")
            found[name] = (path, d)
    require(set(found) == set(ALGORITHMS), "Missing original final evaluation.json files")
    return found


def match_reference(ref, cfg, algorithm, provenance, source, row=None):
    require(ref["algorithm"] == algorithm and ref["scenario_seed"] == 2026
            and ref["offset"] == 3_000_000, "Unexpected final reference namespace")
    require(ref["source_sha256"] == source, "Final reference source differs")
    require(ref["qoe_weights"] == [1.0, 0.5, 2.0], "Final reference QoE weights differ")
    require(physical(ref["config"]) == physical(cfg), "Final reference config differs")
    old = ref["provenance"]
    hashes = old["files"].values() if algorithm == "proposed" else [old["sha256"]]
    require(sorted(hashes) == sorted(provenance["files"].values()), "Final reference checkpoint differs")
    if algorithm == "proposed":
        require(old["pair"] == provenance["pair"], "Final reference checkpoint pair differs")
    else:
        require(old["trained_episodes"] == EXPECTED_EPISODES[algorithm], "Final selected episode differs")
    if row is None:
        return
    matches = [r for r in ref["per_episode"] if r["episode"] == row["episode"]]
    require(len(matches) == 1, "Smoke episode missing/duplicated in final reference")
    # Wall-clock measurements are deliberately excluded; physical and reward values are checked.
    for key, value in matches[0].items():
        if key == "completion_runtime_s":
            continue
        require(key in row, f"Smoke result missing {key}")
        if isinstance(value, (int, float)):
            require(np.isclose(row[key], value, rtol=1e-8, atol=1e-10),
                    f"0 dB equivalence failed: {algorithm}/{key}: {row[key]} vs {value}")
        else:
            require(row[key] == value, f"0 dB equivalence failed: {key}")


def git_info():
    def run(*args):
        p = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"),
            "status_porcelain": run("status", "--porcelain"), "audited_commit": AUDITED_COMMIT}


def run(args, c, torch):
    source = c.source_hashes()
    require(digest(source) == AUDITED_SOURCE_DIGEST, "Source differs from audited 6201893; review before use")
    require(sha256(Path(c.__file__).with_name("ndtvs_analysis.py")) == AUDITED_ANALYSIS_DIGEST,
            "ndtvs_analysis.py differs from audited 6201893; review before use")
    configs, policies, provenance = load_policies(args, c)
    originals = {k: c.jsonable(asdict(v)) for k, v in configs.items()}
    selection = {"source_sha256": source, "checkpoints": c.jsonable(provenance), "configs": originals}
    gate_path = args.smoke_root / "smoke_verification.json"
    references = None
    if args.mode == "smoke":
        references = reference_evaluations(args.reference_root)
        for name, (_, ref) in references.items():
            match_reference(ref, originals[name], name, provenance[name], source)
    else:
        gate = read_json(gate_path)
        require(gate["passed"] and gate["selection"] == selection,
                "Successful 0 dB smoke verification for these exact checkpoints/configs required")
    levels = (0,) if args.mode == "smoke" else OFFSETS
    offset = 3_000_000 if args.mode == "smoke" else 4_000_000
    count = 1 if args.mode == "smoke" else args.episodes
    require(0 < count <= 1000, "Episode count must be in [1,1000]")
    spec = {"mode": args.mode, "selection": selection, "scenario_seed": 2026,
            "offset": offset, "episodes": count, "snr_offsets_db": list(levels),
            "base_noise_psd_w_hz": BASE_NOISE, "trace": args.trace or args.mode == "smoke",
            "analysis_sha256": AUDITED_ANALYSIS_DIGEST,
            "script_sha256": sha256(__file__), "radio_sanity": radio_sanity(configs["proposed"]),
            "smoke_gate_sha256": sha256(gate_path) if args.mode == "sweep" else None,
            "references": {k: {"path": str(p.resolve()), "sha256": sha256(p)}
                           for k, (p, _) in references.items()} if references else None}
    manifest = args.out / "sweep_manifest.json"
    if args.out.exists() and any(args.out.iterdir()):
        require(args.resume and manifest.exists(), "Nonempty output: use another path or --resume")
        require(read_json(manifest) == spec, "Resume settings/source/checkpoints differ")
    else:
        c.atomic(manifest, spec)
    runtime = {"git": git_info(), "python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": True,
               "environment": {k: os.environ.get(k) for k in
                               ("SLURM_JOB_ID", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS")}}
    c.atomic(args.out / "attempts" / f"{uuid.uuid4().hex}.json", runtime)
    fingerprints, verified = {}, {}
    budget = c.Budget(args.walltime_seconds, args.reserve_seconds)
    try:
        for delta in levels:
            for name in ALGORITHMS:
                cfg = replace(configs[name], device="cuda", seed=2026,
                              noise_psd_w_hz=BASE_NOISE * 10.0 ** (-delta / 10.0))
                root = args.out / f"snr_{delta:+03d}dB" / name
                cell = {"algorithm": name, "config": c.jsonable(asdict(cfg)),
                        "logger_config": c.jsonable(asdict(replace(cfg, write_jsonl_trace=spec["trace"],
                                                                   write_human_debug_log=False))),
                        "original_config": originals[name], "offset": offset, "scenario_seed": 2026,
                        "target_episodes": count, "snr_offset_db": delta,
                        "base_noise_psd_w_hz": BASE_NOISE,
                        "effective_noise_psd_w_hz": cfg.noise_psd_w_hz,
                        "provenance": c.jsonable(provenance[name]), "source_sha256": source,
                        "qoe_weights": list(c.QOE_WEIGHTS), "manifest_sha256": digest(spec)}
                partial = root / "evaluation_partial.json"
                rows, durations = [], []
                if partial.exists():
                    previous = read_json(partial)
                    require(previous["spec"] == cell, "Cell resume specification mismatch")
                    rows, durations = previous["rows"], previous["durations"]
                require(len(rows) == len(durations) <= count, "Corrupt partial evaluation")
                for i, row in enumerate(rows):
                    validate_row(row, cfg, name, offset + i)
                    if args.mode == "smoke":
                        match_reference(references[name][1], cell["config"], name,
                                        provenance[name], source, row)
                c.hrl.seed_all(2026)
                c.torch.set_num_threads(cfg.torch_num_threads)
                before = policy_digest(policies[name])
                for i in range(len(rows), count):
                    if budget.expired(max(durations, default=0) * 1.5):
                        print("PAUSED: resubmit the same command with --resume", flush=True)
                        return 75
                    directory = root / "episodes" / f"ep_{i:04d}_{uuid.uuid4().hex[:12]}"
                    started = time.monotonic()
                    ObservedProposed = type("ObservedProposed", (PairingObserver, c.P3HierarchicalEnv), {})
                    ObservedRSU = type("ObservedRSU", (PairingObserver, c.RSUEnv), {})
                    # Only these local names are substituted; all original transitions run via super().
                    with patch.object(c, "P3HierarchicalEnv", ObservedProposed), \
                         patch.object(c, "RSUEnv", ObservedRSU):
                        row = c.episode(cfg, name, policies[name], offset + i, cfg.dual_init,
                                        False, directory, spec["trace"])
                    validate_row(row, cfg, name, offset + i)
                    require(before == policy_digest(policies[name]), "Policy changed during evaluation")
                    if args.mode == "smoke":
                        from ndtvs_analysis import audit
                        require(audit(directory) == 0, "Smoke trace audit failed")
                        match_reference(references[name][1], cell["config"], name,
                                        provenance[name], source, row)
                    row["trace_dir"] = str(directory.relative_to(args.out))
                    rows.append(row)
                    durations.append(time.monotonic() - started)
                    c.atomic(partial, {"spec": cell, "rows": rows, "durations": durations})
                    print(f"{name} SNR={delta:+d} ep={i+1}/{count} "
                          f"QoE={row[METRICS[0]]:.6f} stall={row['stall_ratio']:.6f}", flush=True)
                require(before == policy_digest(policies[name]), "Policy changed during evaluation")
                for row in rows:
                    identity = row["episode"]
                    old = fingerprints.setdefault(identity, row["scenario_sha256"])
                    require(old == row["scenario_sha256"], "Actual mobility/fading pairing mismatch")
                c.atomic(root / "evaluation.json", {**cell, "per_episode": rows,
                         "policy_unchanged": True, "policy_sha256": before})
                c.write_rows(root / "evaluation.csv", rows)
                verified[f"{delta}/{name}"] = sha256(root / "evaluation.json")
    finally:
        budget.close()
    require(c.source_hashes() == source, "Sources changed during evaluation")
    for item in provenance.values():
        for path, expected in item["files"].items():
            require(sha256(path) == expected, "Checkpoint changed during evaluation")
    result = {"passed": True, "selection": selection, "scenario_sha256": fingerprints,
              "evaluation_sha256": verified, "manifest_sha256": digest(spec)}
    c.atomic(args.out / ("smoke_verification.json" if args.mode == "smoke" else "sweep_verification.json"), result)
    print("COMPLETE: all configurations, checkpoints, pairing and invariants verified", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "sweep"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--config", type=Path, default=TRAIN / "resolved_config.json")
    p.add_argument("--ndtvs-checkpoint", type=Path, default=RUNS / "ndtvs_seed2026_ep500/best.pt")
    p.add_argument("--rsu-checkpoint", type=Path, default=RUNS / "hppo_rsu_seed2026_ep850/best.pt")
    p.add_argument("--frame-checkpoint", type=Path, default=TRAIN / "checkpoints/frame_ep00124.pt")
    p.add_argument("--slot-checkpoint", type=Path, default=TRAIN / "checkpoints/slot_ep00124.pt")
    p.add_argument("--reference-root", type=Path, default=RUNS / "final_eval_seed2026_offset3000000")
    p.add_argument("--smoke-root", type=Path, default=RUNS / "snr_smoke_seed2026_offset3000000")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--walltime-seconds", type=float, default=82800)
    p.add_argument("--reserve-seconds", type=float, default=1800)
    args = p.parse_args()
    require(args.walltime_seconds > args.reserve_seconds >= 0, "Invalid walltime budget")
    args.out = args.out.resolve()
    protected = [args.reference_root.resolve(), args.config.resolve().parent]
    protected += [p.resolve().parent for p in (args.ndtvs_checkpoint, args.rsu_checkpoint,
                                               args.frame_checkpoint, args.slot_checkpoint)]
    for path in protected:
        require(args.out != path and path not in args.out.parents and args.out not in path.parents,
                "Output overlaps a protected checkpoint/training/final directory")
    if args.mode == "sweep":
        smoke = args.smoke_root.resolve()
        require(args.out != smoke and smoke not in args.out.parents and args.out not in smoke.parents,
                "Sweep output overlaps smoke directory")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable: no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import ndtvs_common as c
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_name(args.out.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args, c, torch)


if __name__ == "__main__":
    raise SystemExit(main())
