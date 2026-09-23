"""Four fixed policies, paired SNR evaluation. Install beside ndtvs_common.py.

smoke: replay one known validation scenario for every policy, with trace audit.
sweep: new test scenarios, all SNR levels, automatic resume. Exit 75 = paused.
report: validate committed results, create CSV/JSON/PNG/PDF and a compact ZIP.
No training, checkpoint selection, reward changes, or checkpoint writes.
"""
import argparse
import fcntl
import itertools
import json
import math
import os
import platform
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import snr_sweep_eval as s

POLICIES = ("proposed_124", "proposed_200", "hppo_rsu", "ndtvs")
LABELS = {"proposed_124": "Proposed ep124 (DPP-selected)",
          "proposed_200": "Proposed ep200 (final)", "hppo_rsu": "HPPO-RSU ep800", "ndtvs": "NDTVS ep175"}
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "baseline/NDTVS/runs/common_gpu"
TRAIN = ROOT / "proposed/outputs/hppo/hrl_revision_train_seed2026_job142434"
RESUMED = ROOT / "proposed/outputs/hppo/hrl_resume_job142596"
OUTPUT = BASE / "comparison_v2_hashfix_seed2026_offset5000000_n30"
CORE_HASH = "604db64d4148ef9a07c72f947fe16095b5b25f66cf58cbb43df7bafbccf6d4f8"
# The earlier snr_sweep_6201893.zip ships the same code without one final blank line.
DELIVERED_CORE_HASH = "7dc59ce06e0c531c5e8c82a449570d2791efd0c365e03a85e763f111676d6874"
SCOPE = "Pointwise paired scenario bootstrap, conditional on one training seed and fixed checkpoints; not training-seed uncertainty or simultaneous coverage."


def base_algorithm(name):
    return "proposed" if name.startswith("proposed_") else name


def check_rows(rows, cfg, name, ids, fingerprints):
    s.require(len(rows) <= len(ids), "Too many rows")
    for row, expected in zip(rows, ids):
        s.validate_row(row, cfg, base_algorithm(name), expected)
        s.require(row["q_gt_qe_rate"] == 0, "Queue admissibility violation")
        s.require(fingerprints.setdefault(expected, row["scenario_sha256"]) == row["scenario_sha256"],
                  "Actual mobility/fading pairing mismatch")


def load(args, c):
    source = c.source_hashes()
    s.require(s.digest(source) == s.AUDITED_SOURCE_DIGEST, "Training/evaluator source changed")
    core_hash = s.sha256(Path(s.__file__))
    s.require(core_hash in (CORE_HASH, DELIVERED_CORE_HASH),
              f"SNR core changed: found {core_hash}; inspect the server file before continuing")
    s.require(s.sha256(Path(c.__file__).with_name("ndtvs_analysis.py")) == s.AUDITED_ANALYSIS_DIGEST, "Trace auditor changed")
    valid = s.read_json(args.validation)
    s.require(valid["spec"]["source_sha256"] == source and valid["spec"]["qoe_weights"] == list(c.QOE_WEIGHTS), "Validation source/metric mismatch")
    s.require(valid["spec"]["scenario_ids"] == list(range(1000000,1000010)), "Unexpected validation scenarios")
    s.require(valid["spec"]["selection_metric"] == "-dpp_cost_per_user_slot", "Selection objective changed")
    for ep in ("124", "200"):
        rows = valid["per_episode"][ep]
        s.require([r["episode"] for r in rows] == valid["spec"]["scenario_ids"], "Incomplete validation")
        for key, value in valid["means"][ep].items():
            s.require(np.isclose(np.mean([r[key] for r in rows]), value, rtol=0, atol=1e-12), "Validation mean mismatch")
    s.require(min(valid["means"], key=lambda ep: valid["means"][ep]["dpp_cost_per_user_slot"]) == "124", "Review changed validation selection")
    inputs = SimpleNamespace(ndtvs_checkpoint=BASE/"ndtvs_seed2026_ep500/best.pt",
        rsu_checkpoint=BASE/"hppo_rsu_seed2026_ep850/best.pt", config=TRAIN/"resolved_config.json",
        frame_checkpoint=TRAIN/"checkpoints/frame_ep00124.pt", slot_checkpoint=TRAIN/"checkpoints/slot_ep00124.pt")
    configs, policies, provenance = s.load_policies(inputs, c)
    for mapping in (configs, policies, provenance):
        mapping["proposed_124"] = mapping.pop("proposed")
    cfg = configs["proposed_124"]
    s.require(s.physical(c.jsonable(asdict(cfg))) == s.physical(valid["spec"]["config"]), "Validation environment mismatch")
    agents = c.make_agents(replace(cfg, device="cuda"), "proposed")
    paths = [RESUMED/"checkpoints"/f"{role}_latest.pt" for role in ("frame", "slot")]
    extras = [agent.load(path) for agent, path in zip(agents, paths)]
    s.require(extras[0] == extras[1] and extras[0].get("pair_id") and extras[0]["episode"] == 199, "Expected matched ep200 pair")
    configs["proposed_200"], policies["proposed_200"] = cfg, agents
    provenance["proposed_200"] = {"selected_training_episode": 200, "pair": extras,
                                 "files": {str(p.resolve()): s.sha256(p) for p in paths}}
    for ep in (124, 200):
        expected = valid["spec"]["checkpoints"][str(ep)]
        actual = provenance[f"proposed_{ep}"]["files"]
        for role in ("frame", "slot"):
            old = [v for p,v in expected.items() if Path(p).name.startswith(role+"_")]
            new = [v for p,v in actual.items() if Path(p).name.startswith(role+"_")]
            s.require(len(old) == len(new) == 1 and old == new, f"ep{ep} {role} checkpoint changed")
    refs = {f"proposed_{ep}": valid["per_episode"][str(ep)][0] for ep in (124,200)}
    for name, run, ep in (("ndtvs","ndtvs_seed2026_ep500",175), ("hppo_rsu","hppo_rsu_seed2026_ep850",800)):
        choices = [v for v in s.read_json(BASE/run/"validation.json") if v["trained_episodes"] == ep]
        s.require(len(choices) == 1, "Missing/ambiguous baseline validation")
        first = [r for r in choices[0]["per_episode"] if r["episode"] == 1000000]
        s.require(len(first) == 1, "Missing baseline reference episode")
        refs[name] = first[0]
    for group in policies.values():
        for agent in group:
            agent.net.eval()
            s.require(all(c.torch.isfinite(t).all() for t in agent.net.state_dict().values()), "Nonfinite checkpoint")
    selection = c.jsonable({"source_sha256": source, "checkpoints": provenance,
        "configs": {name: asdict(configs[name]) for name in POLICIES}, "validation_sha256": s.sha256(args.validation),
        "reference_rows": refs, "qoe_weights": c.QOE_WEIGHTS, "selection_rule": "ep124 main by validation DPP; ep200 secondary final-checkpoint comparison"})
    return configs, policies, selection


def run(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import ndtvs_common as c
    s.require(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    configs, policies, selection = load(args, c)
    if args.mode == "sweep":
        gate = s.read_json(args.out/"smoke/verification.json")
        s.require(gate["passed"] and gate["selection"] == selection and gate["runner_sha256"] == s.sha256(__file__), "Run matching smoke first")
        verify_saved(args.out/"smoke")
    levels = (0,) if args.mode == "smoke" else s.OFFSETS
    ids = [1000000] if args.mode == "smoke" else list(range(5000000,5000000+args.episodes))
    root = args.out/args.mode
    spec = {"mode": args.mode, "selection": selection, "policy_order": list(POLICIES), "scenario_seed": 2026,
            "scenario_ids": ids, "snr_offsets_db": list(levels), "runner_sha256": s.sha256(__file__),
            "snr_core_sha256": s.sha256(Path(s.__file__)), "metric_window": "All 30 frames / 300 slots; no warmup exclusion",
            "radio_sanity": s.radio_sanity(configs["proposed_124"])}
    state_file = root/"state.json"
    if state_file.exists():
        state = s.read_json(state_file)
        s.require(state["spec"] == spec, "Resume inputs differ; use a new --out")
    else:
        s.require(not root.exists() or not any(root.iterdir()), "Nonempty output has no resumable state")
        state = {"spec": spec, "cells": {}}
        c.atomic(state_file, state)
    before = {name: s.policy_digest(policies[name]) for name in POLICIES}
    fingerprints, durations = {}, []
    for key, cell in state["cells"].items():
        delta, name = key.split("/")
        s.require(int(delta) in levels and name in POLICIES, "Unexpected saved cell")
        s.require(cell["policy_sha256"] == before[name], "Saved policy differs")
        s.require(len(cell["rows"]) == len(cell["durations"]), "Partial row/duration mismatch")
        check_rows(cell["rows"], configs[name], name, ids, fingerprints)
        durations.extend(cell["durations"])
    c.atomic(root/"attempts"/(uuid.uuid4().hex+".json"), {"python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "job_id": os.environ.get("SLURM_JOB_ID"), "git": s.git_info()})
    budget = c.Budget(args.walltime_seconds, args.reserve_seconds)
    try:
        for delta in levels:
            for name in POLICIES:
                key = f"{delta}/{name}"
                cell = state["cells"].setdefault(key, {"rows": [], "durations": [], "policy_sha256": before[name]})
                cfg = replace(configs[name], device="cuda", noise_psd_w_hz=s.BASE_NOISE*10**(-delta/10))
                for i in range(len(cell["rows"]), len(ids)):
                    if budget.expired(1.5*max(durations, default=0)):
                        print("PAUSED: resubmit identical command; completed scenarios are retained.", flush=True)
                        return 75
                    c.hrl.seed_all(2026)
                    directory = root/"episodes"/f"snr{delta:+d}_{name}_{ids[i]}_{uuid.uuid4().hex[:8]}"
                    trace = args.mode == "smoke" or i == 0
                    tick = time.monotonic()
                    observed_p = type("ObservedProposed", (s.PairingObserver,c.P3HierarchicalEnv), {})
                    observed_r = type("ObservedRSU", (s.PairingObserver,c.RSUEnv), {})
                    with patch.object(c,"P3HierarchicalEnv",observed_p), patch.object(c,"RSUEnv",observed_r):
                        row = c.episode(cfg, base_algorithm(name), policies[name], ids[i], cfg.dual_init, False, directory, trace)
                    check_rows([row], cfg, name, [ids[i]], fingerprints)
                    s.require(s.policy_digest(policies[name]) == before[name], "Policy changed during evaluation")
                    if args.mode == "smoke":
                        for metric, value in selection["reference_rows"][name].items():
                            if metric != "completion_runtime_s":
                                s.require(np.isclose(row[metric], value, rtol=1e-8, atol=1e-10), f"Validation replay mismatch: {name}/{metric}")
                    if trace:
                        from ndtvs_analysis import audit
                        s.require(audit(directory) == 0, "Trace physics/QoE audit failed")
                        row["audit_file"] = str((directory/"audit.json").relative_to(root))
                        row["audit_sha256"] = s.sha256(directory/"audit.json")
                    cell["rows"].append(row)
                    elapsed = time.monotonic()-tick
                    cell["durations"].append(elapsed)
                    durations.append(elapsed)
                    c.atomic(state_file, state)
                    print(f"{name} SNR={delta:+d} {i+1}/{len(ids)} QoE={row[s.METRICS[0]]:.6f} stall={row['stall_ratio']:.6f}", flush=True)
    finally:
        budget.close()
    s.require(c.source_hashes() == selection["source_sha256"], "Sources changed during evaluation")
    for item in selection["checkpoints"].values():
        for path, digest in item["files"].items():
            s.require(s.sha256(path) == digest, "Checkpoint changed during evaluation")
    c.atomic(root/"verification.json", {"passed": True, "selection": selection, "runner_sha256": s.sha256(__file__),
        "state_sha256": s.sha256(state_file), "scenario_sha256": fingerprints,
        "policy_unchanged": {name: before[name] == s.policy_digest(policies[name]) for name in POLICIES}})
    verify_saved(root)
    print(f"COMPLETE: {root}", flush=True)
    return 0


def verify_saved(root):
    state, gate = s.read_json(root/"state.json"), s.read_json(root/"verification.json")
    spec = state["spec"]
    s.require(gate["passed"] and gate["state_sha256"] == s.sha256(root/"state.json"), "Incomplete or modified result")
    s.require(gate["selection"] == spec["selection"] and gate["runner_sha256"] == spec["runner_sha256"], "Provenance mismatch")
    s.require(gate["policy_unchanged"] == {p: True for p in POLICIES}, "Policy changed/missing")
    s.require(spec["policy_order"] == list(POLICIES), "Policy set/order mismatch")
    smoke = spec["mode"] == "smoke"
    ids = spec["scenario_ids"]
    s.require(ids == ([1000000] if smoke else list(range(5000000,5000000+len(ids)))), "Scenario namespace/order changed")
    levels = [0] if smoke else list(s.OFFSETS)
    s.require(spec["snr_offsets_db"] == levels and spec["scenario_seed"] == 2026, "Experiment design mismatch")
    s.require(set(state["cells"]) == {f"{d}/{p}" for d in levels for p in POLICIES}, "Incomplete grid")
    fingerprints = {}
    for key, cell in state["cells"].items():
        name = key.split("/")[1]
        rows = cell["rows"]
        s.require(len(rows) == len(cell["durations"]) == len(ids), "Incomplete cell")
        cfg = SimpleNamespace(**spec["selection"]["configs"][name])
        check_rows(rows, cfg, name, ids, fingerprints)
        for i, row in enumerate(rows):
            if smoke or i == 0:
                path = (root/row["audit_file"]).resolve()
                s.require(root.resolve() in path.parents, "Invalid audit path")
                s.require(s.sha256(path) == row["audit_sha256"] and s.read_json(path)["valid"], "Missing/failed/modified trace audit")
    s.require({str(k):v for k,v in fingerprints.items()} == gate["scenario_sha256"], "Fingerprint verification mismatch")
    return state


def report(args):
    smoke = verify_saved(args.out/"smoke")
    state = verify_saved(args.out/"sweep")
    spec, cells = state["spec"], state["cells"]
    s.require(smoke["spec"]["selection"] == spec["selection"] and smoke["spec"]["runner_sha256"] == spec["runner_sha256"], "Smoke/sweep provenance differs")
    n = len(spec["scenario_ids"])
    s.require(n >= 2, "Need at least two test scenarios")
    indices = np.random.default_rng(572913).integers(0,n,(20000,n))
    means, pairs, arrays = [], [], {}
    for delta in s.OFFSETS:
        for name in POLICIES:
            rows = cells[f"{delta}/{name}"]["rows"]
            for metric in s.METRICS:
                values = np.array([r[metric] for r in rows])
                arrays[delta,name,metric] = values
                lo,hi = np.quantile(values[indices].mean(1),[.025,.975])
                means.append(dict(snr_offset_db=delta,policy=name,metric=metric,episodes=n,mean=float(values.mean()),ci95_low=float(lo),ci95_high=float(hi)))
        for a,b in itertools.combinations(POLICIES,2):
            for metric in s.METRICS:
                difference = arrays[delta,a,metric]-arrays[delta,b,metric]
                lo,hi = np.quantile(difference[indices].mean(1),[.025,.975])
                pairs.append(dict(snr_offset_db=delta,difference=f"{a} - {b}",metric=metric,mean=float(difference.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    output = args.out/"summary"
    output.mkdir(exist_ok=True)
    import csv
    for filename, rows in (("means.csv",means),("paired_differences.csv",pairs)):
        with (output/filename).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"comparison.json").write_text(json.dumps({"means":means,"paired_differences":pairs,"uncertainty_scope":SCOPE,
        "bootstrap_resamples":20000,"bootstrap_seed":572913,"input_state_sha256":s.sha256(args.out/"sweep/state.json"),
        "report_script_sha256":s.sha256(__file__),"spec":spec},indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"proposed_124":("#477db3","o","-"),"proposed_200":("#7A5195","D",":"),"hppo_rsu":("#cf4037","s","--"),"ndtvs":("#59a33b","^","-.")}
    plt.rcParams.update({"font.family":"serif","font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    panels = (("performance",((s.METRICS[0],"QoE / user-slot",1),("stall_ratio","Stall (%)",100),("average_quality_utility","Quality utility",1))),
              ("cost",(("dpp_cost_per_user_slot","DPP / user-slot",1),("original_cost_per_user_slot","Original cost / user-slot",1),("energy_consumed_j","UAV energy (MJ / episode)",1e-6))))
    for figure, metrics in panels:
        fig, axes=plt.subplots(1,3,figsize=(13,4.7))
        for ax,(metric,label,scale) in zip(axes,metrics):
            for name in POLICIES:
                values=[next(r for r in means if r["snr_offset_db"]==d and r["policy"]==name and r["metric"]==metric) for d in s.OFFSETS]
                y=np.array([r["mean"] for r in values])*scale
                lo=np.array([r["ci95_low"] for r in values])*scale; hi=np.array([r["ci95_high"] for r in values])*scale
                color,marker,line=styles[name]
                ax.plot(s.OFFSETS,y,label=LABELS[name],color=color,marker=marker,linestyle=line,markerfacecolor="none",linewidth=1.3)
                ax.fill_between(s.OFFSETS,lo,hi,color=color,alpha=.10,linewidth=0)
            ax.set(xlabel="SNR offset (dB)",ylabel=label,xticks=s.OFFSETS)
            ax.grid(alpha=.15)
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,bbox_to_anchor=(.5,.075),fontsize=9)
        fig.suptitle(f"Fixed-checkpoint comparison: {n} paired test scenarios per SNR",y=.98)
        fig.text(.5,.025,"Bands: pointwise 95% scenario-bootstrap intervals; one training seed. Original cost does not include a stall penalty.",ha="center",fontsize=8.5)
        fig.tight_layout(rect=(0,.20,1,.93))
        for ext in ("png","pdf"):fig.savefig(output/f"{figure}.{ext}",dpi=220)
        plt.close(fig)
    with ZipFile(args.out/"comparison_results.zip","w",ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file():z.write(path,str(path.relative_to(args.out)))
        for phase in ("smoke","sweep"):
            for name in ("state.json","verification.json"):z.write(args.out/phase/name,f"{phase}/{name}")
            phase_state = s.read_json(args.out/phase/"state.json")
            audits = {row["audit_file"] for cell in phase_state["cells"].values() for row in cell["rows"] if "audit_file" in row}
            for relative in sorted(audits):
                path = args.out/phase/relative
                z.write(path,str(path.relative_to(args.out)))
            for path in sorted((args.out/phase/"attempts").glob("*.json")):z.write(path,str(path.relative_to(args.out)))
    print(f"REPORT COMPLETE: {args.out/'comparison_results.zip'}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("smoke","sweep","report"))
    p.add_argument("--out",type=Path,default=OUTPUT)
    p.add_argument("--validation",type=Path,default=ROOT/"proposed_validation_124_200/validation_comparison.json")
    p.add_argument("--episodes",type=int,default=30)
    p.add_argument("--walltime-seconds",type=float,default=82800)
    p.add_argument("--reserve-seconds",type=float,default=1800)
    args=p.parse_args();args.out=args.out.resolve()
    s.require(2 <= args.episodes <= 1000,"Use 2–1000 test scenarios per SNR")
    s.require(args.walltime_seconds > args.reserve_seconds >= 0,"Invalid time budget")
    for protected in (TRAIN,RESUMED,BASE/"ndtvs_seed2026_ep500",BASE/"hppo_rsu_seed2026_ep850",args.validation.resolve().parent):
        protected=protected.resolve()
        s.require(args.out != protected and protected not in args.out.parents and args.out not in protected.parents,"Output overlaps training/validation input")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.with_name(args.out.name+".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return report(args) if args.mode=="report" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
