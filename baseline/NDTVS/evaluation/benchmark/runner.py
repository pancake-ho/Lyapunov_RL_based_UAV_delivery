"""Resumable paired scenario grid; common observers and frozen checkpoints."""
import json
import os
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from baseline.NDTVS.common.paths import HERE
from baseline.NDTVS.evaluation.checks import require, sha256, read_json
from baseline.NDTVS.evaluation.policies import policy_digest
from baseline.NDTVS.evaluation.scenario import PairingObserver
from baseline.NDTVS.evaluation.benchmark.settings import validate
from baseline.NDTVS.evaluation.benchmark.models import freeze_models, load_models
from baseline.NDTVS.evaluation.benchmark.radio import at_snr, sanity
from baseline.NDTVS.metrics.service import ServiceLogger
import baseline.NDTVS.training.rollout as rollout


def source_files():
    files = sorted(p for p in (HERE/"evaluation/benchmark").glob("*.py") if p.name != "config.py")
    files += [HERE/"metrics/service.py", HERE/"plot/benchmark.py", HERE/"plot/benchmark_animation.py"]
    return {str(p.relative_to(HERE)):sha256(p) for p in files}


def manifest(s, c, selection, sizes, mode):
    return c.jsonable(dict(schema="paired-v-snr-service-v1", mode=mode,
        seed_list=list(s.SEEDS), episodes_per_seed=1 if mode == "smoke" else s.EPISODES_PER_SEED,
        episode_offset=s.SMOKE_EPISODE_OFFSET if mode == "smoke" else s.TEST_EPISODE_OFFSET,
        snr_mode=s.SNR_MODE, snr_levels=list(s.SNR_OFFSETS_DB if s.SNR_MODE == "offset" else s.SNR_DB),
        reference_distance_m=s.REFERENCE_DISTANCE_M, cost_mode=s.COST_MODE,
        hiring_costs=s.HIRING_COSTS, cost_snr_db=s.COST_SNR_DB, qoe_cost_weight=s.QOE_COST_WEIGHT,
        selection=selection, model_sizes=sizes, source_sha256=c.source_hashes(),
        evaluator_sha256=source_files(), reward_definition=c.reward_spec(),
        metric_window="all slots including initial buffer; no warmup exclusion",
        uncertainty_scope="paired held-out scenario bootstrap, conditional on fixed models; not training-seed uncertainty",
        device=s.DEVICE, trace_all=s.TRACE_ALL))


def cell_grid(spec, configs):
    for snr in spec["snr_levels"]:
        for seed in spec["seed_list"]:
            for name in configs:
                yield dict(experiment="snr", snr_db=snr, seed=seed, model=name,
                           hiring_cost_per_frame=configs[name].hiring_cost_per_frame)
    if spec["cost_mode"] == "reevaluate":
        for cost in spec["hiring_costs"]:
            for seed in spec["seed_list"]:
                for name in configs:
                    yield dict(experiment="cost", snr_db=spec["cost_snr_db"], seed=seed, model=name,
                               hiring_cost_per_frame=cost)


def cell_key(cell):
    return "|".join(str(cell[k]) for k in ("experiment","snr_db","seed","model","hiring_cost_per_frame"))


def validate_episode(row, users, cfg, algorithm, episode):
    require(row["episode"] == episode and row["observed_slots"] == cfg.num_frames*cfg.frame_slots,
            "Incomplete scenario")
    require(row["reserve_violations"] == row["power_violations"] == 0 and row["q_gt_qe_rate"] == 0,
            "Physical/queue invariant failed")
    require(len(users) == cfg.num_users and sorted(r["user"] for r in users) == list(range(cfg.num_users)), "User coverage failed")
    require(all(r["user_slots"] == cfg.num_frames*cfg.frame_slots for r in users), "Incomplete user slots")
    require(sum(r["received_chunks"] for r in users) == row["received_segments_total"] == row["delivered_chunks_total"], "Delivery totals differ")
    require(np.isclose(np.mean([r["stall_time_ratio"] for r in users]),row["stall_time_ratio"], atol=1e-12), "Stall aggregation differs")
    require(np.isclose(np.mean([r["delivery_user_slot_ratio"] for r in users]),row["served_user_ratio"], atol=1e-12), "Legacy served-user metric differs")
    require(np.isclose(row["hiring_cost_total"],cfg.lambda_h*cfg.hiring_cost_per_frame*row["hired_uav_frames"]), "Hiring arithmetic failed")
    if algorithm != "proposed":
        require(all(row[k] == 0 for k in ("hire_rate","hiring_cost_total","energy_consumed_j","hired_uav_frames")), "RSU-only invariant failed")
    for key in ("stall_time_ratio","stall_user_slot_ratio","unique_served_user_ratio","delivery_user_slot_ratio","playback_fulfillment_ratio"):
        require(0 <= row[key] <= 1+1e-12, f"Bad metric {key}")


def run(s, mode="sweep"):
    validate(s)
    import baseline.NDTVS.api as c
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    c.require_device(s.DEVICE)
    c.torch.use_deterministic_algorithms(True)
    c.torch.backends.cudnn.benchmark = False
    c.torch.backends.cudnn.deterministic = True
    c.torch.set_num_threads(1)
    models = freeze_models(s,c)
    configs, policies, selection, sizes = load_models(models,s.DEVICE,c)
    spec = manifest(s,c,selection,sizes,mode)
    spec["configs"] = c.jsonable({name:asdict(cfg) for name,cfg in configs.items()})
    if mode == "sweep":
        smoke = verify_result(s.OUT/"smoke",c)
        expected = dict(spec,mode="smoke",episodes_per_seed=1,episode_offset=s.SMOKE_EPISODE_OFFSET)
        require(smoke["spec"] == expected, "Run smoke with these exact models/settings before sweep")
    root = s.OUT/mode
    state_path = root/"state.json"
    if state_path.exists():
        require(s.RESUME, "Result exists; enable RESUME or change OUT")
        state = read_json(state_path)
        require(state["spec"] == spec, "Resume experiment differs; use a new OUT")
    else:
        state = dict(spec=spec,cells={})
        c.atomic(state_path,state)
    c.atomic(root/"radio_sanity.json", sanity(next(iter(configs.values())),s))
    c.write_rows(root/"model_size.csv",sizes)
    c.atomic(root/"model_size.json",dict(policies=sizes,
        combined={name:sum(r["parameters"] for r in sizes if r["model"] == name) for name in configs},
        note="Combined proposed includes both frame and slot networks; shared weights counted once per network."))
    fingerprints, durations = {}, []
    ids = list(range(spec["episode_offset"],spec["episode_offset"]+spec["episodes_per_seed"]))
    for cell in state["cells"].values():
        require(len(cell["rows"]) <= len(ids), "Too many saved scenarios")
        for i,row in enumerate(cell["rows"]):
            require(row["episode"] == ids[i], "Saved episode sequence changed")
            key = f"{cell['seed']}:{row['episode']}"
            require(fingerprints.setdefault(key,row["scenario_sha256"]) == row["scenario_sha256"], "Saved scenario pairing failed")
            verify_sidecars(root,row,c)
            durations.append(row["elapsed_s"])
    before = {name:policy_digest(policies[name]) for name in configs}
    budget, completed = c.Budget(s.WALLTIME_SECONDS,s.RESERVE_SECONDS), 0
    observed_p = type("BenchmarkProposed",(PairingObserver,rollout.P3HierarchicalEnv),{})
    observed_r = type("BenchmarkRSU",(PairingObserver,rollout.RSUEnv),{})
    try:
        for condition in cell_grid(spec,configs):
            key, name = cell_key(condition), condition["model"]
            cell = state["cells"].setdefault(key,dict(condition,rows=[]))
            algorithm = selection[name]["algorithm"]
            cfg = at_snr(configs[name],s.SNR_MODE,condition["snr_db"],s.REFERENCE_DISTANCE_M)
            cfg = replace(cfg,seed=condition["seed"],device=s.DEVICE,
                          hiring_cost_per_frame=condition["hiring_cost_per_frame"],episode_offset=0)
            for i in range(len(cell["rows"]),len(ids)):
                if budget.expired(1.5*max(durations,default=0)) or (s.MAX_NEW_EPISODES and completed >= s.MAX_NEW_EPISODES):
                    print("PAUSED: completed scenarios saved; resubmit the same config.",flush=True)
                    return 75
                c.hrl.seed_all(condition["seed"])
                directory = root/"episodes"/(f"{condition['experiment']}_{name}_snr{condition['snr_db']}_cost{condition['hiring_cost_per_frame']}_seed{condition['seed']}_ep{ids[i]}_"+uuid.uuid4().hex[:8])
                trace, logs = s.TRACE_ALL or i == 0, []
                class CapturedLogger(ServiceLogger):
                    def __init__(self,*a,**kw):
                        super().__init__(*a,**kw)
                        logs.append(self)
                began = time.monotonic()
                with patch.object(rollout,"QoELogger",CapturedLogger), patch.object(rollout,"P3HierarchicalEnv",observed_p), patch.object(rollout,"RSUEnv",observed_r):
                    row = c.episode(cfg,algorithm,policies[name],ids[i],selection[name]["dual"],False,directory,trace)
                users = logs[0].user_rows()
                validate_episode(row,users,cfg,algorithm,ids[i])
                pairing = f"{condition['seed']}:{ids[i]}"
                require(fingerprints.setdefault(pairing,row["scenario_sha256"]) == row["scenario_sha256"], "Actual mobility/fading differs across policies/SNR/cost")
                require(policy_digest(policies[name]) == before[name], "Evaluation changed policy/buffer/update count")
                c.atomic(directory/"per_user.json",users)
                c.write_rows(directory/"per_user.csv",users)
                row.update(per_user_file=str((directory/"per_user.json").relative_to(root)),
                           per_user_sha256=sha256(directory/"per_user.json"),
                           episode_dir=str(directory.relative_to(root)))
                if s.QOE_COST_WEIGHT is not None:
                    cost = s.QOE_COST_WEIGHT*row["hiring_cost_total"]/(cfg.num_users*cfg.num_frames*cfg.frame_slots)
                    row.update(hiring_penalty_per_user_slot=cost,
                               cost_augmented_qoe_per_user_slot=row["paper_qoe_per_user_slot"]-cost,
                               cost_augmented_qoe_final_per_user=row["paper_qoe_final_per_user"]-
                               s.QOE_COST_WEIGHT*row["hiring_cost_total"]/cfg.num_users)
                if trace:
                    from baseline.NDTVS.analysis.cli import audit
                    require(audit(directory) == 0,"Trace physics/QoE audit failed")
                    row.update(audit_file=str((directory/"audit.json").relative_to(root)),
                               audit_sha256=sha256(directory/"audit.json"),
                               trace_sha256=sha256(directory/"trace.jsonl"))
                row["elapsed_s"] = time.monotonic()-began
                cell["rows"].append(row)
                durations.append(row["elapsed_s"])
                completed += 1
                c.atomic(state_path,state)
                print(f"{mode}: {name} {s.SNR_MODE}={condition['snr_db']} seed={condition['seed']} {i+1}/{len(ids)} stall={row['stall_time_ratio']:.4f} coverage={row['unique_served_user_ratio']:.4f}",flush=True)
    finally:
        budget.close()
    require(c.source_hashes() == spec["source_sha256"] and source_files() == spec["evaluator_sha256"], "Source changed during evaluation")
    for item in read_json(s.OUT/"inputs/manifest.json")["files"]:
        require(sha256(item["snapshot"]) == item["sha256"], "Snapshot changed during evaluation")
    require(all(policy_digest(policies[name]) == before[name] for name in configs), "Policy changed")
    c.atomic(root/"verification.json",dict(passed=True,state_sha256=sha256(state_path),
        scenario_sha256=fingerprints,policy_unchanged={name:True for name in configs}))
    verify_result(root,c)
    return 0


def verify_sidecars(root,row,c):
    path = root/row["per_user_file"]
    require(root.resolve() in path.resolve().parents and sha256(path) == row["per_user_sha256"], "Changed/invalid user data")
    users = read_json(path)
    require(sum(u["received_chunks"] for u in users) == row["received_segments_total"], "Corrupt user totals")
    if "audit_file" in row:
        audit = root/row["audit_file"]
        require(root.resolve() in audit.resolve().parents and sha256(audit) == row["audit_sha256"] and read_json(audit)["valid"], "Changed/failed audit")
        require(sha256(root/row["episode_dir"]/"trace.jsonl") == row["trace_sha256"], "Trace changed after audit")


def verify_result(root,c):
    state, gate = read_json(root/"state.json"),read_json(root/"verification.json")
    require(gate["passed"] and sha256(root/"state.json") == gate["state_sha256"], "Incomplete/changed results")
    spec = state["spec"]
    require(gate["policy_unchanged"] == {name:True for name in spec["selection"]}, "Missing policy verification")
    configs = {name:None for name in spec["selection"]}
    # Construct expected condition keys without loading any models.
    expected = set()
    for db in spec["snr_levels"]:
        for seed in spec["seed_list"]:
            for name in configs:
                expected.add(("snr",db,seed,name))
    if spec["cost_mode"] == "reevaluate":
        for cost in spec["hiring_costs"]:
            for seed in spec["seed_list"]:
                for name in configs:
                    expected.add(("cost",spec["cost_snr_db"],seed,name,cost))
    actual, fingerprints = set(),{}
    for key,cell in state["cells"].items():
        require(key == cell_key(cell), "Invalid cell key")
        signature = (cell["experiment"],cell["snr_db"],cell["seed"],cell["model"])
        if cell["experiment"] == "cost":
            signature += (cell["hiring_cost_per_frame"],)
        actual.add(signature)
        require(len(cell["rows"]) == spec["episodes_per_seed"], "Incomplete cell")
        require("audit_file" in cell["rows"][0], "First scenario must be audited")
        for i,row in enumerate(cell["rows"]):
            require(row["episode"] == spec["episode_offset"]+i, "Wrong episode IDs")
            verify_sidecars(root,row,c)
            scenario = f"{cell['seed']}:{row['episode']}"
            require(fingerprints.setdefault(scenario,row["scenario_sha256"]) == row["scenario_sha256"], "Pairing failed")
    require(expected == actual and len(actual) == len(state["cells"]), "Result grid differs")
    require(fingerprints == gate["scenario_sha256"], "Fingerprint manifest differs")
    return state
