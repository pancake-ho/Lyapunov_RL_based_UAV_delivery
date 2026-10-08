"""Frozen validation-selected SAC and paired held-out SNR scenarios."""
import math
import shutil
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

import baseline.NDTVS.api as c
from baseline.NDTVS.evaluation.checks import require, sha256, read_json
from baseline.NDTVS.evaluation.benchmark.radio import at_snr
from baseline.UAVRTV.common.checkpoint import read, policy_digest, source_hashes
from baseline.UAVRTV.environment.shared import SharedUAVRTVEnv
from baseline.UAVRTV.models.sac import SACAgent
from baseline.UAVRTV.training.rollout import run_episode
from baseline.UAVRTV.evaluation.audit import audit


def selected(path, s):
    saved = read(path)
    require(saved["kind"] == "best", "Use fixed-validation selected best.pt")
    points = [v for v in saved["validation"] if v["trained_episodes"] == saved["next_episode"]]
    require(len(points) == 1 and points[0]["selection_score"] == saved["best"], "Incomplete best validation")
    d = saved["spec"]["common_config"]
    cfg = c.HPPOConfig(**{k: tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
                         if isinstance(v, list) else v for k, v in d.items()})
    reward = saved["spec"]["reward"]
    settings = SimpleNamespace(**vars(s))
    for k, value in saved["spec"]["learning"].items():
        setattr(settings, k, value)
    for k, field in (("BETA", "beta"), ("DELTA", "delta_per_bps"), ("PHI", "phi_per_second"),
                     ("VARSIGMA", "varsigma_per_joule"), ("REWARD_SCALE", "scale")):
        setattr(settings, k, reward[field])
    cfg = replace(cfg, device=s.DEVICE)
    shape = SharedUAVRTVEnv(cfg)
    agent = SACAgent(shape.obs_dim, shape.act_dim, settings)
    agent.restore(saved["agent"])
    for net in agent.networks().values():
        net.eval()
    return cfg, settings, agent, saved


def sweep(s):
    root = Path(s.EVAL_OUT)
    source = Path(s.OUT) / "best.pt"
    snapshot = root / "inputs/best.pt"
    manifest_path = root / "inputs/manifest.json"
    if manifest_path.exists():
        frozen = read_json(manifest_path)
        require(frozen["source"] == str(source.resolve()) and sha256(snapshot) == frozen["sha256"], "Frozen model changed; choose a new EVAL_OUT")
    else:
        fingerprint = sha256(source)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temp = snapshot.with_name(snapshot.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            shutil.copyfile(source, temp)
            require(sha256(temp) == fingerprint == sha256(source), "best.pt advanced during snapshot; retry")
            temp.replace(snapshot)
        finally:
            temp.unlink(missing_ok=True)
        frozen = dict(source=str(source.resolve()), sha256=fingerprint)
        c.atomic(manifest_path, frozen)
    cfg, settings, agent, checkpoint = selected(snapshot, s)
    spec = c.jsonable(dict(checkpoint=frozen, selected_training_episode=checkpoint["next_episode"],
        validation_best=checkpoint["best"], seeds=list(s.TEST_SEEDS), episodes=s.TEST_EPISODES,
        offset=s.TEST_OFFSET, snr_offsets=list(s.SNR_OFFSETS_DB), source_sha256=source_hashes(),
        config=asdict(cfg), training_reward=checkpoint["spec"]["reward"], device=s.DEVICE))
    state_path = root / "state.json"
    state = read_json(state_path) if state_path.exists() else dict(spec=spec, rows=[])
    require(state["spec"] == spec, "Evaluation settings differ; choose a new EVAL_OUT")
    for row in state["rows"]:
        require(sha256(root / row["summary_file"]) == row["summary_sha256"], "Changed saved summary")
        require(sha256(root / row["per_user_file"]) == row["per_user_sha256"], "Changed saved per-user data")
        if "audit_file" in row:
            require(sha256(root / row["audit_file"]) == row["audit_sha256"] and read_json(root / row["audit_file"])["valid"], "Changed/failed audit")
    finished = {(r["snr_offset_db"], r["scenario_seed"], r["episode"]) for r in state["rows"]}
    require(len(finished) == len(state["rows"]), "Duplicate evaluation scenarios")
    fingerprints = {}
    for row in state["rows"]:
        identity = (row["scenario_seed"], row["episode"])
        require(fingerprints.setdefault(identity, row["scenario_sha256"]) == row["scenario_sha256"], "Saved pairing changed")
    before, began, completed = policy_digest(agent), time.monotonic(), 0
    torch.set_num_threads(s.TORCH_THREADS)
    for snr in s.SNR_OFFSETS_DB:
        for seed in s.TEST_SEEDS:
            evaluated = replace(at_snr(cfg, "offset", snr, None), seed=seed, episode_offset=0)
            for i in range(s.TEST_EPISODES):
                number = s.TEST_OFFSET + i
                if (snr, seed, number) in finished:
                    continue
                if time.monotonic() - began >= s.WALLTIME_SECONDS - s.RESERVE_SECONDS or (
                        s.MAX_NEW_EPISODES and completed >= s.MAX_NEW_EPISODES):
                    c.atomic(state_path, state)
                    print("EVAL PAUSED; resubmit with the same settings", flush=True)
                    return 75
                directory = root / "episodes" / f"snr{snr}_seed{seed}_ep{number}_{uuid.uuid4().hex[:8]}"
                row = run_episode(evaluated, settings, agent, number, directory, trace=i == 0)
                if i == 0:
                    audit(directory)
                    row.update(audit_file=str((directory / "audit.json").relative_to(root)), audit_sha256=sha256(directory / "audit.json"))
                identity = (seed, number)
                require(fingerprints.setdefault(identity, row["scenario_sha256"]) == row["scenario_sha256"], "SNR changed the exogenous scenario")
                require(policy_digest(agent) == before, "Evaluation changed SAC weights/temperature/update count")
                row.update(snr_offset_db=snr, scenario_seed=seed,
                    summary_file=str((directory / "summary.json").relative_to(root)), summary_sha256=sha256(directory / "summary.json"),
                    per_user_file=str((directory / "per_user.json").relative_to(root)), per_user_sha256=sha256(directory / "per_user.json"))
                state["rows"].append(row)
                completed += 1
                c.atomic(state_path, state)
                print(f"uavrtv eval offset={snr} seed={seed} {i+1}/{s.TEST_EPISODES} stall={row['stall_time_ratio']:.4f}", flush=True)
    c.write_rows(root / "episodes.csv", state["rows"])
    metrics = ("stall_time_ratio", "stall_user_slot_ratio", "average_quality_utility", "unique_served_user_ratio",
        "delivery_user_slot_ratio", "playback_fulfillment_ratio", "hire_rate", "energy_consumed_j",
        "hiring_cost_total", "uavrtv_reward_per_region_slot", "paper_qoe_per_user_slot")
    means = []
    for snr in s.SNR_OFFSETS_DB:
        cell = [r for r in state["rows"] if r["snr_offset_db"] == snr]
        summaries = {}
        for key in metrics:
            values = [r[key] for r in cell if key != "average_quality_utility" or r["quality_utility_defined"]]
            summaries[key] = float(np.mean(values)) if values else None
        means.append(dict(snr_offset_db=snr, episodes=len(cell),
            quality_defined_episodes=sum(r["quality_utility_defined"] for r in cell), **summaries))
    c.write_rows(root / "means.csv", means)
    c.atomic(root / "verification.json", dict(passed=True, policy_unchanged=policy_digest(agent) == before,
        state_sha256=sha256(state_path), scenarios=len(state["rows"]),
        uncertainty_scope="fixed trained SAC; scenario seeds, not independent training seeds"))
    return 0
