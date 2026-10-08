"""Reward magnitude diagnostics using fixed policies, before SAC training."""
import csv
from pathlib import Path
import numpy as np
import baseline.NDTVS.api as c
from baseline.UAVRTV.training.rollout import run_episode
from baseline.UAVRTV.rewards.paper import COMPONENTS
from baseline.UAVRTV.evaluation.audit import audit


def run(s, cfg, spec):
    from baseline.UAVRTV.common.checkpoint import source_hashes
    sources = source_hashes()
    root = Path(s.OUT) / "preflight"
    report_path = root / "report.json"
    if report_path.exists():
        import json
        saved = json.loads(report_path.read_text())
        if saved["spec"] != spec or saved.get("source_sha256") != sources:
            raise ValueError("Existing preflight differs; select a new OUT")
        print("PREFLIGHT cached: " + str(report_path), flush=True)
        return saved
    report = dict(spec=spec, source_sha256=sources, profiles=[], diagnostics_only=True, coefficient_changes=False)
    for index, profile in enumerate(("nohire", "hired_max", "random")):
        values = {key: [] for key in COMPONENTS}
        audits = []
        for episode in range(s.PREFLIGHT_EPISODES):
            directory = root / profile / f"ep{episode}"
            rng = np.random.default_rng(s.TRAIN_SEED + 741 + index * 31 + episode)
            run_episode(cfg, s, None, 12_000_000 + episode, directory, trace=True, rng=rng, profile=profile)
            audits.append(audit(directory))
            with (directory / "reward_components.csv").open() as f:
                for row in csv.DictReader(f):
                    for key in COMPONENTS:
                        values[key].append(float(row[key]))
        stats = {}
        for key, sequence in values.items():
            a = np.asarray(sequence)
            positive = a[a > 0]
            stats[key] = dict(mean=float(a.mean()), p95=float(np.quantile(a, .95)),
                nonzero_p95=float(np.quantile(positive, .95)) if len(positive) else 0., maximum=float(a.max()))
        quality = max(stats["quality_gain"]["mean"], 1e-12)
        flags = [key for key in COMPONENTS[1:] if stats[key]["mean"] > s.DOMINANCE_RATIO * quality]
        burst = [key for key in COMPONENTS[1:] if stats[key]["nonzero_p95"] >
                 s.DOMINANCE_RATIO * max(stats["quality_gain"]["nonzero_p95"], 1e-12)]
        report["profiles"].append(dict(profile=profile, components=stats,
            dominant_mean_terms=flags, dominant_nonzero_p95_terms=burst, audits=audits))
        print("PREFLIGHT " + profile + " " + " ".join(f"{k}={v['mean']:.4g}" for k, v in stats.items())
              + f" mean_flags={flags} burst_flags={burst}", flush=True)
    report["analytic"] = dict(hover_penalty_per_hired_slot=s.VARSIGMA * cfg.hover_energy_per_slot_j,
        nonzero_relocation_penalty=s.VARSIGMA * cfg.relocation_energy_j,
        hiring_penalty_per_hired_frame=cfg.lambda_h * cfg.hiring_cost_per_frame,
        full_stall_penalty_per_user_slot=s.PHI * cfg.slot_duration_s,
        max_quality_gain_per_successful_user_slot=s.BETA)
    c.atomic(report_path, report)
    print("PREFLIGHT report: " + str(report_path), flush=True)
    return report
