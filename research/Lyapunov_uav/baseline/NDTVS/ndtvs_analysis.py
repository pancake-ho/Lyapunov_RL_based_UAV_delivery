"""Audit one episode, plot training diagnostics, or compare paired evaluations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ndtvs_common import atomic, load_bundle, qoe_terms, read_config, write_rows
from hppo.verify_trace import _verify_physics


def audit(root):
    cfg = read_config(root / "resolved_config.json")
    records = [json.loads(x) for x in (root / "trace.jsonl").read_text().splitlines()]
    errors, active, frame, slot, ended = [], False, 0, 0, False
    episode_ends = 0
    previous_quality = {u: -1 for u in range(cfg.num_users)}
    algorithm = records[0].get("algorithm") if records else None
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(bool(records) and records[0]["event"] == "baseline_start", "missing baseline_start")
    check(algorithm in ("ndtvs", "hppo_rsu", "proposed"), "unknown algorithm")
    episode_id = records[0].get("episode") if records else None
    qoe_total = 0.0
    for r in records[1:]:
        check(not ended, "event after end")
        ev = r["event"]
        if ev in ("frame_start", "slot", "frame_end", "episode_end"):
            check(r["episode"] == episode_id, "episode identity")
        if ev in ("frame_start", "slot", "frame_end"):
            check(set(r["regions"]) == {str(m) for m in range(cfg.num_regions)}, "region coverage")
        if ev == "frame_start":
            check(not active and r["frame"] == frame, "frame order")
            active, slot = True, 0
            for rg in r["regions"].values():
                if algorithm != "proposed":
                    check(rg["executed_hire"] == 0 and not rg["executed_uav_users"], "UAV enabled")
        elif ev == "slot":
            check(active and r["frame"] == frame and r["slot_in_frame"] == slot, "slot order")
            users = [u for rg in r["regions"].values() for u in rg["users"]]
            check(sorted(u["user"] for u in users) == list(range(cfg.num_users)), "user coverage")
            for m, rg in r["regions"].items():
                total = 0.0
                for u in rg["users"]:
                    check(u["last_quality_before"] == previous_quality[u["user"]], "quality continuity")
                    value, switch = qoe_terms(u, cfg)
                    check(np.isclose(value, u["qoe_surrogate"]), "QoE arithmetic")
                    check(np.isclose(switch, u["switch_magnitude"]), "switch arithmetic")
                    check(u["delivered"] in (0, u["req_chunks"]), "atomic delivery violated")
                    check(0 <= u["q_after"] <= cfg.large_queue_level + 1e-8, "queue bound")
                    if algorithm != "proposed":
                        check(u["provider"] in (0, 1), "UAV provider")
                    if u["delivered"]:
                        previous_quality[u["user"]] = u["req_quality"]
                    total += value
                if algorithm == "ndtvs":
                    check(np.isclose(total, r["rewards"][m]["training"]), "NDTVS reward")
                qoe_total += total
            slot += 1
        elif ev == "frame_end":
            check(active and slot == cfg.frame_slots, "incomplete frame")
            active, frame = False, frame + 1
        elif ev == "episode_end":
            episode_ends += 1
            check(not active and frame == cfg.num_frames, "incomplete episode")
            check(np.isclose(r["qoe_surrogate_per_user_slot"],
                             qoe_total / (cfg.num_users * cfg.num_frames * cfg.frame_slots)), "QoE summary")
        elif ev == "baseline_end":
            check(not active and frame == cfg.num_frames and r["status"] == "complete", "incomplete run")
            ended = True
        elif ev not in ("slot_ppo_update", "frame_ppo_update"):
            check(False, "unknown event: " + ev)
    check(ended, "missing baseline_end")
    check(episode_ends == 1, "episode_end count")
    passed, failed, details = _verify_physics(root)
    errors.extend(details)
    result = {"valid": not errors and not failed, "physics_checks_passed": sum(passed.values()),
              "physics_checks_failed": dict(failed), "errors": errors[:50]}
    atomic(root / "audit.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


def progress(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import csv
    rows = list(csv.DictReader((root / "training.csv").open()))
    vals = json.loads((root / "validation.json").read_text())
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    scores = np.asarray([v["selection_score"] for v in vals])
    # A descriptive flag, NOT a statistical proof or automatic stopping rule.
    relative_change = None
    plateau = False
    if len(scores) >= 6:
        before, after = scores[-6:-3].mean(), scores[-3:].mean()
        relative_change = float(abs(after - before) / max(abs(before), 1.0))
        plateau = relative_change <= 0.01
    diagnostic = {"execution_status": status, "training_episodes": len(rows),
                  "validation_points": len(vals), "plateau_candidate": plateau,
                  "last_three_vs_previous_three_change": relative_change,
                  "convergence_certified": False,
                  "interpretation": "A plateau may be a poor local solution. Inspect stalls, quality, failures and independent training seeds."}
    atomic(root / "diagnostics.json", diagnostic)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes[:2], ("qoe_surrogate_per_user_slot", "stall_ratio"), ("Training QoE surrogate", "Training stall ratio")):
        y = np.asarray([float(r[key]) for r in rows])
        ax.plot(np.arange(1, len(y) + 1), y, alpha=.3, lw=.8)
        w = min(25, len(y))
        if w:
            ax.plot(np.arange(w, len(y) + 1), np.convolve(y, np.ones(w) / w, "valid"), lw=1.5)
        ax.set(title=title, xlabel="Completed episodes")
        ax.grid(alpha=.2)
    axes[2].plot([v["trained_episodes"] for v in vals], scores, marker="o")
    axes[2].set(title="Held-out validation selection score", xlabel="Completed episodes")
    axes[2].grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(root / f"learning_diagnostics.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostic, indent=2))
    return 0


def audit_run(root):
    saved = load_bundle(root / "latest.pt")
    results = []
    for row in saved["rows"]:
        if not row["trace_dir"]:
            continue
        directory = root / row["trace_dir"]
        try:
            code = audit(directory)
            detail = json.loads((directory / "audit.json").read_text())
        except (OSError, ValueError, KeyError, IndexError) as exc:
            code, detail = 1, {"valid": False, "errors": [str(exc)]}
        results.append({"episode": row["episode"], "directory": row["trace_dir"],
                        "exit_code": code, **detail})
    result = {"valid": all(r["exit_code"] == 0 for r in results),
              "checkpoint_completed_episodes": saved["next_episode"],
              "traced_episodes_checked": len(results), "per_episode": results}
    atomic(root / "audit_run.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
    return 0 if result["valid"] else 1


def comparable_config(c):
    ignore = {"seed", "episode_offset", "train_episodes", "eval_episodes", "save_every_episodes",
              "device", "torch_num_threads", "hidden_dims", "slot_update_every_frames",
              "frame_update_every_episodes", "write_jsonl_trace", "write_human_debug_log",
              "log_hidden_csi", "log_observation_vectors", "console_log_every_slots"}
    return {k: v for k, v in c.items() if k not in ignore and not k.startswith("ppo_")}


def compare(paths, out):
    data = [json.loads((p / "evaluation.json").read_text()) for p in paths]
    reference = data[0]
    if len({d["algorithm"] for d in data}) != len(data):
        raise ValueError("Use one evaluation per algorithm; repeat the comparison for each training seed")
    for d in data[1:]:
        for key in ("offset", "scenario_seed", "qoe_weights"):
            if d[key] != reference[key]:
                raise ValueError(f"Unpaired evaluations: {key}")
        if comparable_config(d["config"]) != comparable_config(reference["config"]):
            raise ValueError("Physical/control configuration differs")
        if [r["episode"] for r in d["per_episode"]] != [r["episode"] for r in reference["per_episode"]]:
            raise ValueError("Episode identities differ")
        # The adapter file is common too; differing source versions require a fresh evaluation.
        if d["source_sha256"] != reference["source_sha256"]:
            raise ValueError("Evaluation source versions differ")
    metrics = ("qoe_surrogate_per_user_slot", "stall_ratio", "average_quality_utility",
               "switch_magnitude_per_user_slot", "request_failure_ratio", "delivered_chunks_per_user_slot",
               "dpp_cost_per_user_slot", "original_cost_per_user_slot", "hire_rate", "energy_consumed_j")
    summaries = [{"algorithm": d["algorithm"], "episodes": len(d["per_episode"]),
                  **{k: float(np.mean([r[k] for r in d["per_episode"]])) for k in metrics}} for d in data]
    rng = np.random.default_rng(572913)
    pairs = []
    for i, a in enumerate(data):
        for b in data[i + 1:]:
            for key in metrics:
                delta = np.asarray([x[key] - y[key] for x, y in zip(a["per_episode"], b["per_episode"])])
                interval = [None, None]
                if len(delta) >= 2:
                    samples = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(1)
                    interval = np.quantile(samples, [.025, .975]).tolist()
                pairs.append({"difference": a["algorithm"] + " - " + b["algorithm"], "metric": key,
                              "mean": float(delta.mean()), "ci95_low": interval[0], "ci95_high": interval[1]})
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "means.csv", summaries)
    write_rows(out / "paired_differences.csv", pairs)
    selected_episodes = {}
    for d in data:
        provenance = d["provenance"]
        count = provenance.get("trained_episodes")
        if count is None and provenance.get("pair"):
            last = provenance["pair"][0].get("episode")
            count = last + 1 if last is not None else None
        selected_episodes[d["algorithm"]] = count
    atomic(out / "comparison.json", {"means": summaries, "paired_differences": pairs,
           "selected_checkpoint_training_episodes": selected_episodes,
           "same_selected_training_episode": None not in selected_episodes.values()
                                             and len(set(selected_episodes.values())) == 1,
           "checkpoint_provenance": {d["algorithm"]: d["provenance"] for d in data},
           "uncertainty_scope": "Bootstrap across paired test scenarios, conditional on these trained checkpoints; not across independent training seeds.",
           "inputs": [str(p.resolve()) for p in paths]})
    print(json.dumps(summaries, indent=2))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "audit-run", "progress"):
        sub.add_parser(name).add_argument("directory", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("directories", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "audit":
        return audit(a.directory)
    if a.command == "progress":
        return progress(a.directory)
    if a.command == "audit-run":
        return audit_run(a.directory)
    return compare(a.directories, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
