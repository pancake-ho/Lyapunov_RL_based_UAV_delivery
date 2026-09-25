#!/usr/bin/env python3
"""Read-only audit of this training_evidence.zip; requires numpy/pandas/matplotlib.

Outputs go to --out. Never loads checkpoints or changes training configuration.
Episode numbers in output are one-based; input CSV episode numbers are zero-based.
"""
import argparse
import hashlib
import io
import json
import fnmatch
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ppo_rows(df, prefix, algorithm, epoch_limit, complete_count):
    fields = [c for c in df if c.startswith(prefix + "/")]
    rows = df.loc[df[prefix + "/updates"].notna(), ["episode"] + fields].copy()
    require(np.isfinite(rows[fields].to_numpy()).all(), f"nonfinite {algorithm}/{prefix}")
    rows = rows.rename(columns={c: c.split("/", 1)[1] for c in fields})
    rows = rows.sort_values("updates").reset_index(drop=True)
    require(rows.updates.tolist() == list(range(1, len(rows) + 1)), "update sequence mismatch")
    require(rows.epochs_completed.between(1, epoch_limit).all(), "invalid epoch count")
    require(rows.stopped_by_kl.isin([0, 1]).all(), "invalid KL flag")
    rows["kl_threshold_reached"] = rows.stopped_by_kl.astype(bool)
    rows["actual_early_stop"] = rows.kl_threshold_reached & (rows.epochs_completed < epoch_limit)
    rows["algorithm"], rows["agent"], rows["epoch_limit"] = algorithm, prefix, epoch_limit
    rows["episode_number"] = rows.episode + 1
    rows["complete_episode"] = rows.episode < complete_count
    rows["last50_complete"] = rows.complete_episode & (rows.episode >= max(0, complete_count - 50))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trains, validations, updates, summaries = {}, {}, [], {}
    with ZipFile(args.zip) as z:
        require(z.testzip() is None, "ZIP CRC failed")
        require(len(z.namelist()) == len(set(z.namelist())), "duplicate ZIP member")
        read_json = lambda name: json.loads(z.read(name))
        read_csv = lambda name: pd.read_csv(io.BytesIO(z.read(name)))
        manifest = read_json("evidence_manifest.json")
        for name, digest in manifest["sha256"].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"hash mismatch: {name}")
        for alg in ("ndtvs", "hppo_rsu", "proposed"):
            source = "episode_summary.csv" if alg == "proposed" else "training.csv"
            df = read_csv(f"{alg}/{source}").sort_values("episode").reset_index(drop=True)
            n = len(df)
            require(df.episode.tolist() == list(range(n)), f"episode sequence: {alg}")
            summary_cols = [c for c in df if "/" not in c and c != "trace_dir"]
            require(np.isfinite(df[summary_cols].to_numpy()).all(), f"nonfinite summary: {alg}")
            require((df.frames == 30).all(), f"unexpected frame count: {alg}")
            cfg_name = ("proposed/resolved_config.json" if alg == "proposed" else
                        f"{alg}/{df.iloc[0].trace_dir}/resolved_config.json")
            cfg = read_json(cfg_name)["config"]
            epoch_limit = int(cfg["ppo_update_epochs"])
            # NDTVS also uses 4 epochs in the audited agent override; experiment.json is absent.
            stats = {"completed_episodes": n, "ppo_epoch_limit": epoch_limit,
                     "physical_violation_summary": df[["reserve_violations", "power_violations", "q_gt_qe_rate"]].sum().to_dict()}
            trains[alg] = df
            if alg != "proposed":
                stats["status"] = read_json(f"{alg}/status.json")
                vals = []
                reference_ids = None
                for record in read_json(f"{alg}/validation.json"):
                    per = pd.DataFrame(record["per_episode"])
                    ids = sorted(per.episode.tolist())
                    if reference_ids is None:
                        reference_ids = ids
                    require(ids == reference_ids and len(ids) == 10, "validation scenario mismatch")
                    metric = record["selection_metric"]
                    score = per[metric].mean() * (-1 if metric == "dpp_cost_per_user_slot" else 1)
                    require(np.isclose(score, record["selection_score"], rtol=0, atol=1e-10), "selection score mismatch")
                    vals.append({"algorithm": alg, "episode_number": record["trained_episodes"],
                                 "selection_score": score, "selection_metric": metric,
                                 "qoe": per.qoe_surrogate_per_user_slot.mean(),
                                 "stall_ratio": per.stall_ratio.mean(),
                                 "quality": per.average_quality_utility.mean(), "scenarios": len(per)})
                val = pd.DataFrame(vals)
                validations[alg] = val
                best = val.loc[val.selection_score.idxmax()]
                tail = val.tail(6)
                prior, last = tail.head(3).selection_score.mean(), tail.tail(3).selection_score.mean()
                stats.update(best_validation=best.to_dict(), final_validation=val.iloc[-1].to_dict(),
                             last6_validation_range=[float(tail.selection_score.min()), float(tail.selection_score.max())],
                             last3_vs_previous3_absolute=float(last-prior),
                             last3_vs_previous3_relative=float((last-prior)/abs(prior)))
                # trace_dir is intentionally sparse. Resolve missing links by unique
                # train_<episode> segment, then verify the actual summary values.
                # Never include validation episodes or stale live-export snapshots.
                slots = []
                for row in df.itertuples():
                    if pd.notna(row.trace_dir):
                        trace_name = f"{alg}/{row.trace_dir}/episode_summary.csv"
                    else:
                        pattern = f"{alg}/segments/*/train_{row.episode:06d}/episode_summary.csv"
                        matches = fnmatch.filter(z.namelist(), pattern)
                        require(len(matches) == 1, f"ambiguous/missing trace: {pattern}")
                        trace_name = matches[0]
                    trace = read_csv(trace_name)
                    require(len(trace) == 1, "unexpected episode trace length")
                    for col in summary_cols:
                        if col in trace:
                            require(np.isclose(getattr(row, col), trace.iloc[0][col], rtol=1e-10, atol=1e-8), f"summary mismatch: {alg}/{row.episode}/{col}")
                    if alg == "hppo_rsu":
                        slots.append(read_csv(trace_name.replace("episode_summary.csv", "slot_updates.csv")))
                prefix = "ndtvs_ppo" if alg == "ndtvs" else "frame_ppo"
                updates.append(ppo_rows(df, prefix, alg, epoch_limit, n))
                if slots:
                    updates.append(ppo_rows(pd.concat(slots), "slot_ppo", alg, epoch_limit, n))
            else:
                stats["target_episodes"] = int(read_json(cfg_name)["args"]["train_episodes"])
                stats["runtime"] = read_json("proposed/runtime.json")
                for agent in ("frame", "slot"):
                    updates.append(ppo_rows(read_csv(f"proposed/{agent}_updates.csv"), agent+"_ppo", alg, epoch_limit, n))
                scale, v = cfg["ppo_reward_scale"], cfg["lyapunov_v"]
                require(np.allclose(df.slow_reward_sum, -df.dpp_cost_total*scale), "slow reward mismatch")
                require(np.allclose(df.fast_reward_sum, -(df.dpp_cost_total-v*df.hiring_cost_total)*scale), "fast reward mismatch")
            summaries[alg] = stats

    all_updates = pd.concat(updates, ignore_index=True)
    all_updates.to_csv(args.out / "ppo_updates_corrected.csv", index=False)
    pd.concat(validations.values(), ignore_index=True).to_csv(args.out / "validation_recomputed.csv", index=False)
    ppo_summary = []
    for (alg, agent), group in all_updates.groupby(["algorithm", "agent"]):
        for scope, subset in (("all_logged", group), ("last50_complete_episodes", group[group.last50_complete])):
            ppo_summary.append({"algorithm": alg, "agent": agent, "scope": scope, "updates": len(subset),
                                "kl_threshold_reached": int(subset.kl_threshold_reached.sum()),
                                "actual_early_stop": int(subset.actual_early_stop.sum()),
                                "actual_early_stop_fraction": subset.actual_early_stop.mean(),
                                **{k: subset[k].mean() for k in ("epochs_completed", "explained_variance", "approx_kl", "clip_fraction", "grad_norm", "entropy")}})
    pd.DataFrame(ppo_summary).to_csv(args.out / "ppo_summary.csv", index=False)
    p = trains["proposed"]
    blocks = []
    for start, stop in ((0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,len(p))):
        sub = p.iloc[start:stop]
        blocks.append({"first_episode": start+1, "last_episode": stop, "episodes": len(sub),
                       **sub[["dpp_cost_per_user_slot","stall_ratio","average_quality_utility","hire_rate"]].mean().to_dict()})
    pd.DataFrame(blocks).to_csv(args.out / "proposed_training_blocks.csv", index=False)
    result = {"input_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
              "verified_manifest_files": len(manifest["sha256"]), "runs": summaries,
              "ppo_summary": ppo_summary, "proposed_blocks": blocks}
    (args.out / "training_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"ndtvs": "#228833", "hppo_rsu": "#EE7733", "proposed": "#4477AA"}
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, alg, title in zip(axs, trains, ["NDTVS", "HPPO-RSU", "Proposed"]):
        df, color = trains[alg], colors[alg]
        series = df.qoe_surrogate_per_user_slot if alg == "ndtvs" else -df.dpp_cost_per_user_slot
        x = df.episode + 1
        ax.plot(x, series, color=color, alpha=.22, linewidth=.8, label="Training episode")
        ax.plot(x, series.rolling(25, min_periods=25).mean(), color=color, linewidth=2, label="Trailing 25-episode mean")
        if alg in validations:
            val = validations[alg]
            ax.plot(val.episode_number, val.selection_score, "o--", color="#222222", markersize=3.5, linewidth=1, label="Validation: 10 scenarios")
            best = val.loc[val.selection_score.idxmax()]
            ax.axvline(best.episode_number, color="#777777", linestyle=":", linewidth=1)
            ax.text(.96, .05, f"Best validation: ep {int(best.episode_number)}", transform=ax.transAxes, ha="right")
        else:
            ax.axvspan(179, 200, color="#dddddd", alpha=.5)
            ax.set_xlim(1, 200)
            ax.text(.04, .05, "178 complete episodes\nNo validation in this ZIP", transform=ax.transAxes)
        ax.set(title=title, xlabel="Completed training episode", ylabel="QoE surrogate (higher is better)" if alg == "ndtvs" else "Negative DPP / user-slot (higher is better)")
        ax.grid(alpha=.18)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .035))
    fig.suptitle("Training progress and checkpoint validation — one training seed (2026)", y=.98)
    fig.text(.5, .015, "Panels use each algorithm's selection objective; values are not a common performance ranking. Gray area: no complete Proposed episode logged.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"training_curves.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"verified_files": result["verified_manifest_files"], "completed_episodes": {k: len(v) for k,v in trains.items()}, "ppo_update_rows": len(all_updates), "output": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
