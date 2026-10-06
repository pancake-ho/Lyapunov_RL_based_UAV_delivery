"""Validate v2 paired evaluation; plot QoE, actual stalls and PSNR utility.

Usage: plot_snr_sweep.py OUTPUT/sweep --out OUTPUT/summary
Also accepts OUTPUT/eval for nominal comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import itertools
from pathlib import Path
import numpy as np
from baseline.NDTVS.evaluation.checks import METRICS, verify_saved, require
from baseline.NDTVS.common.io import atomic, write_rows


def point(rows, metric, indices=None):
    chosen = rows if indices is None else [rows[int(i)] for i in indices]
    if metric in ("average_quality_utility", "average_received_psnr_db"):
        w = np.asarray([r["received_segments_total"] for r in chosen], dtype=float)
        return float(np.dot([r[metric] for r in chosen], w) / w.sum()) if w.sum() else None
    return float(np.mean([r[metric] for r in chosen]))


def report(root, out, bootstrap=5000):
    state = verify_saved(root)
    require(not out.exists(), "Report output must be absent; choose another --out")
    require(bootstrap > 0, "Positive bootstrap count required")
    spec, cells = state["spec"], state["cells"]
    levels, names = spec["snr_offsets_db"], spec["policy_order"]
    n = len(spec["scenario_ids"])
    rng = np.random.default_rng(572913)
    samples = rng.integers(n, size=(bootstrap, n)) if n >= 2 else None
    summaries, differences = [], []
    for delta in levels:
        for name in names:
            rows = cells[f"{delta}/{name}"]["rows"]
            for metric in METRICS:
                ci = [None, None]
                if samples is not None:
                    values = [point(rows, metric, ix) for ix in samples]
                    if all(x is not None for x in values):
                        ci = np.quantile(values, [.025, .975]).tolist()
                summaries.append({"algorithm": name, "snr_offset_db": delta, "metric": metric,
                                  "estimate": point(rows, metric), "ci95_low": ci[0], "ci95_high": ci[1], "episodes": n})
        for a, b in itertools.combinations(names, 2):
            ar, br = cells[f"{delta}/{a}"]["rows"], cells[f"{delta}/{b}"]["rows"]
            for metric in METRICS:
                av, bv = point(ar, metric), point(br, metric)
                value = av-bv if av is not None and bv is not None else None
                ci = [None, None]
                if samples is not None:
                    values = [(point(ar, metric, ix), point(br, metric, ix)) for ix in samples]
                    if all(x is not None and y is not None for x, y in values):
                        ci = np.quantile([x-y for x, y in values], [.025, .975]).tolist()
                differences.append({"difference": f"{a} - {b}", "snr_offset_db": delta, "metric": metric,
                                    "estimate": value, "ci95_low": ci[0], "ci95_high": ci[1]})
    out.mkdir(parents=True)
    write_rows(out / "means_ci.csv", summaries)
    write_rows(out / "paired_differences.csv", differences)
    atomic(out / "report.json", {"spec": spec, "means": summaries, "paired_differences": differences,
           "quality_aggregation": "pooled received-segment-weighted ratio over episodes",
           "other_aggregation": "arithmetic mean of episode metrics, equal horizons",
           "uncertainty_scope": "pointwise paired scenario bootstrap conditional on fixed checkpoints; not training-seed uncertainty",
           "bootstrap": bootstrap})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("paper_qoe_per_user_slot", "Fixed-weight NDTVS QoE / user-slot", 1),
              ("stall_time_ratio", "Playback stall time (%)", 100),
              ("average_quality_utility", "Received PSNR / 41.64", 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (metric, label, scale) in zip(axes, panels):
        for name in names:
            records = [r for r in summaries if r["metric"] == metric and r["algorithm"] == name]
            y = np.asarray([np.nan if r["estimate"] is None else r["estimate"]*scale for r in records])
            line, = ax.plot(levels, y, marker="o", label=name)
            low = [np.nan if r["ci95_low"] is None else r["ci95_low"]*scale for r in records]
            high = [np.nan if r["ci95_high"] is None else r["ci95_high"]*scale for r in records]
            ax.fill_between(levels, low, high, alpha=.15, color=line.get_color())
        ax.set(xlabel="SNR offset (dB)", ylabel=label, xticks=levels)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"performance.{ext}", dpi=180)
    plt.close(fig)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args(argv)
    return report(args.root, args.out, args.bootstrap)


if __name__ == "__main__":
    raise SystemExit(main())
