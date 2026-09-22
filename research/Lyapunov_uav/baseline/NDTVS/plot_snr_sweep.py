"""Validate and aggregate a complete paired SNR sweep; export three 1x3 figures."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from snr_sweep_eval import ALGORITHMS, OFFSETS, METRICS, BASE_NOISE, digest, read_json, require, sha256

LABELS = {"proposed": "Proposed", "hppo_rsu": "HPPO-RSU", "ndtvs": "NDTVS"}
STYLES = {"proposed": dict(color="#477db3", marker="o", linestyle="-"),
          "hppo_rsu": dict(color="#cf4037", marker="s", linestyle="--"),
          "ndtvs": dict(color="#59a33b", marker="^", linestyle="-.")}
SCOPE = ("Paired test-scenario bootstrap uncertainty conditional on fixed trained checkpoints; "
         "not training-seed uncertainty. Intervals are pointwise, not simultaneous.")


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def collect(root):
    manifest = read_json(root / "sweep_manifest.json")
    verification = read_json(root / "sweep_verification.json")
    require(verification["passed"] and manifest["mode"] == "sweep", "Sweep is incomplete")
    require(verification["manifest_sha256"] == digest(manifest), "Manifest changed")
    require(verification["selection"] == manifest["selection"], "Selection changed")
    require(manifest["snr_offsets_db"] == list(OFFSETS) and manifest["offset"] == 4_000_000
            and manifest["scenario_seed"] == 2026, "Unexpected sweep design")
    count = manifest["episodes"]
    expected = list(range(4_000_000, 4_000_000 + count))
    data, fingerprints, configs = {}, {}, {}
    for delta in OFFSETS:
        for name in ALGORITHMS:
            path = root / f"snr_{delta:+03d}dB" / name / "evaluation.json"
            require(sha256(path) == verification["evaluation_sha256"][f"{delta}/{name}"],
                    "Evaluation file changed after verification")
            d = read_json(path)
            require(d["manifest_sha256"] == digest(manifest) and d["policy_unchanged"],
                    "Evaluation provenance mismatch")
            require(d["algorithm"] == name and d["snr_offset_db"] == delta,
                    "Algorithm/SNR mismatch")
            require(d["source_sha256"] == manifest["selection"]["source_sha256"]
                    and d["provenance"] == manifest["selection"]["checkpoints"][name],
                    "Sources/checkpoints differ")
            require(d["original_config"] == manifest["selection"]["configs"][name],
                    "Original configuration differs")
            nominal = dict(d["config"])
            require(nominal["noise_psd_w_hz"] == BASE_NOISE * 10 ** (-delta / 10), "Wrong noise PSD")
            nominal["noise_psd_w_hz"] = BASE_NOISE
            require(configs.setdefault(name, nominal) == nominal, "Other settings vary with SNR")
            rows = d["per_episode"]
            require([r["episode"] for r in rows] == expected, "Episode pairing/order mismatch")
            for row in rows:
                ep = row["episode"]
                require(fingerprints.setdefault(ep, row["scenario_sha256"]) == row["scenario_sha256"],
                        "Realized exogenous scenarios differ")
                require(verification["scenario_sha256"][str(ep)] == row["scenario_sha256"],
                        "Scenario fingerprint differs from verification")
                require(np.isfinite([row[k] for k in METRICS]).all(), "Nonfinite metrics")
                require(row["reserve_violations"] == row["power_violations"] == 0, "Invariant failure")
                if name != "proposed":
                    require(row["hire_rate"] == row["energy_consumed_j"] == 0, "No-UAV invariant failure")
            data[delta, name] = rows
    return manifest, data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, help="Default: ROOT/summary; must be absent")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()
    require(args.bootstrap >= 1000, "Use at least 1000 bootstrap resamples")
    manifest, data = collect(args.root)
    out = args.out or args.root / "summary"
    require(not out.exists(), "Summary exists: choose a new --out to preserve it")
    means, pairs = [], []
    n = manifest["episodes"]
    # One resampling matrix preserves pairing across algorithms, metrics and SNR points.
    indices = np.random.default_rng(572913).integers(0, n, size=(args.bootstrap, n))
    for delta in OFFSETS:
        for name in ALGORITHMS:
            means.append({"snr_offset_db": delta, "algorithm": name, "episodes": n,
                          **{k: float(np.mean([r[k] for r in data[delta, name]])) for k in METRICS}})
        for a, b in itertools.combinations(ALGORITHMS, 2):
            for key in METRICS:
                difference = np.asarray([x[key] - y[key]
                                         for x, y in zip(data[delta, a], data[delta, b])])
                lo, hi = (np.quantile(difference[indices].mean(1), [.025, .975]).tolist()
                          if n >= 2 else (None, None))
                pairs.append({"snr_offset_db": delta, "difference": f"{a} - {b}", "metric": key,
                              "mean": float(difference.mean()), "ci95_low": lo, "ci95_high": hi})
    (out / "plots").mkdir(parents=True)
    table(out / "snr_means.csv", means)
    table(out / "snr_paired_differences.csv", pairs)
    dump(out / "snr_summary.json", {"means": means, "paired_differences": pairs,
         "uncertainty_scope": SCOPE, "bootstrap_resamples": args.bootstrap, "bootstrap_seed": 572913,
         "aggregation": "Arithmetic mean of episode-level metrics; energy is J/episode.",
         "no_uav": "HPPO-RSU and NDTVS hiring/energy are structurally zero; overlapping curves are expected.",
         "interpretation": "No checkpoint reselection; nominal channel observations and completion rollouts use effective noise PSD.",
         "input_manifest": manifest, "plot_script_sha256": sha256(__file__)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": .6, "xtick.direction": "in", "ytick.direction": "in"})
    figures = (
        ("qoe_vs_snr", ((METRICS[1], "Stall Ratio", 1), (METRICS[2], "Average Quality Utility", 1),
                        (METRICS[3], "Quality Switching / User-slot", 1))),
        ("service_vs_snr", ((METRICS[0], "QoE Surrogate / User-slot", 1),
                            (METRICS[5], "Delivered Chunks / User-slot", 1),
                            (METRICS[4], "Request Failure Ratio", 1))),
        ("resource_vs_snr", ((METRICS[6], "DPP Cost / User-slot", 1), (METRICS[8], "UAV Hiring Rate", 1),
                             (METRICS[9], "UAV Energy [MJ / episode]", 1e-6))),
    )
    for filename, panels in figures:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.15))
        for i, (ax, (key, label, scale)) in enumerate(zip(axes, panels)):
            for name in ALGORITHMS:
                y = [next(r[key] for r in means if r["algorithm"] == name and r["snr_offset_db"] == d)
                     * scale for d in OFFSETS]
                ax.plot(OFFSETS, y, label=LABELS[name], linewidth=1.25, markersize=4.5,
                        markerfacecolor="none", **STYLES[name])
            ax.set(xlabel="Channel SNR Offset [dB]", ylabel=label, xticks=OFFSETS)
            ax.text(.5, -.29, f"({chr(97+i)})", transform=ax.transAxes, ha="center")
            ax.margins(x=.04, y=.10)
            if key in (METRICS[1], METRICS[2], METRICS[4], METRICS[8]):
                bottom, top = ax.get_ylim()
                ax.set_ylim(max(bottom, -.015), min(top, 1.015))
            ax.grid(False)
        axes[0].legend(loc="best", frameon=True, fancybox=False, edgecolor=".8")
        fig.subplots_adjust(left=.075, right=.995, bottom=.25, top=.96, wspace=.48)
        for ext in ("png", "pdf"):
            fig.savefig(out / "plots" / f"{filename}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved summaries and PNG/PDF figures: {out.resolve()}")


if __name__ == "__main__":
    main()
