"""Training curves and descriptive convergence diagnostics."""
from __future__ import annotations

import json
import numpy as np
from baseline.NDTVS.common.io import atomic


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
    for ax, key, title in zip(axes[:2], ("paper_qoe_per_user_slot", "stall_ratio"), ("Training fixed-weight paper QoE", "Training stall ratio")):
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
