"""Learning and reward-component plots; no convergence claim from smoothing."""
import csv
import os
from pathlib import Path
import tempfile
import numpy as np


def plot(s):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="uavrtv_mpl_"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(s.OUT)
    with (root / "training.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No completed training episodes")
    episodes = np.asarray([int(r["episode"]) + 1 for r in rows])
    def series(key):
        return np.asarray([float(r[key]) for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rewards = series("uavrtv_reward_per_region_slot")
    axes[0, 0].plot(episodes, rewards, alpha=.3, label="Training")
    window = min(20, len(rows))
    axes[0, 0].plot(episodes[window-1:], np.convolve(rewards, np.ones(window)/window, mode="valid"), label=f"Mean {window}")
    if (root / "validation.csv").exists():
        with (root / "validation.csv").open() as f:
            validation = list(csv.DictReader(f))
        axes[0, 0].plot([int(r["trained_episodes"]) for r in validation], [float(r["selection_score"]) for r in validation], "o-", label="Fixed validation")
    axes[0, 0].set_ylabel("UAVRTV reward / region-slot")
    for key in ("stall_time_ratio", "average_quality_utility", "unique_served_user_ratio"):
        values = series(key)
        if key == "average_quality_utility":
            values = np.where([r["quality_utility_defined"] == "True" for r in rows], values, np.nan)
        axes[0, 1].plot(episodes, values, label=key)
    axes[0, 1].set_ylabel("Common metrics")
    axes[1, 0].plot(episodes, series("hire_rate"), label="Hire rate")
    axes[1, 0].set_ylabel("Hire rate")
    right = axes[1, 0].twinx()
    right.plot(episodes, series("energy_consumed_j") / 1000, color="tab:orange", label="Energy")
    right.set_ylabel("UAV energy [kJ]")
    import json
    cfg = json.loads((root / "resolved_config.json").read_text())["config"]
    slots = cfg["num_frames"] * cfg["frame_slots"] * cfg["num_regions"]
    for key in ("quality_gain", "switch_penalty", "rebuffer_penalty", "energy_penalty", "hiring_penalty"):
        axes[1, 1].plot(episodes, series(key + "_total") / slots, label=key)
    axes[1, 1].set_ylabel("Reward terms / region-slot")
    for ax in axes.ravel():
        ax.set_xlabel("Completed training episodes")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    destination = root / "plots"
    destination.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / ("learning." + suffix), dpi=180)
    plt.close(fig)
    return 0
