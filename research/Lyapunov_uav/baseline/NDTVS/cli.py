"""Ladipo-adapted RSU PPO and a proposed-without-UAV ablation.

Use the sibling proposed/ physical implementation without editing it.
No GRU, instantaneous CSI, cloud, cache, or bandwidth/power optimization.
This is an adaptation, not a reproduction of the original NDT system."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from baseline.NDTVS.training.train import run_train
from baseline.NDTVS.evaluation.single import run_eval


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "eval"))
    p.add_argument("--algorithm", choices=("ndtvs", "hppo_rsu", "proposed"), required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=10)
    p.add_argument("--trace-every", type=int, default=25)
    p.add_argument("--walltime-seconds", type=float, default=82800)
    p.add_argument("--reserve-seconds", type=float, default=900)
    p.add_argument("--max-new-episodes", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--frame-checkpoint", type=Path)
    p.add_argument("--slot-checkpoint", type=Path)
    p.add_argument("--scenario-seed", type=int, default=2026)
    p.add_argument("--offset", type=int, default=2_000_000)
    p.add_argument("--trace", action="store_true")
    a = p.parse_args(argv)
    if min(a.episodes, a.val_every, a.val_episodes) <= 0 or a.reserve_seconds < 0 or a.walltime_seconds <= 0:
        p.error("positive episode/validation/budget values required")
    if a.mode == "train" and (a.algorithm == "proposed" or not a.config):
        p.error("train requires --config and algorithm ndtvs or hppo_rsu")
    if a.mode == "eval" and ((a.algorithm == "proposed" and not (a.config and a.frame_checkpoint and a.slot_checkpoint))
                             or (a.algorithm != "proposed" and not a.checkpoint)):
        p.error("eval requires the algorithm's checkpoint(s) and proposed configuration")
    a.out = a.out.resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_name(a.out.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_train(a) if a.mode == "train" else run_eval(a)
