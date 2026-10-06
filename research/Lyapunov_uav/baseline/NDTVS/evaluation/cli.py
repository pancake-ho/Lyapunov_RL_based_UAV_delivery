"""Paired v2 QoE evaluation of freshly trained NDTVS and existing policies.

Nominal eval, smoke and SNR sweep share one observer. Exit 75 is resumable."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
from baseline.NDTVS.evaluation.checks import (ALGORITHMS, OFFSETS, METRICS, require, digest,
    sha256, read_json, physical, algorithm, validate_row, check_rows, verify_saved)
from baseline.NDTVS.evaluation.scenario import radio_sanity, PairingObserver
from baseline.NDTVS.evaluation.policies import policy_digest, verify_best, load_policies
from baseline.NDTVS.evaluation.sweep import run
from baseline.NDTVS.common.paths import HERE
ROOT = HERE.parents[1]
RUNS = ROOT / "baseline/NDTVS/runs/common_gpu"
TRAIN = ROOT / "proposed/outputs/hppo/hrl_revision_train_seed2026_job142434"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "eval", "sweep"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--proposed-runtime", type=Path)
    p.add_argument("--ndtvs-checkpoint", type=Path, required=True)
    p.add_argument("--rsu-checkpoint", type=Path, required=True)
    p.add_argument("--frame-checkpoint", type=Path, required=True)
    p.add_argument("--slot-checkpoint", type=Path, required=True)
    p.add_argument("--frame-checkpoint-2", type=Path)
    p.add_argument("--slot-checkpoint-2", type=Path)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--offset", type=int, default=7_000_000)
    p.add_argument("--scenario-seed", type=int, default=2026)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--walltime-seconds", type=float, default=82800)
    p.add_argument("--reserve-seconds", type=float, default=1800)
    args = p.parse_args(argv)
    if args.episodes <= 0 or args.offset < 2_000_000 or args.walltime_seconds <= 0 or args.reserve_seconds < 0:
        p.error("Positive episodes/walltime, nonnegative reserve, final-test offset >= 2000000 required")
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_name(args.out.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
