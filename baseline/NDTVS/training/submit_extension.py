"""Extend an existing NDTVS budget and submit training using config.py."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.NDTVS.training.submit_ndtvs import load_settings


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def build_job(settings):
    root = Path(settings.PROJECT_ROOT).resolve()
    source = Path(settings.EXTEND_SOURCE_RUN).resolve()
    out = Path(settings.EXTEND_OUT).resolve()
    python = Path(settings.EXTENSION_PYTHON).resolve()
    logs = Path(settings.LOG_DIR).resolve()
    helper = root / "baseline/NDTVS/training/extend_training.py"
    batch = root / "baseline/NDTVS/run_common.sbatch"
    target = settings.EXTEND_EPISODES
    if isinstance(target, bool) or not isinstance(target, int) or not 0 < target < 1_000_000:
        raise ValueError("EXTEND_EPISODES must be a total budget in [1, 999999]")
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("EXTEND_OUT must be separate from, and not nested in, EXTEND_SOURCE_RUN")
    for path in (python, helper, batch, source / "latest.pt", source / "experiment.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")
    spec = read_json(source / "experiment.json")
    if spec["algorithm"] != "ndtvs":
        raise ValueError("This submitter extends only the NDTVS run")
    previous = spec["episodes"]
    if target <= previous:
        raise ValueError(f"EXTEND_EPISODES must exceed the original target {previous}")
    status = read_json(source / "status.json")
    if status.get("status") != "complete" or status.get("completed_episodes") != previous:
        raise ValueError("Finish the original run before extending its budget")
    cfg = spec["config"]
    if cfg["episode_offset"] != 0:
        raise ValueError("The original NDTVS training config must have episode_offset=0")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("The checkpoint device must be cpu or cuda")
    for name, minimum in (("CPUS", 1), ("GPUS", 0), ("TRACE_EVERY", 0), ("MAX_NEW_EPISODES", 0)):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if cfg["device"] == "cuda" and settings.GPUS < 1:
        raise ValueError("The original CUDA run requires a requested GPU")
    if not 0 <= settings.RESERVE_SECONDS < settings.WALLTIME_SECONDS:
        raise ValueError("Use WALLTIME_SECONDS > RESERVE_SECONDS >= 0")

    needs_extension = not out.exists()
    if not needs_extension:
        for name in ("latest.pt", "common_config.json", "experiment.json", "extension.json", "status.json"):
            if not (out / name).is_file():
                raise ValueError(f"Existing extension is incomplete: missing {out / name}")
        origin = read_json(out / "extension.json")
        if (origin.get("source") != str(source) or origin.get("previous_target") != previous
                or origin.get("new_target") != target or origin.get("next_episode") != previous):
            raise ValueError("Existing output belongs to a different budget extension")
        extended = read_json(out / "experiment.json")
        # The existing trainer performs strict model/reward/source checks too.
        for key in ("algorithm", "config", "val_every", "val_episodes", "version",
                    "qoe_weights", "qoe_definition", "learning_settings", "model_parameters"):
            if extended.get(key) != spec.get(key):
                raise ValueError(f"Extended run changed the inherited setting: {key}")
        if extended["episodes"] != target or read_json(out / "common_config.json")["config"] != cfg:
            raise ValueError("Extended run has a different budget or environment config")
        completed = read_json(out / "status.json").get("completed_episodes", 0)
        if completed >= target:
            raise ValueError(f"This extension already completed {completed}/{target} episodes")

    extension_command = [str(python), str(helper), "--source", str(source),
                         "--out", str(out), "--episodes", str(target)]
    command = ["sbatch", f"--partition={settings.PARTITION}",
               f"--cpus-per-task={settings.CPUS}", f"--mem={settings.MEMORY}",
               f"--time={settings.TIME_LIMIT}", f"--job-name={settings.EXTEND_JOB_NAME}",
               f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
    if settings.GPUS:
        command.append(f"--gres=gpu:{settings.GPUS}")
    command.extend([str(batch), "train", "--algorithm", "ndtvs",
                    "--config", str(out / "common_config.json"), "--out", str(out),
                    "--episodes", str(target), "--device", cfg["device"], "--seed", str(cfg["seed"]),
                    "--val-every", str(spec["val_every"]), "--val-episodes", str(spec["val_episodes"]),
                    "--trace-every", str(settings.TRACE_EVERY),
                    "--walltime-seconds", str(settings.WALLTIME_SECONDS),
                    "--reserve-seconds", str(settings.RESERVE_SECONDS),
                    "--max-new-episodes", str(settings.MAX_NEW_EPISODES), "--resume"])
    return {"root": root, "source": source, "out": out, "logs": logs, "target": target,
            "previous": previous, "spec": spec, "needs_extension": needs_extension,
            "extension_command": extension_command, "command": command}


def prepare(job, settings):
    if job["needs_extension"]:
        # Uses the existing checked checkpoint tool in lab, even from base Python.
        # It preserves model/optimizer/buffers/RNG and the original 500-episode run.
        subprocess.run(job["extension_command"], cwd=job["root"], check=True)
    prepared = build_job(settings)  # Verify generated provenance before submission.
    if prepared["needs_extension"]:
        raise RuntimeError("Checkpoint extension did not create its output")
    prepared["logs"].mkdir(parents=True, exist_ok=True)
    return prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path,
                        default=PROJECT_ROOT / "baseline/NDTVS/config.py")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print commands without writes or submission")
    mode.add_argument("--prepare-only", action="store_true", help="extend the checkpoint without submitting")
    args = parser.parse_args(argv)
    settings = load_settings(args.settings)
    job = build_job(settings)
    spec = job["spec"]
    print(f"Source NDTVS run: {job['source']}")
    print(f"Extension output: {job['out']}")
    print(f"Total target: {job['previous']} -> {job['target']} episodes")
    print(f"Inherited seed/device: {spec['config']['seed']} / {spec['config']['device']}")
    print(f"Inherited validation: every {spec['val_every']}, {spec['val_episodes']} fixed scenarios")
    if job["needs_extension"]:
        print(shlex.join(job["extension_command"]))
    print(shlex.join(job["command"]), flush=True)
    if args.dry_run:
        return 0
    job = prepare(job, settings)
    if args.prepare_only:
        return 0
    return subprocess.run(job["command"], cwd=job["root"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
