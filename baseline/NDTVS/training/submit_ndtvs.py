"""Prepare a zero-offset NDTVS config and submit the existing training job."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))


def load_settings(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_ndtvs_submission_settings", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load settings: {path}")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings


def build_job(settings):
    root = Path(settings.PROJECT_ROOT).resolve()
    source = Path(settings.SOURCE_CONFIG).resolve()
    target = Path(settings.TRAIN_CONFIG).resolve()
    out = Path(settings.OUT).resolve()
    logs = Path(settings.LOG_DIR).resolve()
    script = root / "baseline/NDTVS/run_common.sbatch"
    if source == target:
        raise ValueError("TRAIN_CONFIG must differ from SOURCE_CONFIG; keep the Proposed JSON intact")
    if not script.is_file():
        raise FileNotFoundError(f"Missing existing batch entry point: {script}")
    if settings.DEVICE not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu or cuda")
    if not isinstance(settings.RESUME, bool):
        raise ValueError("RESUME must be True or False")
    for name in ("EPISODES", "VAL_EVERY", "VAL_EPISODES", "CPUS"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings.EPISODES >= 1_000_000:
        raise ValueError("EPISODES must stay below the validation episode namespace")
    for name in ("GPUS", "TRACE_EVERY", "MAX_NEW_EPISODES"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if settings.DEVICE == "cuda" and settings.GPUS < 1:
        raise ValueError("CUDA training requires at least one requested GPU")
    if not 0 <= settings.RESERVE_SECONDS < settings.WALLTIME_SECONDS:
        raise ValueError("Use WALLTIME_SECONDS > RESERVE_SECONDS >= 0")
    if settings.RESUME:
        if not (out / "latest.pt").is_file():
            raise FileNotFoundError(f"RESUME=True requires this NDTVS run's latest.pt: {out}")
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty NDTVS output: {out}; use its original settings and RESUME=True")

    original = source.read_bytes()
    raw = json.loads(original)
    if not isinstance(raw, dict):
        raise ValueError("SOURCE_CONFIG must contain a config object")
    source_config = raw.get("config", raw)
    if not isinstance(source_config, dict):
        raise ValueError("SOURCE_CONFIG.config must be an object")
    # The submitter needs only the standard library on the login node.
    # The existing trainer still performs the full HPPOConfig validation in lab.
    required = {"num_regions", "users_per_region", "rsu_total_bandwidth_hz",
                "mask_queue_actions", "delivery_mode", "lyapunov_v"}
    if not required <= source_config.keys():
        raise ValueError("Use the complete saved Proposed resolved_config.json")
    if not (source_config["mask_queue_actions"]
            and source_config.get("enforce_queue_admissibility", True)
            and source_config["delivery_mode"] == "all_or_nothing"):
        raise ValueError("Queue-masked atomic delivery is required")
    source_v = source_config["lyapunov_v"]
    if source_v != settings.EXPECTED_LYAPUNOV_V:
        raise ValueError(f"Expected an explicit V={settings.EXPECTED_LYAPUNOV_V}; source has V={source_v}")

    train_config = copy.deepcopy(source_config)
    train_config["episode_offset"] = 0
    # Everything other than the run's episode numbering stays byte-value equivalent.
    assert {k: v for k, v in train_config.items() if k != "episode_offset"} == {
        k: v for k, v in source_config.items() if k != "episode_offset"}
    payload = copy.deepcopy(raw) if "config" in raw else {"config": copy.deepcopy(raw)}
    payload["config"] = train_config
    payload["ndtvs_preparation"] = {
        "source_config": str(source),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "source_episode_offset": source_config.get("episode_offset", 0),
        "training_episode_offset": 0,
        "changed_config_fields": ["episode_offset"] if source_config.get("episode_offset", 0) != 0 else [],
        "reference_proposed_episodes_user_reported": settings.REFERENCE_PROPOSED_EPISODES,
        "note": "Proposed source is preserved; its remaining session budget is not the NDTVS target.",
    }
    if target.exists() and json.loads(target.read_text()) != payload:
        raise ValueError(f"Existing TRAIN_CONFIG differs: {target}; choose a separate config path for a new experiment")

    command = ["sbatch", f"--partition={settings.PARTITION}",
               f"--cpus-per-task={settings.CPUS}", f"--mem={settings.MEMORY}",
               f"--time={settings.TIME_LIMIT}", f"--job-name={settings.JOB_NAME}",
               f"--output={logs}/%x-%j.out", f"--error={logs}/%x-%j.err"]
    if settings.GPUS:
        command.append(f"--gres=gpu:{settings.GPUS}")
    command.extend([str(script), "train", "--algorithm", "ndtvs",
                    "--config", str(target), "--out", str(out),
                    "--device", settings.DEVICE, "--seed", str(settings.SEED),
                    "--episodes", str(settings.EPISODES),
                    "--val-every", str(settings.VAL_EVERY),
                    "--val-episodes", str(settings.VAL_EPISODES),
                    "--trace-every", str(settings.TRACE_EVERY),
                    "--walltime-seconds", str(settings.WALLTIME_SECONDS),
                    "--reserve-seconds", str(settings.RESERVE_SECONDS),
                    "--max-new-episodes", str(settings.MAX_NEW_EPISODES)])
    if settings.RESUME:
        command.append("--resume")
    return {"root": root, "source": source, "original": original,
            "target": target, "logs": logs, "payload": payload, "command": command}


def prepare(job):
    # Detect a concurrent edit before deriving another file from the source.
    if job["source"].read_bytes() != job["original"]:
        raise ValueError("Proposed source changed during preparation; retry with the intended source")
    target = job["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(job["payload"], indent=2, ensure_ascii=False) + "\n").encode()
    try:
        with target.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if json.loads(target.read_text()) != job["payload"]:
            raise ValueError("TRAIN_CONFIG was changed during preparation")
    job["logs"].mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path,
                        default=PROJECT_ROOT / "baseline/NDTVS/config.py")
    parser.add_argument("--dry-run", action="store_true", help="print the job without creating files or submitting")
    parser.add_argument("--prepare-only", action="store_true", help="create the derived JSON without submitting")
    args = parser.parse_args(argv)
    if args.dry_run and args.prepare_only:
        parser.error("Choose --dry-run or --prepare-only")
    job = build_job(load_settings(args.settings))
    print(f"Source: {job['source']}")
    print(f"NDTVS config: {job['target']}")
    print(f"episode_offset: {job['payload']['ndtvs_preparation']['source_episode_offset']} -> 0")
    print(f"V: {job['payload']['config']['lyapunov_v']} (preserved)")
    print(shlex.join(job["command"]), flush=True)
    if args.dry_run:
        return 0
    prepare(job)
    if args.prepare_only:
        return 0
    return subprocess.run(job["command"], cwd=job["root"], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
