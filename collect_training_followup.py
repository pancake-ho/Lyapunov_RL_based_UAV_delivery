#!/usr/bin/env python3
"""Collect original/resumed run metadata without loading or changing checkpoints.

Python standard library only. --root is research/Lyapunov_uav.
--extra may name a Proposed candidate-validation JSON/CSV or its directory.
"""
import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-run", default="hrl_revision_train_seed2026_job142434")
    p.add_argument("--job-id", default="142434")
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    run_root = root / "proposed/outputs/hppo"
    source = (run_root / a.source_run).resolve()
    if not source.is_dir():
        p.error(f"Source run not found: {source}")
    if a.out.exists():
        p.error(f"Output already exists (choose a new name): {a.out}")
    files, errors, runtimes = {}, [], {}

    def add(path, archive_name): 
        if path.is_file():
            files[archive_name] = path.read_bytes()

    for path in sorted(run_root.glob("*/runtime.json")):
        try:
            runtimes[path.parent.resolve()] = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    selected = {source}
    while True:
        found = {run for run, meta in runtimes.items()
                 if meta.get("resume_source") and Path(meta["resume_source"]).resolve() in selected}
        if found <= selected:
            break
        selected |= found
    inventory = []
    names = ("runtime.json", "resolved_config.json", "training_status.json", "status.json",
             "episode_summary.csv", "frame_updates.csv", "slot_updates.csv", "validation.json")
    for run in sorted(selected):
        for name in names:
            add(run / name, f"proposed/{run.name}/{name}")
        summary = run / "episode_summary.csv"
        rows = list(csv.DictReader(io.StringIO(summary.read_text()))) if summary.exists() else []
        episodes = sorted(int(float(row["episode"])) for row in rows)
        inventory.append({"run": str(run), "resume_source": runtimes.get(run, {}).get("resume_source"),
                          "rows_in_segment": len(rows), "first_episode_number": episodes[0]+1 if episodes else None,
                          "last_completed_episode_number": episodes[-1]+1 if episodes else None,
                          "checkpoint_filenames_only": sorted(path.name for path in (run/"checkpoints").glob("*.pt"))})
    for name in ("ndtvs_seed2026_ep500", "hppo_rsu_seed2026_ep850"):
        add(root/"baseline/NDTVS/runs/common_gpu"/name/"experiment.json", f"baseline/{name}/experiment.json")
    for index, extra in enumerate(a.extra):
        extra = extra.resolve()
        if not extra.exists():
            p.error(f"Extra path not found: {extra}")
        paths = sorted(extra.rglob("*")) if extra.is_dir() else [extra]
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
                relative = path.relative_to(extra) if extra.is_dir() else Path(path.name)
                add(path, f"extra_{index}/{relative.as_posix()}")
    command = ["sacct", "-j", a.job_id, "--noheader", "--parsable2",
               "--format=JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
        accounting = {"command": command, "returncode": proc.returncode,
                      "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        accounting = {"command": command, "unavailable": str(exc)}
    for path in sorted((root/"proposed/slurm_logs").glob(f"*{a.job_id}*")):
        if path.is_file():
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size-100000))
                files[f"slurm_tails/{path.name}.last100000bytes.txt"] = handle.read()
    info = {"root": str(root), "runs": inventory, "scheduler": accounting, "errors": errors,
            "scope": "Original run and resume_source descendants directly under proposed/outputs/hppo; optional explicit validation paths.",
            "checkpoints_loaded": False, "sha256": {k: hashlib.sha256(v).hexdigest() for k,v in files.items()}}
    with ZipFile(a.out, "x", compression=ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("followup_inventory.json", json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(a.out.resolve()), "runs": inventory,
                      "metadata_files": len(files), "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
