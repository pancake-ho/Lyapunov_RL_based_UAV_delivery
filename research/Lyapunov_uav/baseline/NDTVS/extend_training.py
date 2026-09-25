"""Explicitly fork a checkpoint to a larger budget; keep the original run intact."""
import argparse
import fcntl
import hashlib
import json
import shlex
import shutil
import sys
from pathlib import Path
from ndtvs_common import atomic, load_bundle, source_hashes


def extend(source, destination, episodes):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Use a separate, non-nested destination")
    if not 0 < episodes < 1_000_000:
        raise ValueError("Budget must stay below the validation episode namespace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.with_name(source.name + ".lock").open("a") as src_lock, \
            destination.with_name(destination.name + ".lock").open("a") as dst_lock:
        fcntl.flock(src_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(dst_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if destination.exists():
            raise FileExistsError("Destination must not exist")
        checkpoint = source / "latest.pt"
        saved = load_bundle(checkpoint)
        spec = saved["spec"]
        if spec["source_sha256"] != source_hashes():
            raise ValueError("Source differs from the checkpoint; extension is not a migration")
        previous = spec["episodes"]
        if episodes <= previous:
            raise ValueError("New budget must exceed the original target")
        interval = spec["config"]["frame_update_every_episodes"]
        if (spec["algorithm"] == "hppo_rsu" and saved["next_episode"] == previous
                and previous % interval != 0):
            raise ValueError("Old final frame-buffer flush occurred off the regular update boundary")
        destination.mkdir()
        for name in ("segments", "original_traces", "reproduced_traces"):
            if (source / name).exists():
                shutil.copytree(source / name, destination / name)
        for name in ("training.csv", "validation.json", "best.pt", "trace_recovery.json"):
            if (source / name).exists():
                shutil.copy2(source / name, destination / name)
        origin = {"source": str(source), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "previous_target": previous, "new_target": episodes,
                  "next_episode": saved["next_episode"],
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "operation": "budget-only fork; model, optimizer, buffers and RNG preserved"}
        spec["episodes"] = episodes
        atomic(destination / "latest.pt", saved, binary=True)
        atomic(destination / "experiment.json", spec)
        atomic(destination / "common_config.json", {"config": spec["config"]})
        atomic(destination / "extension.json", origin)
        atomic(destination / "status.json", {"status": "paused", "reason": "explicit budget extension",
               "completed_episodes": saved["next_episode"], "target_episodes": episodes,
               "convergence": "not certified"})
    command = [sys.executable, str(Path(__file__).with_name("ndtvs_common.py")), "train",
               "--algorithm", spec["algorithm"], "--config", str(destination / "common_config.json"),
               "--out", str(destination), "--episodes", str(episodes),
               "--device", spec["config"]["device"], "--seed", str(spec["config"]["seed"]),
               "--val-every", str(spec["val_every"]), "--val-episodes", str(spec["val_episodes"]), "--resume"]
    print(shlex.join(command))
    return origin


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, required=True)
    a = p.parse_args()
    extend(a.source, a.out, a.episodes)
