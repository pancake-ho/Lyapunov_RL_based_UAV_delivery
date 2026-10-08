"""Episode-boundary recovery for the e1d79cf scheduling HPPO implementation.

Uses the original run_episode unchanged. Exit 75 means safely PAUSED, not COMPLETE.
SIGUSR1/SIGTERM/SIGINT request a stop after the current episode. SIGKILL cannot
be caught; the atomic checkpoint from the previous episode remains usable.
Only load trusted, locally generated checkpoints (torch pickle format).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from config_hppo import HPPOConfig
from hppo.env import P3HierarchicalEnv
from hppo.logger import HistoryLogger
from hppo.ppo import PPOAgent, ARCHITECTURE_VERSION
from hppo.train import run_episode, save_resolved_config, seed_all
from hppo.verify_trace import load_config

FORMAT = "hppo-episode-resume-v1"


def atomic_save(path, payload, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb" if binary else "w") as f:
            if binary:
                torch.save(payload, f)
            else:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in ("config*.py", "hppo/*.py", "env/p3/*.py", "agent/P3/*.py")
            for p in sorted(root.glob(pattern))}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from resume checkpoint")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def agent_state(agent):
    return {"architecture": ARCHITECTURE_VERSION, "name": agent.name,
            "model": agent.net.state_dict(), "optimizer": agent.opt.state_dict(),
            "update_count": agent.update_count, "trajectories": agent.trajectories}


def restore_agent(agent, saved):
    if saved["architecture"] != ARCHITECTURE_VERSION or saved["name"] != agent.name:
        raise RuntimeError("Resume architecture/policy role mismatch")
    if not all(torch.isfinite(v).all() for v in saved["model"].values()):
        raise RuntimeError("Nonfinite checkpoint weights")
    agent.net.load_state_dict(saved["model"], strict=True)
    agent.opt.load_state_dict(saved["optimizer"])
    agent.update_count = int(saved["update_count"])
    agent.trajectories = saved["trajectories"]


def load_source(source, fresh=False):
    source = Path(source).resolve()
    cfg = load_config(source)
    saved_args = json.loads((source / "resolved_config.json").read_text())["args"]
    ck = source / saved_args.get("checkpoint_dir", "checkpoints")
    if fresh:
        return cfg, {"kind": "fresh", "origin": cfg.episode_offset,
                     "next_episode": cfg.episode_offset, "target": cfg.train_episodes}
    bundle_path = ck / "resume_latest.pt"
    if bundle_path.exists():
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if b.get("format") != FORMAT:
            raise RuntimeError("Unsupported resume format")
        cfg = HPPOConfig(**b["config"])
        current = source_hashes()
        changed = [k for k, v in b["code_sha256"].items() if current.get(k) != v]
        if changed:
            raise RuntimeError("Resume source code changed: " + ", ".join(changed))
        return cfg, {"kind": "exact", "origin": b["origin"],
                     "next_episode": b["next_episode"], "target": b["target"],
                     "bundle": b, "path": str(bundle_path)}
    # Legacy pairs are individually atomic, but latest files are not atomic as a pair.
    # Inspect numbered pairs newest-first and reject incomplete or inconsistent pairs.
    errors = []
    for frame in sorted(ck.glob("frame_ep*.pt"), reverse=True):
        slot = frame.with_name(frame.name.replace("frame_", "slot_", 1))
        try:
            f = torch.load(frame, map_location="cpu", weights_only=False)
            s = torch.load(slot, map_location="cpu", weights_only=False)
            ex = f["extra"]
            count = int(frame.stem.removeprefix("frame_ep"))
            if not ex.get("pair_id") or ex != s["extra"]:
                raise ValueError("pair metadata mismatch")
            if ex["episode"] != cfg.episode_offset + count - 1:
                raise ValueError("episode/filename mismatch")
            if count % cfg.frame_update_every_episodes and count != cfg.train_episodes:
                raise ValueError("legacy checkpoint may omit pending slow PPO buffer")
            for state, name in ((f, "frame_ppo"), (s, "slot_ppo")):
                if state["name"] != name or not state["optimizer"]:
                    raise ValueError("missing policy/optimizer")
                if not all(torch.isfinite(x).all() for x in state["model"].values()):
                    raise ValueError("nonfinite checkpoint")
            return cfg, {"kind": "legacy", "origin": cfg.episode_offset,
                         "next_episode": ex["episode"] + 1, "target": cfg.train_episodes,
                         "frame_path": str(frame), "slot_path": str(slot),
                         "extra": ex, "skipped_pairs": errors}
        except Exception as exc:
            errors.append(f"{frame.name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("No valid checkpoint pair. Use server originals, not truncated uploads. "
                       + " | ".join(errors[:5]))


class StopRequest:
    def __init__(self):
        self.reason = ""

    def handle(self, signum, _frame):
        # No IO, exceptions, or checkpoint writes inside the signal handler.
        self.reason = signal.Signals(signum).name


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/hppo"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-episodes", type=int, default=None)
    p.add_argument("--max-seconds", type=float, default=82800)
    p.add_argument("--reserve-seconds", type=float, default=900)
    p.add_argument("--stop-after-episodes", type=int, default=0)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    args = p.parse_args(argv)
    if args.max_seconds <= 0 or args.reserve_seconds < 0 or args.stop_after_episodes < 0:
        p.error("invalid stop budget")
    stop = StopRequest()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop.handle)
    started = time.monotonic()
    cfg, source = load_source(args.source_run, args.fresh)
    target = source["target"] if args.total_episodes is None else args.total_episodes
    if target != source["target"]:
        raise ValueError("Keep the original total episode target; changing it changes final PPO flush timing")
    origin, next_ep = source["origin"], source["next_episode"]
    end = origin + target
    if next_ep > end:
        raise ValueError("checkpoint exceeds target")
    cfg = replace(cfg, device=args.device, episode_offset=next_ep,
                  train_episodes=max(1, end - next_ep))
    seed_all(cfg.seed)
    torch.set_num_threads(cfg.torch_num_threads)
    frame = PPOAgent(cfg.frame_obs_dim, cfg.frame_action_nvec, cfg, "frame_ppo")
    slot = PPOAgent(cfg.slot_obs_dim, cfg.slot_action_nvec, cfg, "slot_ppo")
    dual = cfg.dual_init
    if source["kind"] == "exact":
        b = source["bundle"]
        restore_agent(frame, b["frame"])
        restore_agent(slot, b["slot"])
        dual = b["dual"]
        restore_rng(b["rng"])
    elif source["kind"] == "legacy":
        frame.load(source["frame_path"], load_optimizer=True)
        slot.load(source["slot_path"], load_optimizer=True)
        dual = source["extra"]["dual_lambda_z"]
        seed_all(cfg.seed)  # Legacy files do not contain RNG states.
        print("[LEGACY] Model/optimizer/episode restored; original RNG is unavailable.", flush=True)
    info = {k: v for k, v in source.items() if k != "bundle"}
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.inspect_only:
        return 0
    if next_ep == end:
        print(f"[ALREADY COMPLETE] {target} episodes; evaluate {args.source_run}", flush=True)
        return 0
    root = args.output_dir / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    args.mode, args.checkpoint_dir = "train", "checkpoints"
    recorded_args = argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v
                                          for k, v in vars(args).items()})
    save_resolved_config(cfg, recorded_args, root)
    hashes = source_hashes()
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                      cwd=Path(__file__).resolve().parents[1], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    atomic_save(root / "runtime.json", {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": cfg.device,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "numpy": np.__version__, "base_commit": "e1d79cff46742625fd75fd3fc23917fa93428326",
        "git_head": git_head, "code_sha256": hashes, "argv": sys.argv[1:],
        "resume_source": str(args.source_run.resolve()), "resume_info": info})
    logger = HistoryLogger(cfg, root)
    ck = root / "checkpoints"
    session_start, completed_here, longest = next_ep, 0, 0.0
    committed_next = next_ep

    def save():
        nonlocal committed_next
        # One atomic file commits both models, both optimizers, pending buffers and RNG.
        atomic_save(ck / "resume_latest.pt", {"format": FORMAT, "config": asdict(cfg),
            "origin": origin, "target": target, "next_episode": next_ep, "dual": dual,
            "frame": agent_state(frame), "slot": agent_state(slot), "rng": rng_state(),
            "code_sha256": hashes, "source_run": str(root.resolve())}, binary=True)
        committed_next = next_ep

    def status(state, reason=""):
        atomic_save(root / "training_status.json", {"status": state, "reason": reason,
            "completed_total": committed_next - origin, "target_total": target,
            "next_episode": committed_next, "completed_this_session": committed_next - session_start,
            "session_first_episode": session_start,
            "resume_checkpoint": str((ck / "resume_latest.pt").resolve())})

    logger.event("run_start", {"schema": "scheduling-hppo-v2", "mode": "train",
        "expected_episodes": end - next_ep, "episode_offset": next_ep})
    try:
        save()  # Guarantees a recovery point even if the first episode is killed.
        status("RUNNING")
        env = P3HierarchicalEnv(cfg)
        while next_ep < end:
            guard = max(args.reserve_seconds, 1.5 * longest)
            if stop.reason or time.monotonic() - started + guard >= args.max_seconds:
                stop.reason = stop.reason or "walltime_budget"
                break
            ep = next_ep
            tick = time.monotonic()
            s = run_episode(env, frame, slot, cfg, logger, ep, dual, True, False)
            if cfg.reward_mode == "objective_lagrangian":
                dual = float(np.clip(dual + cfg.dual_lr * (s["z_constraint_mean"] - cfg.z_target_normalized),
                                     0.0, cfg.dual_max))
                s["dual_lambda_z"] = dual
            logger.log_episode(s)
            if ((ep - origin + 1) % cfg.frame_update_every_episodes == 0 or ep == end - 1) and frame.buffer_size():
                metrics = frame.update()
                logger.event("frame_ppo_update", {"episode": ep, **metrics})
                logger.debug(f"[FRAME PPO UPDATE] ep={ep} " + json.dumps(metrics))
            next_ep = ep + 1
            completed_here += 1
            save()
            status("RUNNING")
            longest = max(longest, time.monotonic() - tick)
            print(f"[CHECKPOINT] completed={next_ep-origin}/{target} "
                  f"next_episode={next_ep} episode_seconds={time.monotonic()-tick:.1f}", flush=True)
            if args.stop_after_episodes and completed_here >= args.stop_after_episodes:
                stop.reason = "requested_episode_limit"
                break
        done = next_ep == end
        # Compatible pair for the existing evaluation command; bundle is authoritative.
        extra = {"episode": next_ep - 1, "dual_lambda_z": dual,
                 "pair_id": f"{time.time_ns()}-ep{next_ep-1}-f{frame.update_count}-s{slot.update_count}"}
        frame.save(ck / "frame_latest.pt", extra)
        slot.save(ck / "slot_latest.pt", extra)
        if done:
            if frame.buffer_size() or slot.buffer_size():
                raise RuntimeError("final PPO buffer is not empty")
            logger.event("run_end", {"status": "complete"})
            status("COMPLETE")
        else:
            logger.event("run_pause", {"status": "paused", "reason": stop.reason,
                "completed_episodes": completed_here, "next_episode": next_ep})
            status("PAUSED", stop.reason)
        print(f"[{'COMPLETE' if done else 'PAUSED'}] {root.resolve()}", flush=True)
        return 0 if done else 75
    except Exception as exc:
        # Never serialize partially applied PPO updates as a resumable state.
        status("ERROR", f"{type(exc).__name__}: {exc}; resume from last committed checkpoint")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
