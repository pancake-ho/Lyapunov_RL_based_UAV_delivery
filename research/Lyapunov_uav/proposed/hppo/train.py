from __future__ import annotations

"""Two-timescale control loop (Algorithm 2) with PPO at both levels.

    for frame r:
        frame PPO -> RSU/UAV candidate scheduling; completion -> mu, point
        relocate during control-preparation interval (battery, reachability)
        for slot t in frame:
            slot PPO (per region, shared weights) -> l, k, p        (masked)
            environment realizes hidden fading and applies the configured
            full-request success/failure rule (legacy partial is optional),
            updates Q, Z, battery, mobility
        frame reward, slot rewards -> PPO buffers (per-region trajectories)

Run:
    python -m hppo.train --mode random --num-frames 2 --frame-slots 3 --run-name smoke
    python -m hppo.train --mode train  --train-episodes 200 --run-name hppo_dpp
    python -m hppo.train --mode eval   --frame-checkpoint ... --slot-checkpoint ...
"""

import json
import random
import sys
import time
import platform
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config_hppo import HPPOConfig, parse_config
from hppo.env import P3HierarchicalEnv, frame_training_reward, slot_training_reward
from hppo.logger import HistoryLogger, jsonable
from hppo.scheduling import sample_scheduling
from hppo.completion import FastPolicyCompletion


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


class RandomPolicy:
    """Uniform sampling over the feasible (masked) support; no torch needed."""

    name = "random"

    def __init__(self, seed: int, cfg=None, scheduling=False) -> None:
        self.rng = np.random.default_rng(seed)
        self.cfg, self.scheduling = cfg, scheduling
        self.update_count = 0

    def act(self, obs, masks, deterministic: bool = False):
        action = (sample_scheduling(masks, self.cfg, self.rng) if self.scheduling else
                  np.asarray([int(self.rng.choice(np.flatnonzero(m))) for m in masks], dtype=np.int64))
        return action, 0.0, 0.0, 0.0

    def act_batch(self, observations, masks, deterministic=False, uniforms=None):
        batch, heads = len(observations), len(masks)
        if uniforms is None:
            uniforms = self.rng.random((batch, heads))
        result = np.zeros((batch, heads), dtype=np.int64)
        for j, mask in enumerate(masks):
            for i in range(batch):
                choices = np.flatnonzero(mask[i])
                result[i, j] = choices[min(int(uniforms[i, j] * len(choices)), len(choices) - 1)]
        return result, np.zeros(batch), np.zeros(batch), np.zeros(batch)

    def value(self, obs) -> float:
        return 0.0


def run_episode(env: P3HierarchicalEnv, frame_agent, slot_agent, cfg: HPPOConfig, logger: HistoryLogger,
                episode: int, dual: float, training: bool, deterministic: bool) -> dict:
    env.reset(episode)
    M = cfg.num_regions
    dual_for_log = dual if cfg.reward_mode == "objective_lagrangian" else None
    pending_frame: dict[int, dict] = {}
    completion = FastPolicyCompletion(cfg)
    slow_return = fast_return = completion_seconds = 0.0
    for r in range(cfg.num_frames):
        frame_obs = env.prepare_frame()
        frame_masks = {m: env.frame_action_masks(m) for m in range(M)}

        # ----- bootstrap the previous frame transition with V(s_{r}) of the new frame
        if training and pending_frame:
            for m, item in pending_frame.items():
                frame_agent.store(m, item["obs"], item["masks"], item["action"], item["logp"], item["value"],
                                  item["reward"], next_value=frame_agent.value(frame_obs[m]), done=False)
            pending_frame = {}

        raw_frame = {}
        frame_choice = {}
        for m in range(M):
            a, logp, v, _ = frame_agent.act(frame_obs[m], frame_masks[m], deterministic=deterministic)
            raw_frame[m] = a
            frame_choice[m] = (a, logp, v)
        completed, details = completion.select_all(env, raw_frame, slot_agent, deterministic)
        completion_seconds += sum(d["runtime_s"] for d in details.values())
        frame_info = env.begin_frame(raw_frame, completed, details)
        logger.log_frame_start(frame_info, obs=frame_obs, masks=frame_masks)

        slot_pending: dict[int, dict] = {}
        for t in range(cfg.frame_slots):
            slot_obs = {m: env.get_slot_obs(m) for m in range(M)}
            slot_masks = {m: env.slot_action_masks(m) for m in range(M)}
            # bootstrap previous slot transition with V(s_t) of the current slot state
            if training and slot_pending:
                for m, item in slot_pending.items():
                    slot_agent.store(m, item["obs"], item["masks"], item["action"], item["logp"], item["value"],
                                     item["reward"], next_value=slot_agent.value(slot_obs[m]), done=False)
                slot_pending = {}
            raw_slot = {}
            slot_choice = {}
            for m in range(M):
                a, logp, v, _ = slot_agent.act(slot_obs[m], slot_masks[m], deterministic=deterministic)
                raw_slot[m] = a
                slot_choice[m] = (a, logp, v)
            step = env.step_slot(raw_slot)
            env.assert_consistency()
            rewards = {}
            for m in range(M):
                base, tr = slot_training_reward(cfg, step.metrics[m], dual)
                rewards[m] = {"base": base, "training": tr}
                fast_return += tr
                a, logp, v = slot_choice[m]
                if training:
                    if step.frame_done:
                        # The slot trajectory ends at the frame boundary because the next
                        # frame action changes the fast feasible set (uav_hierarchical_ppo).
                        slot_agent.store(m, slot_obs[m], slot_masks[m], a, logp, v, tr, next_value=0.0, done=True)
                    else:
                        slot_pending[m] = {"obs": slot_obs[m], "masks": slot_masks[m], "action": a,
                                           "logp": logp, "value": v, "reward": tr}
            logger.log_slot(step.info, rewards=rewards, obs=slot_obs, masks=slot_masks, dual=dual_for_log)
            if cfg.console_log_every_slots > 0 and env.global_slot % cfg.console_log_every_slots == 0:
                tot = sum(step.metrics[m].dpp_slot_cost for m in range(M))
                print(f"ep={episode:04d} frame={r:03d} slot={t:02d} J_F={tot:9.2f} "
                      f"meanZ={np.mean(cfg.large_queue_level - env.state.queue):.2f} "
                      f"SoC={np.mean(env.state.battery_j) / cfg.battery_capacity_j:.4f}", flush=True)

        summary = step.info["frame_summary"]
        frame_rewards = {}
        for m in range(M):
            base, tr = frame_training_reward(cfg, summary["regions"][m], dual)
            frame_rewards[m] = {"base": base, "training": tr}
            slow_return += tr
            a, logp, v = frame_choice[m]
            if training:
                if r == cfg.num_frames - 1:
                    frame_agent.store(m, frame_obs[m], frame_masks[m], a, logp, v, tr, next_value=0.0, done=True)
                else:
                    pending_frame[m] = {"obs": frame_obs[m], "masks": frame_masks[m], "action": a,
                                        "logp": logp, "value": v, "reward": tr}
        logger.log_frame_end(summary, rewards=frame_rewards, dual=dual_for_log)

        if training and (r + 1) % cfg.slot_update_every_frames == 0 and slot_agent.buffer_size() > 0:
            metrics = slot_agent.update()
            if metrics:
                logger.event("slot_ppo_update", {"episode": episode, "frame": r, **metrics})
                logger.debug(f"[SLOT PPO UPDATE] ep={episode} frame={r} " + json.dumps(jsonable(metrics)))

    if training and slot_agent.buffer_size() > 0:
        metrics = slot_agent.update()
        if metrics:
            logger.event("slot_ppo_update", {"episode": episode, "frame": cfg.num_frames - 1, **metrics})
            logger.debug(f"[SLOT PPO UPDATE] ep={episode} frame={cfg.num_frames - 1} " + json.dumps(jsonable(metrics)))

    result = env.episode_summary()
    result["dual_lambda_z"] = dual
    result["slow_reward_sum"] = slow_return
    result["fast_reward_sum"] = fast_return
    result["completion_runtime_s"] = completion_seconds
    return result


def save_resolved_config(cfg: HPPOConfig, args, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"args": {k: v for k, v in vars(args).items() if v is not None}, "config": jsonable(asdict(cfg))}
    payload["config"]["distance_bin_edges_m"] = [x if np.isfinite(x) else None for x in cfg.distance_bin_edges_m]
    with open(root / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(argv=None) -> None:
    args, cfg = parse_config(argv)
    seed_all(cfg.seed)
    root = Path(args.output_dir) / args.run_name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"run directory is nonempty; use a new --run-name: {root}")
    env = P3HierarchicalEnv(cfg)
    if args.mode == "eval" and (not args.frame_checkpoint or not args.slot_checkpoint):
        raise ValueError("eval requires both checkpoints")
    loaded = {}

    if args.mode == "random":
        frame_agent = RandomPolicy(cfg.seed + 1, cfg, scheduling=True)
        slot_agent = RandomPolicy(cfg.seed + 2, cfg)
    else:
        import torch
        torch.set_num_threads(cfg.torch_num_threads)
        from hppo.ppo import PPOAgent

        frame_agent = PPOAgent(cfg.frame_obs_dim, cfg.frame_action_nvec, cfg, name="frame_ppo")
        slot_agent = PPOAgent(cfg.slot_obs_dim, cfg.slot_action_nvec, cfg, name="slot_ppo")
        if args.frame_checkpoint:
            loaded["frame"] = frame_agent.load(args.frame_checkpoint)
            print("loaded frame checkpoint:", loaded["frame"])
        if args.slot_checkpoint:
            loaded["slot"] = slot_agent.load(args.slot_checkpoint)
            print("loaded slot checkpoint:", loaded["slot"])
        if len(loaded) == 2 and loaded["frame"].get("pair_id") != loaded["slot"].get("pair_id"):
            raise ValueError("frame/slot checkpoints are from different saved pairs")

    save_resolved_config(cfg, args, root)
    runtime = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
               "mode": args.mode, "argv": list(argv) if argv is not None else sys.argv[1:],
               "base_commit": "a63b1da63addcb94e0f2e0bbe8b92dab5a9df318",
               "checkpoint_usage": "warm-start" if loaded and args.mode == "train" else args.mode,
               "checkpoint_sha256": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
                    for k, v in {"frame": args.frame_checkpoint, "slot": args.slot_checkpoint}.items() if v}}
    try:
        import torch
        runtime.update(torch=torch.__version__, cuda=torch.version.cuda,
                       cuda_available=torch.cuda.is_available(), device=cfg.device,
                       gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    except ModuleNotFoundError:
        runtime["torch"] = None
    source_root = Path(__file__).resolve().parents[1]
    runtime["code_sha256"] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in ("config*.py", "hppo/*.py", "env/p3/*.py", "agent/P3/*.py")
        for p in sorted(source_root.glob(pattern))}
    (root / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")
    logger = HistoryLogger(cfg, root)
    logger.event("run_start", {"schema": "scheduling-hppo-v2", "mode": args.mode,
                              "expected_episodes": cfg.train_episodes if args.mode == "train" else cfg.eval_episodes,
                              "episode_offset": cfg.episode_offset})

    try:
        print("--- P3 scheduling-only hierarchical PPO ---")
        print(f"regions={cfg.num_regions} users={cfg.num_users} J^R={cfg.rsu_capacity} J^U={cfg.uav_capacity} "
              f"frames={cfg.num_frames} slots/frame={cfg.frame_slots} points={cfg.num_candidate_points}")
        print(f"frame obs dim={cfg.frame_obs_dim} heads={len(cfg.frame_action_nvec)} | "
              f"slot obs dim={cfg.slot_obs_dim} heads={len(cfg.slot_action_nvec)}")
        print(f"reward mode={cfg.reward_mode} (CSI hidden from both agents); logs -> {root}")
    
        started = time.perf_counter()
        if args.mode == "train":
            dual = cfg.dual_init
            for ep in range(cfg.train_episodes):
                s = run_episode(env, frame_agent, slot_agent, cfg, logger, ep + cfg.episode_offset, dual, training=True, deterministic=False)
                if cfg.reward_mode == "objective_lagrangian":
                    dual = float(np.clip(dual + cfg.dual_lr * (s["z_constraint_mean"] - cfg.z_target_normalized), 0.0, cfg.dual_max))
                    s["dual_lambda_z"] = dual
                logger.log_episode(s)
                if ((ep + 1) % cfg.frame_update_every_episodes == 0 or ep == cfg.train_episodes - 1) and frame_agent.buffer_size() > 0:
                    metrics = frame_agent.update()
                    if metrics:
                        logger.event("frame_ppo_update", {"episode": ep + cfg.episode_offset, **metrics})
                        logger.debug(f"[FRAME PPO UPDATE] ep={ep} " + json.dumps(jsonable(metrics)))
                if (ep + 1) % cfg.save_every_episodes == 0 or ep == cfg.train_episodes - 1:
                    ck = root / args.checkpoint_dir
                    extra = {"episode": ep + cfg.episode_offset, "dual_lambda_z": dual,
                             "pair_id": f"{time.time_ns()}-ep{ep}-f{frame_agent.update_count}-s{slot_agent.update_count}"}
                    frame_agent.save(ck / f"frame_ep{ep + 1:05d}.pt", extra)
                    slot_agent.save(ck / f"slot_ep{ep + 1:05d}.pt", extra)
                    frame_agent.save(ck / "frame_latest.pt", extra)
                    slot_agent.save(ck / "slot_latest.pt", extra)
                print(f"EP {ep:04d} DPP/us={s['dpp_cost_per_user_slot']:8.3f} orig/us={s['original_cost_per_user_slot']:.4f} "
                      f"stall={s['stall_ratio']:.3f} served={s['served_user_ratio']:.3f} hire={s['hire_rate']:.3f} "
                      f"Zcost={s['z_constraint_mean']:.3f} proj={s['projection_events']} dual={dual:.3f} "
                      f"[{time.perf_counter() - started:.0f}s]", flush=True)
            if frame_agent.buffer_size() > 0:
                raise RuntimeError("unsaved final slow buffer; training did not flush correctly")
        else:
            if args.mode == "eval" and (not args.frame_checkpoint or not args.slot_checkpoint):
                raise ValueError("--mode eval requires --frame-checkpoint and --slot-checkpoint")
            rows = []
            for ep in range(cfg.eval_episodes):
                s = run_episode(env, frame_agent, slot_agent, cfg, logger, ep + cfg.episode_offset, cfg.dual_init, training=False,
                                deterministic=(cfg.deterministic_eval and args.mode == "eval"))
                logger.log_episode(s)
                rows.append(s)
                print(f"{args.mode.upper()} {ep:03d} DPP/us={s['dpp_cost_per_user_slot']:8.3f} orig/us={s['original_cost_per_user_slot']:.4f} "
                      f"stall={s['stall_ratio']:.3f} hire={s['hire_rate']:.3f} Q>Qe={s['q_gt_qe_rate']:.4f} "
                      f"viol(reserve,power)=({s['reserve_violations']},{s['power_violations']})", flush=True)
            if rows:
                for key in ("dpp_cost_per_user_slot", "original_cost_per_user_slot", "stall_ratio", "hire_rate"):
                    vals = np.asarray([r[key] for r in rows])
                    print(f"mean {key}: {vals.mean():.4f} (std {vals.std():.4f})")
        logger.event("run_end", {"status": "complete"})
        print(f"done in {time.perf_counter() - started:.1f}s; trace={root / 'trace.jsonl'} debug={root / 'debug.log'}")
    finally:
        logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])
