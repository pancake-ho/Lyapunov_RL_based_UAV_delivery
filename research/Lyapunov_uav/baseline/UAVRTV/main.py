"""
Train / evaluate the SAC baseline (Wu et al. 2024) in the fixed
RSU+UAV vehicular video-delivery scenario.

Examples
--------
  # environment sanity check with random actions, full per-slot log
  python main.py --agent random --episodes 2 --frames_per_episode 5 --log_dir runs/rand

  # SAC training (reference-paper setting: lr=1e-3, bandwidth allocated by SAC)
  python main.py --agent sac --episodes 300 --log_dir runs/sac --slot_log_every_ep 10

  # scenario-default radio model (fixed reserved RBs) + threshold hiring
  python main.py --bw_mode fixed --hire_mode threshold --log_dir runs/sac_fixedrb

  # evaluate a checkpoint deterministically
  python main.py --eval_only true --load_path runs/sac/sac.pt --episodes 5 --log_dir runs/eval
"""
import os
import time

import numpy as np

from args import parse_args, save_args
from env import VehicularVideoEnv
from logger import EpisodeCSV, SlotLogger, setup_run_logger


class RandomAgent:
    def __init__(self, act_dim, rng):
        self.act_dim, self.rng = act_dim, rng

    def act(self, obs, deterministic=False):
        return self.rng.uniform(-1, 1, self.act_dim)


def run_episode(env, agent, args, ep, mode, buffer, rng, slot_logger, run_log, log_slots, train):
    obs = env.reset()
    M = env.M
    stats = {k: 0.0 for k in ("reward", "quality", "switch", "stall", "energy", "hire",
                              "stalls", "user_slots", "chunks", "chunk_utility", "hired_slots",
                              "uav_energy_J", "uav_users_served", "forced_returns", "switches")}
    losses = []
    done = False
    while not done:
        actions = {}
        for m in range(M):
            if train and args.agent == "sac" and run_episode.total_steps < args.start_steps:
                actions[m] = rng.uniform(-1, 1, env.act_dim)
            else:
                actions[m] = agent.act(obs[m], deterministic=(mode == "eval"))
        next_obs, rewards, done, info = env.step(actions)
        for m in range(M):
            L = info["region_logs"][m]
            stats["reward"] += rewards[m]
            for k in ("quality", "switch", "stall", "energy", "hire"):
                stats[k] += L["reward"][k]
            for v in L["vehicles_after"]:
                stats["stalls"] += v["stall"]
                stats["user_slots"] += 1
                stats["chunks"] += v["d"]
                if v["d"] > 0:
                    stats["chunk_utility"] += v["d"] * env.U[v["k"] - 1]
                if v["switch_bps"] > 0:
                    stats["switches"] += 1
            stats["hired_slots"] += float(L["uav_after"]["hired"] or L["uav_after"]["forced_return"])
            stats["uav_energy_J"] += L["uav_move"].get("e_slot_J", 0.0)
            stats["uav_users_served"] += len(L["uav_decisions"])
            stats["forced_returns"] += sum(1 for e in L["events"] if e.startswith("FORCED"))
            if train and buffer is not None:
                buffer.add(obs[m], actions[m], rewards[m], next_obs[m], done)
                run_episode.total_steps += 1
        if log_slots:
            slot_logger.log(ep, mode, info)
        if train and args.agent == "sac" and buffer.n >= args.batch_size and \
                run_episode.total_steps >= args.start_steps:
            for _ in range(args.updates_per_step):
                losses.append(agent.update(buffer.sample(args.batch_size, rng)))
        obs = next_obs

    slots = env.T * args.frames_per_episode * M
    row = {"episode": ep, "mode": mode, "return_total": round(stats["reward"], 2),
           "return_per_region_slot": round(stats["reward"] / slots, 4),
           "quality": round(stats["quality"], 2), "switch_pen": round(stats["switch"], 3),
           "stall_pen": round(stats["stall"], 2), "energy_pen": round(stats["energy"], 3),
           "hire_pen": round(stats["hire"], 3),
           "stall_ratio": round(stats["stalls"] / max(stats["user_slots"], 1), 4),
           "avg_chunk_utility": round(stats["chunk_utility"] / max(stats["chunks"], 1), 4),
           "chunks": int(stats["chunks"]), "switch_events": int(stats["switches"]),
           "hire_rate": round(stats["hired_slots"] / slots, 4),
           "uav_energy_Wh": round(stats["uav_energy_J"] / 3600, 3),
           "uav_user_slots": int(stats["uav_users_served"]),
           "forced_returns": int(stats["forced_returns"]),
           "steps": run_episode.total_steps}
    if losses:
        row.update({k: round(float(np.mean([l[k] for l in losses])), 4) for k in losses[0]})
    return row


run_episode.total_steps = 0


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    save_args(args, os.path.join(args.log_dir, "config.json"))
    run_log = setup_run_logger(args.log_dir, args.log_level)
    rng = np.random.default_rng(args.seed)
    env = VehicularVideoEnv(args, seed=args.seed)
    run_log.info(f"obs_dim={env.obs_dim} act_dim={env.act_dim} regions={env.M} "
                 f"P_hover={env.P_hover:.1f}W P_prop_max={env.P_prop_max:.1f}W E_max={env.E_max:.0f}J E_th={env.E_th:.0f}J")

    buffer = None
    if args.agent == "sac":
        import torch
        torch.manual_seed(args.seed)
        from sac import ReplayBuffer, SACAgent
        agent = SACAgent(env.obs_dim, env.act_dim, args)
        if args.load_path:
            agent.load(args.load_path)
            run_log.info(f"loaded checkpoint {args.load_path}")
        buffer = ReplayBuffer(env.obs_dim, env.act_dim, args.buffer_size)
    else:
        agent = RandomAgent(env.act_dim, rng)

    slot_logger = SlotLogger(args.log_dir, enabled=args.slot_log, console=args.console_slot_log)
    csv_out = EpisodeCSV(args.log_dir)
    train = not args.eval_only
    t0 = time.time()
    for ep in range(args.episodes):
        mode = "eval" if args.eval_only else "train"
        log_slots = args.slot_log and (args.eval_only or ep % args.slot_log_every_ep == 0)
        row = run_episode(env, agent, args, ep, mode, buffer, rng, slot_logger, run_log, log_slots, train)
        csv_out.write(row)
        run_log.info(" ".join(f"{k}={v}" for k, v in row.items()) + f" elapsed={time.time()-t0:.0f}s")
        if train and args.agent == "sac" and (ep + 1) % args.eval_every == 0:
            row = run_episode(env, agent, args, ep, "eval", None, rng, slot_logger, run_log, args.slot_log, False)
            csv_out.write(row)
            run_log.info("EVAL " + " ".join(f"{k}={v}" for k, v in row.items()))
            agent.save(args.save_path)
    if train and args.agent == "sac":
        agent.save(args.save_path)
        run_log.info(f"saved checkpoint {args.save_path}")
    slot_logger.close()
    csv_out.close()


if __name__ == "__main__":
    main()
