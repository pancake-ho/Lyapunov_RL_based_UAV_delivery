from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent.P3.features import build_candidate_feature_matrix, build_state_features
from agent.P3.ppo_agent import PPOAgent, PPOTransition, finish_trajectory
from agent.P3.slow_rollout_controller import SlowRolloutController
from config_p3 import P3Config
from env.p3.environment import (
    advance_mobility_one_frame,
    apply_region_result,
    generate_frame_trace,
    simulate_region_frame,
)
from env.p3.topology import initialize_state, region_membership, validate_state
from run.p3_common import run_policy, safe_ratio, write_csv


def plot_training_curve(rows: list[dict], output_path: Path) -> None:
    cache = Path(tempfile.gettempdir()) / "p3-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    episodes = [int(row["episode"]) for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    panels = (
        ("scaled_return", "Training return", "scaled negative frame DPP"),
        ("validation_dpp_per_user_slot", "Validation DPP", "lower is better"),
        ("stall_ratio", "Training stall ratio", "fraction"),
        ("average_quality_utility", "Delivered quality", "average utility"),
        ("hire_rate", "UAV hire rate", "fraction"),
        ("entropy", "Policy entropy", "categorical entropy"),
    )
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        values = [float(row[metric]) for row in rows]
        axis.plot(episodes, values, color="#7030A0", linewidth=1.4)
        axis.set_title(title)
        axis.set_xlabel("episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    fig.suptitle("P3 Upper-level PPO Training Diagnostics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def train_episode(
    agent: PPOAgent,
    cfg: P3Config,
) -> tuple[dict, list[PPOTransition]]:
    state = initialize_state(cfg)
    controller = SlowRolloutController(cfg)
    trajectories: dict[int, list[PPOTransition]] = {
        region: [] for region in range(cfg.num_regions)
    }
    totals = {
        "scaled_reward": 0.0,
        "dpp_cost": 0.0,
        "original_cost": 0.0,
        "stall_user_slots": 0.0,
        "delivered_chunks": 0.0,
        "quality_utility": 0.0,
        "hired_uav_frames": 0.0,
        "battery_reserve_violations": 0.0,
        "power_violations": 0.0,
        "provider_violations": 0.0,
    }

    for frame in range(cfg.num_frames):
        membership = region_membership(state, cfg)
        pending: list[dict] = []
        actions = []
        for region in range(cfg.num_regions):
            users = np.flatnonzero(membership == region).tolist()
            candidates = controller.candidate_actions(
                state,
                region,
                users,
                policy="ppo",
                frame=frame,
            )
            state_features = build_state_features(state, region, users, cfg)
            candidate_features = build_candidate_feature_matrix(
                state, candidates.actions, cfg
            )
            choice = agent.select(state_features, candidate_features, deterministic=False)
            action = candidates.actions[choice.action_index]
            actions.append(action)
            pending.append(
                {
                    "region": region,
                    "users": users,
                    "state_features": state_features,
                    "candidate_features": candidate_features,
                    "choice": choice,
                }
            )

        realized_trace = generate_frame_trace(cfg, controller.realized_seed(frame))
        frame_results = []
        for item, action in zip(pending, actions):
            result = simulate_region_frame(
                state,
                action,
                item["users"],
                realized_trace,
                cfg,
                controller.fast_controller,
            )
            frame_results.append(result)
            apply_region_result(state, item["region"], item["users"], result)
            totals["dpp_cost"] += result.frame_dpp_cost
            totals["original_cost"] += result.original_cost
            totals["stall_user_slots"] += result.stall_user_slots
            totals["delivered_chunks"] += result.delivered_chunks
            totals["quality_utility"] += result.quality_utility
            totals["hired_uav_frames"] += action.hired
            totals["battery_reserve_violations"] += result.battery_reserve_violations
            totals["power_violations"] += result.power_violations
            totals["provider_violations"] += result.provider_violations

        advance_mobility_one_frame(state, cfg)
        validate_state(state, cfg)
        done = frame == cfg.num_frames - 1
        next_membership = region_membership(state, cfg)
        for item, result in zip(pending, frame_results):
            region = int(item["region"])
            next_users = np.flatnonzero(next_membership == region).tolist()
            next_features = build_state_features(state, region, next_users, cfg)
            next_value = 0.0 if done else agent.value(next_features)
            reward = -result.frame_dpp_cost * cfg.ppo_reward_scale
            totals["scaled_reward"] += reward
            choice = item["choice"]
            trajectories[region].append(
                PPOTransition(
                    state_features=item["state_features"],
                    candidate_features=item["candidate_features"],
                    action_index=choice.action_index,
                    old_log_prob=choice.log_prob,
                    old_value=choice.value,
                    reward=float(reward),
                    next_value=float(next_value),
                    done=done,
                )
            )

    transitions: list[PPOTransition] = []
    for region_trajectory in trajectories.values():
        finish_trajectory(region_trajectory, cfg)
        transitions.extend(region_trajectory)
    total_user_slots = cfg.num_frames * cfg.num_users * cfg.frame_slots
    total_uav_frames = cfg.num_frames * cfg.num_regions
    summary = {
        "scaled_return": totals["scaled_reward"],
        "dpp_cost_per_user_slot": safe_ratio(totals["dpp_cost"], total_user_slots),
        "original_cost_per_user_slot": safe_ratio(
            totals["original_cost"], total_user_slots
        ),
        "stall_ratio": safe_ratio(totals["stall_user_slots"], total_user_slots),
        "average_quality_utility": safe_ratio(
            totals["quality_utility"], totals["delivered_chunks"]
        ),
        "hire_rate": safe_ratio(totals["hired_uav_frames"], total_uav_frames),
        "battery_reserve_violations": int(totals["battery_reserve_violations"]),
        "power_violations": int(totals["power_violations"]),
        "provider_violations": int(totals["provider_violations"]),
    }
    return summary, transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train upper-level PPO with exact P3 fast solver")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int, default=92026)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/p3_ppo"))
    args = parser.parse_args()
    if args.episodes <= 0 or args.frames <= 0:
        parser.error("episodes and frames must be positive")
    if args.validation_interval <= 0:
        parser.error("validation-interval must be positive")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_cfg = P3Config(seed=args.seed, num_frames=args.frames)
    agent = PPOAgent(agent_cfg, device=args.device)
    rows: list[dict] = []
    best_validation_dpp = math.inf
    started = time.perf_counter()

    for episode in range(args.episodes):
        env_cfg = replace(agent_cfg, seed=args.seed + episode * 1009)
        episode_summary, transitions = train_episode(agent, env_cfg)
        update_summary = agent.update(transitions)
        row = {
            "episode": episode,
            "environment_seed": env_cfg.seed,
            **episode_summary,
            **update_summary,
            "validation_dpp_per_user_slot": math.nan,
        }

        should_validate = (
            (episode + 1) % args.validation_interval == 0
            or episode == args.episodes - 1
        )
        if should_validate:
            validation_cfg = replace(
                agent_cfg,
                seed=args.validation_seed,
                num_frames=args.frames,
            )
            validation = run_policy(
                validation_cfg,
                "ppo",
                output_dir / "validation",
                ppo_agent=agent,
            ).summary
            validation_dpp = float(validation["dpp_cost_per_user_slot"])
            row["validation_dpp_per_user_slot"] = validation_dpp
            if validation_dpp < best_validation_dpp:
                best_validation_dpp = validation_dpp
                agent.save(
                    output_dir / "best.pt",
                    metadata={
                        "episode": episode,
                        "validation_seed": args.validation_seed,
                        "validation_dpp_per_user_slot": validation_dpp,
                    },
                )
        rows.append(row)
        write_csv(output_dir / "training_curve.csv", rows)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    agent.save(
        output_dir / "latest.pt",
        metadata={
            "episodes": args.episodes,
            "best_validation_dpp_per_user_slot": best_validation_dpp,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    plot_training_curve(rows, output_dir / "training_curve.png")
    print(f"checkpoints and logs: {output_dir}")


if __name__ == "__main__":
    main()
