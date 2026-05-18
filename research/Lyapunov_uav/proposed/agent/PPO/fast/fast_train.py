from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from config import EnvConfig
from env.env import Env

from agent.PPO.common import (
    infer_flat_dim,
    split_env_reset,
    split_env_step,
    set_seed,
    ensure_dir,
)
from agent.PPO.common.utils import ScalarLogger, save_json
from agent.PPO.fast.fast_agent import FastPPOAgent, FastPPOConfig


def _find_proposed_root(start: Optional[Path] = None) -> Path:
    """
    proposed root를 자동 탐색함.

    기대 구조:
        proposed/
            config.py
            env/
            agent/
    """
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent

    for parent in [cur, *cur.parents]:
        if (parent / "config.py").exists() and (parent / "env").exists() and (parent / "agent").exists():
            return parent

    raise RuntimeError(
        "proposed root를 찾지 못했습니다.\n"
        "fast_train.py가 research/Lyapunov_uav/proposed/agent/PPO/fast/ 아래에 있는지 확인하세요."
    )

PROPOSED_ROOT = _find_proposed_root()
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-Timescale PPO Agent Training")

    # mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="train이면 PPO 학습, eval이면 평가만 수행.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="eval mode에서 불러올 checkpoint path."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="train mode에서 --checkpoint를 불러와 이어서 학습."
    )

    # experiment
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1000,
        help=(
            "train mode에서 학습할 episode 수. "
            "현재 기준 episode 1개 = slow-timescale round 1개."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20
    )
    parser.add_argument(
        "--save-every-episodes",
        type=int,
        default=50,
        help="train mode에서 몇 episode마다 checkpoint를 저장할지. 0이면 주기 저장 비활성화.",
    )

    # PPO rollout/update
    parser.add_argument(
        "--rollout-slots",
        type=int,
        default=1024,
        help=(
            "PPO update 1회 전에 수집할 fast slot transition 수."
        ),
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--init-log-std", type=float, default=-0.5)

    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Actor/Critic MLP hidden dimensions.",
    )
    parser.add_argument(
        "--no-obs-norm",
        action="store_true",
    )
    parser.add_argument(
        "--no-adv-norm",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='auto, cpu, cuda, cuda:0 등.',
    )
    parser.add_argument(
        "--deterministic-torch",
        action="store_true",
        help="재현성 우선. CUDA deterministic 옵션을 켠다.",
    )

    # env
    parser.add_argument("--num-user", type=int, default=None)
    parser.add_argument("--num-rsu", type=int, default=None)
    parser.add_argument("--num-uav", type=int, default=None)

    parser.add_argument(
        "--slow-T",
        dest="slow_T",
        type=int,
        default=None,
        help=(
            "slow-timescale round 길이. "
            "현재 fast-only 학습에서는 episode 하나의 fast slot 길이와 동일하다."
        ),
    )

    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)

    # reward coeffs
    parser.add_argument(
        "--V",
        type=float,
        default=1.0,
        help="env fast reward의 quality term 계수 V.",
    )

    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    """
    CLI args를 EnvConfig에 반영함.
    """
    cfg = EnvConfig(seed=int(args.seed))

    updates: Dict[str, Any] = {}

    if args.num_user is not None:
        updates["num_user"] = int(args.num_user)

    if args.num_rsu is not None and args.num_uav is not None:
        updates["num_rsu"] = int(args.num_rsu)
        updates["num_uav"] = int(args.num_uav)
    elif args.num_rsu is not None:
        updates["num_rsu"] = int(args.num_rsu)
        updates["num_uav"] = int(args.num_rsu)
    elif args.num_uav is not None:
        updates["num_rsu"] = int(args.num_uav)
        updates["num_uav"] = int(args.num_uav)

    if args.slow_T is not None:
        updates["slow_T"] = int(args.slow_T)

    if args.layer is not None:
        updates["layer"] = int(args.layer)
        updates["quality_weights"] = tuple(float(i + 1) for i in range(int(args.layer))
                                           )
    if args.chunk is not None:
        updates["chunk"] = int(args.chunk)

    if len(updates) > 0:
        cfg = replace(cfg, **updates)

    return cfg


def build_ppo_config(args: argparse.Namespace) -> FastPPOConfig:
    """
    CLI args를 FastPPOConfig로 변환함.
    """
    return FastPPOConfig(
        rollout_steps=int(args.rollout_slots),
        update_epochs=int(args.update_epochs),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        lr=float(args.lr),
        max_grad_norm=float(args.max_grad_norm),
        clip_coef=float(args.clip_coef),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        normalize_obs=not bool(args.no_obs_norm),
        normalize_adv=not bool(args.no_adv_norm),
        hidden_dims=tuple(int(x) for x in args.hidden_dims),
        init_log_std=float(args.init_log_std),
        device=str(args.device),
    )


def make_run_dir(args: argparse.Namespace, env_cfg: EnvConfig) -> Path:
    """
    실행 결과를 저장하는 directory를 생성함.
    """
    if args.run_name is None:
        run_name = (
            f"fast_ppo_{args.mode}"
            f"_ep{int(args.num_episode)}"
            f"_slowT{int(args.slow_T)}"
        )
    else:
        run_name = str(args.run_name)

    run_dir = PROPOSED_ROOT/ "fast" / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "logs")

    return run_dir


def attach_existing_slow_context(env: Env, obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    FastActionCodec이 obs에서 slow decision을 읽을 수 있도록,
    env가 이미 들고 있는 slow-timescale context만 obs에 붙임.

    여기서 slow action을 새로 만들지는 않음.
    """
    if not isinstance(obs, dict):
        raise TypeError(f"obs must be dict, got {type(obs)}")

    merged = dict(obs)

    if hasattr(env, "rsu_scheduling"):
        merged["rsu_scheduling"] = np.asarray(env.rsu_scheduling, dtype=np.int32).copy()
    if hasattr(env, "uav_hiring"):
        merged["uav_hiring"] = np.asarray(env.uav_hiring, dtype=np.int32).copy()
    if hasattr(env, "uav_scheduling"):
        merged["uav_scheduling"] = np.asarray(env.uav_scheduling, dtype=np.int32).copy()

    return merged


def save_configs(
    run_dir: Path,
    args: argparse.Namespace,
    env_cfg: EnvConfig,
    ppo_cfg: FastPPOConfig,
    obs_dim: int,
    action_dim: int,
) -> None:
    """
    실행 설정 저장.
    """
    save_json(vars(args), run_dir / "args.json")
    save_json(asdict(env_cfg), run_dir / "env_config.json")
    save_json(asdict(ppo_cfg), run_dir / "ppo_config.json")
    save_json(
        {
            "proposed_root": str(PROPOSED_ROOT),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        run_dir / "run_info.json",
    )


def reset_env(env: Env) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    env.reset() 결과를 Gymnasium/Gym 호환 형태로 정리.
    """
    obs, info = split_env_reset(env.reset())
    obs = attach_existing_slow_context(env, obs)
    return obs, info

def extract_info_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    """
    env.step() info에서 로그용 metric만 추출.
    """
    reward_components = info.get("reward_components", {})
    fast_components = reward_components.get("fast_reward_components", {})

    stall = info.get("stall", 0.0)
    try:
        stall_sum = float(np.sum(np.asarray(stall, dtype=np.float32)))
    except Exception:
        stall_sum = 0.0

    return {
        "sum_delivery": float(fast_components.get("sum_delivery", 0.0)),
        "sum_quality": float(fast_components.get("sum_quality", 0.0)),
        "sum_consumed_soc": float(fast_components.get("sum_consumed_soc", 0.0)),
        "sum_charged_soc": float(fast_components.get("sum_charged_soc", 0.0)),
        "num_hired_uav": float(fast_components.get("num_hired_uav", 0.0)),
        "num_charging_uav": float(fast_components.get("num_charging_uav", 0.0)),
        "num_outage_uav": float(fast_components.get("num_outage_uav", 0.0)),
        "stall_sum": stall_sum,
    }


def train(args: argparse.Namespace) -> None:
    set_seed(int(args.seed), deterministic=bool(args.deterministic_torch))

    env_cfg = build_env_config(args)
    ppo_cfg = build_ppo_config(args)

    env = Env(env_cfg)
    obs, reset_info = reset_env(env)

    obs_dim = infer_flat_dim(obs)
    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if args.resume:
        if args.checkpoint is None:
            raise ValueError("--resume 사용 시 --checkpoint가 필요합니다.")
        agent.load(
            path=args.checkpoint,
            strict=True,
            load_optimizer=True,
        )

    run_dir = make_run_dir(args, env_cfg)
    save_configs(
        run_dir=run_dir,
        args=args,
        env_cfg=env_cfg,
        ppo_cfg=ppo_cfg,
        obs_dim=obs_dim,
        action_dim=agent.action_dim,
    )

    episode_logger = ScalarLogger(run_dir / "logs" / "episodes.csv")
    update_logger = ScalarLogger(run_dir / "logs" / "updates.csv")

    print("=" * 100, flush=True)
    print("[FAST PPO TRAIN START]", flush=True)
    print(f"proposed_root : {PROPOSED_ROOT}", flush=True)
    print(f"run_dir       : {run_dir}", flush=True)
    print(f"device        : {agent.device}", flush=True)
    print(f"obs_dim       : {obs_dim}", flush=True)
    print(f"action_dim    : {agent.action_dim}", flush=True)
    print(f"slow_T        : {env_cfg.slow_T}", flush=True)
    print(f"rollout_slots : {ppo_cfg.rollout_steps}", flush=True)
    print(f"reset_info    : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    global_slot = 0
    update_idx = 0
    episode_idx = 0

    last_obs = obs
    last_done = False

    while episode_idx < int(args.num_episodes):
        obs, _ = reset_env(env)

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0

        for _ in range(int(env_cfg.slow_T)):
            selected = agent.select_action(
                obs=obs,
                deterministic=False,
                update_norm=True,
            )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )
            next_obs = attach_existing_slow_context(env, next_obs_raw)

            is_round_boundary = bool(info.get("is_round_boundary", False))
            done = bool(terminated or truncated or is_round_boundary)

            agent.store_transition(
                obs_vec=selected["obs_vec"],
                raw_action=selected["raw_action"],
                reward=float(reward),
                done=done,
                value=float(selected["value"]),
                log_prob=float(selected["log_prob"]),
            )

            metrics = extract_info_metrics(info)

            ep_reward += float(reward)
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_steps += 1
            global_slot += 1

            last_obs = next_obs
            last_done = done

            if agent.buffer.is_full:
                agent.finish_rollout(
                    last_obs=last_obs,
                    last_done=last_done,
                )
                update_logs = agent.update()
                update_idx += 1

                update_row = {
                    "update": update_idx,
                    "global_slot": global_slot,
                    "episode": episode_idx + 1,
                    **update_logs,
                }
                update_logger.write(update_row)

                print(
                    "[UPDATE] "
                    f"update={update_idx} "
                    f"slot={global_slot} "
                    f"episode={episode_idx + 1} "
                    f"policy_loss={update_logs['policy_loss']:.6f} "
                    f"value_loss={update_logs['value_loss']:.6f} "
                    f"entropy={update_logs['entropy']:.6f} "
                    f"ev={update_logs['explained_variance']:.4f}",
                    flush=True,
                )

            obs = next_obs

            if done:
                break

        episode_idx += 1

        episode_row = {
            "episode": episode_idx,
            "global_slot": global_slot,
            "episode_steps": ep_steps,
            "episode_reward": ep_reward,
            "episode_delivery": ep_delivery,
            "episode_quality": ep_quality,
            "episode_stall": ep_stall,
            "episode_consumed_soc": ep_consumed_soc,
            "episode_charged_soc": ep_charged_soc,
            "episode_outage": ep_outage,
        }
        episode_logger.write(episode_row)

        print(
            "[EPISODE] "
            f"ep={episode_idx}/{args.num_episodes} "
            f"steps={ep_steps} "
            f"reward={ep_reward:.4f} "
            f"delivery={ep_delivery:.4f} "
            f"quality={ep_quality:.4f} "
            f"stall={ep_stall:.4f}",
            flush=True,
        )

        if (
            int(args.save_every_episodes) > 0
            and episode_idx % int(args.save_every_episodes) == 0
        ):
            ckpt_path = run_dir / "checkpoints" / f"fast_ppo_ep{episode_idx}.pt"
            agent.save(
                ckpt_path,
                extra={
                    "episode": episode_idx,
                    "global_slot": global_slot,
                    "update_idx": update_idx,
                    "env_config": asdict(env_cfg),
                    "ppo_config": asdict(ppo_cfg),
                },
            )
            print(f"[SAVE] {ckpt_path}", flush=True)

    if len(agent.buffer) > 0:
        agent.finish_rollout(
            last_obs=last_obs,
            last_done=last_done,
        )
        update_logs = agent.update()
        update_idx += 1

        update_logger.write(
            {
                "update": update_idx,
                "global_slot": global_slot,
                "episode": episode_idx,
                **update_logs,
            }
        )

        print(
            "[FINAL UPDATE] "
            f"update={update_idx} "
            f"slot={global_slot} "
            f"policy_loss={update_logs['policy_loss']:.6f} "
            f"value_loss={update_logs['value_loss']:.6f}",
            flush=True,
        )

    final_ckpt_path = run_dir / "checkpoints" / "fast_ppo_final.pt"
    agent.save(
        final_ckpt_path,
        extra={
            "episode": episode_idx,
            "global_slot": global_slot,
            "update_idx": update_idx,
            "env_config": asdict(env_cfg),
            "ppo_config": asdict(ppo_cfg),
        },
    )

    save_json(
        {
            "episode": episode_idx,
            "global_slot": global_slot,
            "update_idx": update_idx,
            "final_checkpoint": str(final_ckpt_path),
            "run_dir": str(run_dir),
        },
        run_dir / "train_summary.json",
    )

    print("=" * 100, flush=True)
    print("[FAST PPO TRAIN DONE]", flush=True)
    print(f"final checkpoint: {final_ckpt_path}", flush=True)
    print("=" * 100, flush=True)


def evaluate(args: argparse.Namespace) -> None:
    set_seed(int(args.seed), deterministic=True)

    env_cfg = build_env_config(args)
    ppo_cfg = build_ppo_config(args)

    env = Env(env_cfg)
    obs, reset_info = reset_env(env)

    obs_dim = infer_flat_dim(obs)
    agent = FastPPOAgent(
        env_cfg=env_cfg,
        obs_dim=obs_dim,
        ppo_cfg=ppo_cfg,
    )

    if args.checkpoint is not None:
        agent.load(
            path=args.checkpoint,
            strict=True,
            load_optimizer=False,
        )

    run_dir = make_run_dir(args, env_cfg)
    save_configs(
        run_dir=run_dir,
        args=args,
        env_cfg=env_cfg,
        ppo_cfg=ppo_cfg,
        obs_dim=obs_dim,
        action_dim=agent.action_dim,
    )

    eval_logger = ScalarLogger(run_dir / "logs" / "eval_episodes.csv")

    print("=" * 100, flush=True)
    print("[FAST PPO EVAL START]", flush=True)
    print(f"proposed_root : {PROPOSED_ROOT}", flush=True)
    print(f"run_dir       : {run_dir}", flush=True)
    print(f"device        : {agent.device}", flush=True)
    print(f"obs_dim       : {obs_dim}", flush=True)
    print(f"action_dim    : {agent.action_dim}", flush=True)
    print(f"slow_T        : {env_cfg.slow_T}", flush=True)
    print(f"checkpoint    : {args.checkpoint}", flush=True)
    print(f"reset_info    : {reset_info}", flush=True)
    print("=" * 100, flush=True)

    rewards = []
    deliveries = []
    qualities = []
    stalls = []

    for episode_idx in range(1, int(args.eval_episodes) + 1):
        obs, _ = reset_env(env)

        ep_reward = 0.0
        ep_delivery = 0.0
        ep_quality = 0.0
        ep_stall = 0.0
        ep_consumed_soc = 0.0
        ep_charged_soc = 0.0
        ep_outage = 0.0
        ep_steps = 0

        for _ in range(int(env_cfg.slow_T)):
            selected = agent.select_action(
                obs=obs,
                deterministic=True,
                update_norm=False,
            )

            next_obs_raw, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )
            next_obs = attach_existing_slow_context(env, next_obs_raw)

            metrics = extract_info_metrics(info)

            ep_reward += float(reward)
            ep_delivery += metrics["sum_delivery"]
            ep_quality += metrics["sum_quality"]
            ep_stall += metrics["stall_sum"]
            ep_consumed_soc += metrics["sum_consumed_soc"]
            ep_charged_soc += metrics["sum_charged_soc"]
            ep_outage += metrics["num_outage_uav"]

            ep_steps += 1

            done = bool(
                terminated
                or truncated
                or bool(info.get("is_round_boundary", False))
            )

            obs = next_obs

            if done:
                break

        rewards.append(ep_reward)
        deliveries.append(ep_delivery)
        qualities.append(ep_quality)
        stalls.append(ep_stall)

        eval_logger.write(
            {
                "episode": episode_idx,
                "episode_steps": ep_steps,
                "episode_reward": ep_reward,
                "episode_delivery": ep_delivery,
                "episode_quality": ep_quality,
                "episode_stall": ep_stall,
                "episode_consumed_soc": ep_consumed_soc,
                "episode_charged_soc": ep_charged_soc,
                "episode_outage": ep_outage,
            }
        )

        print(
            "[EVAL] "
            f"ep={episode_idx}/{args.eval_episodes} "
            f"steps={ep_steps} "
            f"reward={ep_reward:.4f} "
            f"delivery={ep_delivery:.4f} "
            f"quality={ep_quality:.4f} "
            f"stall={ep_stall:.4f}",
            flush=True,
        )

    summary = {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "delivery_mean": float(np.mean(deliveries)) if deliveries else 0.0,
        "quality_mean": float(np.mean(qualities)) if qualities else 0.0,
        "stall_mean": float(np.mean(stalls)) if stalls else 0.0,
        "checkpoint": args.checkpoint,
        "run_dir": str(run_dir),
    }

    save_json(summary, run_dir / "eval_summary.json")

    print("=" * 100, flush=True)
    print("[FAST PPO EVAL DONE]", flush=True)
    print(
        f"reward_mean={summary['reward_mean']:.4f}, "
        f"delivery_mean={summary['delivery_mean']:.4f}, "
        f"quality_mean={summary['quality_mean']:.4f}, "
        f"stall_mean={summary['stall_mean']:.4f}",
        flush=True,
    )
    print("=" * 100, flush=True)


def main() -> None:
    args = parse_args()

    if int(args.num_episodes) <= 0:
        raise ValueError("--num-episodes는 양수여야 합니다.")
    if int(args.eval_episodes) <= 0:
        raise ValueError("--eval-episodes는 양수여야 합니다.")
    if int(args.rollout_slots) <= 0:
        raise ValueError("--rollout-slots는 양수여야 합니다.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size는 양수여야 합니다.")
    if int(args.update_epochs) <= 0:
        raise ValueError("--update-epochs는 양수여야 합니다.")

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise ValueError(f"지원하지 않는 mode입니다: {args.mode}")


if __name__ == "__main__":
    main()