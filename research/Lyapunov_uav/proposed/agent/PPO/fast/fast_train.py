from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
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

    if cur.is_file:
        cur = cur.parent
    
    for parent in [cur, *cur.parents]:
        if (parent / "config.py").exists() and (parent / "env").exists():
            return parent
        
    raise RuntimeError(
        "proposed root를 찾지 못했습니다.\n"
        "fast_train.py의 위치를 확인하세요."
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

    # experiment
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
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
        default=20,
        help="eval mode에서 평가할 episode 수.",
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
            "PPO update 한 번 전에 수집할 fast slot transition 수. "
            "episode 길이와 다르며, PPO rollout horizon이다."
        ),
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=5,
        help="PPO update 시 rollout buffer를 반복 학습할 epoch 수.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO minibatch size.",
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
        help="Observation normalization 비활성화.",
    )
    parser.add_argument(
        "--no-adv-norm",
        action="store_true",
        help="Advantage normalization 비활성화.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='auto, cpu, cuda, cuda:0 등.',
    )

    # env
    parser.add_argument("--num-user", type=int, default=None)
    parser.add_argument("--num-rsu", type=int, default=None)
    parser.add_argument("--num-uav", type=int, default=None)

    parser.add_argument(
        "--slow-T",
        type=int,
        default=None,
        help=(
            "slow-timescale 갱신 길이. "
            "현재 fast-only 학습에서는 episode 하나의 fast slot 길이와 동일하다."
        ),
    )

    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)

    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    """
    CLI args를 EnvConfig에 반영함.
    """
    cfg = EnvConfig(seed=int(args.seed))

    updates: Dict[str, Any] = {}

    if args.num_user is not None:
        updates["num_user"] = int(args.num_user)
    if args.num_rsu is not None:
        updates["num_rsu"] = int(args.num_rsu)
    if args.num_uav is not None:
        updates["num_uav"] = int(args.num_uav)

    if args.slow_T is not None:
        updates["slow_T"] = int(args.slow_T)

    if args.layer is not None:
        updates["layer"] = int(args.layer)
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


def make_run_dir(args: argparse.Namespace) -> Path:
    """
    실행 결과 저장 directory 생성.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"fast_ppo_seed{args.seed}_{timestamp}"
    run_dir = PROPOSED_ROOT / "runs" / "fast" / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    return run_dir