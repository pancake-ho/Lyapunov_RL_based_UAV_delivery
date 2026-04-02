from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from config import EnvConfig
from ppo_agent import PPOAgent
from ppo_buffer import RolloutBuffer
from ppo_config import PPOConfig
from env import Env


# utils
def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_state(state: dict) -> np.ndarray:
    return np.concatenate([
        np.asarray(state["Q"], dtype=np.float32).reshape(-1),
        np.asarray(state["Y"], dtype=np.float32).reshape(-1),
        np.asarray(state["E"], dtype=np.float32).reshape(-1),
        np.asarray(state["mu"], dtype=np.float32).reshape(-1),
        np.asarray(state["round_slot"], dtype=np.float32).reshape(-1),
        np.asarray(state["outage"], dtype=np.float32).reshape(-1),
    ]).astype(np.float32)


def train():
    seed = 2026
    set_seed(seed)

    env_cfg = EnvConfig()
    env_cfg.seed = seed

    env = Env(env_cfg)

    ppo_co