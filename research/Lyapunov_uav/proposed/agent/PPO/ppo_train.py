from __future__ import annotations

import os
import sys
import time
import types
import random
import importlib.util
from dataclasses import dataclass
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from env import Env
from ppo_agent import PPOAgent
from ppo_buffer import RolloutBuffer

# 유틸
def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)