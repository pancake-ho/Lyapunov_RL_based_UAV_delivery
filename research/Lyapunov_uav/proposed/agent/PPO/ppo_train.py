import os
import time
import random
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from ppo_agent import PPOAgent
from ppo_buffer import RolloutBuffer

# 유틸
def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)