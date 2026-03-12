import os
import math
import random
from dataclasses import dataclass
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer

# Seed 및 Device 정의
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {DEVICE}")


# Replay Buffer 정의
