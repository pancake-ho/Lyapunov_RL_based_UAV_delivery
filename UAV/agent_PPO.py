"""
PPO 알고리즘 가정
일단 기본적으로 "틀" 만 갖춤

- Actor
1. 세 개의 Head 설정
2. RSU head | UAV head | chunk, layer head

- Critic
1. 가치 함수 출력

- State_dim = 140 (100 + 10 + 10 + 10 + 10)
"""

# Coherence Time -> 이거 고려해서 Channel Modeling 변경 (cap 에 상수 하나 곱해주는 것이라 생각)
# 채널의 상태가 거의 안 변했다고 가정하고, 유지되는 시간? / Random 으로 생성하는 Channel Cap * (time) -> coherence time

import numpy as np
from collections import deque
import copy
import matplotlib.pyplot as plt
import sys
import os
import random
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

directory = './Based_Code/UAV/Result_PPO'
os.makedirs(directory, exist_ok=True)

# 헤드 수 규정
RSU_HEAD = 2
UAV_HEAD = 2
TX_HEAD = 10 # 임의로 가정한 것이라 수정 필요

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")


# 하이퍼파라미터
LEARNING_RATE = 1e-4
GAMMA = 0.96
LAMBDA = 0.97
EPOCHS = 10
BATCH_SIZE = 128
CLIP_EPSILON = 0.2
NUM_STEPS = 1024
MAX_EPISODES = 50000
entropy_coef = 0.01


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super(ActorNetwork, self).__init__()
        self.base = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.RSU_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, RSU_HEAD))
        self.UAV_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, UAV_HEAD))
        self.tx_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, TX_HEAD))
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        base_features = self.base(state)
        RSU_probs = torch.softmax(self.RSU_head(base_features), dim=-1)
        UAV_probs = torch.softmax(self.UAV_head(base_features), dim=-1)
        tx_probs = torch.softmax(self.tx_head(base_features), dim=-1)
        return RSU_probs, UAV_probs, tx_probs
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        return self.network(state)
    

class PPOAgent:
    def __init__(self, state_dim: int):
        self.actor = ActorNetwork(state_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.clear_memory()