"""
DQN 알고리즘 가정

- DQN_MQ: 청크 "전송" 관련 신경망
- DQN_SCH: 스케줄링 관련 신경망
"""

import numpy as np
from collections import defaultdict, deque
import copy
import matplotlib.pyplot as plt
import sys
import os
import random
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

if not os.path.exists('DQN_results'):
    os.makedirs('DQN_results')

if not os.path.exists('DQN_models'):
    os.makedirs('DQN_models')

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

class DQN_MQ(nn.Module):
    """
    Chunk Delivery 전용 DQN
    """
    def __init__(self, input_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc_out(x)
        return q

class DQN_Scheduling(nn.Module):
    """
    Scheduling (MBS, UAV, RSU) 전용 DQN
    """
    def __init__(self, input_size: int, action_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
        nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, state_size_MQ: int, state_size_SCH: int, action_size_MQ: int, action_size_SCH: int):
        self.state_size_MQ = state_size_MQ
        self.state_size_SCH = state_size_SCH
        self.action_size_MQ = action_size_MQ
        self.action_size_SCH = action_size_SCH

        self.lr = 0.001
        self.discount_factor = 0.5
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.batch_size_MQ = 64
        self.batch_size_SCH = 64
        self.train_start = 500
        
        self.memory_MQ: deque = deque(maxlen=2000)
        self.memory_SCH: deque = deque(maxlen=2000)

        self.behavior_MQ_model = DQN_MQ(self.state_size_MQ, self.action_size_MQ)
        self.target_MQ_model = DQN_MQ(self.state_size_MQ, self.action_size_MQ)
        self.behavior_SCH_model = DQN_Scheduling(self.state_size_SCH, self.action_size_SCH)
        self.target_SCH_model = DQN_Scheduling(self.state_size_SCH, self.action_size_SCH)

        self.MQ_optimizer = optim.Adam(self.behavior_MQ_model.parameters(), lr=self.lr)
        self.SCH_optimizer = optim.Adam(self.behavior_SCH_model.parameters(), lr=self.lr)
        
        self.loss_fn = nn.MSELoss()

        # self.update_target_MQ_model()
        # self.update_target_SCH_model()

        self.T = 5 # Large Time Scale (스케줄링)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.behavior_MQ_model.to(self.device)
        self.target_MQ_model.to(self.device)
        self.behavior_SCH_model.to(self.device)
        self.target_SCH_model.to(self.device)
    
    @staticmethod
    def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    
    def update_target_MQ_model(self):
        self.target_MQ_model.load_state_dict(self.behavior_MQ_model.state_dict())
    
    def update_target_S_model(self):
        self.target_SCH_model.load_state_dict(self.behavior_SCH_model.state_dict())
    
    def append_MQ_sample(self, state, action, reward, next_state, done):
        self.memory_MQ.append((state, action, reward, next_state, done))
    
    def append_S_sample(self, state, action, reward, next_state, done):
        self.memory_SCH.append((state, action, reward, next_state, done))
    
    def train_MQ_model(self):
        # 메모리에 쌓인 데이터 부족 시 학습 스킵
        if len(self.memory_MQ) < max(self.train_start, self.batch_size_MQ):
            return
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.memory_MQ, self.batch_size_MQ)

        states = np.array([sample[0] for sample in mini_batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_batch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in mini_batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in mini_batch], dtype=np.float32)

        states_t = self._to_tensor(states, self.device)
        actions_t = self._to_tensor(actions, self.device)
        rewards_t = self._to_tensor(rewards, self.device)
        next_states_t = self._to_tensor(next_states, self.device)
        dones_t = self._to_tensor(dones, self.device)
        
        q_values = self.behavior_MQ_model(states_t) # (Batch_size, action_dim)
        q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_MQ_model(next_states_t)
            max_q_next, _ = torch.max(q_next, dim=-1)
            targets = rewards_t + (1.0 - dones_t) * self.discount_factor * max_q_next
        loss = self.loss_fn(q_pred, targets)

        self.MQ_optimizer.zero_grad()
        loss.backward()
        self.MQ_optimizer.step()
    
    def train_S_model(self):
        # 메모리에 쌓인 데이터 부족 시 학습 스킵
        if len(self.memory_SCH) < max(self.train_start, self.batch_size_SCH):
            return
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.memory_SCH, self.batch_size_SCH)

        states = np.array([sample[0] for sample in mini_batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_batch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in mini_batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in mini_batch], dtype=np.float32)

        states_t = self._to_tensor(states, self.device)
        actions_t = self._to_tensor(actions, self.device)
        rewards_t = self._to_tensor(rewards, self.device)
        next_states_t = self._to_tensor(next_states, self.device)
        dones_t = self._to_tensor(dones, self.device)
        
        q_values = self.behavior_SCH_model(states_t)
        q_pred = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target_SCH_model(next_states_t)
            max_q_next, _ = torch.max(q_next, dim=-1)
            targets = rewards_t + (1.0 - dones_t) * self.discount_factor * max_q_next
        loss = self.loss_fn(q_pred, targets)

        self.SCH_optimizer.zero_grad()
        loss.backward()
        self.SCH_optimizer.step()
    
    def save_models(self, episode):
        """ 특정 에피소드의 모델 저장 """
        model_name_base = f"UAV_DQN_episode_{episode}"
        torch.save(self.behavior_MQ_model.state_dict(), f'models/{model_name_base}_MQ.pth')
        torch.save(self.behavior_SCH_model.state_dict(), f'models/{model_name_base}_SCH.pth')
        print(f"DQN - {episode} 에피소드 모델 저장 완료")
    
    def load_models(self, episode):
        """ 특정 에피소드의 모델을 불러옴 """
        model_name_base = f"UAV_DQN_episode_{episode}"
        try:
            self.behavior_MQ_model.load_state_dict(torch.load(f'models/{model_name_base}_MQ.pth'))
            self.behavior_SCH_model.load_state_dict(torch.load(f'models/{model_name_base}_SCH.pth'))
            print(f"DQN - {episode} 에피소드 모델 불러오기 완료")
        except FileNotFoundError:
            print(f"오류: DQN - {episode} 에피소드의 모델 파일을 찾을 수 없습니다.")

    