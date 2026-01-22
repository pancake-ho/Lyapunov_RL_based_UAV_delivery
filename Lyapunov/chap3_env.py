import numpy as np
import os
import logging
import math
import json
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

class ChannelModel:
    def __init__(self, distance: float, bandwidth: float=1.0, snr_db: float=20.0):
        self.distance = distance
        self.bandwidth = bandwidth
        self.snr_lin = 10 ** (snr_db / 10)
        self.path_loss_exp = 2.0
    
    def get_capacity(self) -> float:
        h_sq = np.random.exponential(1.0)
        signal_power = self.snr_lin * h_sq / (self.distance ** self.path_loss_exp)
        capacity = self.bandwidth * np.log2(1.0 + signal_power)
        return max(0.0, capacity)

class VideoEnv:
    def __init__(self, num_users: int=5, num_rsus: int=3, max_steps: int=1000):
        self.num_users = num_users
        self.num_rsus = num_rsus
        self.max_steps = max_steps

        # 목표 버퍼량
        self.target_buffer = 10.0

        # Trade-off 가중치
        self.V = 2.0 

        self.channel_rsu = ChannelModel(distance=0.5, snr_db=25) # 좀 더 안정적
        self.channel_uav = ChannelModel(distance=0.3, snr_db=30) # UAV 는 더 가까움

        self.playback_buffers = np.zeros(num_users)
        self.virtual_queues = np.zeros(num_users)
        self.user_location = np.random.randint(0, num_rsus, num_users)

        self.t = 0
    
    def reset(self):
        self.t = 0
        self.playback_buffers = np.full(self.num_users, self.target_buffer / 2)
        self._update_virtual_queues()
        return self._get_state()
    
    def _update_virtual_queues(self):
        """
        Q(t) = max(0, Target - Actual Buffer)
        실제 버퍼가 목표(Target)보다 적을수록 Q 가 커져, drift 가 커지고 따라서 스케줄링 우선순위가 올라감
        """
        self.virtual_queues = np.maximum(0, self.target_buffer - self.playback_buffers)
    
    def _get_state(self):
        """
        State: [virtual_queues, user_locations]
        """
        norm_q = self.virtual_queues / self.target_buffer
        return np.concatenate([norm_q, self.user_location])

    def step(self, action: np.ndarray):
        """
        Action 의 구조: [user 0 의 선택, user 1의 선택, ...]
        선택지: 0(대기), 1(RSU 연결), 2(UAV 연결)      
        """
        rewards = 0
        penalty_cost = 0
        sum_rate = np.zeros(self.num_users)

        # 전송률 계산 및 비용 산정
        for n, act in enumerate(action):
            act = int(act)
            rate = 0
            cost = 0

            if act == 1:
                rate = self.channel_rsu.get_capacity()
                cost = 1.0
            elif act == 2:
                rate = self.channel_uav.get_capacity()
                cost = 2.5
            
            sum_rate[n] = rate
            penalty_cost += cost
        
        # 버퍼 업데이트
        playback_rate = 1.0
        self.playback_buffers = np.maximum(0, self.playback_buffers - playback_rate) + sum_rate

        # virtual queue 업데이트
        prev_virtual_queues = self.virtual_queues.copy()
        self._update_virtual_queues()

        # Lyapunov Drift-plus-penalty 계산
        # Drift = 1/2 * (Q(t+1)^2 - Q(t)^2)
        # Approx Reward = Q(t) * mu(t) - V * cost(t)
        # 학습을 위해, "드리프트 감소량" 을 보상으로 제시
        lyapunov_drift = 0.5 * np.sum(prev_virtual_queues ** 2 - self.virtual_queues ** 2)
        reward = lyapunov_drift - (self.V * penalty_cost)

        # stall penalty
        stall_penalty = np.sum(self.playback_buffers == 0) * 10.0
        reward -= stall_penalty

        self.t += 1
        done = (self.t >= self.max_steps)

        info = {
            "avg_buffer": np.mean(self.playback_buffers),
            "avg_virtual_queue": np.mean(self.virtual_queues),
            "cost": penalty_cost
        }

        return self._get_state(), reward, done, info