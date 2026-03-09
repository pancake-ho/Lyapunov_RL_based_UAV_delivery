import os
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
import math
import random
from typing import Dict

import torch

from config import EnvConfig
from channel import ChannelModel
from battery import UAVBattery

seed = 2025
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# 결과 저장을 위한 디렉토리 생성
if not os.path.exists('results'):
    os.makedirs('results')

# 모델 저장을 위한 디렉토리 생성
if not os.path.exists('models'):
    os.makedirs('models')

class Env:
    """
    Scenario 2 전용 Env 클래스
    RSU, UAV, MBS 로 전체 시스템이 구성되며 다음과 같은 기능으로 동작을 수행
    """
    def __init__(self, config: EnvConfig):
        self.cfg = config
        
        # 시스템 설정
        # user, uav, rsu 수 정의
        self.num_rsu = self.cfg.num_rsu
        self.num_user = self.cfg.num_user
        self.num_uav = self.cfg.num_uav
        self.slow_T = self.cfg.slow_T
        self.N0 = self.cfg.N0

        # 비디오 및 캐싱 시스템 설정
        self.num_video = self.cfg.num_video
        self.rsu_caching = self.cfg.rsu_caching
        self.layer = self.cfg.layer
        self.chunk = self.cfg.chunk
        self.rsu_capacity = self.cfg.rsu_capacity

        # 사용자 이동 패턴 설정
        self.spawn_base = self.cfg.spawn_base
        self.spawn_amp = self.cfg.spawn_amp
        self.spawn_period = self.cfg.spawn_period
        self.depart_base = self.cfg.depart_base
        self.depart_amp = self.cfg.depart_amp
        self.depart_period = self.cfg.depart_period

        # 채널 설정
        self.rsu_channel = self.cfg.rsu_channel
        self.uav_channel = self.cfg.uav_channel

        # 배터리 모델
        self.E_max = self.cfg.E_max
        self.E_min = self.cfg.E_min
        self.E = np.full(self.num_uav, self.E_max, dtype=np.float32)
        self.Y = np.zeros(self.num_uav, dtype=np.float32)

        # 라운드 단위 관리
        self.round_slot = 0
        self.round_start_E = self.E.copy()

        # UAV 상태
        self.mu = np.zeros(self.num_uav, dtype=np.int32)
        self.outage = np.zeros(self.num_uav, dtype=np.int32)

        # charging/power 관련 상태
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.tx_power = np.zeros(self.num_uav, dtype=np.float32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.int32)

        # 충전소
        self.charge_rate = self.cfg.e_charge

        # 스케줄링 상태
        self.scheduled_users = np.zeros((self.num_uav, self.num_user), dtype=np.int32)

        self.t = 0
        self.reset()
    
    @property
    def E(self):
        return np.array([b.soc for b in self.batteries], dtype=np.float32)
    
    @property
    def Y(self):
        return np.array([b.virtual_q for b in self.batteries], dtype=np.float32)
    
    def _start_new_round(self):
        self.round_idx = self.t // self.slow_T
        self.round_slot = 0
        self.round_start_soc = np.array([b.soc for b in self.batteries], dtype=np.float32)
        for b in self.batteries:
            b.start_round(round_horizon=self.slow_T)
        
    def reset(self):
        """
        에피소드 초기화 수행 함수
        큐, 배터리, 위치 등 State 변수 초기화
        """
        # 시간 축 및 에피소드 초기화    
        self.t = 0
        self.episode = 0

        # 사용자 큐 초기화
        self.queue = np.zeros(self.num_user, dtype=np.float32)

        # 배터리 큐 및 Virtual queue 초기화
        # 배터리 큐의 경우, 초기에는 Full Charging 을 가정함
        self.E = np.full(self.num_uav, self.E_max, dtype=np.float32)
        self.Y = np.zeros(self.num_uav, dtype=np.float32)

        # 충전 상태 초기화
        # 아무 uav 도 충전 안하고 있다고 가정
        self.charge_counters = np.zeros(self.cfg.num_uav, dtype=np.int32)

        # UAV 고용 상태 초기화
        # 초기 상태이므로, 미고용 상태를 가정함
        self.mu = np.zeros(self.num_uav, dtype=np.int32)

        # 라운드 초기화
        self.round_slot = 0
        self.round_start_E = self.E.copy()
        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charge_state = np.zeros(self.num_uav, dtype=np.int32)
        self.tx_power = np.zeros(self.num_uav, dtype=np.int32)
        self.scheduled_users = np.zeros((self.num_uav, self.num_user), dtype=np.int32)

        return self._get_state()

    def _get_state(self):
        """
        현 상태값을 반환하는 함수
        """
        return {
            "Q": self.queue.copy(),
            "Y": self.Y.copy(),
            "E": self.E.copy(),
            "mu": self.mu.copy(),
            "round_slot": np.array([self.round_slot], dtype=np.int32),
            "outage": self.outage.copy(),
        }
    
    def step(self, action: Dict[str, np.ndarray]):
        """
        환경의 1 time step 진행 함수 (Fast Timescale 기준)

        - Slow Timescale: UAV 고용 및 스케줄링 수행
        - Fast Timescale: 비디오 청크 및 레이어 전송 수행
        """
        # Slow Timescale
        # 매 T 주기마다, UAV 고용 여부를 갱신함
        # 또한, User scheduling을 갱신함
        if self.t % self.slow_T == 0:
            new_mu = action.get("hiring", np.zeros(self.num_uav, dtype=np.int32))
            scheduled_users = action.get("scheduled_users", np.zeros((self.num_uav, self.num_user), dtype=np.int32))
        else:
            new_mu = self.mu.copy()
            scheduled_users = self.scheduled_users.copy()

        # Fast Timescale
        # 매 time-step 마다, l_n (chunk) 및 k_n (layer) 를 전송함
        # 또한, 전송 전력및 충전, served user 수도 선택함
        served_users = action.get("served_users", np.zeros((self.num_uav, self.num_user), dtype=np.int32))
        l_n = action.get("chunk", np.zeros(self.num_user, dtype=np.int32))
        k_n = action.get("layer", np.ones(self.num_user, dtype=np.int32))
        tx_power = action.get("tx_power", np.zeros(self.num_uav, dtype=np.float32))
        charge = action.get("charge", np.zeros(self.num_uav, dtype=np.int32))

        # 배터리 Queue
        # E(t+1) = max(E_t - e_u, 0) + e_c
        prev_E = self.E.copy()
        prev_Y = self.Y.copy()

        for u in range(self.num_uav):
            serving_flag = int(served_users[u].sum() > 0)

            # UAV u가 실제로 전송한 총 chunk 수
            delivery_chunks_u = float(np.sum(served_users[u] * l_n))

            # hovering energy
            hover_e = self.hover_e if serving_flag == 1 else 0.0

            # communication energy
            comm_e = tx_power if serving_flag == 1 else 0.0
            total_consume = hover_e + comm_e

            # charging energy
            charge_e = self.charge_rate if charge[u] == 1 else 0.0

            # actual battery queue update
            self.E[u] = np.clip(self.E[u] - total_consume + charge_e, 0.0, self.E_max)

            # round-based virtual queue update
            allowed_use = max(self.round_start_E[u] - self.E_min, 0.0) / min(1, self.slow_T)
            self.Y[u] = max(0.0, self.Y[u] + total_consume - charge_e - allowed_use)