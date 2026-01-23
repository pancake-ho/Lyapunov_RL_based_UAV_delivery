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
        self.uav_user_cap = self.cfg.uav_user_cap
        self.slow_T = self.cfg.slow_T
        self.N0 = self.cfg.N0

        # 비디오 및 캐싱 시스템 설정
        self.num_video = self.cfg.num_video
        self.rsu_caching = self.cfg.rsu_caching
        self.layer = self.cfg.layer
        self.chunk = self.cfg.chunk
        self.rsu_capacity = self.cfg.rsu_capacity
        self.mbs_capacity = self.cfg.mbs_capacity
        self.mbs_delay = self.cfg.mbs_delay
        self.zipf_alpha = self.cfg.zipf_alpha

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
        self.e_utility = self.cfg.e_utility
        self.e_charge = self.cfg.e_charge
        self.V = 1.0

        self.reset()
        
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

        # UAV 고용 상태 초기화
        # 초기 상태이므로, 미고용 상태를 가정함
        self.mu = np.zeros(self.num_uav, dtype=np.int32)

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
        }
    
    def step(self, action: Dict[str, np.ndarray]):
        """
        환경의 1 time step 진행 함수 (Fast Timescale 기준)

        - Slow Timescale: UAV 고용 및 스케줄링 수행
        - Fast Timescale: 비디오 청크 및 레이어 전송 수행
        """
        # Slow Timescale
        # 매 T 주기마다, UAV 고용 여부를 갱신함
        if self.t % self.slow_T == 0:
            new_mu = action.get("hiring", np.zeros(self.num_uav))

            # 배터리 큐 관련 제약
            # 현 배터리가 최소로 지정한 배터리보다 작은 경우, mu = 0 을 강제함
            available_hiring_mask = (self.E >= self.E_min).astype(np.int32)
            self.mu = new_mu.astype(np.int32) * available_hiring_mask
        
        # Fast Timescale
        # 매 time-step 마다, l_n (chunk) 및 k_n (layer) 를 전송함