import os
import numpy as np
import json
import matplotlib.pyplot as plt
import logging
import math
import random

import torch

from config import EnvConfig

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
        self.config = config
        
        self.uav_battery_queue = []
        self.num_rsu = 10