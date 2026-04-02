from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical, Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def _init_layer(layer: nn.Linear, gain: float) -> None:
    """
    PPO 안정화를 위한 orthogonal initialization 적용 함수
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """
    Fast-timescale low policy용 공통 2층 Fully Connected Network 클래스로,
    slot-level observation과 slow-timescale decision을 받아 latent feature 생성
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim은 양수 값을 가져야 합니다, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim은 양수 값을 가져야 합니다, got {hidden_dim}")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self._init_backbone()
    
    def _init_backbone(self) -> None:
        """
        PPO 안정화를 위해 backbone layer도 orthogonal init 적용, 그를 구현하는 함수
        """
        gain = torch.sqrt(torch.Tensor(2.0)).item()
        for module in self.net:
            if isinstance(module, nn.Linear):
                _init_layer(module, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation 구현 함수
        """
        # 타입 검증
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"MLP Network에는 torch.Tensor 타입의 변수가 들어가야 합니다, got {type(x)}")
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim != 2:
            raise ValueError(
                f"MLP Network에는 (obs_dim,) 또는 (batch, obs_dim) shape이 들어가야 합니다, got {tuple(x.shape)}"
            )
        
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MLP Network는 {self.input_dim}의 input dim을 기대합니다, got {x.shape[-1]}"
            )

        return self.net(x)
    

@dataclass
class FastActionSpaceConfig:
    """
    fast-timescale action head 설정값 클래스

    num_users:
        현재 UAV가 지원 가능한 최대 user 수
    max_chunk_lavel:
        chunk delivery decision 총 수
    max_quality_level:
        layer level 총 수
    """
    num_users: int
    max_chunk_level: int
    max_quality_level: int

    