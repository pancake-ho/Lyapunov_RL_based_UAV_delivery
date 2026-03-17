from typing import Tuple

import torch
import torch.nn as nn

from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class MLP(nn.Module):
    """
    2층 Fully Connected Network로,
    observation을 받아서 action과 critic이 공통으로 쓸 latent feature 얻는 함수
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim) # 각 action dimension의 평균 mean

        # 전체 policy가 공유하는 std를 학습함, 즉 global parameter 느낌
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5)) # 각 action dimension의 로그 표준편차
        self.critic = nn.Linear(hidden_dim, 1)

        # 안정화 기법
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01) # actor output layer gain을 작게 잡음으로써 안정적인 action 유도
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0) # critic은 gain 1.0으로 일반적인 가치 추정 초기화
        nn.init.zeros_(self.critic.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_out = self.backbone(obs)
        mean = self.actor_mean(backbone_out)
        log_std = torch.clamp(self.actor.log_std, LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        value = self.critic(self.backbone).squeeze(-1) # critic 출력 (batch, 1)인데 PPO 계산 편하게 (batch,)으로 바꿈
        return mean, log_std, value
    
    def get_dict(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        actor가 정규분포 정책을 만들게 하는 함수
        """
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, mean, value