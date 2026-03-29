from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _init_layer(layer: nn.Linear, gain: float) -> None:
    """
    PPO에서 쓰이는, orthogonal init 적용 함수
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """
    2층 Fully Connected Network 클래스로,
    observation을 받아서 action과 critic이 공통으로 쓸 latent feature를 생성함.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

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
        for module in self.net:
            if isinstance(module, nn.Linear):
                _init_layer(module, gain=torch.sqrt(torch.tensor(2.0)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation 구현 함수
        """
        # 타입 검증
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"MLP Network에는 torch.Tensor 타입의 변수가 들어가야 합니다, got {type(x)}")
        
        return self.net(x)
    

class ActorCritic(nn.Module):
    """
    PPO용 Actor-Critic 네트워크 클래스
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        # 검증
        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim은 양수 값을 가져야 합니다, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim은 양수 값을 가져야 합니다, got {hidden_dim}")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.backbone = MLP(obs_dim, hidden_dim)

        # Actor
        self.actor_mean = nn.Linear(hidden_dim, action_dim) # 각 action dimension의 평균 mean

        # 각 action dim별 학습 가능한 log std
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5)) # 각 action dimension의 로그 표준편차

        # Critic
        self.critic = nn.Linear(hidden_dim, 1)

        # 안정화 기법
        self._init_heads()

    def _init_heads(self) -> None:
        """
        PPO에서 일반적으로 actor output은 작은 gain이기에,
        critic output은 1.0 gain을 사용. 이를 구현하는 함수
        """
        _init_layer(self.action_dim, gain=0.01)
        _init_layer(self.critic, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        전체 PPO Actor-Critic Forward Propagation 구현 함수
        """
        # 검증
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"obs는 torch.Tensor 타입이어야 합니다, got {type(obs)}")

        # obs가 (obs_dim,) 단일 상태로 들어오는 경우도 허용
        squeeze_batch = False
        if obs.dim() == 1:
            if obs.shape[0] != self.obs_dim:
                raise ValueError(
                    f"Single observation shape mismatch: expected ({self.obs_dim},), got {tuple(obs.shape)}"
                )
            obs = obs.unsqueeze(0)
            squeeze_batch = True
        elif obs.dim() == 2:
            if obs.shape[1] != self.obs_dim:
                raise ValueError(
                    f"Batched observation shape mismatch: expected second dim {self.obs_dim}, got {obs.shape[1]}"
                )
        else:
            raise ValueError(
                f"obs must have shape ({self.obs_dim},) or (batch, {self.obs_dim}), got {tuple(obs.shape)}"
            )

        backbone_out = self.backbone(obs)

        mean = self.actor_mean(backbone_out)

        # Global learnable log_std를 batch shape에 맞게 확장
        log_std = torch.clamp(self.actor_log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.unsqueeze(0).expand_as(mean)

        value = self.critic(backbone_out).squeeze(-1) # critic 출력 (batch, 1)인데 PPO 계산 편하게 (batch,)으로 바꿈

        # 배열 차원 일치
        if squeeze_batch:
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
            value = value.squeeze(0)

        return mean, log_std, value
    
    def get_dict(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        """
        정규분포 정책을 생성하는 함수
        """
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, mean, value