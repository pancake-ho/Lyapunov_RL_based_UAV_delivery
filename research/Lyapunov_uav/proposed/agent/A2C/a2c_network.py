from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _init_layer(layer: nn.Linear, gain: float) -> None:
    """
    orthogonal initialization 적용 함수
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """
    observation을 받아 actor와 critic이 공통으로 사용할 latent feature를 생성하는
    2-layer fully connected backbone
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim은 양수여야 합니다, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim은 양수여야 합니다, got {hidden_dim}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self._init_backbone()

    def _init_backbone(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                _init_layer(module, gain=torch.sqrt(torch.tensor(2.0)).item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x는 torch.Tensor여야 합니다, got {type(x)}")
        return self.net(x)


class ActorCritic(nn.Module):
    """
    A2C용 Actor-Critic 네트워크
    PPO와 동일한 구조를 사용
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()

        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수여야 합니다, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim은 양수여야 합니다, got {action_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim은 양수여야 합니다, got {hidden_dim}")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.backbone = MLP(obs_dim, hidden_dim)

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic = nn.Linear(hidden_dim, 1)

        self._init_heads()

    def _init_heads(self) -> None:
        _init_layer(self.actor_mean, gain=0.01)
        _init_layer(self.critic, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return:
            mean: (..., action_dim)
            log_std: (..., action_dim)
            value: (...,)
        """
        if not isinstance(obs, torch.Tensor):
            raise TypeError(f"obs는 torch.Tensor여야 합니다, got {type(obs)}")

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

        feat = self.backbone(obs)

        mean = self.actor_mean(feat)
        log_std = torch.clamp(self.actor_log_std, LOG_STD_MIN, LOG_STD_MAX)
        log_std = log_std.unsqueeze(0).expand_as(mean)

        value = self.critic(feat).squeeze(-1)

        if squeeze_batch:
            mean = mean.squeeze(0)
            log_std = log_std.squeeze(0)
            value = value.squeeze(0)

        return mean, log_std, value

    def get_dist_value(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist, mean, value