from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_MLP(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """
    각 head에 공통으로 둘 MLP Network 구조.

    현재 시나리오 기준:
        activation은 Tanh로 설정.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation())
        prev_dim = int(hidden_dim)

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> float:
    """
    학습 안정성 향상을 위해, hidden layer에 init 기능을 추가로 설정.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class FastActorCritic(nn.Module):
    """
    Fast-timescale 전용 Actor-Critic Network.

    현재 시나리오 기준:
        input:
            flattened fast_obs
        
        actor output:
            raw continuous fast actionector에 대한 Gaussian mean
        
        또한, env에 들어가는 action은 여기서 직접 만들지 않고
        FastActionCodec class가 decoding을 수행.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()

        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim은 양수 값을 가져야 합니다. 현재 값: {action_dim}")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor_network = _build_MLP(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=nn.Tanh,
        )

        self.critic_network = _build_MLP(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=nn.Tanh,
        )

        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(init_log_std), dtype=torch.float32)
        )

        self.apply(lambda module: _orthogonal_init(module, gain=1.0))

        # actor/critic network의 마지막 hidden layer쪽은 gain을 다르게 적용 (특히 actor)
        if isinstance(self.actor_network[-1], nn.Linear):
            _orthogonal_init(self.actor_network[-1], gain=0.01)
        if isinstance(self.critic_network[-1], nn.Linear):
            _orthogonal_init(self.critic_network[-1], gain=1.0)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """
        Input으로 주어지는 obs에 대하여 mean/std 계산 후(Actor 통과) Normal Distribution 반환.
        """
        mean = self.actor_network(obs)
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)
    
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Input으로 주어지는 obs에 대하여 Critic Network의 output (value) 반환.
        """
        return self.critic_network(obs).squeeze(-1)
    
    @torch.no_grad()
    def act(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input으로 주어지는 obs에 대하여 action, log_prob, value 반환.

        현재 시나리오 기준:
            action: shape (B, action_dim)
            log_prob: shape (B,)
            value: shape (B,)
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        dist = self._distribution(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value(obs)

        return action, log_prob, self.value
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO update를 위해, log_prob/entropy/value를 모두 반환.

        현재 시나리오 기준:
            log_prob: shape (B,)
            entropy: shape (B,)
            value: shape (B,)
        """
        dist = self._distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)

        return log_prob, entropy, value