from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli


def _build_MLP(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """
    actor와 critic에 공통으로 둘 MLP Network 구조.
    Fast-timescale과 MLP 핵심 구조는 동일하게 가져감.

    현재 시나리오 기준:
        activation Tanh로 설정.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation())
        prev_dim = int(hidden_dim)

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """
    학습 안정성 향상을 위해, hidden layer에 init 기능을 추가로 설정.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class SlowActorCritic(nn.Module):
    """
    Slow-timescale 전용 Bernoulli Actor-Critic Network.

    현재 시나리오 기준:
        state:
            s_H(r) = [Z(r), B(r), user_region(r)]
        
        action:
            a_H(r) = [y_mn(r), mu_u(r), phi_un(r)]
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: type[nn.Module] = nn.Tanh,
        init_action_logit: float = 0.0,
        min_logit: float = -20.0,
        max_logit: float = 20.0
    ) -> None:
        super().__init__()

        if int(obs_dim) <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        if int(action_dim) <= 0:
            raise ValueError(f"action_dim은 양수 값을 가져야 합니다. 현재 값: {action_dim}")
        if float(min_logit) >= float(max_logit):
            raise ValueError(
                f"min_logit은 max_logit보다 작은 값을 가져야 합니다, "
                f"현재 값: min_logit={min_logit}, max_logit={max_logit}"
            )
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.min_logit = min_logit
        self.max_logit = max_logit

        self.actor_network = _build_MLP(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=activation,
        )

        self.critic_network = _build_MLP(
            input_dim=self.obs_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
        )

        self.apply(lambda module: _orthogonal_init(module, gain=math.sqrt(2.0)))

        # actor/critic network의 마지막 hidden layer쪽은 gain을 다르게 적용 (특히 actor)
        # 또한 actor network의 마지막 hidden layer의 bias에 initialization 적용
        if isinstance(self.actor_network[-1], nn.Linear):
            _orthogonal_init(self.actor_network[-1], gain=0.01)
            nn.init.constant_(self.actor_network[-1].bias, init_action_logit)

        if isinstance(self.critic_network[-1], nn.Linear):
            _orthogonal_init(self.critic_network[-1], gain=1.0)
    
    def _ensure_obs_batch(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            raise TypeError(f"obs must be torch.Tensor, got {type(obs)}")

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if obs.ndim != 2:
            raise ValueError(
                f"obs must have shape (B, obs_dim), got {tuple(obs.shape)}"
            )

        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: expected {self.obs_dim}, got {obs.shape[-1]}"
            )

        if not torch.is_floating_point(obs):
            obs = obs.float()

        return obs

    def _ensure_action_batch(
        self,
        actions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(actions):
            raise TypeError(f"actions must be torch.Tensor, got {type(actions)}")

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        if actions.ndim != 2:
            raise ValueError(
                f"actions must have shape (B, action_dim), got {tuple(actions.shape)}"
            )

        if actions.shape[0] != int(batch_size):
            raise ValueError(
                f"action batch mismatch: expected batch={batch_size}, "
                f"got {actions.shape[0]}"
            )

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action_dim mismatch: expected {self.action_dim}, "
                f"got {actions.shape[-1]}"
            )

        if not torch.is_floating_point(actions):
            actions = actions.float()

        # PPO buffer should store binary Bernoulli samples.
        # Small numerical deviations are safely mapped back to {0, 1}.
        return (actions >= 0.5).float()

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return Bernoulli logits with shape (B, action_dim).
        """
        obs = self._ensure_obs_batch(obs)
        logits = self.actor_network(obs)
        logits = torch.clamp(logits, min=self.min_logit, max=self.max_logit)
        return logits

    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return Bernoulli probabilities with shape (B, action_dim).
        """
        return torch.sigmoid(self.action_logits(obs))

    def distribution(self, obs: torch.Tensor) -> Bernoulli:
        """
        Return independent Bernoulli distributions for each slow action bit.

        The log-probability is summed over action_dim in act()/evaluate_actions().
        """
        logits = self.action_logits(obs)
        return Bernoulli(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Return critic value with shape (B,).
        """
        obs = self._ensure_obs_batch(obs)
        return self.critic_network(obs).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select a binary slow action.

        Args:
            obs:
                Tensor with shape (obs_dim,) or (B, obs_dim).
            deterministic:
                If False, sample Bernoulli action.
                If True, use probability threshold 0.5.

        Returns:
            action:
                Binary float tensor, shape (B, action_dim).
            log_prob:
                Sum of bit-wise log probabilities, shape (B,).
            value:
                Critic value, shape (B,).
        """
        obs = self._ensure_obs_batch(obs)
        dist = self.distribution(obs)

        if deterministic:
            probs = dist.probs
            action = (probs >= 0.5).float()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value(obs)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate old binary slow actions for PPO update.

        Args:
            obs:
                Tensor with shape (B, obs_dim).
            actions:
                Binary action tensor with shape (B, action_dim).

        Returns:
            log_prob:
                Current log probability of the given actions, shape (B,).
            entropy:
                Sum of bit-wise Bernoulli entropies, shape (B,).
            value:
                Current critic value, shape (B,).
        """
        obs = self._ensure_obs_batch(obs)
        actions = self._ensure_action_batch(
            actions,
            batch_size=obs.shape[0],
        )

        dist = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)

        return log_prob, entropy, value

    @torch.no_grad()
    def action_summary(self, obs: torch.Tensor) -> Dict[str, float]:
        """
        Return lightweight statistics for logging/debugging.

        This function does not change policy behavior and is not required for PPO.
        """
        probs = self.action_probs(obs)

        return {
            "prob_mean": float(probs.mean().detach().cpu().item()),
            "prob_std": float(probs.std(unbiased=False).detach().cpu().item()),
            "prob_min": float(probs.min().detach().cpu().item()),
            "prob_max": float(probs.max().detach().cpu().item()),
        }