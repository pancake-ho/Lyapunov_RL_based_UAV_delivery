from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .slow_action import SlowActionSpec


def _build_body(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> Tuple[nn.Sequential, int]:
    """
    actor와 critic에 공통으로 둘 MLP Network Body 구조.
    Fast-timescale과 MLP 핵심 구조는 동일하게 가져감.

    현재 시나리오 기준:
        activation Tanh로 설정.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        width = int(hidden_dim)
        layers.extend([nn.Linear(prev_dim, width), activation()])
        prev_dim = width

    return nn.Sequential(*layers), prev_dim


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
        action_spec: SlowActionSpec,
        hidden_dims: Sequence[int] = (256, 256),
        rsu_init_logit: float = 0.0,
        hiring_init_logit: float = 0.0,
        uav_init_logit: float = 0.0,
        min_logit: float = -20.0,
        max_logit: float = 20.0
    ) -> None:
        super().__init__()

        if int(obs_dim) <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        if float(min_logit) >= float(max_logit):
            raise ValueError(
                f"min_logit은 max_logit보다 작은 값을 가져야 합니다, "
                f"현재 값: min_logit={min_logit}, max_logit={max_logit}"
            )
        
        self.obs_dim = obs_dim
        self.spec = action_spec
        self.action_dim = action_spec.action_dim
        self.min_logit = min_logit
        self.max_logit = max_logit

        # actor & head
        self.actor_backbone, actor_dim = _build_body(self.obs_dim, hidden_dims)
        self.rsu_head = nn.Linear(actor_dim, self.spec.rsu_dim)
        self.hiring_head = nn.Linear(actor_dim + self.spec.num_user, self.spec.hiring_dim)
        self.uav_head = nn.Linear(actor_dim + self.spec.num_user + self.spec.hiring_dim, self.spec.uav_dim)

        # critic
        critic_body, critic_dim = _build_body(self.obs_dim, hidden_dims)
        self.critic_network = nn.Sequential(*list(critic_body.children()), nn.Linear(critic_dim, 1))

        self.apply(lambda module: _orthogonal_init(module, gain=math.sqrt(2.0)))

        for head, bias in (
            (self.rsu_head, rsu_init_logit),
            (self.hiring_head, hiring_init_logit),
            (self.uav_head, uav_init_logit),
        ):
            _orthogonal_init(head, gain=0.01)
            nn.init.constant_(head.bias, float(bias))
        if isinstance(self.critic_network[-1], nn.Linear):
            _orthogonal_init(self.critic_network[-1], gain=1.0)

    def _ensure_obs_batch(self, obs: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(obs):
            raise TypeError(f"obs must be torch.Tensor, got {type(obs)}")

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs shape must be (B, {self.obs_dim}), got {tuple(obs.shape)}"
            )
        return obs

    def _ensure_action_batch(
        self, action: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape != (batch_size, self.action_dim):
            raise ValueError(
                "action shape mismatch: "
                f"expected={(batch_size, self.action_dim)}, got={tuple(action.shape)}"
            )
        return action.float()

    def _split(
        self, vector: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if vector.shape[-1] != self.action_dim:
            raise ValueError("slow vector last dimension mismatch.")
        return (
            vector[..., self.spec.rsu_slice],
            vector[..., self.spec.hiring_slice],
            vector[..., self.spec.uav_slice],
        )

    def _merge(
        self,
        rsu: torch.Tensor,
        hiring: torch.Tensor,
        uav: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([hiring, rsu, uav], dim=-1)

    def _prepare_mask(
        self, action_mask: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        if action_mask.shape != reference.shape:
            raise ValueError(
                "slow action_mask shape mismatch: "
                f"expected={tuple(reference.shape)}, got={tuple(action_mask.shape)}"
            )
        mask = action_mask.to(device=reference.device, dtype=reference.dtype)
        if torch.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("slow action_mask must contain only 0 or 1.")
        return mask

    def _clamp_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.clamp(logits, self.min_logit, self.max_logit)

    def _residual_users(self, rsu_action: torch.Tensor) -> torch.Tensor:
        rsu_matrix = rsu_action.view(
            -1, self.spec.num_rsu, self.spec.num_user
        )
        return (rsu_matrix.sum(dim=1) == 0).to(rsu_action.dtype)

    def _rsu_distribution(self, feature: torch.Tensor) -> Bernoulli:
        return Bernoulli(logits=self._clamp_logits(self.rsu_head(feature)))

    def _hiring_distribution(
        self, feature: torch.Tensor, rsu_action: torch.Tensor
    ) -> Bernoulli:
        residual = self._residual_users(rsu_action)
        actor_input = torch.cat([feature, residual], dim=-1)
        return Bernoulli(
            logits=self._clamp_logits(self.hiring_head(actor_input))
        )

    def _uav_distribution(
        self,
        feature: torch.Tensor,
        rsu_action: torch.Tensor,
        hiring_action: torch.Tensor,
    ) -> Bernoulli:
        residual = self._residual_users(rsu_action)
        actor_input = torch.cat(
            [feature, residual, hiring_action], dim=-1
        )
        return Bernoulli(logits=self._clamp_logits(self.uav_head(actor_input)))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._ensure_obs_batch(obs)
        return self.critic_network(obs).squeeze(-1)

    def conditional_action_mask(
        self,
        prefix_action: torch.Tensor,
        static_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Add hiring and residual-user conditions to the static mask."""
        batch_size = 1 if prefix_action.ndim == 1 else prefix_action.shape[0]
        prefix_action = self._ensure_action_batch(
            prefix_action, batch_size=batch_size
        )
        static_mask = self._prepare_mask(static_action_mask, prefix_action)
        rsu_action, hiring_action, _ = self._split(prefix_action)
        rsu_mask, hiring_mask, uav_static_mask = self._split(static_mask)

        rsu_action = rsu_action * rsu_mask
        hiring_action = hiring_action * hiring_mask
        residual = self._residual_users(rsu_action)
        hiring_vector = hiring_action.view(-1, self.spec.num_uav, 1)
        uav_mask = uav_static_mask.view(
            -1, self.spec.num_uav, self.spec.num_user
        )
        uav_mask = uav_mask * hiring_vector * residual.unsqueeze(1)
        return self._merge(
            rsu_mask,
            hiring_mask,
            uav_mask.reshape(-1, self.spec.uav_dim),
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        count = mask.sum(dim=-1)
        mean = (value * mask).sum(dim=-1) / count.clamp_min(1.0)
        return torch.where(count > 0.0, mean, torch.zeros_like(mean))

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        *,
        static_action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = self._ensure_obs_batch(obs)
        feature = self.actor_backbone(obs)
        reference = torch.zeros(
            (obs.shape[0], self.action_dim),
            dtype=obs.dtype,
            device=obs.device,
        )
        static_mask = self._prepare_mask(static_action_mask, reference)
        rsu_mask, hiring_mask, _ = self._split(static_mask)

        rsu_dist = self._rsu_distribution(feature)
        rsu_action = (
            (rsu_dist.probs >= 0.5).float()
            if deterministic
            else rsu_dist.sample()
        ) * rsu_mask

        hiring_dist = self._hiring_distribution(feature, rsu_action)
        hiring_action = (
            (hiring_dist.probs >= 0.5).float()
            if deterministic
            else hiring_dist.sample()
        ) * hiring_mask

        prefix_action = self._merge(
            rsu_action,
            hiring_action,
            torch.zeros(
                (obs.shape[0], self.spec.uav_dim),
                dtype=obs.dtype,
                device=obs.device,
            ),
        )
        effective_mask = self.conditional_action_mask(
            prefix_action, static_mask
        )
        _, _, uav_mask = self._split(effective_mask)

        uav_dist = self._uav_distribution(
            feature, rsu_action, hiring_action
        )
        uav_action = (
            (uav_dist.probs >= 0.5).float()
            if deterministic
            else uav_dist.sample()
        ) * uav_mask

        action = self._merge(rsu_action, hiring_action, uav_action)
        log_prob = (
            (rsu_dist.log_prob(rsu_action) * rsu_mask).sum(dim=-1)
            + (hiring_dist.log_prob(hiring_action) * hiring_mask).sum(dim=-1)
            + (uav_dist.log_prob(uav_action) * uav_mask).sum(dim=-1)
        )
        return action, log_prob, self.value(obs), effective_mask

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        obs = self._ensure_obs_batch(obs)
        actions = self._ensure_action_batch(actions, obs.shape[0])
        mask = self._prepare_mask(action_mask, actions)
        if torch.any((actions != 0.0) & (actions != 1.0)):
            raise ValueError("stored slow actions must be binary.")
        if torch.any(actions * (1.0 - mask) != 0.0):
            raise ValueError("stored slow action has active bits outside its mask.")

        rsu_action, hiring_action, uav_action = self._split(actions)
        rsu_mask, hiring_mask, uav_mask = self._split(mask)
        feature = self.actor_backbone(obs)
        rsu_dist = self._rsu_distribution(feature)
        hiring_dist = self._hiring_distribution(feature, rsu_action)
        uav_dist = self._uav_distribution(
            feature, rsu_action, hiring_action
        )

        log_prob = (
            (rsu_dist.log_prob(rsu_action) * rsu_mask).sum(dim=-1)
            + (hiring_dist.log_prob(hiring_action) * hiring_mask).sum(dim=-1)
            + (uav_dist.log_prob(uav_action) * uav_mask).sum(dim=-1)
        )
        rsu_bit_entropy = rsu_dist.entropy()
        hiring_bit_entropy = hiring_dist.entropy()
        uav_bit_entropy = uav_dist.entropy()
        bit_entropy = self._merge(
            rsu_bit_entropy, hiring_bit_entropy, uav_bit_entropy
        )
        return (
            log_prob,
            self._masked_mean(bit_entropy, mask),
            self.value(obs),
            self._masked_mean(rsu_bit_entropy, rsu_mask),
            self._masked_mean(hiring_bit_entropy, hiring_mask),
            self._masked_mean(uav_bit_entropy, uav_mask),
        )

    @staticmethod
    def _masked_probability_mean(
        probability: torch.Tensor, mask: torch.Tensor
    ) -> float:
        count = mask.sum()
        if float(count.detach().cpu().item()) <= 0.0:
            return 0.0
        value = (probability * mask).sum() / count
        return float(value.detach().cpu().item())

    @torch.no_grad()
    def action_summary(
        self, obs: torch.Tensor, static_action_mask: torch.Tensor
    ) -> Dict[str, float]:
        obs = self._ensure_obs_batch(obs)
        feature = self.actor_backbone(obs)
        reference = torch.zeros(
            (obs.shape[0], self.action_dim),
            dtype=obs.dtype,
            device=obs.device,
        )
        static_mask = self._prepare_mask(static_action_mask, reference)
        rsu_mask, hiring_mask, _ = self._split(static_mask)

        rsu_prob = self._rsu_distribution(feature).probs
        rsu_action = (rsu_prob >= 0.5).float() * rsu_mask
        hiring_prob = self._hiring_distribution(feature, rsu_action).probs
        hiring_action = (hiring_prob >= 0.5).float() * hiring_mask
        prefix = self._merge(
            rsu_action,
            hiring_action,
            torch.zeros(
                (obs.shape[0], self.spec.uav_dim),
                dtype=obs.dtype,
                device=obs.device,
            ),
        )
        effective_mask = self.conditional_action_mask(prefix, static_mask)
        _, _, uav_mask = self._split(effective_mask)
        uav_prob = self._uav_distribution(
            feature, rsu_action, hiring_action
        ).probs
        probabilities = self._merge(rsu_prob, hiring_prob, uav_prob)
        return {
            "prob_mean": self._masked_probability_mean(
                probabilities, effective_mask
            ),
            "rsu_prob_mean": self._masked_probability_mean(
                rsu_prob, rsu_mask
            ),
            "hiring_prob_mean": self._masked_probability_mean(
                hiring_prob, hiring_mask
            ),
            "uav_prob_mean": self._masked_probability_mean(
                uav_prob, uav_mask
            ),
            "active_action_dims": float(
                effective_mask.sum(dim=-1).mean().cpu().item()
            ),
            "active_action_ratio": float(effective_mask.mean().cpu().item()),
        }
