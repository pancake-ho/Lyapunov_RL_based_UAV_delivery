from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .fast_action import FastActionSpec


def _build_body(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation())
        prev_dim = int(hidden_dim)

    return nn.Sequential(*layers), prev_dim


def _build_MLP(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    body, last_dim = _build_body(input_dim, hidden_dims)
    return nn.Sequential(
        *list(body.children()),
        nn.Linear(last_dim, int(output_dim)),
    )


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class FastActorCritic(nn.Module):
    """Conditional mixed categorical/Gaussian Fast-timescale actor-critic."""

    def __init__(
        self,
        obs_dim: int,
        action_spec: FastActionSpec,
        hidden_dims: Sequence[int] = (256, 256),
        init_log_std: float = -1.0,
    ) -> None:
        super().__init__()

        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}.")

        self.obs_dim = int(obs_dim)
        self.spec = action_spec
        self.action_dim = int(action_spec.action_dim)

        self.actor_backbone, actor_dim = _build_body(
            self.obs_dim,
            hidden_dims,
        )

        self.rsu_chunk_head = nn.Linear(
            actor_dim,
            self.spec.rsu_link_dim * self.spec.chunk_choices,
        )
        self.rsu_layer_head = nn.Linear(
            actor_dim,
            self.spec.rsu_link_dim * self.spec.layer_choices,
        )
        self.uav_chunk_head = nn.Linear(
            actor_dim,
            self.spec.uav_link_dim * self.spec.chunk_choices,
        )
        self.uav_layer_head = nn.Linear(
            actor_dim,
            self.spec.uav_link_dim * self.spec.layer_choices,
        )
        self.power_mean_head = nn.Linear(
            actor_dim,
            self.spec.uav_link_dim,
        )
        self.power_log_std = nn.Parameter(
            torch.full(
                (self.spec.uav_link_dim,),
                float(init_log_std),
                dtype=torch.float32,
            )
        )

        self.critic_network = _build_MLP(
            self.obs_dim,
            hidden_dims,
            1,
        )

        self.apply(lambda module: _orthogonal_init(module, gain=1.0))
        for head in (
            self.rsu_chunk_head,
            self.rsu_layer_head,
            self.uav_chunk_head,
            self.uav_layer_head,
            self.power_mean_head,
        ):
            _orthogonal_init(head, gain=0.01)

        if isinstance(self.critic_network[-1], nn.Linear):
            _orthogonal_init(self.critic_network[-1], gain=1.0)

    def _policy_distributions(
        self,
        obs: torch.Tensor,
    ) -> tuple[
        Categorical,
        Categorical,
        Categorical,
        Categorical,
        Normal,
    ]:
        feature = self.actor_backbone(obs)
        batch_size = int(feature.shape[0])

        rsu_chunk_logits = self.rsu_chunk_head(feature).view(
            batch_size,
            self.spec.rsu_link_dim,
            self.spec.chunk_choices,
        )
        rsu_layer_logits = self.rsu_layer_head(feature).view(
            batch_size,
            self.spec.rsu_link_dim,
            self.spec.layer_choices,
        )
        uav_chunk_logits = self.uav_chunk_head(feature).view(
            batch_size,
            self.spec.uav_link_dim,
            self.spec.chunk_choices,
        )
        uav_layer_logits = self.uav_layer_head(feature).view(
            batch_size,
            self.spec.uav_link_dim,
            self.spec.layer_choices,
        )

        power_mean = self.power_mean_head(feature)
        power_log_std = torch.clamp(
            self.power_log_std,
            min=-5.0,
            max=1.0,
        )
        power_std = torch.exp(power_log_std).expand_as(power_mean)

        return (
            Categorical(logits=rsu_chunk_logits),
            Categorical(logits=rsu_layer_logits),
            Categorical(logits=uav_chunk_logits),
            Categorical(logits=uav_layer_logits),
            Normal(power_mean, power_std),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_network(obs).squeeze(-1)

    def _split_vector(
        self,
        vector: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if vector.shape[-1] != self.action_dim:
            raise ValueError(
                "action last dim mismatch: "
                f"expected={self.action_dim}, got={vector.shape[-1]}"
            )

        s = self.spec
        return (
            vector[..., s.rsu_chunks_slice],
            vector[..., s.rsu_layers_slice],
            vector[..., s.uav_chunks_slice],
            vector[..., s.uav_layers_slice],
            vector[..., s.uav_power_slice],
        )

    def _prepare_mask(
        self,
        action_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if action_mask.shape != reference.shape:
            raise ValueError(
                "action_mask shape mismatch: "
                f"expected={tuple(reference.shape)}, "
                f"got={tuple(action_mask.shape)}"
            )

        mask = action_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        if torch.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("action_mask must contain only 0 or 1.")
        return mask

    @staticmethod
    def _masked_mean(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        count = mask.sum(dim=-1)
        result = (
            (value * mask).sum(dim=-1)
            / count.clamp_min(1.0)
        )
        return torch.where(
            count > 0.0,
            result,
            torch.zeros_like(result),
        )

    def _sample_components(
        self,
        obs: torch.Tensor,
        deterministic: bool,
    ) -> tuple[
        Categorical,
        Categorical,
        Categorical,
        Categorical,
        Normal,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            rsu_chunk_dist,
            rsu_layer_dist,
            uav_chunk_dist,
            uav_layer_dist,
            power_dist,
        ) = self._policy_distributions(obs)

        if deterministic:
            rsu_chunks = rsu_chunk_dist.logits.argmax(dim=-1)
            rsu_layer_class = rsu_layer_dist.logits.argmax(dim=-1)
            uav_chunks = uav_chunk_dist.logits.argmax(dim=-1)
            uav_layer_class = uav_layer_dist.logits.argmax(dim=-1)
            power_raw = power_dist.mean
        else:
            rsu_chunks = rsu_chunk_dist.sample()
            rsu_layer_class = rsu_layer_dist.sample()
            uav_chunks = uav_chunk_dist.sample()
            uav_layer_class = uav_layer_dist.sample()
            power_raw = power_dist.sample()

        return (
            rsu_chunk_dist,
            rsu_layer_dist,
            uav_chunk_dist,
            uav_layer_dist,
            power_dist,
            rsu_chunks,
            rsu_layer_class,
            uav_chunks,
            uav_layer_class,
            power_raw,
        )

    def _assemble_masked_action(
        self,
        *,
        rsu_chunks: torch.Tensor,
        rsu_layer_class: torch.Tensor,
        uav_chunks: torch.Tensor,
        uav_layer_class: torch.Tensor,
        power_raw: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        provisional = torch.cat(
            [
                rsu_chunks.float(),
                (rsu_layer_class + 1).float(),
                uav_chunks.float(),
                (uav_layer_class + 1).float(),
                power_raw,
            ],
            dim=-1,
        )

        if action_mask is None:
            base_mask = torch.ones_like(provisional)
        else:
            base_mask = self._prepare_mask(action_mask, provisional)

        (
            rsu_chunk_mask,
            rsu_layer_base_mask,
            uav_chunk_mask,
            uav_layer_base_mask,
            power_base_mask,
        ) = self._split_vector(base_mask)

        rsu_chunks = torch.where(
            rsu_chunk_mask > 0.0,
            rsu_chunks,
            torch.zeros_like(rsu_chunks),
        )
        uav_chunks = torch.where(
            uav_chunk_mask > 0.0,
            uav_chunks,
            torch.zeros_like(uav_chunks),
        )

        rsu_layer_mask = (
            rsu_layer_base_mask
            * (rsu_chunks > 0).to(rsu_layer_base_mask.dtype)
        )
        uav_layer_mask = (
            uav_layer_base_mask
            * (uav_chunks > 0).to(uav_layer_base_mask.dtype)
        )
        power_mask = (
            power_base_mask
            * (uav_chunks > 0).to(power_base_mask.dtype)
        )

        rsu_layers = torch.where(
            rsu_layer_mask > 0.0,
            rsu_layer_class + 1,
            torch.zeros_like(rsu_layer_class),
        )
        uav_layers = torch.where(
            uav_layer_mask > 0.0,
            uav_layer_class + 1,
            torch.zeros_like(uav_layer_class),
        )
        power_raw = torch.where(
            power_mask > 0.0,
            power_raw,
            torch.zeros_like(power_raw),
        )

        action = torch.cat(
            [
                rsu_chunks.float(),
                rsu_layers.float(),
                uav_chunks.float(),
                uav_layers.float(),
                power_raw,
            ],
            dim=-1,
        )
        effective_mask = torch.cat(
            [
                rsu_chunk_mask,
                rsu_layer_mask,
                uav_chunk_mask,
                uav_layer_mask,
                power_mask,
            ],
            dim=-1,
        )

        return (
            action,
            effective_mask,
            rsu_chunks,
            rsu_layer_class,
            uav_chunks,
            uav_layer_class,
            power_raw,
            rsu_layer_mask,
            uav_layer_mask,
        )

    @torch.inference_mode()
    def policy_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Slow-DPP candidate forecast 전용 actor-only batch inference.

        critic, log_prob, entropy를 계산하지 않아 candidate batch에서
        불필요한 GPU 연산과 GPU->CPU synchronization을 줄인다.
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        sampled = self._sample_components(obs, deterministic)
        (
            _,
            _,
            _,
            _,
            _,
            rsu_chunks,
            rsu_layer_class,
            uav_chunks,
            uav_layer_class,
            power_raw,
        ) = sampled

        action, effective_mask, *_ = self._assemble_masked_action(
            rsu_chunks=rsu_chunks,
            rsu_layer_class=rsu_layer_class,
            uav_chunks=uav_chunks,
            uav_layer_class=uav_layer_class,
            power_raw=power_raw,
            action_mask=action_mask,
        )
        return action, effective_mask

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        sampled = self._sample_components(obs, deterministic)
        (
            rsu_chunk_dist,
            rsu_layer_dist,
            uav_chunk_dist,
            uav_layer_dist,
            power_dist,
            rsu_chunks,
            rsu_layer_class,
            uav_chunks,
            uav_layer_class,
            power_raw,
        ) = sampled

        (
            action,
            effective_mask,
            rsu_chunks,
            rsu_layer_class,
            uav_chunks,
            uav_layer_class,
            power_raw,
            rsu_layer_mask,
            uav_layer_mask,
        ) = self._assemble_masked_action(
            rsu_chunks=rsu_chunks,
            rsu_layer_class=rsu_layer_class,
            uav_chunks=uav_chunks,
            uav_layer_class=uav_layer_class,
            power_raw=power_raw,
            action_mask=action_mask,
        )

        (
            rsu_chunk_mask,
            _,
            uav_chunk_mask,
            _,
            power_mask,
        ) = self._split_vector(effective_mask)

        log_prob = (
            (rsu_chunk_dist.log_prob(rsu_chunks) * rsu_chunk_mask).sum(dim=-1)
            + (
                rsu_layer_dist.log_prob(rsu_layer_class)
                * rsu_layer_mask
            ).sum(dim=-1)
            + (uav_chunk_dist.log_prob(uav_chunks) * uav_chunk_mask).sum(dim=-1)
            + (
                uav_layer_dist.log_prob(uav_layer_class)
                * uav_layer_mask
            ).sum(dim=-1)
            + (power_dist.log_prob(power_raw) * power_mask).sum(dim=-1)
        )

        return action, log_prob, self.value(obs), effective_mask

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            rsu_chunk_dist,
            rsu_layer_dist,
            uav_chunk_dist,
            uav_layer_dist,
            power_dist,
        ) = self._policy_distributions(obs)

        if action_mask is None:
            mask = torch.ones_like(actions)
        else:
            mask = self._prepare_mask(action_mask, actions)

        (
            rsu_chunks_float,
            rsu_layers_float,
            uav_chunks_float,
            uav_layers_float,
            power_raw,
        ) = self._split_vector(actions)
        (
            rsu_chunk_mask,
            rsu_layer_mask,
            uav_chunk_mask,
            uav_layer_mask,
            power_mask,
        ) = self._split_vector(mask)

        rsu_chunks = rsu_chunks_float.long().clamp(
            0,
            self.spec.max_chunk,
        )
        uav_chunks = uav_chunks_float.long().clamp(
            0,
            self.spec.max_chunk,
        )
        rsu_layer_class = (rsu_layers_float.long() - 1).clamp(
            0,
            self.spec.max_layer - 1,
        )
        uav_layer_class = (uav_layers_float.long() - 1).clamp(
            0,
            self.spec.max_layer - 1,
        )

        log_prob = (
            (rsu_chunk_dist.log_prob(rsu_chunks) * rsu_chunk_mask).sum(dim=-1)
            + (
                rsu_layer_dist.log_prob(rsu_layer_class)
                * rsu_layer_mask
            ).sum(dim=-1)
            + (uav_chunk_dist.log_prob(uav_chunks) * uav_chunk_mask).sum(dim=-1)
            + (
                uav_layer_dist.log_prob(uav_layer_class)
                * uav_layer_mask
            ).sum(dim=-1)
            + (power_dist.log_prob(power_raw) * power_mask).sum(dim=-1)
        )

        categorical_entropy_sum = (
            (rsu_chunk_dist.entropy() * rsu_chunk_mask).sum(dim=-1)
            + (rsu_layer_dist.entropy() * rsu_layer_mask).sum(dim=-1)
            + (uav_chunk_dist.entropy() * uav_chunk_mask).sum(dim=-1)
            + (uav_layer_dist.entropy() * uav_layer_mask).sum(dim=-1)
        )
        categorical_count = (
            rsu_chunk_mask.sum(dim=-1)
            + rsu_layer_mask.sum(dim=-1)
            + uav_chunk_mask.sum(dim=-1)
            + uav_layer_mask.sum(dim=-1)
        )
        categorical_entropy = (
            categorical_entropy_sum
            / categorical_count.clamp_min(1.0)
        )
        categorical_entropy = torch.where(
            categorical_count > 0.0,
            categorical_entropy,
            torch.zeros_like(categorical_entropy),
        )

        power_entropy = self._masked_mean(
            power_dist.entropy(),
            power_mask,
        )

        return (
            log_prob,
            categorical_entropy,
            power_entropy,
            self.value(obs),
        )
