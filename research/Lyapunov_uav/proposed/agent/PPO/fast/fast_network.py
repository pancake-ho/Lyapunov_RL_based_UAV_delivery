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
    """
    각 head에 공통으로 둘 MLP Network body 구조.

    현재 시나리오 기준:
        activation은 Tanh로 설정.
    """
    layers: list[nn.Module] = []
    prev_dim = input_dim

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
    """
    각 head에 공통으로 둘 MLP Network 구조.

    현재 시나리오 기준:
        activation은 Tanh로 설정.
    """
    body, last_dim = _build_body(input_dim, hidden_dims)

    return nn.Sequential(
        *list(body.children()),
        nn.Linear(
            last_dim,
            output_dim,
        ),
    )


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """
    학습 안정성 향상을 위해, hidden layer에 init 기능을 추가로 설정.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class FastActorCritic(nn.Module):
    """
    Fast-timescale 전용 Conditional mixed-action Actor-Critic Network.

    현재 시나리오 기준:
        l(t):
            Categorical {0, ..., L}

        k(t) | l(t)>0:
            Categorical {1, ..., K}

        p(t) | l_u(t)>0:
            Gaussian latent
    """
    def __init__(
        self,
        obs_dim: int,
        action_spec: FastActionSpec,
        hidden_dims: Sequence[int] = (256, 256),
        init_log_std: float = -1.0,
    ) -> None:
        super().__init__()

        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        
        self.obs_dim = obs_dim
        self.spec = action_spec
        self.action_dim = int(action_spec.action_dim)

        self.actor_backbone, actor_dim = _build_body(self.obs_dim, hidden_dims)

        # actor heads
        self.rsu_chunk_head = nn.Linear(actor_dim, (self.spec.rsu_link_dim * self.spec.chunk_choices))
        self.rsu_layer_head = nn.Linear(actor_dim, (self.spec.rsu_link_dim * self.spec.layer_choices))

        self.uav_chunk_head = nn.Linear(actor_dim, (self.spec.uav_link_dim * self.spec.chunk_choices))
        self.uav_layer_head = nn.Linear(actor_dim, (self.spec.uav_link_dim * self.spec.layer_choices))

        self.power_mean_head = nn.Linear(actor_dim, self.spec.uav_link_dim)
        self.power_log_std = nn.Parameter(
            torch.tensor(
                float(init_log_std),
                dtype=torch.float32,
            )
        )

        # network
        self.critic_network = _build_MLP(
            self.obs_dim,
            hidden_dims,
            1,
        )
        self.apply(lambda module: _orthogonal_init(module, gain=1.0))

        # actor/critic network의 마지막 hidden layer쪽은 gain을 다르게 적용 (특히 actor)
        for head in (
            self.rsu_chunk_head,
            self.rsu_layer_head,
            self.uav_chunk_head,
            self.uav_layer_head,
            self.power_mean_head,
        ):
            _orthogonal_init(
                head,
                gain=0.01,
            )

        if isinstance(self.critic_network[-1], nn.Linear):
            _orthogonal_init(self.critic_network[-1], gain=1.0)

    def _policy_distributions(self, obs: torch.Tensor) -> tuple[Categorical, Categorical, Categorical, Categorical, Normal]:
        """
        Input으로 주어지는 obs에 대하여 mean/std 계산 후(Actor 통과) Policy Distribution 반환.
        """
        feature = self.actor_backbone(obs)
        batch_size = feature.shape[0]

        rsu_chunk_logits = self.rsu_chunk_head(feature).view(batch_size, self.spec.rsu_link_dim, self.spec.chunk_choices)
        rsu_layer_logits = self.rsu_layer_head(feature).view(batch_size, self.spec.rsu_link_dim, self.spec.layer_choices)

        uav_chunk_logits = self.uav_chunk_head(feature).view(batch_size, self.spec.uav_link_dim, self.spec.chunk_choices)
        uav_layer_logits = self.uav_layer_head(feature).view(batch_size, self.spec.uav_link_dim, self.spec.layer_choices)

        power_mean = self.power_mean_head(feature)
        power_log_std = torch.clamp(self.power_log_std, min=-5.0, max=1.0)
        power_std = torch.exp(power_log_std).expand_as(power_mean)

        return (
            Categorical(logits=rsu_chunk_logits),
            Categorical(logits=rsu_layer_logits),
            Categorical(logits=uav_chunk_logits),
            Categorical(logits=uav_layer_logits),
            Normal(power_mean, power_std),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Input으로 주어지는 obs에 대하여 Critic Network의 output (value) 반환.
        """
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
        if (
            vector.shape[-1]
            != self.action_dim
        ):
            raise ValueError(
                "action last dim mismatch: "
                f"expected={self.action_dim}, "
                f"got={vector.shape[-1]}"
            )

        s = self.spec

        return (
            vector[
                ...,
                s.rsu_chunks_slice,
            ],
            vector[
                ...,
                s.rsu_layers_slice,
            ],
            vector[
                ...,
                s.uav_chunks_slice,
            ],
            vector[
                ...,
                s.uav_layers_slice,
            ],
            vector[
                ...,
                s.uav_power_slice,
            ],
        )

    def _prepare_mask(
        self,
        action_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            action_mask.shape
            != reference.shape
        ):
            raise ValueError(
                "action_mask shape mismatch: "
                f"expected={tuple(reference.shape)}, "
                f"got={tuple(action_mask.shape)}"
            )

        mask = action_mask.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if torch.any(
            (mask != 0.0)
            & (mask != 1.0)
        ):
            raise ValueError(
                "action_mask는 0 또는 1이어야 합니다."
            )

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

    @staticmethod
    def _sample_common_categorical(
        distribution: Categorical,
    ) -> torch.Tensor:
        """
        Sample one common uniform variate per categorical variable.

        Every batch row keeps its own policy probabilities, while the
        underlying uniform variate is shared across rows. This is the
        vectorized equivalent of evaluating Slow-DPP candidates with common
        random numbers.
        """
        logits = distribution.logits
        if logits.ndim < 2:
            raise ValueError(
                "Categorical logits must include batch and class dims."
            )

        probabilities = torch.softmax(logits, dim=-1)
        cdf = probabilities.cumsum(dim=-1)
        uniform_shape = (1, *logits.shape[1:-1])
        uniform = torch.rand(
            uniform_shape,
            device=logits.device,
            dtype=logits.dtype,
        )
        sample = (
            uniform.unsqueeze(-1)
            > cdf
        ).sum(dim=-1)
        return sample.clamp_max(logits.shape[-1] - 1)

    @staticmethod
    def _sample_common_normal(
        distribution: Normal,
    ) -> torch.Tensor:
        """
        Reparameterized Normal sample with common noise across batch rows.
        """
        mean = distribution.mean
        noise = torch.randn(
            (1, *mean.shape[1:]),
            device=mean.device,
            dtype=mean.dtype,
        )
        return mean + distribution.stddev * noise

    @torch.inference_mode()
    def sample_policy_actions(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
        common_random_across_batch: bool = False,
    ) -> torch.Tensor:
        """
        Forecast-only policy sampling without critic/log-prob computation.

        The realized joint-training path continues to use :meth:`act`.
        Slow-DPP forecasts need only environment actions, so omitting critic,
        log-probability, entropy, and statistics removes unnecessary GPU
        work and synchronization.
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

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
        elif common_random_across_batch:
            rsu_chunks = self._sample_common_categorical(
                rsu_chunk_dist
            )
            rsu_layer_class = self._sample_common_categorical(
                rsu_layer_dist
            )
            uav_chunks = self._sample_common_categorical(
                uav_chunk_dist
            )
            uav_layer_class = self._sample_common_categorical(
                uav_layer_dist
            )
            power_raw = self._sample_common_normal(power_dist)
        else:
            rsu_chunks = rsu_chunk_dist.sample()
            rsu_layer_class = rsu_layer_dist.sample()
            uav_chunks = uav_chunk_dist.sample()
            uav_layer_class = uav_layer_dist.sample()
            power_raw = power_dist.sample()

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
            base_mask = self._prepare_mask(
                action_mask,
                provisional,
            )

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

        return torch.cat(
            [
                rsu_chunks.float(),
                rsu_layers.float(),
                uav_chunks.float(),
                uav_layers.float(),
                power_raw,
            ],
            dim=-1,
        )
    
    @torch.no_grad()
    def act(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input으로 주어지는 obs에 대하여 action, log_prob, value 반환.

        현재 시나리오 기준:
            action: shape (B, action_dim)
            log_prob: shape (B,)
            value: shape (B,)
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        rsu_chunk_dist, rsu_layer_dist, uav_chunk_dist, uav_layer_dist, power_dist = self._policy_distributions(obs)

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
        
        rsu_chunk_mask, rsu_layer_base_mask, uav_chunk_mask, uav_layer_base_mask, power_base_mask = self._split_vector(base_mask)

        rsu_chunks = torch.where(rsu_chunk_mask > 0.0, rsu_chunks, torch.zeros_like(rsu_chunks))
        uav_chunks = torch.where(uav_chunk_mask > 0.0, uav_chunks, torch.zeros_like(uav_chunks))
        rsu_layer_mask = rsu_layer_base_mask * (rsu_chunks > 0).to(rsu_layer_base_mask.dtype)
        uav_layer_mask = uav_layer_base_mask * (uav_chunks > 0).to(uav_layer_base_mask.dtype)
        power_mask = power_base_mask * (uav_chunks > 0).to(power_base_mask.dtype)

        rsu_layers = torch.where(rsu_layer_mask > 0.0, rsu_layer_class + 1, torch.zeros_like(rsu_layer_class))
        uav_layers = torch.where(uav_layer_mask > 0.0, uav_layer_class + 1, torch.zeros_like(uav_layer_class))
        power_raw = torch.where(power_mask > 0.0, power_raw, torch.zeros_like(power_raw))

        action = torch.cat(
            [
                rsu_chunks.float(), rsu_layers.float(),
                uav_chunks.float(), uav_layers.float(),
                power_raw,
            ],
            dim=-1,
        )
        eff_mask = torch.cat(
            [
                rsu_chunk_mask, rsu_layer_mask,
                uav_chunk_mask, uav_layer_mask,
                power_mask,
            ],
            dim=-1,
        )

        log_prob = (
            (
                rsu_chunk_dist.log_prob(
                    rsu_chunks
                )
                * rsu_chunk_mask
            ).sum(dim=-1)

            + (
                rsu_layer_dist.log_prob(
                    rsu_layer_class
                )
                * rsu_layer_mask
            ).sum(dim=-1)

            + (
                uav_chunk_dist.log_prob(
                    uav_chunks
                )
                * uav_chunk_mask
            ).sum(dim=-1)

            + (
                uav_layer_dist.log_prob(
                    uav_layer_class
                )
                * uav_layer_mask
            ).sum(dim=-1)

            + (
                power_dist.log_prob(
                    power_raw
                )
                * power_mask
            ).sum(dim=-1)
        )

        return (
            action,
            log_prob,
            self.value(obs),
            eff_mask,
        )
    
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
        rsu_chunk_dist, rsu_layer_dist, uav_chunk_dist, uav_layer_dist, power_dist = self._policy_distributions(obs)

        if action_mask is None:
            mask = torch.ones_like(actions)
        else:
            mask = self._prepare_mask(action_mask, actions)
        
        rsu_chunks_float, rsu_layers_float, uav_chunks_float, uav_layers_float, power_raw = self._split_vector(actions)
        rsu_chunk_mask, rsu_layer_mask, uav_chunk_mask, uav_layer_mask, power_mask = self._split_vector(mask)

        rsu_chunks = rsu_chunks_float.long().clamp(0, self.spec.max_chunk)
        uav_chunks = uav_chunks_float.long().clamp(0, self.spec.max_chunk)

        # inactive layer의 지정값은 0이며, mask=0이므로 loss에서 제거 처리
        rsu_layer_class = (rsu_layers_float.long() - 1).clamp(0, self.spec.max_layer - 1)
        uav_layer_class = (uav_layers_float.long() - 1).clamp(0, self.spec.max_layer - 1)

        log_prob = (
            (
                rsu_chunk_dist.log_prob(
                    rsu_chunks
                )
                * rsu_chunk_mask
            ).sum(dim=-1)

            + (
                rsu_layer_dist.log_prob(
                    rsu_layer_class
                )
                * rsu_layer_mask
            ).sum(dim=-1)

            + (
                uav_chunk_dist.log_prob(
                    uav_chunks
                )
                * uav_chunk_mask
            ).sum(dim=-1)

            + (
                uav_layer_dist.log_prob(
                    uav_layer_class
                )
                * uav_layer_mask
            ).sum(dim=-1)

            + (
                power_dist.log_prob(
                    power_raw
                )
                * power_mask
            ).sum(dim=-1)
        )

        categorical_entropy_sum = (
            (
                rsu_chunk_dist.entropy()
                * rsu_chunk_mask
            ).sum(dim=-1)

            + (
                rsu_layer_dist.entropy()
                * rsu_layer_mask
            ).sum(dim=-1)

            + (
                uav_chunk_dist.entropy()
                * uav_chunk_mask
            ).sum(dim=-1)

            + (
                uav_layer_dist.entropy()
                * uav_layer_mask
            ).sum(dim=-1)
        )

        categorical_count = (
            rsu_chunk_mask.sum(dim=-1)
            + rsu_layer_mask.sum(dim=-1)
            + uav_chunk_mask.sum(dim=-1)
            + uav_layer_mask.sum(dim=-1)
        )

        categorical_entropy = (
            categorical_entropy_sum
            / categorical_count.clamp_min(
                1.0
            )
        )

        categorical_entropy = torch.where(
            categorical_count > 0.0,
            categorical_entropy,
            torch.zeros_like(
                categorical_entropy
            ),
        )

        power_entropy = (
            self._masked_mean(
                power_dist.entropy(),
                power_mask,
            )
        )

        return (
            log_prob,
            categorical_entropy,
            power_entropy,
            self.value(obs),
        )