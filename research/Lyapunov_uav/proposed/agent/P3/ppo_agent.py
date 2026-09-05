from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from config_p3 import P3Config


ARCHITECTURE_VERSION = "p3-factorized-slow-ppo-v1"


@dataclass(frozen=True)
class PPOChoice:
    action_index: int
    log_prob: float
    value: float
    entropy: float
    normalized_entropy: float
    hire_probability: float


@dataclass
class PPOTransition:
    state_features: np.ndarray
    candidate_signatures: np.ndarray
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    next_value: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class FactorizedActorCritic(nn.Module):
    """Section 8.5 actor: hire -> point -> sequential user providers.

    A complete list of feasible joint signatures provides an exact prefix mask
    at every factor. The actor is factorized while exploration and inference
    retain reachability, capacity, battery and single-provider constraints.
    """

    def __init__(self, cfg: P3Config) -> None:
        super().__init__()
        hidden = cfg.ppo_hidden_dim
        self.cfg = cfg
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.ppo_state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.user_encoder = nn.Sequential(
            nn.Linear(cfg.ppo_user_feature_dim, hidden),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(nn.Linear(4, hidden), nn.Tanh())
        self.hire_head = nn.Linear(hidden, 2)
        self.point_head = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, len(cfg.candidate_offsets_m) + 1),
        )
        self.provider_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(states)

    def hire_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.hire_head(latent)

    def point_logits(self, latent: torch.Tensor, hired: torch.Tensor) -> torch.Tensor:
        hired_one_hot = torch.nn.functional.one_hot(
            hired.long(), num_classes=2
        ).to(dtype=latent.dtype)
        return self.point_head(torch.cat([latent, hired_one_hot], dim=-1))

    def provider_logits(
        self,
        latent: torch.Tensor,
        states: torch.Tensor,
        user_index: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        start = (
            self.cfg.ppo_global_feature_dim
            + int(user_index) * self.cfg.ppo_user_feature_dim
        )
        stop = start + self.cfg.ppo_user_feature_dim
        user = self.user_encoder(states[..., start:stop])
        context_latent = self.context_encoder(context.to(dtype=latent.dtype))
        return self.provider_head(torch.cat([latent, user, context_latent], dim=-1))

    def values_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.critic(latent).squeeze(-1)

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self.values_from_latent(self.encode(states))


def finish_trajectory(transitions: list[PPOTransition], cfg: P3Config) -> None:
    """Compute GAE separately for one region trajectory."""

    gae = 0.0
    for transition in reversed(transitions):
        nonterminal = 0.0 if transition.done else 1.0
        delta = (
            transition.reward
            + cfg.ppo_gamma * nonterminal * transition.next_value
            - transition.old_value
        )
        gae = delta + cfg.ppo_gamma * cfg.ppo_gae_lambda * nonterminal * gae
        transition.advantage = float(gae)
        transition.return_value = float(gae + transition.old_value)


class PPOAgent:
    def __init__(
        self,
        cfg: P3Config,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = cfg
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.network = FactorizedActorCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.ppo_learning_rate
        )

    @staticmethod
    def _masked_distribution(
        logits: torch.Tensor,
        allowed_tokens: torch.Tensor,
    ) -> Categorical:
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[allowed_tokens.long()] = True
        masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked)

    def _evaluate_choice(
        self,
        state: torch.Tensor,
        signatures: torch.Tensor,
        action_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 1 or signatures.ndim != 2:
            raise ValueError("expected one state and a 2-D signature matrix")
        if signatures.shape[1] != self.cfg.ppo_signature_dim:
            raise ValueError("candidate signature width does not match configuration")
        if not 0 <= int(action_index) < signatures.shape[0]:
            raise IndexError("selected action index is outside candidate set")

        latent = self.network.encode(state.unsqueeze(0))
        chosen = signatures[int(action_index)]
        pool = torch.arange(signatures.shape[0], device=self.device)
        log_prob = torch.zeros((), dtype=state.dtype, device=self.device)
        entropy = torch.zeros_like(log_prob)
        max_entropy = torch.zeros_like(log_prob)

        def consume(logits: torch.Tensor, column: int) -> torch.Tensor:
            nonlocal pool, log_prob, entropy, max_entropy
            allowed = torch.unique(signatures[pool, column], sorted=True)
            token = chosen[column].long()
            if not bool(torch.any(allowed == token)):
                raise RuntimeError("chosen factor is not feasible under its prefix")
            distribution = self._masked_distribution(logits.squeeze(0), allowed)
            log_prob = log_prob + distribution.log_prob(token)
            entropy = entropy + distribution.entropy()
            if allowed.numel() > 1:
                max_entropy = max_entropy + math.log(float(allowed.numel()))
            pool = pool[signatures[pool, column] == token]
            return token

        hired = consume(self.network.hire_logits(latent), 0)
        point = consume(self.network.point_logits(latent, hired.reshape(1)), 1)
        rsu_count = 0
        uav_count = 0
        for user in range(self.cfg.num_users):
            context = torch.tensor(
                [[
                    float(hired.item()),
                    float(point.item()) / max(len(self.cfg.candidate_offsets_m), 1),
                    rsu_count / self.cfg.rsu_capacity,
                    uav_count / self.cfg.uav_capacity,
                ]],
                dtype=state.dtype,
                device=self.device,
            )
            token = consume(
                self.network.provider_logits(latent, state.unsqueeze(0), user, context),
                2 + user,
            )
            rsu_count += int(token.item() == 1)
            uav_count += int(token.item() == 2)
        if pool.numel() != 1 or int(pool.item()) != int(action_index):
            raise RuntimeError("factorized signature did not identify one joint action")
        value = self.network.values_from_latent(latent)[0]
        return log_prob, entropy, max_entropy, value

    @torch.no_grad()
    def select(
        self,
        state_features: np.ndarray,
        candidate_signatures: np.ndarray,
        deterministic: bool = False,
    ) -> PPOChoice:
        state = torch.as_tensor(
            state_features, dtype=torch.float32, device=self.device
        )
        signatures = torch.as_tensor(
            candidate_signatures, dtype=torch.long, device=self.device
        )
        latent = self.network.encode(state.unsqueeze(0))
        pool = torch.arange(signatures.shape[0], device=self.device)
        log_prob = 0.0
        entropy = 0.0
        max_entropy = 0.0
        hire_probability = 0.0

        def choose(logits: torch.Tensor, column: int) -> int:
            nonlocal pool, log_prob, entropy, max_entropy, hire_probability
            allowed = torch.unique(signatures[pool, column], sorted=True)
            distribution = self._masked_distribution(logits.squeeze(0), allowed)
            token_tensor = (
                torch.argmax(distribution.logits)
                if deterministic
                else distribution.sample()
            )
            token = int(token_tensor.item())
            if column == 0:
                probabilities = distribution.probs
                hire_probability = (
                    float(probabilities[1].item())
                    if probabilities.numel() > 1
                    else float(token == 1)
                )
            log_prob += float(distribution.log_prob(token_tensor).item())
            entropy += float(distribution.entropy().item())
            if allowed.numel() > 1:
                max_entropy += math.log(float(allowed.numel()))
            pool = pool[signatures[pool, column] == token]
            return token

        hired = choose(self.network.hire_logits(latent), 0)
        point = choose(
            self.network.point_logits(
                latent,
                torch.tensor([hired], dtype=torch.long, device=self.device),
            ),
            1,
        )
        rsu_count = 0
        uav_count = 0
        for user in range(self.cfg.num_users):
            context = torch.tensor(
                [[
                    hired,
                    point / max(len(self.cfg.candidate_offsets_m), 1),
                    rsu_count / self.cfg.rsu_capacity,
                    uav_count / self.cfg.uav_capacity,
                ]],
                dtype=torch.float32,
                device=self.device,
            )
            provider = choose(
                self.network.provider_logits(
                    latent, state.unsqueeze(0), user, context
                ),
                2 + user,
            )
            rsu_count += int(provider == 1)
            uav_count += int(provider == 2)
        if pool.numel() != 1:
            raise RuntimeError("factorized sampling left an ambiguous action")
        value = self.network.values_from_latent(latent)
        return PPOChoice(
            action_index=int(pool.item()),
            log_prob=float(log_prob),
            value=float(value.item()),
            entropy=float(entropy),
            normalized_entropy=float(entropy / max(max_entropy, 1e-8)),
            hire_probability=float(hire_probability),
        )

    @torch.no_grad()
    def value(self, state_features: np.ndarray) -> float:
        state = torch.as_tensor(
            state_features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.network.values(state).item())

    def update(self, transitions: Sequence[PPOTransition]) -> dict[str, float]:
        if not transitions:
            raise ValueError("PPO update requires at least one transition")
        advantages = np.asarray(
            [transition.advantage for transition in transitions], dtype=np.float32
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.asarray(
            [transition.return_value for transition in transitions], dtype=np.float32
        )
        old_log_probs = np.asarray(
            [transition.old_log_prob for transition in transitions], dtype=np.float32
        )

        losses: list[tuple[float, ...]] = []
        total = len(transitions)
        epochs_completed = 0
        stopped_by_kl = False
        for _ in range(self.cfg.ppo_update_epochs):
            epoch_kls: list[float] = []
            permutation = np.random.permutation(total)
            for start in range(0, total, self.cfg.ppo_batch_size):
                indices = permutation[start : start + self.cfg.ppo_batch_size]
                new_logs = []
                entropies = []
                max_entropies = []
                values = []
                for index in indices:
                    transition = transitions[int(index)]
                    state = torch.as_tensor(
                        transition.state_features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    signatures = torch.as_tensor(
                        transition.candidate_signatures,
                        dtype=torch.long,
                        device=self.device,
                    )
                    log_prob, entropy, max_entropy, value = self._evaluate_choice(
                        state, signatures, transition.action_index
                    )
                    new_logs.append(log_prob)
                    entropies.append(entropy)
                    max_entropies.append(max_entropy)
                    values.append(value)

                new_log = torch.stack(new_logs)
                entropy_vector = torch.stack(entropies)
                max_entropy_vector = torch.stack(max_entropies).clamp_min(1e-8)
                normalized_entropy = (entropy_vector / max_entropy_vector).mean()
                entropy = entropy_vector.mean()
                values_tensor = torch.stack(values)
                old_log = torch.as_tensor(
                    old_log_probs[indices], dtype=torch.float32, device=self.device
                )
                advantage = torch.as_tensor(
                    advantages[indices], dtype=torch.float32, device=self.device
                )
                target_return = torch.as_tensor(
                    returns[indices], dtype=torch.float32, device=self.device
                )
                ratio = torch.exp(new_log - old_log)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > self.cfg.ppo_clip_ratio).float()
                )
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.cfg.ppo_clip_ratio,
                    1.0 + self.cfg.ppo_clip_ratio,
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * torch.mean(
                    (values_tensor - target_return) ** 2
                )
                loss = (
                    policy_loss
                    + self.cfg.ppo_value_coef * value_loss
                    - self.cfg.ppo_entropy_coef * normalized_entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite PPO loss")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.cfg.ppo_max_grad_norm
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                approx_kl = torch.mean(old_log - new_log).detach()
                epoch_kls.append(float(approx_kl.item()))
                losses.append(
                    (
                        float(loss.detach().item()),
                        float(policy_loss.detach().item()),
                        float(value_loss.detach().item()),
                        float(entropy.detach().item()),
                        float(approx_kl.item()),
                        float(normalized_entropy.detach().item()),
                        float(clip_fraction.detach().item()),
                        float(grad_norm.detach().item()),
                    )
                )
            epochs_completed += 1
            if epoch_kls and float(np.mean(epoch_kls)) > self.cfg.ppo_target_kl:
                stopped_by_kl = True
                break

        matrix = np.asarray(losses, dtype=np.float64)
        old_values = np.asarray(
            [transition.old_value for transition in transitions], dtype=np.float32
        )
        return_variance = float(np.var(returns))
        explained_variance = (
            1.0 - float(np.var(returns - old_values)) / return_variance
            if return_variance > 1e-12
            else 0.0
        )
        return {
            "loss": float(matrix[:, 0].mean()),
            "policy_loss": float(matrix[:, 1].mean()),
            "value_loss": float(matrix[:, 2].mean()),
            "entropy": float(matrix[:, 3].mean()),
            "approx_kl": float(matrix[:, 4].mean()),
            "normalized_entropy": float(matrix[:, 5].mean()),
            "clip_fraction": float(matrix[:, 6].mean()),
            "grad_norm": float(matrix[:, 7].mean()),
            "explained_variance": explained_variance,
            "update_epochs_completed": float(epochs_completed),
            "stopped_by_kl": float(stopped_by_kl),
            "transitions": float(total),
        }

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
                    "architecture": ARCHITECTURE_VERSION,
                    "model_state_dict": self.network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": asdict(self.cfg),
                    "metadata": metadata or {},
                },
                temporary,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self, path: Path, load_optimizer: bool = False) -> dict:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        architecture = checkpoint.get("architecture")
        if architecture != ARCHITECTURE_VERSION:
            raise RuntimeError(
                f"checkpoint architecture {architecture!r} is incompatible with "
                f"{ARCHITECTURE_VERSION!r}; retrain the slow PPO"
            )
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return dict(checkpoint.get("metadata", {}))
