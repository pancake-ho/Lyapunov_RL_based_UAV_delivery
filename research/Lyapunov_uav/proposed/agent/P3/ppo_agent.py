from __future__ import annotations

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


@dataclass(frozen=True)
class PPOChoice:
    action_index: int
    log_prob: float
    value: float
    entropy: float


@dataclass
class PPOTransition:
    state_features: np.ndarray
    candidate_features: np.ndarray
    action_index: int
    old_log_prob: float
    old_value: float
    reward: float
    next_value: float
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


class CandidateActorCritic(nn.Module):
    """Scores only feasible enumerated frame actions.

    Candidate enumeration is the hard action mask.  This avoids invalid PPO
    actions while retaining a variable number of feasible assignments.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def policy_logits(
        self,
        states: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        state_latent = self.state_encoder(states)
        action_latent = self.action_encoder(candidates)
        expanded_state = state_latent.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        logits = self.actor(torch.cat([expanded_state, action_latent], dim=-1)).squeeze(-1)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)

    def values(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(self.state_encoder(states)).squeeze(-1)


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
        self.network = CandidateActorCritic(
            cfg.ppo_state_dim,
            cfg.ppo_action_dim,
            cfg.ppo_hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=cfg.ppo_learning_rate,
        )

    @torch.no_grad()
    def select(
        self,
        state_features: np.ndarray,
        candidate_features: np.ndarray,
        deterministic: bool = False,
    ) -> PPOChoice:
        states = torch.as_tensor(
            state_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        candidates = torch.as_tensor(
            candidate_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask = torch.ones(
            (1, candidate_features.shape[0]),
            dtype=torch.bool,
            device=self.device,
        )
        logits = self.network.policy_logits(states, candidates, mask)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        value = self.network.values(states)
        return PPOChoice(
            action_index=int(action.item()),
            log_prob=float(distribution.log_prob(action).item()),
            value=float(value.item()),
            entropy=float(distribution.entropy().item()),
        )

    @torch.no_grad()
    def value(self, state_features: np.ndarray) -> float:
        state = torch.as_tensor(
            state_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return float(self.network.values(state).item())

    def update(self, transitions: Sequence[PPOTransition]) -> dict[str, float]:
        if not transitions:
            raise ValueError("PPO update requires at least one transition")
        advantages = np.asarray(
            [transition.advantage for transition in transitions],
            dtype=np.float32,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.asarray(
            [transition.return_value for transition in transitions],
            dtype=np.float32,
        )
        old_log_probs = np.asarray(
            [transition.old_log_prob for transition in transitions],
            dtype=np.float32,
        )
        action_indices = np.asarray(
            [transition.action_index for transition in transitions],
            dtype=np.int64,
        )

        losses: list[tuple[float, float, float, float, float, float, float, float]] = []
        total = len(transitions)
        epochs_completed = 0
        stopped_by_kl = False
        for _ in range(self.cfg.ppo_update_epochs):
            epoch_kls: list[float] = []
            permutation = np.random.permutation(total)
            for start in range(0, total, self.cfg.ppo_batch_size):
                indices = permutation[start : start + self.cfg.ppo_batch_size]
                states = torch.as_tensor(
                    np.stack([transitions[index].state_features for index in indices]),
                    dtype=torch.float32,
                    device=self.device,
                )
                candidates, mask = self._pad_candidates(
                    [transitions[index].candidate_features for index in indices]
                )
                selected = torch.as_tensor(
                    action_indices[indices], dtype=torch.long, device=self.device
                )
                old_log = torch.as_tensor(
                    old_log_probs[indices], dtype=torch.float32, device=self.device
                )
                advantage = torch.as_tensor(
                    advantages[indices], dtype=torch.float32, device=self.device
                )
                target_return = torch.as_tensor(
                    returns[indices], dtype=torch.float32, device=self.device
                )

                logits = self.network.policy_logits(states, candidates, mask)
                distribution = Categorical(logits=logits)
                new_log = distribution.log_prob(selected)
                entropy = distribution.entropy().mean()
                candidate_counts = mask.sum(dim=1).to(dtype=torch.float32)
                entropy_scale = torch.log(candidate_counts).clamp_min(1e-8)
                normalized_entropy = (
                    distribution.entropy() / entropy_scale
                ).mean()
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
                values = self.network.values(states)
                value_loss = 0.5 * torch.mean((values - target_return) ** 2)
                loss = (
                    policy_loss
                    + self.cfg.ppo_value_coef * value_loss
                    - self.cfg.ppo_entropy_coef * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite PPO loss; aborting before optimizer step"
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.cfg.ppo_max_grad_norm,
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        "non-finite PPO gradient norm; aborting optimizer step"
                    )
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
            mean_epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            if mean_epoch_kl > self.cfg.ppo_target_kl:
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

    def _pad_candidates(
        self,
        matrices: Sequence[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_candidates = max(matrix.shape[0] for matrix in matrices)
        padded = np.zeros(
            (len(matrices), max_candidates, self.cfg.ppo_action_dim),
            dtype=np.float32,
        )
        mask = np.zeros((len(matrices), max_candidates), dtype=bool)
        for row, matrix in enumerate(matrices):
            padded[row, : matrix.shape[0]] = matrix
            mask[row, : matrix.shape[0]] = True
        return (
            torch.as_tensor(padded, dtype=torch.float32, device=self.device),
            torch.as_tensor(mask, dtype=torch.bool, device=self.device),
        )

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
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
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return dict(checkpoint.get("metadata", {}))
