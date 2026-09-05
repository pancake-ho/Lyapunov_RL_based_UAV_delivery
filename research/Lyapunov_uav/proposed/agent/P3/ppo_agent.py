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
    def _compact_distribution(
        logits: torch.Tensor,
        allowed_tokens: torch.Tensor,
    ) -> tuple[Categorical | None, torch.Tensor]:
        """Build a categorical only on the feasible support.

        Filling infeasible logits with ``torch.finfo(dtype).min`` is finite in
        the forward pass, but entropy backward can still create non-finite
        gradients on CUDA, especially when a factor has exactly one feasible
        token and normalized entropy amplifies the result.  Removing infeasible
        logits from the autograd graph is mathematically equivalent to an exact
        action mask and avoids that numerical failure mode.

        A singleton support is deterministic.  Returning ``None`` lets callers
        produce an explicit differentiable zero instead of asking
        ``Categorical.entropy`` to operate on a degenerate distribution.
        """

        if logits.ndim != 1:
            raise ValueError("factor logits must be one-dimensional")
        support = allowed_tokens.to(device=logits.device, dtype=torch.long)
        if support.ndim != 1 or support.numel() == 0:
            raise ValueError("a factor must have at least one feasible token")
        if torch.unique(support).numel() != support.numel():
            raise ValueError("feasible factor tokens must be unique")
        compact_logits = logits.index_select(0, support)
        if support.numel() == 1:
            return None, support
        return Categorical(logits=compact_logits), support

    @staticmethod
    def _differentiable_zero(logits: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """Return an exact zero connected to the selected logit's graph."""

        return logits.index_select(0, token.reshape(1).long()).sum() * 0.0

    @staticmethod
    def _bad_named_tensors(
        named_tensors: Sequence[tuple[str, torch.Tensor]],
    ) -> list[str]:
        """Find non-finite tensors with one device synchronization per device."""

        groups: dict[torch.device, list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
        for name, tensor in named_tensors:
            finite = torch.isfinite(tensor).all()
            groups.setdefault(tensor.device, []).append((name, tensor, finite))
        bad: list[str] = []
        for group in groups.values():
            if bool(torch.stack([finite for _, _, finite in group]).all()):
                continue
            bad.extend(
                name
                for name, _, finite in group
                if not bool(finite)
            )
        return bad

    @classmethod
    def _bad_parameter_names(cls, network: nn.Module) -> list[str]:
        return cls._bad_named_tensors(list(network.named_parameters()))

    @classmethod
    def _bad_gradient_names(cls, network: nn.Module) -> list[str]:
        return cls._bad_named_tensors(
            [
                (name, parameter.grad)
                for name, parameter in network.named_parameters()
                if parameter.grad is not None
            ]
        )

    def _bad_optimizer_state_names(self) -> list[str]:
        named_states: list[tuple[str, torch.Tensor]] = []
        parameter_names = {
            id(parameter): name for name, parameter in self.network.named_parameters()
        }
        for parameter, state in self.optimizer.state.items():
            parameter_name = parameter_names.get(id(parameter), "<unknown>")
            for state_name, value in state.items():
                if torch.is_tensor(value):
                    named_states.append((f"{parameter_name}.{state_name}", value))
        return self._bad_named_tensors(named_states)

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
        if signatures.shape[0] == 0:
            raise ValueError("candidate signature matrix must not be empty")
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
            factor_logits = logits.squeeze(0)
            distribution, support = self._compact_distribution(
                factor_logits, allowed
            )
            local_matches = torch.nonzero(support == token, as_tuple=False).flatten()
            if local_matches.numel() != 1:
                raise RuntimeError("chosen factor token has no unique local index")
            if distribution is None:
                factor_zero = self._differentiable_zero(factor_logits, token)
                log_prob = log_prob + factor_zero
                entropy = entropy + factor_zero
            else:
                local_token = local_matches[0]
                log_prob = log_prob + distribution.log_prob(local_token)
                entropy = entropy + distribution.entropy()
                max_entropy = max_entropy + math.log(float(support.numel()))
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
        state_array = np.asarray(state_features)
        signature_array = np.asarray(candidate_signatures)
        if state_array.ndim != 1 or state_array.size != self.cfg.ppo_state_dim:
            raise ValueError("state feature width does not match configuration")
        if not bool(np.isfinite(state_array).all()):
            raise FloatingPointError("non-finite PPO state features")
        if (
            signature_array.ndim != 2
            or signature_array.shape[1] != self.cfg.ppo_signature_dim
        ):
            raise ValueError("candidate signature shape does not match configuration")
        if signature_array.shape[0] == 0:
            raise ValueError("candidate signature matrix must not be empty")
        if not bool(np.isfinite(signature_array).all()):
            raise FloatingPointError("non-finite PPO candidate signatures")
        state = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        )
        signatures = torch.as_tensor(
            signature_array, dtype=torch.long, device=self.device
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
            factor_logits = logits.squeeze(0)
            distribution, support = self._compact_distribution(
                factor_logits, allowed
            )
            if distribution is None:
                local_token = torch.zeros((), dtype=torch.long, device=self.device)
                token_tensor = support[0]
                factor_log_prob = 0.0
                factor_entropy = 0.0
            else:
                local_token = (
                    torch.argmax(distribution.logits)
                    if deterministic
                    else distribution.sample()
                )
                token_tensor = support[local_token]
                factor_log_prob = float(distribution.log_prob(local_token).item())
                factor_entropy = float(distribution.entropy().item())
            token = int(token_tensor.item())
            if column == 0:
                hire_matches = torch.nonzero(
                    support == 1, as_tuple=False
                ).flatten()
                if hire_matches.numel() == 0:
                    hire_probability = 0.0
                elif distribution is None:
                    hire_probability = 1.0
                else:
                    hire_probability = float(
                        distribution.probs[hire_matches[0]].item()
                    )
            log_prob += factor_log_prob
            entropy += factor_entropy
            if distribution is not None:
                max_entropy += math.log(float(support.numel()))
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
        state_array = np.asarray(state_features)
        if state_array.ndim != 1 or state_array.size != self.cfg.ppo_state_dim:
            raise ValueError("state feature width does not match configuration")
        if not bool(np.isfinite(state_array).all()):
            raise FloatingPointError("non-finite PPO state features")
        state = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return float(self.network.values(state).item())

    def update(self, transitions: Sequence[PPOTransition]) -> dict[str, float]:
        if not transitions:
            raise ValueError("PPO update requires at least one transition")
        raw_arrays = {
            "advantages": np.asarray(
                [transition.advantage for transition in transitions], dtype=np.float64
            ),
            "returns": np.asarray(
                [transition.return_value for transition in transitions], dtype=np.float64
            ),
            "old_log_probs": np.asarray(
                [transition.old_log_prob for transition in transitions], dtype=np.float64
            ),
            "old_values": np.asarray(
                [transition.old_value for transition in transitions], dtype=np.float64
            ),
        }
        for name, values in raw_arrays.items():
            if not bool(np.isfinite(values).all()):
                bad_indices = np.flatnonzero(~np.isfinite(values))[:8].tolist()
                raise FloatingPointError(
                    f"non-finite PPO {name} at transition indices {bad_indices}"
                )
        for index, transition in enumerate(transitions):
            state = np.asarray(transition.state_features)
            signatures = np.asarray(transition.candidate_signatures)
            if state.shape != (self.cfg.ppo_state_dim,):
                raise ValueError(
                    f"invalid PPO state shape at transition {index}: {state.shape}"
                )
            if not bool(np.isfinite(state).all()):
                raise FloatingPointError(
                    f"non-finite PPO state features at transition {index}"
                )
            if (
                signatures.ndim != 2
                or signatures.shape[0] == 0
                or signatures.shape[1] != self.cfg.ppo_signature_dim
            ):
                raise ValueError(
                    "invalid PPO candidate signature shape at transition "
                    f"{index}: {signatures.shape}"
                )
            if not bool(np.isfinite(signatures).all()):
                raise FloatingPointError(
                    f"non-finite PPO candidate signatures at transition {index}"
                )
            if not 0 <= int(transition.action_index) < signatures.shape[0]:
                raise IndexError(
                    f"PPO action index is outside candidate set at transition {index}"
                )
        advantages64 = raw_arrays["advantages"]
        advantage_scale = max(float(advantages64.std()), 1e-8)
        advantages = (
            (advantages64 - float(advantages64.mean())) / advantage_scale
        ).astype(np.float32)
        returns = raw_arrays["returns"].astype(np.float32)
        old_log_probs = raw_arrays["old_log_probs"].astype(np.float32)
        old_values = raw_arrays["old_values"].astype(np.float32)

        bad_parameters = self._bad_parameter_names(self.network)
        bad_optimizer_states = self._bad_optimizer_state_names()
        if bad_parameters or bad_optimizer_states:
            raise FloatingPointError(
                "PPO update started from non-finite state: "
                f"parameters={bad_parameters}, optimizer={bad_optimizer_states}"
            )

        losses: list[tuple[float, ...]] = []
        total = len(transitions)
        epochs_completed = 0
        stopped_by_kl = False
        for epoch_index in range(self.cfg.ppo_update_epochs):
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
                max_entropy_vector = torch.stack(max_entropies)
                normalized_entropy_vector = torch.where(
                    max_entropy_vector > 0.0,
                    entropy_vector / max_entropy_vector.clamp_min(1e-8),
                    torch.zeros_like(entropy_vector),
                )
                normalized_entropy = normalized_entropy_vector.mean()
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
                tensors_to_check = {
                    "new_log_probs": new_log,
                    "entropy": entropy_vector,
                    "max_entropy": max_entropy_vector,
                    "values": values_tensor,
                    "old_log_probs": old_log,
                    "advantages": advantage,
                    "returns": target_return,
                }
                for name, tensor in tensors_to_check.items():
                    if not bool(torch.isfinite(tensor).all()):
                        raise FloatingPointError(
                            "non-finite PPO minibatch tensor "
                            f"{name} at epoch={epoch_index + 1}, start={start}"
                        )
                log_ratio = new_log - old_log
                if not bool(torch.isfinite(log_ratio).all()):
                    raise FloatingPointError(
                        "non-finite PPO log ratio at "
                        f"epoch={epoch_index + 1}, start={start}"
                    )
                ratio = torch.exp(log_ratio)
                if not bool(torch.isfinite(ratio).all()):
                    raise FloatingPointError(
                        "non-finite PPO probability ratio at "
                        f"epoch={epoch_index + 1}, start={start}; "
                        f"log_ratio_range=({float(log_ratio.min().item()):.6g}, "
                        f"{float(log_ratio.max().item()):.6g})"
                    )
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
                    raise FloatingPointError(
                        "non-finite PPO loss at "
                        f"epoch={epoch_index + 1}, start={start}; "
                        f"policy={float(policy_loss.detach().item()):.6g}, "
                        f"value={float(value_loss.detach().item()):.6g}, "
                        f"entropy={float(normalized_entropy.detach().item()):.6g}"
                    )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                bad_gradients = self._bad_gradient_names(self.network)
                if bad_gradients:
                    raise FloatingPointError(
                        "non-finite PPO gradients before clipping at "
                        f"epoch={epoch_index + 1}, start={start}; "
                        f"parameters={bad_gradients}"
                    )
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.cfg.ppo_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as error:
                    raise FloatingPointError(
                        "non-finite PPO gradient norm at "
                        f"epoch={epoch_index + 1}, start={start}"
                    ) from error
                self.optimizer.step()
                bad_parameters = self._bad_parameter_names(self.network)
                bad_optimizer_states = self._bad_optimizer_state_names()
                if bad_parameters or bad_optimizer_states:
                    raise FloatingPointError(
                        "PPO optimizer produced non-finite state at "
                        f"epoch={epoch_index + 1}, start={start}; "
                        f"parameters={bad_parameters}, optimizer={bad_optimizer_states}"
                    )
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
        bad_parameters = self._bad_parameter_names(self.network)
        bad_optimizer_states = self._bad_optimizer_state_names()
        if bad_parameters or bad_optimizer_states:
            raise FloatingPointError(
                "refusing to save a non-finite PPO checkpoint: "
                f"parameters={bad_parameters}, optimizer={bad_optimizer_states}"
            )
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
        bad_parameters = self._bad_parameter_names(self.network)
        bad_optimizer_states = (
            self._bad_optimizer_state_names() if load_optimizer else []
        )
        if bad_parameters or bad_optimizer_states:
            raise FloatingPointError(
                "loaded PPO checkpoint contains non-finite state: "
                f"parameters={bad_parameters}, optimizer={bad_optimizer_states}"
            )
        return dict(checkpoint.get("metadata", {}))
