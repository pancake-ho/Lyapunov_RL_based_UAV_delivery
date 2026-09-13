from __future__ import annotations

"""Factorized MultiDiscrete PPO actor-critic (from uav_hierarchical_ppo/ppo.py)
with per-head action masks and the finite-value safety checks used by the P3
slow PPO (agent/P3/ppo_agent.py).

One ``PPOAgent`` instance is used for the frame level (``name='frame_ppo'``)
and one for the slot level (``name='slot_ppo'``). Both are shared across
regions: each region contributes its own trajectory with its own GAE.
"""

import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config_hppo import HPPOConfig


ARCHITECTURE_VERSION = "p3-scheduling-only-hppo-v2"
MASK_FILL = -1.0e9  # finite "minus infinity" for masked logits


class MaskedMultiDiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int, nvec: Sequence[int], hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.nvec = [int(n) for n in nvec]
        layers: list[nn.Module] = []
        in_dim = self.obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(h)), nn.Tanh()])
            in_dim = int(h)
        self.trunk = nn.Sequential(*layers)
        self.actor_heads = nn.ModuleList([nn.Linear(in_dim, n) for n in self.nvec])
        self.critic = nn.Linear(in_dim, 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def distributions(self, obs: torch.Tensor, masks: list[torch.Tensor] | None):
        """masks: list (one per head) of bool tensors shaped (batch, n_i)."""

        z = self.trunk(obs)
        dists = []
        for i, head in enumerate(self.actor_heads):
            logits = head(z)
            if masks is not None:
                logits = logits.masked_fill(~masks[i], MASK_FILL)
            dists.append(Categorical(logits=logits))
        return dists, self.critic(z).squeeze(-1)

    def value(self, obs):
        return self.critic(self.trunk(obs)).squeeze(-1)

    def _joint(self, obs, masks):
        z = self.trunk(obs)
        # Same independent fast heads as Claude, evaluated as one batched distribution.
        flat = F.linear(z, torch.cat([h.weight for h in self.actor_heads]),
                        torch.cat([h.bias for h in self.actor_heads]))
        parts = flat.split(self.nvec, dim=-1)
        width = max(self.nvec)
        logits = []
        for i, part in enumerate(parts):
            if masks is not None:
                part = part.masked_fill(~masks[i], MASK_FILL)
            logits.append(F.pad(part, (0, width - self.nvec[i]), value=MASK_FILL))
        return Categorical(logits=torch.stack(logits, dim=1)), self.critic(z).squeeze(-1)

    @torch.no_grad()
    def act_batch(self, obs, masks, deterministic=False, uniforms=None):
        dist, value = self._joint(obs, masks)
        if deterministic:
            action = dist.logits.argmax(dim=-1)
        elif uniforms is None:
            action = dist.sample()
        else:
            # Inverse-CDF draws preserve the categorical policy without changing global RNG.
            probs = dist.probs.to(torch.float64)
            cdf = probs.cumsum(dim=-1) / probs.sum(dim=-1, keepdim=True)
            action = (uniforms.unsqueeze(-1) >= cdf).sum(dim=-1)
            action = torch.minimum(action, torch.as_tensor(self.nvec, device=obs.device) - 1)
        return action, dist.log_prob(action).sum(-1), value, dist.entropy().sum(-1)

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return tuple(x.squeeze(0) for x in self.act_batch(obs, masks, deterministic))

    def evaluate_actions(self, obs, actions, masks):
        dist, value = self._joint(obs, masks)
        return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1), value


class SchedulingActorCritic(MaskedMultiDiscreteActorCritic):
    """Autoregressive provider proposals; internal prefixes do not change env observations.

    User IDs use a fixed order. Each step sees earlier provider selections,
    and its conditional support respects remaining RSU/UAV capacity.
    There is one joint log probability and one frame reward per proposal.
    """
    def __init__(self, obs_dim, nvec, hidden_dims, cfg):
        super().__init__(obs_dim, nvec, hidden_dims)
        self.rsu_capacity, self.uav_capacity = cfg.rsu_capacity, cfg.uav_capacity
        self.prefix = nn.Linear(3 * len(nvec), int(hidden_dims[-1]), bias=False)
        nn.init.zeros_(self.prefix.weight)

    def _decode(self, obs, masks, actions=None, deterministic=False):
        if masks is None:
            raise ValueError("scheduling requires stored member/feasibility masks")
        z = self.trunk(obs)
        prefix = z.new_zeros((len(obs), 3 * len(self.nvec)))
        nr = torch.zeros(len(obs), dtype=torch.long, device=obs.device)
        nu = torch.zeros_like(nr)
        choices, logps, entropies = [], [], []
        for i, head in enumerate(self.actor_heads):
            valid = masks[i].clone()
            valid[:, 1] &= nr < self.rsu_capacity
            valid[:, 2] &= nu < self.uav_capacity
            # Invalid data must fail explicitly, not be projected to another action.
            if not bool(valid.any(-1).all()):
                raise ValueError("empty conditional scheduling support")
            logits = head(torch.tanh(z + self.prefix(prefix)))
            dist = Categorical(logits=logits.masked_fill(~valid, MASK_FILL))
            a = actions[:, i] if actions is not None else (
                dist.logits.argmax(-1) if deterministic else dist.sample())
            if not bool(valid.gather(1, a.unsqueeze(1)).all()):
                raise ValueError("stored scheduling action violates its prefix mask")
            choices.append(a); logps.append(dist.log_prob(a)); entropies.append(dist.entropy())
            nr = nr + (a == 1).long(); nu = nu + (a == 2).long()
            token = F.one_hot(a, 3).to(z.dtype)
            prefix = prefix + F.pad(token, (3 * i, 3 * (len(self.nvec) - i - 1)))
        return (torch.stack(choices, 1), torch.stack(logps, 1).sum(1),
                self.critic(z).squeeze(-1), torch.stack(entropies, 1).sum(1))

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return tuple(x.squeeze(0) for x in self._decode(obs, masks, deterministic=deterministic))

    def evaluate_actions(self, obs, actions, masks):
        _, logp, value, entropy = self._decode(obs, masks, actions=actions)
        return logp, entropy, value


@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray          # concatenated bool mask over all heads (sum(nvec),)
    action: np.ndarray
    logp: float
    value: float
    reward: float
    next_value: float         # bootstrap value of the successor state (0 if done)
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


def compute_gae(trajectory: list[Transition], gamma: float, lam: float) -> None:
    """GAE over one region trajectory (same as agent/P3/ppo_agent.finish_trajectory)."""

    gae = 0.0
    for tr in reversed(trajectory):
        nonterminal = 0.0 if tr.done else 1.0
        delta = tr.reward + gamma * nonterminal * tr.next_value - tr.value
        gae = delta + gamma * lam * nonterminal * gae
        tr.advantage = float(gae)
        tr.return_value = float(gae + tr.value)


class PPOAgent:
    def __init__(self, obs_dim: int, nvec: Sequence[int], cfg: HPPOConfig, name: str) -> None:
        self.cfg = cfg
        self.name = name
        self.nvec = [int(n) for n in nvec]
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        self.scheduling = name == "frame_ppo"
        self.net = (SchedulingActorCritic(obs_dim, self.nvec, cfg.hidden_dims, cfg)
                    if self.scheduling else MaskedMultiDiscreteActorCritic(obs_dim, self.nvec, cfg.hidden_dims)).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo_learning_rate)
        # per-trajectory buffers (key = region); GAE is computed per trajectory
        self.trajectories: dict[int, list[Transition]] = {}
        self.update_count = 0

    # ------------------------------------------------------------------
    def _mask_tensors(self, masks: Sequence[np.ndarray] | None, batch: int = 1) -> list[torch.Tensor] | None:
        if masks is None:
            return None
        out = []
        for i, m in enumerate(masks):
            t = torch.as_tensor(np.asarray(m, dtype=bool), device=self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0).expand(batch, -1)
            if t.shape[-1] != self.nvec[i]:
                raise ValueError(f"mask {i} width {t.shape[-1]} != head width {self.nvec[i]}")
            if not bool(t.any(dim=-1).all()):
                raise ValueError(f"mask {i} has an empty feasible support")
            out.append(t)
        return out

    @staticmethod
    def flatten_masks(masks: Sequence[np.ndarray] | None, nvec: Sequence[int]) -> np.ndarray:
        if masks is None:
            return np.ones(int(sum(nvec)), dtype=bool)
        return np.concatenate([np.asarray(m, dtype=bool) for m in masks])

    def _split_masks(self, flat: torch.Tensor) -> list[torch.Tensor]:
        out, start = [], 0
        for n in self.nvec:
            out.append(flat[:, start:start + n])
            start += n
        return out

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, masks: Sequence[np.ndarray] | None = None, deterministic: bool = False):
        obs_arr = np.asarray(obs, dtype=np.float32)
        if not np.isfinite(obs_arr).all():
            raise FloatingPointError(f"{self.name}: non-finite observation")
        t = torch.as_tensor(obs_arr, device=self.device)
        a, logp, value, entropy = self.net.act(t, self._mask_tensors(masks), deterministic=deterministic)
        action = a.detach().cpu().numpy().astype(np.int64)
        if masks is not None:
            for i, m in enumerate(masks):
                if not bool(np.asarray(m)[action[i]]):
                    raise RuntimeError(f"{self.name}: sampled a masked action on head {i}")
        return action, float(logp), float(value), float(entropy)

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).unsqueeze(0)
        return float(self.net.value(t).squeeze(0).cpu())

    @torch.no_grad()
    def act_batch(self, observations, masks, deterministic=False, uniforms=None):
        if self.scheduling:
            raise ValueError("batched virtual rollout is for the fast policy only")
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.net.obs_dim or not np.isfinite(obs).all():
            raise ValueError("invalid batched observations")
        ts = torch.as_tensor(obs, device=self.device)
        ms = self._mask_tensors(masks, batch=len(obs))
        us = None
        if uniforms is not None:
            arr = np.asarray(uniforms, dtype=np.float64)
            if arr.shape != (len(obs), len(self.nvec)) or not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr >= 1):
                raise ValueError("uniforms must have shape (batch, heads) in [0,1)")
            us = torch.as_tensor(arr, device=self.device, dtype=torch.float64)
        a, lp, v, ent = self.net.act_batch(ts, ms, deterministic, us)
        if ms is not None and any(not bool(mask.gather(1, a[:, i:i+1]).all()) for i, mask in enumerate(ms)):
            raise RuntimeError("fast policy sampled a masked action")
        return a.cpu().numpy().astype(np.int64), lp.cpu().numpy(), v.cpu().numpy(), ent.cpu().numpy()

    def store(self, key: int, obs, masks, action, logp, value, reward, next_value, done) -> None:
        if not np.isfinite([logp, value, reward, next_value]).all():
            raise FloatingPointError("nonfinite transition")
        self.trajectories.setdefault(int(key), []).append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                mask=self.flatten_masks(masks, self.nvec),
                action=np.asarray(action, dtype=np.int64).copy(),
                logp=float(logp),
                value=float(value),
                reward=float(reward),
                next_value=float(next_value),
                done=bool(done),
            )
        )

    def buffer_size(self) -> int:
        return sum(len(t) for t in self.trajectories.values())

    # ------------------------------------------------------------------
    def update(self) -> dict[str, float]:
        cfg = self.cfg
        transitions: list[Transition] = []
        for traj in self.trajectories.values():
            compute_gae(traj, cfg.ppo_gamma, cfg.ppo_gae_lambda)
            transitions.extend(traj)
        self.trajectories = {}
        if not transitions:
            return {}

        obs = torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor(np.stack([t.action for t in transitions]), dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor([t.logp for t in transitions], dtype=torch.float32, device=self.device)
        old_values = np.asarray([t.value for t in transitions], dtype=np.float64)
        adv_np = np.asarray([t.advantage for t in transitions], dtype=np.float64)
        ret_np = np.asarray([t.return_value for t in transitions], dtype=np.float64)
        for nm, arr in (("advantages", adv_np), ("returns", ret_np)):
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"{self.name}: non-finite {nm}")
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + cfg.ppo_adv_eps)
        adv = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)
        mask_list = self._split_masks(masks)

        n = len(transitions)
        idx = np.arange(n)
        stats = {k: [] for k in ("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
        epochs_done, stopped_by_kl = 0, False
        for _ in range(cfg.ppo_update_epochs):
            np.random.shuffle(idx)
            epoch_kls = []
            for start in range(0, n, cfg.ppo_minibatch_size):
                mb = torch.as_tensor(idx[start:start + cfg.ppo_minibatch_size], device=self.device)
                mb_masks = [m.index_select(0, mb) for m in mask_list]
                logp, entropy, value = self.net.evaluate_actions(obs[mb], actions[mb], mb_masks)
                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip_ratio, 1.0 + cfg.ppo_clip_ratio) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (returns[mb] - value).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + cfg.ppo_value_coef * value_loss - cfg.ppo_entropy_coef * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{self.name}: non-finite PPO loss")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), cfg.ppo_max_grad_norm, error_if_nonfinite=True)
                self.opt.step()
                with torch.no_grad():
                    approx_kl = (old_logp[mb] - logp).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.ppo_clip_ratio).float().mean()
                stats["loss"].append(float(loss)); stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss)); stats["entropy"].append(float(entropy_mean))
                stats["approx_kl"].append(float(approx_kl)); stats["clip_fraction"].append(float(clip_frac))
                stats["grad_norm"].append(float(grad_norm))
                epoch_kls.append(float(approx_kl))
            epochs_done += 1
            if epoch_kls and float(np.mean(epoch_kls)) > cfg.ppo_target_kl:
                stopped_by_kl = True
                break

        self.update_count += 1
        ret_var = float(np.var(ret_np))
        explained = 1.0 - float(np.var(ret_np - old_values)) / ret_var if ret_var > 1e-12 else 0.0
        out = {f"{self.name}/{k}": float(np.mean(v)) for k, v in stats.items() if v}
        out.update({
            f"{self.name}/explained_variance": explained,
            f"{self.name}/transitions": float(n),
            f"{self.name}/epochs_completed": float(epochs_done),
            f"{self.name}/stopped_by_kl": float(stopped_by_kl),
            f"{self.name}/updates": float(self.update_count),
        })
        return out

    # ------------------------------------------------------------------
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        payload = {
            "architecture": ARCHITECTURE_VERSION,
            "name": self.name,
            "scheduling": self.scheduling,
            "nvec": self.nvec,
            "obs_dim": self.net.obs_dim,
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": asdict(self.cfg),
            "update_count": self.update_count,
            "extra": extra or {},
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load(self, path: str | Path, load_optimizer: bool = False) -> dict:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if payload.get("architecture") != ARCHITECTURE_VERSION:
            raise RuntimeError(f"incompatible checkpoint architecture {payload.get('architecture')!r}")
        if list(payload.get("nvec", [])) != self.nvec or int(payload.get("obs_dim", -1)) != self.net.obs_dim:
            raise RuntimeError("checkpoint action/observation geometry does not match this configuration")
        if payload.get("name") != self.name or payload.get("scheduling") != self.scheduling:
            raise RuntimeError("checkpoint policy role mismatch")
        # Training budgets/runtime switches may differ; physical/control semantics may not.
        ignored = {"seed", "episode_offset", "train_episodes", "eval_episodes", "save_every_episodes",
                   "device", "torch_num_threads", "hidden_dims", "slot_update_every_frames",
                   "frame_update_every_episodes", "write_jsonl_trace", "write_human_debug_log",
                   "log_hidden_csi", "log_observation_vectors", "console_log_every_slots"}
        def canonical(v):
            return tuple(canonical(x) for x in v) if isinstance(v, (tuple, list)) else v
        current = asdict(self.cfg)
        mismatch = [key for key, val in current.items()
                    if key not in ignored and not (key.startswith("ppo_") and key != "ppo_reward_scale")
                    and canonical(payload["config"].get(key)) != canonical(val)]
        if mismatch:
            raise RuntimeError("checkpoint config mismatch: " + ", ".join(mismatch))
        self.net.load_state_dict(payload["model"])
        if load_optimizer and "optimizer" in payload:
            self.opt.load_state_dict(payload["optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        return dict(payload.get("extra", {}))
