#!/usr/bin/env python3
"""
RSU-only PPO algorithmic baseline adapted from:
  Ladipo, Okegbile, Cai,
  "Network Digital Twin-enhanced QoE Optimization for Adaptive Video
   Streaming in 6G IoV Networks", VTC 2025-Fall.

Purpose in this project
-----------------------
This is NOT a paper-reproduction script.  It is an algorithmic baseline for
fair comparison with the proposed RSU-UAV-Vehicle formulation.

Professor-feedback adaptation:
  * GRU / multi-step bandwidth prediction is removed completely.
  * NDT prediction loss / predicted-bandwidth constraint is removed.
  * The common physical environment is synchronized to the current P3
    implementation: RSU-only topology, frame/slot timing, mobility, fixed
    reserved RSU resource groups, RSU channel model, video ladder, and
    playback-queue dynamics.
  * The baseline-specific control mechanism retained from Ladipo et al. is PPO
    based joint resource-user selection and video quality adaptation with a
    QoE-style reward.

Important fairness rule
-----------------------
Environment/application constants below are deliberately NOT exposed as CLI
arguments.  They are fixed so that a run cannot silently drift away from the
proposed P3 environment.  Runtime/training controls (seed, episodes, output,
checkpoint, device) remain configurable.

Synced against project branch `feat/new-form-p3`, `config_p3.py` / P3 radio and
mobility logic as inspected on 2026-09-08.  If the proposed environment is
changed later, update COMMON below at the same time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ============================================================================
# Fixed common environment: synchronized to the current proposed P3 code.
# ============================================================================
@dataclass(frozen=True)
class CommonScenario:
    # Time / topology
    num_regions: int = 2
    users_per_region: int = 4
    num_frames: int = 30
    frame_slots: int = 10
    slot_duration_s: float = 1.0
    region_length_m: float = 400.0

    # Heights / vehicle mobility
    rsu_height_m: float = 8.0
    user_height_m: float = 1.5
    vehicle_speed_min_mps: float = 5.0
    vehicle_speed_max_mps: float = 20.0

    # RSU radio: fixed resource groups W_R/J_R and P_R/J_R
    rsu_capacity: int = 3
    rsu_total_bandwidth_hz: float = 20e6
    rsu_total_power_w: float = 40.0
    noise_psd_w_hz: float = 2e-20
    shannon_gap: float = 2.0
    rsu_beta0: float = 1e-7
    rsu_pathloss_exp: float = 3.0

    # Match P3 fading trace generation: power fading ~ Exp(1), clipped.
    fading_min: float = 0.05
    fading_max: float = 10.0

    # Common application model
    quality_utility: tuple[float, ...] = (0.55, 0.72, 0.86, 1.0)
    chunk_size_bits: tuple[float, ...] = (0.5e6, 1e6, 2e6, 4e6)
    max_chunks_per_slot: int = 3
    playback_chunks_per_slot: float = 1.0
    initial_playback_queue: float = 3.0
    large_queue_level: float = 100.0

    @property
    def num_users(self) -> int:
        return self.num_regions * self.users_per_region

    @property
    def road_length_m(self) -> float:
        return self.num_regions * self.region_length_m

    @property
    def num_quality_levels(self) -> int:
        return len(self.quality_utility)

    @property
    def rsu_user_bandwidth_hz(self) -> float:
        return self.rsu_total_bandwidth_hz / self.rsu_capacity

    @property
    def rsu_user_power_w(self) -> float:
        return self.rsu_total_power_w / self.rsu_capacity

    @property
    def num_slots(self) -> int:
        return self.num_frames * self.frame_slots

    def rsu_x(self, region: int) -> float:
        return (float(region) + 0.5) * self.region_length_m

    def validate(self) -> None:
        if self.num_regions <= 0 or self.users_per_region <= 0:
            raise ValueError("num_regions and users_per_region must be positive")
        if self.rsu_capacity <= 0:
            raise ValueError("rsu_capacity must be positive")
        if not (0.0 < self.slot_duration_s):
            raise ValueError("slot_duration_s must be positive")
        if self.vehicle_speed_min_mps > self.vehicle_speed_max_mps:
            raise ValueError("vehicle speed range is invalid")
        if len(self.quality_utility) != len(self.chunk_size_bits):
            raise ValueError("quality_utility and chunk_size_bits must align")
        if tuple(sorted(self.quality_utility)) != self.quality_utility:
            raise ValueError("quality_utility must be nondecreasing")
        if tuple(sorted(self.chunk_size_bits)) != self.chunk_size_bits:
            raise ValueError("chunk_size_bits must be nondecreasing")
        if self.max_chunks_per_slot <= 0:
            raise ValueError("max_chunks_per_slot must be positive")
        if self.playback_chunks_per_slot <= 0.0:
            raise ValueError("playback_chunks_per_slot must be positive")
        if self.initial_playback_queue < 0.0:
            raise ValueError("initial_playback_queue cannot be negative")
        if not (0.0 < self.fading_min <= self.fading_max):
            raise ValueError("invalid fading clipping interval")


COMMON = CommonScenario()
COMMON.validate()


# ============================================================================
# Baseline-specific PPO / QoE settings.
# These are algorithm settings, not common physical-environment parameters.
# ============================================================================
@dataclass(frozen=True)
class BaselinePPOConfig:
    # Paper explicitly reports actor=1e-5, critic=1e-4, hidden=(128, 64),
    # mini-batch=64. Other PPO choices below are implementation choices.
    hidden_dims: tuple[int, ...] = (128, 64)
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50

    # Common fixed QoE surrogate after removing VUEDT beta fitting.
    # It preserves Ladipo's quality - variation - rebuffering structure while
    # using the project's common chunk/queue definitions.
    beta_quality: float = 1.0
    beta_switch: float = 0.5
    beta_stall: float = 2.0


PPOCFG = BaselinePPOConfig()


# ============================================================================
# Runtime arguments only. Environment values are intentionally not overridable.
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "RSU-only PPO baseline (Ladipo-adapted, GRU removed, P3-common env)"
    )
    p.add_argument("--episodes", type=int, default=1500,
                   help="PPO training episodes; paper reports 1500")
    p.add_argument("--eval_episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out_dir", default="runs/ndtv_rsu_only_ppo")
    p.add_argument("--log_slots", choices=["eval", "all", "none"], default="eval")
    p.add_argument("--log_text", action="store_true")
    p.add_argument("--checkpoint", default="",
                   help="Optional PPO checkpoint to load")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training and evaluate --checkpoint")
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


# ============================================================================
# Environment
# ============================================================================
class RSUOnlyVideoEnv:
    """Common RSU-only environment used by the PPO baseline.

    Time logic:
      * At the beginning of each frame, each user is associated with the region
        containing its current position.
      * That association is fixed throughout the frame, matching the project's
        slow/fast timing convention.
      * Positions and fading/capacity are updated every slot.

    Mobility logic follows current P3:
      x <- (x + v * Delta) mod road_length.
      Users move in the positive road direction; no reflection model is used.

    Channel logic follows current P3 RSU radio:
      g = beta0 * fading / d^alpha
      C = Wbar * log2(1 + Pbar*g/(Gamma*N0*Wbar))
      fading ~ Exp(mean=1), clipped to [0.05, 10].
    """

    def __init__(self, cfg: CommonScenario):
        self.cfg = cfg
        self.M = cfg.num_regions
        self.N = cfg.num_users
        self.K = cfg.num_quality_levels
        self.Wbar = cfg.rsu_user_bandwidth_hz
        self.Pbar = cfg.rsu_user_power_w
        self.rsu_x = np.asarray([cfg.rsu_x(m) for m in range(self.M)], dtype=np.float64)
        self.utility = np.asarray(cfg.quality_utility, dtype=np.float64)
        self.chunk_bits = np.asarray(cfg.chunk_size_bits, dtype=np.float64)
        self._rng: Optional[np.random.Generator] = None

    def reset(self, seed: int) -> None:
        cfg = self.cfg
        # Separate streams make initial topology reproducible independent of
        # how many fading samples are generated later.
        topo_rng = np.random.default_rng(int(seed))
        self._rng = np.random.default_rng(int(seed) + 17_171)

        self.x = np.empty(self.N, dtype=np.float64)
        for region in range(cfg.num_regions):
            start = region * cfg.region_length_m
            sl = slice(
                region * cfg.users_per_region,
                (region + 1) * cfg.users_per_region,
            )
            self.x[sl] = topo_rng.uniform(
                start + 0.05 * cfg.region_length_m,
                start + 0.95 * cfg.region_length_m,
                size=cfg.users_per_region,
            )
        self.v = topo_rng.uniform(
            cfg.vehicle_speed_min_mps,
            cfg.vehicle_speed_max_mps,
            self.N,
        )

        self.Q = np.full(self.N, cfg.initial_playback_queue, dtype=np.float64)
        self.k_last = np.full(self.N, -1, dtype=np.int64)  # 0-based quality index
        self.qoe_last = np.zeros(self.N, dtype=np.float64)
        self.utility_last = np.zeros(self.N, dtype=np.float64)
        self.last_stalled = np.zeros(self.N, dtype=bool)

        self.t = 0
        self.frame = -1
        self._start_frame()
        self._observe_channel()
        self._validate_state()

    def _start_frame(self) -> None:
        self.frame += 1
        region = np.floor(self.x / self.cfg.region_length_m).astype(np.int32)
        self.region = np.clip(region, 0, self.M - 1)
        self.members = [np.where(self.region == m)[0] for m in range(self.M)]

    def _observe_channel(self) -> None:
        if self._rng is None:
            raise RuntimeError("environment must be reset before stepping")
        cfg = self.cfg
        serving_x = self.rsu_x[self.region]
        horizontal = np.abs(self.x - serving_x)
        vertical = cfg.rsu_height_m - cfg.user_height_m
        self.distance_m = np.hypot(horizontal, vertical)

        self.fading = np.clip(
            self._rng.exponential(scale=1.0, size=self.N),
            cfg.fading_min,
            cfg.fading_max,
        )
        gain = cfg.rsu_beta0 * self.fading / np.maximum(self.distance_m, 1.0) ** cfg.rsu_pathloss_exp
        noise = cfg.shannon_gap * cfg.noise_psd_w_hz * self.Wbar
        snr = self.Pbar * gain / max(noise, 1e-30)
        self.capacity_bps = self.Wbar * np.log2(1.0 + snr)

        # Relative position to the associated RSU; observable mobility context.
        self.dx_rel = (self.x - serving_x) / cfg.region_length_m

        if not np.all(np.isfinite(self.capacity_bps)):
            raise RuntimeError("non-finite RSU capacity")

    def snapshot(self) -> dict:
        return {
            "x": self.x.copy(),
            "v": self.v.copy(),
            "region": self.region.copy(),
            "distance_m": self.distance_m.copy(),
            "fading": self.fading.copy(),
            "capacity_bps": self.capacity_bps.copy(),
            "Q": self.Q.copy(),
            "k_last": self.k_last.copy(),
            "qoe_last": self.qoe_last.copy(),
            "members": [m.copy() for m in self.members],
        }

    def _validate_actions(self, actions: list[list[tuple[int, int]]]) -> None:
        cfg = self.cfg
        if len(actions) != self.M:
            raise ValueError("actions must contain one list per RSU region")
        for m, region_actions in enumerate(actions):
            if len(region_actions) > cfg.rsu_capacity:
                raise ValueError("RSU capacity violation")
            members = set(int(x) for x in self.members[m])
            seen: set[int] = set()
            for user, q_token in region_actions:
                user = int(user)
                q_token = int(q_token)
                if user not in members:
                    raise ValueError(f"cross-region association: region={m}, user={user}")
                if user in seen:
                    raise ValueError(f"duplicate user assignment: region={m}, user={user}")
                if not 0 <= q_token <= self.K:
                    raise ValueError(f"quality token must be in [0,{self.K}]")
                seen.add(user)

    def step(self, actions: list[list[tuple[int, int]]]) -> tuple[np.ndarray, dict, bool]:
        """Execute one fast-timescale slot.

        Each action tuple is (user_id, q_token):
          q_token=0      -> selected resource group is intentionally idle
          q_token=1..K   -> use quality index q_token-1

        Number of delivered chunks is not an independent PPO action.  It is the
        largest feasible integer under the common RSU capacity, capped by
        L_max, matching the project flow.
        """
        self._validate_actions(actions)
        cfg = self.cfg

        chunks = np.zeros(self.N, dtype=np.int64)
        q_token = np.zeros(self.N, dtype=np.int64)
        scheduled = np.zeros(self.N, dtype=bool)

        for m, region_actions in enumerate(actions):
            del m  # validation already tied users to their region
            for user, token in region_actions:
                user = int(user)
                token = int(token)
                scheduled[user] = True
                q_token[user] = token
                if token == 0:
                    continue
                q_idx = token - 1
                feasible = int(
                    (self.capacity_bps[user] * cfg.slot_duration_s)
                    // self.chunk_bits[q_idx]
                )
                chunks[user] = min(cfg.max_chunks_per_slot, max(feasible, 0))

        delivered = chunks > 0
        q_idx_delivered = q_token - 1

        # Common physical playback queue.
        q_before = self.Q.copy()
        actual_departure = np.minimum(q_before, cfg.playback_chunks_per_slot)
        stalled = q_before < cfg.playback_chunks_per_slot
        q_after = q_before - actual_departure + chunks
        if np.any(q_after < -1e-9):
            raise RuntimeError("negative playback queue")

        # QoE surrogate adapted to the project's multi-chunk semantics.
        # Quality reward is proportional to delivered chunks; switching is paid
        # once when the delivered quality changes; stall is per user-slot.
        utility_per_chunk = np.zeros(self.N, dtype=np.float64)
        utility_per_chunk[delivered] = self.utility[q_idx_delivered[delivered]]
        utility_total = utility_per_chunk * chunks

        switch_mag = np.zeros(self.N, dtype=np.float64)
        has_previous = self.k_last >= 0
        switch_mask = delivered & has_previous
        if self.K > 1:
            switch_mag[switch_mask] = (
                np.abs(q_idx_delivered[switch_mask] - self.k_last[switch_mask])
                / float(self.K - 1)
            )

        stall_time = stalled.astype(np.float64) * cfg.slot_duration_s
        qoe = (
            PPOCFG.beta_quality * utility_total
            - PPOCFG.beta_switch * switch_mag
            - PPOCFG.beta_stall * stall_time
        )
        rewards = np.asarray(
            [float(qoe[ids].sum()) for ids in self.members],
            dtype=np.float64,
        )

        degradation = np.zeros(self.N, dtype=np.float64)
        degradation[delivered] = (
            (max(cfg.quality_utility) - utility_per_chunk[delivered])
            * chunks[delivered]
        )

        rec = {
            "scheduled": scheduled,
            "q_token": q_token,
            "quality_index": np.where(delivered, q_idx_delivered, -1),
            "chunks": chunks,
            "Q_before": q_before,
            "Q_next": q_after,
            "actual_departure": actual_departure,
            "stall": stalled,
            "stall_event": stalled & ~self.last_stalled,
            "utility_per_chunk": utility_per_chunk,
            "utility_total": utility_total,
            "degradation": degradation,
            "switch_mag": switch_mag,
            "qoe": qoe,
        }

        # State transition.
        self.Q = q_after
        self.k_last = np.where(delivered, q_idx_delivered, self.k_last)
        self.qoe_last = qoe
        self.utility_last = utility_per_chunk
        self.last_stalled = stalled

        self.x = np.mod(
            self.x + self.v * cfg.slot_duration_s,
            cfg.road_length_m,
        )
        self.t += 1
        done = self.t >= cfg.num_slots
        if not done:
            if self.t % cfg.frame_slots == 0:
                self._start_frame()
            self._observe_channel()
            self._validate_state()
        return rewards, rec, done

    def _validate_state(self) -> None:
        if self.x.shape != (self.N,) or self.Q.shape != (self.N,):
            raise RuntimeError("state shape mismatch")
        if np.any(self.x < 0.0) or np.any(self.x >= self.cfg.road_length_m + 1e-9):
            raise RuntimeError("user position outside road")
        if np.any(self.Q < -1e-9) or not np.all(np.isfinite(self.Q)):
            raise RuntimeError("invalid playback queue")
        if np.any(self.k_last < -1) or np.any(self.k_last >= self.K):
            raise RuntimeError("invalid last-quality index")


# ============================================================================
# PPO model
# ============================================================================
def mlp(sizes: list[int] | tuple[int, ...], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor/critic encoders so reported actor/critic LRs are real.

    Actor per region:
      1) sequentially selects at most J_R distinct users;
      2) for each selected user, chooses q_token in {0,1,...,K}.
         q_token=0 is idle, allowing the effective service set to be < J_R.

    Resource allocation is therefore represented by fixed-RB user selection,
    which is the project-consistent replacement for the source paper's
    bandwidth-allocation variable.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...], K: int):
        super().__init__()
        H = hidden[-1]

        # Actor
        self.actor_phi = mlp([obs_dim, *hidden])
        self.actor_ctx = mlp([H + 1, H])
        self.actor_mix = mlp([2 * H, H, H])
        self.user_score = nn.Linear(H, 1)
        self.quality_head = nn.Linear(H, K + 1)

        # Critic
        self.critic_phi = mlp([obs_dim, *hidden])
        self.critic_ctx = mlp([H + 1, H, H])
        self.value_head = nn.Linear(H, 1)

    def actor_parameters(self):
        modules = [
            self.actor_phi,
            self.actor_ctx,
            self.actor_mix,
            self.user_score,
            self.quality_head,
        ]
        for module in modules:
            yield from module.parameters()

    def critic_parameters(self):
        modules = [self.critic_phi, self.critic_ctx, self.value_head]
        for module in modules:
            yield from module.parameters()

    def actor_encode(self, obs: torch.Tensor, mask: torch.Tensor, load: torch.Tensor):
        e = torch.tanh(self.actor_phi(obs))
        m = mask.unsqueeze(-1)
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        ctx = torch.tanh(self.actor_ctx(torch.cat([pooled, load.unsqueeze(-1)], dim=-1)))
        h = torch.tanh(
            self.actor_mix(torch.cat([e, ctx.unsqueeze(1).expand_as(e)], dim=-1))
        )
        return h

    def critic_value(self, obs: torch.Tensor, mask: torch.Tensor, load: torch.Tensor):
        e = torch.tanh(self.critic_phi(obs))
        m = mask.unsqueeze(-1)
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        ctx = torch.tanh(self.critic_ctx(torch.cat([pooled, load.unsqueeze(-1)], dim=-1)))
        return self.value_head(ctx).squeeze(-1)


def run_policy(
    net: ActorCritic,
    obs: torch.Tensor,
    mask: torch.Tensor,
    load: torch.Tensor,
    J: int,
    picks: Optional[torch.Tensor] = None,
    quals: Optional[torch.Tensor] = None,
    deterministic: bool = False,
):
    h = net.actor_encode(obs, mask, load)
    scores = net.user_score(h).squeeze(-1)
    qlogits = net.quality_head(h)
    value = net.critic_value(obs, mask, load)

    B, _N = mask.shape
    avail = mask.bool().clone()
    # When #users <= J, all users receive one reserved candidate group, so user
    # ordering is not a stochastic resource-allocation decision.
    need_sample = mask.sum(1) > J

    out_p = torch.full((B, J), -1, dtype=torch.long, device=obs.device)
    out_q = torch.full((B, J), -1, dtype=torch.long, device=obs.device)
    logp = torch.zeros(B, device=obs.device)
    entropy = torch.zeros(B, device=obs.device)
    ar = torch.arange(B, device=obs.device)

    for j in range(J):
        active = avail.any(1)
        if not bool(active.any()):
            break
        logits = scores.masked_fill(~avail, -1e9)
        d_user = Categorical(logits=logits)

        if picks is not None:
            idx = picks[:, j].clamp(min=0)
        elif deterministic:
            idx = logits.argmax(1)
        else:
            idx = d_user.sample()

        select_term = active & need_sample
        logp = logp + torch.where(
            select_term,
            d_user.log_prob(idx),
            torch.zeros_like(logp),
        )
        entropy = entropy + torch.where(
            select_term,
            d_user.entropy(),
            torch.zeros_like(entropy),
        )

        d_quality = Categorical(logits=qlogits[ar, idx])
        if quals is not None:
            qq = quals[:, j].clamp(min=0)
        elif deterministic:
            qq = qlogits[ar, idx].argmax(1)
        else:
            qq = d_quality.sample()

        logp = logp + torch.where(
            active,
            d_quality.log_prob(qq),
            torch.zeros_like(logp),
        )
        entropy = entropy + torch.where(
            active,
            d_quality.entropy(),
            torch.zeros_like(entropy),
        )

        out_p[:, j] = torch.where(active, idx, torch.full_like(idx, -1))
        out_q[:, j] = torch.where(active, qq, torch.full_like(qq, -1))
        avail[ar, idx] = avail[ar, idx] & ~active

    return out_p, out_q, logp, entropy, value


# ============================================================================
# Observation
# ============================================================================
OBS_DIM = 10


def build_obs(env: RSUOnlyVideoEnv) -> np.ndarray:
    """Current-observation state; no future-bandwidth information exists.

    Per-user features:
      0 current RSU capacity (log normalized)
      1 playback queue / Q_large
      2 virtual shortage Z/Q_large = (Q_large-Q)/Q_large
      3 current stall indicator
      4 previous quality index normalized
      5 previous per-slot QoE (bounded scale)
      6 previous delivered utility per chunk
      7 position relative to frame-associated RSU
      8 normalized velocity
      9 normalized absolute RSU distance
    """
    cfg = env.cfg
    z = cfg.large_queue_level - env.Q
    last_q = np.where(env.k_last >= 0, (env.k_last + 1) / env.K, 0.0)
    # asinh keeps the feature finite and approximately linear near zero.
    qoe_feature = np.arcsinh(env.qoe_last) / 3.0
    max_dist = math.hypot(cfg.region_length_m, cfg.rsu_height_m - cfg.user_height_m)
    return np.stack(
        [
            np.log1p(env.capacity_bps / 1e6) / 4.0,
            env.Q / cfg.large_queue_level,
            z / cfg.large_queue_level,
            (env.Q < cfg.playback_chunks_per_slot).astype(np.float64),
            last_q,
            qoe_feature,
            env.utility_last,
            env.dx_rel,
            env.v / cfg.vehicle_speed_max_mps,
            env.distance_m / max(max_dist, 1e-9),
        ],
        axis=1,
    ).astype(np.float32)


# ============================================================================
# PPO utilities
# ============================================================================
def gae(rewards: list[float], values: list[float], gamma: float, lam: float):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float64)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def ppo_update(net: ActorCritic, opt: torch.optim.Optimizer, buf: dict, device: torch.device):
    obs = torch.tensor(np.asarray(buf["obs"]), dtype=torch.float32, device=device)
    mask = torch.tensor(np.asarray(buf["mask"]), dtype=torch.float32, device=device)
    load = torch.tensor(np.asarray(buf["load"]), dtype=torch.float32, device=device)
    picks = torch.tensor(np.asarray(buf["picks"]), dtype=torch.long, device=device)
    quals = torch.tensor(np.asarray(buf["quals"]), dtype=torch.long, device=device)
    old_logp = torch.tensor(np.asarray(buf["logp"]), dtype=torch.float32, device=device)
    old_value = torch.tensor(np.asarray(buf["value"]), dtype=torch.float32, device=device)
    adv = torch.tensor(np.asarray(buf["adv"]), dtype=torch.float32, device=device)
    ret = torch.tensor(np.asarray(buf["ret"]), dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    indices = np.arange(len(obs))

    rows = []
    for _ in range(PPOCFG.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), PPOCFG.minibatch_size):
            mb = indices[start:start + PPOCFG.minibatch_size]
            _, _, logp, ent, value = run_policy(
                net,
                obs[mb],
                mask[mb],
                load[mb],
                COMMON.rsu_capacity,
                picks[mb],
                quals[mb],
                deterministic=False,
            )
            log_ratio = logp - old_logp[mb]
            ratio = log_ratio.exp()
            unclipped = ratio * adv[mb]
            clipped = ratio.clamp(
                1.0 - PPOCFG.clip_ratio,
                1.0 + PPOCFG.clip_ratio,
            ) * adv[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (ret[mb] - value).pow(2).mean()
            entropy = ent.mean()
            loss = (
                policy_loss
                + PPOCFG.value_coef * value_loss
                - PPOCFG.entropy_coef * entropy
            )

            if not torch.isfinite(loss):
                raise RuntimeError("non-finite PPO loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(net.parameters(), PPOCFG.max_grad_norm))
            opt.step()

            with torch.no_grad():
                approx_kl = float((old_logp[mb] - logp).mean().cpu())
                clipfrac = float(((ratio - 1.0).abs() > PPOCFG.clip_ratio).float().mean().cpu())
            rows.append(
                (
                    float(policy_loss.item()),
                    float(value_loss.item()),
                    float(entropy.item()),
                    approx_kl,
                    clipfrac,
                    grad_norm,
                )
            )

    arr = np.asarray(rows, dtype=np.float64)
    ev = explained_variance(
        old_value.detach().cpu().numpy(),
        ret.detach().cpu().numpy(),
    )
    return {
        "policy_loss": float(arr[:, 0].mean()),
        "value_loss": float(arr[:, 1].mean()),
        "entropy": float(arr[:, 2].mean()),
        "approx_kl": float(arr[:, 3].mean()),
        "clipfrac": float(arr[:, 4].mean()),
        "grad_norm": float(arr[:, 5].mean()),
        "explained_variance_preupdate": ev,
    }


# ============================================================================
# Logging / metrics
# ============================================================================
class SlotLogger:
    def __init__(self, jsonl_path: Path, txt_path: Optional[Path] = None):
        self.fj = jsonl_path.open("w", encoding="utf-8")
        self.ft = txt_path.open("w", encoding="utf-8") if txt_path else None

    def log(self, episode: int, snap: dict, rec: dict, rewards: np.ndarray, env: RSUOnlyVideoEnv):
        cfg = env.cfg
        regions = []
        for m, ids in enumerate(snap["members"]):
            users = []
            for raw_n in ids:
                n = int(raw_n)
                users.append(
                    {
                        "id": n,
                        "x_m": float(snap["x"][n]),
                        "v_mps": float(snap["v"][n]),
                        "distance_m": float(snap["distance_m"][n]),
                        "fading": float(snap["fading"][n]),
                        "capacity_mbps": float(snap["capacity_bps"][n] / 1e6),
                        "Q": float(snap["Q"][n]),
                        "Z": float(cfg.large_queue_level - snap["Q"][n]),
                        "last_quality_index": int(snap["k_last"][n]),
                        "scheduled": int(rec["scheduled"][n]),
                        "q_token": int(rec["q_token"][n]),
                        "quality_index": int(rec["quality_index"][n]),
                        "chunks": int(rec["chunks"][n]),
                        "Q_next": float(rec["Q_next"][n]),
                        "stall": int(rec["stall"][n]),
                        "stall_event": int(rec["stall_event"][n]),
                        "utility_total": float(rec["utility_total"][n]),
                        "degradation": float(rec["degradation"][n]),
                        "switch_mag": float(rec["switch_mag"][n]),
                        "qoe": float(rec["qoe"][n]),
                    }
                )
            regions.append(
                {
                    "region": m,
                    "n_users": len(ids),
                    "reward": float(rewards[m]),
                    "users": users,
                }
            )

        record = {
            "episode": int(episode),
            "frame": int(env.frame),
            "slot": int(env.t - 1),
            "slot_in_frame": int((env.t - 1) % cfg.frame_slots),
            "regions": regions,
        }
        self.fj.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.ft:
            self.ft.write(
                f"[ep={episode} frame={record['frame']} slot={record['slot']} "
                f"in_frame={record['slot_in_frame']}]\n"
            )
            for rg in regions:
                self.ft.write(
                    f"  RSU{rg['region']} |N|={rg['n_users']} J_R={cfg.rsu_capacity} "
                    f"reward={rg['reward']:+.4f}\n"
                )
                for u in rg["users"]:
                    self.ft.write(
                        f"    u{u['id']:02d} x={u['x_m']:7.1f} v={u['v_mps']:5.1f} "
                        f"d={u['distance_m']:6.1f} C={u['capacity_mbps']:8.3f} "
                        f"Q={u['Q']:6.2f} Z={u['Z']:7.2f} "
                        f"sched={u['scheduled']} qtok={u['q_token']} chunks={u['chunks']} "
                        f"Q'={u['Q_next']:6.2f} stall={u['stall']} "
                        f"util={u['utility_total']:.3f} deg={u['degradation']:.3f} "
                        f"switch={u['switch_mag']:.3f} qoe={u['qoe']:+.3f}\n"
                    )
            self.ft.write("\n")

    def close(self):
        self.fj.close()
        if self.ft:
            self.ft.close()


class Metrics:
    def __init__(self, cfg: CommonScenario):
        self.cfg = cfg
        self.user_slots = 0
        self.qoe = 0.0
        self.stalls = 0.0
        self.stall_events = 0.0
        self.scheduled = 0.0
        self.served = 0.0
        self.chunks = 0.0
        self.utility = 0.0
        self.degradation = 0.0
        self.quality_level_sum = 0.0
        self.switch_user_slots = 0.0
        self.switch_mag = 0.0
        self.quality_transitions = 0.0
        self.queue_sum = 0.0
        self.large_q_exceed = 0.0
        self.max_queue = 0.0

    def update(self, rec: dict) -> None:
        n = len(rec["qoe"])
        self.user_slots += n
        self.qoe += float(np.sum(rec["qoe"]))
        self.stalls += float(np.sum(rec["stall"]))
        self.stall_events += float(np.sum(rec["stall_event"]))
        self.scheduled += float(np.sum(rec["scheduled"]))
        served = rec["chunks"] > 0
        self.served += float(np.sum(served))
        self.chunks += float(np.sum(rec["chunks"]))
        self.utility += float(np.sum(rec["utility_total"]))
        self.degradation += float(np.sum(rec["degradation"]))
        if np.any(served):
            self.quality_level_sum += float(
                np.sum((rec["quality_index"][served] + 1) * rec["chunks"][served])
            )
        switched = rec["switch_mag"] > 0.0
        self.switch_user_slots += float(np.sum(switched))
        self.switch_mag += float(np.sum(rec["switch_mag"]))
        # Any delivered sample with an existing previous quality manifests either
        # zero or non-zero switch. Infer transitions from delivered & not first.
        # We approximate by counting delivered slots after the user's first delivery
        # through a separate field is not available here; switch_user_slot_rate is
        # therefore the primary reported switching metric.
        self.queue_sum += float(np.sum(rec["Q_next"]))
        self.large_q_exceed += float(np.sum(rec["Q_next"] > self.cfg.large_queue_level))
        self.max_queue = max(self.max_queue, float(np.max(rec["Q_next"])))

    def summary(self) -> dict[str, float]:
        n = max(self.user_slots, 1)
        chunks = max(self.chunks, 1.0)
        return {
            "avg_qoe_per_user_slot": self.qoe / n,
            "stall_ratio": self.stalls / n,
            "stall_events_per_user_slot": self.stall_events / n,
            "scheduled_ratio": self.scheduled / n,
            "service_rate": self.served / n,
            "chunks_per_user_slot": self.chunks / n,
            "quality_utility_per_chunk": self.utility / chunks,
            "degradation_per_chunk": self.degradation / chunks,
            "avg_quality_level_per_chunk": self.quality_level_sum / chunks,
            "switch_user_slot_rate": self.switch_user_slots / n,
            "avg_switch_magnitude_per_user_slot": self.switch_mag / n,
            "avg_playback_queue": self.queue_sum / n,
            "large_Q_exceed_rate": self.large_q_exceed / n,
            "max_playback_queue": self.max_queue,
        }


# ============================================================================
# Episode runner
# ============================================================================
def run_episode(
    env: RSUOnlyVideoEnv,
    net: ActorCritic,
    device: torch.device,
    episode_seed: int,
    deterministic: bool,
    logger: Optional[SlotLogger] = None,
    episode_index: int = 0,
):
    env.reset(episode_seed)
    cfg = env.cfg
    keep_buffer = not deterministic

    buf = {
        k: [[] for _ in range(cfg.num_regions)]
        for k in ["obs", "mask", "load", "picks", "quals", "logp", "value", "reward"]
    }
    metrics = Metrics(cfg)
    done = False

    while not done:
        snap = env.snapshot()
        full_obs = build_obs(env)
        if full_obs.shape != (cfg.num_users, OBS_DIM):
            raise RuntimeError(f"observation shape mismatch: {full_obs.shape}")
        if not np.all(np.isfinite(full_obs)):
            raise RuntimeError("non-finite observation")

        obs = np.zeros(
            (cfg.num_regions, cfg.num_users, OBS_DIM),
            dtype=np.float32,
        )
        mask = np.zeros((cfg.num_regions, cfg.num_users), dtype=np.float32)
        load = np.zeros(cfg.num_regions, dtype=np.float32)

        for m in range(cfg.num_regions):
            ids = env.members[m]
            obs[m, : len(ids)] = full_obs[ids]
            mask[m, : len(ids)] = 1.0
            load[m] = len(ids) / float(cfg.rsu_capacity)

        with torch.no_grad():
            picks_t, quals_t, logp_t, _ent_t, value_t = run_policy(
                net,
                torch.tensor(obs, dtype=torch.float32, device=device),
                torch.tensor(mask, dtype=torch.float32, device=device),
                torch.tensor(load, dtype=torch.float32, device=device),
                cfg.rsu_capacity,
                deterministic=deterministic,
            )
        picks = picks_t.cpu().numpy()
        quals = quals_t.cpu().numpy()

        actions: list[list[tuple[int, int]]] = []
        for m in range(cfg.num_regions):
            local_members = env.members[m]
            region_actions: list[tuple[int, int]] = []
            for local_idx, qtok in zip(picks[m], quals[m]):
                if local_idx < 0:
                    continue
                if local_idx >= len(local_members):
                    raise RuntimeError("policy selected padded user index")
                region_actions.append((int(local_members[local_idx]), int(qtok)))
            actions.append(region_actions)

        rewards, rec, done = env.step(actions)

        if keep_buffer:
            for m in range(cfg.num_regions):
                buf["obs"][m].append(obs[m])
                buf["mask"][m].append(mask[m])
                buf["load"][m].append(load[m])
                buf["picks"][m].append(picks[m])
                buf["quals"][m].append(quals[m])
                buf["logp"][m].append(float(logp_t[m].cpu()))
                buf["value"][m].append(float(value_t[m].cpu()))
                buf["reward"][m].append(float(rewards[m]))

        metrics.update(rec)
        if logger:
            logger.log(episode_index, snap, rec, rewards, env)

    flat = None
    if keep_buffer:
        flat = {
            k: []
            for k in [
                "obs", "mask", "load", "picks", "quals", "logp", "value", "adv", "ret"
            ]
        }
        for m in range(cfg.num_regions):
            adv, ret = gae(
                buf["reward"][m],
                buf["value"][m],
                PPOCFG.gamma,
                PPOCFG.gae_lambda,
            )
            for k in ["obs", "mask", "load", "picks", "quals", "logp", "value"]:
                flat[k].extend(buf[k][m])
            flat["adv"].extend(adv.tolist())
            flat["ret"].extend(ret.tolist())

    return metrics.summary(), flat


# ============================================================================
# Main / checkpoints
# ============================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is unavailable; refusing silent CPU fallback"
            )
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def checkpoint_payload(net: ActorCritic, args: argparse.Namespace, episode: int) -> dict:
    return {
        "model_state_dict": net.state_dict(),
        "episode": int(episode),
        "runtime_args": vars(args),
        "common_scenario": asdict(COMMON),
        "ppo_config": asdict(PPOCFG),
        "obs_dim": OBS_DIM,
    }


def load_checkpoint(net: ActorCritic, path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    net.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


def main() -> None:
    args = get_args()
    if args.episodes <= 0 and not args.eval_only:
        raise ValueError("--episodes must be positive for training")
    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive")

    device = resolve_device(args.device)
    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "common_scenario.json", asdict(COMMON))
    save_json(out_dir / "ppo_config.json", asdict(PPOCFG))
    save_json(out_dir / "run_args.json", vars(args))

    print("[CONFIG] GRU/predicted bandwidth: REMOVED")
    print(
        "[CONFIG] common env: "
        f"regions={COMMON.num_regions} users={COMMON.num_users} "
        f"J_R={COMMON.rsu_capacity} W_R={COMMON.rsu_total_bandwidth_hz/1e6:.1f}MHz "
        f"P_R={COMMON.rsu_total_power_w:.1f}W frames={COMMON.num_frames} "
        f"T={COMMON.frame_slots}"
    )
    print(
        "[CONFIG] PPO: "
        f"hidden={PPOCFG.hidden_dims} actor_lr={PPOCFG.actor_lr:g} "
        f"critic_lr={PPOCFG.critic_lr:g} device={device}"
    )

    env = RSUOnlyVideoEnv(COMMON)
    net = ActorCritic(OBS_DIM, PPOCFG.hidden_dims, COMMON.num_quality_levels).to(device)

    actor_params = list(net.actor_parameters())
    critic_params = list(net.critic_parameters())
    if {id(x) for x in actor_params} & {id(x) for x in critic_params}:
        raise RuntimeError("actor/critic parameter groups overlap")
    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": PPOCFG.actor_lr},
            {"params": critic_params, "lr": PPOCFG.critic_lr},
        ]
    )

    if args.checkpoint:
        ckpt = load_checkpoint(net, args.checkpoint, device)
        print(f"[CHECKPOINT] loaded {args.checkpoint} episode={ckpt.get('episode', 'unknown')}")
    elif args.eval_only:
        raise ValueError("--eval_only requires --checkpoint")

    # --------------------------------------------------------------------- train
    if not args.eval_only:
        train_csv = out_dir / "train_log.csv"
        fields = [
            "episode",
            "avg_qoe_per_user_slot",
            "stall_ratio",
            "stall_events_per_user_slot",
            "scheduled_ratio",
            "service_rate",
            "chunks_per_user_slot",
            "quality_utility_per_chunk",
            "degradation_per_chunk",
            "avg_quality_level_per_chunk",
            "switch_user_slot_rate",
            "avg_playback_queue",
            "large_Q_exceed_rate",
            "max_playback_queue",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clipfrac",
            "grad_norm",
            "explained_variance_preupdate",
            "elapsed_sec",
        ]
        with train_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            start = time.time()

            for ep in range(args.episodes):
                logger = None
                if args.log_slots == "all":
                    logger = SlotLogger(
                        out_dir / f"slots_train_ep{ep:04d}.jsonl",
                        out_dir / f"slots_train_ep{ep:04d}.txt" if args.log_text else None,
                    )

                # Episode-specific seed; reproducible and independent of eval.
                episode_seed = args.seed + ep
                metrics, buf = run_episode(
                    env,
                    net,
                    device,
                    episode_seed=episode_seed,
                    deterministic=False,
                    logger=logger,
                    episode_index=ep,
                )
                if logger:
                    logger.close()
                if buf is None:
                    raise RuntimeError("training rollout buffer missing")
                ppo_stats = ppo_update(net, optimizer, buf, device)

                row = {
                    "episode": ep,
                    **metrics,
                    **ppo_stats,
                    "elapsed_sec": time.time() - start,
                }
                writer.writerow({k: row[k] for k in fields})
                f.flush()

                if ep % args.print_every == 0 or ep == args.episodes - 1:
                    print(
                        f"[TRAIN] ep={ep:4d} "
                        f"qoe={metrics['avg_qoe_per_user_slot']:+.4f} "
                        f"stall={metrics['stall_ratio']:.4f} "
                        f"service={metrics['service_rate']:.4f} "
                        f"quality={metrics['quality_utility_per_chunk']:.4f} "
                        f"deg={metrics['degradation_per_chunk']:.4f} "
                        f"KL={ppo_stats['approx_kl']:.5f} "
                        f"ent={ppo_stats['entropy']:.4f} "
                        f"t={time.time()-start:.1f}s"
                    )

        torch.save(
            checkpoint_payload(net, args, args.episodes - 1),
            out_dir / "ppo_final.pt",
        )
        print(f"[CHECKPOINT] saved {out_dir / 'ppo_final.pt'}")

    # --------------------------------------------------------------------- eval
    results = []
    # Separate evaluation seed namespace so results do not depend on number of
    # training episodes or RNG consumption during PPO updates.
    eval_seed_base = args.seed + 1_000_000
    for ep in range(args.eval_episodes):
        logger = None
        if args.log_slots != "none":
            logger = SlotLogger(
                out_dir / f"slots_eval_ep{ep:02d}.jsonl",
                out_dir / f"slots_eval_ep{ep:02d}.txt" if args.log_text else None,
            )
        metrics, _ = run_episode(
            env,
            net,
            device,
            episode_seed=eval_seed_base + ep,
            deterministic=True,
            logger=logger,
            episode_index=ep,
        )
        if logger:
            logger.close()
        results.append(metrics)
        print(
            f"[EVAL] ep={ep} "
            + " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
        )

    keys = list(results[0])
    mean = {k: float(np.mean([r[k] for r in results])) for k in keys}
    std = {k: float(np.std([r[k] for r in results], ddof=0)) for k in keys}
    save_json(
        out_dir / "eval_metrics.json",
        {
            "per_episode": results,
            "mean": mean,
            "std": std,
            "eval_seed_base": eval_seed_base,
        },
    )
    print("[EVAL-MEAN] " + " ".join(f"{k}={v:.5f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
