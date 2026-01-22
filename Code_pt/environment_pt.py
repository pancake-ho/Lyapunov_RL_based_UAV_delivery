# -*- coding: utf-8 -*-
"""
Paper-faithful environment (HRL video delivery, 논문식 완전 구현 전용)
- Rayleigh × Log-normal channel
- INR in dB -> linear
- Capacity C = W * log2(1 + SNR*|h|^2 / (1+INR_lin))
- Constraint: M*q <= C ; else transmission fails (M_eff=0)
- Queue: Q_{t+1} = max(Q_t - 1, 0) + M_eff
- Reward: k1*sqrt(M_eff) + k2*K - k3*1{stall} - k4*|K_l - K_p|
Interface kept for agent_pt.py: check_in_cell -> (idxs, node_state)
"""

from __future__ import annotations
import math, random
from typing import List, Tuple, Optional
import numpy as np


class Env:
    def __init__(
        self,
        # --- topology / mobility ---
        num_node: int = 50,
        field_x: float = 20.0,         # rectangle width (normalized)
        field_y: float = 2.0,          # rectangle height (normalized)
        user_radius: float = 1.0,      # cell radius (~50 m if 1.0 == 50 m)
        user_mobility: float = 0.04,   # per-small-step move in x
        # --- radio / channel ---
        path_loss_exponent: float = 3.0,
        Bandwidth: float = 1.0,        # normalized so that C is in "chunk units"
        transmit_SNR_dB: float = 30.0, # dB
        INR_db: float = 0.0,           # dB, external interference
        shadowing_sigma_dB: float = 2.0,  # log-normal sigma (dB)
        # --- time scales ---
        agent_T: int = 5,              # large step = 5 small steps
        horizon_small: int = 300,      # episode small-time length
        # --- M-q action space (paper: quality 1..3; M up to 9) ---
        max_quality: int = 3,
        max_M: int = 9,
        # --- reward weights (paper default) ---
        k1: float = 1.0,
        k2: float = 30.0,
        k3: float = 10.0,
        k4: float = 1.0,
        # --- misc ---
        seed: Optional[int] = 42,
        case_id: int = 1,              # caching case: 1/2/3
        tx_efficiency: float = 1.0,    # optional scaling (<=1)
    ):
        # RNG
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # store
        self.num_node = num_node
        self.field_x, self.field_y = field_x, field_y
        self.user_radius = user_radius
        self.user_mobility = user_mobility

        self.path_loss_exponent = path_loss_exponent
        self.Bandwidth = Bandwidth
        self.transmit_SNR_dB = transmit_SNR_dB
        self.transmit_SNR = 10.0 ** (transmit_SNR_dB / 10.0)
        self.INR_db = INR_db
        self.INR_lin = 10.0 ** (INR_db / 10.0)
        self.shadowing_sigma_dB = shadowing_sigma_dB

        self.agent_T = agent_T
        self.horizon_small = horizon_small

        self.max_quality = max_quality
        self.max_M = max_M
        # Build action list [(M,q)] including (0,0)
        self.Mq_list: List[Tuple[int, int]] = [(0, 0)]
        for q in range(1, max_quality + 1):
            for M in range(1, max_M + 1):
                self.Mq_list.append((M, q))

        # reward weights
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4

        self.tx_efficiency = tx_efficiency

        # Node placement (PPP ~ uniform over rectangle)
        self.node_position_x = np.random.uniform(0.0, field_x, size=num_node)
        self.node_position_y = np.random.uniform(0.0, field_y, size=num_node)

        # user trajectory (move along x, center y)
        self.user_position_y = field_y / 2.0
        # small-time indexed positions
        self.user_position_x_t = np.clip(
            np.linspace(0.0, field_x, horizon_small), 0.0, field_x
        )

        # distances[small_t][i]
        self.distances: List[List[float]] = []
        self._precompute_distances()

        # caching qualities per node (case)
        self.case_id = case_id
        self.node_quality: List[int] = self._assign_caching_by_case(case_id)

        # runtime variables
        self.reset()

    # ---------------- core helpers ----------------
    def _precompute_distances(self):
        self.distances.clear()
        for t in range(self.horizon_small):
            ux = self.user_position_x_t[t]
            dy = (self.node_position_y - self.user_position_y)
            dx = (self.node_position_x - ux)
            d = np.sqrt(dx * dx + dy * dy)
            # normalize by radius so that "in-cell" test is d_norm <= 1
            d_norm = (d / max(self.user_radius, 1e-9)).tolist()
            self.distances.append(d_norm)

    def _assign_caching_by_case(self, case_id: int) -> List[int]:
        """
        Assign max quality each node can serve (1..max_quality), i.i.d. by case distribution.
        Case1: [4/7, 2/7, 1/7]
        Case3: [1/3, 1/3, 1/3]
        Case2: [2/7, 4/7, 1/7]  (more mid-quality bias)
        """
        if self.max_quality != 3:
            # generalize: geometric fall-off default
            probs = np.array([0.6, 0.3, 0.1])
            probs = probs[: self.max_quality]
            probs = probs / probs.sum()
        else:
            if case_id == 1:
                probs = np.array([4/7, 2/7, 1/7])
            elif case_id == 2:
                probs = np.array([1/3, 1/3, 1/3])
            else:
                probs = np.array([1/7, 2/7, 4/7])
        quals = np.arange(1, self.max_quality + 1)
        return np.random.choice(quals, size=self.num_node, p=(probs / probs.sum())).tolist()

    # ---------------- episode control ----------------
    def reset(self):
        self.small_time = 0   # fine time
        self.large_time = 0   # decision slot counter (every agent_T)
        self.Q = 0            # playback buffer in chunks
        self.last_K = 0       # previous quality
        self.buffering_count = 0
        self.total_reward = 0.0
        self.ChannelCapacity_list: List[float] = [0.0 for _ in range(self.num_node)]
        return self._get_obs()

    def _get_obs(self):
        """State fed to higher-level S agent (example: [Q, last_K]) — keep minimal here."""
        return np.array([float(self.Q), float(self.last_K)], dtype=np.float32)

    # ---------------- visibility to agent ----------------
    def check_in_cell(self, large_time: int) -> Optional[Tuple[List[int], List[List[float]]]]:
        """
        Returns (idxs, node_state) for the 5 nearest nodes within radius (normalized distance<=1).
        node_state = [ [d1..d5], [q1..q5] ]
        Pads with (-1, d=1.0, q=1) if fewer than 5.
        """
        t = min(self.small_time, self.horizon_small - 1)
        dlist = self.distances[t]
        cand = [(i, dlist[i], self.node_quality[i]) for i in range(self.num_node) if dlist[i] <= 1.0]
        if not cand:
            return None
        cand.sort(key=lambda x: x[1])
        cand = cand[:5]
        while len(cand) < 5:
            cand.append((-1, 1.0, 1))
        idxs = [c[0] for c in cand]
        node_state = [
            [c[1] for c in cand],  # distances (normalized)
            [c[2] for c in cand],  # max qualities available
        ]
        return idxs, node_state

    def calculate_capacity(self, large_time: int):
        """
        Populate self.ChannelCapacity_list for current small_time index using paper channel.
        """
        t = min(self.small_time, self.horizon_small - 1)
        dlist = self.distances[t]
        caps: List[float] = []
        for i in range(self.num_node):
            d_norm = max(dlist[i], 1e-6)                 # normalized by radius
            d = d_norm * self.user_radius                # physical (relative) distance
            # log-normal shadowing (amplitude): w_dB ~ N(0, sigma), amp = 10^(w_dB/20)
            w_db = np.random.normal(0.0, self.shadowing_sigma_dB)
            shadow_amp = 10.0 ** (w_db / 20.0)
            # Rayleigh amplitude
            u = np.random.rayleigh(scale=1.0)
            # channel amplitude with path loss
            h_amp = (shadow_amp / (d ** self.path_loss_exponent)) * u
            gamma = (self.transmit_SNR * (h_amp ** 2)) / (1.0 + self.INR_lin)
            C = self.Bandwidth * math.log2(1.0 + gamma)
            C *= float(self.tx_efficiency)
            caps.append(C)
        self.ChannelCapacity_list = caps

    # ---------------- environment step ----------------
    def step(
        self,
        alpha: Optional[int],
        action_M: int,
        action_q: int,
        K_pred: int,
        K_learn: int,
    ):
        """
        One small-time step. Agent chooses a node alpha (index), M and q for download,
        and K_pred (predicted quality) / K_learn (actual played quality) for k4 term.
        - Enforce q <= node_quality[alpha]
        - Enforce M*q <= C[alpha]; else transmission fails (M_eff=0)
        - Queue/playback update and reward per paper
        Returns: obs, reward, done, info
        """
        # channel capacity for this small slot
        self.calculate_capacity(self.large_time)

        # stall indicator (before playout)
        stall_flag = 1 if self.Q <= 0 else 0

        M_eff, q_eff = 0, 0
        if alpha is not None and 0 <= alpha < self.num_node:
            # cap q by node's max quality
            q_cap = int(min(action_q, int(self.node_quality[alpha])))
            if q_cap > 0 and action_M > 0:
                need = float(action_M * q_cap)
                cap = float(self.ChannelCapacity_list[alpha])
                if need <= cap + 1e-9:
                    M_eff, q_eff = int(action_M), int(q_cap)
                else:
                    # capacity violated -> no delivery in this slot
                    M_eff, q_eff = 0, 0

        # playback (consume 1 chunk if available)
        if self.Q > 0:
            self.Q -= 1
        else:
            self.buffering_count += 1

        # arrival
        self.Q += M_eff

        # reward
        reward = (
            self.k1 * math.sqrt(max(M_eff, 0))
            + self.k2 * float(K_learn)
            - self.k3 * float(stall_flag)
            - self.k4 * abs(int(K_learn) - int(K_pred))
        )
        self.total_reward += reward
        self.last_K = int(K_learn)

        # time update
        self.small_time += 1
        if self.small_time % self.agent_T == 0:
            self.large_time += 1

        done = self.small_time >= self.horizon_small

        info = {
            "M_eff": M_eff,
            "q_eff": q_eff,
            "stall": stall_flag,
            "cap_alpha": float(self.ChannelCapacity_list[alpha]) if (alpha is not None and 0 <= alpha < self.num_node) else 0.0,
            "Q": int(self.Q),
            "buffering_count": int(self.buffering_count),
        }
        return self._get_obs(), float(reward), bool(done), info

    # ---------------- utility ----------------
    def set_case(self, case_id: int):
        self.case_id = int(case_id)
        self.node_quality = self._assign_caching_by_case(self.case_id)

    def snr_set_db(self, snr_db: float):
        self.transmit_SNR_dB = float(snr_db)
        self.transmit_SNR = 10.0 ** (self.transmit_SNR_dB / 10.0)

    def inr_set_db(self, inr_db: float):
        self.INR_db = float(inr_db)
        self.INR_lin = 10.0 ** (self.INR_db / 10.0)
