"""Fixed-weight NDTVS QoE adapted to the shared atomic-delivery simulator.

Eq. (6)/(7) use successful received segments (each chunk is one segment).
Eq. (9) is mapped to observed cumulative playback stall seconds; the simulator
does not retain per-segment download times. See README_NDTVS_QOE_KR.md.
"""
from __future__ import annotations

import math
import numpy as np

REWARD_VERSION = "ndtvs-fixed-paper-qoe-v2"
PSNR_DB = (34.0, 36.64, 39.11, 41.64)
CHUNK_DURATION_S = 1.0
QOE_WEIGHTS = (1.0, (PSNR_DB[-1] - PSNR_DB[0]) / 3.0,
               (PSNR_DB[-1] - PSNR_DB[0]) / CHUNK_DURATION_S)
REWARD_SCALE = 1.0 / PSNR_DB[-1]


def reward_spec():
    return {"version": REWARD_VERSION, "psnr_db": list(PSNR_DB),
            "weights": list(QOE_WEIGHTS), "training_scale": REWARD_SCALE,
            "chunk_duration_s": CHUNK_DURATION_S,
            "vq": "Eq6: first request PSNR minus mean signed gap over received segments 2..S",
            "qv": "mean absolute consecutive received segment quality-index difference",
            "re": "cumulative observed queue stall seconds, including unscheduled users",
            "s0": "Vq=0, Qv=0", "s1": "Vq=first received PSNR, Qv=0",
            "aggregation": "sum per-region QoE each slot; shared policy, regional GAE",
            "evaluation_quality": "received-chunk-weighted PSNR / 41.64",
            "coefficients_origin": "fixed designed exchange assumptions; not author-fitted weights"}


class QoEHistory:
    """Episode-scoped state indexed by user, never reset on frame/region moves."""
    def __init__(self, cfg):
        if cfg.num_quality_levels != len(PSNR_DB):
            raise ValueError("This PSNR ladder requires exactly four quality levels")
        if not math.isclose(cfg.playback_chunks_per_slot * CHUNK_DURATION_S,
                            cfg.slot_duration_s, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Common playback rate must match one-second segments")
        self.cfg = cfg
        N = cfg.num_users
        self.segments = np.zeros(N, dtype=np.int64)
        self.first_requested_psnr = np.full(N, np.nan)
        self.first_received_psnr = np.zeros(N)
        self.gap_sum = np.zeros(N)
        self.switch_sum = np.zeros(N)
        self.last_quality = np.full(N, -1, dtype=np.int64)
        self.rebuffer_s = np.zeros(N)
        self.vq = np.zeros(N)
        self.qv = np.zeros(N)
        self.qoe = np.zeros(N)
        self.received_psnr_sum = 0.0

    def update(self, rec):
        u, d, k = int(rec["user"]), int(rec["delivered"]), int(rec["req_quality"])
        if d < 0 or d != rec["delivered"] or not 0 <= k < len(PSNR_DB):
            raise ValueError("Invalid segment count or quality index")
        if rec["req_chunks"] > 0 and np.isnan(self.first_requested_psnr[u]):
            self.first_requested_psnr[u] = PSNR_DB[k]
        stall_s = max(self.cfg.playback_chunks_per_slot - float(rec["q_before"]), 0.0) * CHUNK_DURATION_S
        if not -1e-10 <= stall_s <= self.cfg.slot_duration_s + 1e-10:
            raise ValueError("Invalid playback stall duration")
        self.rebuffer_s[u] += stall_s
        switch_delta = 0.0
        if d:
            if np.isnan(self.first_requested_psnr[u]):
                raise ValueError("Delivered segment without a request")
            old_s = int(self.segments[u])
            if old_s:
                switch_delta = float(abs(k - self.last_quality[u]))
                tail_segments = d
            else:
                self.first_received_psnr[u] = PSNR_DB[k]
                tail_segments = d - 1
            # The d-1 within-batch transitions have zero magnitude, but count
            # in the S-1 denominator. Do not divide differences by K-1.
            self.switch_sum[u] += switch_delta
            self.gap_sum[u] += tail_segments * (self.first_requested_psnr[u] - PSNR_DB[k])
            self.segments[u] += d
            self.last_quality[u] = k
            self.received_psnr_sum += d * PSNR_DB[k]
        S = int(self.segments[u])
        if S >= 2:
            self.vq[u] = self.first_requested_psnr[u] - self.gap_sum[u] / (S - 1)
            self.qv[u] = self.switch_sum[u] / (S - 1)
        elif S == 1:
            self.vq[u] = self.first_received_psnr[u]
        b1, b2, b3 = QOE_WEIGHTS
        self.qoe[u] = b1 * self.vq[u] - b2 * self.qv[u] - b3 * self.rebuffer_s[u]
        return {"paper_qoe": float(self.qoe[u]), "paper_vq_db": float(self.vq[u]),
                "paper_qv_index": float(self.qv[u]), "rebuffer_s_cumulative": float(self.rebuffer_s[u]),
                "stall_duration_s": float(stall_s), "received_segments": S,
                "first_requested_psnr_db": (None if np.isnan(self.first_requested_psnr[u])
                                            else float(self.first_requested_psnr[u])),
                "quality_switch_sum": float(self.switch_sum[u]),
                "quality_switch_increment": switch_delta,
                "paper_reward_scaled": float(self.qoe[u] * REWARD_SCALE)}
