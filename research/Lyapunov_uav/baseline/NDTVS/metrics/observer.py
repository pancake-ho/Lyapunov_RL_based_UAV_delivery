"""QoE observer and logging; one history update per user-slot."""
from __future__ import annotations

import numpy as np
from baseline.NDTVS.common.paths import HERE
from hppo.logger import HistoryLogger
from baseline.NDTVS.rewards.qoe import QoEHistory, PSNR_DB


class QoELogger(HistoryLogger):
    """One shared observer; state updates once per slot before NDTVS storage."""
    def __init__(self, cfg, root):
        super().__init__(cfg, root)
        self.history = QoEHistory(cfg)
        self.qoe = self.vq = self.qv = self.re = self.stall_seconds = 0.0
        self.failures = self.requests = 0.0
        self.user_slots = self.stall_events = 0
        self.last_qoe = np.zeros(cfg.num_users)
        self.last_utility = np.zeros(cfg.num_users)
        self.last_stall = np.zeros(cfg.num_users, dtype=bool)
        self._observed_key = None
        self._observed_info = None

    def observe_slot(self, info):
        key = (info["episode"], info["frame"], info["slot_in_frame"])
        if key == self._observed_key:
            if info is not self._observed_info:
                raise ValueError("Duplicate slot represented by a different record")
            return
        users = [u for rg in info["regions"].values() for u in rg["users"]]
        if sorted(u["user"] for u in users) != list(range(self.cfg.num_users)):
            raise ValueError("QoE observer requires exactly one record for every user")
        for u in users:
            terms = self.history.update(u)
            u.update(terms)
            i = u["user"]
            self.qoe += terms["paper_qoe"]
            self.vq += terms["paper_vq_db"]
            self.qv += terms["paper_qv_index"]
            self.re += terms["rebuffer_s_cumulative"]
            self.stall_seconds += terms["stall_duration_s"]
            self.requests += u["req_chunks"] > 0
            self.failures += u["transmission_failed"]
            self.user_slots += 1
            self.stall_events += bool(u["stall"] and not self.last_stall[i])
            self.last_stall[i] = bool(u["stall"])
            self.last_qoe[i] = terms["paper_reward_scaled"]
            self.last_utility[i] = terms["paper_vq_db"] / PSNR_DB[-1]
        self._observed_key, self._observed_info = key, info

    def log_slot(self, info, **kwargs):
        self.observe_slot(info)
        super().log_slot(info, **kwargs)

    def measures(self):
        h, us = self.history, max(self.user_slots, 1)
        S = int(h.segments.sum())
        mean_psnr = h.received_psnr_sum / S if S else 0.0
        transitions = int(np.maximum(h.segments - 1, 0).sum())
        return {"paper_qoe_per_user_slot": self.qoe / us,
                "paper_vq_per_user_slot": self.vq / us,
                "paper_qv_per_user_slot": self.qv / us,
                "cumulative_rebuffer_s_per_user_slot": self.re / us,
                "paper_qoe_final_per_user": float(h.qoe.mean()),
                "paper_vq_final_per_user": float(h.vq.mean()),
                "paper_qv_final_per_user": float(h.qv.mean()),
                "rebuffer_s_per_user": float(h.rebuffer_s.mean()),
                "stall_time_ratio": self.stall_seconds / (us * self.cfg.slot_duration_s),
                "average_received_psnr_db": mean_psnr,
                "average_quality_utility": mean_psnr / PSNR_DB[-1],
                "quality_utility_defined": bool(S),
                "received_segments_total": S,
                "quality_switch_per_received_transition": float(h.switch_sum.sum()) / max(transitions, 1),
                "stall_events_per_user_slot": self.stall_events / us,
                "request_failure_ratio": self.failures / max(self.requests, 1),
                "requested_user_slots": self.requests}
