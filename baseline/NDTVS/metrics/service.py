"""All-user service distribution, separate from the training QoE observer."""
import math
import numpy as np
from baseline.NDTVS.metrics.observer import QoELogger
from baseline.NDTVS.rewards.qoe import PSNR_DB


def jain(values):
    x = np.asarray(values, dtype=float)
    denominator = len(x)*float(x @ x)
    return float(x.sum()**2/denominator) if denominator else None


class ServiceLogger(QoELogger):
    def __init__(self, cfg, root):
        super().__init__(cfg, root)
        self.counts = {k: np.zeros(cfg.num_users, dtype=float) for k in
                       ("slots", "scheduled", "requested", "delivery_slots", "delivered", "psnr_sum", "stall_s", "stall_slots", "played", "qoe_sum")}
        self.hired_frames = 0
        self.chunk_histogram = np.zeros(cfg.max_chunks_per_slot+1, dtype=int)
        self.quality_histogram = np.zeros(cfg.num_quality_levels, dtype=int)
        self.service_gaps = np.zeros(cfg.num_users, dtype=int)
        self.max_service_gaps = np.zeros(cfg.num_users, dtype=int)

    def log_frame_start(self, info, **kwargs):
        self.hired_frames += sum(int(rg["executed_hire"]) for rg in info["regions"].values())
        super().log_frame_start(info, **kwargs)

    def observe_slot(self, info):
        key = (info["episode"], info["frame"], info["slot_in_frame"])
        if key == self._observed_key:
            super().observe_slot(info)
            return
        super().observe_slot(info)
        c = self.counts
        for rg in info["regions"].values():
            for u in rg["users"]:
                i, received = u["user"], int(u["delivered"])
                c["slots"][i] += 1
                c["scheduled"][i] += u["provider"] != 0
                c["requested"][i] += u["req_chunks"] > 0
                c["delivery_slots"][i] += received > 0
                c["delivered"][i] += received
                c["psnr_sum"][i] += received*PSNR_DB[u["req_quality"]]
                c["stall_s"][i] += u["stall_duration_s"]
                c["stall_slots"][i] += bool(u["stall"])
                c["played"][i] += min(u["q_before"], self.cfg.playback_chunks_per_slot)
                c["qoe_sum"][i] += u["paper_qoe"]
                self.chunk_histogram[int(u["req_chunks"])] += 1
                if received:
                    self.quality_histogram[int(u["req_quality"])] += received
                    self.service_gaps[i] = 0
                else:
                    self.service_gaps[i] += 1
                self.max_service_gaps[i] = max(self.max_service_gaps[i], self.service_gaps[i])

    def user_rows(self):
        c, cfg = self.counts, self.cfg
        result = []
        for i in range(cfg.num_users):
            n, received = c["slots"][i], c["delivered"][i]
            if n != cfg.num_frames*cfg.frame_slots:
                raise ValueError("Incomplete user history")
            psnr = c["psnr_sum"][i]/received if received else None
            result.append(dict(user=i, user_slots=int(n), received_chunks=int(received),
                delivery_user_slot_ratio=c["delivery_slots"][i]/n,
                scheduled_user_slot_ratio=c["scheduled"][i]/n,
                requested_user_slot_ratio=c["requested"][i]/n,
                stall_user_slot_ratio=c["stall_slots"][i]/n,
                stall_time_ratio=c["stall_s"][i]/(n*cfg.slot_duration_s),
                delivered_chunks_per_slot=received/n,
                playback_fulfillment_ratio=c["played"][i]/(n*cfg.playback_chunks_per_slot),
                received_psnr_db=psnr, received_quality_utility=psnr/PSNR_DB[-1] if received else None,
                paper_qoe_per_slot=c["qoe_sum"][i]/n,
                longest_no_delivery_slots=int(self.max_service_gaps[i])))
        return result

    def measures(self):
        rows, result = self.user_rows(), super().measures()
        c = self.counts
        total, us = c["delivered"].sum(), c["slots"].sum()
        stall = np.array([r["stall_time_ratio"] for r in rows])
        quality = [r["received_quality_utility"] for r in rows if r["received_quality_utility"] is not None]
        top = max(1, math.ceil(self.cfg.num_users*.2))
        result.update(unique_served_user_ratio=float(np.mean(c["delivered"] > 0)),
            never_served_user_ratio=float(np.mean(c["delivered"] == 0)),
            scheduled_user_slot_ratio=float(c["scheduled"].sum()/us),
            delivery_user_slot_ratio=float(c["delivery_slots"].sum()/us),
            playback_fulfillment_ratio=float(c["played"].sum()/(us*self.cfg.playback_chunks_per_slot)),
            delivery_jain_index=jain(c["delivered"]),
            delivery_jain_defined=bool(total),
            playback_jain_index=jain(c["played"]),
            top20_delivery_share=float(np.sort(c["delivered"])[-top:].sum()/total) if total else None,
            per_user_quality_mean=float(np.mean(quality)) if quality else None,
            worst_user_stall_time_ratio=float(stall.max()),
            p90_user_stall_time_ratio=float(np.quantile(stall, .9)),
            p10_user_delivery_chunks_per_slot=float(np.quantile(c["delivered"]/c["slots"], .1)),
            hired_uav_frames=self.hired_frames,
            requested_chunk_histogram=self.chunk_histogram.tolist(),
            received_quality_histogram=self.quality_histogram.tolist())
        # Reconstruct key pooled metrics from independent per-user counters.
        for actual, expected in ((result["received_segments_total"], total),
                (result["stall_time_ratio"], c["stall_s"].sum()/(us*self.cfg.slot_duration_s)),
                (result["paper_qoe_per_user_slot"], c["qoe_sum"].sum()/us)):
            if not np.isclose(actual, expected, rtol=0, atol=1e-8):
                raise AssertionError("User/aggregate metric mismatch")
        return result
