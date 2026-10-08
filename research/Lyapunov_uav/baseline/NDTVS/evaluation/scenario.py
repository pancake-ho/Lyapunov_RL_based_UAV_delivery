"""SNR offsets and observation-only mobility/fading fingerprints."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
import numpy as np
from baseline.NDTVS.common.paths import HERE
from baseline.NDTVS.evaluation.checks import OFFSETS, require


def radio_sanity(cfg):
    from env.p3.radio import capacity_bps, rsu_link_capacity_bps
    from hppo.env import uav_link_capacity_bps
    rows = []
    for delta in OFFSETS:
        noise = cfg.noise_psd_w_hz * 10.0 ** (-delta / 10.0)
        effective = replace(cfg, noise_psd_w_hz=noise)
        for bw in (cfg.rsu_total_bandwidth_hz / cfg.rsu_capacity,
                   cfg.uav_user_bandwidth_hz):
            require(effective.shannon_gap * noise * bw > 1e-30,
                    "Noise floor clamp would invalidate the exact SNR offset")
            snr = 1e-12 / (effective.shannon_gap * noise * bw)
            base_snr = 1e-12 / (cfg.shannon_gap * cfg.noise_psd_w_hz * bw)
            require(math.isclose(10 * math.log10(snr / base_snr), delta, abs_tol=1e-12),
                    "SNR shift check failed")
            require(math.isfinite(capacity_bps(bw, 1, 1e-12, effective)), "Bad capacity")
        rows.append({"snr_offset_db": delta, "noise_psd_w_hz": noise,
                     "rsu_bps": rsu_link_capacity_bps(150, 1, effective),
                     "uav_bps": uav_link_capacity_bps(150, 1, 1, effective)[0]})
    require(rows[2]["noise_psd_w_hz"] == cfg.noise_psd_w_hz,
            "Nominal checkpoint noise mismatch")
    for key in ("rsu_bps", "uav_bps"):
        require(all(a[key] < b[key] for a, b in zip(rows, rows[1:])),
                "Fixed-link capacity is not strictly increasing")
    return rows


class PairingObserver:
    """Observe actual exogenous arrays; never change transitions or actions."""
    def add_arrays(self, tag, *arrays):
        self._scenario.update(tag.encode())
        for values in arrays:
            a = np.ascontiguousarray(values, dtype="<f8")
            require(np.isfinite(a).all(), "Nonfinite exogenous state")
            self._scenario.update(str(a.shape).encode())
            self._scenario.update(a.tobytes())

    def reset(self, episode=0, seed=None):
        super().reset(episode, seed)
        self._scenario = hashlib.sha256(f"{self.cfg.seed}:{episode}".encode())
        self._observed_slots = 0
        self.add_arrays("initial", self.state.user_x, self.state.user_speed)

    def prepare_frame(self):
        obs = super().prepare_frame()
        # Hash every potential RSU/UAV link, including unselected UAV points.
        self.add_arrays(f"frame:{self.frame}", self.trace.rsu_fading, self.trace.uav_fading)
        return obs

    def step_slot(self, actions):
        result = super().step_slot(actions)
        self.add_arrays(f"slot:{self._observed_slots}", self.state.user_x, self.state.user_speed)
        self._observed_slots += 1
        for rg in result.info["regions"].values():
            values = [rg[k] for k in ("total_requested_power_w", "total_executed_power_w",
                      "p_eff_w", "battery_before_j", "battery_after_j",
                      "hover_energy_j", "communication_energy_j")]
            require(np.isfinite(values).all(), "Nonfinite physical metric")
            require(rg["reserve_ok"], "Reserve violation")
            require(-1e-9 <= rg["total_executed_power_w"] <= rg["p_eff_w"] + 1e-9,
                    "Executed power violates feasibility")
        return result

    def episode_summary(self):
        row = super().episode_summary()
        row.update(scenario_sha256=self._scenario.hexdigest(),
                   observed_slots=self._observed_slots)
        return row
