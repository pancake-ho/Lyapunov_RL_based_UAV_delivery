"""Explicit SNR convention with unchanged transmit power and geometry."""
import math
from dataclasses import replace
from baseline.NDTVS.common.paths import HERE
from env.p3.radio import rsu_link_capacity_bps, link_gain
from hppo.env import uav_link_capacity_bps


def at_snr(cfg, mode, db, reference_distance=None):
    bw = cfg.rsu_total_bandwidth_hz / cfg.rsu_capacity
    power = cfg.rsu_total_power_w / cfg.rsu_capacity
    ratio = 10.0 ** (float(db) / 10.0)
    if mode == "offset":
        noise = cfg.noise_psd_w_hz / ratio
    elif mode == "transmit":
        # Paper: W log2(1 + Gamma*|h|^2/(INR+1)).
        # Gamma excludes channel gain and Shannon gap. Our existing gap stays.
        noise = power / (bw * ratio)
    elif mode == "received":
        if reference_distance is None:
            raise ValueError("Received SNR requires an explicit reference distance")
        distance = math.hypot(reference_distance, cfg.rsu_height_m-cfg.user_height_m)
        gain = link_gain(cfg.rsu_beta0, distance, cfg.rsu_pathloss_exp, 1.)
        # Physical received SNR P*g/(N0*W); gap remains separately in capacity.
        noise = power * gain / (bw * ratio)
    else:
        raise ValueError("Choose transmit, received or offset SNR")
    if not math.isfinite(noise) or cfg.shannon_gap * noise * min(bw, cfg.uav_user_bandwidth_hz) <= 1e-30:
        raise ValueError("Noise invalid or numerical floor would invalidate SNR")
    result = replace(cfg, noise_psd_w_hz=noise)
    actual = (cfg.noise_psd_w_hz/noise if mode == "offset" else
              power/(noise*bw) if mode == "transmit" else power*gain/(noise*bw))
    if not math.isclose(10*math.log10(actual), float(db), abs_tol=1e-10):
        raise AssertionError("SNR mapping failed")
    return result


def sanity(cfg, s):
    levels = s.SNR_OFFSETS_DB if s.SNR_MODE == "offset" else s.SNR_DB
    rows = []
    for db in levels:
        c = at_snr(cfg, s.SNR_MODE, db, s.REFERENCE_DISTANCE_M)
        bw, p = c.rsu_total_bandwidth_hz/c.rsu_capacity, c.rsu_total_power_w/c.rsu_capacity
        rows.append(dict(snr_db=db, mode=s.SNR_MODE, noise_psd_w_hz=c.noise_psd_w_hz,
            transmit_snr_rsu_db=10*math.log10(p/(c.noise_psd_w_hz*bw)),
            nominal_transmit_snr_rsu_db=10*math.log10(p/(cfg.noise_psd_w_hz*bw)),
            rsu_bps_at_150m_fading1=rsu_link_capacity_bps(150., 1., c),
            rsu_bps_at_nearest_point_fading1=rsu_link_capacity_bps(0., 1., c),
            uav_bps_at_150m_fading1=uav_link_capacity_bps(150., 1., c.uav_max_total_power_w/c.uav_capacity, c)[0],
            minimum_chunk_rate_bps=min(c.chunk_size_bits)/c.slot_duration_s))
    ordered = sorted(rows, key=lambda x: x["snr_db"])
    if any(a["rsu_bps_at_150m_fading1"] >= b["rsu_bps_at_150m_fading1"] for a,b in zip(ordered, ordered[1:])):
        raise AssertionError("Fixed-link capacity must increase with SNR")
    return rows


def axis_label(mode, reference_distance=None):
    return {"offset": "SNR offset from training channel (dB)",
            "transmit": "RSU transmit SNR P / (N0 W) (dB)",
            "received": f"RSU reference received SNR at {reference_distance} m (dB)"}[mode]
