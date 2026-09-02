from __future__ import annotations

import math

from config_p3 import P3Config


def link_gain(beta0: float, distance_m: float, exponent: float, fading: float) -> float:
    distance = max(float(distance_m), 1.0)
    return float(beta0) * max(float(fading), 0.0) / distance ** float(exponent)


def capacity_bps(bandwidth_hz: float, power_w: float, gain: float, cfg: P3Config) -> float:
    noise = cfg.shannon_gap * cfg.noise_psd_w_hz * bandwidth_hz
    snr = max(power_w, 0.0) * max(gain, 0.0) / max(noise, 1e-30)
    return float(bandwidth_hz * math.log2(1.0 + snr))


def rsu_link_capacity_bps(
    horizontal_distance_m: float,
    fading: float,
    cfg: P3Config,
) -> float:
    bandwidth = cfg.rsu_total_bandwidth_hz / cfg.rsu_capacity
    power = cfg.rsu_total_power_w / cfg.rsu_capacity
    vertical = cfg.rsu_height_m - cfg.user_height_m
    gain = link_gain(
        cfg.rsu_beta0,
        math.hypot(horizontal_distance_m, vertical),
        cfg.rsu_pathloss_exp,
        fading,
    )
    return capacity_bps(bandwidth, power, gain, cfg)


def required_uav_power_w(
    chunks: int,
    quality_index: int,
    horizontal_distance_m: float,
    fading: float,
    cfg: P3Config,
) -> float:
    """Equation (7.3) using the fixed per-user UAV resource group W^U/J^U."""

    if chunks <= 0:
        return 0.0
    if not 0 <= quality_index < cfg.num_quality_levels:
        raise IndexError(f"invalid quality index: {quality_index}")
    bandwidth = cfg.uav_user_bandwidth_hz
    vertical = cfg.uav_height_m - cfg.user_height_m
    gain = link_gain(
        cfg.uav_beta0,
        math.hypot(horizontal_distance_m, vertical),
        cfg.uav_pathloss_exp,
        fading,
    )
    if gain <= 0.0:
        return math.inf
    spectral_term = (
        chunks
        * cfg.chunk_size_bits[quality_index]
        / (bandwidth * cfg.slot_duration_s)
    )
    noise = cfg.shannon_gap * cfg.noise_psd_w_hz * bandwidth
    return float(noise * (2.0**spectral_term - 1.0) / gain)
