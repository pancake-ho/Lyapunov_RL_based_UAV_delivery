from __future__ import annotations

import math

from config_p1 import P1Config


def link_gain(
    beta0: float,
    distance_m: float,
    exponent: float,
    fading: float,
) -> float:
    distance = max(float(distance_m), 1.0)
    return (
        float(beta0)
        * max(float(fading), 0.0)
        / distance ** float(exponent)
    )


def capacity_bps(
    bandwidth_hz: float,
    power_w: float,
    gain: float,
    cfg: P1Config,
) -> float:
    noise = cfg.shannon_gap * cfg.noise_psd_w_hz * bandwidth_hz
    snr = max(power_w, 0.0) * max(gain, 0.0) / max(noise, 1e-30)
    return float(bandwidth_hz * math.log2(1.0 + snr))


def rsu_link_capacity_bps(
    horizontal_distance_m: float,
    fading: float,
    cfg: P1Config,
) -> float:
    """Fixed RSU resource-block bandwidth and power, W^R/J^R and P^R/J^R."""

    bandwidth = cfg.rsu_total_bandwidth_hz / cfg.rsu_capacity
    power = cfg.rsu_total_power_w / cfg.rsu_capacity
    vertical = cfg.rsu_height_m - cfg.user_height_m
    distance = math.hypot(horizontal_distance_m, vertical)
    gain = link_gain(
        cfg.rsu_beta0,
        distance,
        cfg.rsu_pathloss_exp,
        fading,
    )
    return capacity_bps(bandwidth, power, gain, cfg)


def required_uav_power_w(
    chunks: int,
    quality_index: int,
    horizontal_distance_m: float,
    fading: float,
    cfg: P1Config,
) -> float:
    """Closed-form minimum power satisfying the selected payload rate."""

    if chunks <= 0:
        return 0.0
    if not 0 <= quality_index < len(cfg.chunk_size_bits):
        raise IndexError(f"invalid quality index: {quality_index}")
    bandwidth = cfg.uav_total_bandwidth_hz  # J^U=1
    vertical = cfg.uav_height_m - cfg.user_height_m
    distance = math.hypot(horizontal_distance_m, vertical)
    gain = link_gain(
        cfg.uav_beta0,
        distance,
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
