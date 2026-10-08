"""Wu et al. Eq.13/14 adapted to common buffers and frame hiring."""
import numpy as np
from baseline.NDTVS.rewards.qoe import PSNR_DB

COMPONENTS = ("quality_gain", "switch_penalty", "rebuffer_penalty", "energy_penalty", "hiring_penalty")


def definition(s, cfg):
    return dict(version="uavrtv-psnr-rebuffer-energy-hiring-v1", psnr_db=list(PSNR_DB),
        beta=s.BETA, delta_per_bps=s.DELTA, phi_per_second=s.PHI, varsigma_per_joule=s.VARSIGMA,
        lambda_h=cfg.lambda_h, hiring_cost_per_frame=cfg.hiring_cost_per_frame, scale=s.REWARD_SCALE,
        quality="PSNR/PSNRmax once per successful user-slot; no chunk-count multiplication",
        switching="absolute bitrate change between successful deliveries; first delivery has zero switching",
        delay="adapted current rebuffer seconds, not the original download-delay equation",
        energy="actual common relocation+hover+communication energy; charging energy is excluded",
        hiring="lambda_h*c_h once at the first slot of each hired frame; no Lyapunov V multiplier",
        scope="all region members, including unscheduled users", rsu="low-buffer scheduling, mean-fading max-quality requests")


def components(region, cfg, s, boundary=None):
    chunk_seconds = cfg.slot_duration_s / cfg.playback_chunks_per_slot
    rates = np.asarray(cfg.chunk_size_bits) / chunk_seconds
    quality = switching = rebuffer = 0.
    for u in region["users"]:
        rebuffer += max(cfg.playback_chunks_per_slot - u["q_before"], 0.) * chunk_seconds
        if u["delivered"] > 0:
            k, previous = u["req_quality"], u["last_quality_before"]
            quality += PSNR_DB[k] / PSNR_DB[-1]
            if previous >= 0:
                switching += abs(rates[k] - rates[previous])
    move = boundary["relocation_energy_j"] if boundary else 0.
    hire = boundary["hiring_cost_weighted"] if boundary else 0.
    energy = move + region["hover_energy_j"] + region["communication_energy_j"]
    out = dict(quality_gain=s.BETA * quality, switch_penalty=s.DELTA * switching,
        rebuffer_penalty=s.PHI * rebuffer, energy_penalty=s.VARSIGMA * energy,
        hiring_penalty=hire, rebuffer_seconds=rebuffer, uav_consumed_j=energy)
    out["raw_reward"] = out["quality_gain"] - sum(out[k] for k in COMPONENTS[1:])
    out["training"] = out["raw_reward"] * s.REWARD_SCALE
    if not np.isfinite(list(out.values())).all():
        raise ValueError("Nonfinite UAVRTV reward")
    return out
