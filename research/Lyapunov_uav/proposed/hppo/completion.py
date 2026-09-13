"""Compare no-hire and every feasible point with a fixed fast policy.

Rollouts are isolated from real transitions and buffers. All candidates use
common channel and policy random numbers from a separate SeedSequence domain.
Only candidate proposals, not actual future fading, enter the comparator.
"""
from __future__ import annotations

import time
import numpy as np

from config_hppo import HPPOConfig
from env.p3.environment import generate_frame_trace
from env.p3.types import RegionAction


class FastPolicyCompletion:
    def __init__(self, cfg: HPPOConfig):
        self.cfg = cfg

    def select(self, env, region: int, raw, fast_policy, deterministic: bool = False):
        started = time.perf_counter()
        cfg = self.cfg
        proposal = env.proposal(region, raw)
        points = env.feasible_hover_points(region)
        candidates = [proposal.execute(0, -1)] + [proposal.execute(1, point) for point in points]
        samples = np.zeros((len(candidates), cfg.rollout_scenarios), dtype=np.float64)
        domains = []
        for scenario in range(cfg.rollout_scenarios):
            # List components are separate seed domains, avoiding scalar-offset collisions.
            domain = [int(cfg.seed), int(cfg.completion_seed_offset), int(env.episode),
                      int(env.frame), int(region), int(scenario)]
            channel_seq, action_seq = np.random.SeedSequence(domain).spawn(2)
            channel_seed = int(channel_seq.generate_state(1, dtype=np.uint64)[0])
            action_rng = np.random.default_rng(action_seq)
            trace = generate_frame_trace(cfg, channel_seed)
            domains.append({"domain": domain, "channel_seed": channel_seed})
            sims = [env.fork_for_rollout(region, trace) for _ in candidates]
            for sim, action in zip(sims, candidates):
                sim.begin_frame({region: raw}, {region: action})
            for _ in range(cfg.frame_slots):
                obs = np.stack([sim.get_slot_obs(region) for sim in sims])
                mask_rows = [sim.slot_action_masks(region) for sim in sims]
                masks = [np.stack([row[h] for row in mask_rows]) for h in range(len(cfg.slot_action_nvec))]
                # The same uniform for a head is mapped through each candidate's own CDF.
                uniforms = np.broadcast_to(action_rng.random(len(cfg.slot_action_nvec)),
                                           (len(sims), len(cfg.slot_action_nvec))).copy()
                actions, _, _, _ = fast_policy.act_batch(obs, masks, deterministic, uniforms)
                for i, (sim, action) in enumerate(zip(sims, actions)):
                    step = sim.step_slot({region: action})
                    if step.frame_done:
                        samples[i, scenario] = step.info["frame_summary"]["regions"][region]["frame_dpp_cost"]
        if not np.isfinite(samples).all():
            raise FloatingPointError("nonfinite candidate DPP")
        scores = samples.mean(axis=1)
        # Stable exact tie: no-hire first, then increasing point index.
        best = int(np.argmin(scores))
        info = {
            "method": "fixed_fast_policy_mc_dpp", "scenarios": cfg.rollout_scenarios,
            "fast_deterministic": bool(deterministic),
            "fast_update_count": int(getattr(fast_policy, "update_count", 0)),
            "seed_domains": domains, "selected_index": best,
            "selected_score": float(scores[best]),
            "runtime_s": time.perf_counter() - started,
            "candidates": [
                {"hired": a.hired, "point": a.point_index, "rsu_users": list(a.rsu_users),
                 "uav_users": list(a.uav_users), "sample_dpp": samples[i].tolist(),
                 "mean_dpp": float(scores[i]),
                 "hiring_dpp": cfg.lyapunov_v * cfg.lambda_h * cfg.hiring_cost_per_frame * a.hired}
                for i, a in enumerate(candidates)
            ],
        }
        return candidates[best], info

    def select_all(self, env, proposals, fast_policy, deterministic=False):
        actions, details = {}, {}
        for region in env.regions:
            actions[region], details[region] = self.select(
                env, region, proposals[region], fast_policy, deterministic)
        return actions, details
