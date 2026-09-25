"""Compare no-hire and every feasible point with a fixed fast policy.

Rollouts are isolated from real transitions and buffers. All candidates use
common channel and policy random numbers from a separate SeedSequence domain.
Only candidate proposals, not actual future fading, enter the comparator.
"""
from __future__ import annotations

import time
import numpy as np

from config_hppo import HPPOConfig
from env.p3.battery import activation_energy_required_j, relocation_energy_j
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
        components = [[] for _ in candidates]
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
            scenario_slots = [[] for _ in candidates]
            scenario_users = [{u: {"user": u, "queue_drift": 0., "quality_dpp": 0.,
                                   "delivered": 0, "failed_requests": 0}
                               for u in env.region_users[region]} for _ in candidates]
            for slot in range(cfg.frame_slots):
                obs = np.stack([sim.get_slot_obs(region) for sim in sims])
                mask_rows = [sim.slot_action_masks(region) for sim in sims]
                masks = [np.stack([row[h] for row in mask_rows]) for h in range(len(cfg.slot_action_nvec))]
                # The same uniform for a head is mapped through each candidate's own CDF.
                uniforms = np.broadcast_to(action_rng.random(len(cfg.slot_action_nvec)),
                                           (len(sims), len(cfg.slot_action_nvec))).copy()
                actions, _, _, _ = fast_policy.act_batch(obs, masks, deterministic, uniforms)
                for i, (sim, action) in enumerate(zip(sims, actions)):
                    step = sim.step_slot({region: action})
                    records = step.info["regions"][region]["users"]
                    drift = sum(u["queue_drift_term"] for u in records)
                    quality = sum(u["quality_dpp_term"] for u in records)
                    scenario_slots[i].append({"slot": slot, "queue_drift": drift,
                                              "quality_dpp": quality, "dpp": drift + quality,
                                              "delivered": sum(u["delivered"] for u in records),
                                              "failed_requests": sum(u["transmission_failed"] for u in records)})
                    for u in records:
                        agg = scenario_users[i][u["user"]]
                        agg["queue_drift"] += u["queue_drift_term"]
                        agg["quality_dpp"] += u["quality_dpp_term"]
                        agg["delivered"] += u["delivered"]
                        agg["failed_requests"] += int(u["transmission_failed"])
                    if step.frame_done:
                        summary = step.info["frame_summary"]["regions"][region]
                        samples[i, scenario] = summary["frame_dpp_cost"]
                        terms = {"scenario": scenario,
                                 "queue_drift": sum(v["queue_drift"] for v in scenario_slots[i]),
                                 "quality_dpp": sum(v["quality_dpp"] for v in scenario_slots[i]),
                                 "hiring_dpp": cfg.lyapunov_v * summary["hiring_cost_weighted"],
                                 "frame_dpp": summary["frame_dpp_cost"],
                                 "slots": scenario_slots[i],
                                 "users": list(scenario_users[i].values())}
                        if not np.isclose(terms["queue_drift"] + terms["quality_dpp"] + terms["hiring_dpp"],
                                          terms["frame_dpp"], rtol=1e-10, atol=1e-8):
                            raise RuntimeError("candidate DPP decomposition mismatch")
                        components[i].append(terms)
        if not np.isfinite(samples).all():
            raise FloatingPointError("nonfinite candidate DPP")
        scores = samples.mean(axis=1)
        # Stable exact tie: no-hire first, then increasing point index.
        best = int(np.argmin(scores))
        point_checks = []
        previous = float(env.state.uav_x[region])
        battery = float(env.state.battery_j[region])
        for point, target in enumerate(cfg.candidate_points(region)):
            distance = abs(target - previous)
            energy = relocation_energy_j(previous, target, cfg)
            reasons = []
            if distance > cfg.reachable_distance_m + 1e-9:
                reasons.append("unreachable")
            if battery - energy + 1e-9 < activation_energy_required_j(cfg):
                reasons.append("activation_reserve")
            point_checks.append({"point": point, "target_x_m": target, "distance_m": distance,
                                 "relocation_energy_j": energy, "battery_after_move_j": battery-energy,
                                 "feasible": not reasons, "excluded_reasons": reasons})
        info = {
            "dpp_formula": "mean_s[V*lambda_H*c_H*hire + sum_t,n(alpha_Z*Z*(departure-delivered) + V*delivered*(quality_max-quality[k]))]",
            "estimate_scope": "Monte Carlo estimate under fixed fast policy; not exact expected DPP or realized future",
            "point_checks": point_checks,
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
                 "sample_components": components[i],
                 "mean_queue_drift": float(np.mean([c["queue_drift"] for c in components[i]])),
                 "mean_quality_dpp": float(np.mean([c["quality_dpp"] for c in components[i]])),
                 "delta_from_no_hire": float(scores[i] - scores[0]),
                 "delta_from_selected": float(scores[i] - scores[best]),
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
