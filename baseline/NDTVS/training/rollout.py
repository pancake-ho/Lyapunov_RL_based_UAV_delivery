"""Observations, masks and episode rollout with unchanged event order."""
from __future__ import annotations

from dataclasses import asdict, replace
import numpy as np
import torch
from baseline.NDTVS.common.config import VERSION
from baseline.NDTVS.environment.rsu import RSUEnv, NoUAVCompletion, P3HierarchicalEnv
from baseline.NDTVS.metrics.observer import QoELogger
from baseline.NDTVS.common.io import atomic
from baseline.NDTVS.rewards.qoe import reward_spec
from hppo import train as hrl


def ndt_observation(env, m, log, boundary):
    c, s = env.cfg, env.state
    x = np.zeros((env.N, 10), dtype=np.float32)
    for u in env.region_users[m]:
        x[u] = (1, s.queue[u] / c.large_queue_level,
                (c.large_queue_level - s.queue[u]) / c.large_queue_level,
                (s.user_x[u] - c.rsu_x(m)) / c.region_length_m,
                s.user_speed[u] / c.vehicle_speed_max_mps,
                (s.last_quality_index[u] + 1) / env.K,
                np.arcsinh(log.last_qoe[u]) / 3, log.last_utility[u],
                float(s.queue[u] < c.playback_chunks_per_slot),
                0 if boundary else float(env.provider[u] == 1))
    obs = np.concatenate(([float(boundary), env.frame / c.num_frames,
                           (0 if boundary else env.local_slot) / c.frame_slots,
                           len(env.region_users[m]) / env.N], x.ravel())).astype(np.float32)
    pointer = np.zeros(env.N + 1, dtype=bool)
    pointer[-1] = True
    if boundary:
        pointer[list(env.region_users[m])] = True
    masks = [pointer.copy() for _ in range(c.rsu_capacity)]
    for u in range(env.N):
        cap = env.guard.max_queue_admissible_chunks(c.large_queue_level - float(s.queue[u]))
        valid = np.zeros(1 + c.max_chunks_per_slot * env.K, dtype=bool)
        valid[0] = True
        if u in env.region_users[m]:
            valid[1:1 + cap * env.K] = True
        masks.append(valid)
    return obs, masks


def ndt_episode(env, agent, log, episode, training, deterministic):
    env.reset(episode)
    c, pending = env.cfg, {}
    for frame in range(c.num_frames):
        env.prepare_frame()
        for slot in range(c.frame_slots):
            data = {m: ndt_observation(env, m, log, slot == 0) for m in env.regions}
            # Regions share weights but have separate rewards/GAE trajectories.
            batch_obs = torch.as_tensor(np.stack([data[m][0] for m in env.regions]), device=agent.device)
            batch_masks = [torch.as_tensor(np.stack([data[m][1][j] for m in env.regions]), device=agent.device)
                           for j in range(len(agent.nvec))]
            with torch.no_grad():
                draws = agent.net.decode(batch_obs, batch_masks, deterministic=deterministic)
            aa, ll, vv, ee = [x.cpu().numpy() for x in draws]
            choices = {m: (aa[i], float(ll[i]), float(vv[i]), float(ee[i]))
                       for i, m in enumerate(env.regions)}
            for m, prev in pending.items():
                agent.store(m, *prev, next_value=choices[m][2], done=False)
            pending = {}
            if slot == 0:
                raw = {}
                for m, (a, _, _, _) in choices.items():
                    raw[m] = np.zeros(env.N, dtype=np.int64)
                    picks = a[:c.rsu_capacity]
                    raw[m][picks[picks < env.N]] = 1
                completed, details = NoUAVCompletion(c).select_all(env, raw, agent)
                log.log_frame_start(env.begin_frame(raw, completed, details))
            actions = {}
            for m, (a, _, _, _) in choices.items():
                tokens = a[c.rsu_capacity:]
                raw = np.zeros(3 * env.N, dtype=np.int64)
                raw[:env.N] = np.where(tokens > 0, (tokens - 1) // env.K + 1, 0)
                raw[env.N:2 * env.N] = np.where(tokens > 0, (tokens - 1) % env.K, 0)
                actions[m] = raw
            step = env.step_slot(actions)
            env.assert_consistency()
            log.observe_slot(step.info)
            rewards = {m: sum(u["paper_reward_scaled"] for u in rg["users"])
                       for m, rg in step.info["regions"].items()}
            log.log_slot(step.info, rewards={m: {"training": r} for m, r in rewards.items()})
            if training:
                terminal = frame == c.num_frames - 1 and step.frame_done
                for m, (a, lp, value, _) in choices.items():
                    prev = (*data[m], a, lp, value, rewards[m])
                    if terminal:
                        agent.store(m, *prev, next_value=0, done=True)
                    else:
                        pending[m] = prev
            if step.frame_done:
                log.log_frame_end(step.info["frame_summary"])
    return env.episode_summary()


def episode(cfg, algorithm, agents, number, dual, training, root, trace):
    lc = replace(cfg, write_jsonl_trace=trace, write_human_debug_log=False)
    atomic(root / "resolved_config.json", {"config": asdict(lc), "algorithm": algorithm})
    log = QoELogger(lc, root)
    log.event("baseline_start", {"version": VERSION, "algorithm": algorithm, "episode": number,
                                 "qoe_definition": reward_spec()})
    env = P3HierarchicalEnv(cfg) if algorithm == "proposed" else RSUEnv(cfg)
    original = hrl.FastPolicyCompletion
    try:
        if algorithm == "ndtvs":
            result = ndt_episode(env, agents[0], log, number, training, not training)
        else:
            if algorithm == "hppo_rsu":
                hrl.FastPolicyCompletion = NoUAVCompletion
            result = hrl.run_episode(env, *agents, cfg, log, number, dual, training, not training)
        result["legacy_average_quality_utility"] = result["average_quality_utility"]
        result.update(log.measures())
        result["stall_user_slot_ratio"] = result["stall_ratio"]
        if algorithm != "proposed" and (result["hire_rate"] != 0 or result["hiring_cost_total"] != 0):
            raise AssertionError("RSU-only invariant failed")
        log.log_episode(result)
        log.event("baseline_end", {"status": "complete"})
        return result
    finally:
        hrl.FastPolicyCompletion = original
        log.close()
