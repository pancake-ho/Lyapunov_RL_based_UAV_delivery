"""Shared transition event order, baseline decisions and common reporting."""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

import baseline.NDTVS.api as c
from baseline.NDTVS.metrics.service import ServiceLogger
from baseline.UAVRTV.environment.shared import SharedUAVRTVEnv
from baseline.UAVRTV.rewards.paper import COMPONENTS, components, definition


def run_episode(cfg, s, agent, number, root, training=False, replay=None, rng=None,
                counters=None, trace=False, profile=None):
    root = Path(root)
    lc = replace(cfg, write_jsonl_trace=trace, write_human_debug_log=False, log_hidden_csi=True)
    c.atomic(root / "resolved_config.json", dict(config=asdict(lc), algorithm="uavrtv", reward=definition(s, cfg)))
    log, env = ServiceLogger(lc, root), SharedUAVRTVEnv(cfg)
    log.event("baseline_start", dict(algorithm="uavrtv", episode=number, reward=definition(s, cfg)))
    totals = dict.fromkeys(COMPONENTS, 0.)
    samples, losses = [], []
    env.reset(number)
    env.prepare_frame()
    boundary, terminal = True, False
    data = {m: env.observation(m, True) for m in env.regions}
    before_updates = agent.updates if agent is not None else 0
    try:
        while not terminal:
            obs, masks = np.stack([data[m][0] for m in env.regions]), np.stack([data[m][1] for m in env.regions])
            if profile is not None:
                choices = np.ones_like(masks)
                if profile == "nohire":
                    choices[:, 0] = -1
                elif profile == "random":
                    choices = rng.uniform(-1, 1, masks.shape).astype(np.float32)
                else:
                    choices[:, 1] = 0.  # Middle feasible hovering point.
                choices *= masks
            elif training and counters["transitions"] < s.START_TRANSITIONS:
                choices = rng.uniform(-1, 1, masks.shape).astype(np.float32) * masks
            else:
                choices = agent.act(obs, masks, deterministic=not training)
            actions = {m: choices[i] for i, m in enumerate(env.regions)}
            frame_info = env.start_from_sac(actions) if boundary else None
            if frame_info:
                log.log_frame_start(frame_info)
            executed = {m: env.requests(m, actions[m]) for m in env.regions}
            step = env.step_slot(executed)
            step.info["uavrtv_continuous_actions"] = {m: actions[m].tolist() for m in env.regions}
            step.info["uavrtv_action_support"] = {m: data[m][1].tolist() for m in env.regions}
            log.observe_slot(step.info)
            rewards = {}
            for m, rg in step.info["regions"].items():
                rewards[m] = components(rg, cfg, s, frame_info["regions"][m] if frame_info else None)
                for key in COMPONENTS:
                    totals[key] += rewards[m][key]
                samples.append(dict(episode=number, frame=step.info["frame"], slot=step.info["slot_in_frame"], region=m, **rewards[m]))
            log.log_slot(step.info, rewards=rewards)
            if step.frame_done:
                log.log_frame_end(step.info["frame_summary"])
            terminal = step.frame_done and env.frame == cfg.num_frames
            next_boundary = step.frame_done and not terminal
            if next_boundary:
                env.prepare_frame()
            next_data = ({m: env.observation(m, next_boundary) for m in env.regions} if not terminal else
                         {m: (np.zeros(env.obs_dim, np.float32), np.zeros(env.act_dim, np.float32)) for m in env.regions})
            if training:
                for m in env.regions:
                    replay.add(data[m][0], actions[m], rewards[m]["training"], next_data[m][0], terminal, data[m][1], next_data[m][1])
                    counters["transitions"] += 1
                if replay.n >= s.BATCH_SIZE and counters["transitions"] >= s.START_TRANSITIONS:
                    for _ in range(s.UPDATES_PER_SLOT):
                        losses.append(agent.update(replay.sample(s.BATCH_SIZE, rng)))
            data, boundary = next_data, next_boundary

        row = env.episode_summary()
        row["legacy_average_quality_utility"] = row["average_quality_utility"]
        row.update(log.measures())
        slots = cfg.num_frames * cfg.frame_slots * cfg.num_regions
        raw = totals["quality_gain"] - sum(totals[k] for k in COMPONENTS[1:])
        row.update(uavrtv_reward_total=raw, uavrtv_reward_per_region_slot=raw / slots,
            uavrtv_qoe_total=totals["quality_gain"] - totals["switch_penalty"] - totals["rebuffer_penalty"],
            **{key + "_total": value for key, value in totals.items()})
        row["stall_user_slot_ratio"] = row["stall_ratio"]
        if not np.isclose(totals["hiring_penalty"], row["hiring_cost_total"], rtol=0, atol=1e-8):
            raise AssertionError("Hiring reward differs from once-per-frame physical accounting")
        if not np.isclose(totals["energy_penalty"], s.VARSIGMA * row["energy_consumed_j"], rtol=0, atol=1e-7):
            raise AssertionError("Energy reward omits/double-counts a physical energy term")
        if row["reserve_violations"] or row["power_violations"] or row["q_gt_qe_rate"]:
            raise AssertionError("Shared physical invariant failed")
        if not training and agent is not None and agent.updates != before_updates:
            raise AssertionError("Evaluation updated SAC")
        if losses:
            row.update({key: float(np.mean([r[key] for r in losses])) for key in losses[0]})
        users = log.user_rows()
        c.atomic(root / "per_user.json", users)
        c.write_rows(root / "per_user.csv", users)
        c.write_rows(root / "reward_components.csv", samples)
        c.atomic(root / "summary.json", row)
        log.log_episode(row)
        log.event("baseline_end", dict(status="complete"))
        return row
    finally:
        log.close()
