"""Ladipo-adapted RSU PPO and a proposed-without-UAV ablation.

Use the sibling proposed/ physical implementation without editing it.
No GRU, instantaneous CSI, cloud, cache, or bandwidth/power optimization.
This is an adaptation, not a reproduction of the original NDT system.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPOSED = HERE.parents[1] / "proposed"
sys.path.insert(0, str(PROPOSED))

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from config_hppo import HPPOConfig
from hppo.env import P3HierarchicalEnv
from hppo.logger import HistoryLogger, jsonable
from hppo.ppo import PPOAgent
from hppo import train as hrl

VERSION = "ndtvs-common-v1"
BASE_COMMIT = "e1d79cff46742625fd75fd3fc23917fa93428326"
QOE_WEIGHTS = (1.0, 0.5, 2.0)


def atomic(path, obj, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb" if binary else "w") as f:
            if binary:
                torch.save(obj, f)
            else:
                json.dump(jsonable(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def read_config(path):
    d = json.loads(Path(path).read_text())
    d = d.get("config", d)
    for k, v in list(d.items()):
        if isinstance(v, list):
            d[k] = tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
    required = {"num_regions", "users_per_region", "rsu_total_bandwidth_hz",
                "mask_queue_actions", "delivery_mode"}
    if not required <= d.keys():
        raise ValueError("Use a complete current proposed resolved_config.json")
    cfg = HPPOConfig(**d)
    if not (cfg.mask_queue_actions and cfg.enforce_queue_admissibility
            and cfg.delivery_mode == "all_or_nothing"):
        raise ValueError("This adaptation requires current queue-masked, atomic delivery")
    return cfg


def source_hashes():
    files = [HERE / "ndtvs_common.py"]
    for pattern in ("config*.py", "hppo/*.py", "env/p3/*.py", "agent/P3/*.py"):
        files.extend(sorted(PROPOSED.glob(pattern)))
    return {str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2].cpu())
    if state[3] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state[3]])


def load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class RSUEnv(P3HierarchicalEnv):
    def frame_action_masks(self, region):
        masks = super().frame_action_masks(region)
        for mask in masks:
            mask[2] = False
        return masks

    def proposal(self, region, raw):
        if np.any(np.asarray(raw) == 2):
            raise ValueError("RSU-only policy requested a UAV")
        return super().proposal(region, raw)

    def begin_frame(self, raw_actions, completed_actions, completion_info=None):
        if any(a.hired or a.uav_users for a in completed_actions.values()):
            raise ValueError("UAV service is disabled")
        return super().begin_frame(raw_actions, completed_actions, completion_info)


class NoUAVCompletion:
    def __init__(self, cfg):
        self.cfg = cfg

    def select_all(self, env, raw, fast_policy, deterministic=False):
        actions = {m: env.proposal(m, a).execute(0, -1) for m, a in raw.items()}
        detail = {m: {"reason": "UAV disabled", "runtime_s": 0.0, "scenarios": 0,
                      "fast_update_count": fast_policy.update_count,
                      "selected_index": None, "candidates": []} for m in raw}
        return actions, detail


def qoe_terms(rec, cfg):
    """Existing baseline surrogate: delivered utility - switching - stall.

    A quality change is charged once per delivery batch, including changes
    across a stall. This is not MOS or played-video PSNR.
    """
    d, k, previous = rec["delivered"], rec["req_quality"], rec["last_quality_before"]
    switch = abs(k - previous) / max(cfg.num_quality_levels - 1, 1) if d and previous >= 0 else 0.0
    stall_s = rec["stall"] * cfg.slot_duration_s
    utility = d * cfg.quality_utility[k]
    return utility - 0.5 * switch - 2.0 * stall_s, switch


class QoELogger(HistoryLogger):
    """The same observer measures all three algorithms, outside their rewards."""
    def __init__(self, cfg, root):
        super().__init__(cfg, root)
        self.qoe = self.switch = self.failures = self.requests = 0.0
        self.user_slots = self.stall_events = 0
        self.last_qoe = np.zeros(cfg.num_users)
        self.last_utility = np.zeros(cfg.num_users)
        self.last_stall = np.zeros(cfg.num_users, dtype=bool)

    def log_slot(self, info, **kwargs):
        for rg in info["regions"].values():
            for u in rg["users"]:
                qoe, switch = qoe_terms(u, self.cfg)
                i = u["user"]
                u.update(qoe_surrogate=qoe, switch_magnitude=switch)
                self.qoe += qoe
                self.switch += switch
                self.requests += u["req_chunks"] > 0
                self.failures += u["transmission_failed"]
                self.user_slots += 1
                self.stall_events += bool(u["stall"] and not self.last_stall[i])
                self.last_stall[i] = bool(u["stall"])
                self.last_qoe[i] = qoe
                self.last_utility[i] = self.cfg.quality_utility[u["req_quality"]] if u["delivered"] else 0
        super().log_slot(info, **kwargs)

    def measures(self):
        return {"qoe_surrogate_per_user_slot": self.qoe / max(self.user_slots, 1),
                "switch_magnitude_per_user_slot": self.switch / max(self.user_slots, 1),
                "stall_events_per_user_slot": self.stall_events / max(self.user_slots, 1),
                "request_failure_ratio": self.failures / max(self.requests, 1),
                "requested_user_slots": self.requests}


def mlp(d):
    return nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh())


class NDTVSNet(nn.Module):
    """One PPO policy: frame-gated scheduling and slot chunk/quality choices.

    J pointer draws select distinct users or STOP. Then each scheduled user
    selects idle or (chunks, quality). Replay reconstructs exactly the same
    conditional support. Unused choices have probability one.
    """
    def __init__(self, cfg):
        super().__init__()
        self.N, self.J = cfg.num_users, cfg.rsu_capacity
        self.D = 1 + cfg.max_chunks_per_slot * cfg.num_quality_levels
        self.obs_dim = 4 + 10 * self.N
        self.actor = mlp(10)
        self.context = mlp(68)
        self.score = nn.Linear(128, 1)
        self.stop = nn.Linear(64, 1)
        self.download = nn.Linear(128, self.D)
        self.critic = mlp(10)
        self.value_head = nn.Sequential(mlp(68), nn.Linear(64, 1))
        for head in (self.score, self.stop, self.download):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def encode(self, obs, encoder):
        x = obs[:, 4:].reshape(-1, self.N, 10)
        e = encoder(x)
        present = x[:, :, 0:1]
        pooled = (e * present).sum(1) / present.sum(1).clamp(min=1)
        return x, e, torch.cat((obs[:, :4], pooled), -1)

    def value(self, obs):
        return self.value_head(self.encode(obs, self.critic)[2]).squeeze(-1)

    def decode(self, obs, masks, actions=None, deterministic=False):
        x, e, ctx = self.encode(obs, self.actor)
        context = self.context(ctx)
        h = torch.cat((e, context[:, None].expand(-1, self.N, -1)), -1)
        scores = torch.cat((self.score(h).squeeze(-1), self.stop(context)), -1)
        boundary = obs[:, 0].bool()
        available = masks[0].clone()
        selected = torch.zeros_like(x[:, :, 0], dtype=torch.bool)
        ended = ~boundary
        chosen, lp, ent = [], obs.new_zeros(len(obs)), obs.new_zeros(len(obs))
        for j in range(self.J):
            valid = available.clone()
            valid[:, :self.N] &= ~ended[:, None]
            dist = Categorical(logits=scores.masked_fill(~valid, -1e9))
            a = actions[:, j] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
            if not valid.gather(1, a[:, None]).all():
                raise ValueError("invalid stored pointer action")
            chosen.append(a)
            lp, ent = lp + dist.log_prob(a), ent + dist.entropy()
            one = torch.nn.functional.one_hot(a, self.N + 1).bool()
            selected |= one[:, :self.N]
            available = available & ~one
            available[:, self.N] = True
            ended = ended | (a == self.N)
        scheduled = torch.where(boundary[:, None], selected, x[:, :, 9].bool())
        valid = torch.stack(masks[self.J:], dim=1).clone()
        valid[:, :, 1:] &= scheduled[:, :, None]
        dist = Categorical(logits=self.download(h).masked_fill(~valid, -1e9))
        a = actions[:, self.J:] if actions is not None else (dist.logits.argmax(-1) if deterministic else dist.sample())
        if not valid.gather(2, a[:, :, None]).all():
            raise ValueError("invalid stored download action")
        return (torch.cat((torch.stack(chosen, 1), a), 1),
                lp + dist.log_prob(a).sum(1), self.value(obs), ent + dist.entropy().sum(1))

    @torch.no_grad()
    def act(self, obs, masks, deterministic=False):
        return tuple(x.squeeze(0) for x in self.decode(obs.reshape(1, -1), masks, deterministic=deterministic))

    def evaluate_actions(self, obs, actions, masks):
        _, lp, value, ent = self.decode(obs, masks, actions)
        return lp, ent, value


def ndt_agent(cfg):
    # Learning settings remain those of the existing adapted baseline.
    learning = replace(cfg, hidden_dims=(128, 64), ppo_gamma=0.95,
                       ppo_minibatch_size=64, ppo_gae_lambda=0.95)
    nvec = (cfg.num_users + 1,) * cfg.rsu_capacity + (
        1 + cfg.max_chunks_per_slot * cfg.num_quality_levels,) * cfg.num_users
    agent = PPOAgent(4 + 10 * cfg.num_users, nvec, learning, "ndtvs_ppo")
    agent.net = NDTVSNet(cfg).to(agent.device)
    critic = list(agent.net.critic.parameters()) + list(agent.net.value_head.parameters())
    ids = {id(p) for p in critic}
    actor = [p for p in agent.net.parameters() if id(p) not in ids]
    agent.opt = torch.optim.Adam([{"params": actor, "lr": 1e-5}, {"params": critic, "lr": 1e-4}])
    return agent


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
            rewards = {m: sum(qoe_terms(u, c)[0] for u in rg["users"])
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


def make_agents(cfg, algorithm):
    if algorithm == "ndtvs":
        return [ndt_agent(cfg)]
    return [PPOAgent(cfg.frame_obs_dim, cfg.frame_action_nvec, cfg, "frame_ppo"),
            PPOAgent(cfg.slot_obs_dim, cfg.slot_action_nvec, cfg, "slot_ppo")]


def episode(cfg, algorithm, agents, number, dual, training, root, trace):
    lc = replace(cfg, write_jsonl_trace=trace, write_human_debug_log=False)
    atomic(root / "resolved_config.json", {"config": asdict(lc), "algorithm": algorithm})
    log = QoELogger(lc, root)
    log.event("baseline_start", {"version": VERSION, "algorithm": algorithm, "episode": number})
    env = P3HierarchicalEnv(cfg) if algorithm == "proposed" else RSUEnv(cfg)
    original = hrl.FastPolicyCompletion
    try:
        if algorithm == "ndtvs":
            result = ndt_episode(env, agents[0], log, number, training, not training)
        else:
            if algorithm == "hppo_rsu":
                hrl.FastPolicyCompletion = NoUAVCompletion
            result = hrl.run_episode(env, *agents, cfg, log, number, dual, training, not training)
        result.update(log.measures())
        if algorithm != "proposed" and (result["hire_rate"] != 0 or result["hiring_cost_total"] != 0):
            raise AssertionError("RSU-only invariant failed")
        log.log_episode(result)
        log.event("baseline_end", {"status": "complete"})
        return result
    finally:
        hrl.FastPolicyCompletion = original
        log.close()


def policy_state(agents):
    return [{"model": a.net.state_dict(), "optimizer": a.opt.state_dict(),
             "trajectories": a.trajectories, "updates": a.update_count} for a in agents]


def restore_agents(agents, payload):
    if len(agents) != len(payload):
        raise ValueError("checkpoint policy count mismatch")
    for a, p in zip(agents, payload):
        a.net.load_state_dict(p["model"])
        a.opt.load_state_dict(p["optimizer"])
        a.trajectories = p["trajectories"]
        a.update_count = p["updates"]


class Budget:
    def __init__(self, seconds, reserve):
        self.started, self.seconds, self.reserve = time.monotonic(), seconds, reserve
        self.reason = ""
        self.old = {}
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.old[sig] = signal.signal(sig, self.stop)

    def stop(self, signum, _frame):
        self.reason = signal.Signals(signum).name

    def expired(self, estimate=0):
        if self.reason:
            return True
        if time.monotonic() - self.started + estimate + self.reserve >= self.seconds:
            self.reason = "walltime budget"
            return True
        return False

    def close(self):
        for sig, handler in self.old.items():
            signal.signal(sig, handler)


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def run_train(args):
    root = args.out.resolve()
    cfg = replace(read_config(args.config), device=args.device)
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    if not 0 < args.episodes < 1_000_000:
        raise ValueError("training episodes must be in [1, 999999]")
    if cfg.episode_offset != 0:
        raise ValueError("Training uses episode IDs starting at zero; use a source config with episode_offset=0")
    if args.algorithm == "hppo_rsu" and cfg.reward_mode != "dpp":
        raise ValueError("The requested proposed ablation uses the current DPP objective")
    spec = {"config": jsonable(asdict(cfg)), "algorithm": args.algorithm,
            "episodes": args.episodes, "val_every": args.val_every, "val_episodes": args.val_episodes,
            "source_sha256": source_hashes(), "version": VERSION, "base_commit": BASE_COMMIT,
            "qoe_weights": QOE_WEIGHTS,
            "learning_settings": ({"hidden_dims": [128, 64], "actor_lr": 1e-5, "critic_lr": 1e-4,
                                   "gamma": 0.95, "gae_lambda": 0.95, "minibatch_size": 64,
                                   "update_every_episodes": 1} if args.algorithm == "ndtvs"
                                  else "identical to proposed config")}
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Nonempty output directory; use --resume or another --out")
    root.mkdir(parents=True, exist_ok=True)
    hrl.seed_all(cfg.seed)
    torch.set_num_threads(cfg.torch_num_threads)
    agents = make_agents(cfg, args.algorithm)
    rows, vals, start, dual, best = [], [], 0, cfg.dual_init, -math.inf
    durations = []
    if args.resume:
        saved = load_bundle(root / "latest.pt")
        if jsonable(saved["spec"]) != jsonable(spec):
            raise ValueError("Resume configuration/source/budget changed; restore the original settings")
        restore_agents(agents, saved["policies"])
        restore_rng(saved["rng"])
        rows, vals, start, dual, best = (saved[k] for k in ("rows", "validation", "next_episode", "dual", "best"))
        durations = saved["durations"]
    atomic(root / "experiment.json", spec)
    attempt = root / "segments" / str(time.time_ns())
    attempt.mkdir(parents=True)
    budget = Budget(args.walltime_seconds, args.reserve_seconds)

    def save(next_episode):
        payload = {"spec": spec, "policies": policy_state(agents), "rng": rng_state(),
                   "next_episode": next_episode, "rows": rows, "validation": vals,
                   "dual": dual, "best": best, "durations": durations}
        atomic(root / "latest.pt", payload, binary=True)
        write_rows(root / "training.csv", rows)
        atomic(root / "validation.json", vals)
        return payload

    def validate(completed):
        nonlocal best
        if not completed or any(v["trained_episodes"] == completed for v in vals):
            return
        if completed % args.val_every and completed != args.episodes:
            return
        if budget.expired(max(durations[-10:], default=0) * (args.val_episodes + 1)):
            return  # Resume retries this validation before advancing training.
        state = rng_state()
        try:
            scores = [episode(cfg, args.algorithm, agents, 1_000_000 + i, dual, False,
                              attempt / f"val_{completed:06d}_{i:03d}", False)
                      for i in range(args.val_episodes)]
        finally:
            restore_rng(state)
        key = "qoe_surrogate_per_user_slot" if args.algorithm == "ndtvs" else "dpp_cost_per_user_slot"
        score = float(np.mean([r[key] for r in scores])) * (1 if args.algorithm == "ndtvs" else -1)
        vals.append({"trained_episodes": completed, "selection_score": score,
                     "selection_metric": key, "per_episode": scores})
        improved = score > best
        if improved:
            best = score
        payload = save(completed)
        if improved:
            atomic(root / "best.pt", payload, binary=True)

    try:
        if not args.resume:
            save(0)  # Even an immediate hard kill has a restart point.
        completed = start
        if args.resume:
            # Recover a crash between committing validation and exporting best.pt.
            if vals and vals[-1]["trained_episodes"] == start and vals[-1]["selection_score"] == best:
                atomic(root / "best.pt", saved, binary=True)
            validate(start)
        for ep in range(start, args.episodes):
            if budget.expired(max(durations[-10:], default=0) * 1.5):
                break
            began = time.monotonic()
            trace = args.trace_every > 0 and ((ep + 1) % args.trace_every == 0 or ep == 0)
            directory = attempt / f"train_{ep:06d}"
            row = episode(cfg, args.algorithm, agents, ep, dual, True, directory, trace)
            if args.algorithm == "ndtvs":
                row.update(agents[0].update())
            elif (ep + 1) % cfg.frame_update_every_episodes == 0 or ep + 1 == args.episodes:
                row.update(agents[0].update())
            row["trace_dir"] = str(directory.relative_to(root)) if trace else ""
            rows.append(row)
            completed = ep + 1
            durations.append(time.monotonic() - began)
            save(completed)  # Commit before potentially expensive validation.
            validate(completed)
            print(f"{args.algorithm} {completed}/{args.episodes} "
                  f"QoE={row['qoe_surrogate_per_user_slot']:.5f} stall={row['stall_ratio']:.4f} "
                  f"{durations[-1]:.2f}s", flush=True)
            if args.max_new_episodes and completed - start >= args.max_new_episodes:
                budget.reason = "max-new-episodes"
                break
        done = completed == args.episodes
        atomic(root / "status.json", {"status": "complete" if done else "paused",
               "completed_episodes": completed, "target_episodes": args.episodes,
               "reason": budget.reason, "convergence": "not certified",
               "validation_points": len(vals), "checkpoint": "latest.pt",
               "final_validation_pending": done and not any(v["trained_episodes"] == completed for v in vals)})
        return 0 if done else 75
    finally:
        budget.close()


def run_eval(args):
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise FileExistsError("Evaluation output must be empty")
    if args.algorithm == "proposed":
        cfg = replace(read_config(args.config), device=args.device)
        agents = make_agents(cfg, "proposed")
        extras = [a.load(p) for a, p in zip(agents, (args.frame_checkpoint, args.slot_checkpoint))]
        if not extras[0].get("pair_id") or extras[0]["pair_id"] != extras[1].get("pair_id"):
            raise ValueError("Proposed checkpoints must be a matched pair")
        provenance = {"pair": extras, "files": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                      for p in (args.frame_checkpoint, args.slot_checkpoint)}}
    else:
        saved = load_bundle(args.checkpoint)
        if saved["spec"]["algorithm"] != args.algorithm or saved["spec"]["source_sha256"] != source_hashes():
            raise ValueError("Checkpoint algorithm/source mismatch")
        cfg = HPPOConfig(**{k: tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
                           if isinstance(v, list) else v for k, v in saved["spec"]["config"].items()})
        cfg = replace(cfg, device=args.device)
        agents = make_agents(cfg, args.algorithm)
        restore_agents(agents, saved["policies"])
        provenance = {"trained_episodes": saved["next_episode"], "file": str(args.checkpoint),
                      "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    cfg = replace(cfg, seed=args.scenario_seed)
    torch.set_num_threads(cfg.torch_num_threads)
    hrl.seed_all(args.scenario_seed)
    if args.offset < 2_000_000:
        raise ValueError("Final evaluation uses offset >= 2000000, disjoint from training/validation")
    spec = {"algorithm": args.algorithm, "config": asdict(cfg), "offset": args.offset,
            "scenario_seed": args.scenario_seed, "target_episodes": args.episodes,
            "provenance": provenance, "source_sha256": source_hashes(), "qoe_weights": QOE_WEIGHTS}
    rows, durations = [], []
    if args.resume:
        previous = json.loads((args.out / "evaluation_partial.json").read_text())
        if previous["spec"] != jsonable(spec):
            raise ValueError("Evaluation resume settings/checkpoints differ")
        rows, durations = previous["rows"], previous["durations"]
    budget = Budget(args.walltime_seconds, args.reserve_seconds)
    try:
        for i in range(len(rows), args.episodes):
            if budget.expired(max(durations, default=0) * 1.5):
                break
            began = time.monotonic()
            rows.append(episode(cfg, args.algorithm, agents, args.offset + i, cfg.dual_init, False,
                                args.out / f"ep_{i:04d}", args.trace))
            durations.append(time.monotonic() - began)
            atomic(args.out / "evaluation_partial.json", {"spec": spec, "rows": rows, "durations": durations})
        atomic(args.out / "evaluation_partial.json", {"spec": spec, "rows": rows, "durations": durations})
    finally:
        budget.close()
    if len(rows) != args.episodes:
        print("Evaluation paused; resubmit with --resume", flush=True)
        return 75
    atomic(args.out / "evaluation.json", {**spec, "per_episode": rows})
    write_rows(args.out / "evaluation.csv", rows)
    print(json.dumps({k: float(np.mean([r[k] for r in rows])) for k in
                     ("qoe_surrogate_per_user_slot", "stall_ratio", "average_quality_utility", "hire_rate")}, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "eval"))
    p.add_argument("--algorithm", choices=("ndtvs", "hppo_rsu", "proposed"), required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-episodes", type=int, default=10)
    p.add_argument("--trace-every", type=int, default=25)
    p.add_argument("--walltime-seconds", type=float, default=82800)
    p.add_argument("--reserve-seconds", type=float, default=900)
    p.add_argument("--max-new-episodes", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--frame-checkpoint", type=Path)
    p.add_argument("--slot-checkpoint", type=Path)
    p.add_argument("--scenario-seed", type=int, default=2026)
    p.add_argument("--offset", type=int, default=2_000_000)
    p.add_argument("--trace", action="store_true")
    a = p.parse_args(argv)
    if min(a.episodes, a.val_every, a.val_episodes) <= 0 or a.reserve_seconds < 0 or a.walltime_seconds <= 0:
        p.error("positive episode/validation/budget values required")
    if a.mode == "train" and (a.algorithm == "proposed" or not a.config):
        p.error("train requires --config and algorithm ndtvs or hppo_rsu")
    if a.mode == "eval" and ((a.algorithm == "proposed" and not (a.config and a.frame_checkpoint and a.slot_checkpoint))
                             or (a.algorithm != "proposed" and not a.checkpoint)):
        p.error("eval requires the algorithm's checkpoint(s) and proposed configuration")
    a.out = a.out.resolve()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_name(a.out.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_train(a) if a.mode == "train" else run_eval(a)


if __name__ == "__main__":
    raise SystemExit(main())
