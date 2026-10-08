"""Episode commits; validation RNG isolation; Slurm pause and exact recovery."""
import copy
import signal
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch

import baseline.NDTVS.api as c
from baseline.UAVRTV.common.settings import experiment_spec
from baseline.UAVRTV.common.checkpoint import FORMAT, source_hashes, atomic_checkpoint, read
from baseline.UAVRTV.environment.shared import SharedUAVRTVEnv
from baseline.UAVRTV.models.sac import SACAgent, ReplayBuffer
from baseline.UAVRTV.training.rollout import run_episode
from baseline.UAVRTV.evaluation.preflight import run as preflight
from baseline.UAVRTV.evaluation.audit import audit


def validation(s, cfg, agent, root, completed):
    state = c.rng_state()
    try:
        rows = []
        for i in range(s.VAL_EPISODES):
            directory = root / "validation" / f"ep{completed}_{i}_{uuid.uuid4().hex[:8]}"
            row = run_episode(replace(cfg, seed=s.VAL_SEED), s, agent, s.VAL_OFFSET + i, directory, trace=i == 0)
            if i == 0:
                audit(directory)
            rows.append(row)
        score = float(np.mean([r["uavrtv_reward_per_region_slot"] for r in rows]))
        qualities = [r["average_quality_utility"] for r in rows if r["quality_utility_defined"]]
        return dict(trained_episodes=completed, selection_score=score,
            mean_stall_time_ratio=float(np.mean([r["stall_time_ratio"] for r in rows])),
            mean_quality=float(np.mean(qualities)) if qualities else None,
            quality_defined_episodes=len(qualities), rows=rows)
    finally:
        c.restore_rng(state)


def train(s, cfg):
    spec, sources = experiment_spec(s, cfg), source_hashes()
    root = Path(s.OUT)
    latest = root / "latest.pt"
    saved = read(latest) if latest.exists() else None
    if saved:
        if not s.RESUME or saved["spec"] != spec or saved["kind"] != "resume":
            raise ValueError("Resume settings differ; use a new OUT")
        if s.TRAIN_EPISODES < saved["next_episode"]:
            raise ValueError("Training budget is below the completed checkpoint")
    else:
        if root.exists() and any(p.name != "preflight" for p in root.iterdir()):
            raise ValueError("Nonempty run without latest.pt; select a new OUT")
        root.mkdir(parents=True, exist_ok=True)

    preflight(s, cfg, spec)
    c.hrl.seed_all(s.TRAIN_SEED)
    torch.set_num_threads(s.TORCH_THREADS)
    shape = SharedUAVRTVEnv(cfg)
    agent = SACAgent(shape.obs_dim, shape.act_dim, s)
    replay = ReplayBuffer(shape.obs_dim, shape.act_dim, s.BUFFER_SIZE)
    rng = np.random.default_rng(s.TRAIN_SEED + 93071)
    counters = dict(transitions=0)
    next_episode, rows, validations, best = 0, [], [], None
    if saved:
        agent.restore(saved["agent"], training=True)
        replay.restore(saved["replay"])
        counters = saved["counters"]
        rng.bit_generator.state = saved["sampler_rng"]
        c.restore_rng(saved["rng"])
        next_episode, rows, validations, best = saved["next_episode"], saved["rows"], saved["validation"], saved["best"]
    else:
        c.atomic(root / "resolved_config.json", dict(config=asdict(cfg), experiment=spec))
        c.atomic(root / "runtime.json", dict(torch=torch.__version__, device=s.DEVICE,
            source_sha256=sources, source_config=str(s.SOURCE_CONFIG),
            note="UAVRTV shared-environment adaptation; old SAC checkpoints are incompatible"))
        c.atomic(root / "model_size.json", dict(networks=agent.sizes(),
            policy_parameters=agent.sizes()["actor"],
            online_parameters=sum(agent.sizes()[k] for k in ("actor", "q1", "q2")),
            all_parameters=sum(agent.sizes().values()),
            note="Target critics are counted separately from online actor/critics."))

    stop, started, longest, completed_here = [], time.monotonic(), 0., 0
    def request_stop(signum, frame):
        stop.append(signal.Signals(signum).name)
    previous_handlers = {sig: signal.signal(sig, request_stop)
                         for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)}
    def bundle(kind="resume"):
        b = dict(format=FORMAT, kind=kind, spec=spec, source_sha256=sources,
            agent=agent.state(training=kind == "resume"), next_episode=next_episode,
            validation=validations, best=best, rows=rows)
        if kind == "resume":
            b.update(replay=replay.state(), counters=dict(counters), sampler_rng=copy.deepcopy(rng.bit_generator.state), rng=c.rng_state())
        return b
    def commit():
        atomic_checkpoint(latest, bundle())
        c.write_rows(root / "training.csv", rows)
        c.write_rows(root / "validation.csv", [{k: v for k, v in x.items() if k != "rows"} for x in validations])
    def status(state, reason=""):
        c.atomic(root / "status.json", dict(status=state, reason=reason, completed_episodes=next_episode,
            target_episodes=s.TRAIN_EPISODES, final_validation_pending=not any(v["trained_episodes"] == next_episode for v in validations)))
    try:
        if not saved:
            commit()
        # A hard cancellation during scheduled validation leaves a complete
        # training commit. Re-run that missing validation before advancing.
        if saved and next_episode and next_episode % s.VAL_EVERY == 0 and not any(
                v["trained_episodes"] == next_episode for v in validations):
            point = validation(s, cfg, agent, root, next_episode)
            validations.append(point)
            if best is None or point["selection_score"] > best:
                best = point["selection_score"]
                atomic_checkpoint(root / "best.pt", bundle("best"))
            commit()
        while next_episode < s.TRAIN_EPISODES:
            if stop or (s.MAX_NEW_EPISODES and completed_here >= s.MAX_NEW_EPISODES) or (
                time.monotonic() - started + max(1.5 * longest, 0) >= s.WALLTIME_SECONDS - s.RESERVE_SECONDS):
                status("PAUSED", ",".join(stop) or "budget")
                print(f"PAUSED ep={next_episode}; resubmit the same config to continue", flush=True)
                return 75
            tick = time.monotonic()
            traced = next_episode % s.TRACE_EVERY == 0
            directory = root / "episodes" / f"train_{next_episode}_{uuid.uuid4().hex[:8]}"
            row = run_episode(cfg, s, agent, next_episode, directory, training=True,
                replay=replay, rng=rng, counters=counters, trace=traced)
            if traced:
                audit(directory)
            rows.append(dict(row, sac_updates=agent.updates, transitions=counters["transitions"],
                             trace_dir=str(directory.relative_to(root)) if traced else ""))
            next_episode += 1
            completed_here += 1
            commit()  # Persist before any potentially interrupted validation.
            status("RUNNING")
            if next_episode % s.VAL_EVERY == 0:
                point = validation(s, cfg, agent, root, next_episode)
                validations.append(point)
                if best is None or point["selection_score"] > best:
                    best = point["selection_score"]
                    atomic_checkpoint(root / "best.pt", bundle("best"))
                commit()
                status("RUNNING")
            longest = max(longest, time.monotonic() - tick)
            print(f"uavrtv {next_episode}/{s.TRAIN_EPISODES} reward={row['uavrtv_reward_per_region_slot']:.5f} "
                  f"stall={row['stall_time_ratio']:.4f} hire={row['hire_rate']:.4f} updates={agent.updates}", flush=True)
        if not any(v["trained_episodes"] == next_episode for v in validations):
            point = validation(s, cfg, agent, root, next_episode)
            validations.append(point)
            if best is None or point["selection_score"] > best:
                best = point["selection_score"]
                atomic_checkpoint(root / "best.pt", bundle("best"))
            commit()
        status("COMPLETE")
        print("COMPLETE " + str(root.resolve()), flush=True)
        return 0
    except Exception as exc:
        committed = read(latest, sources=False)
        c.atomic(root / "status.json", dict(status="ERROR", reason=str(exc),
            completed_episodes=committed["next_episode"], recovery="resume latest.pt; incomplete episode data is not committed"))
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
