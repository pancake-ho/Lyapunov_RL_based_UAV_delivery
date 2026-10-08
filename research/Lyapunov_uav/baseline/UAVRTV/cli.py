"""Config-driven inspect/preflight/smoke/train/eval/plot/audit commands."""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict, replace
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.UAVRTV.common.settings import load, validate, common_config, experiment_spec, DEFAULT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "preflight", "smoke", "train", "eval", "plot", "audit"))
    parser.add_argument("--settings", type=Path, default=DEFAULT)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args(argv)
    s = validate(load(args.settings))
    if args.mode == "plot":
        from baseline.UAVRTV.plot.learning import plot
        return plot(s)
    if args.mode == "audit":
        if args.directory is None:
            parser.error("audit requires --directory pointing to one traced episode")
        from baseline.UAVRTV.evaluation.audit import audit
        print(json.dumps(audit(args.directory), indent=2))
        return 0

    import baseline.NDTVS.api as c
    if args.mode == "inspect":
        cfg, verified = common_config(s)
        from baseline.UAVRTV.environment.shared import SharedUAVRTVEnv
        from baseline.UAVRTV.models.sac import SACAgent
        from baseline.UAVRTV.common.checkpoint import read
        settings = SimpleNamespace(**vars(s))
        settings.DEVICE = "cpu"
        env = SharedUAVRTVEnv(cfg)
        agent = SACAgent(env.obs_dim, env.act_dim, settings)
        info = dict(source_config=str(s.SOURCE_CONFIG), source_verified=verified,
            scenario=c.jsonable(asdict(cfg)), reward=experiment_spec(s, cfg)["reward"],
            sac_learning=experiment_spec(s, cfg)["learning"],
            observation_dim=env.obs_dim, action_dim=env.act_dim, networks=agent.sizes(), out=str(s.OUT))
        if (s.OUT / "latest.pt").exists():
            b = read(s.OUT / "latest.pt")
            info["checkpoint"] = dict(completed_episodes=b["next_episode"], sac_updates=b["agent"]["updates"],
                replay_transitions=b["replay"]["n"], best_validation_score=b["best"])
        print(json.dumps(info, indent=2))
        return 0
    c.require_device(s.DEVICE)
    if args.mode == "eval":
        from baseline.UAVRTV.evaluation.sweep import sweep
        return sweep(s)
    cfg, verified = common_config(s)
    if args.mode == "preflight":
        from baseline.UAVRTV.evaluation.preflight import run
        run(s, cfg, experiment_spec(s, cfg))
        return 0
    if args.mode == "smoke":
        s = SimpleNamespace(**vars(s))
        s.OUT = Path(s.OUT).with_name(Path(s.OUT).name + "_smoke")
        s.TRAIN_EPISODES, s.VAL_EVERY, s.VAL_EPISODES = 2, 1, 1
        s.HIDDEN_DIMS, s.BATCH_SIZE, s.BUFFER_SIZE = (64, 64), 32, 4096
        s.START_TRANSITIONS, s.MAX_NEW_EPISODES = 0, 0
        print("SMOKE: shared physical scenario unchanged; two-episode small SAC is not a research result", flush=True)
    from baseline.UAVRTV.training.engine import train
    code = train(s, cfg)
    if code == 0:
        from baseline.UAVRTV.plot.learning import plot
        plot(s)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
