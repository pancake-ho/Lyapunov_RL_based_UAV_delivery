"""Commands backed by config.py: inspect, smoke, sweep, plot, visuals, all."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import argparse
import json
from dataclasses import replace
from baseline.NDTVS.evaluation.benchmark.settings import load, validate, DEFAULT_CONFIG


def inspect(s):
    """Read-only inspection; works before the requested checkpoint is ready."""
    import baseline.NDTVS.api as c
    from baseline.NDTVS.evaluation.benchmark.models import load_models, PATH_KEYS
    from baseline.NDTVS.evaluation.benchmark.radio import sanity
    missing = [str(item[k]) for item in s.MODELS for k in PATH_KEYS if k in item and not Path(item[k]).is_file()]
    config_path = next((Path(m["config"]) for m in s.MODELS if "config" in m and Path(m["config"]).is_file()),
                       s.PROJECT_ROOT/"baseline/NDTVS/configs/ndtvs_V50_seed2026.json")
    print(json.dumps(dict(snr_mode=s.SNR_MODE,cost_mode=s.COST_MODE,
        scenario_seeds=s.SEEDS,episodes_per_seed=s.EPISODES_PER_SEED,models=[m["name"] for m in s.MODELS],missing_files=missing),indent=2))
    if config_path.is_file():
        cfg = c.read_config(config_path)
        for mode in ((s.SNR_MODE,) if s.SNR_MODE else ("transmit","offset")):
            settings = __import__("types").SimpleNamespace(**vars(s))
            settings.SNR_MODE = mode
            print(json.dumps(dict(radio=mode,rows=sanity(cfg,settings)),indent=2))
    if not missing:
        _,_,provenance,sizes = load_models(s.MODELS,"cpu",c)
        print(json.dumps(dict(checkpoints=provenance,parameter_counts=sizes),indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("inspect","smoke","sweep","plot","visuals","all"))
    p.add_argument("--settings",type=Path,default=DEFAULT_CONFIG)
    p.add_argument("--result-mode",choices=("smoke","sweep"),default="sweep")
    a = p.parse_args(argv)
    s = validate(load(a.settings),require_snr=a.mode not in ("inspect","plot","visuals"))
    if a.mode == "inspect":
        return inspect(s)
    if a.mode in ("smoke","sweep","all"):
        from baseline.NDTVS.evaluation.benchmark.runner import run
        if a.mode == "all":
            for mode in ("smoke","sweep"):
                code = run(s,mode)
                if code:
                    return code
        else:
            return run(s,a.mode)
    if a.mode in ("plot","all"):
        from baseline.NDTVS.plot.benchmark import report
        report(s,a.result_mode)
    if a.mode == "visuals" or (a.mode == "all" and s.EXPORT_ANIMATIONS):
        from baseline.NDTVS.plot.benchmark_animation import export
        export(s,a.result_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
