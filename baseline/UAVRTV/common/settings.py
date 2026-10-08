"""Python settings and the verified, unchanged common scenario."""
import importlib.util
import math
from dataclasses import asdict, replace
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "config.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path=DEFAULT):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("uavrtv_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SETTINGS_PATH = path
    return module


def validate(s):
    require(s.DEVICE in ("cpu", "cuda"), "Bad DEVICE")
    for key in ("TRAIN_EPISODES", "BATCH_SIZE", "BUFFER_SIZE", "VAL_EVERY", "VAL_EPISODES",
                "TEST_EPISODES", "PREFLIGHT_EPISODES", "UPDATES_PER_SLOT", "TORCH_THREADS"):
        require(type(getattr(s, key)) is int and getattr(s, key) > 0, "Bad " + key)
    require(s.BATCH_SIZE <= s.BUFFER_SIZE and s.START_TRANSITIONS >= 0, "Bad replay/warmup")
    require(s.HIDDEN_DIMS and all(type(n) is int and n > 0 for n in s.HIDDEN_DIMS), "Bad network")
    require(all(math.isfinite(float(getattr(s, k))) and getattr(s, k) >= 0
                for k in ("BETA", "DELTA", "PHI", "VARSIGMA")), "Bad reward coefficients")
    require(s.REWARD_SCALE > 0 and math.isfinite(s.REWARD_SCALE), "Bad reward scale")
    require(s.LR > 0 and math.isfinite(s.LR) and s.ALPHA > 0 and math.isfinite(s.ALPHA)
            and 0 <= s.GAMMA < 1 and 0 < s.TAU <= 1, "Bad SAC settings")
    require(s.WALLTIME_SECONDS > s.RESERVE_SECONDS >= 0 and s.MAX_NEW_EPISODES >= 0, "Bad budget")
    require(s.TRACE_EVERY > 0 and s.DOMINANCE_RATIO > 1, "Bad trace/diagnostic settings")
    require(s.TRAIN_SEED >= 0 and s.VAL_SEED >= 0 and s.VAL_OFFSET >= 0 and s.TEST_OFFSET >= 0, "Bad seeds/IDs")
    require(s.TRAIN_EPISODES < min(s.VAL_OFFSET, s.TEST_OFFSET), "Training/evaluation IDs overlap")
    require(not (s.TEST_OFFSET < s.VAL_OFFSET + s.VAL_EPISODES
                 and s.VAL_OFFSET < s.TEST_OFFSET + s.TEST_EPISODES), "Validation/test IDs overlap")
    require(s.TEST_SEEDS and len(set(s.TEST_SEEDS)) == len(s.TEST_SEEDS)
            and all(type(v) is int and v >= 0 for v in s.TEST_SEEDS), "Bad test seeds")
    require(s.SNR_OFFSETS_DB and len(set(s.SNR_OFFSETS_DB)) == len(s.SNR_OFFSETS_DB)
            and all(math.isfinite(float(v)) for v in s.SNR_OFFSETS_DB), "Bad SNR offsets")
    return s


def common_config(s):
    import baseline.NDTVS.api as c
    from baseline.NDTVS.evaluation.checks import read_json
    cfg = c.read_config(Path(s.SOURCE_CONFIG))
    raw = read_json(s.SOURCE_CONFIG)
    runtime = read_json(s.SOURCE_RUNTIME)
    expected = {(k if k.startswith("proposed/") else "proposed/" + k): v
                for k, v in runtime["code_sha256"].items()}
    current = {k: v for k, v in c.source_hashes().items() if k.startswith("proposed/")}
    verification = c.verify_shared_sources(expected, current, raw.get("config", raw))
    require(cfg.num_quality_levels == 4, "Four common PSNR levels required")
    require(math.isclose(cfg.playback_chunks_per_slot, cfg.slot_duration_s, abs_tol=1e-10),
            "Shared NDTVS observer requires one-second chunks")
    cfg = replace(cfg, device=s.DEVICE, seed=s.TRAIN_SEED, episode_offset=0,
                  write_human_debug_log=False, console_log_every_slots=0,
                  torch_num_threads=s.TORCH_THREADS)
    return cfg, verification


def experiment_spec(s, cfg):
    import baseline.NDTVS.api as c
    from baseline.UAVRTV.rewards.paper import definition
    learning = {key: getattr(s, key) for key in ("HIDDEN_DIMS", "LR", "GAMMA", "TAU", "ALPHA",
        "AUTO_ALPHA", "BATCH_SIZE", "BUFFER_SIZE", "START_TRANSITIONS", "UPDATES_PER_SLOT")}
    return c.jsonable(dict(version="uavrtv-shared-sac-v1", common_config=asdict(cfg),
        learning=learning, reward=definition(s, cfg), validation=dict(seed=s.VAL_SEED,
        offset=s.VAL_OFFSET, episodes=s.VAL_EPISODES, every=s.VAL_EVERY)))
