"""Load a Python config and validate explicit scientific choices."""
import importlib.util
from pathlib import Path
import math


DEFAULT_CONFIG = Path(__file__).with_name("config.py")


def load(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "ndtvs_benchmark_settings", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SETTINGS_PATH = path
    return module


def validate(s, require_snr=True):
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    require(s.DEVICE in ("cpu", "cuda"), "DEVICE must be cpu or cuda")
    require(
        s.SEEDS
        and len(set(s.SEEDS)) == len(s.SEEDS)
        and all(isinstance(n, int) and n >= 0 for n in s.SEEDS),
        "Bad scenario seeds",
    )
    require(
        isinstance(s.EPISODES_PER_SEED, int)
        and s.EPISODES_PER_SEED > 0,
        "Bad episode count",
    )
    require(
        s.TEST_EPISODE_OFFSET >= 0 and s.SMOKE_EPISODE_OFFSET >= 0,
        "Bad scenario offset",
    )
    require(
        not (
            s.TEST_EPISODE_OFFSET
            <= s.SMOKE_EPISODE_OFFSET
            < s.TEST_EPISODE_OFFSET + s.EPISODES_PER_SEED
        ),
        "Smoke/test IDs overlap",
    )
    require(
        s.SNR_MODE in (None, "transmit", "received", "offset"),
        "Bad SNR_MODE",
    )

    if require_snr:
        require(
            s.SNR_MODE is not None,
            "Choose SNR_MODE in evaluation/benchmark/config.py; "
            "transmit and received SNR are different.",
        )

    levels = (
        s.SNR_OFFSETS_DB if s.SNR_MODE == "offset" else s.SNR_DB
    )
    require(
        levels
        and len(set(levels)) == len(levels)
        and all(math.isfinite(float(value)) for value in levels),
        "Bad SNR levels",
    )

    if s.SNR_MODE == "received":
        require(
            s.REFERENCE_DISTANCE_M is not None
            and math.isfinite(s.REFERENCE_DISTANCE_M)
            and s.REFERENCE_DISTANCE_M >= 0,
            "Set REFERENCE_DISTANCE_M explicitly for received SNR",
        )

    require(
        s.MODELS
        and len({model["name"] for model in s.MODELS}) == len(s.MODELS),
        "Duplicate/empty models",
    )
    for model in s.MODELS:
        require(
            model["name"].replace("_", "").replace("-", "").isalnum(),
            "Unsafe model name",
        )
        require(
            model["algorithm"] in ("proposed", "ndtvs", "hppo_rsu"),
            "Unsupported algorithm",
        )

        if model["algorithm"] == "proposed":
            require(
                all(key in model for key in ("config", "runtime")),
                f"Missing proposed config/runtime: {model['name']}",
            )
            resume = "resume_checkpoint" in model
            pair = (
                "frame_checkpoint" in model
                and "slot_checkpoint" in model
            )
            any_pair = (
                "frame_checkpoint" in model
                or "slot_checkpoint" in model
            )
            require(
                (resume and not any_pair) or (not resume and pair),
                "Select resume_checkpoint OR both frame/slot checkpoints: "
                + model["name"],
            )
            require(
                "checkpoint" not in model,
                f"Use a proposed checkpoint key: {model['name']}",
            )
        else:
            require(
                "checkpoint" in model,
                f"Missing model checkpoint: {model['name']}",
            )

    require(
        s.COST_MODE in (None, "reevaluate", "accounting"),
        "Bad COST_MODE",
    )
    if s.COST_MODE:
        require(
            s.HIRING_COSTS is not None
            and len(s.HIRING_COSTS) > 0
            and len(set(s.HIRING_COSTS)) == len(s.HIRING_COSTS)
            and all(
                math.isfinite(value) and value >= 0
                for value in s.HIRING_COSTS
            ),
            "Set nonnegative HIRING_COSTS",
        )
        require(
            s.QOE_COST_WEIGHT is not None
            and math.isfinite(s.QOE_COST_WEIGHT)
            and s.QOE_COST_WEIGHT >= 0,
            "Set QOE_COST_WEIGHT explicitly",
        )
        require(
            s.COST_SNR_DB in levels,
            "COST_SNR_DB must be in the SNR grid",
        )

    require(
        s.VISUAL_SLOT_STRIDE > 0
        and s.VISUAL_MAX_IMAGES > 0
        and s.GIF_FPS > 0,
        "Bad visualization settings",
    )
    require(
        s.BOOTSTRAP_SAMPLES >= 100
        and s.WALLTIME_SECONDS > s.RESERVE_SECONDS >= 0,
        "Bad bootstrap/budget",
    )
    require(s.MAX_NEW_EPISODES >= 0, "Bad MAX_NEW_EPISODES")
    return s