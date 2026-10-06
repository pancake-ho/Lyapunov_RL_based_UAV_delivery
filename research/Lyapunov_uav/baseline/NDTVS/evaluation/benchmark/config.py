"""All experiment and submission settings live here."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT = PROJECT_ROOT / "baseline/NDTVS/runs/evaluation/snr_v_service_resume"

DEVICE = "cuda"
SEEDS = (2026, 2027, 2028)
EPISODES_PER_SEED = 30
TEST_EPISODE_OFFSET = 7_000_000
SMOKE_EPISODE_OFFSET = 8_000_000

# Relative SNR offsets from the trained channel.
SNR_MODE = "offset"
SNR_DB = (25, 30, 35, 40, 45)  # Inactive in offset mode.
SNR_OFFSETS_DB = (-10, -5, 0, 5, 10)
REFERENCE_DISTANCE_M = None

V50_RUN = PROJECT_ROOT / "proposed/outputs/hppo/hrl_resume_job145847"
NDTVS_RUN = (
    PROJECT_ROOT
    / "baseline/NDTVS/runs/common_gpu/ndtvs_paper_261006_seed2026_ep1000"
)

MODELS = [
    dict(
        name="proposed_V50",
        algorithm="proposed",
        expected_v=50,
        config=V50_RUN / "resolved_config.json",
        runtime=V50_RUN / "runtime.json",
        resume_checkpoint=V50_RUN / "checkpoints/resume_latest.pt",
        selection=(
            "user-selected longest-trained run; "
            "last committed resume checkpoint"
        ),
    ),
    dict(
        name="ndtvs",
        algorithm="ndtvs",
        checkpoint=NDTVS_RUN / "best.pt",
        completion_status=NDTVS_RUN / "status.json",
        selection="fixed-validation mean-QoE selected best.pt",
    ),
]

# Additional proposed V models can use either checkpoint format.
# Specify each model's own config/runtime and expected_v.

COST_MODE = None
HIRING_COSTS = None
QOE_COST_WEIGHT = None
COST_SNR_DB = None

TRACE_ALL = False
EXPORT_ANIMATIONS = True
VISUAL_SEEDS = (2026,)
VISUAL_SNR_DB = None
VISUAL_SLOT_STRIDE = 10
VISUAL_MAX_IMAGES = 30
GIF_FPS = 2

BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 572913
WALLTIME_SECONDS = 82800
RESERVE_SECONDS = 900
MAX_NEW_EPISODES = 0
RESUME = True

PYTHON = Path("/data/surt321/anaconda3/envs/lab/bin/python")
PARTITION = "batch_eebme_ugrad"
GPUS = 1
CPUS = 16
MEMORY = "29G"
TIME_LIMIT = "1-00:00:00"
JOB_NAME = "ndtvs-v-snr-eval"
LOG_DIR = PROJECT_ROOT / "baseline/NDTVS/slurm_logs"