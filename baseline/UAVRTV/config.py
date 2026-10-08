"""Edit settings here; no shell exports or new Conda environment are needed."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = PROJECT_ROOT / "proposed/outputs/hppo/hrl_resume_job145847"
SOURCE_CONFIG = SOURCE_RUN / "resolved_config.json"
SOURCE_RUNTIME = SOURCE_RUN / "runtime.json"
OUT = PROJECT_ROOT / "baseline/UAVRTV/runs/shared_sac_seed2026"
DEVICE = "cuda"
TRAIN_SEED = 2026
TRAIN_EPISODES = 500  # Initial budget; extend this value and resubmit to continue.
RESUME = True

# Paper coefficients; phi multiplies adapted rebuffer duration in seconds.
BETA = 1.0
DELTA = 1e-6
PHI = 10.0
VARSIGMA = 0.01
# lambda_h and hiring_cost_per_frame are taken from SOURCE_CONFIG unchanged.
REWARD_SCALE = 1.0

# Preserve the existing UAVRTV SAC architecture and learning defaults.
HIDDEN_DIMS = (256, 256)
LR = 1e-3
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
AUTO_ALPHA = True
BATCH_SIZE = 256
BUFFER_SIZE = 200_000
START_TRANSITIONS = 2_000
UPDATES_PER_SLOT = 1
TORCH_THREADS = 1

VAL_EVERY = 20
VAL_EPISODES = 5
VAL_SEED = 2026
VAL_OFFSET = 10_000_000
TEST_SEEDS = (2026, 2027, 2028)
TEST_EPISODES = 30
TEST_OFFSET = 7_000_000
SNR_OFFSETS_DB = (-10, -5, 0, 5, 10)
EVAL_OUT = PROJECT_ROOT / "baseline/UAVRTV/runs/evaluation/shared_sac"

# Runs before learning. Fixed profiles expose every reward component.
PREFLIGHT_EPISODES = 1
DOMINANCE_RATIO = 10.0  # Diagnostic flag only; never changes coefficients.
TRACE_EVERY = 25
WALLTIME_SECONDS = 82_800
RESERVE_SECONDS = 900
MAX_NEW_EPISODES = 0

PYTHON = Path("/data/surt321/anaconda3/envs/lab/bin/python")
PARTITION = "batch_eebme_ugrad"
GPUS = 1
CPUS = 16
MEMORY = "29G"
TIME_LIMIT = "1-00:00:00"
JOB_NAME = "uavrtv-shared-sac"
LOG_DIR = PROJECT_ROOT / "baseline/UAVRTV/slurm_logs"
