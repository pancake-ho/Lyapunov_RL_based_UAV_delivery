"""Edit this file to configure NDTVS submission; keep the proposed JSON intact."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Comparison reference: user-selected Proposed V=50, trained for 700 episodes.
# The resume JSON's train_episodes can be the remaining session budget, not 700.
SOURCE_CONFIG = PROJECT_ROOT / "proposed/outputs/hppo/hrl_resume_job145847/resolved_config.json"
EXPECTED_LYAPUNOV_V = 50.0
REFERENCE_PROPOSED_EPISODES = 700  # Reference note; does not override any source field.

# The submitter copies SOURCE_CONFIG and changes only episode_offset to zero.
TRAIN_CONFIG = PROJECT_ROOT / "baseline/NDTVS/configs/ndtvs_V50_seed2026.json"
OUT = PROJECT_ROOT / "baseline/NDTVS/runs/common_gpu/ndtvs_paper_261006_seed2026_ep500"

# NDTVS run settings: independent from the Proposed run's completed episodes.
EPISODES = 500
SEED = 2026
DEVICE = "cuda"
RESUME = False  # Set True only to continue this same NDTVS run.
VAL_EVERY = 25
VAL_EPISODES = 10
TRACE_EVERY = 25
WALLTIME_SECONDS = 82800
RESERVE_SECONDS = 900
MAX_NEW_EPISODES = 0

# Existing Seraph/lab resources. run_common.sbatch activates the lab environment.
PARTITION = "batch_eebme_ugrad"
GPUS = 1
CPUS = 16
MEMORY = "29G"
TIME_LIMIT = "1-00:00:00"
JOB_NAME = "ndtvs-paper"
LOG_DIR = PROJECT_ROOT / "baseline/NDTVS/slurm_logs"

# Budget extension: use training/submit_extension.py instead of submit_ndtvs.py.
# EPISODES/RESUME above describe the original run; leave them as they were.
# The source checkpoint supplies the environment, reward, seed, device,
# learning settings and validation settings. Only the total budget increases.
EXTEND_SOURCE_RUN = OUT
EXTEND_OUT = PROJECT_ROOT / "baseline/NDTVS/runs/common_gpu/ndtvs_paper_261006_seed2026_ep1000"
EXTEND_EPISODES = 1000  # TOTAL episodes: continue 501..1000, not 1000 more.
EXTEND_JOB_NAME = "ndtvs-extend"
# The checkpoint tool imports torch; run it in the existing lab environment.
EXTENSION_PYTHON = Path("/data/surt321/anaconda3/envs/lab/bin/python")
