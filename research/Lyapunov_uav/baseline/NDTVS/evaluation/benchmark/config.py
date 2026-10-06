"""All experiment and submission settings. Edit this file, not shell variables."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUT = PROJECT_ROOT / "baseline/NDTVS/runs/evaluation/snr_v_service"
DEVICE = "cuda"
SEEDS = (2026, 2027, 2028)  # Evaluation scenarios, not independent training seeds.
EPISODES_PER_SEED = 30
TEST_EPISODE_OFFSET = 7_000_000
SMOKE_EPISODE_OFFSET = 8_000_000

# A scientific choice is needed. No answer was supplied to the SNR question.
# 'transmit': Gamma = RSU per-user transmit power / (N0 * per-user BW).
# 'received': RSU reference-link SNR, at REFERENCE_DISTANCE_M and fading=1.
# 'offset': exact -10..+10 dB relative to the trained noise PSD; axis says offset.
SNR_MODE = None
SNR_DB = (25, 30, 35, 40, 45)
SNR_OFFSETS_DB = (-10, -5, 0, 5, 10)
REFERENCE_DISTANCE_M = None  # Required only for 'received'; not chosen silently.

# Each proposed model has its OWN config/runtime and compatible checkpoint pair.
# Add as many V models as needed. No automatic choice by test-set performance.
V50_RUN = PROJECT_ROOT / "proposed/outputs/hppo/hrl_resume_job145847"
NDTVS_RUN = PROJECT_ROOT / "baseline/NDTVS/runs/common_gpu/ndtvs_paper_261006_seed2026_ep1000"
MODELS = [
    dict(name="proposed_V50", algorithm="proposed", expected_v=50,
         config=V50_RUN / "resolved_config.json", runtime=V50_RUN / "runtime.json",
         frame_checkpoint=V50_RUN / "checkpoints/frame_latest.pt",
         slot_checkpoint=V50_RUN / "checkpoints/slot_latest.pt",
         selection="user-selected converged checkpoint"),
    dict(name="ndtvs", algorithm="ndtvs", checkpoint=NDTVS_RUN / "best.pt",
         completion_status=NDTVS_RUN / "status.json",
         selection="fixed-validation mean-QoE selected best.pt"),
]
# Example additional model (replace RUN and expected_v with your actual run):
# RUN = PROJECT_ROOT / "proposed/outputs/hppo/<your_V_run>"
# MODELS.append(dict(name="proposed_V20", algorithm="proposed", expected_v=20,
#     config=RUN/"resolved_config.json", runtime=RUN/"runtime.json",
#     frame_checkpoint=RUN/"checkpoints/frame_latest.pt",
#     slot_checkpoint=RUN/"checkpoints/slot_latest.pt", selection="user-selected"))
# Optional existing RSU ablation: MODELS.append(dict(name='hppo_rsu',
#     algorithm='hppo_rsu', checkpoint=Path('<validation-selected best.pt>')))

# Common reporting utility is received PSNR / 41.64. Training utility is retained.
# Common QoE uses the existing NDTVS observer, not the proposed DPP reward.
# A COST-AUGMENTED reporting objective is distinct from the original paper QoE.
# No cost/PSNR conversion coefficient or cost list was approved in this turn.
COST_MODE = None  # None: no cost sweep; 'reevaluate' or 'accounting'.
HIRING_COSTS = None  # Suggested pending confirmation: (0., 5., 10., 20.).
QOE_COST_WEIGHT = None  # Suggested pending confirmation: 1.0.
COST_SNR_DB = None  # Choose a value on your selected SNR axis for the cost sweep.

# Trace/audit the first scenario of EVERY method/SNR/seed/cost cell.
TRACE_ALL = False
EXPORT_ANIMATIONS = True
VISUAL_SEEDS = (2026,)  # Render representative traced episodes for these seeds.
VISUAL_SNR_DB = None  # None=all SNR levels; tuple restricts visualization only.
VISUAL_SLOT_STRIDE = 10  # One slot image per frame by default; set 1 for every slot.
VISUAL_MAX_IMAGES = 30
GIF_FPS = 2
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 572913
WALLTIME_SECONDS = 82800
RESERVE_SECONDS = 900
MAX_NEW_EPISODES = 0  # 0=unlimited; supports short, resumable checks.
RESUME = True  # New output also works. Exact spec must match to resume.

# Submission settings; no terminal exports or echo are needed.
PYTHON = Path("/data/surt321/anaconda3/envs/lab/bin/python")
PARTITION = "batch_eebme_ugrad"
GPUS = 1
CPUS = 16
MEMORY = "29G"
TIME_LIMIT = "1-00:00:00"
JOB_NAME = "ndtvs-v-snr-eval"
LOG_DIR = PROJECT_ROOT / "baseline/NDTVS/slurm_logs"
