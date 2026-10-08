#!/usr/bin/bash
#SBATCH --job-name=p3-ppo
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/p3-ppo-%j.out
#SBATCH --error=logs/p3-ppo-%j.err

set -Eeuo pipefail

readonly EXPECTED_BRANCH="feat/new-form-p3"
readonly PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"

readonly EPISODES="${EPISODES:-100}"
readonly FRAMES="${FRAMES:-400}"
readonly SEED="${SEED:-2026}"
readonly VALIDATION_SEEDS="${VALIDATION_SEEDS:-92026,92027,92028}"
readonly VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
readonly EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
readonly MIN_DELTA="${MIN_DELTA:-0.0001}"
readonly CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
readonly PROGRESS_INTERVAL_FRAMES="${PROGRESS_INTERVAL_FRAMES:-20}"
readonly PLOT_INTERVAL="${PLOT_INTERVAL:-5}"
readonly ROLLING_WINDOW="${ROLLING_WINDOW:-10}"
readonly ACCEPTANCE_SEEDS="${ACCEPTANCE_SEEDS:-93026,93027,93028}"
readonly ACCEPTANCE_FRAMES="${ACCEPTANCE_FRAMES:-100}"
readonly RUN_ACCEPTANCE="${RUN_ACCEPTANCE:-1}"

# The first run showed a deterministic-policy regime shift.  Use a smaller
# learning rate and a KL guard while keeping the formulation and reward intact.
readonly LEARNING_RATE="${LEARNING_RATE:-0.0001}"
readonly ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
readonly TARGET_KL="${TARGET_KL:-0.015}"

readonly OUTPUT_DIR="${OUTPUT_DIR:-outputs/p3_ppo_seed${SEED}_f${FRAMES}_e${EPISODES}_${SLURM_JOB_ID}}"
readonly CONSOLE_LOG="$OUTPUT_DIR/train_console.log"

job_started_epoch="$(date +%s)"
on_exit() {
    local exit_code="$?"
    local finished_epoch
    local runtime_seconds
    finished_epoch="$(date +%s)"
    runtime_seconds="$((finished_epoch - job_started_epoch))"
    echo "[SBATCH-END] job_id=${SLURM_JOB_ID:-unknown} exit_code=$exit_code runtime_seconds=$runtime_seconds output=$OUTPUT_DIR"
    trap - EXIT
    exit "$exit_code"
}
trap on_exit EXIT

if [[ ! -f "$PROJECT_DIR/config_p3.py" || ! -f "$PROJECT_DIR/run/p3_train_ppo.py" ]]; then
    echo "Invalid PROJECT_DIR: $PROJECT_DIR" >&2
    echo "Submit this script from research/Lyapunov_uav/proposed or set PROJECT_DIR explicitly." >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda activation script: $CONDA_SH" >&2
    exit 2
fi

cd "$PROJECT_DIR"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH, got $(git branch --show-current)" >&2
    exit 2
fi
if [[ -n "$(git status --short)" ]]; then
    echo "Refusing a non-reproducible dirty worktree" >&2
    git status --short >&2
    exit 2
fi
mkdir -p logs "$OUTPUT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_PER_GPU:-16}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

{
    echo "[SBATCH-START] timestamp=$(date --iso-8601=seconds)"
    echo "[SBATCH-START] job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
    echo "[SBATCH-START] project=$PROJECT_DIR output=$OUTPUT_DIR"
    echo "[SBATCH-START] branch=$(git branch --show-current 2>/dev/null || echo unknown)"
    echo "[SBATCH-START] commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "[SBATCH-START] dirty_files=$(git status --short 2>/dev/null | wc -l)"
    echo "[SBATCH-START] episodes=$EPISODES frames=$FRAMES seed=$SEED"
    echo "[SBATCH-START] validation_seeds=$VALIDATION_SEEDS interval=$VALIDATION_INTERVAL patience=$EARLY_STOPPING_PATIENCE"
    echo "[SBATCH-START] run_acceptance=$RUN_ACCEPTANCE acceptance_seeds=$ACCEPTANCE_SEEDS acceptance_frames=$ACCEPTANCE_FRAMES"
    echo "[SBATCH-START] lr=$LEARNING_RATE entropy_coef=$ENTROPY_COEF target_kl=$TARGET_KL"
} | tee -a "$CONSOLE_LOG"

python -u - <<'PY' 2>&1 | tee -a "$CONSOLE_LOG"
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA allocation exists but PyTorch cannot open the GPU; refusing CPU fallback"
    )
print("torch", torch.__version__, flush=True)
print("cuda", torch.version.cuda, flush=True)
print("gpu", torch.cuda.get_device_name(0), flush=True)
PY

python -u -m unittest discover -s tests -p 'test_p3_*.py' -v 2>&1 \
    | tee -a "$CONSOLE_LOG"

python -u -m run.p3_train_ppo \
    --episodes "$EPISODES" \
    --frames "$FRAMES" \
    --seed "$SEED" \
    --validation-seeds "$VALIDATION_SEEDS" \
    --validation-interval "$VALIDATION_INTERVAL" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --min-delta "$MIN_DELTA" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --progress-interval-frames "$PROGRESS_INTERVAL_FRAMES" \
    --plot-interval "$PLOT_INTERVAL" \
    --rolling-window "$ROLLING_WINDOW" \
    --learning-rate "$LEARNING_RATE" \
    --entropy-coef "$ENTROPY_COEF" \
    --target-kl "$TARGET_KL" \
    --device cuda \
    --output "$OUTPUT_DIR" 2>&1 \
    | tee -a "$CONSOLE_LOG"

if [[ "$RUN_ACCEPTANCE" == "1" ]]; then
    python -u -m run.p3_acceptance \
        --checkpoint "$OUTPUT_DIR/best.pt" \
        --seeds "$ACCEPTANCE_SEEDS" \
        --frames "$ACCEPTANCE_FRAMES" \
        --device cuda \
        --output "$OUTPUT_DIR/acceptance.json" 2>&1 \
        | tee -a "$CONSOLE_LOG"
elif [[ "$RUN_ACCEPTANCE" != "0" ]]; then
    echo "RUN_ACCEPTANCE must be 0 or 1, got: $RUN_ACCEPTANCE" >&2
    exit 2
fi
