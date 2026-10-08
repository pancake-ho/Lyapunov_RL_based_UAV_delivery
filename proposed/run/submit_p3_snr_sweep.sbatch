#!/usr/bin/bash
#SBATCH --job-name=p3-snr
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/p3-snr-%A_%a.out
#SBATCH --error=logs/p3-snr-%A_%a.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"
readonly SNR_ROOT="${SNR_ROOT:?SNR_ROOT must be exported}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:?BEST_CHECKPOINT must be exported}"
readonly SNR_VALUES="${SNR_VALUES:-20:25:30:35:40}"
readonly SNR_BASELINE_DB="${SNR_BASELINE_DB:-30}"
readonly SNR_SEEDS="${SNR_SEEDS:-120026:120027:120028:120029:120030}"
readonly SNR_POLICIES="${SNR_POLICIES:-proposed:always_hire:slow_ppo:rsu_only}"
readonly SNR_FRAMES="${SNR_FRAMES:-400}"
readonly SNR_ROLLOUTS="${SNR_ROLLOUTS:-4}"
readonly SNR_SELECTION_WORKERS="${SNR_SELECTION_WORKERS:-8}"
readonly SNR_PROGRESS_INTERVAL="${SNR_PROGRESS_INTERVAL:-10}"
readonly SNR_JOB_WORKERS="${SNR_JOB_WORKERS:-2}"
readonly SNR_DEVICE="${SNR_DEVICE:-cpu}"
readonly WORKER_INDEX="${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}"

started="$(date +%s)"
on_exit() {
    local rc="$?"
    local ended
    ended="$(date +%s)"
    echo "[SBATCH-END] job=${SLURM_JOB_ID:-unknown} worker=$WORKER_INDEX exit=$rc runtime=$((ended-started))s"
    trap - EXIT
    exit "$rc"
}
trap on_exit EXIT

cd "$PROJECT_DIR"
mkdir -p logs "$SNR_ROOT"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    echo "Missing BEST_CHECKPOINT: $BEST_CHECKPOINT" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda activation script: $CONDA_SH" >&2
    exit 2
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[SBATCH-START] timestamp=$(date --iso-8601=seconds) job=${SLURM_JOB_ID:-unknown} array=${SLURM_ARRAY_JOB_ID:-unknown} worker=$WORKER_INDEX/$SNR_JOB_WORKERS host=$(hostname)"
echo "[SBATCH-START] branch=$(git branch --show-current 2>/dev/null || echo unknown) commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) dirty_files=$(git status --short 2>/dev/null | wc -l)"
echo "[SBATCH-START] root=$SNR_ROOT snrs=$SNR_VALUES seeds=$SNR_SEEDS frames=$SNR_FRAMES rollouts=$SNR_ROLLOUTS workers=$SNR_SELECTION_WORKERS"

python -u -m run.p3_snr_sweep \
    --worker-index "$WORKER_INDEX" \
    --worker-count "$SNR_JOB_WORKERS" \
    --snrs-db "$SNR_VALUES" \
    --baseline-snr-db "$SNR_BASELINE_DB" \
    --seeds "$SNR_SEEDS" \
    --policies "$SNR_POLICIES" \
    --frames "$SNR_FRAMES" \
    --rollouts "$SNR_ROLLOUTS" \
    --selection-workers "$SNR_SELECTION_WORKERS" \
    --progress-interval "$SNR_PROGRESS_INTERVAL" \
    --device "$SNR_DEVICE" \
    --best-checkpoint "$BEST_CHECKPOINT" \
    --output "$SNR_ROOT" \
    --resume

(
    flock 9
    python -u -m run.p3_snr_sweep_aggregate \
        --if-complete \
        --input "$SNR_ROOT" \
        --output "$SNR_ROOT/aggregate" \
        --snrs-db "$SNR_VALUES" \
        --seeds "$SNR_SEEDS" \
        --policies "$SNR_POLICIES"
) 9>"$SNR_ROOT/.aggregate.lock"
