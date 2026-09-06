#!/usr/bin/bash

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"
readonly P3_RUN_DIR="${P3_RUN_DIR:-$PROJECT_DIR/outputs/p3_ppo_seed2026_f400_e100_139723}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:-$P3_RUN_DIR/best.pt}"
readonly SNR_ROOT="${SNR_ROOT:-$PROJECT_DIR/outputs/p3_snr_sweep_s5_f400}"
readonly SNR_VALUES="${SNR_VALUES:-20:25:30:35:40}"
readonly SNR_BASELINE_DB="${SNR_BASELINE_DB:-30}"
readonly SNR_SEEDS="${SNR_SEEDS:-120026:120027:120028:120029:120030}"
readonly SNR_POLICIES="${SNR_POLICIES:-proposed:always_hire:slow_ppo:rsu_only}"
readonly SNR_FRAMES="${SNR_FRAMES:-400}"
readonly SNR_ROLLOUTS="${SNR_ROLLOUTS:-4}"
readonly SNR_SELECTION_WORKERS="${SNR_SELECTION_WORKERS:-8}"
readonly SNR_JOB_WORKERS="${SNR_JOB_WORKERS:-2}"

cd "$PROJECT_DIR"
mkdir -p logs "$SNR_ROOT"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda activation script: $CONDA_SH" >&2
    exit 2
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
    echo "Missing best checkpoint: $BEST_CHECKPOINT" >&2
    exit 2
fi
if ! [[ "$SNR_JOB_WORKERS" =~ ^[12]$ ]]; then
    echo "SNR_JOB_WORKERS must be 1 or 2 for the current Seraph user job limit." >&2
    exit 2
fi
if ! [[ "$SNR_SELECTION_WORKERS" =~ ^[1-9][0-9]*$ ]] || (( SNR_SELECTION_WORKERS > 16 )); then
    echo "SNR_SELECTION_WORKERS must be between 1 and 16." >&2
    exit 2
fi

echo "[SNR-PREFLIGHT] branch=$(git branch --show-current)"
echo "[SNR-PREFLIGHT] commit=$(git rev-parse HEAD)"
echo "[SNR-PREFLIGHT] dirty_files=$(git status --short | wc -l)"
echo "[SNR-PREFLIGHT] checkpoint=$BEST_CHECKPOINT"
echo "[SNR-PREFLIGHT] snrs=$SNR_VALUES seeds=$SNR_SEEDS policies=$SNR_POLICIES"
echo "[SNR-PREFLIGHT] frames=$SNR_FRAMES rollouts=$SNR_ROLLOUTS selection_workers=$SNR_SELECTION_WORKERS job_workers=$SNR_JOB_WORKERS"

task_count="$(
python -m run.p3_snr_sweep \
    --print-task-count \
    --worker-index 0 \
    --worker-count "$SNR_JOB_WORKERS" \
    --snrs-db "$SNR_VALUES" \
    --baseline-snr-db "$SNR_BASELINE_DB" \
    --seeds "$SNR_SEEDS" \
    --policies "$SNR_POLICIES" \
    --frames "$SNR_FRAMES" \
    --rollouts "$SNR_ROLLOUTS" \
    --selection-workers "$SNR_SELECTION_WORKERS" \
    --best-checkpoint "$BEST_CHECKPOINT" \
    --output "$SNR_ROOT"
)"
echo "[SNR-PREFLIGHT] total_tasks=$task_count"

export PROJECT_DIR BEST_CHECKPOINT SNR_ROOT SNR_VALUES SNR_BASELINE_DB
export SNR_SEEDS SNR_POLICIES SNR_FRAMES SNR_ROLLOUTS
export SNR_SELECTION_WORKERS SNR_JOB_WORKERS

array_end=$((SNR_JOB_WORKERS - 1))
sbatch --array="0-${array_end}%${SNR_JOB_WORKERS}" run/submit_p3_snr_sweep.sbatch
