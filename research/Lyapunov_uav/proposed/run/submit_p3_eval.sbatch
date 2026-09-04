#!/usr/bin/bash
#SBATCH --job-name=p3-eval
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/p3-eval-%A_%a.out
#SBATCH --error=logs/p3-eval-%A_%a.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_final}"
readonly EVAL_SEEDS="${EVAL_SEEDS:-3026:3027:3028:3029:3030:3031:3032:3033:3034:3035}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:load_threshold:always_hire:fixed_rsu:nearest_hotspot:rsu_only}"
readonly EVAL_FRAMES="${EVAL_FRAMES:-400}"
readonly EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
readonly EVAL_PROGRESS_INTERVAL="${EVAL_PROGRESS_INTERVAL:-10}"
readonly EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
readonly EVAL_SELECTION_WORKERS="${EVAL_SELECTION_WORKERS:-4}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:?BEST_CHECKPOINT must be exported}"
readonly LATEST_CHECKPOINT="${LATEST_CHECKPOINT:?LATEST_CHECKPOINT must be exported}"
readonly TASK_INDEX="${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}"

started_epoch="$(date +%s)"
on_exit() {
    local exit_code="$?"
    local finished_epoch
    finished_epoch="$(date +%s)"
    echo "[SBATCH-END] job=${SLURM_JOB_ID:-unknown} task=$TASK_INDEX exit=$exit_code runtime=$((finished_epoch - started_epoch))s"
    trap - EXIT
    exit "$exit_code"
}
trap on_exit EXIT

if [[ ! -f "$PROJECT_DIR/config_p3.py" || ! -f "$PROJECT_DIR/run/p3_eval_shard.py" ]]; then
    echo "Invalid PROJECT_DIR: $PROJECT_DIR" >&2
    exit 2
fi
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Missing conda activation script: $CONDA_SH" >&2
    exit 2
fi
if [[ ! -f "$BEST_CHECKPOINT" || ! -f "$LATEST_CHECKPOINT" ]]; then
    echo "Missing PPO checkpoint(s): best=$BEST_CHECKPOINT latest=$LATEST_CHECKPOINT" >&2
    exit 2
fi

cd "$PROJECT_DIR"
mkdir -p logs "$EVAL_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[SBATCH-START] timestamp=$(date --iso-8601=seconds) job=${SLURM_JOB_ID:-unknown} array=${SLURM_ARRAY_JOB_ID:-unknown} task=$TASK_INDEX host=$(hostname)"
echo "[SBATCH-START] branch=$(git branch --show-current 2>/dev/null || echo unknown) commit=$(git rev-parse HEAD 2>/dev/null || echo unknown) dirty_files=$(git status --short 2>/dev/null | wc -l)"
echo "[SBATCH-START] root=$EVAL_ROOT frames=$EVAL_FRAMES rollouts=$EVAL_ROLLOUTS selection_workers=$EVAL_SELECTION_WORKERS device=$EVAL_DEVICE"

python -u -m run.p3_eval_shard \
    --task-index "$TASK_INDEX" \
    --seeds "$EVAL_SEEDS" \
    --baseline-policies "$EVAL_BASELINE_POLICIES" \
    --frames "$EVAL_FRAMES" \
    --rollouts "$EVAL_ROLLOUTS" \
    --best-checkpoint "$BEST_CHECKPOINT" \
    --latest-checkpoint "$LATEST_CHECKPOINT" \
    --device "$EVAL_DEVICE" \
    --selection-workers "$EVAL_SELECTION_WORKERS" \
    --progress-interval "$EVAL_PROGRESS_INTERVAL" \
    --resume \
    --fail-on-hard-violation \
    --output "$EVAL_ROOT"
