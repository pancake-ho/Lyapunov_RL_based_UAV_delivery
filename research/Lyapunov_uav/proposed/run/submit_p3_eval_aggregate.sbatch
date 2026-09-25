#!/usr/bin/bash
#SBATCH --job-name=p3-eval-agg
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/p3-eval-agg-%j.out
#SBATCH --error=logs/p3-eval-agg-%j.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_final}"
readonly EVAL_SEEDS="${EVAL_SEEDS:-3026:3027:3028:3029:3030:3031:3032:3033:3034:3035}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:load_threshold:always_hire:fixed_rsu:nearest_hotspot:rsu_only}"

cd "$PROJECT_DIR"
mkdir -p logs "$EVAL_ROOT/aggregate"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/p3-matplotlib-${SLURM_JOB_ID:-manual}"
mkdir -p "$MPLCONFIGDIR"

python -u -m run.p3_eval_aggregate \
    --seeds "$EVAL_SEEDS" \
    --baseline-policies "$EVAL_BASELINE_POLICIES" \
    --input "$EVAL_ROOT" \
    --output "$EVAL_ROOT/aggregate"
