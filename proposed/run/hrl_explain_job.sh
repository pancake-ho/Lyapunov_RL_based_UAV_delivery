#!/usr/bin/env bash
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit using run/hrl_explain_submit.sh}"
project_root="$1"
run_dir="$2"
shift 2
cd -- "$project_root"
task_user="${USER:-$(id -un)}"
set +u
source "${HPPO_CONDA_SH:-/data/$task_user/anaconda3/etc/profile.d/conda.sh}"
conda activate "${HPPO_CONDA_ENV:-lab}"
set -u
echo "[EXPLAIN] job=$SLURM_JOB_ID host=$(hostname) source=$run_dir"
exec bash run/hrl_explain.sh "$run_dir" "$@"
