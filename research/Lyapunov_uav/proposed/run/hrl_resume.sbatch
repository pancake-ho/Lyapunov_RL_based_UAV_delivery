#!/usr/bin/env bash
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit with hrl_resume_submit.sh}"
: "${HPPO_PROJECT_ROOT:?Missing project root}"
: "${HPPO_SOURCE_RUN:?Missing source run}"
cd -- "$HPPO_PROJECT_ROOT"
task_user="${USER:-$(id -un)}"
set +u
source "${HPPO_CONDA_SH:-/data/$task_user/anaconda3/etc/profile.d/conda.sh}"
conda activate "${HPPO_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hppo-resume-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
run_name="hrl_resume_job${SLURM_JOB_ID}"
printf '[SOURCE_RUN] %s\n' "$HPPO_SOURCE_RUN"
printf '[RUN_DIR] %s/%s\n' "${HPPO_OUTPUT_DIR:-$PWD/outputs/hppo}" "$run_name"
# Do not add commands after exec. Evaluation/export run in a separate allocation.
# Exit 75 is a saved PAUSED session, intentionally distinguishable from COMPLETE.
exec python -u -m hppo.resume \
    --source-run "$HPPO_SOURCE_RUN" \
    --output-dir "${HPPO_OUTPUT_DIR:-$PWD/outputs/hppo}" --run-name "$run_name" \
    --device cuda --max-seconds 82800 --reserve-seconds 900
