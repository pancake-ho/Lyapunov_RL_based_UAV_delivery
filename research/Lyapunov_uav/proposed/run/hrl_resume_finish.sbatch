#!/usr/bin/env bash
# Separate allocation: verify the saved segment, evaluate, then export.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
: "${HPPO_PROJECT_ROOT:?Set proposed directory}"
: "${HPPO_FINISHED_RUN:?Set the final run directory}"
cd -- "$HPPO_PROJECT_ROOT"
task_user="${USER:-$(id -un)}"
set +u
source "${HPPO_CONDA_SH:-/data/$task_user/anaconda3/etc/profile.d/conda.sh}"
conda activate "${HPPO_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hppo-finish-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import json, os
from pathlib import Path
import torch
p = Path(os.environ['HPPO_FINISHED_RUN']) / 'training_status.json'
if p.exists() and json.loads(p.read_text())['status'] != 'COMPLETE':
    raise RuntimeError('Training is not COMPLETE; resume it before final evaluation')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable; no CPU fallback')
PY
python -m hppo.verify_trace "$HPPO_FINISHED_RUN"
eval_name="$(basename -- "$HPPO_FINISHED_RUN")_eval_job${SLURM_JOB_ID}"
eval_root="$(dirname -- "$HPPO_FINISHED_RUN")"
python -m hppo.evaluate --train-run "$HPPO_FINISHED_RUN" \
    --run-name "$eval_name" --output-dir "$eval_root" \
    --eval-episodes "${HPPO_EVAL_EPISODES:-5}" --episode-offset 1000000 --device cuda
python -m hppo.verify_trace "$eval_root/$eval_name"
python -m hppo.export_files "$HPPO_FINISHED_RUN" --out "$HPPO_FINISHED_RUN/artifacts/final" --dpi 100
python -m hppo.explain_trace "$HPPO_FINISHED_RUN" --out "$HPPO_FINISHED_RUN/artifacts/explanation"
python -m hppo.export_files "$eval_root/$eval_name" --out "$eval_root/$eval_name/artifacts/final" --dpi 100
printf '[EVALUATION_COMPLETE] %s\n' "$eval_root/$eval_name"
