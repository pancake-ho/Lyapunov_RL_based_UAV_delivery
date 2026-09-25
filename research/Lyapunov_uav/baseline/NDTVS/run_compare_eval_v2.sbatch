#!/usr/bin/env bash
# Submit from research/Lyapunov_uav. Arguments: smoke | sweep [Python options]
#SBATCH --job-name=compare-eval-v2
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from research/Lyapunov_uav}"
mode="${1:?Usage: sbatch run_compare_eval_v2.sbatch smoke|sweep}"
shift
case "$mode" in smoke|sweep) ;; *) echo 'Mode must be smoke or sweep' >&2; exit 2 ;; esac
set +u
source "${NDT_CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
conda activate "${NDT_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=2026
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/compare-eval-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
exec python -u - "$mode" "$@" <<'PY'
import sys
sys.path.insert(0, 'baseline/NDTVS')
import compare_eval_v2 as evaluation
sys.argv = ['compare_eval_v2.py', *sys.argv[1:]]
result = evaluation.main()
if result in (None, 0) and sys.argv[1] == 'sweep':
    sys.argv[1] = 'report'
    result = evaluation.main()
raise SystemExit(result)
PY
