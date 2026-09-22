#!/usr/bin/env bash
# Submit from research/Lyapunov_uav; one sequential GPU process.
#SBATCH --job-name=paper-snr-eval
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
test -f baseline/NDTVS/snr_sweep_eval.py
set +u
source "${NDT_CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
conda activate "${NDT_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=2026
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/snr-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
git branch --show-current
git rev-parse HEAD
git status --short
nvidia-smi
child=''
pending=0
interrupted=0
forward_stop() {
    pending=1
    interrupted=1
    if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM INT
python -u baseline/NDTVS/snr_sweep_eval.py "$@" &
child=$!
if [[ "$pending" -eq 1 ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
set +e
while true; do
    interrupted=0
    wait "$child"
    result=$?
    if [[ "$interrupted" -eq 0 ]]; then break; fi
done
set -e
if [[ "$result" -eq 75 ]]; then
    echo '[PAUSED] Incomplete sweep. Resubmit identical arguments with --resume.'
fi
# Preserve nonzero status: afterok must never accept a paused sweep.
exit "$result"
