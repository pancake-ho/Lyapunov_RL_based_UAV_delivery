#!/usr/bin/env bash
# Submit from research/Lyapunov_uav; see RUN_COMMANDS.sh.
#SBATCH --job-name=rsu-common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -Eeuo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit using sbatch}"
set +u
source "${NDT_CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
conda activate "${NDT_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/ndtvs-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
child=''
pending_signal=''
interrupted=0
forward_stop() {
    pending_signal=USR1
    interrupted=1
    if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM INT
python -u baseline/NDTVS/ndtvs_common.py "$@" &
child=$!
if [[ -n "$pending_signal" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
set +e
while true; do
    interrupted=0
    wait "$child"
    result=$?
    if [[ "$interrupted" -eq 0 ]]; then break; fi
done
set -e
if [[ ( "$result" -eq 0 || "$result" -eq 75 ) && "${1:-}" == train ]]; then
    output_dir=''
    take_next=0
    for argument in "$@"; do
        if [[ "$take_next" -eq 1 ]]; then output_dir="$argument"; break; fi
        case "$argument" in
            --out) take_next=1 ;;
            --out=*) output_dir="${argument#--out=}" ;;
        esac
    done
    python baseline/NDTVS/ndtvs_analysis.py audit-run "${output_dir:?Missing output directory}"
fi
if [[ "$result" -eq 75 ]]; then
    echo '[PAUSED] Checkpoint saved. Training target is NOT complete. Submit the same command with --resume.'
    exit 0
fi
exit "$result"
