#!/usr/bin/env bash
# Resource flags come from submit.py and config.py.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@900
set -Eeuo pipefail
python_path=${1:?Configured Python required}
mode=${2:?Mode required}
settings=${3:?Settings required}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/ndtvs-eval-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
child=''
interrupted=0
pending_signal=''
forward_stop() {
    interrupted=1
    pending_signal=USR1
    if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM INT
"$python_path" -u baseline/NDTVS/evaluation/benchmark/cli.py "$mode" --settings "$settings" &
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
if [[ "$result" -eq 75 ]]; then
    echo 'PAUSED: paired scenarios saved. Resubmit the same config to continue.'
    exit 0
fi
exit "$result"
