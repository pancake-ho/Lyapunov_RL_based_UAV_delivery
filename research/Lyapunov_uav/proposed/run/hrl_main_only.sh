#!/usr/bin/env bash
#SBATCH --job-name=hrl_main
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm_logs/hrl_main-%j.out
#SBATCH --error=slurm_logs/hrl_main-%j.err

set -Eeuo pipefail

: "${SLURM_JOB_ID:?Submit this script using sbatch}"
cd -- "$SLURM_SUBMIT_DIR"

# Activate the existing training environment.
set +u
source /data/surt321/anaconda3/etc/profile.d/conda.sh
conda activate lab
set -u

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hppo-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

# Same settings as the existing main-training command.
seed=2026
episodes=100
output_root="$PWD/outputs/hppo"
run_name="hrl50_seed${seed}_job${SLURM_JOB_ID}"
run_dir="$output_root/$run_name"

mkdir -p "$output_root"

echo "[START] job=$SLURM_JOB_ID host=$(hostname)"
echo "[RUN_DIR] $run_dir"
echo "[EPISODES] $episodes"

observer_pid=""

cleanup() {
    if [[ -n "$observer_pid" ]]; then
        kill "$observer_pid" 2>/dev/null || true
        wait "$observer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for training logs, then refresh file visualizations.
# Observer errors do not interrupt training.
(
    while [[ ! -s "$run_dir/resolved_config.json" ||
             ! -s "$run_dir/trace.jsonl" ]]; do
        sleep 5
    done

    exec bash run/hrl_export.sh "$run_dir" \
        --out "$run_dir/artifacts/files_live" \
        --watch 300
) > "slurm_logs/hrl_main-${SLURM_JOB_ID}-export.log" 2>&1 &
observer_pid=$!

# Start main training directly: no unit tests or smoke run.
python -u -m hppo.train \
    --mode train \
    --device cuda \
    --torch-num-threads 1 \
    --num-regions 10 \
    --users-per-region 5 \
    --seed "$seed" \
    --hidden-dims 256,256 \
    --train-episodes "$episodes" \
    --num-frames 30 \
    --frame-slots 10 \
    --rollout-scenarios 2 \
    --reward-mode dpp \
    --save-every-episodes 2 \
    --console-log-every-slots 10 \
    --output-dir "$output_root" \
    --run-name "$run_name"

echo "[TRAIN_COMPLETE] $run_dir"

# The live exporter exits when it observes run_end.
if wait "$observer_pid"; then
    echo "[LIVE_EXPORT_COMPLETE]"
else
    echo "[LIVE_EXPORT_FAILED] Check the export log." >&2
fi
observer_pid=""

# Export every episode after training has completed.
if bash run/hrl_export.sh "$run_dir" \
    --out "$run_dir/artifacts/files_final" \
    --all-episodes; then
    echo "[ALL_COMPLETE] $run_dir"
else
    echo "[FINAL_EXPORT_FAILED] Training completed; check export diagnostics." >&2
    exit 2
fi
