#!/usr/bin/bash
#SBATCH --job-name=ndtv-rsu-smoke
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/ndtv-rsu-smoke-%j.out
#SBATCH --error=logs/ndtv-rsu-smoke-%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate lab

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Avoid matplotlib cache/config permission issues on Seraph.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

SEED="${SEED:-2026}"
OUT_DIR="runs/ndtv_rsu_smoke_rewardplot_seed${SEED}_${SLURM_JOB_ID:-local}"

printf '[SBATCH-START] %s\n' "$(date -Is)"
printf '[JOB] id=%s host=%s pwd=%s seed=%s\n' \
  "${SLURM_JOB_ID:-NA}" "$(hostname)" "$PWD" "$SEED"
printf '[PYTHON] %s\n' "$(which python)"
printf '[MPLCONFIGDIR] %s\n' "$MPLCONFIGDIR"
python --version

python -m py_compile ndtv_rsu_only_ppo_baseline_rewardplot.py

python ndtv_rsu_only_ppo_baseline_rewardplot.py \
  --episodes 5 \
  --eval_episodes 1 \
  --seed "$SEED" \
  --device cpu \
  --out_dir "$OUT_DIR" \
  --log_slots eval \
  --log_text \
  --print_every 1 \
  --plot_every 1 \
  --reward_ma_window 3

test -s "$OUT_DIR/train_log.csv"
test -s "$OUT_DIR/training_reward.png"
test -s "$OUT_DIR/ppo_final.pt"
test -s "$OUT_DIR/eval_metrics.json"

printf '[VERIFY] reward plot: %s\n' "$OUT_DIR/training_reward.png"
printf '[RESULT] %s\n' "$OUT_DIR"
printf '[SBATCH-END] %s\n' "$(date -Is)"
