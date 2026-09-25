#!/usr/bin/bash
#SBATCH --job-name=ndtv-rsu-train
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=29G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/ndtv-rsu-train-%j.out
#SBATCH --error=logs/ndtv-rsu-train-%j.err

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate lab

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Headless matplotlib cache/config path on the compute node.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

SEED="${SEED:-2026}"
RUN_TAG="${RUN_TAG:-seed${SEED}_${SLURM_JOB_ID:-local}}"
OUT_DIR="runs/ndtv_rsu_ppo_${RUN_TAG}"

printf '[SBATCH-START] %s\n' "$(date -Is)"
printf '[JOB] id=%s host=%s pwd=%s seed=%s\n' \
  "${SLURM_JOB_ID:-NA}" "$(hostname)" "$PWD" "$SEED"
printf '[PYTHON] %s\n' "$(which python)"
printf '[MPLCONFIGDIR] %s\n' "$MPLCONFIGDIR"
python --version

python -m py_compile ndtv_rsu_only_ppo_baseline_rewardplot.py

python ndtv_rsu_only_ppo_baseline_rewardplot.py \
  --episodes 5000 \
  --eval_episodes 3 \
  --seed "$SEED" \
  --device cpu \
  --out_dir "$OUT_DIR" \
  --log_slots eval \
  --log_text \
  --print_every 10 \
  --plot_every 25 \
  --reward_ma_window 50

test -s "$OUT_DIR/train_log.csv"
test -s "$OUT_DIR/training_reward.png"
test -s "$OUT_DIR/ppo_final.pt"
test -s "$OUT_DIR/eval_metrics.json"

printf '[REWARD-PLOT] %s\n' "$OUT_DIR/training_reward.png"
printf '[RESULT] %s\n' "$OUT_DIR"
printf '[SBATCH-END] %s\n' "$(date -Is)"
