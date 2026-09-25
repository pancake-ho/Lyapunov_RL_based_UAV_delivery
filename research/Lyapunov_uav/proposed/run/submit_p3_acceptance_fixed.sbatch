#!/usr/bin/bash
#SBATCH --job-name=p3-accept
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/p3-accept-%j.out
#SBATCH --error=logs/p3-accept-%j.err

set -Eeuo pipefail

PROJECT_DIR="/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"
CONDA_SH="/data/$USER/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="lab"

OUT="outputs/p3_changedform8_numfix_seed2026_20260906_021351"
CHECKPOINT="${CHECKPOINT:-$OUT/best.pt}"
FRAMES="${FRAMES:-400}"

cd "$PROJECT_DIR"
mkdir -p logs

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1

echo "[START] $(date --iso-8601=seconds)"
echo "[HOST] $(hostname)"
echo "[PWD] $PWD"
echo "[BRANCH] $(git branch --show-current)"
echo "[COMMIT] $(git rev-parse HEAD)"
echo "[PYTHON] $(which python)"
python --version

python - <<'PY'
import torch

print("torch =", torch.__version__)
print("cuda runtime =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(
        "GPU allocation exists but PyTorch cannot access CUDA; "
        "refusing CPU fallback."
    )

print("gpu =", torch.cuda.get_device_name(0))
PY

python -u -m run.p3_acceptance \
    --checkpoint "$CHECKPOINT" \
    --seeds 93026,93027,93028 \
    --frames "$FRAMES" \
    --device cuda \
    --output "$OUT/acceptance_best_f400.json
    "
echo "[DONE] $(date --iso-8601=seconds)"
echo "[RESULT] $OUT/acceptance_fixed.json"
