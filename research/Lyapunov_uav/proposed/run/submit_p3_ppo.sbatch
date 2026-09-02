#!/usr/bin/bash
#SBATCH --job-name=p3-ppo
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/p3-ppo-%j.out
#SBATCH --error=logs/p3-ppo-%j.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="/data/$USER/anaconda3/etc/profile.d/conda.sh"
readonly EPISODES="${EPISODES:-100}"
readonly FRAMES="${FRAMES:-30}"
readonly SEED="${SEED:-2026}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-outputs/p3_ppo_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
source "$CONDA_SH"
conda activate lab

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA allocation exists but PyTorch cannot open the GPU; refusing CPU fallback")
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY

python -m unittest discover -s tests -p 'test_p3_*.py' -v
python -m run.p3_train_ppo \
  --episodes "$EPISODES" \
  --frames "$FRAMES" \
  --seed "$SEED" \
  --device cuda \
  --output "$OUTPUT_DIR"
