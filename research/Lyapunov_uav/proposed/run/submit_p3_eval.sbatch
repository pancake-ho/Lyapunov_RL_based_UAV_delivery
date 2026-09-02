#!/usr/bin/bash
#SBATCH --job-name=p3-eval
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/p3-eval-%j.out
#SBATCH --error=logs/p3-eval-%j.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="/data/$USER/anaconda3/etc/profile.d/conda.sh"
readonly CHECKPOINT="${CHECKPOINT:-outputs/p3_ppo/best.pt}"
readonly SEEDS="${SEEDS:-3026,3027,3028}"
readonly FRAMES="${FRAMES:-100}"
readonly ROLLOUTS="${ROLLOUTS:-4}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-outputs/p3_eval_${SLURM_JOB_ID}}"

cd "$PROJECT_DIR"
mkdir -p logs "$OUTPUT_DIR"
source "$CONDA_SH"
conda activate lab

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing PPO checkpoint: $CHECKPOINT" >&2
  exit 2
fi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA allocation exists but PyTorch cannot open the GPU; refusing CPU fallback")
print("gpu", torch.cuda.get_device_name(0))
PY

python -m run.p3_compare \
  --policies dpp,ppo,rsu_only,always_hire,fixed_rsu,nearest_hotspot,load_threshold \
  --seeds "$SEEDS" \
  --frames "$FRAMES" \
  --rollouts "$ROLLOUTS" \
  --ppo-checkpoint "$CHECKPOINT" \
  --device cuda \
  --output "$OUTPUT_DIR"
