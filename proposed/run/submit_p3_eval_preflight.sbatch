#!/usr/bin/bash
#SBATCH --job-name=p3-eval-check
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/p3-eval-check-%j.out
#SBATCH --error=logs/p3-eval-check-%j.err

set -Eeuo pipefail

readonly PROJECT_DIR="${PROJECT_DIR:-/data/$USER/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed}"
readonly CONDA_SH="${CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
readonly CONDA_ENV="${CONDA_ENV:-lab}"
readonly EVAL_SEEDS="${EVAL_SEEDS:-3026:3027:3028:3029:3030:3031:3032:3033:3034:3035}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:load_threshold:always_hire:fixed_rsu:nearest_hotspot:rsu_only}"
readonly EVAL_JOB_WORKERS="${EVAL_JOB_WORKERS:-2}"
readonly BEST_CHECKPOINT="${BEST_CHECKPOINT:?BEST_CHECKPOINT must be exported}"
readonly LATEST_CHECKPOINT="${LATEST_CHECKPOINT:?LATEST_CHECKPOINT must be exported}"

cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[PREFLIGHT] timestamp=$(date --iso-8601=seconds) host=$(hostname)"
echo "[PREFLIGHT] branch=$(git branch --show-current) commit=$(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)"
test -f "$BEST_CHECKPOINT"
test -f "$LATEST_CHECKPOINT"
command -v flock >/dev/null
task_count="$(python -m run.p3_eval_shard \
    --seeds "$EVAL_SEEDS" \
    --baseline-policies "$EVAL_BASELINE_POLICIES" \
    --print-task-count)"
echo "[PREFLIGHT] tasks=$task_count job_workers=$EVAL_JOB_WORKERS"
if (( EVAL_JOB_WORKERS <= 0 || EVAL_JOB_WORKERS > 2 || EVAL_JOB_WORKERS > task_count )); then
    echo "Invalid EVAL_JOB_WORKERS=$EVAL_JOB_WORKERS for task_count=$task_count" >&2
    exit 2
fi
python -m unittest discover -s tests -p 'test_p3_*.py' -v
python - "$BEST_CHECKPOINT" "$LATEST_CHECKPOINT" <<'PY'
import sys
from pathlib import Path

from agent.P3.ppo_agent import PPOAgent
from config_p3 import P3Config

agent = PPOAgent(P3Config(), device="cpu")
for raw in sys.argv[1:]:
    path = Path(raw).resolve()
    metadata = agent.load(path)
    print(f"[PREFLIGHT] checkpoint={path} episode={metadata.get('episode')}")
PY
echo "[PREFLIGHT-DONE]"
