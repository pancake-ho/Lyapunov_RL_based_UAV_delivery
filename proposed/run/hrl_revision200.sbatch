#!/usr/bin/env bash
# Submit using run/hrl_revision_submit.sh from any working directory.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Use sbatch via hrl_revision_submit.sh}"
: "${HPPO_PROJECT_ROOT:?Project root was not provided}"
cd -- "$HPPO_PROJECT_ROOT"
task_user="${USER:-$(id -un)}"
mode="${HPPO_MODE:-train}"
if [[ "$mode" != train && "$mode" != smoke ]]; then
    echo "Invalid mode: $mode" >&2; exit 2
fi
set +u
source "${HPPO_CONDA_SH:-/data/$task_user/anaconda3/etc/profile.d/conda.sh}"
conda activate "${HPPO_CONDA_ENV:-lab}"
set -u
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/hppo-revision-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable: CPU fallback is disabled')
print(sys.version, torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)
PY
seed="${HPPO_SEED:-2026}"
episodes="${HPPO_EPISODES:-200}"
frames=30
if [[ "$mode" == smoke ]]; then
    python -m unittest tests.test_hppo_env tests.test_hppo_revision -v
    episodes=3; frames=3
fi
output_root="${HPPO_OUTPUT_DIR:-$PWD/outputs/hppo}"
run_name="hrl_revision_${mode}_seed${seed}_job${SLURM_JOB_ID}"
run_dir="$output_root/$run_name"
mkdir -p "$output_root"
printf '[RUN_DIR] %s\n' "$run_dir"
git rev-parse HEAD || true
git status --short || true
observer_pid=''
cleanup() {
    if [[ -n "$observer_pid" ]]; then
        kill "$observer_pid" 2>/dev/null || true
        wait "$observer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
(
    while [[ ! -s "$run_dir/resolved_config.json" || ! -s "$run_dir/trace.jsonl" ]]; do
        sleep 5
    done
    exec python -m hppo.export_files "$run_dir" --out "$run_dir/artifacts/live" --watch 300 --dpi 100
) > "$output_root/${run_name}_export.log" 2>&1 &
observer_pid=$!
python -u -m hppo.train \
    --mode train --device cuda --torch-num-threads 1 \
    --num-regions 10 --users-per-region 5 --hidden-dims 256,256 \
    --seed "$seed" --train-episodes "$episodes" --num-frames "$frames" \
    --frame-slots 10 --rollout-scenarios 2 --reward-mode dpp \
    --rsu-total-bandwidth-hz 3000000 --uav-total-bandwidth-hz 5000000 \
    --mask-queue-actions true --delivery-mode all_or_nothing \
    --save-every-episodes 2 --console-log-every-slots 10 \
    --output-dir "$output_root" --run-name "$run_name"
python -m hppo.verify_trace "$run_dir"
python -m hppo.evaluate --train-run "$run_dir" --run-name "${run_name}_eval" \
    --eval-episodes "${HPPO_EVAL_EPISODES:-5}" --episode-offset 1000000 --device cuda
python -m hppo.verify_trace "${run_dir}_eval"
if ! wait "$observer_pid"; then
    echo "Live export failed; see ${run_name}_export.log" >&2
fi
observer_pid=''
# All trace/CSV rows are retained. PNG/GIF export defaults to the final episode.
# Use the documented --all-episodes command to render all 6,000 frame images.
python -m hppo.export_files "$run_dir" --out "$run_dir/artifacts/final" --dpi 100
python -m hppo.explain_trace "$run_dir" --out "$run_dir/artifacts/explanation"
printf '[COMPLETE] %s\n' "$run_dir"
