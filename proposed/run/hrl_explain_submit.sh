#!/usr/bin/env bash
# Submit a read-only explanation job using the project's existing Seraph partition.
# No training/evaluation/testing is started. CLI options are forwarded unchanged.
set -Eeuo pipefail
if (( $# < 1 )); then
    echo 'Usage: bash run/hrl_explain_submit.sh RUN_DIR [options]' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
run_dir="$(cd -- "$1" && pwd -P)"
shift
for file in trace.jsonl resolved_config.json; do
    [[ -f "$run_dir/$file" ]] || { echo "Missing: $run_dir/$file" >&2; exit 2; }
done
mkdir -p "$project_root/slurm_logs"
sbatch --job-name=hrl_explain --partition="${HPPO_PARTITION:-batch_eebme_ugrad}" \
    --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-gpu=16 --mem-per-gpu=29G \
    --time="${HPPO_EXPLAIN_TIME:-02:00:00}" --no-requeue --chdir="$project_root" \
    --output="$project_root/slurm_logs/hrl_explain-%j.out" \
    --error="$project_root/slurm_logs/hrl_explain-%j.err" \
    "$script_dir/hrl_explain_job.sh" "$project_root" "$run_dir" "$@"
