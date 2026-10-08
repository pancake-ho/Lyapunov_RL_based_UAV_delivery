#!/usr/bin/env bash
# Run from any directory: bash /path/to/proposed/run/hrl_submit.sh train
# This wrapper only checks files/Slurm status and submits; Python runs on the GPU node.
set -Eeuo pipefail

mode="${1:-train}"
if [[ $# -gt 1 || ( "$mode" != train && "$mode" != smoke ) ]]; then
    echo "Usage: bash run/hrl_submit.sh [train|smoke]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/.." && pwd -P)"
task_user="${USER:-$(id -un)}"
for rel in config_hppo.py hppo/train.py hppo/completion.py hppo/evaluate.py \
           tests/test_hppo_env.py tests/claude_physics_fixture.json run/hrl_gpu_job.sh; do
    if [[ ! -f "$project_root/$rel" ]]; then
        echo "Required file missing: $project_root/$rel" >&2
        exit 2
    fi
done
for command in sbatch sinfo squeue scontrol; do
    command -v "$command" >/dev/null || { echo "Slurm command missing: $command" >&2; exit 2; }
done
partition="${HPPO_PARTITION:-batch_eebme_ugrad}"
if [[ -z "$(sinfo -h -p "$partition" -o '%P')" ]]; then
    echo "Partition unavailable: $partition. Check sinfo and set HPPO_PARTITION." >&2
    exit 2
fi
sinfo -p "$partition" -o '%P %a %l %D %G'
squeue -u "$task_user"
scontrol show partition "$partition"
if command -v show-qos >/dev/null; then show-qos || true; fi
if command -v show-assoc >/dev/null; then show-assoc || true; fi

log_dir="$project_root/slurm_logs"
mkdir -p "$log_dir"
export HPPO_PROJECT_ROOT="$project_root"
export HPPO_MODE="$mode"
default_time="1-00:00:00"
if [[ "$mode" == smoke ]]; then default_time="01:00:00"; fi

# One process, one GPU. 29G is host RAM, not GPU VRAM.
sbatch --job-name="hrl_${mode}" --partition="$partition" \
    --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-gpu=16 --mem-per-gpu=29G \
    --time="${HPPO_TIME:-$default_time}" --no-requeue --chdir="$project_root" \
    --output="$log_dir/%x-%j.out" --error="$log_dir/%x-%j.err" \
    --export=ALL "$script_dir/hrl_gpu_job.sh"
echo "Submission requested. Logs: $log_dir/hrl_${mode}-JOBID.out and .err"
