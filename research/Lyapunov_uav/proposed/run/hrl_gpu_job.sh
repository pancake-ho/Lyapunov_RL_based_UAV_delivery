#!/usr/bin/env bash
#SBATCH --job-name=hrl_train
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/hrl-%j.out
#SBATCH --error=logs/hrl-%j.err

# Prefer: bash run/hrl_submit.sh train
# Direct sbatch is supported when submitted from the proposed/ project root.
set -Eeuo pipefail
stage="initialization"
task_user="${USER:-$(id -un)}"
on_exit() {
    rc=$?
    trap - EXIT
    if (( rc == 0 )); then
        echo "[SBATCH-END] status=complete job=${SLURM_JOB_ID:-unknown}"
    else
        echo "[SBATCH-FAILED] stage=$stage exit=$rc job=${SLURM_JOB_ID:-unknown}" >&2
    fi
    exit "$rc"
}
trap on_exit EXIT
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use bash run/hrl_submit.sh train; this worker requires a Slurm allocation." >&2
    exit 2
fi
# Slurm copies this script to a spool directory; do not derive the project from $0.
project_root="${HPPO_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ ! -f "$project_root/config_hppo.py" || ! -f "$project_root/hppo/train.py" ]]; then
    echo "Project root not found: $project_root. Submit with run/hrl_submit.sh." >&2
    exit 2
fi
cd -- "$project_root"
mode="${HPPO_MODE:-train}"
if [[ "$mode" != train && "$mode" != smoke ]]; then echo "Invalid HPPO_MODE=$mode" >&2; exit 2; fi

threads="${HPPO_THREADS:-1}"
allocated_cpus="${SLURM_CPUS_PER_GPU:-16}"
if [[ ! "$threads" =~ ^[1-9][0-9]*$ || ! "$allocated_cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "HPPO_THREADS and allocated CPU count must be positive integers" >&2; exit 2
fi
if (( threads > allocated_cpus )); then echo "HPPO_THREADS exceeds allocated CPUs" >&2; exit 2; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"
task_tmp="${SLURM_TMPDIR:-/tmp}/hrl-${task_user}-${SLURM_JOB_ID}"
mkdir -p "$task_tmp/matplotlib" "$task_tmp/pycache"
export MPLCONFIGDIR="$task_tmp/matplotlib"
export PYTHONPYCACHEPREFIX="$task_tmp/pycache"

stage="conda"
conda_setup="${HPPO_CONDA_SH:-/data/$task_user/anaconda3/etc/profile.d/conda.sh}"
if [[ ! -f "$conda_setup" ]]; then echo "Set HPPO_CONDA_SH: missing $conda_setup" >&2; exit 2; fi
set +u
source "$conda_setup"
conda activate "${HPPO_CONDA_ENV:-lab}"
set -u

echo "[SBATCH-START] job=$SLURM_JOB_ID host=$(hostname) cwd=$PWD mode=$mode"
echo "[RESOURCES] GPU=1 CPUs_per_GPU=$allocated_cpus host_RAM=${SLURM_MEM_PER_GPU:-29G} torch_threads=$threads"
echo "[PYTHON] $(command -v python)"

stage="CUDA validation"
python -u - <<'PY'
import sys, numpy, torch, matplotlib
from pathlib import Path
if sys.version_info < (3,10):
    raise RuntimeError('Python 3.10 or newer is required')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable in the allocated GPU job; check lab PyTorch and the Slurm error log')
from config_hppo import HPPOConfig
from hppo.ppo import PPOAgent
cfg=HPPOConfig(device='cuda',num_regions=10,users_per_region=5)
agent=PPOAgent(cfg.slot_obs_dim,cfg.slot_action_nvec,cfg,'slot_ppo')
assert next(agent.net.parameters()).is_cuda
x=torch.randn(64,64,device='cuda',requires_grad=True)
(x@x).square().mean().backward()
torch.cuda.synchronize()
if not Path('tests/claude_physics_fixture.json').is_file():
    raise FileNotFoundError('Restore tests/claude_physics_fixture.json from the supplied fix package')
print('[CUDA-OK]',sys.version.split()[0],torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),flush=True)
print('[DEPENDENCIES] numpy=',numpy.__version__,'matplotlib=',matplotlib.__version__,flush=True)
PY

stage="HPPO contract tests"
python -u - <<'PY'
import unittest
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_hppo_env')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.testsRun == 0:
    raise SystemExit('HPPO tests failed, were skipped, or did not run')
PY

output_root="${HPPO_OUTPUT_DIR:-$project_root/outputs/hppo}"
mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
seed="${HPPO_SEED:-2026}"
episodes="${HPPO_EPISODES:-100}"
eval_episodes="${HPPO_EVAL_EPISODES:-5}"
for value in "$seed" "$episodes" "$eval_episodes"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then echo "Seed and episode counts must be integers" >&2; exit 2; fi
done
if (( episodes < 1 || eval_episodes < 1 )); then echo "Episode counts must be positive" >&2; exit 2; fi
run_name="hrl50_seed${seed}_job${SLURM_JOB_ID}"
smoke_name="${run_name}_smoke"
export HPPO_OUTPUT_ROOT_RESOLVED="$output_root"
export HPPO_RUN_NAME_RESOLVED="$run_name"
export HPPO_REQUESTED_EPISODES="$episodes"

stage="run metadata"
python -u - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
import torch
def git(*args):
    try:return subprocess.check_output(['git',*args],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):return None
payload={'job_id':os.environ['SLURM_JOB_ID'],'project_root':str(Path.cwd()),
         'git_commit':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),
         'git_status':git('status','--short'),'python':sys.version,'torch':torch.__version__,
         'gpu':torch.cuda.get_device_name(0),'run_name':os.environ['HPPO_RUN_NAME_RESOLVED'],
         'train_episodes':int(os.environ['HPPO_REQUESTED_EPISODES']),
         'slurm':{key:os.environ.get(key) for key in ('SLURM_JOB_PARTITION','SLURM_CPUS_PER_GPU','SLURM_MEM_PER_GPU','CUDA_VISIBLE_DEVICES')},
         'scope':'fresh training; checkpoint loading is not full training resume'}
path=Path(os.environ['HPPO_OUTPUT_ROOT_RESOLVED'])/('batch_'+os.environ['SLURM_JOB_ID']+'.json')
path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('[RUN-METADATA]',path,flush=True)
PY

# Same 50-user geometry, 256x256 network and 2-scenario comparator as training.
# The smoke horizon is shorter; its weights are never reused for the main run.
stage="GPU smoke train"
python -u -m hppo.train --mode train --device cuda --torch-num-threads "$threads" \
    --num-regions 10 --users-per-region 5 --seed "$seed" --hidden-dims 256,256 \
    --train-episodes 3 --num-frames 2 --frame-slots 10 --rollout-scenarios 2 \
    --reward-mode dpp --save-every-episodes 2 --console-log-every-slots 10 \
    --output-dir "$output_root" --run-name "$smoke_name"
stage="GPU smoke verification and evaluation"
python -u -m hppo.verify_trace "$output_root/$smoke_name"
python -u -m hppo.evaluate --train-run "$output_root/$smoke_name" --run-name "${smoke_name}_eval" \
    --eval-episodes 1 --episode-offset 1000000 --device cuda
python -u -m hppo.verify_trace "$output_root/${smoke_name}_eval"
python -u -m hppo.plot "$output_root/$smoke_name"
python -u -m hppo.plot "$output_root/${smoke_name}_eval"
echo "[GPU-SMOKE-PASS] $output_root/$smoke_name"
if [[ "$mode" == smoke ]]; then exit 0; fi

stage="main GPU training"
python -u -m hppo.train --mode train --device cuda --torch-num-threads "$threads" \
    --num-regions 10 --users-per-region 5 --seed "$seed" --hidden-dims 256,256 \
    --train-episodes "$episodes" --num-frames 30 --frame-slots 10 --rollout-scenarios 2 \
    --reward-mode dpp --save-every-episodes 2 --console-log-every-slots 10 \
    --output-dir "$output_root" --run-name "$run_name"
stage="main trace verification"
python -u -m hppo.verify_trace "$output_root/$run_name"
stage="held-out GPU evaluation"
python -u -m hppo.evaluate --train-run "$output_root/$run_name" --run-name "${run_name}_eval" \
    --eval-episodes "$eval_episodes" --episode-offset 1000000 --device cuda
stage="evaluation verification and plots"
python -u -m hppo.verify_trace "$output_root/${run_name}_eval"
python -u -m hppo.plot "$output_root/$run_name"
python -u -m hppo.plot "$output_root/${run_name}_eval"
echo "[RESULTS] train=$output_root/$run_name eval=$output_root/${run_name}_eval"
