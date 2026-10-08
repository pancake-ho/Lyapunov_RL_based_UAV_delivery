#!/usr/bin/bash

set -Eeuo pipefail

readonly EXPECTED_BRANCH="feat/new-form-p3"
readonly PROJECT_DIR="${PROJECT_DIR:-$PWD}"
readonly P3_RUN_DIR="${P3_RUN_DIR:?export P3_RUN_DIR to the accepted Stage-2 training output}"

# Objective default for the current code:
# one evaluation trajectory = one independent seed, each kept at the same
# 400-frame horizon used by training/validation. 30 held-out seeds provide
# 30 independent trajectory-level samples (960,000 user-slots per policy for
# the current 8-user, 10-slot/frame config). Final rule baselines are reduced to
# Proposed(DPP), Always-hire, and RSU-only; PPO best/latest are added by the shard runner.
readonly EVAL_SEEDS="${EVAL_SEEDS:-120026:120027:120028:120029:120030:120031:120032:120033:120034:120035:120036:120037:120038:120039:120040:120041:120042:120043:120044:120045:120046:120047:120048:120049:120050:120051:120052:120053:120054:120055}"
readonly EVAL_FRAMES="${EVAL_FRAMES:-400}"
readonly EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
readonly EVAL_BASELINE_POLICIES="${EVAL_BASELINE_POLICIES:-dpp:always_hire:rsu_only}"
readonly EVAL_ROOT="${EVAL_ROOT:-$PROJECT_DIR/outputs/p3_eval_stage2_reduced_s30_f400}"
readonly EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
readonly EVAL_SELECTION_WORKERS="${EVAL_SELECTION_WORKERS:-8}"
readonly MAX_PARALLEL="${MAX_PARALLEL:-2}"

cd "$PROJECT_DIR"
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
    echo "Expected branch $EXPECTED_BRANCH, got $(git branch --show-current)" >&2
    exit 2
fi
if [[ -n "$(git status --short)" ]]; then
    echo "Refusing dirty worktree for final evaluation" >&2
    git status --short >&2
    exit 2
fi
if [[ ! -f "$P3_RUN_DIR/run_config.json" ]]; then
    echo "Missing training run_config.json: $P3_RUN_DIR/run_config.json" >&2
    exit 2
fi
if [[ ! -f "$P3_RUN_DIR/best.pt" || ! -f "$P3_RUN_DIR/latest.pt" ]]; then
    echo "Missing best/latest checkpoint in $P3_RUN_DIR" >&2
    exit 2
fi

# Enforce held-out evaluation seeds. The original default 3026..3035 includes
# 3035, which is also seed=2026 + 1*1009 from the 100-episode training schedule.
python - "$P3_RUN_DIR/run_config.json" "$EVAL_SEEDS" "${P3_RUN_DIR}/acceptance.json" <<'PY'
import json
import sys

run_config_path, seed_text, acceptance_path = sys.argv[1:]
with open(run_config_path, encoding="utf-8") as stream:
    run = json.load(stream)

eval_seeds = {int(value) for value in seed_text.replace(":", ",").split(",") if value}
base_seed = int(run["seed"])
episodes = int(run["episodes"])
training_seeds = {base_seed + episode * 1009 for episode in range(episodes)}
validation_seeds = {int(value) for value in run.get("validation_seeds", [])}
acceptance_seeds = set()
try:
    with open(acceptance_path, encoding="utf-8") as stream:
        acceptance = json.load(stream)
    acceptance_seeds = {
        int(row["seed"])
        for row in acceptance.get("ppo_seed_summaries", acceptance.get("seed_summaries", []))
        if isinstance(row, dict) and "seed" in row
    }
except (FileNotFoundError, json.JSONDecodeError):
    pass

overlap = {
    "training": sorted(eval_seeds & training_seeds),
    "validation": sorted(eval_seeds & validation_seeds),
    "acceptance": sorted(eval_seeds & acceptance_seeds),
}
if any(overlap.values()):
    raise SystemExit(f"Evaluation seeds are not held out: {overlap}")
print(
    f"[SEED-AUDIT] held-out eval seeds={len(eval_seeds)} "
    f"training={len(training_seeds)} validation={len(validation_seeds)} "
    f"acceptance={len(acceptance_seeds)}"
)
PY

export PROJECT_DIR P3_RUN_DIR EVAL_ROOT EVAL_SEEDS EVAL_FRAMES EVAL_ROLLOUTS
export EVAL_BASELINE_POLICIES EVAL_DEVICE EVAL_SELECTION_WORKERS MAX_PARALLEL

exec bash run/p3_submit_eval.sh
