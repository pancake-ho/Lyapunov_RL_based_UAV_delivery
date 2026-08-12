#!/usr/bin/bash

#SBATCH -J full-dpp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -t 1-0
#SBATCH --exclude=moana-y5
#SBATCH -o logs/slurm-labmeeting-%A.out

set -euo pipefail
umask 027


# =====================================================================
# Fixed paths
# =====================================================================

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"

CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

EXPECTED_BRANCH="feat/no-hrl"

JOB_ID="${SLURM_JOB_ID:-manual}"

MASTER_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}/labmeeting-master"

mkdir -p \
    "${MASTER_LOG_ROOT}"


die() {
    echo "[ERROR] $*" >&2
    exit 2
}


# =====================================================================
# Static repository contract
# =====================================================================

cd "${PROJECT_ROOT}"

CURRENT_BRANCH="$(git branch --show-current)"

[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die \
    "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"

[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || die \
    "Working tree must be clean before overnight evaluation."

COMMIT_SHA="$(git rev-parse HEAD)"

echo "============================================================"
echo "[LABMEETING OVERNIGHT START]"
echo "time=$(date --iso-8601=seconds)"
echo "job_id=${JOB_ID}"
echo "branch=${CURRENT_BRANCH}"
echo "commit=${COMMIT_SHA}"
echo "============================================================"


# =====================================================================
# Code preflight
# =====================================================================

"${PYTHON_BIN}" -m py_compile \
    agent/PPO/config.py \
    agent/PPO/fast/fast_train.py \
    agent/PPO/fast/plot_eval_diagnostics.py \
    agent/PPO/fast/plot_slot_diagnostics.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/test_fast_slow.py


bash -n \
    run/slow_full_eval.sh


"${PYTHON_BIN}" -m unittest -v \
    env.delivery.test_rsu_delivery_crn \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract


# =====================================================================
# Helper
#
# IMPORTANT:
# This is NOT another sbatch.
#
# slow_full_eval.sh is executed as an ordinary child bash process
# inside the current Slurm GPU allocation.
#
# Its internal `srun`s therefore create job steps under the same
# SLURM_JOB_ID.
# =====================================================================

run_eval() {
    local method="$1"
    local trace="$2"
    local snr="$3"
    local profile="$4"

    echo
    echo "============================================================"
    echo "[RUN START]"
    echo "method=${method}"
    echo "trace=${trace}"
    echo "snr=${snr}"
    echo "profile=${profile}"
    echo "time=$(date --iso-8601=seconds)"
    echo "============================================================"

    bash \
        run/slow_full_eval.sh \
        "${method}" \
        "${trace}" \
        "${snr}" \
        "${profile}"

    echo
    echo "============================================================"
    echo "[RUN DONE]"
    echo "method=${method}"
    echo "trace=${trace}"
    echo "snr=${snr}"
    echo "profile=${profile}"
    echo "time=$(date --iso-8601=seconds)"
    echo "============================================================"
}


# =====================================================================
# PHASE 1
# Core paper / lab-meeting benchmark
#
# Same frozen Fast ep300 checkpoint
# Same 5 seeds
# Same 50 rounds / seed
#
# Priority:
#   Random
#   RSU-only DPP
#   Full DPP
#
# These MUST finish before diagnostic experiments.
# =====================================================================

echo
echo "####################################################################"
echo "# PHASE 1: CORE 5-SEED COMPARISON"
echo "####################################################################"


run_eval \
    random \
    round \
    0 \
    full


run_eval \
    rsu_only \
    round \
    0 \
    full


run_eval \
    full_dpp \
    round \
    0 \
    full


# =====================================================================
# PHASE 2
# Generate core comparison figures immediately.
#
# Even if the job later reaches its wall-time during SNR diagnostics,
# the primary lab-meeting figures are already saved.
# =====================================================================

echo
echo "####################################################################"
echo "# PHASE 2: CORE FIGURES"
echo "####################################################################"


RANDOM_ROOT="${PROJECT_ROOT}/eval/full_ep300_random_full_round_snrp0db_job${JOB_ID}"

RSU_ONLY_ROOT="${PROJECT_ROOT}/eval/full_ep300_rsu_only_full_round_snrp0db_job${JOB_ID}"

FULL_DPP_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_full_round_snrp0db_job${JOB_ID}"

CORE_FIGURE_ROOT="${PROJECT_ROOT}/eval/figures/labmeeting_job${JOB_ID}/core"


"${PYTHON_BIN}" \
    -m \
    agent.PPO.fast.plot_eval_diagnostics \
    --random-root \
    "${RANDOM_ROOT}" \
    --rsu-only-root \
    "${RSU_ONLY_ROOT}" \
    --full-dpp-root \
    "${FULL_DPP_ROOT}" \
    --output-dir \
    "${CORE_FIGURE_ROOT}" \
    --seeds \
    2026 \
    2027 \
    2028 \
    2029 \
    2030 \
    --slow-t \
    3600


echo "[CORE FIGURES DONE]"
echo "output=${CORE_FIGURE_ROOT}"

find \
    "${CORE_FIGURE_ROOT}" \
    -maxdepth 1 \
    -type f \
    -printf '%f\n' \
    | sort


# =====================================================================
# PHASE 3
# Full-DPP slot-level diagnostic
#
# seed 2026
# 5 episodes × 10 rounds
# 180,000 REAL slots
#
# This is not a smoke test.
# =====================================================================

echo
echo "####################################################################"
echo "# PHASE 3: SLOT-LEVEL PHYSICAL TRACE"
echo "####################################################################"


run_eval \
    full_dpp \
    slot \
    0 \
    single


SLOT_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_slot_snrp0db_job${JOB_ID}"

SLOT_CSV="${SLOT_ROOT}/seed2026/logs/eval_slots.csv"

SLOT_FIGURE_ROOT="${PROJECT_ROOT}/eval/figures/labmeeting_job${JOB_ID}/slot_seed2026"


[[ -f "${SLOT_CSV}" ]] \
    || die \
    "Slot CSV was not created: ${SLOT_CSV}"


SLOT_ROWS="$(
    tail -n +2 \
    "${SLOT_CSV}" \
    | wc -l
)"


[[ "${SLOT_ROWS}" -eq 180000 ]] \
    || die \
    "Expected 180000 slot rows, got=${SLOT_ROWS}"


"${PYTHON_BIN}" \
    -m \
    agent.PPO.fast.plot_slot_diagnostics \
    --slot-csv \
    "${SLOT_CSV}" \
    --output-dir \
    "${SLOT_FIGURE_ROOT}" \
    --smooth-window \
    500


echo "[SLOT FIGURES DONE]"
echo "output=${SLOT_FIGURE_ROOT}"

find \
    "${SLOT_FIGURE_ROOT}" \
    -maxdepth 1 \
    -type f \
    -printf '%f\n' \
    | sort


# =====================================================================
# PHASE 4
# 3-point SNR sensitivity diagnostic
#
# Baseline 0 dB:
#   reuse PHASE 3 evaluation
#
# Additional:
#   -10 dB
#   +10 dB
#
# One seed only.
# This is explicitly a diagnostic, NOT final paper statistics.
# =====================================================================

echo
echo "####################################################################"
echo "# PHASE 4: SNR SENSITIVITY"
echo "####################################################################"


run_eval \
    full_dpp \
    round \
    -10 \
    single


run_eval \
    full_dpp \
    round \
    10 \
    single


SNR_M10_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_round_snrm10db_job${JOB_ID}"

SNR_P0_ROOT="${SLOT_ROOT}"

SNR_P10_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_round_snrp10db_job${JOB_ID}"

SNR_FIGURE_ROOT="${PROJECT_ROOT}/eval/figures/labmeeting_job${JOB_ID}/snr_seed2026"

mkdir -p \
    "${SNR_FIGURE_ROOT}"


# =====================================================================
# Generate 3-point SNR diagnostic.
#
# IMPORTANT:
# Only one seed is used.
# Therefore this is NOT plotted as a confidence interval.
# =====================================================================

"${PYTHON_BIN}" - \
    "${SNR_M10_ROOT}" \
    "${SNR_P0_ROOT}" \
    "${SNR_P10_ROOT}" \
    "${SNR_FIGURE_ROOT}" <<'PY'

import csv
import math
import sys

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


roots = {
    -10.0: Path(
        sys.argv[1]
    ),
    0.0: Path(
        sys.argv[2]
    ),
    10.0: Path(
        sys.argv[3]
    ),
}

output_dir = Path(
    sys.argv[4]
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = (
    (
        "realized_round_cost",
        "Realized DPP Cost",
        "snr_realized_dpp_cost.png",
    ),
    (
        "scheduled_stall_rate",
        "Scheduled Stall Rate",
        "snr_scheduled_stall_rate.png",
    ),
    (
        "delivery_per_scheduled_user_slot",
        "Delivery / Scheduled User-Slot",
        "snr_delivery_per_scheduled_user_slot.png",
    ),
    (
        "quality_per_chunk",
        "Quality / Chunk",
        "snr_quality_per_chunk.png",
    ),
)


summary = []


for snr_offset, root in roots.items():

    path = (
        root
        / "seed2026"
        / "logs"
        / "eval_rounds.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    if len(rows) != 50:
        raise RuntimeError(
            f"Expected 50 rows at {path}, "
            f"got {len(rows)}."
        )

    result = {
        "snr_offset_db":
            float(
                snr_offset
            ),
    }

    for key, _, _ in metrics:
        values = np.asarray(
            [
                float(
                    row[key]
                )
                for row in rows
            ],
            dtype=np.float64,
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        if finite.size == 0:
            result[key] = float(
                "nan"
            )
        else:
            result[key] = float(
                np.mean(
                    finite
                )
            )

    summary.append(
        result
    )


summary.sort(
    key=lambda item:
        item[
            "snr_offset_db"
        ]
)


summary_csv = (
    output_dir
    / "snr_sensitivity_seed2026.csv"
)


fieldnames = [
    "snr_offset_db",
    *[
        key
        for key, _, _
        in metrics
    ],
]


with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        summary
    )


x = np.asarray(
    [
        item[
            "snr_offset_db"
        ]
        for item
        in summary
    ],
    dtype=np.float64,
)


for key, ylabel, filename in metrics:

    y = np.asarray(
        [
            item[key]
            for item
            in summary
        ],
        dtype=np.float64,
    )

    fig, axis = plt.subplots(
        figsize=(
            7.2,
            4.8,
        )
    )

    axis.plot(
        x,
        y,
        marker="o",
        linewidth=1.8,
    )

    axis.set_xlabel(
        "Common SNR Offset [dB]"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_xticks(
        x
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.set_title(
        "Frozen Fast-PPO SNR Sensitivity "
        "(Seed 2026 Diagnostic)"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / filename,
        dpi=220,
    )

    plt.close(
        fig
    )


print(
    "[SNR FIGURES DONE]",
    output_dir,
)

for row in summary:
    print(
        row
    )

PY


# =====================================================================
# Final inventory
# =====================================================================

echo
echo "####################################################################"
echo "# FINAL INVENTORY"
echo "####################################################################"


echo
echo "[CORE FIGURES]"
find \
    "${CORE_FIGURE_ROOT}" \
    -maxdepth 1 \
    -type f \
    -printf '%p\n' \
    | sort


echo
echo "[SLOT FIGURES]"
find \
    "${SLOT_FIGURE_ROOT}" \
    -maxdepth 1 \
    -type f \
    -printf '%p\n' \
    | sort


echo
echo "[SNR FIGURES]"
find \
    "${SNR_FIGURE_ROOT}" \
    -maxdepth 1 \
    -type f \
    -printf '%p\n' \
    | sort


echo
echo "============================================================"
echo "[LABMEETING OVERNIGHT DONE]"
echo "time=$(date --iso-8601=seconds)"
echo "job_id=${JOB_ID}"
echo "commit=${COMMIT_SHA}"
echo "============================================================"