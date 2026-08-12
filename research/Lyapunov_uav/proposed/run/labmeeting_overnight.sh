#!/usr/bin/bash

#SBATCH -J final-dpp
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
# Fixed project/runtime paths
# =====================================================================

PROJECT_ROOT="/data/surt321/repos/lab/uav_rsu/env/Lyapunov_RL_based_UAV_delivery/research/Lyapunov_uav/proposed"

CONDA_ROOT="/data/surt321/anaconda3"
CONDA_ENV_PATH="${CONDA_ROOT}/envs/lab"
PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

EXPECTED_BRANCH="feat/no-hrl"

JOB_ID="${SLURM_JOB_ID:-manual}"

MASTER_LOG_ROOT="${PROJECT_ROOT}/logs/jobs/${JOB_ID}/labmeeting-master"

STATUS_LOG="${MASTER_LOG_ROOT}/phase-status.tsv"

FIGURE_ROOT="${PROJECT_ROOT}/eval/figures/labmeeting_job${JOB_ID}"

CORE_FIGURE_ROOT="${FIGURE_ROOT}/core"

SLOT_FIGURE_ROOT="${FIGURE_ROOT}/slot_seed2026"

SNR_FIGURE_ROOT="${FIGURE_ROOT}/snr_seed2026"

START_EPOCH="$(date +%s)"

FAILURE_COUNT=0


# =====================================================================
# Helpers
# =====================================================================

die() {
    echo "[ERROR] $*" >&2
    exit 2
}


elapsed_seconds() {
    local now

    now="$(date +%s)"

    echo $((now - START_EPOCH))
}


record_status() {
    local phase="$1"
    local status="$2"
    local message="${3:-}"

    printf \
        '%s\t%s\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" \
        "${phase}" \
        "${status}" \
        "${message}" \
        >> "${STATUS_LOG}"
}


print_phase() {
    local title="$1"

    echo
    echo "####################################################################"
    echo "# ${title}"
    echo "####################################################################"
    echo
}


on_exit() {
    local exit_code=$?

    set +e

    echo
    echo "============================================================"
    echo "[LABMEETING MASTER EXIT]"
    echo "time=$(date --iso-8601=seconds)"
    echo "job_id=${JOB_ID}"
    echo "elapsed_seconds=$(elapsed_seconds)"
    echo "failure_count=${FAILURE_COUNT}"
    echo "exit_code=${exit_code}"
    echo "============================================================"

    if [[ -f "${STATUS_LOG}" ]]; then
        echo
        echo "[PHASE STATUS]"
        cat "${STATUS_LOG}" || true
    fi
}


trap on_exit EXIT


# =====================================================================
# Repository contract
# =====================================================================

[[ -d "${PROJECT_ROOT}" ]] \
    || die \
    "Project root does not exist: ${PROJECT_ROOT}"

[[ -x "${PYTHON_BIN}" ]] \
    || die \
    "Python executable does not exist: ${PYTHON_BIN}"

[[ -f "${PROJECT_ROOT}/run/slow_full_eval.sh" ]] \
    || die \
    "Missing run/slow_full_eval.sh"


cd "${PROJECT_ROOT}"


CURRENT_BRANCH="$(
    git branch --show-current
)"


[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || die \
    "Wrong branch: expected=${EXPECTED_BRANCH}, actual=${CURRENT_BRANCH}"


[[ -z "$(
    git status \
        --porcelain \
        --untracked-files=normal
)" ]] \
    || die \
    "Working tree must be clean before overnight evaluation."


COMMIT_SHA="$(
    git rev-parse HEAD
)"


mkdir -p \
    "${MASTER_LOG_ROOT}" \
    "${FIGURE_ROOT}"


: > "${STATUS_LOG}"


echo "============================================================"
echo "[LABMEETING OVERNIGHT START]"
echo "time=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
echo "job_id=${JOB_ID}"
echo "partition=${SLURM_JOB_PARTITION:-NOT_SET}"
echo "branch=${CURRENT_BRANCH}"
echo "commit=${COMMIT_SHA}"
echo "python=${PYTHON_BIN}"
echo "figure_root=${FIGURE_ROOT}"
echo "============================================================"


record_status \
    "master" \
    "START" \
    "commit=${COMMIT_SHA}"


# =====================================================================
# Static preflight
# =====================================================================

print_phase \
    "PREFLIGHT"


bash -n \
    run/slow_full_eval.sh \
    run/labmeeting_overnight.sh


"${PYTHON_BIN}" -m py_compile \
    agent/PPO/config.py \
    agent/PPO/fast/fast_train.py \
    agent/PPO/fast/plot_eval_diagnostics.py \
    agent/PPO/fast/plot_slot_diagnostics.py \
    agent/PPO/slow/slow_matching.py \
    agent/PPO/slow/test_fast_slow.py


"${PYTHON_BIN}" -m unittest -v \
    env.delivery.test_rsu_delivery_crn \
    agent.PPO.slow.test_fast_slow \
    agent.PPO.fast.test_fast_pretrain_contract


# ---------------------------------------------------------------------
# Protect against the exact Bash arithmetic bug seen in job 134712.
#
# We require arithmetic expansion:
#
#     $(( ... ))
#
# rather than command substitution:
#
#     $( ... )
# ---------------------------------------------------------------------

grep -Eq \
    '^[[:space:]]*EXPECTED_ROUND_ROWS=\$\(\(' \
    run/slow_full_eval.sh \
    || die \
    "slow_full_eval.sh: EXPECTED_ROUND_ROWS is not Bash arithmetic expansion."


grep -Eq \
    '^[[:space:]]*EXPECTED_SLOT_ROWS_PER_ROUND=\$\(\(' \
    run/slow_full_eval.sh \
    || die \
    "slow_full_eval.sh: EXPECTED_SLOT_ROWS_PER_ROUND is not Bash arithmetic expansion."


grep -Eq \
    '^[[:space:]]*EXPECTED_SLOT_ROWS=\$\(\(' \
    run/slow_full_eval.sh \
    || die \
    "slow_full_eval.sh: EXPECTED_SLOT_ROWS is not Bash arithmetic expansion."


# ---------------------------------------------------------------------
# Make sure the diagnostic arguments are actually wired to Fast config.
# ---------------------------------------------------------------------

grep -Fq \
    'export FAST_PPO_EVAL_SLOT_LOGGING=' \
    run/slow_full_eval.sh \
    || die \
    "FAST_PPO_EVAL_SLOT_LOGGING is not exported."


grep -Fq \
    'export FAST_PPO_EVAL_SLOT_LOG_STRIDE=' \
    run/slow_full_eval.sh \
    || die \
    "FAST_PPO_EVAL_SLOT_LOG_STRIDE is not exported."


grep -Fq \
    'export FAST_PPO_CHANNEL_SNR_OFFSET_DB=' \
    run/slow_full_eval.sh \
    || die \
    "FAST_PPO_CHANNEL_SNR_OFFSET_DB is not exported."


record_status \
    "preflight" \
    "PASS" \
    "syntax+compile+tests+launcher-contract"


echo "[PREFLIGHT] PASS"


# =====================================================================
# Evaluation helper
#
# IMPORTANT:
#
# slow_full_eval.sh is executed by bash, NOT sbatch.
#
# Therefore this master script still consumes exactly one Slurm job.
# slow_full_eval.sh creates only srun job steps inside this allocation.
# =====================================================================

run_eval() {
    local phase_key="$1"
    local method="$2"
    local trace="$3"
    local snr="$4"
    local profile="$5"

    local rc=0

    echo
    echo "============================================================"
    echo "[EVALUATION START]"
    echo "phase=${phase_key}"
    echo "method=${method}"
    echo "trace=${trace}"
    echo "snr_offset_db=${snr}"
    echo "profile=${profile}"
    echo "time=$(date --iso-8601=seconds)"
    echo "elapsed_seconds=$(elapsed_seconds)"
    echo "============================================================"

    record_status \
        "${phase_key}" \
        "START" \
        "method=${method},trace=${trace},snr=${snr},profile=${profile}"

    if bash \
        run/slow_full_eval.sh \
        "${method}" \
        "${trace}" \
        "${snr}" \
        "${profile}"
    then
        echo
        echo "[EVALUATION PASS] ${phase_key}"

        record_status \
            "${phase_key}" \
            "PASS" \
            "elapsed_seconds=$(elapsed_seconds)"

        return 0
    else
        rc=$?

        echo
        echo "[EVALUATION FAIL]"
        echo "phase=${phase_key}"
        echo "exit_code=${rc}"

        record_status \
            "${phase_key}" \
            "FAIL" \
            "exit_code=${rc}"

        FAILURE_COUNT=$((FAILURE_COUNT + 1))

        return "${rc}"
    fi
}


# =====================================================================
# Expected output roots
# =====================================================================

SLOT_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_slot_snrp0db_job${JOB_ID}"

SLOT_CSV="${SLOT_ROOT}/seed2026/logs/eval_slots.csv"


SNR_M10_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_round_snrm10db_job${JOB_ID}"

SNR_P0_ROOT="${SLOT_ROOT}"

SNR_P10_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_single_round_snrp10db_job${JOB_ID}"


FULL_DPP_ROOT="${PROJECT_ROOT}/eval/full_ep300_full_dpp_full_round_snrp0db_job${JOB_ID}"

RSU_ONLY_ROOT="${PROJECT_ROOT}/eval/full_ep300_rsu_only_full_round_snrp0db_job${JOB_ID}"

RANDOM_ROOT="${PROJECT_ROOT}/eval/full_ep300_random_full_round_snrp0db_job${JOB_ID}"


# =====================================================================
# State flags
# =====================================================================

SLOT_OK=0

SNR_M10_OK=0
SNR_P10_OK=0

FULL_DPP_OK=0
RSU_ONLY_OK=0
RANDOM_OK=0


# =====================================================================
# PHASE 1
#
# Priority for tomorrow's lab meeting:
#
#   Full-DPP real slot trace
#
# This directly produces:
#   - weighted reward terms
#   - queue
#   - UAV Tx power
#   - capacity / throughput
#   - energy
#   - SoC
#   - SNR
#
# 1 seed × 5 episodes × 10 rounds
# = 50 full rounds
# = 180,000 real slots
# =====================================================================

print_phase \
    "PHASE 1: FULL-DPP SLOT-LEVEL PHYSICAL TRACE"


if run_eval \
    "slot_full_dpp_0db" \
    full_dpp \
    slot \
    0 \
    single
then
    SLOT_OK=1
else
    echo \
        "[WARN] Slot evaluation failed. Continuing with remaining phases."
fi


# ---------------------------------------------------------------------
# Generate slot figures immediately so they survive even if a later
# phase reaches the Slurm wall-time.
# ---------------------------------------------------------------------

if [[ "${SLOT_OK}" -eq 1 ]]; then

    if [[ ! -f "${SLOT_CSV}" ]]; then
        echo \
            "[WARN] Slot evaluation passed but eval_slots.csv is missing."

        record_status \
            "slot_plot" \
            "FAIL" \
            "missing=${SLOT_CSV}"

        FAILURE_COUNT=$((FAILURE_COUNT + 1))

    else

        SLOT_ROWS="$(
            tail -n +2 \
            "${SLOT_CSV}" \
            | wc -l
        )"

        echo \
            "[SLOT CSV] rows=${SLOT_ROWS}"

        if [[ "${SLOT_ROWS}" -ne 180000 ]]; then
            echo \
                "[WARN] Expected 180000 slot rows, got=${SLOT_ROWS}."

            record_status \
                "slot_plot" \
                "FAIL" \
                "unexpected_rows=${SLOT_ROWS}"

            FAILURE_COUNT=$((FAILURE_COUNT + 1))

        else

            print_phase \
                "PHASE 1B: SLOT FIGURES"

            if "${PYTHON_BIN}" \
                -m \
                agent.PPO.fast.plot_slot_diagnostics \
                --slot-csv \
                "${SLOT_CSV}" \
                --output-dir \
                "${SLOT_FIGURE_ROOT}" \
                --smooth-window \
                500
            then
                record_status \
                    "slot_plot" \
                    "PASS" \
                    "output=${SLOT_FIGURE_ROOT}"

                echo \
                    "[SLOT FIGURES DONE] ${SLOT_FIGURE_ROOT}"

                find \
                    "${SLOT_FIGURE_ROOT}" \
                    -maxdepth 1 \
                    -type f \
                    -printf '%f\n' \
                    | sort
            else
                rc=$?

                echo \
                    "[WARN] Slot plotting failed with exit=${rc}."

                record_status \
                    "slot_plot" \
                    "FAIL" \
                    "exit_code=${rc}"

                FAILURE_COUNT=$((FAILURE_COUNT + 1))
            fi

        fi

    fi

fi


# =====================================================================
# PHASE 2
#
# 3-point channel sensitivity diagnostic.
#
# 0 dB:
#   reuse PHASE 1 Full-DPP slot evaluation.
#
# Additional:
#   -10 dB
#   +10 dB
#
# One seed only.
#
# This is a lab-meeting diagnostic, not final paper statistics.
# =====================================================================

print_phase \
    "PHASE 2: SNR SENSITIVITY"


if run_eval \
    "snr_m10db" \
    full_dpp \
    round \
    -10 \
    single
then
    SNR_M10_OK=1
else
    echo \
        "[WARN] -10 dB SNR run failed. Continuing."
fi


if run_eval \
    "snr_p10db" \
    full_dpp \
    round \
    10 \
    single
then
    SNR_P10_OK=1
else
    echo \
        "[WARN] +10 dB SNR run failed. Continuing."
fi


# ---------------------------------------------------------------------
# Generate SNR sensitivity figure only when all three points exist.
# ---------------------------------------------------------------------

if [[ "${SLOT_OK}" -eq 1 \
    && "${SNR_M10_OK}" -eq 1 \
    && "${SNR_P10_OK}" -eq 1 ]]
then

    print_phase \
        "PHASE 2B: SNR FIGURES"

    mkdir -p \
        "${SNR_FIGURE_ROOT}"

    if "${PYTHON_BIN}" - \
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
    ),
    (
        "scheduled_stall_rate",
        "Scheduled Stall Rate",
    ),
    (
        "delivery_per_scheduled_user_slot",
        "Delivery / Scheduled User-Slot",
    ),
    (
        "quality_per_chunk",
        "Quality / Chunk",
    ),
    (
        "quality_degradation_per_chunk",
        "Quality Degradation / Chunk",
    ),
    (
        "service_rate",
        "Service Rate",
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
            f"Expected 50 rows in {path}, "
            f"got={len(rows)}"
        )

    result = {
        "snr_offset_db":
            float(
                snr_offset
            ),
    }

    for key, _ in metrics:
        values = np.asarray(
            [
                float(
                    row[key]
                )
                for row
                in rows
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
        for key, _
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


fig, axes = plt.subplots(
    3,
    2,
    figsize=(
        11,
        11,
    ),
)


axes_flat = axes.reshape(
    -1
)


for axis, (
    key,
    ylabel,
) in zip(
    axes_flat,
    metrics,
):
    y = np.asarray(
        [
            item[key]
            for item
            in summary
        ],
        dtype=np.float64,
    )

    axis.plot(
        x,
        y,
        marker="o",
        linewidth=1.7,
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


fig.suptitle(
    "Frozen Fast-PPO SNR Sensitivity "
    "(Seed 2026 Diagnostic)"
)

fig.tight_layout()


figure_path = (
    output_dir
    / "snr_sensitivity_seed2026.png"
)


fig.savefig(
    figure_path,
    dpi=220,
)


plt.close(
    fig
)


print(
    "[SNR SUMMARY]"
)

for row in summary:
    print(
        row
    )


print(
    "[SNR FIGURE]",
    figure_path,
)

PY
    then
        record_status \
            "snr_plot" \
            "PASS" \
            "output=${SNR_FIGURE_ROOT}"

        echo \
            "[SNR FIGURES DONE] ${SNR_FIGURE_ROOT}"

        find \
            "${SNR_FIGURE_ROOT}" \
            -maxdepth 1 \
            -type f \
            -printf '%f\n' \
            | sort
    else
        rc=$?

        echo \
            "[WARN] SNR plotting failed with exit=${rc}."

        record_status \
            "snr_plot" \
            "FAIL" \
            "exit_code=${rc}"

        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi

else
    echo \
        "[WARN] Skipping SNR figure: one or more SNR evaluations are missing."

    record_status \
        "snr_plot" \
        "SKIP" \
        "slot=${SLOT_OK},m10=${SNR_M10_OK},p10=${SNR_P10_OK}"
fi


# =====================================================================
# PHASE 3
#
# Core 5-seed benchmark.
#
# Order is intentionally:
#
#   Full-DPP
#   RSU-only DPP
#   Random
#
# The expensive and research-critical proposed method is completed
# first. Random is expected to be the cheapest method and is left last.
#
# Every method:
#   5 seeds
#   × 5 episodes
#   × 10 rounds
# =====================================================================

print_phase \
    "PHASE 3: CORE 5-SEED FULL-DPP"


if run_eval \
    "core_full_dpp" \
    full_dpp \
    round \
    0 \
    full
then
    FULL_DPP_OK=1
else
    echo \
        "[WARN] Full-DPP 5-seed run failed. Continuing."
fi


print_phase \
    "PHASE 4: CORE 5-SEED RSU-ONLY DPP"


if run_eval \
    "core_rsu_only" \
    rsu_only \
    round \
    0 \
    full
then
    RSU_ONLY_OK=1
else
    echo \
        "[WARN] RSU-only 5-seed run failed. Continuing."
fi


print_phase \
    "PHASE 5: CORE 5-SEED RANDOM BASELINE"


if run_eval \
    "core_random" \
    random \
    round \
    0 \
    full
then
    RANDOM_OK=1
else
    echo \
        "[WARN] Random 5-seed run failed. Continuing."
fi


# =====================================================================
# PHASE 6
#
# Core Random / RSU-only / Full-DPP plots.
# =====================================================================

print_phase \
    "PHASE 6: CORE COMPARISON FIGURES"


if [[ "${FULL_DPP_OK}" -eq 1 \
    && "${RSU_ONLY_OK}" -eq 1 \
    && "${RANDOM_OK}" -eq 1 ]]
then

    if "${PYTHON_BIN}" \
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
    then
        record_status \
            "core_plot" \
            "PASS" \
            "output=${CORE_FIGURE_ROOT}"

        echo \
            "[CORE FIGURES DONE] ${CORE_FIGURE_ROOT}"

        find \
            "${CORE_FIGURE_ROOT}" \
            -maxdepth 1 \
            -type f \
            -printf '%f\n' \
            | sort
    else
        rc=$?

        echo \
            "[WARN] Core plotting failed with exit=${rc}."

        record_status \
            "core_plot" \
            "FAIL" \
            "exit_code=${rc}"

        FAILURE_COUNT=$((FAILURE_COUNT + 1))
    fi

else
    echo \
        "[WARN] Core plot skipped because at least one 5-seed method is incomplete."

    record_status \
        "core_plot" \
        "SKIP" \
        "full_dpp=${FULL_DPP_OK},rsu_only=${RSU_ONLY_OK},random=${RANDOM_OK}"
fi


# =====================================================================
# Final inventory
# =====================================================================

print_phase \
    "FINAL INVENTORY"


echo "[JOB]"
echo "job_id=${JOB_ID}"
echo "commit=${COMMIT_SHA}"
echo "elapsed_seconds=$(elapsed_seconds)"
echo "failure_count=${FAILURE_COUNT}"


echo
echo "[SLOT ROOT]"
echo "${SLOT_ROOT}"


echo
echo "[SNR ROOTS]"
echo "-10 dB: ${SNR_M10_ROOT}"
echo "  0 dB: ${SNR_P0_ROOT}"
echo "+10 dB: ${SNR_P10_ROOT}"


echo
echo "[CORE ROOTS]"
echo "Full-DPP: ${FULL_DPP_ROOT}"
echo "RSU-only: ${RSU_ONLY_ROOT}"
echo "Random:   ${RANDOM_ROOT}"


echo
echo "[FIGURES]"

if [[ -d "${FIGURE_ROOT}" ]]; then
    find \
        "${FIGURE_ROOT}" \
        -type f \
        -printf '%p\n' \
        | sort
else
    echo "(none)"
fi


echo
echo "[PHASE STATUS]"

cat \
    "${STATUS_LOG}"


record_status \
    "master" \
    "DONE" \
    "failure_count=${FAILURE_COUNT}"


echo
echo "============================================================"

if [[ "${FAILURE_COUNT}" -eq 0 ]]; then
    echo "[LABMEETING OVERNIGHT DONE: PASS]"
else
    echo "[LABMEETING OVERNIGHT DONE: PARTIAL]"
fi

echo "time=$(date --iso-8601=seconds)"
echo "job_id=${JOB_ID}"
echo "commit=${COMMIT_SHA}"
echo "failure_count=${FAILURE_COUNT}"
echo "============================================================"


# =====================================================================
# Make the final Slurm state explicit.
#
# We intentionally waited until ALL independent phases had a chance
# to run. A partial result therefore leaves all successful artifacts
# intact but returns non-zero at the very end.
# =====================================================================

if [[ "${FAILURE_COUNT}" -gt 0 ]]; then
    exit 1
fi

exit 0