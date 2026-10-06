# NDTVS 기능별 폴더 모듈화 — 전체 소스와 실행 명령

이번 첨부본은 직전 `ndtvs_modular_update.zip`의 기능을 그대로 유지하면서
**파일뿐 아니라 실제 구현의 폴더도 담당 기능에 따라 나눈 버전**이다.
기준 저장소는 `pancake-ho/Lyapunov_RL_based_UAV_delivery`, `exp/v-sweep`,
`40f0ba62526b19eb744579dc74389f84392cca3f`다.
GitHub 원격에는 변경을 적용하지 않았다.

`files/research/Lyapunov_uav/baseline/NDTVS/`에 교체·추가 파일 53개의 전문이 있다.
기존 저장소의 `proposed/` 공통 환경·PPO 코드는 그대로 사용한다.

## 1. 기능별 폴더 구성

| 폴더 | 실제 구현 파일 | 담당 기능 |
|---|---|---|
| `NDTVS/common/` | `paths.py`, `config.py`, `io.py`, `runtime.py`, `checkpoint.py`, `source_compat.json` | 경로/config, 저장·읽기, RNG/device/walltime, checkpoint와 source 호환 검사. |
| `NDTVS/environment/` | `rsu.py` | RSU-only 제약과 UAV 미사용 completion adapter. |
| `NDTVS/models/` | `policy.py` | NDTVS network와 알고리즘별 PPO agent 생성. |
| `NDTVS/rewards/` | `qoe.py` | PSNR·계수·누적 사용자 history·보상식. |
| `NDTVS/metrics/` | `observer.py` | slot observer, QoE 기록, stall/quality 등 공통 metric 집계. |
| `NDTVS/training/` | `rollout.py`, `train.py`, `extend_training.py` | observation/episode 실행, 학습·validation-best·resume, 명시적 budget 연장. |
| `NDTVS/evaluation/` | `single.py`, `checks.py`, `policies.py`, `scenario.py`, `sweep.py`, `cli.py` | 단독 eval, checkpoint/pair 로딩, 동일 scenario 검증, paired smoke/eval/SNR sweep. |
| `NDTVS/analysis/` | `audit.py`, `compare.py`, `cli.py` | 물리·QoE 독립 검산, evaluation 설정 검증·비교, 기존 분석 CLI. |
| `NDTVS/plot/` | `plot_snr_sweep.py`, `plot_final_comparison.py`, `learning_diagnostics.py` | SNR/개별 scenario 비교 그래프, 학습 curve와 진단 그래프. **그림을 생성하는 구현은 여기에 있다.** |
| `NDTVS/tests/` | `test_ndtvs_common.py`, `test_ndtvs_modular.py` | 기존 의미 검증 15개 + 이동 호환/namespace 검증 9개. |

각 폴더의 `__init__.py`는 Python package 표시다.
`NDTVS/__main__.py`, `evaluation/__main__.py`, `analysis/__main__.py`는 `python -m` 진입점이다.

루트의 `api.py`는 공통 API를 모으고 `cli.py`는 기존 train/eval 인자를 처리한다.
루트에 남은 아래 파일은 기존 실행 명령을 유지하기 위한 **10줄짜리 연결 파일**이다.
학습·평가·그림 구현은 위 기능 폴더에만 있다.

- `ndtvs_common.py` → `api.py`와 학습/평가 구현.
- `ndtvs_analysis.py` → `analysis/cli.py`.
- `snr_sweep_eval.py`, `compare_eval_v2.py` → `evaluation/cli.py`.
- `plot_snr_sweep.py`, `plot_final_comparison.py` → `plot/`의 실제 구현.
- `extend_training.py` → `training/extend_training.py`.

기존 `run_common.sbatch`, `run_snr_sweep.sbatch`, `run_compare_eval_v2.sbatch`는
제출 진입점으로 루트에 유지했고 내용도 변경하지 않았다.
앞선 첨부본의 평면 구현 파일 23개는 적용 시 정리한다. 별도의 frozen 소스나 새 branch를 만들지 않는다.
`manifest.json.remove_files`에서 정확한 정리 목록을 확인할 수 있다.

## 2. 유지한 기능과 호환성

보상 계수·PSNR ladder·누적 stall·S=0/1 처리·quality 집계·모델 크기·PPO 설정·행동 순서·seed를 유지한다.
RSU region 제한, RSU-only, 자체 pointer scheduling, 공통 channel/mobility/queue와 전송 순서도 그대로다.
Proposed/HPPO-RSU의 학습 목적과 모델은 바꾸지 않았다.
상세 보상·metric 정의는 `REWARD_AND_METRICS_KR.md`에 있다.
`rewards/qoe.py`는 직전 보상 구현과 파일 내용이 동일하다.

폴더 이동에 필요한 import, 기준 경로, 구현 파일 hash 목록을 정리했다.
기존 두 첨부본(`ndtvs_qoe_update.zip`, `ndtvs_modular_update.zip`)의 **정확히 알려진 v2 소스**는
파일 이동으로 인정하므로 해당 checkpoint의 모델·optimizer·buffer·RNG를 유지하며 resume할 수 있다.
동일한 checkpoint/config/seed/offset을 사용한 partial eval과 paired sweep도 이어 실행할 수 있다.
새로 저장하는 결과에는 새 폴더 구조의 source metadata를 기록한다.

보상 변경 전 NDTVS v1은 기존처럼 거부한다. 다른 물리 소스나 임의 수정본을 무조건 허용하지 않는다.
기존 HPPO v1 observer migration 및 완전한 저장 config를 요구하는 검토된 default migration도 유지한다.
폴더 이동 자체 때문에 v2 run을 처음부터 다시 학습할 필요는 없다.

## 3. 적용

ZIP을 해제하면 `ndtvs_folder_modular_update/`가 생긴다.
현재 학습/평가 job이 종료된 뒤 적용한다.
본인의 수정은 먼저 git/local backup으로 보관한다.

```bash
REPO=/data/$USER/Lyapunov_RL_based_UAV_delivery
PACKAGE=/data/$USER/ndtvs_folder_modular_update

# 기존 checkout에서 적용한다.
git -C "$REPO" branch --show-current
git -C "$REPO" status --short

# 읽기 전용 사전 검사.
python "$PACKAGE/apply_update.py" "$REPO" --check

# 기능 폴더의 소스 적용 + 알려진 기존 평면 파일만 정리.
python "$PACKAGE/apply_update.py" "$REPO"

cd "$REPO/research/Lyapunov_uav"
source "${NDT_CONDA_SH:-/data/$USER/anaconda3/etc/profile.d/conda.sh}"
conda activate "${NDT_CONDA_ENV:-lab}"
python -m unittest discover -s baseline/NDTVS/tests -p 'test_ndtvs*.py' -v
```

`apply_update.py`는 모든 교체·정리 대상의 hash를 **쓰기 전에** 검사한다.
원 브랜치와 이전 두 첨부본, 이미 적용된 이번 첨부본의 파일을 식별한다.
다른 개인 수정이 있으면 해당 파일을 표시하고 아무것도 변경하지 않는다. 그 차이를 먼저 병합한다.
적용은 NDTVS의 지정된 source 파일만 대상으로 하며 기존 run/checkpoint/log 디렉터리를 건드리지 않는다.

폴더를 복사하는 것만으로는 앞선 평면 파일이 남으므로 위 적용 명령을 사용한다.
`changes_from_flat.patch`는 직전 평면 모듈화본과의 변경·삭제 확인용이다.
추가 dependency 설치나 별도 Conda 환경 준비는 필요 없다.

## 4. 새 폴더/패키지 실행 경로

아래 명령은 `research/Lyapunov_uav/`에서 실행한다.

| 작업 | 새 경로/명령 | 그대로 사용할 수 있는 기존 경로 |
|---|---|---|
| train 또는 단독 eval | `python -m baseline.NDTVS train ...` / `eval ...` | `python baseline/NDTVS/ndtvs_common.py train ...` / `eval ...` |
| paired smoke/eval/sweep | `python -m baseline.NDTVS.evaluation --mode ...` | `python baseline/NDTVS/snr_sweep_eval.py --mode ...` |
| audit/audit-run/progress/compare | `python -m baseline.NDTVS.analysis ...` | `python baseline/NDTVS/ndtvs_analysis.py ...` |
| SNR 그래프 | `python baseline/NDTVS/plot/plot_snr_sweep.py ...` | `python baseline/NDTVS/plot_snr_sweep.py ...` |
| 개별 scenario 그래프 | `python baseline/NDTVS/plot/plot_final_comparison.py ...` | `python baseline/NDTVS/plot_final_comparison.py ...` |
| 명시적 학습 연장 | `python baseline/NDTVS/training/extend_training.py ...` | `python baseline/NDTVS/extend_training.py ...` |

예를 들어 `progress`도 실제 graph 코드는 `plot/learning_diagnostics.py`에 있고
분석 CLI가 이를 호출한다. CLI 인자, output 형식과 계산 방법은 그대로다.

```bash
python -m baseline.NDTVS --help
python -m baseline.NDTVS.evaluation --help
python -m baseline.NDTVS.analysis --help
python baseline/NDTVS/plot/plot_snr_sweep.py --help
python baseline/NDTVS/training/extend_training.py --help
```

## 5. NDTVS 재학습/이어 학습 → 단독 eval

기존 Slurm 제출 파일과 명령을 그대로 사용할 수 있다.
`sinfo`/`scontrol show partition`에서 현재 사용할 GPU partition을 확인해 설정한다.
실제 선택한 proposed run의 완전한 저장 config와 matched checkpoint pair를 사용한다.
현재 사용 중인 run의 V 등 설정을 유지한다. 아래 경로는 직전 안내와 같은 예시다.

```bash
cd "$REPO/research/Lyapunov_uav"
mkdir -p slurm_logs
PARTITION=확인한_GPU_partition

PROPOSED_RUN=proposed/outputs/hppo/hrl_revision_train_seed2026_job142434
CONFIG="$PROPOSED_RUN/resolved_config.json"
PROPOSED_RUNTIME="$PROPOSED_RUN/runtime.json"
FRAME="$PROPOSED_RUN/checkpoints/frame_ep00124.pt"
SLOT="$PROPOSED_RUN/checkpoints/slot_ep00124.pt"
RSU=baseline/NDTVS/runs/common_gpu/hppo_rsu_seed2026_ep850/best.pt

NEW_TRAIN=baseline/NDTVS/runs/common_gpu/ndtvs_paper_qoe_seed2026_ep500
NEW_EVAL=baseline/NDTVS/runs/common_gpu/ndtvs_paper_qoe_comparison_seed2026

# 새 보상 학습을 처음 시작할 경우: 비어 있는 output.
sbatch --partition="$PARTITION" --gres=gpu:1 \
  baseline/NDTVS/run_common.sbatch \
  train --algorithm ndtvs --config "$CONFIG" --out "$NEW_TRAIN" \
  --device cuda --seed 2026 --episodes 500 \
  --val-every 25 --val-episodes 10 --trace-every 25
```

이미 이전 첨부본으로 v2 학습을 진행했다면 그 run의 **기존 인자와 output을 유지하고**
학습 명령에 `--resume`을 붙인다. 위 예시는 budget 500/seed 2026인 경우다.
목표 episode나 validation 간격을 모듈화 때문에 바꾸지 않는다.
보상 변경 전 NDTVS v1을 새 보상 학습의 초기 checkpoint로 사용하지 않는다.

```bash
# 위와 동일한 설정의 v2 run 이어 학습.
sbatch --partition="$PARTITION" --gres=gpu:1 \
  baseline/NDTVS/run_common.sbatch \
  train --algorithm ndtvs --config "$CONFIG" --out "$NEW_TRAIN" \
  --device cuda --seed 2026 --episodes 500 \
  --val-every 25 --val-episodes 10 --trace-every 25 --resume

# 진행 상태/학습 curve, 기록된 trace 검산.
srun --partition="$PARTITION" --gres=gpu:1 --time=00:10:00 \
  python -m baseline.NDTVS.analysis progress "$NEW_TRAIN"
srun --partition="$PARTITION" --gres=gpu:1 --time=00:10:00 \
  python -m baseline.NDTVS.analysis audit-run "$NEW_TRAIN"
```

학습 완료와 final validation을 확인한 뒤 아래 eval을 제출한다.
500은 기존 budget 예시이며 수렴을 보장하는 수치는 아니다.

```bash
sbatch --partition="$PARTITION" --gres=gpu:1 \
  baseline/NDTVS/run_common.sbatch \
  eval --algorithm ndtvs --checkpoint "$NEW_TRAIN/best.pt" \
  --out "$NEW_EVAL/ndtvs_only" --device cuda --episodes 30 \
  --scenario-seed 2026 --offset 7000000 --trace
```

단독 eval이 중단되면 동일 명령에 `--resume`을 붙인다.
단독 결과는 기존대로 `evaluation.json`, `evaluation.csv`, episode별 trace에 저장된다.

## 6. 공통 scenario 평가 → SNR 그래프

모든 방법을 동일 observer/scenario로 평가한다.
입력 경로는 실제 서버 파일을 지정하고, 해당 output을 쓰는 다른 작업이 종료된 뒤 제출한다.

```bash
test -f "$CONFIG"
test -f "$PROPOSED_RUNTIME"
test -f "$FRAME"
test -f "$SLOT"
test -f "$RSU"
test -f "$NEW_TRAIN/best.pt"

EVAL_INPUTS=(
  --out "$NEW_EVAL" --config "$CONFIG" --proposed-runtime "$PROPOSED_RUNTIME"
  --ndtvs-checkpoint "$NEW_TRAIN/best.pt" --rsu-checkpoint "$RSU"
  --frame-checkpoint "$FRAME" --slot-checkpoint "$SLOT"
  --device cuda --scenario-seed 2026
)

SMOKE_JOB=$(sbatch --parsable --partition="$PARTITION" \
  baseline/NDTVS/run_snr_sweep.sbatch \
  --mode smoke --offset 6000000 "${EVAL_INPUTS[@]}")

sbatch --partition="$PARTITION" --dependency="afterok:$SMOKE_JOB" \
  baseline/NDTVS/run_snr_sweep.sbatch \
  --mode sweep --offset 7000000 --episodes 30 "${EVAL_INPUTS[@]}"
```

sweep이 중단되면 동일 명령에 `--resume`을 붙인다.
기존 v2의 partial sweep도 같은 checkpoint·seed·offset·output을 유지하면 이어 실행할 수 있다.
완료 후 `verification.json`의 passed를 확인하고 새 plot 폴더 경로를 실행한다.

```bash
srun --partition="$PARTITION" --gres=gpu:1 --time=00:15:00 \
  python baseline/NDTVS/plot/plot_snr_sweep.py "$NEW_EVAL/sweep" \
  --out "$NEW_EVAL/summary"
```

nominal SNR만 평가할 때는 sweep 명령의 `--mode sweep`을 `--mode eval`로 바꾸고,
plot 입력도 `"$NEW_EVAL/eval"`로 바꾼다. 동일 output lock을 사용하므로 순차 실행한다.
추가 proposed pair는 기존 `--frame-checkpoint-2`, `--slot-checkpoint-2` 인자를 사용한다.
새 pair를 선택하면 새 평가 output에서 smoke부터 진행한다.

## 7. 검증 기록

- 테스트 **24개 통과**: 기존 15개, 이전 파일 이동 호환 6개, 이전 평면 구조/namespace 검증 3개.
- 기존 진입점 7개의 `--help` 출력 동일. 새 package/폴더 명령 6개도 실행 확인.
- 이전 평면 코드와 새 폴더 코드를 각각 별도 프로세스로 실행:
  NDTVS/HPPO-RSU/Proposed의 stochastic 행동·보상·metric·trace와
  update 후 모델·optimizer·buffer·RNG가 정확히 일치.
- 작은 동일 config에서 NDTVS/HPPO-RSU 3-episode 학습, validation-best,
  원 보상 수정본과 평면 모듈화본의 실제 checkpoint를 이어 학습한 상태도 일치.
- 이전 평면 코드의 partial 단독 eval과 5 SNR × 3방법 × 2 episode sweep을
  새 폴더 코드로 이어 실행하여 metric/scenario/policy 일치를 확인.
- 같은 평가 입력에서 새 `plot/` 코드의 summary/CI/paired difference와 PNG가
  이전 plot 출력과 byte 단위로 동일. 양쪽 PDF 생성도 확인.
- 보상 파일은 byte 단위 동일. 이동한 정의 72개 중 68개는 import 경로를 대응시키면 AST가 동일.
  나머지 4개는 알려진 source layout 허용, evaluator 파일 경로, 연장 명령 경로 처리다.
- Python 3.10 문법·compile, sbatch/문서 shell 문법, 적용 script와 patch, ZIP 무결성 검사.

수치 비교에서는 wall-clock 시간, timestamp 기반 trace directory, 변경된 source layout metadata를 제외했다.
CPU / Python 3.12 / PyTorch 2.14.1+cpu의 작은 검증 실험이다.
서버의 장시간 CUDA 재학습이나 논문용 전체 성능 실험은 실행하지 않았다.
`VALIDATION.json`, `MOVE_VALIDATION.json`, `CLI_VALIDATION.json`, `TEST_RESULTS.txt`에 기록을 첨부했다.
