# UAVRTV: common-environment SAC baseline

기준 브랜치: `exp/v-sweep`, `2cb80f90e900f71c31f10c0e5b5e54a9bbe670ac`.
참고 논문: Dan Wu et al., *UAV-Assisted Real-Time Video Transmission for Vehicles:
A Soft Actor–Critic DRL Approach*, IEEE IoT Journal 11(8), 2024.

이 구현은 공통 RSU/UAV/buffer 시나리오에 맞춘 UAVRTV adaptation이다.
원 논문의 SUMO 도로, 연속 UAV trajectory, bandwidth 최적화를 그대로 재현한
실험이 아니다. SAC와 논문 형태의 QoE/energy 보상을 유지하며 사용자 요청으로
hiring decision/cost를 추가했다. 기존 UAVRTV 모델은 환경/보상/action 차원이 달라
새 코드에서 resume할 수 없다. 새 run에서 다시 학습해야 한다.

## 확정한 설계

| 항목 | 구현 |
|---|---|
| 물리 환경 | `proposed/hppo/env.py`의 `P3HierarchicalEnv` 상속. transition 재구현 없음 |
| 설정 | `proposed/outputs/hppo/hrl_resume_job145847/resolved_config.json`에서 공통 설정 읽기 |
| mobility/channel | 같은 seed/episode/frame의 초기 위치, ring-road mobility, 전체 fading trace |
| scheduling | 프레임 시작 현재 region 사용자 중 `(buffer, user ID)` 순서; RSU 우선, 다음 UAV |
| RSU 전송 | 거리/평균 fading=1로 highest-feasible-quality, 그 품질에서 최대 admissible chunk 요청 |
| 실제 전송 | 공통 actual fading에서 all-or-nothing 성공/실패. RSU가 실제 fading을 미리 읽지 않음 |
| SAC 제어 | 프레임 hiring/feasible hovering point; 슬롯 UAV chunk/quality/discrete-power |
| 자원 | 공통 fixed per-user bandwidth, 공통 total-power/battery projection |
| UAV 이동 | 공통 이산 hovering point 및 control-preparation interval reachability |
| 배터리 | 공통 relocation/hover/communication/자동 charging/reserve 제약 |
| 평가 지표 | NDTVS의 공통 `ServiceLogger`: stall-time/count, PSNR quality, 서비스 범위, per-user 통계 |

SAC는 한 actor와 twin online Q critics를 모든 region에 공유한다.
원 코드의 `(256,256)`, lr=0.001, gamma=0.99, tau=0.005, alpha=0.2를 기본 유지한다.
State에서 이미 비활성인 action 차원은 0으로 고정하며 그 차원은 entropy에서 제외한다.
프레임 boundary의 잠재 UAV 사용자 차원은 실제 hiring 결정 전에 열려 있다.
Gaussian latent action을 허용된 이산 품질/chunk/power/point로 대응시키는 adaptation이며
정확한 categorical/discrete SAC 구현으로 주장하지 않는다.
프레임 이후 hiring/point가 바뀌지 않는다. Proposed PPO나 DPP completion을 사용하지 않는다.

## 보상

각 region의 현재 slot 보상:

`r = beta * sum_success(PSNR_k / 41.64)
     - delta * sum_success(abs(bitrate_k - last_successful_bitrate))
     - phi * sum_all_users(rebuffer_seconds)
     - varsigma * actual_UAV_consumed_energy_J
     - frame_first_slot * lambda_h * hiring_cost_per_frame * hired`

- 논문 VI-A의 계수: beta=1, delta=1e-6 [/bps], phi=10 [/s], varsigma=0.01 [/J].
- 공통 PSNR ladder: `(34.0,36.64,39.11,41.64)`.
  품질 gain은 성공한 user-slot당 한 번이며 받은 chunk 수를 곱하지 않는다.
  이는 원 논문의 PSNR normalization 형태를 공통 quality ladder에 적용한 것이다.
- Bitrate는 `chunk_size_bits / chunk_playback_seconds`.
  첫 성공 수신의 switch penalty는 0. 실패/무수신은 품질 gain/switch penalty가 0.
- Rebuffer 초는 `(playback_chunks_per_slot - Q_before)^+ * chunk_playback_seconds`.
  부분적인 부족도 실제 초로 계산한다. 원 논문의 download-delay 식을 그대로 재현한
  것이 아니라 **adapted rebuffer-second penalty**이다.
- Stall count는 공통 평가 지표로 별도 기록하며 학습 보상에는 들어가지 않는다.
- 고용 비용은 공통 lambda_h/c_h, 프레임 첫 슬롯에서 한 번. V를 곱하지 않는다.
- 에너지는 공통 실제 relocation+hover+communication의 합. relocation도 한 번만
  포함하며 미고용 후 depot 복귀 에너지도 실제 소비된 만큼 포함한다.
  charging으로 배터리에 추가된 에너지는 소비 에너지 penalty에 넣지 않는다.
- Unscheduled user의 rebuffer도 포함한다.

`paper_qoe_*`는 비교용 **공통 NDTVS QoE observer 지표**이다.
UAVRTV 학습 보상은 `uavrtv_reward_*`, 고유 QoE 합은 `uavrtv_qoe_total`.
공통 환경의 DPP 관련 열은 공통 진단값이며 SAC가 그 값을 학습하지 않는다.

## 기능별 파일

| 경로 | 담당 기능 |
|---|---|
| `config.py` | 실험·보상·학습·Slurm 설정 |
| `common/settings.py` | 설정 검증, 저장된 공통 환경/source 검증 |
| `common/checkpoint.py` | atomic checkpoint, source hash, policy digest |
| `environment/shared.py` | 관측/action adapter, low-buffer scheduling, 평균 채널 RSU 규칙 |
| `models/sac.py` | actor/twin Q/target/temperature, replay buffer |
| `rewards/paper.py` | UAVRTV 보상 정의·항별 계산 |
| `training/rollout.py` | 공통 transition 순서, 데이터 수집, shared metrics |
| `training/engine.py` | 학습, validation, pause/resume, best 선택 |
| `evaluation/preflight.py` | 학습 전 reward magnitude 비교 |
| `evaluation/audit.py` | 보상 독립 재계산, 공통 physics verifier |
| `evaluation/sweep.py` | frozen best, 동일 SNR offset/seed의 독립 평가 |
| `plot/learning.py` | reward/validation/stall/quality/service/resource/component 그림 |
| `submit.py`, `job.sbatch` | config 기반 Slurm 실행 |
| `main.py` | 호환 entry point |
| `args.py`, `env.py`, `sac.py`, `logger.py` | 옛 별도 구현을 제거한 얇은 import 경로 |

## 적용 및 실행

ZIP의 `baseline/UAVRTV/`를 `baseline/UAVRTV/`에 덮어쓴다.
기존 run을 삭제할 필요는 없다. Proposed/NDTVS 파일은 교체하지 않는다.
이후 `repository root`에서 실행한다. 기존 lab 환경을 사용한다.

```bash
/data/surt321/anaconda3/envs/lab/bin/python baseline/UAVRTV/main.py inspect
python baseline/UAVRTV/submit.py --mode preflight
```

Preflight 결과는 `config.OUT/preflight/report.json`, 상세 항은 각 profile의
`reward_components.csv`, trace와 독립 `audit.json`에 생긴다.
profile은 `nohire`, `hired_max`, `random`이다. 모든 profile은 같은 시나리오를 사용한다.
mean/p95/nonzero-p95/max와 dominance flag를 출력한다. flag는 진단이며 계수를
자동 보정하거나 학습 결과로 간주하지 않는다. 현재 실제 환경의 항별 magnitude를
이 로그로 먼저 확인한다. 1회 테스트가 모든 상태/학습 정책을 대표하지는 않는다.
`analytic`은 실제 공통 설정에서 hover/move/hire/full-stall 항의 크기를 출력한다.

```bash
python baseline/UAVRTV/submit.py --mode smoke
python baseline/UAVRTV/submit.py --mode train --dry-run
python baseline/UAVRTV/submit.py --mode train
```

smoke와 preflight가 끝난 뒤 train을 제출한다. smoke는 공통 물리 환경을 그대로
유지하고 작은 네트워크로 2회 학습하는 소프트웨어 점검이다. 연구 결과로 쓰지 않는다.
smoke 출력은 `OUT`과 같은 부모 폴더의 `<OUT 이름>_smoke/`에 저장한다.
예: `runs/shared_sac_seed2026_smoke/`. 본 학습의 새 출력 폴더와 분리한다.
train은 기본 500회, 고정 validation 20회마다 5개 scenario, 마지막 episode에서도 validation.
설정은 `config.py`에서 바꾼다. export/echo는 필요 없다.
로그: `baseline/UAVRTV/slurm_logs/uavrtv-shared-sac-<mode>-<jobid>.out/.err`.

## 출력과 추가 학습

`config.OUT` 아래:

- `latest.pt`: 마지막 완료 에피소드의 모델/target/optimizer/temperature/replay/RNG/회차.
- `best.pt`: 고정 validation의 mean UAVRTV reward가 가장 높은 모델. test로 선택하지 않는다.
- `training.csv`, `validation.csv`, `status.json`, `runtime.json`, `model_size.json`.
- `episodes/`, `validation/`: summary/per-user/reward-component, trace 선택 저장.
- `plots/learning.png`, `plots/learning.pdf`: 정상 학습 종료 후 자동 생성.

PAUSED(exit 75), 시간 예산, SIGTERM/SIGUSR1/SIGINT는 에피소드 경계에서 멈춘다.
SIGKILL/scancel 강제 종료는 진행 중인 에피소드/validation을 잃을 수 있으나 마지막
atomic episode commit에서 재개한다. 중단된 scheduled validation은 다음 학습 전에
재수행한다. 실패 중 부분 적용된 SAC update를 새 checkpoint로 저장하지 않는다.
기존 replay/optimizer를 버린 채 weights만 로드하는 재학습은 exact resume가 아니다.

추가 학습은 `TRAIN_EPISODES`를 늘리고 같은 OUT/RESUME=True로 같은 train 명령을
제출한다. 학습 계수/physics/SAC learning/validation/source를 바꾸면 새 OUT이 필요하다.
`MAX_NEW_EPISODES`는 짧은 중단 점검용이다. 완료 또는 중단된 결과의 그림은:

```bash
/data/surt321/anaconda3/envs/lab/bin/python baseline/UAVRTV/main.py plot
```

smoke/장기 작업은 login node가 아니라 sbatch에서 실행한다. train 실행 중에는
다른 train을 같은 OUT에 중복 제출하지 않는다.

## Evaluation

```bash
python baseline/UAVRTV/submit.py --mode eval
```

`best.pt`를 `config.EVAL_OUT/inputs/`에 한 번 freeze한 뒤 SNR offset
`(-10,-5,0,5,10)` × scenario seed `(2026,2027,2028)` × 30개 episode를 평가한다.
조건/가중치는 학습 checkpoint에서 읽고, 비교 지표는 동일 ServiceLogger를 사용한다.
각 SNR/seed 첫 episode는 full trace audit한다. `state.json`, `episodes.csv`,
`means.csv`, `verification.json`, `episodes/*/per_user.csv`가 생긴다.
pause 후 같은 명령으로 누락된 scenario만 진행한다. zero-delivery quality는
undefined flag와 함께 기록하며 평균 quality에서 제외한다.
validation quality 평균과 학습 그래프에서도 zero-delivery quality는 제외한다.
UAVRTV는 기존 Proposed/NDTVS benchmark MODELS에 아직 등록하지 않는다.
현재 실행 중인 그 evaluator는 수정하지 않는다. UAVRTV 학습 완료 후 공통 세 방법
benchmark 연결과 그래프 합치는 작업을 진행할 수 있는 동일 seed/ID/fingerprint다.

## 로컬 소프트웨어 검증

```bash
/data/surt321/anaconda3/envs/lab/bin/python -m unittest baseline.UAVRTV.tests.test_shared -v
```

이 테스트는 CPU에서 작은 별도 설정으로 실제 공통 environment를 실행한다.
형식/물리 검증 목적이며 실제 145847 시나리오의 수렴/성능이나 GPU 검증을 대체하지 않는다.
