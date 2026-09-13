# SAC Baseline (Wu et al., IEEE IoT-J 2024) in the RSU + UAV Vehicular Video-Delivery Scenario

Reference 논문 *"UAV-Assisted Real-Time Video Transmission for Vehicles: A Soft Actor–Critic DRL Approach"*의
결정 구조(UAV trajectory + bandwidth allocation + SVC layer selection, SAC, reward = QoE − ςE)를
**연구 시나리오(Lyapunov guide)에 고정된 환경** 위에 baseline으로 구현한 코드입니다.

```
sac_uav_baseline/
├── args.py      # 모든 파라미터 (argparse)
├── env.py       # 시나리오 환경 (RSU+UAV, frame/slot, playback queue, battery, hiring cost)
├── sac.py       # SAC (Gaussian actor, twin soft-Q, auto temperature) — torch
├── logger.py    # per-slot debug log (txt + jsonl), episodes.csv
├── main.py      # train / eval loop
└── requirements.txt
```

## 1. 실행

```bash
pip install -r requirements.txt

# 환경/로그 sanity check (torch 불필요)
python main.py --agent random --episodes 2 --frames_per_episode 5 --log_dir runs/rand

# reference 설정으로 SAC 학습 (lr=1e-3, SAC가 UAV bandwidth 배분)
python main.py --agent sac --episodes 300 --log_dir runs/sac --slot_log_every_ep 10

# 시나리오 기본 radio 모델(fixed reserved RB) + 규칙 기반 hiring
python main.py --bw_mode fixed --hire_mode threshold --log_dir runs/sac_fixedrb

# 체크포인트 deterministic 평가
python main.py --eval_only true --load_path runs/sac/sac.pt --episodes 5 --log_dir runs/eval

python main.py -h   # 전체 파라미터 목록
```

## 2. Reference 논문 → 연구 시나리오 mapping

| Reference (Wu et al.) | 이 코드 (연구 시나리오) |
|---|---|
| 단일 UAV, 다수 GMV | region(RSU)당 persistent UAV 1대. 모든 region이 **하나의 SAC policy를 공유**(parameter sharing), replay buffer 공유 |
| state: UAV/GMV 위치·속도, bandwidth, 현재 layer | obs = UAV(위치, 속도, SoC, hired, slot-in-frame, 사용자 수) + N_max 개 user slot × [present, by_uav, by_rsu, 상대위치, 속도, Q/Qe, last_k, gain, last_stall] |
| action: UAV velocity, bandwidth b_m, layer l_m | action = `[v_x, v_y, hire, bw_1..bw_Nmax, k_1..k_Nmax, l_1..l_Nmax]` ∈ [−1,1]. `hire`는 frame 첫 slot에서만 사용, bw는 UAV-scheduled user들 간 softmax, k → quality level, l → chunk 수(0..L_max) |
| 매 slot 하나의 GOP 전송 | chunk 단위 전송. 실제 전송 chunk = min(l_req, ⌊CΔ/S_k⌋, buffer room). playback queue Q(t+1)=[Q−b]⁺+d |
| time delay penalty φ·D | stall penalty φ·1{Q_n(t)<b} (rate 제약을 항상 만족시키므로 transmission delay = 0) |
| bitrate switching δ·\|R_k−R_k'\| | 동일 (전송이 있는 slot에서 quality 변경 시) |
| video quality β·PSNR/PSNR_max | β·Σ l·U_k (`--quality_reward per_chunk`) 또는 β·U_k·1{l>0} (`per_user`) |
| ς·E_total (rotary-wing 모델 식 (3)) | 동일 propulsion 모델 + RF 에너지 Δ/η_PA·Σp, **physical battery E_u(t)** 추적 |
| 없음 | hiring cost λ_H·c_H/T (hired slot마다), RSU(J_R명, fixed RB, rule 기반), 배터리 threshold/automatic return/charging |

### 시나리오 고정 요소 (guide 문서 기준)
* Frame r = T slots. Frame 시작: N_m(r) 고정, hiring 결정, association(RSU가 Q 낮은 순 J_R명, UAV가 다음 J_U명).
  Frame 중 도착한 차량은 다음 frame까지 unserved.
* RSU: `W_R/J_R`, `P_R/J_R` fixed reserved RB, 규칙 기반 (l,k) 결정 (`--rsu_rule maxq|maxchunks`). RSU 결정은 agent 외부.
* UAV radio: `--bw_mode sac`(reference처럼 SAC가 bandwidth/power share 결정) 또는 `fixed`(W_U/J_U, P_U_max/J_U).
* Battery: frame 시작 hiring feasibility `E ≥ E_th + T·P_prop,max·Δ`, 매 slot reserve 검사(위반 시 FORCED_RETURN),
  미고용 시 depot(RSU 위치)에서 charging, depot 복귀 시 e_rel 차감.
* Z_n = Qe − Q_n 은 로그/지표용으로 기록됩니다 (baseline reward에는 사용되지 않음).

### Reference 대비 의도된 단순화 (필요 시 수정)
* UAV trajectory는 reference처럼 **연속 velocity 제어** (discrete hovering point 아님). 비행 영역은 region ± `--flight_x_margin`.
* 미고용→고용 시 depot에서 이륙, 고용→미고용 시 즉시 depot 복귀(e_rel 차감). Control-preparation interval τ^c는 명시적으로 모델링하지 않음.
* 도로는 직선 corridor(양방향 2차로), `--hotspot_region`에서 속도 저하로 정체를 만듦. SUMO trace 연동은 `_spawn`/mobility 부분을 교체하면 됨.

## 3. Debug log (매 time slot, 매 region)

`--slot_log true`(기본)이면 `<log_dir>/slot_log.txt`, `slot_log.jsonl`에 기록됩니다.
한 slot·region 블록 구조:

```
[ep 0 train | t=12 | region 0] UAV before: hired=1 pos=(194,-19) vel=(..) E=..J soc=.. | members=[..] rsu=[..] uav=[..] overflow=[..] (hired)
  STATE  : veh21: x=.. y=.. v=.. Q=5 Z=45 lastk=0 by=uav slot=3 r_now=0 | ...        ← 이번 slot 시작 상태
  ACTION : raw=[...]                                                                   ← SAC 원시 action
  UAV    : vel_cmd=(..) -> vel=(..) |v|=.. pos=(..) P_prop=..W e_com=..J e_slot=..J    ← trajectory/energy 결정
  UAV-TX : veh21 bw=0.84 W=4.21MHz p=1.69W d=202m snr=38dB C=53.7Mbps k=2 l_req=2 l_max=53 room=46 -> l=2 | ...
  RSU-TX : veh6 d=169m C=26.2Mbps lmax/k=[52,26,13,6] room=50 -> k=4 l=4 | ...
  NEXT   : veh0 Q 35->35 Z=15 d=1 k=2 stall=0 sw=3.0Mbps | ...                          ← queue/stall/switch 갱신
  MOVED  : veh0@256(r0), ...                                                           ← mobility 후 위치/region
  UAV after: hired=1 pos=(..) E=..J soc=..   [FORCED_RETURN]
  EVENTS : ARRIVE vid=16 (unserved until next frame); DEPART vid=3; FORCED_RETURN: ...
  REWARD : total=+x = quality a - switch b - stall c - energy d - hire e
```

`slot_log.jsonl`은 같은 내용을 JSON 한 줄/레코드로 담고 있어 pandas로 바로 분석할 수 있습니다:

```python
import pandas as pd, json
rows = [json.loads(l) for l in open("runs/sac/slot_log.jsonl")]
df = pd.json_normalize(rows)          # reward.total, uav_after.soc, ... 컬럼
```

`episodes.csv`: episode별 return, stall_ratio, avg_chunk_utility, hire_rate, uav_energy_Wh, switch_events, SAC loss/alpha.

## 4. Calibration 메모
* 기본 radio 값에서 RSU/UAV rate가 quality ladder 대비 여유가 큽니다(guide §10.5 지적). `--P_R --alpha_R --W_U --P_U_max --alpha_U --beta_*_dB`로 rate를,
  `--J_R --J_U`로 동시 서비스 수(시나리오의 실제 bottleneck)를 조정하세요.
* `--E_max_Wh`를 작게(예 2–5) 두면 episode 안에서 battery/charging cycle이 나타납니다.
* reward 스케일: `--phi`(stall), `--varsigma`(에너지 $/J), `--c_H --lambda_H`(hiring)를 함께 조정.
