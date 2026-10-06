# 직전 보상 수정본의 정의 — 이번 모듈화에서 동일하게 유지

## 3. 정확한 보상식

### PSNR과 계수

```text
P = (34.0, 36.64, 39.11, 41.64) dB
beta1 = 1
beta2 = (41.64 - 34.0) / 3 = 2.5466666666666664
beta3 = (41.64 - 34.0) / 1 second = 7.64
training_scale = 1 / 41.64
```


a) `beta2`: 평균 quality-index variation 3이 같은 시점의 Vq ladder 전체 차이
7.64 dB와 맞바뀐다는 **설계 가정**이다.

b) `beta3`: 누적 stall 1초가 같은 시점의 Vq ladder 전체 차이와 맞바뀐다는 설계 가정이다.
MOS나 실측 사용자의 선호에서 추정한 계수는 아니다.
`Qv`의 의미를 quality-index 차이로 유지하므로 `(K-1)`로 나누지 않는다.

c) raw 사용자 QoE:

\[
q_{u,t}=Vq_{u,t}-2.5466666667Qv_{u,t}-7.64Re_{u,t}.
\]

현 구조는 region별 rollout/GAE를 사용하므로 PPO에 넣는 값은

\[
r_{m,t}=\frac{1}{41.64}\sum_{u\in\mathcal U_m(t)}q_{u,t}.
\]

모든 region의 raw reward 합은 전체 사용자 QoE 합이다.
region별 policy credit assignment는 원 논문의 중앙화된 전역 policy와 구분되는 현재 adaptation이다.
전체 양의 상수 scaling은 선형 QoE 항 사이의 상대 가중치를 보존한다.
PPO의 finite-step 학습 궤적까지 scale에 무관하다는 주장은 하지 않는다.
`cfg.ppo_reward_scale`을 NDTVS reward에 다시 곱하지 않는다. 0–5 clipping도 하지 않는다.
로그와 그래프의 QoE는 **raw score**이며 MOS가 아니다.

### Vq: 식 (6), segment 2..S 기준

사용자별 `P*`는 episode의 **첫 요청** quality의 PSNR이다.
실패한 첫 요청도 P*를 정하지만 수신 segment 수 S에는 들어가지 않는다.
실제로 받은 각 chunk를 한 video segment로 해석하고 성공한 수신 sequence를 저장한다.
`S>=2`에서

\[
Vq=P^*-\frac{1}{S-1}\sum_{s=2}^{S}(P^*-P_s).
\]

성공 수신 sequence에서 z=1이므로 segment 2..S의 PSNR 평균과 같다.
첫 segment를 포함하는 보통의 전체 PSNR 평균으로 교체하지 않는다.
나중 PSNR이 P*보다 높을 때 signed gap이 음수가 될 수 있으며 이를 0으로 자르지 않는다.
한 번에 d개를 받으면 같은 quality의 d개 segment로 계산한다.

논문이 정의하지 않는 `S=0,1`의 코드 경계조건은 각각
`Vq=0`과 `Vq=첫 성공 수신 PSNR`, 두 경우 모두 `Qv=0`이다.
초기 Q=3 chunk는 공통 환경의 가상 prebuffer이고 quality 기록이 없으므로
임의의 PSNR을 부여하거나 수신 S에 넣지 않는다.

### Qv: 식 (7)의 연속 수신 quality-index 변화 평균

\[
Qv=\frac{1}{S-1}\sum_{s=2}^{S}|k_s-k_{s-1}|,\qquad S\ge2.
\]

한 batch 안의 d-1개 변화는 0이지만 분모에는 들어간다.
전송 실패나 idle은 segment를 만들지 않아 quality history가 바뀌지 않는다.
frame 또는 RSU region 이동으로 history를 초기화하지 않는다. episode 시작에서만 초기화한다.

### Re: 식 (9)의 공통 환경 대응

원 논문은 다운로드 시간과 download 시작 전 buffer를 이용해 rebuffering을 합산한다.
현 환경은 실패한 요청을 slot 말에 atomic discard하고 부분 다운로드 시간을 저장하지 않는다.
따라서 `요청 bits/capacity`를 억지로 넣어 논문 식 (9)의 정확한 재현이라고 하지 않는다.

현재 queue 단위가 chunk이고 한 chunk가 1초이므로, departure-before-arrival 순서에서

\[
\Delta Re_{u,t}=\max(b-Q^{before}_{u,t},0)\times1\text{ second},
\quad Re_{u,t}=\sum_{\tau\le t}\Delta Re_{u,\tau}.
\]

현재 b=1, slot=1초에서는 stall indicator와 같은 수치다.
fractional queue에서는 부족한 재생 시간만 계산한다.
스케줄되지 않은 사용자도 모두 반영한다.
과거 stall은 이후 slot의 QoE에도 계속 들어간다. 누적 Re 대신 이번 slot stall만 넣거나
QoE의 증가분 `q_t-q_{t-1}`을 보상으로 쓰지 않는다.
queue/전송 이벤트 순서를 바꾸어 이 보상만 유리하게 만들지 않는다.

## 4. 그래프의 stall과 quality를 읽는 방법

| 출력 | 정의 |
|---|---|
| `stall_ratio`, `stall_user_slot_ratio` | 기존 metric 보존. 전체 user-slot 중 `Q_before<b`인 비율. 전송 요청 실패율이 아니다. |
| `stall_time_ratio` | 실제 stall 초 합 / 전체 사용자 관측 초. 새 대표 그래프는 이 값을 %로 표시한다. 현재 정수 queue와 1초 slot에서는 기존 stall_ratio와 같다. |
| `average_received_psnr_db` | 성공 수신 chunk의 PSNR 합 / 성공 수신 chunk 수. |
| `average_quality_utility` | 위 평균 PSNR / 41.64. ladder는 `(0.8165225744, 0.8799231508, 0.9392411143, 1)`이다. `0.55/0.72/0.86/1`을 역산한 것이 아니다. |
| `paper_vq_per_user_slot` | 보상에 사용한 Vq를 전체 user-slot에서 평균. 전체 수신 PSNR 평균과 다른 metric이다. |
| `paper_qv_per_user_slot` | 사용자별 누적 sequence의 평균 index variation을 user-slot에서 평균. 이전 surrogate switching metric과 다르다. |
| `rebuffer_s_per_user` | episode 끝의 누적 Re를 전체 사용자에서 평균. |
| `cumulative_rebuffer_s_per_user_slot` | 각 시점의 누적 Re를 시간과 사용자에 대해 평균. 최종 Re와 다르다. |
| `paper_qoe_per_user_slot` | raw QoE 합 / 전체 user-slot. validation-best의 선택 기준이다. |
| `paper_qoe_final_per_user` | episode 마지막 raw QoE를 사용자에서 평균. 시간평균 QoE와 구분한다. |
| `legacy_average_quality_utility` | 기존 config의 heuristic U 기준 값. 과거 결과 추적용이며 새 대표 quality 그래프에 쓰지 않는다. |

새 대표 그래프의 quality는 모든 test episode의 수신 chunk를 pooling하여 계산한다.
수신 chunk가 하나도 없으면 quality를 정의할 수 없으므로 보고서에서 null로 처리한다.
episode 로그에는 0 sentinel와 `quality_utility_defined=false`가 함께 기록된다.
이 quality는 **수신 quality**다. 초기 buffer와 재생 순서의 quality가 기록되지 않으므로
playback PSNR이나 시청한 segment의 quality라고 부르면 안 된다.
quality가 일정하다는 것만으로 버그라고 판단하지 않는다. 수신이 성공한 요청의 quality 선택이
일정하거나, 높은 quality 실패가 수신 평균에 포함되지 않으면 충분히 나타날 수 있다.
`request_failure_ratio`, 수신량, stall을 함께 확인한다.

`ndtvs_analysis.py compare`와 기존 per-scenario plot은 episode 평균을 그대로 비교하는
보조 경로다. 논문용 SNR 요약의 pooled quality와 CI는 아래 `plot_snr_sweep.py` 경로를 사용한다.

