## Overview

RSU-UAV-user의 통합 구조 하에서 차량 네트워크에서의 UAV-Assisted Communication Optimization 연구를 다루는 코드를 관리합니다.

해당 연구는 **two-timescale joint optimization problem**에 중점을 맞추며, QoE와 QoS의 공동 최적화를 목표로 진행하고 있습니다.
구체적인 사항은 다음과 같습니다.:

- **Slow Timescale**
  - UAV 고용
  - RSU/UAV 스케줄링

- **Fast Timescale**
  - chunk/layer 전송
  - UAV 전력 할당 및 충전 결정


## Structure

```text
Lyapunov_RL_based_UAV_delivery/
├── Paper/                         # papers, notes, references
├── research/
│   └── Lyapunov_uav/
│       └── proposed/              # main implementation
│           ├── agent/             # PPO / RL agents
│           ├── env/               # RSU/UAV delivery environments
│           ├── config.py          # experiment configuration
│           ├── env.py             # environment wrapper / interface
│           ├── validators.py      # consistency / config validation
│           └── ... 
└── README.md
