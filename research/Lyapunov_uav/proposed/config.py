from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class ChannelConfig:
    distance: float = 20.0
    bandwidth: float = 1e6
    gamma_db: float = 25.0
    sigma_db: float = 4.0
    beta: float = 2.0
    mu_db: float = 0.0
    min_distance: float = 1.0
    seed: int = 42

@dataclass
class BatteryConfig:
    # SoC queue
    e_max: int = 100
    e_init: int = 100
    e_min: float = 10.0 # 하한

    # hovering 에너지 모델
    # e_hover(t) = (p_0 + p_i) * slot_duration
    p_0: float = 80.0 # blade profile power [W]
    p_i: float = 70.0 # induced power [W]

    tx_energy_coeff: float = 1.0

    # 충전 모델
    charging_rate: float = 120.0 # Charging power [W]
    eta_c: float = 1.0
    enable_charging: bool = True
    allow_charge: bool = False

    # time slot
    slot_duration: float = 1.0
    target_service_slots_per_round: int = 5

    energy_to_soc_factor: float = 0.05 

    # 최대 통신 power bound
    max_tx_power: float = 10.0

@dataclass
class EnvConfig:
    # 시스템 설정
    num_user: int = 10
    num_rsu: int = 10
    num_uav: int = 10
    uav_user_cap: int = 2 # UAV 1대당 동시 서비스 가능 사용자 수
    slow_T: int = 5
    N0: int = 3

    # 비디오 및 캐싱
    num_video: int = 3
    rsu_caching: int = 2
    layer: int = 5
    chunk: int = 5
    rsu_capacity: int = 3
    mbs_capacity: int = 3
    mbs_delay: int = 2
    zipf_alpha: float = 1.1

    # 사용자 이동 패턴
    spawn_base: float = 0.10
    spawn_amp: float = 0.05
    spawn_period: float = 200.0
    depart_base: float = 0.05
    depart_amp: float = 0.02
    depart_period: float = 300.0

    # 채널
    rsu_channel: ChannelConfig = field(default_factory=lambda: ChannelConfig(distance=0.6))
    uav_channel: ChannelConfig = field(default_factory=lambda: ChannelConfig(distance=0.4))

    # 배터리
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    # Queue / Playback model
    init_queue: float = 0.0
    playback_rate: float = 1.0
    max_queue: float = 100.0 # 임시 설정

    # delivery (비트 당 청크 사이즈 정의)
    base_chunk_size_bits: float = 2e5

    # reward 계수
    stall_penalty: float = 4.0
    battery_virtual_penalty: float = 0.2
    outage_penalty: float = 5.0

    # 각 layer에 대한 quality 가중치
    quality_weights: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # indicator
    sch_indicator: float = 0.0
    hir_indicator: float = 0.0

    # seed
    seed: int = 2026