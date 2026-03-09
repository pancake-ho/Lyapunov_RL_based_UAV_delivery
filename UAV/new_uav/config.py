from dataclasses import dataclass, field

@dataclass
class ChannelConfig:
    distance: float = 20.0
    bandwidth: float = 1e6
    gamma_db: float = 30.0
    sigma_db: int = 4
    beta: int = 2
    mu_db: int = 0

@dataclass
class BatteryConfig:
    # SoC 기준: [0, e_max]
    e_max: int = 100
    e_init: int = 100
    e_min: float = 1.0 # 하한

    # hovering 에너지 모델
    p_0: float = 100 # blade profile power
    p_i: float = 100 # induced power

    # 충전 모델
    charging_rate: float = 4.0
    eta_c: float = 1.0
    allow_charge: bool = False

    # time slot
    slot_duration: float = 1.0


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

    # indicator
    sch_indicator: float = 0.0
    hir_indicator: float = 0.0