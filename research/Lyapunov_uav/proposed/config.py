from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class ChannelConfig:
    """
    Channel 및 PHY parameter config 클래스
    """
    distance: float = 20.0
    bandwidth: float = 1e6
    gamma_db: float = 25.0
    sigma_db: float = 4.0
    beta: float = 2.0
    mu_db: float = 0.0
    min_distance: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.bandwidth <= 0.0:
            raise ValueError("bandwidth는 양수 값을 가져야 합니다.")
        if self.min_distance <= 0.0:
            raise ValueError("min_distance는 양수 값을 가져야 합니다.")
        if self.distance < self.min_distance:
            self.distance = float(self.min_distance)

@dataclass
class BatteryConfig:
    # SoC Actual queue
    e_max: int = 100
    e_init: int = 100
    e_min: float = 10.0

    # hovering 에너지 모델
    # e_hover(t) = (p_0 + p_i) * slot_duration
    p_0: float = 80.0 # blade profile power [W]
    p_i: float = 70.0 # induced power [W]

    tx_energy_coeff: float = 1.0

    # 충전 모델
    charging_rate: float = 120.0 # Charging power [W]
    eta_c: float = 1.0
    enable_charging: bool = True
    allow_charge: bool = True

    # time slot
    slot_duration: float = 1.0
    target_service_slots_per_round: int = 5

    # SoC conversion term
    # None으로 설정되면, SoC 단위 사용 X
    energy_to_soc_factor: Optional[float] = 0.05

    # 최대 통신 power bound
    max_tx_power: float = 10.0

    def __post_init__(self) -> None:
        if self.e_max <= 0:
            raise ValueError("e_max는 양수 값을 가져야 합니다.")
        if self.slot_duration <= 0.0:
            raise ValueError("slot_duration은 양수 값을 가져야 합니다.")
        if self.target_service_slots_per_round <= 0:
            raise ValueError("target_service_slots_per_round는 양수 값을 가져야 합니다.")
        if self.p_0 < 0.0 or self.p_i < 0.0:
            raise ValueError(f"p_0와 p_i는 앙수 값을 가져야 합니다. 현재 두 값은 각각 {self.p_0}, {self.p_i}입니다.")
        if self.tx_energy_coeff <= 0.0:
            raise ValueError("tx_energy_coeff는 양수 값을 가져야 합니다.")
        if self.charging_rate < 0.0:
            raise ValueError("charging_rate는 0 이상의 값을 가져야 합니다.")
        if self.eta_c <= 0.0:
            raise ValueError("eta_c는 양수 값을 가져야 합니다.")
        if self.max_tx_power <= 0.0:
            raise ValueError("max_tx_power는 양수 값을 가져야 합니다.")
        
        self.e_init = float(min(max(self.e_init, 0.0), float(self.e_max)))
        self.e_min = float(min(max(self.e_min, 0.0), float(self.e_max)))

        self.energy_to_soc_factor = 100.0 / float(self.e_max)


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
    playback_rate: float = 1.0 # queue update 산식의 b에 대응
    max_queue: float = 100.0 # Q_bar에 대응

    # delivery (비트 당 청크 사이즈 정의)
    base_chunk_size_bits: float = 2e5

    # reward 계수
    # 아직 최적화 산식이 완료되지 않았으므로, 주석 처리.
    # stall_penalty: float = 4.0
    # battery_virtual_penalty: float = 0.2
    # outage_penalty: float = 5.0

    # 각 layer에 대한 quality 가중치
    quality_weights: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # indicator
    sch_indicator: float = 0.0
    hir_indicator: float = 0.0

    # seed
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.num_user <= 0:
            raise ValueError("num_user는 양수 값을 가져야 합니다.")
        if self.num_rsu <= 0:
            raise ValueError("num_rsu는 양수 값을 가져야 합니다.")
        if self.num_uav <= 0:
            raise ValueError("num_uav는 양수 값을 가져야 합니다.")
        if self.uav_user_cap:
            raise ValueError("uav_user_cap은 양수 값을 가져야 합니다.")
        if self.slow_T <= 0:
            raise ValueError("slow_T는 양수 값을 가져야 합니다.")
        if self.N0 < 0:
            raise ValueError("N0는 0 이상의 값을 가져야 합니다.")
        
        if self.num_video <= 0:
            raise ValueError("num_video는 양수 값을 가져야 합니다.")
        if not (0 <= self.rsu_caching <= self.num_video):
            raise ValueError(f"rsu_caching은 [0, num_video] 범위 내의 값을 가져야 합니다. \
                             \n현재 num_video 값은 {self.num_video}, rsu_caching 값은 {self.rsu_caching}입니다.")
        if self.layer <= 0 or self.chunk <= 0:
            raise ValueError(f"layer와 chunk는 모두 양수 값을 가져야 합니다. 현재 두 값은 각각 {self.layer}, {self.chunk}입니다.")
        if self.rsu_capacity <= 0 or self.mbs_capacity <= 0:
            raise ValueError(f"rsu_capacity와 mbs_capacity는 모두 양수 값을 가져야 합니다. \
                             현재 두 값은 각각 {self.rsu_capacity}, {self.mbs_capacity}입니다.")
        if self.mbs_delay < 0:
            raise ValueError("mbs_delay는 0 이상의 값을 가져야 합니다.")
        if self.zipf_alpha <= 0.0:
            raise ValueError("zipf_alpha는 양수 값을 가져야 합니다.")
        
        if not (0.0 <= self.spawn_base <= 1.0):
            raise ValueError("spawn_base는 [0, 1] 범위 내의 값을 가져야 합니다.")
        if not (0.0 <= self.depart_base <= 1.0):
            raise ValueError("depart_base는 [0, 1] 범위 내의 값을 가져야 합니다.")
        if self.spawn_amp < 0.0 or self.depart_amp < 0.0:
            raise ValueError(f"spawn_amp와 depart_map는 모두 양수 값을 가져야 합니다. \
                             현재 두 값은 각각 {self.spawn_amp}, {self.depart_amp}입니다.")
        if self.spawn_period < 0.0 or self.depart_period < 0.0:
            raise ValueError(f"spawn_period와 depart_period는 모두 양수 값을 가져야 합니다. \
                             현재 두 값은 각각 {self.spawn_period}, {self.depart_period}입니다.")
        
        if self.init_queue < 0.0 or self.max_queue <= 0.0:
            raise ValueError("init_queue는 0 이상의 값을, max_queue는 양수 값을 가져야 합니다.")
        if self.playback_rate < 0.0:
            raise ValueError("playback_rate는 0 이상의 값을 가져야 합니다.")
        if self.base_chunk_size_bits <= 0.0:
            raise ValueError("base_chunk_size_bits는 양수 값을 가져야 합니다.")
    
        if len(self.quality_weights) != self.layer:
            raise ValueError(f"quality_weights의 len {len(self.quality_weights)}는 layer와 같아야 합니다.")
        
        # 하나의 UAV는 coverage region(RSU) 당 한 대 고용될 수 있음
        if self.num_uav != self.num_rsu:
            raise ValueError(f"coverage region 당 하나의 UAV 고용을 가정합니다. 현재는 \
                             NUM_UAV: {self.num_uav}, NUM_RSU: {self.num_rsu}입니다.")