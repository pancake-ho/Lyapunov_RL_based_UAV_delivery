from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class ChannelConfig:
    """
    Channel 및 PHY parameter config 클래스
    """
    distance: float = 20.0 # random (범위: 5 ~ 25 - uniform)
    bandwidth: float = 1e6

    gamma_db: float = 35.0
    inr_db: float = 5.0
    sigma_db: float = 4.0
    mu_db: float = 0.0

    beta: float = 2.0
    min_distance: float = 1.0
    seed: int = 42

    altitude: float = 100.0 # 감소
    beta_zero: float = 1e-3
    noise_power: float = 1e-13
    capacity_gap: float = 1.0

    def __post_init__(self) -> None:
        self.distance = float(self.distance)
        self.bandwidth = float(self.bandwidth)
        self.gamma_db = float(self.gamma_db)
        self.inr_db = float(self.inr_db)
        self.sigma_db = float(self.sigma_db)
        self.mu_db = float(self.mu_db)
        self.beta = float(self.beta)
        self.min_distance = float(self.min_distance)

        self.altitude = float(self.altitude)
        self.beta_zero = float(self.beta_zero)
        self.noise_power = float(self.noise_power)
        self.capacity_gap = float(self.capacity_gap)

        if self.distance < 0.0:
            raise ValueError(f"distance는 0 이상이어야 합니다. 현재 값: {self.distance}")
        if self.bandwidth <= 0.0:
            raise ValueError(f"bandwidth는 양수 값을 가져야 합니다. 현재 값: {self.bandwidth}")
        if self.sigma_db < 0.0:
            raise ValueError(f"sigma_db는 0 이상이어야 합니다. 현재 값: {self.sigma_db}")
        if self.beta <= 0.0:
            raise ValueError(f"beta는 양수 값을 가져야 합니다. 현재 값: {self.beta}")
        if self.min_distance <= 0.0:
            raise ValueError(f"min_distance는 양수 값을 가져야 합니다. 현재 값: {self.min_distance}")

        if self.altitude < 0.0:
            raise ValueError(f"altitude는 0 이상이어야 합니다. 현재 값: {self.altitude}")
        if self.beta_zero <= 0.0:
            raise ValueError(f"beta_zero는 양수 값을 가져야 합니다. 현재 값: {self.beta_zero}")
        if self.noise_power <= 0.0:
            raise ValueError(f"noise_power는 양수 값을 가져야 합니다. 현재 값: {self.noise_power}")
        if self.capacity_gap <= 0.0:
            raise ValueError(
                f"capacity_gap은 양수 값을 가져야 합니다. 현재 값: {self.capacity_gap}"
            )

@dataclass
class BatteryConfig:
    # SoC Actual queue
    e_max: int = 100
    e_init: int = 100
    e_min: float = 10.0

    # hovering 에너지 모델
    # e_hover(t) = (p_0 + p_i) * slot_duration
    p_0: float = 580.65 # blade profile power [W]
    p_i: float = 790.67 # induced power [W]

    tx_energy_coeff: float = 1.0

    # 충전 모델
    charging_rate: float = 1500.0 # Charging power [W]
    eta_c: float = 1.0
    enable_charging: bool = True
    allow_charge: bool = True

    # time slot
    slot_duration: float = 0.05
    target_service_slots_per_round: int = 5

    # SoC conversion term
    # None으로 설정되면, SoC 단위 사용 X
    # SAC/DDPG도 추가적으로 고려
    battery_capacity_joule: float = 100000.0 # 감소
    energy_to_soc_factor: Optional[float] = None

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
            raise ValueError(
                f"p_0와 p_i는 양수 값을 가져야 합니다. "
                f"현재 두 값은 각각 {self.p_0}, {self.p_i}입니다."
            )
        if self.tx_energy_coeff <= 0.0:
            raise ValueError("tx_energy_coeff는 양수 값을 가져야 합니다.")
        if self.charging_rate < 0.0:
            raise ValueError("charging_rate는 0 이상의 값을 가져야 합니다.")
        if self.eta_c <= 0.0:
            raise ValueError("eta_c는 양수 값을 가져야 합니다.")
        if self.max_tx_power <= 0.0:
            raise ValueError("max_tx_power는 양수 값을 가져야 합니다.")
        if self.battery_capacity_joule <= 0.0:
            raise ValueError("battery_capacity_joule은 양수 값을 가져야 합니다.")

        if self.energy_to_soc_factor is None:
            self.energy_to_soc_factor = 100.0 / float(self.battery_capacity_joule)
        else:
            self.energy_to_soc_factor = float(self.energy_to_soc_factor)

        self.e_init = float(min(max(self.e_init, 0.0), float(self.e_max)))
        self.e_min = float(min(max(self.e_min, 0.0), float(self.e_max)))


@dataclass
class EnvConfig:
    # 시스템 설정
    num_user: int = 100
    num_rsu: int = 63
    num_uav: int = 63
    uav_user_cap: int = 2
    slow_T: int = 25
    N0: int = 3

    # 비디오 및 캐싱
    num_video: int = 3
    rsu_caching: int = 2
    layer: int = 4
    chunk: int = 9
    rsu_capacity: int = 3
    zipf_alpha: float = 1.1

    # 사용자 이동 패턴
    spawn_base: float = 0.10
    spawn_amp: float = 0.05
    spawn_period: float = 200.0
    depart_base: float = 0.05
    depart_amp: float = 0.02
    depart_period: float = 300.0

    # Channel
    # HRL video delivery 논문은 user radius R=50 m를 사용.
    # 여기의 distance는 fallback/default distance일 뿐이며,
    # 실제 env에서는 rsu_user_distance/uav_user_distance를 meter 단위로 넘기는 것이 원칙.
    rsu_channel: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            distance=20.0,
            bandwidth=1e6,
            gamma_db=25.0,
            inr_db=5.0,
            sigma_db=4.0,
            beta=2.0,
            mu_db=0.0,
            min_distance=1.0,
            seed=42,
        )
    )
    uav_channel: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            distance=40.0,
            bandwidth=1e6,
            altitude=100.0,
            beta_zero=1e5,
            noise_power=1.0,
            capacity_gap=1.0,
            min_distance=1.0,
            seed=42,
        )
    )

    # 배터리
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    # Queue / Playback model
    init_queue: float = 0.0
    playback_rate: float = 1.0 # queue update 산식의 b에 대응
    max_queue: float = 100.0 # Q_bar에 대응

    # delivery (비트 당 청크 사이즈 정의)
    base_chunk_size_bits: float = 2e5

    # 각 layer에 대한 quality 가중치
    quality_weights: Tuple[float, ...] = (34.0, 36.64, 39.11, 41.64)
    chunk_size_bits: Tuple[float, ...] = (
        2.621e3,
        5.073e3,
        10.658e3,
        26.496e3,
    )

    # indicator
    sch_indicator: float = 0.0
    hir_indicator: float = 0.0

    # seed
    seed: int = 2026

    # User video virtual queue target
    # Z_n(t) = max_queue - Q_n(t)를 theta_z 근처로 유지하기 위한 perturbed target
    theta_z: Optional[Tuple[float, ...]] = None
    V: float = 10.0

    def __post_init__(self) -> None:
        if self.num_user <= 0:
            raise ValueError("num_user는 양수 값을 가져야 합니다.")
        if self.num_rsu <= 0:
            raise ValueError("num_rsu는 양수 값을 가져야 합니다.")
        if self.num_uav <= 0:
            raise ValueError("num_uav는 양수 값을 가져야 합니다.")
        if self.uav_user_cap <= 0:
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
    
        if self.theta_z is None:
            self.theta_z = tuple([0.7 * float(self.max_queue)] * self.num_user)
        elif len(self.theta_z) != self.num_user:
            raise ValueError(
                f"theta_z 길이는 num_user={self.num_user}와 같아야 합니다. "
                f"현재 len(theta_z)={len(self.theta_z)}입니다."
            )
        
        if len(self.chunk_size_bits) != self.layer:
            raise ValueError(
                f"chunk_size_bits의 len {len(self.chunk_size_bits)}는 "
                f"layer={self.layer}와 같아야 합니다."
            )
        if any(float(s) <= 0.0 for s in self.chunk_size_bits):
            raise ValueError("chunk_size_bits의 모든 값은 양수여야 합니다.")