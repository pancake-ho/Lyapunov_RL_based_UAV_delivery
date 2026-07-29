from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

@dataclass
class ChannelConfig:
    """
    Channel 및 PHY parameter config 클래스
    """
    # distance uniform sampling
    # USER와 RSU/UAV 사이 거리 (UAV는 고도 추가)
    distance: float = 15.0
    min_distance: float = 1.0 

    distance_min: float = 5.0
    distance_max: float = 25.0
    distance_sampling: str = "fixed" # 일단 not uniform

    # common params
    seed: int = 42
    bandwidth: float = 1e6

    # RSU Channel Params
    gamma_db: float = 35.0
    inr_db: float = 5.0
    sigma_db: float = 4.0
    mu_db: float = 0.0
    beta: float = 2.0
    
    # UAV LoS Channel Params
    altitude: float = 20.0
    beta_zero: float = 2e-9
    noise_power: float = 1e-13
    capacity_gap: float = 1.0

    def __post_init__(self) -> None:
        self.distance = float(self.distance)
        self.min_distance = float(self.min_distance)
        self.distance_min = float(self.distance_min)
        self.distance_max = float(self.distance_max)
        self.distance_sampling = str(self.distance_sampling).lower().strip()

        self.bandwidth = float(self.bandwidth)
        self.gamma_db = float(self.gamma_db)
        self.inr_db = float(self.inr_db)
        self.sigma_db = float(self.sigma_db)
        self.mu_db = float(self.mu_db)
        self.beta = float(self.beta)

        self.altitude = float(self.altitude)
        self.beta_zero = float(self.beta_zero)
        self.noise_power = float(self.noise_power)
        self.capacity_gap = float(self.capacity_gap)

        if self.distance < 0.0:
            raise ValueError(f"distance는 0 이상이어야 합니다. 현재 값: {self.distance}")
        if self.min_distance <= 0.0:
            raise ValueError(f"min_distance는 양수여야 합니다. 현재 값: {self.min_distance}")
        if self.distance_min < 0.0:
            raise ValueError(f"distance_min은 0 이상이어야 합니다. 현재 값: {self.distance_min}")
        if self.distance_max < self.distance_min:
            raise ValueError(
                f"distance_max는 distance_min 이상이어야 합니다. "
                f"현재 distance_min={self.distance_min}, distance_max={self.distance_max}"
            )
        if self.distance_sampling not in {"fixed", "uniform"}:
            raise ValueError(
                "distance_sampling은 'fixed' 또는 'uniform'이어야 합니다.  "
                f"현재 값={self.distance_sampling}"
            )
        
        if self.bandwidth <= 0.0:
            raise ValueError(f"bandwidth는 양수여야 합니다. 현재 값: {self.bandwidth}")
        if self.sigma_db < 0.0:
            raise ValueError(f"sigma_db는 0 이상이어야 합니다. 현재 값: {self.sigma_db}")
        if self.beta <= 0.0:
            raise ValueError(f"beta는 양수여야 합니다. 현재 값: {self.beta}")

        if self.altitude < 0.0:
            raise ValueError(f"altitude는 0 이상이어야 합니다. 현재 값: {self.altitude}")
        if self.beta_zero <= 0.0:
            raise ValueError(f"beta_zero는 양수여야 합니다. 현재 값: {self.beta_zero}")
        if self.noise_power <= 0.0:
            raise ValueError(f"noise_power는 양수여야 합니다. 현재 값: {self.noise_power}")
        if self.capacity_gap <= 0.0:
            raise ValueError(f"capacity_gap은 양수여야 합니다. 현재 값: {self.capacity_gap}")


@dataclass
class BatteryConfig:
    # SoC Actual queue
    e_max: int = 100
    e_init: int = 100
    e_min: float = 20.0

    # hovering 에너지 모델
    # e_hover(t) = (p_0 + p_i) * slot_duration
    p_0: float = 580.65 # blade profile power [W]
    p_i: float = 790.67 # induced power [W]

    tx_energy_coeff: float = 100.0

    # 충전 모델
    charging_rate: float = 5000.0 # Charging power [W]
    eta_c: float = 1.0
    enable_charging: bool = True
    allow_charge: bool = True

    # time slot
    slot_duration: float = 1.0
    target_service_slots_per_round: int = 5

    # SoC conversion term
    # None으로 설정되면, SoC 단위 사용 X
    # SAC/DDPG도 추가적으로 고려
    battery_capacity_joule: float = 5_000_000.0 
    energy_to_soc_factor: Optional[float] = None

    # 최대 통신 power bound
    min_tx_power: float = 1e-5
    max_tx_power: float = 1.0

    def __post_init__(self) -> None:
        if self.e_max <= 0:
            raise ValueError("e_max는 양수여야 합니다.")
        if self.slot_duration <= 0.0:
            raise ValueError("slot_duration은 양수여야 합니다.")
        if self.target_service_slots_per_round <= 0:
            raise ValueError("target_service_slots_per_round는 양수여야 합니다.")
        if self.p_0 < 0.0 or self.p_i < 0.0:
            raise ValueError(
                f"p_0와 p_i는 0 이상이어야 합니다. "
                f"현재 p_0={self.p_0}, p_i={self.p_i}"
            )
        if self.tx_energy_coeff <= 0.0:
            raise ValueError("tx_energy_coeff는 양수여야 합니다.")
        if self.charging_rate < 0.0:
            raise ValueError("charging_rate는 0 이상이어야 합니다.")
        if self.eta_c <= 0.0:
            raise ValueError("eta_c는 양수여야 합니다.")
        
        if self.min_tx_power <= 0.0:
            raise ValueError("min_tx_power는 양수여야 합니다.")
        if self.max_tx_power <= self.min_tx_power:
            raise ValueError(
                "max_tx_power는 min_tx_power보다 커야 합니다."
            )
        if self.battery_capacity_joule <= 0.0:
            raise ValueError("battery_capacity_joule은 양수여야 합니다.")

        self.e_init = float(min(max(float(self.e_init), 0.0), float(self.e_max)))
        self.e_min = float(min(max(float(self.e_min), 0.0), float(self.e_max)))

        if self.energy_to_soc_factor is None:
            self.energy_to_soc_factor = 100.0 / float(self.battery_capacity_joule)
        else:
            self.energy_to_soc_factor = float(self.energy_to_soc_factor)
            if self.energy_to_soc_factor <= 0.0:
                raise ValueError("energy_to_soc_factor는 양수여야 합니다.")

@dataclass
class EnvConfig:
    # 시스템 설정
    # user/rsu/uav 수 수정
    num_user: int = 20
    num_rsu: int = 10
    num_uav: int = 10
    uav_user_cap: int = 2
    
    slow_T: int = 3600
    episode_slots: Optional[int] = None
    N0: int = 3

    # 비디오 및 캐싱
    num_video: int = 3
    rsu_caching: int = 2
    layer: int = 4
    chunk: int = 9
    rsu_capacity: int = 3
    zipf_alpha: float = 1.1

    # FSMC mobility
    # user는 매 slot 확률 p로 왼쪽 region으로 이동
    # region 0에서 이동이 발생하면 오른쪽 끝 region으로 다른 user 재진입
    move_prob: float = 1e-4
    region_len: float = 50.0

    # Channel
    # HRL video delivery 논문은 user radius R=50 m를 사용.
    # 여기의 distance는 fallback/default distance일 뿐이며,
    # 실제 env에서는 rsu_user_distance/uav_user_distance를 meter 단위로 넘기는 것이 원칙.
    rsu_channel: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            distance=15.0,
            min_distance=1.0,
            distance_min=5.0,
            distance_max=25.0,
            distance_sampling="fixed",
            bandwidth=1e6,
            gamma_db=25.0,
            inr_db=5.0,
            sigma_db=4.0,
            beta=2.0,
            mu_db=0.0,
            seed=42,
        )
    )
    uav_channel: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(
            distance=15.0,
            min_distance=1.0,
            distance_min=5.0,
            distance_max=25.0,
            distance_sampling="fixed",
            bandwidth=1e6,
            altitude=20.0,
            beta_zero=2e-9,
            noise_power=1e-13,
            capacity_gap=1.0,
            seed=42,
        )
    )

    # 배터리
    battery: BatteryConfig = field(default_factory=BatteryConfig)

    # Queue / Playback model
    init_queue: float = 20.0
    playback_rate: float = 1.0 # queue update 산식의 b에 대응
    max_queue: float = 100.0 # Q_bar에 대응

    # 각 layer에 대한 quality 가중치
    quality_weights: Tuple[float, ...] = (34.0, 36.64, 39.11, 41.64)
    chunk_size_bits: Tuple[float, ...] = (
        2.621e3,
        5.073e3,
        10.658e3,
        26.496e3,
    )

    # scaled Lyapunov queue coefficients
    # alpha_Z = 1.0 설정
    # alpha_B = 30.0 설정
    alpha_Z: float = 1.0
    alpha_B: float = 30.0

    # slow-timescale decision에 반영
    # 추후 scale에 따라 수정 필요
    uav_hiring_cost: float = 5000.0
    hire_weight: float = 1.0

    # Lyapunov trade-off parameter
    V: float = 10.0

    # seed
    seed: int = 2026

    # quality objective
    quality_obj: str = "max_quality_degradation"

    def __post_init__(self) -> None:
        if self.num_user <= 0:
            raise ValueError("num_user는 양수여야 합니다.")
        if self.num_rsu <= 0:
            raise ValueError("num_rsu는 양수여야 합니다.")
        if self.num_uav <= 0:
            raise ValueError("num_uav는 양수여야 합니다.")
        if self.num_uav != self.num_rsu:
            raise ValueError(
                "현재 시나리오에서는 coverage region마다 최대 UAV 1대를 가정하므로 "
                f"num_uav와 num_rsu가 같아야 합니다. "
                f"현재 num_uav={self.num_uav}, num_rsu={self.num_rsu}"
            )
        if self.uav_user_cap <= 0:
            raise ValueError("uav_user_cap은 양수여야 합니다.")
        if self.slow_T <= 0:
            raise ValueError("slow_T는 양수여야 합니다.")
        if self.episode_slots is not None and self.episode_slots <= 0:
            raise ValueError("episode_slots는 None 또는 양수여야 합니다.")

        if self.num_video <= 0:
            raise ValueError("num_video는 양수여야 합니다.")
        if not (0 <= self.rsu_caching <= self.num_video):
            raise ValueError("rsu_caching은 [0, num_video] 범위여야 합니다.")
        if self.layer <= 0:
            raise ValueError("layer는 양수여야 합니다.")
        if self.chunk <= 0:
            raise ValueError("chunk는 양수여야 합니다.")
        if self.rsu_capacity <= 0:
            raise ValueError("rsu_capacity는 양수여야 합니다.")
        if self.zipf_alpha <= 0.0:
            raise ValueError("zipf_alpha는 양수여야 합니다.")

        if not (0.0 <= self.move_prob <= 1.0):
            raise ValueError("move_prob는 [0, 1] 범위여야 합니다.")
        if self.region_len <= 0.0:
            raise ValueError("region_len은 양수여야 합니다.")

        if self.init_queue < 0.0:
            raise ValueError("init_queue는 0 이상이어야 합니다.")
        if self.playback_rate < 0.0:
            raise ValueError("playback_rate는 0 이상이어야 합니다.")
        if self.max_queue <= 0.0:
            raise ValueError("max_queue는 양수여야 합니다.")

        if len(self.quality_weights) != self.layer:
            raise ValueError(
                f"quality_weights 길이는 layer와 같아야 합니다. "
                f"len={len(self.quality_weights)}, layer={self.layer}"
            )
        if len(self.chunk_size_bits) != self.layer:
            raise ValueError(
                f"chunk_size_bits 길이는 layer와 같아야 합니다. "
                f"len={len(self.chunk_size_bits)}, layer={self.layer}"
            )
        if any(float(v) < 0.0 for v in self.quality_weights):
            raise ValueError("quality_weights는 모두 0 이상이어야 합니다.")
        if any(float(v) <= 0.0 for v in self.chunk_size_bits):
            raise ValueError("chunk_size_bits는 모두 양수여야 합니다.")
        
        if self.alpha_Z <= 0.0:
            raise ValueError("alpha_Z는 양수여야 합니다.")
        if self.alpha_B <= 0.0:
            raise ValueError("alpha_B는 양수여야 합니다.")

        if self.uav_hiring_cost < 0.0:
            raise ValueError("uav_hiring_cost는 0 이상이어야 합니다.")
        if self.hire_weight < 0.0:
            raise ValueError("hire_weight는 0 이상이어야 합니다.")
        if self.V < 0.0:
            raise ValueError("V는 0 이상이어야 합니다.")
        
        self.quality_obj = (
            str(self.quality_obj)
            .lower()
            .strip()
        )
        if self.quality_obj != "max_quality_degradation":
            raise ValueError(
                "현재 formulation 기준 "
                "'max_quality_degradation' obj만 지원합니다. "
                f"현재 값: {self.quality_obj}"
            )
        
    @property
    def P_bar(self) -> float:
        """
        최대 quality P_bar
        """
        return float(max(self.quality_weights))

    def reward_coefficients(self) -> Dict[str, float]:
        return {
            "alpha_Z": float(self.alpha_Z),
            "alpha_B": float(self.alpha_B),
            "V": float(self.V),
            "P_bar": float(self.P_bar),
            "hire_weight": float(self.hire_weight)
        }

    def as_dict(self) -> Dict:
        return asdict(self)