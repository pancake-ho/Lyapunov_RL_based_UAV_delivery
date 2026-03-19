from dataclasses import dataclass, field
from typing import List

@dataclass
class CommLinkInput:
    """
    user n에 대한 UAV 통신 입력 클래스로
    "통신 시 UAV 배터리 소모량" 모델링을 위해 선언

    - scheduled: phi_un(t)
    - delivered_chunks: l_un(t)
    - chunk_size_bits: S(k_un(t))
    - channel_gain: g_un(t)
    - noise_power: sigma_un(t)^2
    """
    scheduled: bool
    delivered_chunks: int
    chunk_size_bits: float
    channel_gain: float
    noise_power: float

@dataclass
class BatteryAction:
    """
    fast-timescale(slot) 에서 battery 모듈이 받는 UAV 동작 입력 클래스
    """
    mu_active: bool # hiring
    do_charge: bool
    links: List[CommLinkInput] = field(default_factory=list)

@dataclass
class BatteryState:
    """
    UAV Battery 내부 상태 클래스
    """
    soc: float # actual q
    virtual_q: float # virtual q
    round_start_soc: float
    round_horizon: int

@dataclass
class BatteryStepInfo:
    """
    한 time step 이후 battery transition 결과 클래스
    """
    hover_energy: float
    comm_energy: float
    total_consumed: float
    charged_energy: float

    consumed_soc: float
    charged_soc: float

    soc_before: float
    soc_after: float

    virtual_before: float
    virtual_after: float
    
    outage: bool