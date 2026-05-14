from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class UAVBatteryMode(str, Enum):
    """
    한 slot에 대한 UAV의 battery/service mode로
    현재 기준 다음과 같은 mode가 존재.

        IDLE:
            UAV가 고용되어 있지만 해당 slot에서 아무 동작도 하지 않는 상태. (단순 hovering)
            consume_hover_when_idle=True 이면 hovering energy를 소모함.
        
        SERVE:
            UAV가 고용되어 있고 해당 slot에서 user에게 delivery를 수행하는 상태.
            hovering energy + communication energy를 모두 소모함.
        
        CHARGE:
            UAV가 해당 slot에서 충전하는 상태.

        OUTAGE:
            UAV battery가 service 불가능한 상태.
    """
    IDLE = "idle"
    SERVE = "serve"
    CHARGE = "charge"
    OUTAGE = "outage"


@dataclass
class CommLinkInput:
    """
    UAV communication energy 계산을 위한 link-level 입력.

    이는 uav_delivery.py에서 active UAV-user link에 대해 생성하고,
    uav_battery.py에서 communication energy 계싼에 사용함.
    """
    scheduled: bool
    delivered_layers: int
    delivered_chunks: int
    payload_bits: float
    channel_gain: float
    noise_power: float

    tx_power: float | None = None
    user_idx: int = -1
    layer_idx: int = -1
    link_capacity_bps: float = 0.0
    tx_time: int = 0


@dataclass
class BatteryAction:
    """
    한 slot에 대한 UAV 하나의 Battery Action.
    이때 uav_charge는 policy가 직접 출력하는 action이 아니라,
    env로 들어오는 값으로 해석함.
    """
    uav_idx: int
    mu_active: bool # hiring
    mode: UAVBatteryMode
    links: List[CommLinkInput] = field(default_factory=list)


@dataclass
class BatteryState:
    """
    UAV Battery 내부 상태.

    round_start_soc:
        round 시작 시점에서의 SoC로, 한 round 내 battery feasibility 확인에 사용됨.
    
    round_total_slots:
        현재 round horizon.
    
    round_remaining_slots:
        현재 round에서 남은 slot 수
    """
    soc: float # actual q
    virtual_q: float # virtual q
    round_start_soc: float
    round_total_slots: int
    round_remaining_slots: int


@dataclass
class BatteryStepInfo:
    """
    한 time slot 이후 battery transition 결과.
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