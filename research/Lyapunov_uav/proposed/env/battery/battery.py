from dataclasses import dataclass
from typing import Dict

from config import BatteryConfig

@dataclass
class BatteryStepInfo:
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

class UAVBattery:
    """
    UAV 1대의 배터리 관련 클래스. 모든 UAV는 같은 배터리 종류를 공유한다고 가정.
    """
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.soc = float(config.e_init) # actual queue
        self.virtual_q = 0.0 # battery usage virtual queue 
        self.round_start_soc = self.soc
        self.round_horizon = max(1, int(config.target_service_slots_per_round))

    def reset_episode(self) -> None:
        """
        에피소드마다 배터리 SoC를 초기화하는 함수
        """
        self.soc = float(self.config.e_init)
        self.virtual_q = 0.0
        self.round_start_soc = self.soc
        self.round_horizon = max(1, int(self.config.target_service_slots_per_round))
    
    def energy_to_soc(self, energy: float) -> float:
        """
        일반적인 에너지 단위(Wh 등)에서 SoC로 변환하는 함수
        """
        return float(energy) * float(self.config.energy_to_soc_factor)
    
    def hover_energy(self) -> float:
        """
        UAV가 hovering할 때 소모하는 에너지 수식을 구현하는 함수
        """
        return (self.config.p_0 + self.config.p_i) * self.config.slot_duration
    
    def comm_energy(self, tx_power: float, is_serving: bool) -> float:
        """
        UAV가 user에게 video delivery 작업을 수행할 때 소모하는 에너지 수식을 구현하는 함수
        """
        # delivery를 수행하고 있지 않으면 comm energy를 소비하지 않음
        if not is_serving:
            return 0.0
        return self.config.tx_energy_coeff * max(0.0, tx_power) * self.config.slot_duration
    
    def total_consumption(self, tx_power: float, delivered_chunks: int, is_serving: bool) -> Dict[str, float]:
        """
        UAV가 소모하는 총 에너지 수식을 구현하는 함수로
        hovering 에너지와 comm 에너지의 합으로 정의됨
        """
        hover_e = self.hover_energy() if is_serving else 0.0
        comm_e = self.comm_energy(tx_power=tx_power, is_serving=is_serving)
        total_e = hover_e + comm_e

        return {
            "hover_energy": hover_e,
            "comm_energy": comm_e,
            "total_energy": total_e,
        }
    
    def charge_energy(self, do_charge: bool) -> float:
        """
        UAV가 충전할 때 증가하는 에너지 수식을 구현하는 함수로
        각 충전소들은 일정한 에너지 공급량을 가지고 있고, 충전량은 시간에 비례함
        """
        if not do_charge or not self.config.enable_charging:
            return 0.0
        return self.config.eta_c * self.config.charging_rate * self.config.slot_duration
    
    def start_round(self, round_horizon: int, reset_virtual_queue: bool = True) -> None:
        """
        라운드마다 UAV를 처음 고용하는 경우, UAV의 배터리는 풀 충전된 상태에서 시작. 이를 구현하는 함수
        """
        self.round_start_soc = float(self.soc)
        self.round_horizon = max(1, int(round_horizon))
        if reset_virtual_queue:
            self.virtual_q = 0.0
    
    def step(self, tx_power: float, delivered_chunks: int, is_serving: bool, do_charge: bool) -> BatteryStepInfo:
        """
        매 time-step마다 UAV의 배터리 변화를 추적하는 함수
        """
        del delivered_chunks

        soc_before = self.soc
        virtual_q_before = self.virtual_q

        # 예외처리
        if is_serving and do_charge and not self.config.allow_charge:
            raise ValueError("UAV는 service와 charge를 동시에 수행할 수 없습니다.")
        
        # 충전 및 소모 동작
        use_info = self.total_consumption(tx_power=tx_power, is_serving=is_serving)
        consume_e = use_info["total_energy"]
        charge_e = self.charge_energy(do_charge)

        # SoC로 변환
        consume_soc = self.energy_to_soc(consume_e)
        charge_soc = self.energy_to_soc(charge_e)

        # SoC queue 업데이트
        next_soc = self.soc - consume_soc + charge_soc
        next_soc = max(0.0, min(self.config.e_max, next_soc))

        # virtual queue 업데이트
        allowed_use = max(self.round_start_soc - self.config.e_min, 0.0) / self.round_horizon
        next_virtual_q = max(0.0, self.virtual_q + consume_soc - charge_soc - allowed_use)

        self.soc = next_soc
        self.virtual_q = next_virtual_q
        outage = (self.soc <= self.config.e_min)

        return BatteryStepInfo(
            hover_energy=use_info["hover_energy"],
            comm_energy=use_info["comm_energy"],
            total_consumed=consume_e,
            charged_energy=charge_e,
            consumed_soc=consume_soc,
            charged_soc=charge_soc,
            soc_before=soc_before,
            soc_after=self.soc,
            virtual_before=virtual_q_before,
            virtual_after=self.virtual_q,
            outage=outage,
        )