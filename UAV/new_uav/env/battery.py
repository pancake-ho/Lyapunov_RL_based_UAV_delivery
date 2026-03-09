from dataclasses import dataclass
from typing import Dict
from config import BatteryConfig

@dataclass
class BatteryStepInfo:
    hover_energy: float
    delivery_energy: float
    total_consumed: float
    charged_energy: float
    soc_before: float
    soc_after: float
    virtual_before: float
    virtual_after: float
    outage: bool

class UAVBattery:
    """
    임의의 UAV의 배터리 관련 클래스. 모든 UAV는 같은 배터리 종류를 공유한다고 가정.
    """
    def __init__(self, config: BatteryConfig):
        self.config = config
        self.soc = float(config.e_init) # actual queue
        self.virtual_q = 0.0 # battery usage virtual queue 
        
    def reset_episode(self):
        self.soc = float(self.config.e_init)
        self.virtual_q = 0.0
    
    def hover_energy(self) -> float:
        return (self.config.p_0 + self.config.p_i) * self.config.slot_duration
    
    def comm_energy(self, tx_power: float, delivered_chunks: int) -> float:
        return self.config.tx_energy_coeff * tx_power * delivered_chunks * self.config.slot_duration
    
    def total_consumption(self, tx_power: float, delivered_chunks: int, is_serving: bool) -> Dict[str, float]:
        hover_e = self.hover_energy() if is_serving else 0.0
        comm_e = self.comm_energy(tx_power, delivered_chunks) if is_serving else 0.0
        total_e = hover_e + comm_e

        return {
            "hover_energy": hover_e,
            "comm_energy": comm_e,
            "total_energy": total_e,
        }
    
    def charge_energy(self, do_charge: bool) -> float:
        if not do_charge:
            return 0.0
        return self.config.eta_c * self.config.charging_rate * self.config.slot_duration
    
    def step(self, tx_power: float, delivered_chunks: int, is_serving: bool, do_charge: bool) -> BatteryStepInfo:
        soc_before = self.soc
        virtual_q_before = self.virtual_q

        use_info = self.total_consumption(
            tx_power=tx_power,
            delivered_chunks=delivered_chunks,
            is_serving=is_serving,
        )
        consume_e = use_info["total_energy"]

        if is_serving and do_charge and not self.config.allow_charge:
            raise ValueError("UAV는 service와 charge를 동시에 수행할 수 없습니다.")
        
        charge_e = self.charge_energy(do_charge)

        # actual SoC queue update
        next_soc = self.soc - consume_e + charge_e
        next_soc = max(0.0, min(self.config.e_max, next_soc))

        # virtual queue update
        service_budget = self.config.target_service_slots_per_round
        allowed_use = self.round_start_soc / max(1, service_budget)
        next_virtual_q = max(0.0, self.virtual_q + consume_e - charge_e - allowed_use)

        self.soc = next_soc
        self.virtual_q = next_virtual_q

        outage = (self.soc <= self.config.e_min)

        return BatteryStepInfo(
            hover_energy=use_info["hover_energy"],
            delivery_energy=use_info["delivery_energy"],
            total_consumed=consume_e,
            charged_energy=charge_e,
            soc_before=soc_before,
            soc_after=self.soc,
            virtual_before=virtual_q_before,
            virtual_after=self.virtual_q,
            outage=outage,
        )