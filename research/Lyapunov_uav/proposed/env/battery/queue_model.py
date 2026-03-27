from typing import Tuple

from config import BatteryConfig


def energy_to_soc(
    config: BatteryConfig,
    energy: float,
) -> float:
    """
    일반적인 에너지 단위(Wh 등)에서 SoC(%)로 변환하는 함수
    """
    return max(0.0, float(energy)) * float(config.energy_to_soc_factor)


def soc_to_virtual_q(
    config: BatteryConfig,
    soc: float,
) -> float:
    """
    Battery(SoC) Queue를 Virtual Queue로 변환하는 함수
    """
    return min(float(config.e_max), max(0.0, float(config.e_max) - float(soc)))


def update_soc(
    config: BatteryConfig,
    soc: float,
    consumed_energy: float,
    charged_energy: float,
) -> Tuple[float, float, float]:
    """
    actual queue E_u(t+1)에 대응하는 함수로,
    consumed_soc, charged_soc, next_soc를 반환
    """
    consumed_soc = energy_to_soc(config=config, energy=consumed_energy)
    charged_soc = energy_to_soc(config=config, energy=charged_energy)
    
    next_soc = max(0.0, float(soc) - consumed_soc) + charged_soc
    next_soc = min(float(config.e_max), next_soc)

    return float(consumed_soc), float(charged_soc), float(next_soc)


def update_soc_virtual_q(
    config: BatteryConfig,
    soc: float,
    consumed_energy: float,
    charged_energy: float,
) -> Tuple[float, float, float, float]:
    """
    virtual queue B_u(t+1)에 대응하는 함수
    """
    consumed_soc, charged_soc, next_soc = update_soc(
        config=config,
        soc=soc,
        consumed_energy=consumed_energy,
        charged_energy=charged_energy,
    )
    next_virtual_q = soc_to_virtual_q(config=config, soc=next_soc)
    return consumed_soc, charged_soc, next_soc, next_virtual_q


def check_outage(
    config: BatteryConfig,
    soc: float
) -> bool:
    """
    현재 soc queue가 최솟값을 벗어났는 지 확인하는 함수
    """
    return float(soc) <= float(config.e_min)