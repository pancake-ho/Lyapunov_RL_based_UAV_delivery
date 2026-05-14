from __future__ import annotations

from typing import Tuple

try:
    from proposed.config import BatteryConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import BatteryConfig


def energy_to_soc(
    config: BatteryConfig,
    energy: float,
) -> float:
    """
    energy 값을 SoC 단위로 변환함.
    """
    return max(0.0, float(energy)) * float(config.energy_to_soc_factor)


def soc_to_virtual_q(
    config: BatteryConfig,
    soc: float,
) -> float:
    """
    Battery actual SoC queue E_u(t)를 Virtual queue B_u(t)로 변환함.

    현재 시나리오 기준:
        B_u(t) = E_max - E_u(t)
    """
    soc_clipped = min(float(config.e_max), max(0.0, float(soc)))
    return float(config.e_max) - soc_clipped


def update_soc(
    config: BatteryConfig,
    soc: float,
    consumed_energy: float,
    charged_energy: float,
) -> Tuple[float, float, float]:
    """
    Battery SoC queue update 산식.

    산식 대응:
        E_u(t+1) = clip(E_u(t) - eta_u e_u(t) + eta_u e_c(t), 0, E_max)
    """
    soc_now = min(float(config.e_max), max(0.0, float(soc)))
    consumed_soc = energy_to_soc(config=config, energy=consumed_energy)
    charged_soc = energy_to_soc(config=config, energy=charged_energy)
    
    next_soc = soc_now - consumed_soc + charged_soc
    next_soc = min(float(config.e_max), max(0.0, next_soc))

    return float(consumed_soc), float(charged_soc), float(next_soc)


def update_soc_virtual_q(
    config: BatteryConfig,
    soc: float,
    consumed_energy: float,
    charged_energy: float,
) -> Tuple[float, float, float, float]:
    """
    actual SoC queue와 battery virtual queue를 함께 update함.

    현재 시나리오 기준 구현은 항상 actual SoC queue를 먼저 update하고,
    그 결과로부터
    
        B_u(t+1) = E_max - E_u(t+1)
    
    를 재계산함.
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
    soc: float,
    consumed_soc: float = 0.0,
    soc_before: float | None = None,
) -> bool:
    """
    UAV battery outage 여부를 판단함.

    현재 시나리오 기준 다음과 같은 outage 기준을 사용:
        1) update 이후 soc <= 0
        2) slot에서 요구된 consumed_soc가 soc_before보다 커서 실제로는 service feasibility가 깨진 경우 (안전장치)
    """
    if float(soc) <= 0.0:
        return True
    
    if soc_before is not None:
        if float(consumed_soc) > float(soc_before) + 1e-9:
            return True
        
    return False