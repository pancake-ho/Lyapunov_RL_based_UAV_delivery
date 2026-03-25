from typing import Dict, List

from config import BatteryConfig
from .battery_types import CommLinkInput


def compute_hover_energy(
    config: BatteryConfig,
    is_hovering: bool,
) -> float:
    """
    UAV가 hovering할 때 소모하는 에너지 산식을 구현하는 함수로,
    e_hover(t) = (p_0 + p_i) * delta_t 와 같이 계산됨.
    """
    if not is_hovering:
        print("UAV는 hovering 상태가 아닙니다.")
        return 0.0
    
    return float(config.p_0 + config.p_i) * float(config.slot_duration)


def compute_comm_energy(
    config: BatteryConfig,
    bandwidth: float,
    links: List[CommLinkInput]
) -> float:
    """
    UAV가 user에게 video delivery 작업을 수행할 때 소모하는 에너지 산식을 구현하는 함수
    """
    delta_t = float(config.slot_duration)
    bw = float(bandwidth)

    total = 0.0
    for link in links:
        if not link.scheduled:
            print("UAV와 scheduling된 user가 없습니다.")
            continue
        if link.delivered_chunks <= 0:
            print("UAV에게 chunk를 전송받는 user가 없습니다.")
            continue
        if link.chunk_size_bits <= 0.0:
            print("UAV에게 받는 chunk size가 0 이하입니다.")
            continue
        if link.channel_gain <= 0.0:
            print("UAV와 user 사이의 channel gain이 0 이하입니다.")
            continue
        if link.noise_power <= 0.0:
            print("UAV와 user 사이의 noise power가 0 이하입니다.")
            continue

        exponent = (
            float(link.delivered_chunks) * float(link.chunk_size_bits)
        ) / (bw * delta_t)

        required_power = (
            float(link.noise_power) / float(link.channel_gain)
        ) * (2.0 ** exponent - 1.0)

        required_power = min(required_power, float(config.max_tx_power))

        communication_energy = required_power * delta_t * float(config.tx_energy_coeff)
        total += communication_energy

    return total


def compute_total_energy(
    mu_active: bool,
    hover_energy: float,
    comm_energy: float,
) -> float:
    """
    UAV가 소모하는 총 에너지 수식을 구현하는 함수로
    hovering 에너지와 comm 에너지의 합으로 정의됨
    """
    if not mu_active:
        print("UAV를 현재 고용하고 있지 않습니다.")
        return 0.0
    
    return float(hover_energy + comm_energy)


def compute_charge_energy(
    config: BatteryConfig,
    mu_active: bool,
    do_charge: bool,
) -> float:
    """
    UAV가 충전할 때 증가하는 에너지 수식을 구현하는 함수로
    각 충전소들은 일정한 에너지 공급량을 가지고 있고, 충전량은 시간에 비례함
    """
    if not config.enable_charging:
        print("UAV는 현재 충전이 불가합니다.")
        return 0.0
    if not mu_active:
        print("UAV를 현재 고용하고 있지 않습니다.")
        return 0.0
    if not do_charge:
        print("UAV는 현재 충전을 진행하고 있지 않습니다.")
        return 0.0

    return float(config.charging_rate) * float(config.slot_duration)


def compute_energy_summary(
    config: BatteryConfig,
    bandwidth: float,
    mu_active: bool,
    links: List[CommLinkInput],
    hover_only_when_serving: bool = True
) -> Dict[str, float]:
    """
    energy summary 구조 확립 함수
    """
    if not mu_active:
        print("UAV를 현재 고용하고 있지 않습니다.")
        return {
            "hover_energy": 0.0,
            "comm_energy": 0.0,
            "total_energy": 0.0,
        }
    
    is_serving = any(link.scheduled and link.delivered_chunks > 0 for link in links)

    if hover_only_when_serving:
        is_hovering = is_serving if hover_only_when_serving else (mu_active and not do_charge)
    else:
        is_hovering = True

    hover_e = compute_hover_energy(config=config, is_hovering=is_hovering)
    comm_e = compute_comm_energy(
        config=config,
        bandwidth=bandwidth,
        links=links,
    )
    total_e = compute_total_energy(
        config=config,
        mu_active=mu_active,
        hover_energy=hover_e,
        comm_energy=comm_e,
    )

    return {
        "hover_energy": hover_e,
        "comm_energy": comm_e,
        "total_energy": total_e,
    }