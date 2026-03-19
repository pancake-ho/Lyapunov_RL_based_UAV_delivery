from typing import List

from config import BatteryConfig
from .types import CommLinkInput


def validate_service_charge(
    config: BatteryConfig,
    is_serving: bool,
    do_charge: bool,
) -> None:
    """
    UAV가 service가 charging을 동시 수행하고 있는 지 검증하는 함수
    """
    if is_serving and do_charge and not config.allow_charge:
        raise ValueError("UAV는 service와 charging을 동시 수행할 수 없습니다.")


def validate_links(
    links: List[CommLinkInput],
) -> List[CommLinkInput]:
    """
    step 전에, 이상한 값의 입력을 걸러내는 함수
    """
    validated = []
    for link in links:
        validated.append(
            CommLinkInput(
                scheduled=bool(link.scheduled),
                delivered_chunks=max(0, int(link.delivered_chunks)),
                chunk_size_bits=max(0.0, float(link.chunk_size_bits)),
                channel_gain=max(0.0, float(link.channel_gain)),
                noise_power=max(0.0, float(link.noise_power)),
            )
        )
    return validated


def can_serve(
    config: BatteryConfig,
    soc: float,
) -> bool:
    return float(soc) > float(config.e_min)