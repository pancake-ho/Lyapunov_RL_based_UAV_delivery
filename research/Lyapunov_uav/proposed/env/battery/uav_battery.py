from typing import List

from config import BatteryConfig
from .types import (
    BatteryAction,
    BatteryState,
    BatteryStepInfo,
    CommLinkInput,
)
from .constraints import (
    can_serve,
    validate_links,
    validate_service_charge,
)
from .energy_model import (
    compute_charge_energy,
    compute_energy_summary,
)
from .queue_model import (
    check_outage,
    update_soc,
    update_virtual_queue,
)


class UAVBattery:
    """
    UAV 1대의 battery state 관리 클래스
    """
    def __init__(
        self,
        config: BatteryConfig,
        bandwidth: float,
        hover_only_when_serving: bool = True,
    ):
        self.config = config
        self.bandwidth = bandwidth
        self.hover_only_when_seving = bool(hover_only_when_serving)

        self.soc = float(config.e_init)
        self.virtual_q = 0.0
        self.round_start_soc = float(self.soc)
        self.round_horizon = max(1, int(config.target_service_slots_per_round))
        
    def reset_episode(self) -> None:
        self.soc = float(self.config.e_init)
        self.virtual_q = 0.0
        self.round_start_soc = float(self.soc)
        self.round_horizon = max(
            1,
            int(self.config.target_service_slots_per_round),
        )

    def start_round(
        self,
        round_horizon: int,
        reset_virtual_queue: bool = True,
    ) -> None:
        """
        round 시작 시점 battery budget 기준점 저장 함수
        """
        self.round_start_soc = float(self.soc)
        self.round_horizon = max(1, int(round_horizon))

        if reset_virtual_queue:
            self.virtual_q = 0.0

    def get_state(self) -> BatteryState:
        return BatteryState(
            soc=float(self.soc),
            virtual_q=float(self.virtual_q),
            round_start_soc=float(self.round_start_soc),
            round_horizon=int(self.round_horizon),
        )

    def step(
        self,
        mu_active: bool,
        links: List[CommLinkInput],
        do_charge: bool,
    ) -> BatteryStepInfo:
        soc_before = float(self.soc)
        virtual_before = float(self.virtual_q)

        links = validate_links(links)
        is_serving = any(link.scheduled for link in links)

        validate_service_charge(
            config=self.config,
            is_serving=is_serving,
            do_charge=do_charge,
        )

        # battery 하한 이하면 service 강제 차단
        if not can_serve(config=self.config, soc=self.soc):
            links = []
            is_serving = False

        use_info = compute_energy_summary(
            config=self.config,
            bandwidth=self.bandwidth,
            mu_active=mu_active,
            links=links,
            hover_only_when_serving=self.hover_only_when_serving,
        )

        charge_e = compute_charge_energy(
            config=self.config,
            mu_active=mu_active,
            do_charge=do_charge,
        )

        consumed_soc, charged_soc, next_soc = update_soc(
            config=self.config,
            soc=self.soc,
            consumed_energy=use_info["total_energy"],
            charged_energy=charge_e,
        )

        next_virtual_q = update_virtual_queue(
            config=self.config,
            virtual_q=self.virtual_q,
            consumed_energy=use_info["total_energy"],
            charged_energy=charge_e,
        )

        self.soc = next_soc
        self.virtual_q = next_virtual_q

        outage = check_outage(config=self.config, soc=self.soc)

        return BatteryStepInfo(
            hover_energy=use_info["hover_energy"],
            comm_energy=use_info["comm_energy"],
            total_consumed=use_info["total_energy"],
            charged_energy=charge_e,
            consumed_soc=consumed_soc,
            charged_soc=charged_soc,
            soc_before=soc_before,
            soc_after=float(self.soc),
            virtual_before=virtual_before,
            virtual_after=float(self.virtual_q),
            outage=outage,
        )

    def step_with_action(
        self,
        action: BatteryAction,
    ) -> BatteryStepInfo:
        return self.step(
            mu_active=action.mu_active,
            links=action.links,
            do_charge=action.do_charge,
        )