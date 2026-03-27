from typing import List

from config import BatteryConfig
from .battery_types import (
    BatteryAction,
    BatteryState,
    BatteryStepInfo,
    CommLinkInput,
    UAVBatteryMode,
)
from .constraints import (
    can_serve,
    validate_links,
    validate_action_mode,
)
from .energy_model import (
    compute_charge_energy,
    compute_energy_summary,
)
from .queue_model import (
    check_outage,
    soc_to_virtual_q,
    update_soc_virtual_q,
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
        self.hover_only_when_serving = bool(hover_only_when_serving)

        self.soc = float(config.e_init)
        self.virtual_q = soc_to_virtual_q(config=self.config, soc=self.soc)
        self.round_start_soc = float(self.soc)
        self.round_total_slots = max(1, int(config.target_service_slots_per_round))
        self.round_remaining_slots = self.round_total_slots

    def reset_episode(self) -> None:
        self.soc = float(self.config.e_init)
        self.virtual_q = soc_to_virtual_q(config=self.config, soc=self.soc)
        self.round_start_soc = float(self.soc)
        self.round_horizon = max(1, int(self.config.target_service_slots_per_round))
        self.round_remaining_slots = self.round_total_slots

    def start_round(
        self,
        round_horizon: int,
    ) -> None:
        """
        round 시작 시점 battery 기준점 저장 함수
        """
        self.round_start_soc = float(self.soc)
        self.round_total_slots = max(1, int(round_horizon))
        self.round_remaining_slots = self.round_total_slots

    def get_state(self) -> BatteryState:
        return BatteryState(
            soc=float(self.soc),
            virtual_q=float(self.virtual_q),
            round_start_soc=float(self.round_start_soc),
            round_total_slots=int(self.round_total_slots),
            round_remaining_slots=int(self.round_remaining_slots),
        )

    def step(
        self,
        mu_active: bool,
        links: List[CommLinkInput],
        mode: UAVBatteryMode,
    ) -> BatteryStepInfo:
        soc_before = float(self.soc)
        virtual_before = float(self.virtual_q)

        links = validate_links(links)

        is_serving = any(
            bool(link.scheduled) and int(link.delivered_chunks) > 0
            for link in links
        )

        action = BatteryAction(
            uav_idx=-1,
            mu_active=bool(mu_active),
            mode=mode,
            links=links,
        )
        validate_action_mode(action)

        # battery 하한 이하면 service 강제 차단
        if not can_serve(config=self.config, soc=self.soc):
            if mode == UAVBatteryMode.SERVE:
                mode = UAVBatteryMode.OUTAGE
                links = []

        use_info = compute_energy_summary(
            config=self.config,
            mode=mode,
            mu_active=mu_active,
            links=links,
            hover_only_when_serving=self.hover_only_when_serving,
        )

        charge_e = compute_charge_energy(
            config=self.config,
            mu_active=mu_active,
            mode=mode,
        )

        consumed_soc, charged_soc, next_soc, next_virtual_q = update_soc_virtual_q(
            config=self.config,
            soc=self.soc,
            consumed_energy=use_info["total_energy"],
            charged_energy=charge_e,
        )

        self.soc = float(next_soc)
        self.virtual_q = float(next_virtual_q)
        self.round_remaining_slots = max(0, self.round_remaining_slots - 1)

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
            mode=action.mode,
        )