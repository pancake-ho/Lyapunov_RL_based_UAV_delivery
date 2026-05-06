from __future__ import annotations

from typing import List

try:
    from proposed.config import BatteryConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
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
from .energy_model import compute_energy_summary
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
        consume_hover_when_idle: bool = False,
    ):
        self.config = config
        self.bandwidth = float(bandwidth)
        self.consume_hover_when_idle = bool(consume_hover_when_idle)

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

        action = BatteryAction(
            uav_idx=-1,
            mu_active=bool(mu_active),
            mode=mode,
            links=links,
        )
        validate_action_mode(action)

        # battery 하한 이하면 service 강제 차단
        if mode == UAVBatteryMode.SERVE and not can_serve(config=self.config, soc=self.soc):
            mode = UAVBatteryMode.OUTAGE
            links = []

        energy_info = compute_energy_summary(
            config=self.config,
            mode=mode,
            mu_active=bool(mu_active),
            links=links,
            consume_hover_when_idle=self.consume_hover_when_idle,
        )


        consumed_soc, charged_soc, next_soc, next_virtual_q = update_soc_virtual_q(
            config=self.config,
            soc=self.soc,
            consumed_energy=energy_info["total_energy"],
            charged_energy=energy_info["charge_energy"],
        )

        self.soc = float(next_soc)
        self.virtual_q = float(next_virtual_q)
        self.round_remaining_slots = max(0, self.round_remaining_slots - 1)

        outage = check_outage(self.soc)

        return BatteryStepInfo(
            hover_energy=float(energy_info["hover_energy"]),
            comm_energy=float(energy_info["comm_energy"]),
            total_consumed=float(energy_info["total_energy"]),
            charged_energy=float(energy_info["charge_energy"]),
            consumed_soc=float(consumed_soc),
            charged_soc=float(charged_soc),
            soc_before=float(soc_before),
            soc_after=float(self.soc),
            virtual_before=float(virtual_before),
            virtual_after=float(self.virtual_q),
            outage=bool(outage),
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
