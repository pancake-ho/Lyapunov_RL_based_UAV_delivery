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
    UAV 1대의 battery state 관리 클래스.

    현재 시나리오 기준:
        - battery actual queue는 SoC E_u(t)로 관리함.
        - battery virtual queue는 B_u(t) = E_max - E_u(t)로 관리함.
        - UAV의 이동 에너지는 무시하고, hovering 및 communication/charging energy만 반영함.
    """
    def __init__(
        self,
        config: BatteryConfig,
        bandwidth: float,
        consume_hover_when_idle: bool = True,
    ):
        self.config = config
        self.bandwidth = float(bandwidth)
        self.consume_hover_when_idle = bool(consume_hover_when_idle)

        if self.bandwidth <= 0.0:
            raise ValueError(f"bandwidth는 양수여야 합니다. 현재 값: {self.bandwidth}")
        
        self.soc = float(config.e_init)
        self.virtual_q = soc_to_virtual_q(config=self.config, soc=self.soc)

        self.round_start_soc = float(self.soc)
        self.round_total_slots = max(1, int(config.target_service_slots_per_round))
        self.round_remaining_slots = self.round_total_slots

    def reset_episode(self) -> None:
        """
        episode 시작 시 battery state를 초기화함.
        """
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
        round 시작 시점 battery 기준점을 저장함. (round_horizon은 slow_T와 연결되는 값)
        """
        self.round_start_soc = float(self.soc)
        self.round_total_slots = max(1, int(round_horizon))
        self.round_remaining_slots = self.round_total_slots

    def get_state(self) -> BatteryState:
        """
        현재 battery state를 dataclass 형태로 반환함.
        """
        return BatteryState(
            soc=float(self.soc),
            virtual_q=float(self.virtual_q),
            round_start_soc=float(self.round_start_soc),
            round_total_slots=int(self.round_total_slots),
            round_remaining_slots=int(self.round_remaining_slots),
        )
    
    @staticmethod
    def _normalize_mode(mode: UAVBatteryMode | str) -> UAVBatteryMode:
        """
        str 또는 enum으로 들어온 mode를 UAVBatteryMode로 정규화함.
        """
        return UAVBatteryMode(mode)

    def step(
        self,
        mu_active: bool,
        links: List[CommLinkInput],
        mode: UAVBatteryMode | str,
    ) -> BatteryStepInfo:
        """
        한 slot 동안 battery transition을 수행함.

        현재 시나리오 기준 처리 순서:
            1) mode/link 정규화
            2) 고용되지 않은 UAV는 강제 IDLE 처리
            3) SERVE mode이지만 active link가 없다면 IDLE 처리
            4) e_min 이하이면 SERVE를 OUTAGE로 차단
            5) energy 계산
            6) SoC 및 virtual queue update
            7) outage 여부 기록
        """
        soc_before = float(self.soc)
        virtual_before = float(self.virtual_q)

        mu_active = bool(mu_active)
        mode = self._normalize_mode(mode)

        raw_links = [] if links is None else links
        validated_links = validate_links(raw_links)

        # 고용되지 않은 UAV는 service/charging을 수행하지 않음
        if not mu_active:
            mode = UAVBattery.IDLE
            validate_links = []
        
        # SERVE mode인데 실제 active link가 없으면 IDLE로 처리함
        if mode == UAVBatteryMode.SERVE and len(validated_links) == 0:
            mode = UAVBatteryMode.IDLE

        # battery 하한 이하면 service를 강제로 차단함
        if mode == UAVBatteryMode.SERVE and not can_serve(config=self.config, soc=self.soc):
            mode = UAVBatteryMode.OUTAGE
            validated_links = []

        action = BatteryAction(
            uav_idx=-1,
            mu_active=mu_active,
            mode=mode,
            links=validate_links,
        )
        validate_action_mode(action)

        energy_info = compute_energy_summary(
            config=self.config,
            mode=mode,
            mu_active=mu_active,
            links=validate_links,
            bandwidth=self.bandwidth,
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
        self.round_remaining_slots = max(0, int(self.round_remaining_slots) - 1)

        outage = check_outage(
            soc=self.soc,
            consumed_soc=consumed_soc,
            soc_before=soc_before,
        )

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
        """
        BatteryAction dataclass를 받아 step을 수행함.
        """
        return self.step(
            mu_active=action.mu_active,
            links=action.links,
            mode=action.mode,
        )
