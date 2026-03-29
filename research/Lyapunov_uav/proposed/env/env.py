from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from config import EnvConfig
from .action_types import EnvAction, ParsedAction, StepResult
from .validators import parse_action
from .channel import RSUChannelModel, UAVChannelModel
from .delivery.rsu_delivery import compute_rsu_delivery
from .delivery.uav_delivery import compute_uav_delivery
from .battery import UAVBattery
from .battery.battery_types import BatteryStepInfo, UAVBatteryMode
from .battery.energy_model import compute_energy_summary
from .battery.queue_model import check_outage, update_soc_virtual_q


class Env:
    """
    Modified Joint Lyapunov Optimization Scenario용 전체 환경 클래스

    Slow-timescale (round level):
        1) RSU scheduling y_mn(r)
        2) UAV hiring mu_m(r)
        3) UAV scheduling phi_un(r)

    Fast-timescale (slot level):
        1) RSU/UAV chunk, layer delivery
        2) UAV power allocation p_un(t)
        3) UAV charging/service mode I_u(t)
    """
    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)
        
        # 시스템 설정
        # user, uav, rsu 수 정의
        self.num_rsu = int(self.cfg.num_rsu)
        self.num_user = int(self.cfg.num_user)
        self.num_uav = int(self.cfg.num_uav)
        self.slow_T = int(self.cfg.slow_T)

        # 채널 객체
        self.rsu_channel = RSUChannelModel(self.cfg.rsu_channel)
        self.uav_channel = UAVChannelModel(self.cfg.uav_channel)

        # 배터리 객체
        self.batteries = [
            UAVBattery(
                config=self.cfg.battery,
                bandwidth=float(self.cfg.uav_channel.bandwidth),
                consume_hover_when_idle=False,
            )
            for _ in range(self.num_uav)
        ]

        # reward 산식은 확정되지 않았으므로 미선언
        
        # runtime state
        self.t = 0
        self.episode = 0
        self.round_idx = 0
        self.round_slot = 0

        # user video queue
        self.queue = np.zeros(self.num_user, dtype=np.float32)

        # slow-timescale decisions
        # rsu/uav의 스케줄링 및 uav 고용
        self.rsu_scheduling = np.zeros((self.num_rsu, self.num_user), dtype=np.int32)
        self.uav_hiring = np.zeros(self.num_uav, dtype=np.int32)
        self.uav_scheduling = np.zeros((self.num_uav, self.num_user), dtype=np.int32)

        # user/content 정보
        self.requested_content = np.zeros(self.num_user, dtype=np.int32)
        self.uav_cached_content = np.zeros(self.num_uav, dtype=np.int32)

        # UAV 상태
        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.int32)

        # round energy
        self.round_start_E = np.zeros(self.num_uav, dtype=np.float32)
    
    @property
    def E(self) -> np.ndarray:
        """
        UAV Actual SoC Queue
        """
        return np.array([b.soc for b in self.batteries], dtype=np.float32)
    
    @property
    def Y(self) -> np.ndarray:
        """
        UAV Virtual Soc Queue
        """
        return np.array([b.virtual_q for b in self.batteries], dtype=np.float32)
    
    @property
    def Z(self) -> np.ndarray:
        """
        User virtual video queue
        """
        return np.clip(
            float(self.cfg.max_queue) - self.queue,
            0.0,
            float(self.cfg.max_queue),
        ).astype(np.float32)
    
    def _sample_requested_content(self) -> np.ndarray:
        """
        사용자가 요청하는 content 샘플링하는 함수로,
        Zipf 기반으로 [0, num_video-1] 중 하나를 선택함.
        """
        video_ids = np.arange(1, self.cfg.num_video+1, dtype=np.float64)
        probs = 1.0 / np.power(video_ids, float(self.cfg.zipf_alpha))
        probs = probs / probs.sum()
        sampled = self.rng.choice(self.cfg.num_video, size=self.num_user, p=probs)
        return sampled.astype(np.int32)
    
    def _sample_uav_cached_content(self) -> np.ndarray:
        """
        UAV cache content 초기화하는 함수로,
        현재는 각 UAV가 하나의 content만 cache 가능하다고 보고 균등 샘플링 적용
        """
        sampled = self.rng.integers(
            low=0,
            high=self.cfg.num_video,
            size=self.num_uav,
            dtype=np.int32,
        )
        return sampled.astype(np.int32)
    
    def _start_new_round(self) -> None:
        """
        라운드 초기화 함수
        """
        self.round_idx = self.t // self.slow_T
        self.round_slot = 0
        self.round_start_E = self.E.copy()

        for battery in self.batteries:
            battery.start_round(round_horizon=self.slow_T)

    def _build_effective_action(self, parsed: ParsedAction) -> ParsedAction:
        """
        slow-timescale decision은 round 경계에서 갱신되고,
        나머지 slot에서는 이전의 round decision을 유지하도록 하는 effective action을 생성하는 함수
        """
        residual_users = (
            (self.rsu_scheduling.sum(axis=0) == 0).astype(np.int32)
        )

        return ParsedAction(
            rsu_scheduling=self.rsu_scheduling.copy(),
            uav_hiring=self.uav_hiring.copy(),
            uav_scheduling=(self.uav_scheduling * residual_users[None, :]).copy(),
            rsu_chunks=parsed.rsu_chunks.copy(),
            rsu_layers=parsed.rsu_layers.copy(),
            uav_chunks=(parsed.uav_chunks * residual_users[None, :]).copy(),
            uav_layers=(parsed.uav_layers * residual_users[None, :]).copy(),
            uav_power=(parsed.uav_power * residual_users[None, :].astype(np.float32)).copy(),
            uav_charge=parsed.uav_charge.copy(),
            playback=parsed.playback.copy(),
            rsu_user_distance=parsed.rsu_user_distance.copy(),
            uav_user_distance=parsed.uav_user_distance.copy(),
            residual_users=residual_users.astype(np.int32),
            user_virtual_queue=self.Z.copy(),
            requested_content=self.requested_content.copy(),
            uav_cached_content=self.uav_cached_content.copy(),
        )
    
    def _apply_battery_transition(
        self,
        uav_idx: int,
        mu_active: bool,
        mode: UAVBatteryMode,
        links,
    ) -> Dict[str, Any]:
        """
        energy_model 및 queue_model 기준으로 동일 의미 transition 수행하는 함수
        """
        battery = self.batteries[uav_idx]

        soc_before = float(battery.soc)
        virtual_before = float(battery.virtual_q)

        energy_info = compute_energy_summary(
            config=self.cfg.battery,
            mode=mode,
            mu_active=bool(mu_active),
            links=links,
            consume_hover_when_idle=battery.consume_hover_when_idle,
        )
        
        consumed_soc, charged_soc, next_soc, next_virtual_q = update_soc_virtual_q(
            config=self.cfg.battery,
            soc=battery.soc,
            consumed_energy=float(energy_info["total_energy"]),
            charged_energy=float(energy_info["charge_energy"]),
        )

        battery.soc = float(next_soc)
        battery_virtual_q = float(next_virtual_q)
        battery.round_remaining_slots = max(0, int(battery.round_remaining_slots) - 1)

        is_outage = bool(check_outage(battery.soc))
        self.outage[uav_idx] = int(is_outage)

        step_info = BatteryStepInfo(
            hover_energy=float(energy_info["hover_energy"]),
            comm_energy=float(energy_info["comm_energy"]),
            total_consumed=float(energy_info["total_energy"]),
            charged_energy=float(energy_info["charge_energy"]),
            consumed_soc=float(consumed_soc),
            charged_soc=float(charged_soc),
            soc_before=float(soc_before),
            soc_after=float(battery.soc),
            virtual_before=float(virtual_before),
            virtual_after=float(battery_virtual_q),
            outage=bool(is_outage),
        )

        return asdict(step_info)
    
    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        현 상태값을 반환하는 함수
        """
        return {
            "Q": self.queue.copy(),
            "Z": self.Z.copy(),
            "E": self.E.copy(),
            "Y": self.Y.copy(),
            "uav_hiring": self.uav_hiring.copy(),
            "rsu_scheduling": self.rsu_scheduling.copy(),
            "uav_scheduling": self.uav_scheduling.copy(),
            "requested_content": self.requested_content.copy(),
            "uav_cached_content": self.uav_cached_content.copy(),
            "outage": self.outage.copy(),
            "charging_state": self.charging_state.copy(),
            "round_idx": np.array([self.round_idx], dtype=np.int32),
            "round_slot": np.array([self.round_slot], dtype=np.int32),
            "time": np.array([self.t], dtype=np.int32),
        }
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        에피소드 초기화 수행 함수로
        큐, 배터리, 위치 등 State 변수 초기화
        """
        # time slot 및 round 초기화 / 동시에 에피소드는 1 증가
        self.t = 0
        self.episode += 1
        self.round_idx = 0
        self.round_slot = 0

        # 사용자 큐 초기화
        self.queue = np.full(self.num_user, float(self.cfg.init_queue), dtype=np.float32)

        # slow-timescale decision 초기화
        self.rsu_scheduling = np.zeros(
            (self.num_rsu, self.num_user), dtype=np.int32
        )
        self.uav_hiring = np.zeros(self.num_uav, dtype=np.int32)
        self.uav_scheduling = np.zeros(
            (self.num_uav, self.num_user), dtype=np.int32
        )

        # content 초기화
        self.requested_content = self._sample_requested_content()
        self.uav_cached_content = self._sample_uav_cached_content()

        # battery 초기화
        for battery in self.batteries:
            battery.reset_episode()
            battery.start_round(round_horizon=self.slow_T)

        self.outage = np.zeros(self.num_uav, dtype=np.int32)
        self.charging_state = np.zeros(self.num_uav, dtype=np.int32)
        self.charge_counters = np.zeros(self.num_uav, dtype=np.float32)

        self.round_start_E = self.E.copy()

        return self._get_state()

    def step(self, action: EnvAction) -> tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        환경의 1-slot 진행 함수

        - Slow Timescale: UAV 고용 및 스케줄링 수행
        - Fast Timescale: 비디오 청크 및 레이어 전송 수행
        """
        parsed = parse_action(action, self.cfg)

        # round 경계에서 slow-timescale decision을 갱신함
        if self.t % self.slow_T == 0:
            self.rsu_scheduling = parsed.rsu_scheduling.copy()
            self.uav_hiring = parsed.uav_hiring.copy()
            self.uav_scheduling = parsed.uav_scheduling.copy()
            self._start_new_round()
        
        effective = self._build_effective_action(parsed)

        prev_E = self.E.copy()
        prev_Y = self.Y.copy()
        prev_Q = self.queue.copy()
        prev_Z = self.Z.copy()

        # RSU delivery
        rsu_result = compute_rsu_delivery(
            cfg=self.cfg,
            parsed=effective,
            rsu_channel=self.rsu_channel,
            rng=self.rng,
        )

        # UAV delivery
        battery_soc_before_uav = self.E.copy()
        uav_result = compute_uav_delivery(
            cfg=self.cfg,
            parsed=effective,
            battery_parsed=battery_soc_before_uav,
            uav_channel=self.uav_channel,
            rng=self.rng,
        )

        # Battery transition per UAV
        battery_step_info: list[Dict[str, Any]] = []

        for u in range(self.num_uav):
            mu_active = bool(self.uav_hiring[u])
            charge_flag = bool(effective.uav_charge[u])

            if not mu_active:
                mode = UAVBatteryMode.IDLE
                links = []
            elif charge_flag:
                mode = UAVBatteryMode.CHARGE
                links = []
            elif len(uav_result.links_uav[u]) > 0:
                mode = UAVBatteryMode.SERVE
                links = uav_result.links_uav[u]
            elif self.outage[u] == 1:
                mode = UAVBatteryMode.OUTAGE
                links = []
            else:
                mode = UAVBatteryMode.IDLE
                links = []
            
            self.charging_state[u] = int(mode == UAVBatteryMode.CHARGE)
            self.charge_counters[u] += int(mode == UAVBatteryMode.CHARGE)

            battery_info_u = self._apply_battery_transition(
                uav_idx=u,
                mu_active=mu_active,
                mode=mode,
                links=links,
            )
            battery_info_u["mode"] = str(mode.value)
            battery_step_info.append(battery_info_u)
        
        # delivery 총계산
        delivered_rsu_per_user = rsu_result.delivered_per_user.astype(np.float32)
        delivered_uav_per_user = uav_result.delivered_per_user.astype(np.float32)
        delivered_total_per_user = delivered_rsu_per_user + delivered_uav_per_user

        quality_rsu_per_user = rsu_result.quality_per_user.astype(np.float32)
        quality_uav_per_user = uav_result.quality_per_user.astype(np.float32)
        quality_total_per_user = quality_rsu_per_user + quality_uav_per_user

        playback = effective.playback.astype(np.float32)
        consumed = np.minimum(self.queue, playback)
        stall = np.maximum(playback - self.queue, 0.0)

        self.queue = np.clip(
            self.queue - consumed + delivered_total_per_user,
            0.0,
            float(self.cfg.max_queue),
        ).astype(np.float32)


        # reward 아직 확정되지않았으므로, 미구현

        # time update
        self.t += 1
        self.round_slot = self.t % self.slow_T

        done = False

        info: Dict[str, Any] = {
            "prev_Q": prev_Q.copy(),
            "next_Q": self.queue.copy(),
            "prev_Z": prev_Z.copy(),
            "next_Z": self.Z.copy(),
            "prev_E": prev_E.copy(),
            "next_E": self.E.copy(),
            "prev_Y": prev_Y.copy(),
            "next_Y": self.Y.copy(),
            "round_idx": int(self.round_idx),
            "round_slot": int(self.round_slot),
            "uav_hiring": self.uav_hiring.copy(),
            "rsu_scheduling": self.rsu_scheduling.copy(),
            "uav_scheduling": self.uav_scheduling.copy(),
            "requested_content": self.requested_content.copy(),
            "uav_cached_content": self.uav_cached_content.copy(),
            "playback": playback.copy(),
            "consumed": consumed.copy(),
            "stall": stall.copy(),
            "delivered_rsu_per_user": delivered_rsu_per_user.copy(),
            "delivered_uav_per_user": delivered_uav_per_user.copy(),
            "delivered_total_per_user": delivered_total_per_user.copy(),
            "quality_rsu_per_user": quality_rsu_per_user.copy(),
            "quality_uav_per_user": quality_uav_per_user.copy(),
            "quality_total_per_user": quality_total_per_user.copy(),
            "outage": self.outage.copy(),
            "charging_state": self.charging_state.copy(),
            "battery_step_info": battery_step_info,
            "rsu_result": {
                "requested_mask": rsu_result.requested_mask.copy(),
                "capped_mask": rsu_result.capped_mask.copy(),
                "active_mask": rsu_result.active_mask.copy(),
                "delivered_chunks": rsu_result.delivered_chunks.copy(),
                "delivered_bits": rsu_result.delivered_bits.copy(),
                "delivered_quality": rsu_result.delivered_quality.copy(),
                "raw_channel_gain": rsu_result.raw_channel_gain.copy(),
                "link_capacity_bps": rsu_result.link_capacity_bps.copy(),
            },
            "uav_result": {
                "requested_mask": uav_result.requested_mask.copy(),
                "capped_mask": uav_result.capped_mask.copy(),
                "active_mask": uav_result.active_mask.copy(),
                "delivered_chunks": uav_result.delivered_chunks.copy(),
                "delivered_bits": uav_result.delivered_bits.copy(),
                "delivered_quality": uav_result.delivered_quality.copy(),
                "raw_channel_gain": uav_result.raw_channel_gain.copy(),
                "link_capacity_bps": uav_result.link_capacity_bps.copy(),
                "tx_power": uav_result.tx_power.copy(),
                "charge_mask": uav_result.charge_mask.copy(),
            },
        }

        # 원래는 reward도 추가해야함
        return self._get_state(), done, info