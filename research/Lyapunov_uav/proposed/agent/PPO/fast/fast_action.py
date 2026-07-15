from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from config import EnvConfig


@dataclass(frozen=True)
class FastActionSpec:
    """
    PPO가 출력할 raw action vector의 구조를 정의함.

    Ray action은 PPO policy가 출력하는 continuous vector로,
    이 vector를 env가 요구하는 dict action으로 decoding 처리.

    현재 시나리오 기준:
        action vector layout:
            1) rsu_chunks: shape (M, N) | {0, ..., L}
            2) rsu_layers: shape (M, N) | chunks = 0이면 0, chunks > 0이면 {1, ..., K}
            3) uav_chunks: shape (U, N) | {0, ..., L}
            4) uav_layers: shape (U, N) | chunks = 0이면 0, chunks > 0이면 {1, ..., K}
            5) uav_power: shape (U, N)  | Gaussian latent var
        
        Slow decision은 fast action vector에 포함하지 않고,
        obs 또는 env state에서 가져와 조건으로만 사용함.
    """
    num_rsu: int
    num_user: int
    num_uav: int

    max_chunk: int
    max_layer: int

    max_tx_power: float
    min_tx_power: float

    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "FastActionSpec":
        """
        config로부터 state 반환을 수행.
        """
        return cls(
            num_rsu=int(cfg.num_rsu),
            num_user=int(cfg.num_user),
            num_uav=int(cfg.num_uav),
            max_chunk=int(cfg.chunk),
            max_layer=int(cfg.layer),
            min_tx_power=float(cfg.battery.min_tx_power),
            max_tx_power=float(cfg.battery.max_tx_power),
        )
    
    @property
    def rsu_shape(self) -> Tuple[int, int]:
        """
        RSU와 관련된 action의 shape을 반환.
        """
        return (self.num_rsu, self.num_user)
    
    @property
    def uav_shape(self) -> Tuple[int, int]:
        """
        UAV와 관련된 action의 shape을 반환.
        """
        return (self.num_uav, self.num_user)

    @property
    def rsu_link_dim(self) -> int:
        return (
            self.num_rsu
            * self.num_user
        )

    @property
    def uav_link_dim(self) -> int:
        return (
            self.num_uav
            * self.num_user
        )

    @property
    def chunk_choices(self) -> int:
        # 0, 1, ..., L
        return self.max_chunk + 1

    @property
    def layer_choices(self) -> int:
        # service 상태에서는 1, ..., K
        return self.max_layer
    
    @property
    def action_dim(self) -> int:
        """
        전체 action의 dim을 반환.
        """
        return (
            2 * self.rsu_link_dim
            + 3 * self.uav_link_dim
        )

    @property
    def rsu_chunks_slice(self) -> slice:
        return slice(
            0,
            self.rsu_link_dim,
        )

    @property
    def rsu_layers_slice(self) -> slice:
        start = self.rsu_link_dim
        return slice(
            start,
            start + self.rsu_link_dim,
        )

    @property
    def uav_chunks_slice(self) -> slice:
        start = 2 * self.rsu_link_dim
        return slice(
            start,
            start + self.uav_link_dim,
        )

    @property
    def uav_layers_slice(self) -> slice:
        start = (
            2 * self.rsu_link_dim
            + self.uav_link_dim
        )
        return slice(
            start,
            start + self.uav_link_dim,
        )

    @property
    def uav_power_slice(self) -> slice:
        start = (
            2 * self.rsu_link_dim
            + 2 * self.uav_link_dim
        )
        return slice(
            start,
            start + self.uav_link_dim,
        )


class FastActionCodec:
    """
    Mixed policy action을 env action으로 변환.

    중요한 원칙:
        - chunk/layer는 더 이상 Gaussian scaling X, Network가 선택한 categorical index를 그대로 사용
        - UAV power만 Gaussian latent를 실제 power로 변환
    """
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.spec = FastActionSpec.from_config(cfg)
    
    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    def split_policy_vector(
        self,
        action: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        arr = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if arr.shape != (
            self.action_dim,
        ):
            raise ValueError(
                "Fast policy action dim mismatch: "
                f"expected={(self.action_dim,)}, "
                f"got={arr.shape}"
            )

        s = self.spec

        return {
            "rsu_chunks": (
                arr[s.rsu_chunks_slice]
                .reshape(s.rsu_shape)
            ),
            "rsu_layers": (
                arr[s.rsu_layers_slice]
                .reshape(s.rsu_shape)
            ),
            "uav_chunks": (
                arr[s.uav_chunks_slice]
                .reshape(s.uav_shape)
            ),
            "uav_layers": (
                arr[s.uav_layers_slice]
                .reshape(s.uav_shape)
            ),
            "uav_power_raw": (
                arr[s.uav_power_slice]
                .reshape(s.uav_shape)
            ),
        }

    @staticmethod
    def _discrete(
        value: np.ndarray,
        low: int,
        high: int,
    ) -> np.ndarray:
        """
        Network가 이미 categorical index를 출력하므로
        여기의 rint는 외부 입력에 대한 방어적 검증일 뿐이다.
        """
        arr = np.nan_to_num(
            value,
            nan=float(low),
            posinf=float(high),
            neginf=float(low),
        )

        return np.clip(
            np.rint(arr),
            low,
            high,
        ).astype(np.int32)

    @staticmethod
    def _decode_power(
        raw: np.ndarray,
        min_power: float,
        max_power: float,
    ) -> np.ndarray:
        if (
            min_power <= 0.0
            or max_power <= min_power
        ):
            raise ValueError(
                "0 < min_power < max_power "
                "조건이 필요합니다."
            )

        # Gaussian latent를 hard clipping하지 않는다.
        normalized = np.tanh(
            np.asarray(
                raw,
                dtype=np.float32,
            )
        )

        fraction = (
            normalized + 1.0
        ) * 0.5

        log_min = np.log(min_power)
        log_max = np.log(max_power)

        power = np.exp(
            log_min
            + fraction
            * (log_max - log_min)
        )

        return power.astype(np.float32)

    def decode(
        self,
        policy_action: np.ndarray,
        obs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        parts = self.split_policy_vector(
            policy_action
        )
        s = self.spec

        rsu_chunks = self._discrete(
            parts["rsu_chunks"],
            0,
            s.max_chunk,
        )
        rsu_layers = self._discrete(
            parts["rsu_layers"],
            0,
            s.max_layer,
        )

        uav_chunks = self._discrete(
            parts["uav_chunks"],
            0,
            s.max_chunk,
        )
        uav_layers = self._discrete(
            parts["uav_layers"],
            0,
            s.max_layer,
        )

        uav_power = self._decode_power(
            parts["uav_power_raw"],
            min_power=s.min_tx_power,
            max_power=s.max_tx_power,
        )

        if obs is not None:
            base_mask = (
                self.build_base_action_mask(
                    obs
                )
            )
            mask_parts = (
                self.split_policy_vector(
                    base_mask
                )
            )

            rsu_link_mask = (
                mask_parts["rsu_chunks"]
                > 0.0
            )
            uav_link_mask = (
                mask_parts["uav_chunks"]
                > 0.0
            )

            rsu_chunks = np.where(
                rsu_link_mask,
                rsu_chunks,
                0,
            )
            rsu_layers = np.where(
                rsu_link_mask,
                rsu_layers,
                0,
            )

            uav_chunks = np.where(
                uav_link_mask,
                uav_chunks,
                0,
            )
            uav_layers = np.where(
                uav_link_mask,
                uav_layers,
                0,
            )
            uav_power = np.where(
                uav_link_mask,
                uav_power,
                0.0,
            )

            Z = np.asarray(
                obs["Z"],
                dtype=np.float32,
            )

            if Z.shape != (
                s.num_user,
            ):
                raise ValueError(
                    "Z shape mismatch: "
                    f"expected={(s.num_user,)}, "
                    f"got={Z.shape}"
                )

            # Q(t+1)
            # = min(max(Q(t)-b,0)+d,Q_bar)
            #
            # 에서 유도되는 slot별 수용 가능 chunk.
            user_headroom = np.floor(
                np.minimum(
                    Z
                    + float(
                        self.cfg.playback_rate
                    ),
                    float(
                        self.cfg.max_queue
                    ),
                )
            ).astype(np.int32)

            user_headroom = np.clip(
                user_headroom,
                0,
                s.max_chunk,
            )

            rsu_chunks = np.minimum(
                rsu_chunks,
                user_headroom[None, :],
            )
            uav_chunks = np.minimum(
                uav_chunks,
                user_headroom[None, :],
            )

        # Conditional policy 불변식
        rsu_service = (
            rsu_chunks > 0
        )
        uav_service = (
            uav_chunks > 0
        )

        rsu_layers = np.where(
            rsu_service,
            rsu_layers,
            0,
        ).astype(np.int32)

        uav_layers = np.where(
            uav_service,
            uav_layers,
            0,
        ).astype(np.int32)

        uav_power = np.where(
            uav_service,
            uav_power,
            0.0,
        ).astype(np.float32)

        # 외부 action에 대한 최종 안전장치
        rsu_valid = (
            rsu_service
            & (rsu_layers > 0)
        )
        uav_valid = (
            uav_service
            & (uav_layers > 0)
        )

        rsu_chunks = np.where(
            rsu_valid,
            rsu_chunks,
            0,
        ).astype(np.int32)

        rsu_layers = np.where(
            rsu_valid,
            rsu_layers,
            0,
        ).astype(np.int32)

        uav_chunks = np.where(
            uav_valid,
            uav_chunks,
            0,
        ).astype(np.int32)

        uav_layers = np.where(
            uav_valid,
            uav_layers,
            0,
        ).astype(np.int32)

        uav_power = np.where(
            uav_valid,
            uav_power,
            0.0,
        ).astype(np.float32)

        return {
            "rsu_chunks": rsu_chunks,
            "rsu_layers": rsu_layers,
            "uav_chunks": uav_chunks,
            "uav_layers": uav_layers,
            "uav_power": uav_power,
        }

    def zeros_env_action(
        self,
        obs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "rsu_chunks": np.zeros(
                self.spec.rsu_shape,
                dtype=np.int32,
            ),
            "rsu_layers": np.zeros(
                self.spec.rsu_shape,
                dtype=np.int32,
            ),
            "uav_chunks": np.zeros(
                self.spec.uav_shape,
                dtype=np.int32,
            ),
            "uav_layers": np.zeros(
                self.spec.uav_shape,
                dtype=np.int32,
            ),
            "uav_power": np.zeros(
                self.spec.uav_shape,
                dtype=np.float32,
            ),
        }

    def build_base_action_mask(
        self,
        obs: Dict[str, Any],
    ) -> np.ndarray:
        """
        Slow scheduling 및 charging state로부터
        사전 유효 Fast action mask를 생성한다.

        이 mask 구조는 현재 시나리오와 맞으므로 유지한다.
        """
        s = self.spec

        rsu_connection = np.asarray(
            obs["rsu_connection"],
            dtype=np.float32,
        )
        uav_connection = np.asarray(
            obs["uav_connection"],
            dtype=np.float32,
        )

        if (
            rsu_connection.shape
            != s.rsu_shape
        ):
            raise ValueError(
                "rsu_connection shape mismatch: "
                f"expected={s.rsu_shape}, "
                f"got={rsu_connection.shape}"
            )

        if (
            uav_connection.shape
            != s.uav_shape
        ):
            raise ValueError(
                "uav_connection shape mismatch: "
                f"expected={s.uav_shape}, "
                f"got={uav_connection.shape}"
            )

        rsu_mask = (
            rsu_connection > 0
        ).astype(np.float32)

        uav_mask = (
            uav_connection > 0
        ).astype(np.float32)

        if (
            bool(
                self.cfg.battery.allow_charge
            )
            and bool(
                self.cfg.battery.enable_charging
            )
        ):
            B = np.asarray(
                obs["B"],
                dtype=np.float32,
            )

            if B.shape != (
                s.num_uav,
            ):
                raise ValueError(
                    "B shape mismatch: "
                    f"expected={(s.num_uav,)}, "
                    f"got={B.shape}"
                )

            current_soc = (
                float(
                    self.cfg.battery.e_max
                )
                - B
            )

            charging = (
                current_soc
                <= float(
                    self.cfg.battery.e_min
                )
            )

            uav_mask *= (
                ~charging[:, None]
            ).astype(np.float32)

        action_mask = np.concatenate(
            [
                rsu_mask.reshape(-1),
                rsu_mask.reshape(-1),
                uav_mask.reshape(-1),
                uav_mask.reshape(-1),
                uav_mask.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        if action_mask.shape != (
            self.action_dim,
        ):
            raise RuntimeError(
                "Fast action mask dim mismatch: "
                f"got={action_mask.shape}"
            )

        return action_mask

    # 기존 caller 호환
    def build_action_mask(
        self,
        obs: Dict[str, Any],
    ) -> np.ndarray:
        return (
            self.build_base_action_mask(
                obs
            )
        )

    def action_statistics(
        self,
        policy_action: np.ndarray,
        effective_mask: np.ndarray,
    ) -> Dict[str, float]:
        parts = self.split_policy_vector(
            policy_action
        )
        masks = self.split_policy_vector(
            effective_mask
        )

        chunks = np.concatenate(
            [
                parts["rsu_chunks"].reshape(-1),
                parts["uav_chunks"].reshape(-1),
            ]
        )

        chunk_mask = (
            np.concatenate(
                [
                    masks[
                        "rsu_chunks"
                    ].reshape(-1),
                    masks[
                        "uav_chunks"
                    ].reshape(-1),
                ]
            )
            > 0.0
        )

        layers = np.concatenate(
            [
                parts["rsu_layers"].reshape(-1),
                parts["uav_layers"].reshape(-1),
            ]
        )

        layer_mask = (
            np.concatenate(
                [
                    masks[
                        "rsu_layers"
                    ].reshape(-1),
                    masks[
                        "uav_layers"
                    ].reshape(-1),
                ]
            )
            > 0.0
        )

        active_chunks = chunks[
            chunk_mask
        ]

        active_layers = (
            layers[layer_mask]
            .astype(np.int64)
        )

        stats: Dict[str, float] = {
            "service_rate": (
                float(
                    np.mean(
                        active_chunks > 0.0
                    )
                )
                if active_chunks.size > 0
                else 0.0
            ),
            "mean_requested_chunks": (
                float(
                    np.mean(active_chunks)
                )
                if active_chunks.size > 0
                else 0.0
            ),
        }

        denominator = max(
            int(active_layers.size),
            1,
        )

        for layer in range(
            1,
            self.spec.max_layer + 1,
        ):
            stats[
                f"layer_{layer}_ratio"
            ] = float(
                np.sum(
                    active_layers == layer
                )
                / denominator
            )

        return stats