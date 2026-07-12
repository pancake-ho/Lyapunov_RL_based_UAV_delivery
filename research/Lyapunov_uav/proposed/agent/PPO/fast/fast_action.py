from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:
    from config import EnvConfig
except ImportError:
    from config import EnvConfig


@dataclass(frozen=True)
class FastActionSpec:
    """
    PPO가 출력할 raw action vector의 구조를 정의함.

    Ray action은 PPO policy가 출력하는 continuous vector로,
    이 vector를 env가 요구하는 dict action으로 decoding 처리.

    현재 시나리오 기준:
        action vector layout:
            1) rsu_chunks: shape (M, N)
            2) rsu_layers: shape (M, N)
            3) uav_chunks: shape (U, N)
            4) uav_layers: shape (U, N)
            5) uav_power: shape (U, N)
        
        Slow decision은 fast action vector에 포함하지 않고,
        obs 또는 env state에서 가져와 조건으로만 사용함.
    """
    num_rsu: int
    num_user: int
    num_uav: int
    max_chunk: int
    max_layer: int
    max_tx_power: float

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
    def action_dim(self) -> int:
        """
        전체 action의 dim을 반환.
        """
        m = self.num_rsu
        n = self.num_user
        u = self.num_uav

        return (
            m * n       # rsu_chunks
            + m * n     # rsu_layers
            + u * n     # uav_chunks
            + u * n     # uav_layers
            + u * n     # uav_power
        )
    

class FastActionCodec:
    """
    PPO의 output으로 나온 continuous raw action을 실제 env.step()에 넣을 수 있는 dict 형태로 변환. (호환성)

    중요한 원칙:
        - PPO는 raw continuous action만 학습함.
        - env에는 validators.py가 요구하는 dict action을 넘김.
        - slow decision은 fast policy의 action이 아니라 condition임.
    """
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.spec = FastActionSpec.from_config(cfg)
    
    @property
    def action_dim(self) -> int:
        """
        spec의 action dim을 반환.
        """
        return self.spec.action_dim
    
    def _split_raw_action(self, raw_action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        PPO의 output으로 나오는 긴 raw vector를 action 종류별 vector로 분리.
        """
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)

        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Fast raw action dim mismatch: expected {self.action_dim}, "
                f"got {action.shape[0]}"
            )
        
        m = self.spec.num_rsu
        n = self.spec.num_user
        u = self.spec.num_uav

        idx = 0

        rsu_chunks = action[idx: idx + m * n].reshape(m, n)
        idx += m * n

        rsu_layers = action[idx: idx + m * n].reshape(m, n)
        idx += m * n

        uav_chunks = action[idx: idx + u * n].reshape(u, n)
        idx += u * n

        uav_layers = action[idx: idx + u * n].reshape(u, n)
        idx += u * n

        uav_power = action[idx: idx + u * n].reshape(u, n)
        idx += u * n

        if idx != self.action_dim:
            raise RuntimeError("내부 action split 로직에서 error가 발생했습니다.")

        return {
            "rsu_chunks": rsu_chunks,
            "rsu_layers": rsu_layers,
            "uav_chunks": uav_chunks,
            "uav_layers": uav_layers,
            "uav_power": uav_power,
        }
    
    @staticmethod
    def _scale_to_int(
        raw: np.ndarray,
        min_value: int,
        max_value: int,
    ) -> np.ndarray:
        """
        raw continuous value를 [min_value, max_value] 범위의 int type으로 변환.

        또한 tanh를 사용하지 않아도 Gaussian raw action이 큰 값을 뽑아낼 수 있으므로,
        sigmoid-like clipping 대신 단순 clip 기반 scaling을 사용.

        현재 시나리오 기준:
            chunk 및 layer 수에 사용.
            raw <= -1 -> min_value
            raw >= +1 -> max_value
        """
        clipped = np.clip(raw, -1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5
        value = np.rint(min_value + scaled * (max_value - min_value))
        return np.clip(value, min_value, max_value).astype(np.int32)
    
    @staticmethod
    def _scale_to_float(
        raw: np.ndarray,
        min_value: float,
        max_value: float,
    ) -> np.ndarray:
        """
        raw continuous value를 연속 범위로 변환.

        현재 시나리오 기준:
            UAV power allocation에 사용.
        """
        clipped = np.clip(raw, -1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5
        value = min_value + scaled * (max_value - min_value)
        return np.clip(value, min_value, max_value).astype(np.float32)
    
    @staticmethod
    def _scale_to_power(
        raw: np.ndarray,
        min_positive_power: float,
        max_power: float,
    ) -> np.ndarray:
        """
        Gaussian raw action을 log-uniform power scale로 변환한다.

            raw <= -1 -> min_positive_power
            raw ==  0 -> geometric mean
            raw >= +1 -> max_power

        chunk 또는 layer가 0인 경우에는 decode 마지막 service mask에서
        power를 정확히 0으로 다시 설정한다.
        """
        min_positive_power = float(
            min_positive_power
        )
        max_power = float(max_power)

        if min_positive_power <= 0.0:
            raise ValueError(
                "min_positive_power는 양수여야 합니다."
            )
        if max_power <= min_positive_power:
            raise ValueError(
                "max_power는 min_positive_power보다 커야 합니다."
            )

        clipped = np.clip(
            np.asarray(raw, dtype=np.float32),
            -1.0,
            1.0,
        )

        fraction = (
            clipped + 1.0
        ) * 0.5

        log_min = np.log(min_positive_power)
        log_max = np.log(max_power)

        power = np.exp(
            log_min
            + fraction * (log_max - log_min)
        )

        return np.clip(
            power,
            min_positive_power,
            max_power,
        ).astype(np.float32)

    def decode(self, raw_action: np.ndarray, obs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        raw PPO action을 env.step()용 fast action dict로 변환한다.

        obs는 signature compatibility용으로만 받는다.
        현재 decode에는 slow decision이 필요 없다.
        """
        parts = self._split_raw_action(raw_action)

        rsu_chunks = self._scale_to_int(
            parts["rsu_chunks"],
            0,
            self.spec.max_chunk,
        )
        rsu_layers = self._scale_to_int(
            parts["rsu_layers"],
            0,
            self.spec.max_layer,
        )

        uav_chunks = self._scale_to_int(
            parts["uav_chunks"],
            0,
            self.spec.max_chunk,
        )
        uav_layers = self._scale_to_int(
            parts["uav_layers"],
            0,
            self.spec.max_layer,
        )

        uav_power = self._scale_to_power(
            parts["uav_power"],
            min_positive_power=float(
                self.cfg.battery.min_tx_power
            ),
            max_power=float(
                self.spec.max_tx_power
            ),
        )

        if obs is not None:
            mask_parts = self._split_raw_action(
                self.build_action_mask(obs)
            )

            rsu_mask = mask_parts[
                "rsu_chunks"
            ]
            uav_mask = mask_parts[
                "uav_chunks"
            ]

            rsu_chunks = (
                rsu_chunks * rsu_mask
            ).astype(np.int32)
            rsu_layers = (
                rsu_layers * rsu_mask
            ).astype(np.int32)

            uav_chunks = (
                uav_chunks * uav_mask
            ).astype(np.int32)
            uav_layers = (
                uav_layers * uav_mask
            ).astype(np.int32)
            uav_power = (
                uav_power * uav_mask
            ).astype(np.float32)

        if obs is not None:
            Z = np.asarray(
                obs["Z"],
                dtype=np.float32,
            )

            if Z.shape != (self.spec.num_user,):
                raise ValueError(
                    "Z shape mismatch: "
                    f"expected={(self.spec.num_user,)}, "
                    f"got={Z.shape}"
                )

            # 재생 이후 추가로 수용할 수 있는 최대 chunk 수:
            # min{Z_n(t)+b, Q_bar}
            user_chunk_headroom = np.floor(
                np.minimum(
                    Z + float(self.cfg.playback_rate),
                    float(self.cfg.max_queue),
                )
            ).astype(np.int32)

            user_chunk_headroom = np.clip(
                user_chunk_headroom,
                0,
                self.spec.max_chunk,
            )

            rsu_chunks = np.minimum(
                rsu_chunks,
                user_chunk_headroom[None, :],
            ).astype(np.int32)

            uav_chunks = np.minimum(
                uav_chunks,
                user_chunk_headroom[None, :],
            ).astype(np.int32)
            
        # chunk/layer 중 하나라도 0이면 실제 전송은 없음
        rsu_service_mask = (
            (rsu_chunks > 0)
            & (rsu_layers > 0)
        )
        uav_service_mask = (
            (uav_chunks > 0)
            & (uav_layers > 0)
        )

        rsu_chunks = np.where(
            rsu_service_mask,
            rsu_chunks,
            0,
        ).astype(np.int32)
        rsu_layers = np.where(
            rsu_service_mask,
            rsu_layers,
            0,
        ).astype(np.int32)

        uav_chunks = np.where(
            uav_service_mask,
            uav_chunks,
            0,
        ).astype(np.int32)
        uav_layers = np.where(
            uav_service_mask,
            uav_layers,
            0,
        ).astype(np.int32)
        uav_power = np.where(
            uav_service_mask,
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

    def zeros_env_action(self, obs: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
    
    def build_action_mask(
        self,
        obs: Dict[str, Any],
    ) -> np.ndarray:
        """
        현재 slot에서 formulation 상 유효한 Fast action dimension mask 계산.
        """
        rsu_connection = np.asarray(
        obs["rsu_connection"],
        dtype=np.float32,
        )
        uav_connection = np.asarray(
            obs["uav_connection"],
            dtype=np.float32,
        )

        if rsu_connection.shape != self.spec.rsu_shape:
            raise ValueError(
                "rsu_connection shape mismatch: "
                f"expected={self.spec.rsu_shape}, "
                f"got={rsu_connection.shape}"
            )

        if uav_connection.shape != self.spec.uav_shape:
            raise ValueError(
                "uav_connection shape mismatch: "
                f"expected={self.spec.uav_shape}, "
                f"got={uav_connection.shape}"
            )
        
        rsu_mask = (
            rsu_connection > 0
        ).astype(np.float32)

        uav_mask = (
            uav_connection > 0
        ).astype(np.float32)

        # I_u(t)=1{E_u(t)<=E_u^TH}
        if (
            bool(self.cfg.battery.allow_charge)
            and bool(
                self.cfg.battery.enable_charging
            )
        ):
            B = np.asarray(
                obs["B"],
                dtype=np.float32,
            )

            if B.shape != (self.spec.num_uav,):
                raise ValueError(
                    "B shape mismatch: "
                    f"expected={(self.spec.num_uav,)}, "
                    f"got={B.shape}"
                )

            # B_u(t)=E_bar-E_u(t)
            current_soc = (
                float(self.cfg.battery.e_max)
                - B
            )

            charging_uav = (
                current_soc
                <= float(self.cfg.battery.e_min)
            ).astype(np.float32)

            uav_mask *= (
                1.0
                - charging_uav[:, None]
            )

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
                f"expected={(self.action_dim,)}, "
                f"got={action_mask.shape}"
            )

        return action_mask