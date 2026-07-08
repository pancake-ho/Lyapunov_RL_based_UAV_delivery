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

        uav_power = self._scale_to_float(
            parts["uav_power"],
            0.0,
            self.spec.max_tx_power,
        )

        return {
            "rsu_chunks": rsu_chunks,
            "rsu_layers": rsu_layers,
            "uav_chunks": uav_chunks,
            "uav_layers": uav_layers,
            "uav_power": uav_power,
        }

    def zeros_env_action(self, obs: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raw = np.zeros(self.action_dim, dtype=np.float32)
        return self.decode(raw, obs=obs)