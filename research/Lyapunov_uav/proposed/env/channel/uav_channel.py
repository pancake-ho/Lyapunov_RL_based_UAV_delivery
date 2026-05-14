from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from proposed.config import ChannelConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import ChannelConfig

from .base_channel import BaseChannelModel


class UAVChannelModel(BaseChannelModel):
    """
    UAV-user LoS A2G Channel Model 클래스

    현재 시나리오 기준:
        g_un(t) = beta_zero / (H^2 + ||q_u(t) - w_n(t)||^2)

        C_un(t) = W log2(1 + p_un(t) g_un(t) / (sigma_un^2 * Gamma))

    UAV trajectory optimization은 고려하지 않고, hovering 상태에서
    user와의 horizontal distance만 slot별 상태로 반영한다.
    """

    def __init__(self, config: ChannelConfig):
        super().__init__(config)

        self.altitude = float(getattr(config, "altitude", 50.0))
        self.beta_zero = float(getattr(config, "beta_zero", 1.0))
        self.receiver_noise_power = float(getattr(config, "noise_power", 1.0))
        self.capacity_gap = float(getattr(config, "capacity_gap", 1.0))

        if self.altitude < 0.0:
            raise ValueError(f"altitude는 0 이상이어야 합니다. 현재 값: {self.altitude}")
        if self.beta_zero <= 0.0:
            raise ValueError(f"beta_zero는 양수여야 합니다. 현재 값: {self.beta_zero}")
        if self.receiver_noise_power <= 0.0:
            raise ValueError(
                f"noise_power는 양수여야 합니다. 현재 값: {self.receiver_noise_power}"
            )
        if self.capacity_gap <= 0.0:
            raise ValueError(
                f"capacity_gap은 양수여야 합니다. 현재 값: {self.capacity_gap}"
            )

        self.noise_power = float(self.receiver_noise_power)

    def effective_horizontal_distance(
        self,
        distance: Optional[float] = None,
    ) -> float:
        """
        UAV와 user 사이의 horizontal distance ||q_u(t) - w_n(t)||를 반환한다.
        """
        d = self.distance if distance is None else float(distance)
        if d < 0.0:
            raise ValueError(f"distance는 0 이상이어야 합니다. 현재 값: {d}")
        return max(float(d), 0.0)

    def compute_gain(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        UAV-user channel power gain g_un(t)를 계산한다.

        rng는 RSUChannelModel과 동일한 인터페이스를 맞추기 위한 인자이며,
        현재 LoS free-space UAV channel에서는 사용하지 않는다.
        """
        _ = rng

        horizontal_distance = self.effective_horizontal_distance(distance)
        distance_square_3d = float(self.altitude ** 2 + horizontal_distance ** 2)
        distance_square_3d = max(distance_square_3d, 1e-12)

        return float(self.beta_zero / distance_square_3d)

    def snr_from_gain(
        self,
        tx_power: float,
        gain: float,
    ) -> float:
        """
        UAV transmit power와 UAV channel gain으로부터 SNR term을 계산한다.
        """
        tx_power = float(tx_power)
        gain = float(gain)

        if tx_power < 0.0:
            raise ValueError(f"tx_power는 0 이상이어야 합니다. 현재 값: {tx_power}")
        if gain < 0.0:
            raise ValueError(f"gain은 0 이상이어야 합니다. 현재 값: {gain}")

        if tx_power == 0.0 or gain == 0.0:
            return 0.0

        denominator = max(
            1e-12,
            float(self.receiver_noise_power) * float(self.capacity_gap),
        )
        return float(tx_power * gain / denominator)

    def compute_snr(
        self,
        tx_power: float,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        UAV transmit power p_un(t)와 distance로부터 SNR을 계산한다.
        """
        tx_power = float(tx_power)
        if tx_power < 0.0:
            raise ValueError(f"tx_power는 0 이상이어야 합니다. 현재 값: {tx_power}")
        if tx_power == 0.0:
            return 0.0

        gain = self.compute_gain(distance=distance, rng=rng)
        return self.snr_from_gain(tx_power=tx_power, gain=gain)

    def capacity(
        self,
        tx_power: float,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        UAV-user channel capacity를 계산한다.
        """
        snr = self.compute_snr(tx_power=tx_power, distance=distance, rng=rng)
        return float(self.bandwidth * math.log2(1.0 + snr))

    def capacity_from_gain(
        self,
        tx_power: float,
        gain: float,
    ) -> float:
        """
        이미 계산된 g_un(t)를 이용하여 UAV-user capacity를 계산한다.
        """
        snr = self.snr_from_gain(tx_power=tx_power, gain=gain)
        return float(self.bandwidth * math.log2(1.0 + snr))