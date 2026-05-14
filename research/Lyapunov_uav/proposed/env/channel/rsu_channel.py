from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from proposed.config import ChannelConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import ChannelConfig

from .base_channel import BaseChannelModel


class RSUChannelModel(BaseChannelModel):
    """
    RSU-user channel model 클래스
    """
    def __init__(self, config: ChannelConfig, inr_db: Optional[float] = None):
        super().__init__(config)
        
        if inr_db is None:
            inr_db = getattr(config, "inr_db", 0.0)
        self.inr_db = float(inr_db)
    
    @property
    def inr_linear(self) -> float:
        """
        INR_dB를 linear scale로 변환해 주는 함수
        """
        return self.db_to_linear(self.inr_db)
    
    @property
    def gamma_linear(self) -> float:
        """
        gamma_dB를 linear scale로 변환해 주는 함수
        """
        return self.db_to_linear(self.gamma_db)
    
    def compute_channel_coeff(self, distance: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> complex:
        """
        RSU와 user 사이 channel gain h_mn(t)룰 반환해 주는 함수
        """
        return self.sample_channel_gain(distance=distance, rng=rng)
    
    def compute_gain(self, distance: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> float:
        """
        RSU와 user 사이 channel capacity 계산을 위한 |h_mn(t)|^2를 반환해 주는 함수
        """
        channel_coeff = self.compute_channel_coeff(distance=distance, rng=rng)
        return float(abs(channel_coeff) ** 2)
    
    def snr_from_gain(self, gain: float) -> float:
        """
        RSU와 user 사이 channel capacity 계산을 위해, gain으로부터 SNR을 계산하는 함수
        """
        channel_power_gain = max(0.0, float(gain))
        denominator = max(1e-12, 1.0 + self.inr_linear)
        return float(self.gamma_linear * channel_power_gain / denominator)

    def compute_snr(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        gain = self.compute_gain(distance=distance, rng=rng)
        return self.snr_from_gain(gain)
    
    def capacity(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        RSU에서 Channel Capacity를 계산하는 함수
        """
        gain = self.compute_gain(distance=distance, rng=rng)
        return self.capacity_from_gain(gain)
    
    def capacity_from_gain(
        self,
        gain: float,
    ) -> float:
        """'
        gain으로부터 Capacity를 계산하는 함수
        """
        sinr = self.snr_from_gain(gain)
        return float(self.bandwidth * math.log2(1.0 + sinr))