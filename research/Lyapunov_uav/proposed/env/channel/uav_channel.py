import math
from typing import Optional

import numpy as np
from config import ChannelConfig

from .base_channel import BaseChannelModel

class UAVChannelModel(BaseChannelModel):
    """
    UAV-Vehicle 지상 link용 channel model 클래스로,
    BaseChannelModel class가 생성한 normalized channel power gain을 사용하여
    UAV의 slot별 transmit power에 따른 SNR 및 Shannon Capacity 계산을 담당.
    """
    def compute_gain(
        self,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        return self.sample_channel_gain(distance=distance, rng=rng)
    
    def compute_snr(
        self,
        tx_power: float,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        UAV의 Transmit SNR을 계산하는 함수로,
        RSU와 다르게 UAV는 Transmit Power를 조절할 수 있다고 가정한다.
        """
        if tx_power < 0.0:
            raise ValueError(f"tx_power는 0 이상이어야 합니다, 현재 값은 {tx_power}입니다.")
        if tx_power == 0.0:
            return 0.0
        
        gain = self.compute_gain(distance=distance, rng=rng)
        return self.snr_from_gain(tx_power, gain)
    
    def snr_from_gain(
        self,
        tx_power: float,
        gain: float,
    ) -> float:
        """
        이미 샘플링된 channel gain을 이용하여 SNR을 계산하는 함수
        """
        if tx_power < 0.0:
            raise ValueError(f"tx_power는 0 이상이어야 합니다, 현재 값은 {tx_power}입니다.")
        if tx_power == 0.0:
            return 0.0
        
        reference_snr = self.db_to_linear(self.gamma_db)
        return float(float(tx_power) * reference_snr * max(0.0, float(gain)))

    def capacity(
        self,
        tx_power: float,
        distance: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        snr = self.compute_snr(tx_power=tx_power, distance=distance, rng=rng)
        return self.bandwidth * math.log2(1.0 + snr)
    
    def capacity_from_gain(
        self,
        tx_power: float,
        gain: float,
    ) -> float:
        """
        이미 샘플링된 gain을 이용하여 capacity를 계산하는 함수
        """
        snr = self.snr_from_gain(tx_power=tx_power, gain=gain)
        return self.bandwidth * math.log2(1.0 + snr)