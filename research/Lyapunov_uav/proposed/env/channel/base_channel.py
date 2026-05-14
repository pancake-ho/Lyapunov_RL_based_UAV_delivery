from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from proposed.config import ChannelConfig
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from config import ChannelConfig


class BaseChannelModel:
    """
    공통 channel utility base class.

    RSU/UAV channel에서 공통으로 쓰는 path loss, log-normal shadowing, Rayleigh fast fading sampling을 제공함.
    normalized channel power gain 생성을 담당
    """
    def __init__(self, config: ChannelConfig):
        self.config = config

        self.distance = float(config.distance)
        self.bandwidth = float(config.bandwidth)

        # RSU 식의 gamma 또는 reference SNR로 사용
        self.gamma_db = float(config.gamma_db)

        # Shadowing parameter
        self.shadowing_sigma_db = float(config.sigma_db)
        self.shadowing_mu_db = float(config.mu_db)

        # Path-loss exponent
        self.beta = float(config.beta)

        self.min_distance = float(config.min_distance)

        if self.distance < 0.0:
            raise ValueError(f"distance는 0 이상이어야 합니다. 현재 값: {self.distance}")
        if self.bandwidth <= 0.0:
            raise ValueError(f"bandwidth는 양수여야 합니다. 현재 값: {self.bandwidth}")
        if self.beta <= 0.0:
            raise ValueError(f"beta는 양수여야 합니다, 현재 값: {self.beta}")
        if self.min_distance <= 0.0:
            raise ValueError(f"min_distance는 양수여야 합니다. 현재 값: {self.min_distance}")
        if self.shadowing_sigma_db < 0.0:
            raise ValueError(f"sigma_db는 0 이상이어야 합니다, 현재 값: {self.shadowing_sigma_db}")
        
        seed = getattr(config, "seed", None)
        self.rng = np.random.default_rng(seed)

        # normalized noise power used by delivery/battery link info.
        self.noise_power = float(getattr(config, "noise_power", 1.0))
        
    @staticmethod
    def db_to_linear(db_value: float) -> float:
        """
        단위를 db(데시벨)에서 linear로 변경해 주는 함수
        """
        return 10.0 ** (float(db_value) / 10.0)
    
    def _rng(self, rng: Optional[np.random.Generator]) -> np.random.Generator:
        """
        난수 생성기 함수
        """
        return rng if rng is not None else self.rng
    
    def effective_distance(self, distance: Optional[float] = None) -> float:
        """
        RSU/UAV와 user 사이의 거리를 계산하는 함수
        """
        d = self.distance if distance is None else float(distance)
        return max(float(d), float(self.min_distance))
    
    def pathloss_gain(self, distance: Optional[float] = None) -> float:
        """
        RSU와 user 사이의 path loss를 계산하는 함수
        """
        d = self.effective_distance(distance)
        return float(d ** (-self.beta))
    
    def sample_shadowing_db(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        RSU에서 shadowing component를 계산하는 함수
        """
        generator = self._rng(rng)
        shadowing_db = generator.normal(
            loc=float(self.shadowing_mu_db), 
            scale=float(self.shadowing_sigma_db),
        )
        return float(shadowing_db)
    
    def sample_shadowing_linear(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        RSU에서 shadowing component(linear)를 계산하는 함수
        """
        shadowing_db = self.sample_shadowing_db(rng=rng)
        return float(self.db_to_linear(shadowing_db))
    
    def sample_rayleigh_complex(self, rng: Optional[np.random.Generator] = None) -> complex:
        """
        RSU에서 Rayleigh fast fading component를 계산하는 함수
        """
        generator = self._rng(rng)
        real = generator.normal(loc=0.0, scale=1.0)
        imag = generator.normal(loc=0.0, scale=1.0)
        return complex(real, imag) / np.sqrt(2.0)
    
    def sample_channel_gain(self, distance: Optional[float]=None, rng: Optional[np.random.Generator]=None) -> complex:
        """
        RSU에서 channel gain을 계산하는 함수
        """
        path_loss = self.pathloss_gain(distance=distance)
        shadowing_linear = self.sample_shadowing_linear(rng=rng)
        fast_fading = self.sample_rayleigh_complex(rng=rng)

        scale = np.sqrt(max(0.0, path_loss * shadowing_linear))
        return complex(scale * fast_fading)