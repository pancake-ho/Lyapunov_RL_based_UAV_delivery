from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


class RunningMeanStd:
    """
    Running mean/std estiator.

    PPO에서 observation space가 커지면 학습이 불안정해질 경우를 대비하여,
    fast_obs flatten 이후 reward를 제외한 normalization에 사용.
    """
    def __init__(
        self,
        shape: int | tuple[int, ...],
        eps: float = 1e-4,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = float(eps)

    def update(self, x: np.ndarray) -> None:
        """
        외부 데이터를 받아 batch 통계량을 계산하는 함수.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x, dtype=np.float64)
            batch_count = 1.0
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = float(x.shape[0])
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: float,
    ) -> None:
        """
        새롭게 들어오는 batch 통계량을 이용해 기존 통계량을 실제 갱신하는 함수.
        """
        delta = batch_mean - self.mean
        total_count = batch_count + self.mean

        new_mean = self.mean + delta * batch_count / total_count

        old_var_sum = self.var * self.count
        batch_var_sum = batch_var * batch_count
        mean_diff_var_correction = np.square(delta) * self.count * batch_count / total_count

        combined_var_sum = (
            old_var_sum
            + batch_var_sum
            + mean_diff_var_correction
        )
        new_var = combined_var_sum / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        """
        표준편차 계산 함수.
        """
        return np.sqrt(self.var)
    
    def state_dict(self) -> Dict[str, np.ndarray | float]:
        """
        디버깅을 위해, state를 Dict 형태로 반환해 주는 함수.
        """
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }
    
    def load_state_dict(self, state: Dict[str, np.ndarray | float]) -> None:
        """
        Dict 형태와 호환되는 load_state_dict 함수.
        """
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


class ObsNormalizer:
    """
    Observation Normalizer.

    현재 시나리오 기준:
        사용 위치:
            obs_vec = flatten_obs(obs)
            obs_vec = obs_normalizer.normalize(obs_vec, update=True)
    
    또한 eval 단계에서는 update=False 로 설정.
    """
    def __init__(
        self,
        obs_dim: int,
        clip: float = 10.0,
        eps: float = 1e-8,
    ) -> float:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        
        self.obs_dim = int(obs_dim)
        self.clip = float(clip)
        self.eps = float(eps)
        self.rms = RunningMeanStd(shape=(obs_dim,))

    def normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """
        input으로 주어지는 obs에 대해 실제 normalization을 적용.
        """
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)

        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs_dim mismatch: expected {self.obs_dim}, got {obs_arr.shape[0]}"
            )
        
        if update:
            self.rms.update(obs_arr)
        
        normalized = (obs_arr - self.rms.mean) / (self.rms.std + self.eps)
        normalized = np.clip(normalized, -self.clip, self.clip)

        return normalized.astype(np.float32)
    
    def state_dict(self) -> Dict[str, object]:
        """
        디버깅을 위해, state를 Dict 형태로 반환해 주는 함수.
        """
        return {
            "obs_dim": self.obs_dim,
            "clip": self.clip,
            "eps": self.eps,
            "rms": self.rms.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """
        Dict 형태와 호환되는 load_state_dict 함수.
        """
        self.obs_dim = int(state["obs_dim"])
        self.clip = float(state["clip"])
        self.eps = float(state["eps"])
        self.rms.load_state_dict(state["rms"])  # type: ignore[arg-type]