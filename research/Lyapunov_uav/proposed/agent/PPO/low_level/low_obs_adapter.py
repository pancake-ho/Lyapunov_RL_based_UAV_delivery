from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    from proposed.env.interface import VectorSpec, fast_obs_spec, flatten_fast_obs
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from env.interface import VectorSpec, fast_obs_spec, flatten_fast_obs


@dataclass(frozen=True)
class LowObsSpec:
    """
    Low-level PPO observation vector 구조 정보.
    """
    obs_dim: int
    obs: Dict[str, slice]


class LowObsAdapter:
    """
    FastEnv observation dict를 low-level PPO용 1D float32 vector로 변환한다.

    Low-level policy는 slot 단위 fast-timescale controller이므로,
    round-level slow decision은 관측 정보로 포함하되 직접 변경하지 않는다.
    """
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._spec: VectorSpec = fast_obs_spec(cfg)
        self.spec = LowObsSpec(obs_dim=self._spec.dim, obs=dict(self._spec.slices))
        self.obs_dim = self.spec.obs_dim

    def transform(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return flatten_fast_obs(obs, self.cfg)

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self.transform(obs)
