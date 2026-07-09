from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    from config import EnvConfig
except ImportError:
    from config import EnvConfig


@dataclass(frozen=True)
class SlowActionSpec:
    """
    PPO가 출력할 Slow-Timescale raw action vector의 구조를 정의함.

    현재 시나리오 기준:
        y_mn(r)    : RSU-user scheduling, shape (M, N)
        mu_u(r)    : UAV hiring, shape (U,)
        phi_un(r)  : UAV-user scheduling, shape (U, N)
    """
    num_rsu: int
    num_user: int
    num_uav: int
    
    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "SlowActionSpec":
        """
        config로부터 state 반환을 수행.
        """
        return cls(
            num_rsu=int(cfg.num_rsu),
            num_user=int(cfg.num_user),
            num_uav=int(cfg.num_uav),
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
        Slow-Timescale action의 dim을 반환.
        """
        m = self.num_rsu
        n = self.num_user
        u = self.num_uav

        return (
            m * n       # rsu-user scheduling
            + u         # uav hiring
            + u * n     # uav-user scheduling
        )
    

class SlowActionCodec:
    """
    Slow-timescale PPO action codec.

    PPO policy output은 다음과 같이 env와 호환되는 dict로 변환:
        {
            "rsu_scheduling": np.ndarray[int32], shape (M, N),
            "uav_hiring": np.ndarray[int32], shape (U,),
            "uav_scheduling": np.ndarray[int32], shape (U, N),
        }
    """
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.spec = SlowActionSpec.from_config(cfg)
    
    @property
    def action_dim(self) -> int:
        """
        spec의 action dim을 반환.
        """
        return self.spec.action_dim
    
    @property
    def hiring_shape(self) -> Tuple[int]:
        """
        spec의 hiring action dim을 반환.
        """
        return (self.num_uav,)
    
    def _require_binary_flat_action(self, action: np.ndarray) -> np.ndarray:
        """
        Slow-Timescale PPO actor가 binary action만 뽑게 하는 기능 수행
        """
        arr = np.asarray(action, dtype=np.float32).reshape(-1)

        # 안전장치
        if arr.shape != (self.action_dim,):
            raise ValueError(
                f"Slow action dim mismatch: expected {(self.action_dim,)}, "
                f"got {arr.shape}"
            )
        
        if not np.all(np.isfinite(arr)):
            raise RuntimeError("Slow action contains NaN or Inf.")

        is_zero = np.isclose(arr, 0.0, atol=1e-6)
        is_one = np.isclose(arr, 1.0, atol=1e-6)

        if not np.all(is_zero | is_one):
            bad_idx = np.flatnonzero(~(is_zero | is_one))
            preview_idx = bad_idx[:10]
            preview_values = arr[preview_idx]
            raise ValueError(
                "SlowActionCodec은 binary Bernoulli sample만 받습니다. "
                "logit/probability/continuous action을 넘기지 마세요. "
                f"invalid_count={bad_idx.size}, "
                f"invalid_idx_preview={preview_idx.tolist()}, "
                f"invalid_value_preview={preview_values.tolist()}"
            )

        return (arr >= 0.5).astype(np.int32)
    
    def _split_binary_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        PPO의 output으로 나오는 긴 raw vector를 action 종류별 binary type으로 분리.
        """
        binary = self._require_binary_flat_action(action)

        m = self.spec.num_rsu
        n = self.spec.num_user
        u = self.spec.num_uav

        idx = 0

        rsu_scheduling = binary[idx: idx + m * n].reshape(m, n)
        idx += m * n

        uav_hiring = binary[idx: idx + u].reshape(u)
        idx += u

        uav_scheduling = binary[idx: idx + u * n].reshape(u, n)
        idx += u * n

        if idx != self.action_dim:
            raise RuntimeError("내부 action split 로직에서 error가 발생했습니다.")
        
        return {
            "rsu_scheduling": rsu_scheduling.astype(np.int32, copy=False),
            "uav_hiring": uav_hiring.astype(np.int32, copy=False),
            "uav_scheduling": uav_scheduling.astype(np.int32, copy=False),
        }
    
    def _parse_user_region(self, obs: Mapping[str, Any]) -> np.ndarray:
        """
        slow observation으로부터 user region을 파싱하는 기능 수행
        """
        if not isinstance(obs, Mapping):
            raise TypeError(f"slow obs must be a mapping/dict, got {type(obs)}")

        if "user_region" not in obs:
            raise KeyError(
                "slow obs에 'user_region'이 없습니다. "
                "현재 시나리오에서 x(r)은 user_region으로 구현되어야 합니다."
            )

        user_region = np.asarray(obs["user_region"], dtype=np.int32).reshape(-1)

        if user_region.shape != (self.spec.num_user,):
            raise ValueError(
                f"user_region shape mismatch: expected {(self.spec.num_user,)}, "
                f"got {user_region.shape}"
            )

        if np.any(user_region < 0) or np.any(user_region >= self.spec.num_rsu):
            raise ValueError(
                "user_region contains invalid region index. "
                f"valid range=[0, {self.spec.num_rsu - 1}], "
                f"value={user_region.tolist()}"
            )

        return user_region.astype(np.int32, copy=False)

    @staticmethod
    def _context_get(
        context: Optional[Any],
        obs: Optional[Mapping[str, Any]],
        name: str,
    ) -> Any:
        """
        Read auxiliary feasibility context.

        In the current scenario, requested_content and uav_cached_content are not
        policy-selected slow actions. They are environment states needed to remove
        physically impossible UAV links.
        """
        if isinstance(context, Mapping) and name in context:
            return context[name]

        if context is not None and hasattr(context, name):
            return getattr(context, name)

        if isinstance(obs, Mapping) and name in obs:
            return obs[name]

        return None
    
    def _parse_required_content_context(
        self,
        obs: Optional[Mapping[str, Any]],
        context: Optional[Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse content context required by the current UAV caching scenario.

        requested_content:
            shape (N,)
            requested_content[n] is the content requested by user n.

        uav_cached_content:
            shape (U,)
            uav_cached_content[u] is the single content cached by UAV u.

        This function intentionally raises an error when content context is missing.
        Reason:
            In the current scenario, UAV-user scheduling must be cache-feasible.
            Silently ignoring cache feasibility would make slow scheduling inconsistent
            with the UAV delivery model.
        """
        requested = self._context_get(context, obs, "requested_content")
        cached = self._context_get(context, obs, "uav_cached_content")

        if requested is None:
            raise KeyError(
                "requested_content가 없습니다. "
                "slow_action decode 시 context=env 또는 "
                "context={'requested_content': ..., 'uav_cached_content': ...}를 넘겨야 합니다."
            )

        if cached is None:
            raise KeyError(
                "uav_cached_content가 없습니다. "
                "slow_action decode 시 context=env 또는 "
                "context={'requested_content': ..., 'uav_cached_content': ...}를 넘겨야 합니다."
            )

        requested_arr = np.asarray(requested, dtype=np.int32).reshape(-1)
        cached_arr = np.asarray(cached, dtype=np.int32).reshape(-1)

        if requested_arr.shape != (self.spec.num_user,):
            raise ValueError(
                f"requested_content shape mismatch: expected {(self.spec.num_user,)}, "
                f"got {requested_arr.shape}"
            )

        if cached_arr.shape != (self.spec.num_uav,):
            raise ValueError(
                f"uav_cached_content shape mismatch: expected {(self.spec.num_uav,)}, "
                f"got {cached_arr.shape}"
            )

        return (
            requested_arr.astype(np.int32, copy=False),
            cached_arr.astype(np.int32, copy=False),
        )

    def _region_mask_rsu(self, user_region: np.ndarray) -> np.ndarray:
        """
        RSU m can schedule only users in region m.
        """
        rsu_idx = np.arange(self.spec.num_rsu, dtype=np.int32)[:, None]
        return (rsu_idx == user_region[None, :]).astype(np.int32)

    def _region_mask_uav(self, user_region: np.ndarray) -> np.ndarray:
        """
        UAV u is mapped to coverage region u.
        UAV u can schedule only users in region u.
        """
        uav_idx = np.arange(self.spec.num_uav, dtype=np.int32)[:, None]
        return (uav_idx == user_region[None, :]).astype(np.int32)

    def _cache_match_mask(
        self,
        requested_content: np.ndarray,
        uav_cached_content: np.ndarray,
    ) -> np.ndarray:
        """
        UAV u can serve user n only when:
            uav_cached_content[u] == requested_content[n]
        """
        return (
            uav_cached_content[:, None].astype(np.int32)
            == requested_content[None, :].astype(np.int32)
        ).astype(np.int32)

    def decode_with_info(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Decode a binary slow action into env.apply_slow_action()-compatible action.

        Args:
            action:
                Flattened binary Bernoulli sample with shape (M*N + U + U*N,).

                Layout:
                    [rsu_scheduling.flatten(),
                     uav_hiring,
                     uav_scheduling.flatten()]

            obs:
                Slow observation from env.get_slow_obs().
                Must include user_region.

            context:
                Must provide requested_content and uav_cached_content.
                Usually pass context=env.

        Returns:
            env_action:
                Dict compatible with env.apply_slow_action().

            info:
                Projection/feasibility metadata for logging/debugging.
        """
        parts = self._split_binary_action(action)

        user_region = self._parse_user_region(obs)
        requested_content, uav_cached_content = self._parse_required_content_context(
            obs=obs,
            context=context,
        )

        raw_rsu_scheduling = parts["rsu_scheduling"]
        raw_uav_hiring = parts["uav_hiring"]
        raw_uav_scheduling = parts["uav_scheduling"]

        rsu_region_mask = self._region_mask_rsu(user_region)
        uav_region_mask = self._region_mask_uav(user_region)
        cache_mask = self._cache_match_mask(
            requested_content=requested_content,
            uav_cached_content=uav_cached_content,
        )

        # ------------------------------------------------------------
        # 1) RSU scheduling
        # ------------------------------------------------------------
        # RL이 선택한 RSU-user scheduling 중 같은 region link만 유지한다.
        # capacity trimming은 여기서 하지 않는다.
        # 이유: 사용자를 고르는 선택은 slow policy가 해야 하며,
        #      codec이 top-k 같은 heuristic 선택을 하면 policy action 의미가 흐려진다.
        rsu_scheduling = (
            raw_rsu_scheduling
            * rsu_region_mask
        ).astype(np.int32)

        rsu_scheduled_user = (
            rsu_scheduling.sum(axis=0) > 0
        ).astype(np.int32)

        residual_user = (
            1 - rsu_scheduled_user
        ).astype(np.int32)

        # ------------------------------------------------------------
        # 2) UAV hiring
        # ------------------------------------------------------------
        # UAV hiring은 RL이 직접 결정한 값을 그대로 유지한다.
        # scheduling이 0이어도 hiring=1이면 그대로 둔다.
        # 이유: 불필요한 hiring은 slow reward의 hiring cost로 학습되어야 한다.
        uav_hiring = raw_uav_hiring.astype(np.int32, copy=True)

        # ------------------------------------------------------------
        # 3) UAV-user scheduling
        # ------------------------------------------------------------
        # RL이 선택한 UAV-user scheduling 중 물리적으로 가능한 link만 유지한다.
        #
        # 유지 조건:
        #   - UAV가 고용됨.
        #   - user가 RSU에 scheduling되지 않은 residual user임.
        #   - UAV와 user가 같은 region에 있음.
        #   - UAV cached content == user requested content.
        #
        # 여기서 residual user를 임의로 추가하지 않는다.
        # 즉, RSU가 scheduling하지 않은 user 중 어떤 user를 UAV가 맡을지는
        # 전적으로 raw_uav_scheduling, 즉 slow policy가 결정한다.
        uav_scheduling = (
            raw_uav_scheduling
            * uav_hiring[:, None]
            * residual_user[None, :]
            * uav_region_mask
            * cache_mask
        ).astype(np.int32)

        env_action: Dict[str, np.ndarray] = {
            "rsu_scheduling": rsu_scheduling.astype(np.int32, copy=False),
            "uav_hiring": uav_hiring.astype(np.int32, copy=False),
            "uav_scheduling": uav_scheduling.astype(np.int32, copy=False),
        }

        info: Dict[str, Any] = {
            "raw_rsu_links": int(np.sum(raw_rsu_scheduling)),
            "effective_rsu_links": int(np.sum(rsu_scheduling)),
            "raw_hired_uav": int(np.sum(raw_uav_hiring)),
            "effective_hired_uav": int(np.sum(uav_hiring)),
            "raw_uav_links": int(np.sum(raw_uav_scheduling)),
            "effective_uav_links": int(np.sum(uav_scheduling)),
            "num_rsu_scheduled_users": int(np.sum(rsu_scheduled_user)),
            "num_residual_users": int(np.sum(residual_user)),
            "num_cache_feasible_uav_user_pairs": int(np.sum(cache_mask)),
            "num_region_feasible_rsu_user_pairs": int(np.sum(rsu_region_mask)),
            "num_region_feasible_uav_user_pairs": int(np.sum(uav_region_mask)),
            "masked_rsu_wrong_region_links": int(
                np.sum(raw_rsu_scheduling * (1 - rsu_region_mask))
            ),
            "masked_uav_not_hired_links": int(
                np.sum(raw_uav_scheduling * (1 - uav_hiring[:, None]))
            ),
            "masked_uav_non_residual_links": int(
                np.sum(raw_uav_scheduling * uav_hiring[:, None] * (1 - residual_user[None, :]))
            ),
            "masked_uav_wrong_region_links": int(
                np.sum(raw_uav_scheduling * uav_hiring[:, None] * residual_user[None, :] * (1 - uav_region_mask))
            ),
            "masked_uav_cache_mismatch_links": int(
                np.sum(
                    raw_uav_scheduling
                    * uav_hiring[:, None]
                    * residual_user[None, :]
                    * uav_region_mask
                    * (1 - cache_mask)
                )
            ),
            "user_region": user_region.astype(np.int32, copy=True),
            "requested_content": requested_content.astype(np.int32, copy=True),
            "uav_cached_content": uav_cached_content.astype(np.int32, copy=True),
            "residual_user": residual_user.astype(np.int32, copy=True),
        }

        return env_action, info

    def decode(
        self,
        action: np.ndarray,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Decode a binary slow action into env.apply_slow_action()-compatible action.
        """
        env_action, _ = self.decode_with_info(
            action=action,
            obs=obs,
            context=context,
        )
        return env_action

    def zeros_env_action(self) -> Dict[str, np.ndarray]:
        """
        Explicit no-scheduling/no-hiring slow action.

        This is useful for baseline or reset tests.
        """
        return {
            "rsu_scheduling": np.zeros(self.spec.rsu_shape, dtype=np.int32),
            "uav_hiring": np.zeros(self.spec.hiring_shape, dtype=np.int32),
            "uav_scheduling": np.zeros(self.spec.uav_shape, dtype=np.int32),
        }

    def random_binary_action(
        self,
        *,
        rng: Optional[np.random.Generator] = None,
        rsu_user_prob: float = 0.5,
        uav_hire_prob: float = 0.35,
        uav_user_prob: float = 0.4,
    ) -> np.ndarray:
        """
        Generate a random binary slow action for smoke tests only.

        This does not apply scenario projection.
        Use decode() or decode_with_info() after this function.
        """
        generator = rng if rng is not None else np.random.default_rng()

        for name, value in {
            "rsu_user_prob": rsu_user_prob,
            "uav_hire_prob": uav_hire_prob,
            "uav_user_prob": uav_user_prob,
        }.items():
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        m = self.spec.num_rsu
        n = self.spec.num_user
        u = self.spec.num_uav

        rsu_scheduling = (
            generator.random((m, n)) < float(rsu_user_prob)
        ).astype(np.int32)

        uav_hiring = (
            generator.random(u) < float(uav_hire_prob)
        ).astype(np.int32)

        uav_scheduling = (
            generator.random((u, n)) < float(uav_user_prob)
        ).astype(np.int32)

        action = np.concatenate(
            [
                rsu_scheduling.reshape(-1),
                uav_hiring.reshape(-1),
                uav_scheduling.reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)

        if action.shape != (self.action_dim,):
            raise RuntimeError(
                f"random binary action shape mismatch: expected {(self.action_dim,)}, "
                f"got {action.shape}"
            )

        return action

    def random_env_action(
        self,
        obs: Mapping[str, Any],
        *,
        rng: Optional[np.random.Generator] = None,
        context: Optional[Any] = None,
        rsu_user_prob: float = 0.5,
        uav_hire_prob: float = 0.35,
        uav_user_prob: float = 0.4,
    ) -> Dict[str, np.ndarray]:
        """
        Scenario-projected random slow env action for smoke tests only.

        Main slow-HRL training should use the Bernoulli action sampled by SlowActorCritic.
        """
        action = self.random_binary_action(
            rng=rng,
            rsu_user_prob=rsu_user_prob,
            uav_hire_prob=uav_hire_prob,
            uav_user_prob=uav_user_prob,
        )

        return self.decode(
            action=action,
            obs=obs,
            context=context,
        )