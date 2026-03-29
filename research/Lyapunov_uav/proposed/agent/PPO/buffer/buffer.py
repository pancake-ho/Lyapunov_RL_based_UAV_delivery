import numpy as np
import torch

from typing import Dict, List


class RolloutBuffer:
    """
    PPO on-policy rollout buffer 클래스로,
    rollout 종료 후 adv 및 returns 계산도 수행
    """
    def __init__(self):
        self.reset()

    def __len__(self) -> int:
        return len(self.rewards)
    
    def reset(self) -> None:
        """
        (PPO) 현재 policy로 모은 rollout만 저장하고, 
        업데이트 후 버퍼를 비워주는 함수
        """
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

        # rollout 내 shape 일관성 검사용 변수
        self._obs_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        value: float,
        log_prob: float,
    ) -> None:
        """
        매 step마다 rollout transition을 저장하는 함수.

        - obs: 상태
        - action: env에 실제로 적용된 action
        - reward: 보상
        - done: episode 종료 여부 (0 or 1)
        - value: critic V(s)
        - log_prob: old policy 기준 log_pi(a|s)
        """
        # 차원 검증
        obs_arr = np.asarray(obs, dtype=np.float32)
        action_arr = np.atleast_1d(np.asarray(action, dtype=np.float32))

        if obs_arr.ndim == 0:
            raise ValueError("상태 변수는 least 1D shape이어야 합니다. Scalah type 상태 변수는 불가합니다.")
        
        if self._obs_shape is None:
            self._obs_shape = obs_arr.shape
        elif obs_arr.shape != self._obs_shape:
            raise ValueError(
                f"Rollout에서의 obs shape이 일치하지 않습니다: expected {self._obs_shape}, got {obs_arr.shape}"
            )
        
        if self._action_shape is None:
            self._action_shape = action_arr.shape
        elif action_arr.shape != self._action_shape:
            raise ValueError(
                f"Rollout에서의 action shape이 일치하지 않습니다: expected {self._action_shape}, got {action_arr.shape}"
            )
        
        done_f = float(done)
        if done_f not in (0.0, 1.0):
            raise ValueError(f"done은 0.0이거나 1.0의 값을 가져야 합니다, got {done_f}")
        
        self.obs.append(obs_arr.copy())
        self.actions.append(action_arr.copy())
        self.rewards.append(float(reward))
        self.dones.append(done_f)
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def compute_return_and_advantages(
        self,
        last_value: float,
        last_done: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        rollout 끝에서부터 거꾸로 GAE advantage 및 return을 계산하는 함수
        """
        # 검증
        if len(self) == 0:
            raise ValueError("비어있는 rollout buffer에 대해서는 adv를 계산할 수 없습니다.")
        
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma는 [0, 1] 범위 안에 존재해야 합니다, got {gamma}")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError(f"gae_lambda는 [0, 1] 범위 안에 존재해야 합니다, got {gae_lambda}")
        
        last_done_f = float(last_done)
        if last_done_f not in (0.0, 1.0):
            raise ValueError(f"last_done은 0.0이거나 1.0의 값을 가져야 합니다, got {last_done_f}")
        
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - last_done_f
            else:
                # 현재 transition이 terminal이면 다음 state bootstrap 차단
                next_non_terminal = 1.0 - dones[t]

            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]

        self.advantages = advantages
        self.returns = returns
    
    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        rollout 데이터를 torch.tensor type으로 변환하고,
        advantage normalization을 수행하는 함수 (안정적인 policy gradient를 위함)
        """
        # 검증
        if len(self) == 0:
            raise ValueError("RolloutBuffer가 비어 있습니다. tensor type으로 변환할 요소가 없습니다.")
        
        if self.advantages is None or self.returns is None:
            raise ValueError(
                "advantages/returns 변수가 계산되지 않았습니다.\n"
                "get_tensors() 호출 전 compute_return_and_advantages() 함수를 호출하세요,"
            )
        
        obs_np = np.stack(self.obs, axis=0).astype(np.float32, copy=False)
        actions_np = np.stack(self.actions, axis=0).astype(np.float32, copy=False)
        log_probs_np = np.asarray(self.log_probs, dtype=np.float32)
        values_np = np.asarray(self.values, dtype=np.float32)
        returns_np = np.asarray(self.returns, dtype=np.float32)
        adv_np = np.asarray(self.advantages, dtype=np.float32)

        if not (
            len(obs_np)
            == len(actions_np)
            == len(log_probs_np)
            == len(values_np)
            == len(returns_np)
            == len(adv_np)
        ):
            raise RuntimeError("RolloutBuffer 내부 변수의 length가 맞지 않습니다.")

        adv_mean = float(adv_np.mean())
        adv_std = float(adv_np.std())
    
        if not np.isfinite(adv_mean) or not np.isfinite(adv_std):
            raise RuntimeError(
                f"advantage 정보가 유효하지 않습니다: mean={adv_mean}, std={adv_std}"
            )

        adv_np = (adv_np - adv_mean) / (adv_std + 1e-8)

        return {
            "obs": torch.as_tensor(obs_np, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(actions_np, dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(log_probs_np, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns_np, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(adv_np, dtype=torch.float32, device=device),
            "values": torch.as_tensor(values_np, dtype=torch.float32, device=device),
        }