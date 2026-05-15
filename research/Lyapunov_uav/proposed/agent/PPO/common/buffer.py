from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """
    PPO mini-batch Rollout용 컨테이너.
    """
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    old_values: torch.Tensor


class RolloutBuffer:
    """
    PPO Rollout Buffer.

    현재 시나리오 기준:
        - fast PPO:
            obs     = flattened fast obs
            action  = flattened fast action
            reward  = slot-level fast reward 
            done    = terminated / truncated
        
        - slow PPO:
            obs     = flattened slow obs
            action  = flattened slow action
            reward  = round-level slow reward
            done    = episode done
    
    또한, common 폴더 내의 buffer는 action의 의미를 해석하지 않음.
    fast/slow action flattening은 각각 fast_action.py 및 slow_action.py에서 처리함.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device | str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if obs_dim <= 0:
            raise ValueError(f"obs_dim은 양수 값을 가져야 합니다. 현재 값: {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim은 양수 값을 가져야 합니다. 현재 값: {action_dim}")
        if capacity <= 0:
            raise ValueError(f"capacity는 양수 값을 가져야 합니다. 현재 값: {capacity}")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.cnt = 0
        self.full = False
        self.advantages_ready = False
    
    def __len__(self) -> int:
        """
        현재 Buffer의 길이/크기를 반환.
        """
        return self.capacity if self.full else self.cnt
    
    @property
    def is_full(self) -> bool:
        """
        현재 Buffer가 꽉 찼는 지 여부를 반환.
        """
        return self.full
    
    def reset(self) -> None:
        """
        Buffer의 정보(cnt, full, adv_ready)를 모두 초기화.
        """
        self.cnt = 0
        self.full = False
        self.advantages_ready = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Buffer에 experience push를 수행.
        """
        if self.full:
            raise RuntimeError("RolloutBuffer가 이미 full 상태입니다. update 전 reset() 함수를 호출하세요.")
        
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)

        if obs_arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs shape mismatch: expected ({self.obs_dim},),"
                f"got {obs_arr.shape}"
            )
        if action_arr.shape[0] != self.action_dim:
            raise ValueError(
                f"action shape mismatch: expected ({self.action_dim},), "
                f"got {action_arr.shape}"
            )
        
        self.obs[self.cnt] = obs_arr
        self.actions[self.cnt] = action_arr
        self.rewards[self.cnt] = reward
        self.dones[self.cnt] = done
        self.values[self.cnt] = value
        self.log_probs[self.cnt] = log_prob

        self.cnt += 1
        if self.cnt >= self.capacity:
            self.full = True
        
        self.advantages_ready = False

    def compute_returns_and_advs(
        self,
        last_value: float,
        last_done: bool,
        normalize_adv: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Generalized Advantage Estimation.

        현재 시나리오 기준:
            last_value:
                rollout buffer 마지막 next_obs에 대한 critic value.
                episode가 종료되었으면 보통 0.0의 값.
            
            last_done:
                마지막 next_state가 terminal이면 True.
        """
        size = len(self)
        if size == 0:
            raise RuntimeError("비어있는 Buffer에 대해서는 GAE를 계산할 수 없습니다.")
        
        last_gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - last_done

        for step in reversed(range(size)):
            if step == size - 1:
                non_terminal = next_non_terminal
                next_val = next_value
            else:
                non_terminal = 1.0 - self.dones[step + 1]
                next_val = self.values[step + 1]
            
            delta = (
                self.rewards[step]
                + self.gamma * next_val * non_terminal
                - self.values[step]
            )
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            self.advantages[step] = last_gae
        
        self.returns[:size] = self.advantages[:size] + self.values[:size]

        if normalize_adv and size > 1:
            adv = self.advantages[:size]
            self.advantages[:size] = (adv - adv.mean()) / (adv.std() + eps)

        self.advantages_ready = True
    
    def get_tensors(self) -> Tuple[torch.Tensor, ...]:
        """
        신경망으로의 입력을 위해, Buffer에 저장된 experience를 torch.Tensor 형태로 변환.
        """
        if not self.advantages_ready:
            raise ValueError("get_tensors() 호출 전 compute_returns_and_advs() 함수를 호출하세요.")
        
        size = len(self)
        obs = torch.as_tensor(self.obs[:size], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.actions[:size], dtype=torch.float32, device=self.device)

        old_log_probs = torch.as_tensor(self.log_probs[:size], dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(self.values[:size], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(self.returns[:size], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(self.advantages[:size], dtype=torch.float32, device=self.device)

        return obs, actions, old_log_probs, returns, advantages, old_values,
    
    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[RolloutBuffer]:
        """
        Buffer에 저장된 전체 experience 데이터를 batch_size 단위로 잘라서 하나씩 반환.
        또한 mini-batch를 반복적으로 꺼내기 위해 yield를 사용한 generator 역할을 수행.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size는 양수 값을 가져야 합니다. 현재 값: {batch_size}")
        
        obs, actions, old_log_probs, returns, advantages, old_values = self.get_tensors()
        size = obs.shape[0]

        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, size, batch_size):
            end = start + batch_size
            mini_batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long, device=self.device)

            yield RolloutBatch(
                obs=obs[mini_batch_idx],
                actions=actions[mini_batch_idx],
                old_log_probs=old_log_probs[mini_batch_idx],
                returns=returns[mini_batch_idx],
                advantages=advantages[mini_batch_idx],
                old_values=old_values[mini_batch_idx],
            )
    
    def summary(self) -> Dict[str, float]:
        """
        Buffer에 저장된 전체 experience 데이터를 요약해 반환. (디버깅 확인용)
        """
        size = len(self)
        if size == 0:
            return {
                "size": 0.0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "done_ratio": 0.0
            }
        
        return {
            "size": float(size),
            "reward_mean": float(np.mean(self.rewards[:size])),
            "reward_std": float(np.std(self.rewards[:size])),
            "done_ratio": float(np.mean(self.dones[:size])),
        }