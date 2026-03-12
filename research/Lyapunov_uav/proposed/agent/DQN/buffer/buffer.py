from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np

@dataclass
class Transition:
    state: np.ndarray
    action: Dict[str, np.ndarray]
    action_mask: Dict[str, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    """
    DQN Agent를 위한 replay buffer 정의
    """
    def __init__(self, cap: int, seed: int=2026):
        self.cap = int(cap)
        self.buffer: Deque[Transition] = deque(maxlen=self.cap)
        self.rng = random.Random(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: Dict[str, np.ndarray],
        action_mask: Dict[str, np.ndarray],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        action_np = {
            k: np.asarray(v, dtype=np.int32).copy()
            for k, v in action.items()
        }
        action_mask_np = {
            k: np.asarray(v, dtype=np.float32).copy()
            for k, v in action_mask.items()
        }
        
        if set(action_np.keys()) != set(action_mask_np.keys()):
            raise ValueError("action과 action_mask는 identical keys를 가져야 합니다.")

        self.buffer.append(
            Transition(
                state=state.copy(),
                action=action_np,
                action_mask=action_mask_np,
                reward=float(reward),
                next_state=next_state.copy(),
                done=bool(done),
            )
        )
    
    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)

        states = np.stack([b.state for b in batch], axis=0).astype(np.float32)
        next_states = np.stack([b.next_state for b in batch], axis=0).astype(np.float32)
        rewards = np.asarray([b.reward for b in batch], dtype=np.float32)
        dones = np.asarray([b.done for b in batch], dtype=np.float32)

        action_keys = list(batch[0].action.keys())
        actions = {
            key: np.stack([b.action[key] for b in batch], axis=0).astype(np.int64)
            for key in action_keys
        }
        action_masks = {
            key: np.stack([b.action_mask[key] for b in batch], axis=0).astype(np.float32)
            for key in action_keys
        }

        return states, actions, action_masks, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)