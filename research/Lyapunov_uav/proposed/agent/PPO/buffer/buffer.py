import numpy as np
import torch
from typing import List, Dict

class RolloutBuffer:
    """
    """
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        value: float,
        log_prob: float,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def compute_return_and_advantages(
        self,
        last_value: float,
        last_done: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - (last_done if t == len(rewards) - 1 else dones[t + 1])
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
    
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        self.advantages = advantages
        self.returns = returns
    
    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        assert self.advantages is not None and self.returns is not None
        adv = self.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": torch.as_tensor(np.asarray(self.obs, dtype=np.float32), device=device),
            "actions": torch.as_tensor(np.asarray(self.actions, dtype=np.float32), device=device),
            "log_probs": torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32), device=device),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(adv, dtype=torch.float32, device=device),
            "values": torch.as_tensor(np.asarray(self.values, dtype=np.float32), device=device),
        }