import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from .ppo_config import PPOConfig
    from .ppo_network import ActorCritic
    from .ppo_buffer import RolloutBuffer
except ImportError:  # pragma: no cover - script-style fallback
    from ppo_config import PPOConfig
    from ppo_network import ActorCritic
    from ppo_buffer import RolloutBuffer

EPS = 1e-6


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: PPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.model = ActorCritic(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
    
    @staticmethod
    def _atanh(x: torch.Tensor) -> torch.Tensor:
        """
        tanh-squashed action을 raw Gaussian action으로 되돌리기 위한 inverse tanh 구현 함수로,
        정의역 문제를 피하기 위하 (-1 + EPS, 1 - EPS)로 clamp 적용
        """
        x = torch.clamp(x, -1.0 + EPS, 1.0 + EPS)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))
    
    @staticmethod
    def _squash_log_prob(dist: Normal, raw_action: torch.Tensor, squashed_action: torch.Tensor) -> torch.Tensor:
        """
        tanh transform의 Jacobian 보정 포함 log_pi(a|s) term을 구현하는 함수로,
        action dim별 log_prob을 합산하여 (batch,) shape 반환.
        """
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed_action.pow(2) + EPS)
        return log_prob.sum(dim=-1)
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        env와 상호작용할 때 action을 샘플링하는 함수
        """
        obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) # batch 차원 추가

        with torch.no_grad():
            dist, mean, value = self.model.get_dict(obs_torch)
            raw_action = mean if deterministic else dist.rsample() # action 샘플링 (deterministic이면 mean)
            action = torch.tanh(raw_action) # [-1, 1] 범위의 action 생성
            log_prob = self._squash_log_prob(dist, raw_action, action)

        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )
    
    def value(self, obs: np.ndarray) -> float:
        """
        buffer의 rollout 끝에서 bootstrap value 계산할 때 사용하는 함수
        """
        obs_torch = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, _, value = self.model.get_dict(obs_torch)

        return float(value.item()) 
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO update에서 현재 정책을 기준으로
        - new_log_prob
        - values
        - entropy
        값을 계산하는 함수로, 입력 actions는 이미 tanh-squashed action이라고 가정.
        """
        # 검증
        if obs.ndim != 2:
            raise ValueError(f"obs는 2D tensor (batch, obs_dim) shape이어야 합니다, got shape {tuple(obs.shape)}")
        
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        elif actions.ndim != 2:
            raise ValueError(
                f"actions는 1D 혹은 2D tensor shape이어야 합니다, got shape {tuple(actions.shape)}"
            )
        
        if obs.shape[0] != actions.shape[0]:
            raise ValueError(
                f"obs와 actions 사이에 Batch size가 일치하지 않습니다, obs: {obs.shape[0]} vs actions: {actions.shape[0]}"
            )
        
        dist, _, values = self.model.get_dict(obs)

        raw_action = self._atanh(actions)
        log_prob = self._squash_log_prob(dist, raw_action, actions)
        
        # raw Gaussian entropy
        entropy = dist.entropy().sum(dim=-1) # 이 entropy는 raw Gaussian의 entropy

        return log_prob, values, entropy
    
    def update(self, rollout: RolloutBuffer) -> Dict[str, float]:
        """
        PPO 학습 함수로,
        rollout은 이미 adv/returns 계산이 끝난 상태여야 함.
        """
        # 1. 데이터 준비
        batch = rollout.get_tensors(self.device) # rollout 전체를 batch로 바꿈
        num_samples = batch["obs"].shape[0]

        # 검증
        if num_samples <= 0:
            raise ValueError("Rollout batch가 비어 있습니다.")
        
        batch_size = min(self.cfg.batch_size, num_samples) # 미니배치 학습 준비

        # 학습 모니터링용
        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl_div": 0.0,
            "clip_frac": 0.0,
        }
        num_updates = 0

        self.model.train()

        # 여러 epochs 반복
        for _ in range(self.cfg.update_epochs):
            indices = torch.randperm(num_samples, device=self.device)

            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]

                obs_batch = batch["obs"][batch_idx]
                act_batch = batch["actions"][batch_idx]
                old_log_prob_batch = batch["log_probs"][batch_idx]
                returns_batch = batch["returns"][batch_idx]
                adv_batch = batch["advantages"][batch_idx]

                # 새 정책 평가
                new_log_prob, values, entropy = self.evaluate_actions(obs_batch, act_batch)
                ratio = torch.exp(new_log_prob - old_log_prob_batch)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv_batch               
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # critic loss
                value_loss = F.mse_loss(values, returns_batch)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_log_prob - old_log_prob_batch
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.cfg.clip_ratio).float())
                
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["approx_kl_div"] += float(approx_kl_div.item())
                stats["clip_frac"] += float(clip_frac.item())
                num_updates += 1

        for k in stats:
            stats[k] /= max(num_updates, 1)

        return stats

    def save(self, path: str) -> None:
        """
        model 및 optimizer 설정을 저장하는 함수
        """
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": vars(self.cfg)
            },
            path,
        )
    
    def load(self, path: str) -> None:
        """
        model 및 optimizer 상태를 로드하는 함수
        """
        ckpt = torch.load(path, map_location=self.device)

        if "model" not in ckpt:
            raise KeyError("체크포인트에 'model' 요소가 존재하지 않습니다.")
        
        self.model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

            # optimizer state tensor를 현 device로 이동
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
