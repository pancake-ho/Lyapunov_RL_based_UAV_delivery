from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path

from config import EnvConfig

try:
    from agent.PPO.common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )
except ModuleNotFoundError:  # package 상대 import fallback
    from ..common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )

from .fast_action import FastActionCodec
from .fast_network import FastActorCritic


@dataclass
class FastPPOConfig:
    """
    Fast-timescale PPO hyperparameters.

    현재 시나리오 기준:
        수치 최적화 전 단계이므로, 기본값은 안정적인 training을 추구하는 쪽으로 설정함.
    """
    rollout_steps: int = 1024
    update_epochs: int = 5
    batch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    lr: float = 3e-4
    max_grad_norm: float = 0.5

    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    normalize_obs: bool = True
    normalize_adv: bool = True

    hidden_dims: Tuple[int, ...] = (256, 256)
    init_log_std: float = -0.5

    device: str = "auto"


class FastPPOAgent:
    """
    Fast-timescale PPO Agent.

    slot-level action만 학습하는 agent로,
    slow decision은 obs에서 읽어서 FastActionCodec이 env action에 붙임.
    """
    def __init__(
        self,
        env_cfg: EnvConfig,
        obs_dim: int,
        ppo_cfg: Optional[FastPPOConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg or FastPPOConfig()

        self.device = get_device(self.ppo_cfg.device)
        self.codec = FastActionCodec(env_cfg)

        self.obs_dim = obs_dim
        self.action_dim = self.codec.action_dim

        self.model = FastActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.ppo_cfg.hidden_dims,
            init_log_std=self.ppo_cfg.init_log_std,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ppo_cfg.lr,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.ppo_cfg.rollout_steps,
            device=self.device,
            gamma=self.ppo_cfg.gamma,
            gae_lambda=self.ppo_cfg.gae_lambda,
        )
        
        self.obs_normalizer = Optional[ObsNormalizer]
        if self.ppo_cfg.normalize_obs:
            self.obs_normalizer = ObsNormalizer(obs_dim=self.obs_dim)
        else:
            self.obs_normalizer = None

    def obs_to_vec(
        self,
        obs: Dict[str, Any],
        update_norm: bool = True,
    ) -> np.ndarray:
        """
        Input으로 주어지는 obs에 대하여 vector로 변환 및 normalization 수행.
        """
        obs_vec = flatten_obs(obs).astype(np.float32)

        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(
                f"obs dim mismatch: expected {self.obs_dim}, got {obs_vec.shape[0]}"
            )

        if self.obs_normalizer is not None:
            obs_vec = self.obs_normalizer.normalize(
                obs_vec,
                update=update_norm,
            )

        return obs_vec.astype(np.float32)
    
    @torch.no_grad()
    def select_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Dict[str, Any]:
        """
        실제 Fast Agent가 선택하는 action을 반환 및 env와 호환되게 설정.
        """
        obs_vec = self.obs_to_vec(
            obs,
            update_normalizer=update_norm,
        )

        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)

        raw_action_tensor, log_prob_tensor, value_tensor = self.model.act(
            obs_tensor,
            deterministic=deterministic,
        )

        raw_action = raw_action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        log_prob = float(log_prob_tensor.squeeze(0).detach().cpu().item())
        value = float(value_tensor.squeeze(0).detach().cpu().item())

        env_action = self.codec.decode(raw_action, obs)

        return {
            "obs_vec": obs_vec,
            "raw_action": raw_action,
            "env_action": env_action,
            "log_prob": log_prob,
            "value": value,
        }
    
    def store_transition(
        self,
        obs_vec: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        RolloutBuffer에 experience를 저장.
        """
        self.buffer.add(
            obs=obs_vec,
            action=raw_action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )

    @torch.no_grad()
    def estimate_value(
        self,
        obs: Dict[str, Any],
        update_norm: bool = False,
    ) -> float:
        """
        Input으로 주어지는 obs에 대하여 value 계산 후 detach/값 반환.
        """
        obs_vec = self.obs_to_vec(
            obs,
            update_normalizer=update_norm,
        )
        obs_tensor = to_tensor(obs_vec, device=self.device).unsqueeze(0)
        value = self.model.value(obs_tensor)
        return float(value.squeeze(0).detach().cpu().item())
    
    def finish_rollout(
        self,
        last_obs: Dict[str, Any],
        last_done: bool,
    ) -> None:
        """
        GAE 계산 및 last_done=False인 경우 value 추정까지 수행.
        """
        if last_done:
            last_value = 0.0
        else:
            last_value = self.estimate_value(
                last_obs,
                update_norm=False,
            )

        self.buffer.compute_returns_and_advs(
            last_value=last_value,
            last_done=last_done,
            normalize_adv=self.ppo_cfg.normalize_adv,
        )
    
    def update(self) -> Dict[str, float]:
        """
        Buffer에서 experience를 꺼내어 Agent 업데이트 및 로그 확인용 정보 반환.
        """
        if len(self.buffer) == 0:
            raise RuntimeError("비어있는 buffer로는 PPO Agent를 update할 수 없습니다.")
        
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kl_divs: list[float] = []
        clip_fracs: list[float] = []

        for _ in range(self.ppo_cfg.update_epochs):
            for batch in self.buffer.iter_minibatches(
                batch_size=self.ppo_cfg.batch_size,
                shuffle=True,
            ):
                new_log_prob, entropy, new_value = self.model.evaluate_actions(obs=batch.obs, actions=batch.actions)

                log_ratio = new_log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl_div = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = ((torch.abs(ratio - 1.0) > self.ppo_cfg.clip_coef).float().mean())
                
                adv = batch.advantages
                policy_grad_loss_1 = -adv * ratio
                policy_grad_loss_2 = -adv * torch.clamp(ratio, 1.0 - self.ppo_cfg.clip_coef, 1.0 + self.ppo_cfg.clip_coef)
                
                policy_loss = torch.max(policy_grad_loss_1, policy_grad_loss_2).mean()
                value_loss = F.mse_loss(new_value, batch.returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + self.ppo_cfg.value_coef * value_loss - self.ppo_cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.ppo_cfg.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))
                approx_kl_divs.append(float(approx_kl_div.detach().cpu().item()))
                clip_fracs.append(float(clip_frac.detach().cpu().item()))
        
        obs, actions, old_log_probs, returns, advantages, old_values = self.buffer.get_tensors()

        with torch.no_grad():
            _, _, value_after = self.model.evaluate_actions(obs, actions)
            explained_v = explained_var(y_pred=value_loss.detach().cpu().numpy(), y_true=returns.detach().cpu().numpy())
        
        logs = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl_divs": float(np.mean(approx_kl_divs)),
            "clip_fracs": float(np.mean(clip_fracs)),
            "explained_variance": float(explained_v),
            "buffer_reward_mean": float(self.buffer.summary()["reward_mean"]),
            "buffer_reward_std": float(self.buffer.summary()["reward_std"]),
        }
        self.buffer.reset()

        return logs
    
    def save(self, path: str | Path, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        학습된 model/optimizer 등 저장용.
        """
        merged_extra: Dict[str, Any] = {
            "fast_ppo_config": asdict(self.ppo_cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

        if self.obs_normalizer is not None:
            merged_extra["obs_normalizer"] = self.obs_normalizer.state_dict()

        if extra is not None:
            merged_extra.update(extra)

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            extra=merged_extra,
        )
    
    def load(
        self,
        path: str | Path,
        strict: bool = True,
        load_optimizer: bool = False,
    ) -> Dict[str, Any]:
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer if load_optimizer else None,
            device=self.device,
            strict=bool(strict),
        )

        extra = checkpoint.get("extra", {})
        obs_norm_state = extra.get("obs_normalizer", None)

        if obs_norm_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(obs_norm_state)

        return checkpoint