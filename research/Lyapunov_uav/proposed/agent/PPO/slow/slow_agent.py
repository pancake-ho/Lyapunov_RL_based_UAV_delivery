from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import EnvConfig

try:
    from agent.PPO.common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_slow_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )
except ModuleNotFoundError:  # pragma: no cover - package relative import fallback
    from ..common import (
        ObsNormalizer,
        RolloutBuffer,
        explained_var,
        flatten_slow_obs,
        get_device,
        load_checkpoint,
        save_checkpoint,
        to_tensor,
    )

from .slow_action import SlowActionCodec
from .slow_network import SlowActorCritic


@dataclass
class SlowPPOConfig:
    """
    Slow-timescale PPO hyperparameters.
    """
    # ------------------------------------------------------------
    # 1) rollout / PPO update
    # ------------------------------------------------------------
    rollout_rounds: int = 1024
    update_epochs: int = 4
    batch_size: int = 128

    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ------------------------------------------------------------
    # 2) optimization
    # ------------------------------------------------------------
    lr: float = 1e-5
    max_grad_norm: float = 0.5

    clip_coef: float = 0.10
    value_coef: float = 0.5

    # Bernoulli entropy is summed over all slow action bits.
    # action_dim can be large, so keep entropy_coef conservative.
    entropy_coef: float = 1e-4

    # ------------------------------------------------------------
    # 3) normalization
    # ------------------------------------------------------------
    normalize_obs: bool = True
    normalize_adv: bool = True

    # ------------------------------------------------------------
    # 4) network
    # ------------------------------------------------------------
    hidden_dims: Tuple[int, ...] = (256, 256)

    # Initial Bernoulli probability = sigmoid(init_action_logit).
    # -1.5 gives about 0.18 initial probability.
    # This avoids every RSU/UAV/user link being active at the beginning.
    init_action_logit: float = -1.5

    min_logit: float = -20.0
    max_logit: float = 20.0

    # ------------------------------------------------------------
    # 5) critic stability
    # ------------------------------------------------------------
    use_value_huber_loss: bool = True
    use_value_clip: bool = True
    value_clip_coef: float = 50.0

    # ------------------------------------------------------------
    # 6) debug / device
    # ------------------------------------------------------------
    fail_on_nan: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.rollout_rounds) <= 0:
            raise ValueError("rollout_rounds must be positive.")
        if int(self.update_epochs) <= 0:
            raise ValueError("update_epochs must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")

        if int(self.rollout_rounds) < int(self.batch_size):
            raise ValueError("rollout_rounds must be >= batch_size.")

        if int(self.rollout_rounds) % int(self.batch_size) != 0:
            raise ValueError("rollout_rounds must be divisible by batch_size.")

        if not (0.0 < float(self.gamma) <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

        if not (0.0 < float(self.gae_lambda) <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1].")

        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive.")

        if float(self.max_grad_norm) <= 0.0:
            raise ValueError("max_grad_norm must be positive.")

        if float(self.clip_coef) <= 0.0:
            raise ValueError("clip_coef must be positive.")

        if float(self.value_coef) < 0.0:
            raise ValueError("value_coef must be non-negative.")

        if float(self.entropy_coef) < 0.0:
            raise ValueError("entropy_coef must be non-negative.")

        if not isinstance(self.hidden_dims, tuple):
            self.hidden_dims = tuple(int(x) for x in self.hidden_dims)  # type: ignore[misc]

        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty.")

        if any(int(x) <= 0 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers.")

        if float(self.min_logit) >= float(self.max_logit):
            raise ValueError("min_logit must be smaller than max_logit.")

        if float(self.value_clip_coef) <= 0.0:
            raise ValueError("value_clip_coef must be positive.")
        

class SlowPPOAgent:
    """
    Slow-timescale PPO Agent.
    """
    def __init__(
        self,
        env_cfg: EnvConfig,
        obs_dim: int,
        ppo_cfg: Optional[SlowPPOConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.ppo_cfg = ppo_cfg or SlowPPOConfig()

        self.device = get_device(self.ppo_cfg.device)
        self.codec = SlowActionCodec(env_cfg)

        self.obs_dim = obs_dim
        self.action_dim = self.codec.action_dim

        self.model = SlowActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=self.ppo_cfg.hidden_dims,
            init_action_logit=float(self.ppo_cfg.init_action_logit),
            min_logit=float(self.ppo_cfg.min_logit),
            max_logit=float(self.ppo_cfg.max_logit),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ppo_cfg.lr,
            eps=1e-5,
        )

        self.buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=self.ppo_cfg.rollout_rounds,
            device=self.device,
            gamma=self.ppo_cfg.gamma,
            gae_lambda=self.ppo_cfg.gae_lambda,
        )
        
        if bool(self.ppo_cfg.normalize_obs):
            self.obs_normalizer: Optional[ObsNormalizer] = ObsNormalizer(obs_dim=self.obs_dim)
        else:
            self.obs_normalizer = None
    
    def _check_finite_array(self, name: str, arr: np.ndarray) -> None:
        if not bool(self.ppo_cfg.fail_on_nan):
            return

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def _check_binary_array(self, name: str, arr: np.ndarray) -> None:
        if not bool(self.ppo_cfg.fail_on_nan):
            return

        flat = np.asarray(arr, dtype=np.float32).reshape(-1)

        is_zero = np.isclose(flat, 0.0, atol=1e-6)
        is_one = np.isclose(flat, 1.0, atol=1e-6)

        if not np.all(is_zero | is_one):
            bad_idx = np.flatnonzero(~(is_zero | is_one))[:10]
            raise RuntimeError(
                f"{name} must be binary. "
                f"bad_idx_preview={bad_idx.tolist()}, "
                f"bad_value_preview={flat[bad_idx].tolist()}"
            )

    def _check_finite_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not bool(self.ppo_cfg.fail_on_nan):
            return

        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN or Inf.")

    def obs_to_vec(
        self,
        obs: Mapping[str, Any],
        update_norm: bool = True,
    ) -> np.ndarray:
        """
        Convert slow observation dict to normalized vector.

        Must use flatten_slow_obs() so that slow policy input remains:
            [Z, B, user_region]

        Do not include connection state here.
        connection state is fast-timescale condition.
        """
        obs_vec = flatten_slow_obs(dict(obs)).astype(np.float32)

        if obs_vec.shape[0] != self.obs_dim:
            raise ValueError(
                f"slow obs dim mismatch: expected {self.obs_dim}, "
                f"got {obs_vec.shape[0]}"
            )

        self._check_finite_array("slow_obs_vec_before_norm", obs_vec)

        if self.obs_normalizer is not None:
            obs_vec = self.obs_normalizer.normalize(
                obs_vec,
                update=bool(update_norm),
            )

        self._check_finite_array("slow_obs_vec_after_norm", obs_vec)

        return obs_vec.astype(np.float32, copy=False)

    @torch.no_grad()
    def select_action(
        self,
        obs: Mapping[str, Any],
        *,
        context: Optional[Any] = None,
        deterministic: bool = False,
        update_norm: bool = True,
    ) -> Dict[str, Any]:
        """
        Select one slow-timescale action.

        Args:
            obs:
                env.get_slow_obs().

            context:
                Usually env.
                SlowActionCodec uses this to read:
                    - requested_content
                    - uav_cached_content

            deterministic:
                False for training.
                True for evaluation.

            update_norm:
                True during training.
                False during evaluation.

        Returns:
            {
                "obs_vec":
                    flattened slow obs

                "binary_action":
                    Bernoulli sample before feasibility projection

                "raw_action":
                    alias of binary_action for PPO buffer compatibility

                "env_action":
                    env.apply_slow_action()-compatible action dict

                "log_prob":
                    log probability of binary_action

                "value":
                    critic value V_H(s_H)

                "action_info":
                    projection/logging info from SlowActionCodec
            }
        """
        obs_vec = self.obs_to_vec(
            obs=obs,
            update_norm=bool(update_norm),
        )

        obs_tensor = to_tensor(
            obs_vec,
            device=self.device,
        ).unsqueeze(0)

        action_tensor, log_prob_tensor, value_tensor = self.model.act(
            obs_tensor,
            deterministic=bool(deterministic),
        )

        binary_action = (
            action_tensor.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        log_prob = float(log_prob_tensor.squeeze(0).detach().cpu().item())
        value = float(value_tensor.squeeze(0).detach().cpu().item())

        self._check_finite_array("slow_binary_action", binary_action)
        self._check_binary_array("slow_binary_action", binary_action)

        env_action, action_info = self.codec.decode_with_info(
            action=binary_action,
            obs=obs,
            context=context,
        )

        return {
            "obs_vec": obs_vec,
            "binary_action": binary_action,
            "raw_action": binary_action,
            "env_action": env_action,
            "log_prob": log_prob,
            "value": value,
            "action_info": action_info,
        }

    def store_transition(
        self,
        obs_vec: np.ndarray,
        binary_action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        Store one round-level slow PPO transition.

        The stored action must be the original Bernoulli sample before projection.
        Do not store the projected env_action because PPO log_prob was computed
        for the original binary action.
        """
        if bool(self.ppo_cfg.fail_on_nan):
            if not np.isfinite(float(reward)):
                raise RuntimeError(f"slow reward is NaN or Inf: {reward}")
            if not np.isfinite(float(value)):
                raise RuntimeError(f"slow value is NaN or Inf: {value}")
            if not np.isfinite(float(log_prob)):
                raise RuntimeError(f"slow log_prob is NaN or Inf: {log_prob}")

        action_arr = np.asarray(binary_action, dtype=np.float32).reshape(-1)
        self._check_binary_array("stored_slow_action", action_arr)

        self.buffer.add(
            obs=np.asarray(obs_vec, dtype=np.float32).reshape(-1),
            action=action_arr,
            reward=float(reward),
            done=bool(done),
            value=float(value),
            log_prob=float(log_prob),
        )

    @torch.no_grad()
    def estimate_value(
        self,
        obs: Mapping[str, Any],
        update_norm: bool = False,
    ) -> float:
        """
        Estimate V_H(s_H) for GAE bootstrapping.
        """
        obs_vec = self.obs_to_vec(
            obs=obs,
            update_norm=bool(update_norm),
        )

        obs_tensor = to_tensor(
            obs_vec,
            device=self.device,
        ).unsqueeze(0)

        value = self.model.value(obs_tensor)

        self._check_finite_tensor("slow_estimated_value", value)

        return float(value.squeeze(0).detach().cpu().item())

    def finish_rollout(
        self,
        last_obs: Mapping[str, Any],
        last_done: bool,
    ) -> None:
        """
        Compute GAE for the current slow rollout.
        """
        if bool(last_done):
            last_value = 0.0
        else:
            last_value = self.estimate_value(
                obs=last_obs,
                update_norm=False,
            )

        self.buffer.compute_returns_and_advs(
            last_value=float(last_value),
            last_done=bool(last_done),
            normalize_adv=bool(self.ppo_cfg.normalize_adv),
        )

    def _compute_value_loss(
        self,
        new_value: torch.Tensor,
        old_value: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Critic loss with optional PPO-style value clipping and Huber loss.
        """
        if bool(self.ppo_cfg.use_value_clip):
            value_clipped = old_value + torch.clamp(
                new_value - old_value,
                -float(self.ppo_cfg.value_clip_coef),
                float(self.ppo_cfg.value_clip_coef),
            )

            if bool(self.ppo_cfg.use_value_huber_loss):
                value_loss_unclipped = F.smooth_l1_loss(
                    new_value,
                    returns,
                    reduction="none",
                )
                value_loss_clipped = F.smooth_l1_loss(
                    value_clipped,
                    returns,
                    reduction="none",
                )
            else:
                value_loss_unclipped = (new_value - returns).pow(2)
                value_loss_clipped = (value_clipped - returns).pow(2)

            return torch.max(
                value_loss_unclipped,
                value_loss_clipped,
            ).mean()

        if bool(self.ppo_cfg.use_value_huber_loss):
            return F.smooth_l1_loss(new_value, returns)

        return F.mse_loss(new_value, returns)

    def update(self) -> Dict[str, float]:
        """
        Run PPO update using the accumulated slow rollout.
        """
        if len(self.buffer) == 0:
            raise RuntimeError(
                "비어있는 slow buffer로는 PPO update를 수행할 수 없습니다."
            )

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        approx_kl_values: list[float] = []
        clip_frac_values: list[float] = []

        for _ in range(int(self.ppo_cfg.update_epochs)):
            for batch in self.buffer.iter_minibatches(
                batch_size=int(self.ppo_cfg.batch_size),
                shuffle=True,
            ):
                new_log_prob, entropy, new_value = self.model.evaluate_actions(
                    obs=batch.obs,
                    actions=batch.actions,
                )

                self._check_finite_tensor("slow_new_log_prob", new_log_prob)
                self._check_finite_tensor("slow_entropy", entropy)
                self._check_finite_tensor("slow_new_value", new_value)

                log_ratio = new_log_prob - batch.old_log_probs
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        torch.abs(ratio - 1.0)
                        > float(self.ppo_cfg.clip_coef)
                    ).float().mean()

                adv = batch.advantages

                policy_loss_1 = -adv * ratio
                policy_loss_2 = -adv * torch.clamp(
                    ratio,
                    1.0 - float(self.ppo_cfg.clip_coef),
                    1.0 + float(self.ppo_cfg.clip_coef),
                )
                policy_loss = torch.max(
                    policy_loss_1,
                    policy_loss_2,
                ).mean()

                value_loss = self._compute_value_loss(
                    new_value=new_value,
                    old_value=batch.old_values,
                    returns=batch.returns,
                )

                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + float(self.ppo_cfg.value_coef) * value_loss
                    - float(self.ppo_cfg.entropy_coef) * entropy_mean
                )

                self._check_finite_tensor("slow_ppo_loss", loss)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.ppo_cfg.max_grad_norm),
                )

                self.optimizer.step()

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_values.append(float(entropy_mean.detach().cpu().item()))
                approx_kl_values.append(float(approx_kl.detach().cpu().item()))
                clip_frac_values.append(float(clip_frac.detach().cpu().item()))

        obs, actions, old_log_probs, returns, advantages, old_values = (
            self.buffer.get_tensors()
        )

        with torch.no_grad():
            _, _, value_after = self.model.evaluate_actions(
                obs,
                actions,
            )
            explained_v = explained_var(
                y_pred=value_after.detach().cpu().numpy(),
                y_true=returns.detach().cpu().numpy(),
            )

        buffer_summary = self.buffer.summary()

        logs = {
            "policy_loss": (
                float(np.mean(policy_losses)) if policy_losses else 0.0
            ),
            "value_loss": (
                float(np.mean(value_losses)) if value_losses else 0.0
            ),
            "entropy": (
                float(np.mean(entropy_values)) if entropy_values else 0.0
            ),
            "approx_kl": (
                float(np.mean(approx_kl_values)) if approx_kl_values else 0.0
            ),
            "clipfrac": (
                float(np.mean(clip_frac_values)) if clip_frac_values else 0.0
            ),
            "explained_variance": float(explained_v),
            "buffer_reward_mean": float(buffer_summary["reward_mean"]),
            "buffer_reward_std": float(buffer_summary["reward_std"]),
            "buffer_done_ratio": float(buffer_summary["done_ratio"]),
        }

        self.buffer.reset()

        return logs

    @torch.no_grad()
    def action_summary(
        self,
        obs: Mapping[str, Any],
        update_norm: bool = False,
    ) -> Dict[str, float]:
        """
        Return Bernoulli probability statistics for logging.
        """
        obs_vec = self.obs_to_vec(
            obs=obs,
            update_norm=bool(update_norm),
        )

        obs_tensor = to_tensor(
            obs_vec,
            device=self.device,
        ).unsqueeze(0)

        return self.model.action_summary(obs_tensor)

    def save(
        self,
        path: str | Path,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save slow policy checkpoint.
        """
        merged_extra: Dict[str, Any] = {
            "slow_ppo_config": asdict(self.ppo_cfg),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
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
        """
        Load slow policy checkpoint.
        """
        checkpoint = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer if bool(load_optimizer) else None,
            device=self.device,
            strict=bool(strict),
        )

        extra = checkpoint.get("extra", {})
        obs_norm_state = extra.get("obs_normalizer", None)

        if obs_norm_state is not None and self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(obs_norm_state)

        return checkpoint