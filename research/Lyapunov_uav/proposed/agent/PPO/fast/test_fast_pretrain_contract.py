from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import EnvConfig
from env.env import Env

from agent.PPO.config import FastTrainConfig, get_fast_ppo_config
from agent.PPO.common import infer_fast_obs_dim, split_env_reset, split_env_step
from agent.PPO.fast.fast_agent import FastPPOAgent
from agent.PPO.fast.fast_train import (
    build_agent_ppo_config,
    sample_random_slow_action,
)


class FastPretrainContractTest(unittest.TestCase):
    def test_environment_overrides_reach_agent_config(self) -> None:
        environment = {
            "FAST_PPO_PHASE": "pretrain",
            "FAST_PPO_DEVICE": "cpu",
            "FAST_PPO_FAIL_IF_CUDA_UNAVAILABLE": "0",
            "FAST_PPO_NUM_EPISODES": "2",
            "FAST_PPO_TARGET_TOTAL_EPISODES": "2",
            "FAST_PPO_ROLLOUT_SLOTS": "8",
            "FAST_PPO_BATCH_SIZE": "4",
            "FAST_PPO_UPDATE_EPOCHS": "2",
            "FAST_PPO_ACTOR_LR": "1.5e-5",
            "FAST_PPO_CRITIC_LR": "3e-5",
            "FAST_PPO_CAT_ENTROPY_COEF": "1e-4",
            "FAST_PPO_TARGET_KL": "none",
        }
        with patch.dict(os.environ, environment, clear=True):
            train_cfg = get_fast_ppo_config()
        ppo_cfg = build_agent_ppo_config(train_cfg)

        self.assertEqual(ppo_cfg.rollout_steps, 8)
        self.assertEqual(ppo_cfg.batch_size, 4)
        self.assertEqual(ppo_cfg.update_epochs, 2)
        self.assertAlmostEqual(ppo_cfg.actor_lr, 1.5e-5)
        self.assertAlmostEqual(ppo_cfg.critic_lr, 3e-5)
        self.assertAlmostEqual(ppo_cfg.categorical_entropy_coef, 1e-4)
        self.assertIsNone(ppo_cfg.target_kl)

    def test_one_round_update_and_checkpoint_roundtrip(self) -> None:
        env_cfg = EnvConfig(slow_T=8, episode_slots=8, seed=2026)
        env_cfg.battery.target_service_slots_per_round = env_cfg.slow_T
        env = Env(env_cfg)
        _, _ = split_env_reset(env.reset())

        train_cfg = FastTrainConfig(
            device="cpu",
            fail_if_cuda_unavailable=False,
            num_episodes=1,
            target_total_episodes=1,
            rounds_per_episode=1,
            rollout_slots=8,
            batch_size=4,
            update_epochs=2,
        )
        rng = np.random.default_rng(train_cfg.seed + 10_007)
        env.apply_slow_action(sample_random_slow_action(env, rng, train_cfg))
        obs = env.get_fast_obs()

        ppo_cfg = build_agent_ppo_config(train_cfg)
        agent = FastPPOAgent(
            env_cfg=env_cfg,
            obs_dim=infer_fast_obs_dim(obs),
            ppo_cfg=ppo_cfg,
        )

        actor_ids = {id(parameter) for parameter in agent.actor_parameters}
        critic_ids = {id(parameter) for parameter in agent.critic_parameters}
        self.assertFalse(actor_ids & critic_ids)
        self.assertEqual(
            [group.get("name") for group in agent.optimizer.param_groups],
            ["actor", "critic"],
        )

        raw_observations: list[np.ndarray] = []
        for slot_idx in range(env_cfg.slow_T):
            selected = agent.select_action(
                obs,
                deterministic=False,
                update_norm=False,
            )
            raw_observations.append(selected["raw_obs_vec"])
            next_obs, reward, terminated, truncated, info = split_env_step(
                env.step(selected["env_action"])
            )
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            boundary = slot_idx == env_cfg.slow_T - 1
            self.assertEqual(bool(info["is_round_boundary"]), boundary)
            agent.store_transition(
                obs_vec=selected["obs_vec"],
                raw_action=selected["raw_action"],
                action_mask=selected["action_mask"],
                reward=float(reward) * train_cfg.ppo_reward_scale,
                done=boundary,
                value=selected["value"],
                log_prob=selected["log_prob"],
            )
            obs = next_obs

        agent.finish_rollout(last_obs=obs, last_done=True)
        logs = agent.update()
        agent.update_obs_normalizer(np.stack(raw_observations, axis=0))

        self.assertEqual(logs["completed_minibatches"], 4.0)
        self.assertEqual(logs["expected_minibatches"], 4.0)
        for key in (
            "approx_kl_post",
            "clipfrac_post",
            "value_rmse",
            "actor_grad_norm",
            "critic_grad_norm",
        ):
            self.assertTrue(np.isfinite(logs[key]), key)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "roundtrip.pt"
            agent.save(checkpoint, extra={"resume_signature": "test"})
            restored = FastPPOAgent(
                env_cfg=env_cfg,
                obs_dim=agent.obs_dim,
                ppo_cfg=ppo_cfg,
            )
            payload = restored.load(checkpoint, load_optimizer=True)
            self.assertEqual(payload["extra"]["resume_signature"], "test")
            self.assertEqual(
                [
                    group.get("name")
                    for group in restored.optimizer.param_groups
                ],
                ["actor", "critic"],
            )


if __name__ == "__main__":
    unittest.main()
