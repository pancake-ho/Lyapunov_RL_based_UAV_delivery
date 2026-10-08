"""Behavior tests for queue support, outage semantics and candidate arithmetic."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from config_hppo import HPPOConfig
from hppo.env import P3HierarchicalEnv
from hppo.ppo import PPOAgent
from hppo.completion import FastPolicyCompletion
from hppo.train import RandomPolicy


def started_env(**kw):
    params = dict(num_regions=1, users_per_region=4, num_frames=1, frame_slots=2,
                  hidden_dims=(16,16), rollout_scenarios=2)
    params.update(kw)
    cfg = HPPOConfig(**params)
    env = P3HierarchicalEnv(cfg); env.reset()
    env.state.user_x[:] = cfg.rsu_x(0) + 50
    env.prepare_frame()
    raw = np.array([1,2,0,0])
    env.begin_frame({0:raw}, {0:env.proposal(0,raw).execute(1,2)})
    return cfg, env


class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(2026)

    def test_full_queue_agent_sampling_replay_and_invalid_request_rejection(self):
        cfg, env = started_env(initial_playback_queue=100.)
        masks = env.slot_action_masks(0)
        np.testing.assert_array_equal(masks[0], [True,True,False,False])
        # Playback in this slot frees one place even when the starting Q=Qe.
        agent = PPOAgent(cfg.slot_obs_dim, cfg.slot_action_nvec, cfg, 'slot_ppo')
        obs = env.get_slot_obs(0)
        for _ in range(30):
            a, lp, v, _ = agent.act(obs, masks)
            self.assertLessEqual(a[0],1); self.assertLessEqual(a[1],1)
            with torch.no_grad():
                replay, _, _ = agent.net.evaluate_actions(torch.tensor(obs[None]), torch.tensor(a[None]), agent._mask_tensors(masks))
            self.assertAlmostEqual(lp, float(replay[0]), places=5)
            agent.store(0, obs, masks, a, lp, v, .1, 0., True)
        self.assertTrue(all(np.isfinite(x) for x in agent.update().values()))
        bat = env.state.battery_j.copy(); q = env.state.queue.copy()
        bad = np.zeros(3*cfg.num_users, dtype=int); bad[0] = 3
        with self.assertRaisesRegex(ValueError, 'queue mask'):
            env.step_slot({0:bad})
        np.testing.assert_array_equal(bat,env.state.battery_j)
        np.testing.assert_array_equal(q,env.state.queue)
        env.state.queue[0] = 98.
        self.assertTrue(env.slot_action_masks(0)[0].all())

    def test_insufficient_rate_zero_delivery_but_uav_energy_is_spent(self):
        cfg, env = started_env()
        env.trace.rsu_fading[:] = .7; env.trace.uav_fading[:] = 1.
        raw = np.zeros(3*cfg.num_users, dtype=int)
        raw[0] = raw[1] = 3
        raw[cfg.num_users] = raw[cfg.num_users+1] = 3
        raw[2*cfg.num_users+1] = 3  # 1.5 W
        step = env.step_slot({0:raw}); rg = step.info['regions'][0]
        for u in rg['users'][:2]:
            self.assertLess(u['feasible_by_rate'], 3)
            self.assertEqual(u['delivered'],0)
            self.assertTrue(u['transmission_failed'])
            self.assertEqual(u['q_after'], 2.)
        self.assertAlmostEqual(rg['communication_energy_j'],1.5*cfg.slot_duration_s/cfg.pa_efficiency)
        self.assertAlmostEqual(rg['battery_before_j']-rg['battery_after_j'],cfg.hover_energy_per_slot_j+rg['communication_energy_j'])
        # Same physical state/channel, one requested RSU chunk fits and succeeds.
        cfg2, env2 = started_env(); env2.trace.rsu_fading[:] = .7
        raw[:] = 0; raw[0] = 1; raw[cfg2.num_users] = 3
        u = env2.step_slot({0:raw}).info['regions'][0]['users'][0]
        self.assertEqual(u['delivered'],1)
        self.assertFalse(u['transmission_failed'])

    def test_every_candidate_sample_decomposes_and_infeasible_points_are_visible(self):
        cfg = HPPOConfig(num_regions=1, users_per_region=4, frame_slots=2)
        env = P3HierarchicalEnv(cfg); env.reset()
        env.state.uav_x[0] = cfg.candidate_points(0)[0]
        env.prepare_frame()
        before = copy.deepcopy(env.state)
        _, detail = FastPolicyCompletion(cfg).select(env,0,np.array([1,2,0,0]),RandomPolicy(8,cfg))
        self.assertEqual(len(detail['point_checks']),5)
        self.assertEqual([p['point'] for p in detail['point_checks'] if not p['feasible']],[3,4])
        self.assertEqual(len(detail['candidates']),4)
        for row in detail['candidates']:
            for sample, score in zip(row['sample_components'],row['sample_dpp']):
                self.assertEqual(len(sample['slots']),2)
                self.assertAlmostEqual(sum(x['dpp'] for x in sample['slots'])+sample['hiring_dpp'],score)
                self.assertAlmostEqual(sum(x['queue_drift']+x['quality_dpp'] for x in sample['users'])+sample['hiring_dpp'],score)
        np.testing.assert_array_equal(before.queue, env.state.queue)
        np.testing.assert_array_equal(before.battery_j, env.state.battery_j)

    def test_revision_checkpoint_roundtrip(self):
        cfg, env = started_env()
        agent = PPOAgent(cfg.slot_obs_dim,cfg.slot_action_nvec,cfg,'slot_ppo')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'slot.pt'; agent.save(p)
            loaded = PPOAgent(cfg.slot_obs_dim,cfg.slot_action_nvec,cfg,'slot_ppo'); loaded.load(p)
            o,m = env.get_slot_obs(0),env.slot_action_masks(0)
            np.testing.assert_array_equal(agent.act(o,m,True)[0], loaded.act(o,m,True)[0])


if __name__ == '__main__':
    unittest.main()
