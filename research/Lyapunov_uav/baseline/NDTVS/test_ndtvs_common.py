"""Semantic regressions: shared physics, policy likelihoods, and exact resume."""
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
import ndtvs_common as n
from ndtvs_analysis import audit, compare


class BaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = replace(n.HPPOConfig(), num_regions=2, users_per_region=4,
                           num_frames=2, frame_slots=3, hidden_dims=(32, 32),
                           ppo_minibatch_size=32, ppo_update_epochs=2, device="cpu")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.json"
        n.atomic(self.config, {"config": asdict(self.cfg)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_initial_state_fading_and_rsu_transition(self):
        a, b = n.RSUEnv(self.cfg), n.P3HierarchicalEnv(self.cfg)
        for env in (a, b):
            env.reset(17)
            env.prepare_frame()
        np.testing.assert_array_equal(a.state.user_x, b.state.user_x)
        np.testing.assert_array_equal(a.trace.rsu_fading, b.trace.rsu_fading)
        raw = {}
        for m in a.regions:
            raw[m] = np.zeros(a.N, dtype=np.int64)
            raw[m][list(a.region_users[m])[:self.cfg.rsu_capacity]] = 1
        for env in (a, b):
            done = {m: env.proposal(m, x).execute(0, -1) for m, x in raw.items()}
            env.begin_frame(raw, done)
        actions = {}
        for m in a.regions:
            x = np.zeros(3 * a.N, dtype=np.int64)
            x[:a.N][raw[m] == 1] = 3
            x[a.N:2*a.N][raw[m] == 1] = 3
            actions[m] = x
        for _ in range(self.cfg.frame_slots):
            one, two = a.step_slot(actions), b.step_slot(actions)
            np.testing.assert_array_equal(a.state.queue, b.state.queue)
            np.testing.assert_array_equal(a.state.user_x, b.state.user_x)
            self.assertEqual(one.info, two.info)

    def test_hidden_csi_and_conditional_log_probability(self):
        env = n.RSUEnv(self.cfg)
        env.reset(0)
        env.prepare_frame()
        log = n.QoELogger(replace(self.cfg, write_jsonl_trace=False, write_human_debug_log=False), self.root)
        try:
            obs, masks = n.ndt_observation(env, 0, log, True)
            env.trace.rsu_fading[:] = 9.0
            obs2, masks2 = n.ndt_observation(env, 0, log, True)
            np.testing.assert_array_equal(obs, obs2)
            for x, y in zip(masks, masks2):
                np.testing.assert_array_equal(x, y)
            agent = n.ndt_agent(self.cfg)
            for _ in range(15):
                action, lp, _, _ = agent.act(obs, masks)
                picks = action[:self.cfg.rsu_capacity]
                chosen = picks[picks < env.N]
                self.assertEqual(len(set(chosen)), len(chosen))
                self.assertTrue(set(chosen) <= set(env.region_users[0]))
                for u, token in enumerate(action[self.cfg.rsu_capacity:]):
                    if u not in chosen:
                        self.assertEqual(token, 0)
                replay = agent.net.evaluate_actions(torch.tensor(obs[None]), torch.tensor(action[None]), agent._mask_tensors(masks))
                self.assertAlmostEqual(lp, float(replay[0]), places=5)
        finally:
            log.close()

    def test_queue_support_and_no_uav(self):
        env = n.RSUEnv(self.cfg)
        env.reset()
        env.prepare_frame()
        env.state.queue[:] = self.cfg.large_queue_level
        log = n.QoELogger(replace(self.cfg, write_jsonl_trace=False, write_human_debug_log=False), self.root)
        try:
            _, masks = n.ndt_observation(env, 0, log, True)
            for u in env.region_users[0]:
                self.assertFalse(masks[self.cfg.rsu_capacity + u][1 + env.K:].any())
            self.assertTrue(all(not x[2] for x in env.frame_action_masks(0)))
            raw = np.zeros(env.N, dtype=np.int64)
            raw[env.region_users[0][0]] = 2
            with self.assertRaises(ValueError):
                env.proposal(0, raw)
        finally:
            log.close()

    def test_qoe_first_delivery_switch_and_stall(self):
        rec = {"delivered": 2, "req_quality": 3, "last_quality_before": -1, "stall": 1}
        self.assertEqual(n.qoe_terms(rec, self.cfg), (0.0, 0.0))
        rec["last_quality_before"] = 0
        self.assertEqual(n.qoe_terms(rec, self.cfg), (-0.5, 1.0))
        rec["delivered"] = 0
        self.assertEqual(n.qoe_terms(rec, self.cfg), (-2.0, 0.0))

    def check_resume(self, algorithm):
        full, split = self.root / "full", self.root / "split"
        common = ["train", "--algorithm", algorithm, "--config", str(self.config),
                  "--episodes", "2", "--val-every", "1", "--val-episodes", "1", "--trace-every", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(n.main(common + ["--out", str(full)]), 0)
            self.assertEqual(n.main(common + ["--out", str(split), "--max-new-episodes", "1"]), 75)
            intermediate = n.load_bundle(split / "latest.pt")
            if algorithm == "hppo_rsu":
                self.assertTrue(intermediate["policies"][0]["trajectories"])
            self.assertEqual(n.main(common + ["--out", str(split), "--resume"]), 0)
            for directory in split.glob("segments/*/train_*"):
                self.assertEqual(audit(directory), 0)
        one, two = n.load_bundle(full / "latest.pt"), n.load_bundle(split / "latest.pt")
        def same(a, b):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for k in a:
                    same(a[k], b[k])
            elif isinstance(a, (tuple, list)):
                self.assertEqual(len(a), len(b))
                for x, y in zip(a, b):
                    same(x, y)
            else:
                self.assertEqual(a, b)
        same(one["policies"], two["policies"])
        self.assertEqual(one["validation"], two["validation"])
        self.assertTrue(torch.equal(one["rng"][2], two["rng"][2]))
        trace = next(split.glob("segments/*/train_*/trace.jsonl"))
        text = trace.read_text().splitlines()
        trace.write_text("\n".join(text[:-1]) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit(trace.parent), 1)

    def test_ndtvs_exact_resume(self):
        self.check_resume("ndtvs")

    def test_hppo_resume_pending_frame_buffer(self):
        self.check_resume("hppo_rsu")

    def test_reject_different_physical_evaluations(self):
        for name, bandwidth in (("ndtvs", 3e6), ("hppo_rsu", 20e6)):
            n.atomic(self.root / name / "evaluation.json", {
                "algorithm": name, "config": {"rsu_total_bandwidth_hz": bandwidth},
                "offset": 2_000_000, "scenario_seed": 2026, "qoe_weights": [1, .5, 2],
                "per_episode": []})
        with self.assertRaisesRegex(ValueError, "Physical/control"):
            compare([self.root / "ndtvs", self.root / "hppo_rsu"], self.root / "comparison")

    def test_explicit_budget_extension(self):
        from extend_training import extend
        source, extended, full = self.root / "source", self.root / "extended", self.root / "full"
        common = ["train", "--algorithm", "ndtvs", "--config", str(self.config),
                  "--val-every", "1", "--val-episodes", "1", "--trace-every", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(n.main(common + ["--out", str(source), "--episodes", "2"]), 0)
            original = (source / "latest.pt").read_bytes()
            extend(source, extended, 4)
            self.assertEqual(original, (source / "latest.pt").read_bytes())
            self.assertEqual(n.main(common + ["--out", str(extended), "--episodes", "4", "--resume"]), 0)
            self.assertEqual(n.main(common + ["--out", str(full), "--episodes", "4"]), 0)
        one, two = n.load_bundle(extended / "latest.pt"), n.load_bundle(full / "latest.pt")
        for k, v in one["policies"][0]["model"].items():
            self.assertTrue(torch.equal(v, two["policies"][0]["model"][k]))
        self.assertTrue(torch.equal(one["rng"][2], two["rng"][2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
