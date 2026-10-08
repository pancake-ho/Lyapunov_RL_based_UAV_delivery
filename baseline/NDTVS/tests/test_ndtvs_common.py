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
import baseline.NDTVS.api as n
from baseline.NDTVS.analysis.cli import audit, compare


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

    def test_paper_reward_sequence_boundaries_and_cumulative_stall(self):
        from baseline.NDTVS.rewards.qoe import QoEHistory
        h = QoEHistory(self.cfg)
        def record(d, k, q=1, req=None):
            return {"user": 0, "delivered": d, "req_quality": k,
                    "req_chunks": d if req is None else req, "q_before": q}
        # First request fails: it sets P*, but creates no received segment.
        a = h.update(record(0, 3, q=0, req=1))
        self.assertEqual(a["paper_vq_db"], 0)
        self.assertAlmostEqual(a["paper_qoe"], -7.64)
        self.assertEqual(a["first_requested_psnr_db"], 41.64)
        b = h.update(record(1, 0))
        self.assertAlmostEqual(b["paper_qoe"], 34 - 7.64)
        self.assertEqual(b["paper_qv_index"], 0)
        # Received [0,3,3,3]: Vq excludes the first segment; Qv=3/3.
        c = h.update(record(3, 3))
        self.assertAlmostEqual(c["paper_vq_db"], 41.64)
        self.assertEqual(c["paper_qv_index"], 1)
        self.assertAlmostEqual(c["paper_qoe"], 41.64 - 7.64/3 - 7.64)
        # Failed request changes neither segment count nor switching history.
        d = h.update(record(0, 1, q=0, req=2))
        self.assertEqual(d["received_segments"], 4)
        self.assertAlmostEqual(d["paper_qoe"], c["paper_qoe"] - 7.64)
        e = h.update(record(0, 0, q=1, req=0))
        self.assertEqual(e["paper_qoe"], d["paper_qoe"])
        fresh = QoEHistory(self.cfg)
        self.assertEqual(fresh.update(record(1, 3))["rebuffer_s_cumulative"], 0)

    def test_signed_quality_gap_and_multisegment_first_batch(self):
        from baseline.NDTVS.rewards.qoe import QoEHistory
        h = QoEHistory(self.cfg)
        for d, k in ((1, 0), (2, 3), (1, 1)):
            a = h.update({"user": 0, "delivered": d, "req_quality": k,
                          "req_chunks": d, "q_before": 1})
        # [0,3,3,1], tail mean=(41.64+41.64+36.64)/3, switch=(3+0+2)/3.
        self.assertAlmostEqual(a["paper_vq_db"], (41.64+41.64+36.64)/3)
        self.assertAlmostEqual(a["paper_qv_index"], 5/3)
        h = QoEHistory(self.cfg)
        a = h.update({"user": 1, "delivered": 3, "req_quality": 2,
                      "req_chunks": 3, "q_before": .25})
        self.assertAlmostEqual(a["paper_vq_db"], 39.11)
        self.assertEqual(a["paper_qv_index"], 0)
        self.assertAlmostEqual(a["rebuffer_s_cumulative"], .75)

    def test_observer_updates_once_and_retains_user_history_across_regions(self):
        cfg = replace(self.cfg, write_jsonl_trace=False, write_human_debug_log=False)
        log = n.QoELogger(cfg, self.root)
        try:
            def slot(frame, index, reverse=False):
                users = [{"user": i, "delivered": 1 if i == 0 else 0,
                          "req_chunks": 1 if i == 0 else 0, "req_quality": 0,
                          "q_before": 0, "stall": 1, "transmission_failed": False}
                         for i in range(cfg.num_users)]
                split = cfg.num_users // 2
                chunks = [users[:split], users[split:]]
                if reverse:
                    chunks.reverse()
                return {"episode": 0, "frame": frame, "slot_in_frame": index,
                        "regions": {m: {"users": group} for m, group in enumerate(chunks)}}
            info = slot(0, 0)
            log.observe_slot(info)
            log.observe_slot(info)
            self.assertEqual(log.history.segments[0], 1)
            self.assertEqual(log.user_slots, cfg.num_users)
            log.observe_slot(slot(1, 0, True))
            self.assertEqual(log.history.segments[0], 2)
            self.assertEqual(log.history.rebuffer_s[0], 2)
        finally:
            log.close()

    def test_refuse_old_ndtvs_checkpoint(self):
        old = {"spec": {"algorithm": "ndtvs", "version": "ndtvs-common-v1",
                        "source_sha256": n.source_hashes()}}
        with self.assertRaisesRegex(ValueError, "retrain from scratch"):
            n.verify_checkpoint_source(old, "ndtvs")

    def test_hppo_legacy_migration_requires_identical_physics(self):
        hashes = {k: v for k, v in n.source_hashes().items()
                  if not k.startswith("baseline/NDTVS/")}
        hashes["ndtvs_common.py"] = n.LEGACY_ADAPTER_SHA256
        old = {"spec": {"algorithm": "hppo_rsu", "version": "ndtvs-common-v1",
                        "source_sha256": hashes, "config": asdict(self.cfg)}}
        self.assertEqual(n.verify_checkpoint_source(old, "hppo_rsu"),
                         "verified-hppo-v1-observer-migration")
        hashes["proposed/hppo/env.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "physical/control"):
            n.verify_checkpoint_source(old, "hppo_rsu")

    def test_pooled_quality_ratio_not_mean_of_ratios(self):
        from baseline.NDTVS.plot.plot_snr_sweep import point
        rows = [{"received_segments_total": 1, "average_quality_utility": .5},
                {"received_segments_total": 3, "average_quality_utility": 1}]
        self.assertEqual(point(rows, "average_quality_utility"), .875)
        self.assertEqual(point(rows, "average_quality_utility", [0, 0]), .5)
        self.assertIsNone(point([{"received_segments_total": 0, "average_quality_utility": 0}],
                                "average_quality_utility"))

    def test_reviewed_config_defaults_require_complete_saved_config(self):
        expected, current = {}, {}
        for key, pair in n.REVIEWED_CONFIG_DEFAULT_HASHES.items():
            expected[key], current[key] = pair
        self.assertEqual(n.verify_shared_sources(expected, current, asdict(self.cfg)),
                         "reviewed-explicit-config-default-migration")
        with self.assertRaisesRegex(ValueError, "complete explicit config"):
            n.verify_shared_sources(expected, current, {"lyapunov_v": 60})
        current["proposed/config_p3.py"] = "unreviewed-change"
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            n.verify_shared_sources(expected, current, asdict(self.cfg))

    def test_paired_smoke_eval_sweep_report_and_resume(self):
        import baseline.NDTVS.evaluation.cli as s
        from baseline.NDTVS.plot.plot_snr_sweep import report
        ndt, rsu = self.root / "ndt", self.root / "rsu"
        common = ["train", "--config", str(self.config), "--episodes", "2",
                  "--val-every", "1", "--val-episodes", "1", "--trace-every", "1"]
        with contextlib.redirect_stdout(io.StringIO()):
            for name, out in (("ndtvs", ndt), ("hppo_rsu", rsu)):
                self.assertEqual(n.main(common + ["--algorithm", name, "--out", str(out)]), 0)
            frame, slot = self.root / "frame.pt", self.root / "slot.pt"
            for a, path in zip(n.make_agents(self.cfg, "proposed"), (frame, slot)):
                a.save(path, {"pair_id": "unit-test-fixture", "episode": 0})
            runtime = self.root / "runtime.json"
            n.atomic(runtime, {"code_sha256": {k.removeprefix("proposed/"): v
                for k, v in n.source_hashes().items() if k.startswith("proposed/")}})
            output = self.root / "comparison"
            args = ["--out", str(output), "--config", str(self.config), "--device", "cpu",
                    "--ndtvs-checkpoint", str(ndt / "best.pt"), "--rsu-checkpoint", str(rsu / "best.pt"),
                    "--frame-checkpoint", str(frame), "--slot-checkpoint", str(slot), "--episodes", "2"]
            self.assertEqual(s.main(args + ["--mode", "smoke", "--offset", "6000000"]), 0)
            self.assertEqual(s.main(args + ["--mode", "eval", "--offset", "7000000"]), 0)
            self.assertEqual(s.main(args + ["--mode", "sweep", "--offset", "7000000"]), 0)
            self.assertEqual(s.main(args + ["--mode", "sweep", "--offset", "7000000", "--resume"]), 0)
            self.assertEqual(report(output / "sweep", output / "report", 20), 0)
        state = s.verify_saved(output / "sweep")
        self.assertEqual(len(state["cells"]), 15)
        self.assertTrue((output / "report/performance.png").exists())
        for cell in state["cells"].values():
            self.assertEqual(len(cell["rows"]), 2)

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
                "offset": 2_000_000, "scenario_seed": 2026, "qoe_weights": list(n.QOE_WEIGHTS), "qoe_definition": n.reward_spec(),
                "per_episode": []})
        with self.assertRaisesRegex(ValueError, "Physical/control"):
            compare([self.root / "ndtvs", self.root / "hppo_rsu"], self.root / "comparison")

    def test_explicit_budget_extension(self):
        from baseline.NDTVS.training.extend_training import extend
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
