"""Portable CPU integration; tiny runs are software checks, not research results."""
import ast
import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

import baseline.NDTVS.api as c
from baseline.UAVRTV.common.settings import load, common_config, experiment_spec
from baseline.UAVRTV.common.checkpoint import read, policy_digest
from baseline.UAVRTV.environment.shared import SharedUAVRTVEnv
from baseline.UAVRTV.models.sac import SACAgent
from baseline.UAVRTV.rewards.paper import components
from baseline.UAVRTV.training.rollout import run_episode
from baseline.UAVRTV.training import engine
from baseline.UAVRTV.evaluation.preflight import run as preflight
from baseline.UAVRTV.evaluation.sweep import sweep
from baseline.UAVRTV.evaluation.audit import audit
from baseline.UAVRTV.plot.learning import plot
from baseline.UAVRTV import submit
from baseline.UAVRTV import cli
from baseline.NDTVS.evaluation.scenario import PairingObserver
from hppo.env import P3HierarchicalEnv
from baseline.NDTVS.environment.rsu import RSUEnv
from env.p3.types import RegionAction


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(tempfile.mkdtemp(prefix="uavrtv_test_"))
        cls.cfg = replace(c.HPPOConfig(), num_regions=2, users_per_region=4, num_frames=2,
            frame_slots=3, rsu_capacity=3, uav_capacity=2, rsu_total_bandwidth_hz=3e6,
            hidden_dims=(16, 16), device="cpu", episode_offset=700)
        c.atomic(cls.root / "source.json", dict(config=asdict(cls.cfg)))
        c.atomic(cls.root / "runtime.json", dict(code_sha256={k.removeprefix("proposed/"): v
            for k, v in c.source_hashes().items() if k.startswith("proposed/")}))

    def settings(self, name):
        s = load()
        s.SOURCE_CONFIG, s.SOURCE_RUNTIME = self.root / "source.json", self.root / "runtime.json"
        s.DEVICE, s.TRAIN_EPISODES, s.OUT = "cpu", 2, self.root / name
        s.HIDDEN_DIMS, s.BATCH_SIZE, s.BUFFER_SIZE = (16, 16), 4, 64
        s.START_TRANSITIONS, s.VAL_EVERY, s.VAL_EPISODES = 0, 2, 1
        s.WALLTIME_SECONDS, s.RESERVE_SECONDS = 3600, 0
        s.TEST_EPISODES, s.TEST_SEEDS, s.SNR_OFFSETS_DB = 1, (2026, 2027), (-5, 5)
        s.EVAL_OUT = self.root / (name + "_eval")
        return s

    def test_reward_units_and_delivery(self):
        s = self.settings("reward")
        cfg, _ = common_config(s)
        users = [dict(user=0, q_before=.5, delivered=3, req_quality=3, last_quality_before=0),
                 dict(user=1, q_before=0., delivered=0, req_quality=3, last_quality_before=2)]
        r = components(dict(users=users, hover_energy_j=597., communication_energy_j=3.), cfg, s,
                       dict(relocation_energy_j=10000., hiring_cost_weighted=5.))
        self.assertEqual(r["quality_gain"], 1.)
        self.assertEqual(r["switch_penalty"], 3.5)
        self.assertEqual(r["rebuffer_penalty"], 15.)
        self.assertEqual(r["energy_penalty"], 106.)
        self.assertEqual(r["hiring_penalty"], 5.)
        self.assertEqual(r["raw_reward"], -128.5)
        users[0]["delivered"] = 0
        r = components(dict(users=users, hover_energy_j=0., communication_energy_j=0.), cfg, s)
        self.assertEqual(r["quality_gain"], 0.)
        self.assertEqual(r["switch_penalty"], 0.)
        self.assertEqual(r["hiring_penalty"], 0.)

    def test_no_csi_region_rules_atomic_delivery(self):
        cfg, _ = common_config(self.settings("csi"))
        env = SharedUAVRTVEnv(cfg)
        env.reset(0)
        env.prepare_frame()
        for m in env.regions:
            members = env.region_users[m]
            for i, u in enumerate(members):
                env.state.queue[u] = float(len(members) - i)
            rsu, uav = env.priorities(m)
            order = sorted(members, key=lambda u: (env.state.queue[u], u))
            self.assertEqual(rsu, tuple(order[:3]))
            self.assertEqual(uav, tuple(order[3:5]))
        actions = {m: np.ones(env.act_dim, np.float32) for m in env.regions}
        info = env.start_from_sac(actions)
        original = {m: (env.observation(m, False), env.requests(m, actions[m])) for m in env.regions}
        env.trace.rsu_fading[:] = 1e-30
        env.trace.uav_fading[:] = 1e-30
        for m in env.regions:
            obs, mask = env.observation(m, False)
            np.testing.assert_array_equal(obs, original[m][0][0])
            np.testing.assert_array_equal(mask, original[m][0][1])
            np.testing.assert_array_equal(env.requests(m, actions[m]), original[m][1])
            self.assertTrue(set(info["regions"][m]["executed_rsu_users"]) <= set(env.region_users[m]))
        result = env.step_slot({m: env.requests(m, actions[m]) for m in env.regions})
        self.assertTrue(any(u["req_chunks"] > 0 for rg in result.info["regions"].values() for u in rg["users"]))
        self.assertTrue(all(u["delivered"] == 0 for rg in result.info["regions"].values() for u in rg["users"]))

    def test_shared_scenario_fingerprint_and_energy_accounting(self):
        s = self.settings("pairing")
        cfg, _ = common_config(s)
        ObservedRSU = type("ObservedRSU", (PairingObserver, RSUEnv), {})
        rsu = ObservedRSU(cfg)
        rsu.reset(47)
        for _ in range(cfg.num_frames):
            rsu.prepare_frame()
            raw = {m: np.zeros(cfg.num_users, np.int64) for m in rsu.regions}
            completed = {m: RegionAction(m, 0, -1, (), ()) for m in rsu.regions}
            rsu.begin_frame(raw, completed)
            for _ in range(cfg.frame_slots):
                rsu.step_slot({m: np.zeros(3 * cfg.num_users, np.int64) for m in rsu.regions})
        for profile in ("nohire", "hired_max"):
            directory = self.root / ("pair_" + profile)
            row = run_episode(cfg, s, None, 47, directory, trace=True, profile=profile, rng=np.random.default_rng(9))
            self.assertEqual(row["scenario_sha256"], rsu.episode_summary()["scenario_sha256"])
            self.assertTrue(audit(directory)["valid"])
            self.assertEqual(row["hiring_penalty_total"], row["hiring_cost_total"])
            self.assertAlmostEqual(row["energy_penalty_total"], s.VARSIGMA * row["energy_consumed_j"])
            self.assertEqual(row["hired_uav_frames"], 0 if profile == "nohire" else cfg.num_regions * cfg.num_frames)
        depleted = replace(cfg, initial_battery_j=cfg.reserve_battery_j + cfg.frame_slots * cfg.hover_energy_per_slot_j - 1.)
        env = SharedUAVRTVEnv(depleted)
        env.reset(0)
        env.prepare_frame()
        info = env.start_from_sac({m: np.ones(env.act_dim, np.float32) for m in env.regions})
        self.assertTrue(all(rg["executed_hire"] == 0 for rg in info["regions"].values()))
        constrained = replace(cfg, initial_battery_j=cfg.reserve_battery_j + cfg.frame_slots * cfg.hover_energy_per_slot_j + 3.)
        row = run_episode(constrained, s, None, 47, self.root / "constrained", trace=True,
                          profile="hired_max", rng=np.random.default_rng(1))
        self.assertTrue(audit(self.root / "constrained")["valid"])
        self.assertEqual(row["hired_uav_frames"], cfg.num_regions)
        self.assertEqual(row["reserve_violations"], 0)

    def test_state_only_masks(self):
        s = self.settings("masks")
        cfg, _ = common_config(s)
        env = SharedUAVRTVEnv(cfg)
        env.reset(0)
        env.prepare_frame()
        obs, mask = env.observation(0, True)
        agent = SACAgent(env.obs_dim, env.act_dim, s)
        draw = agent.act([obs], [mask], deterministic=True)[0]
        self.assertTrue(np.all(draw[mask == 0] == 0))
        before = c.rng_state()[2]
        agent.act([obs], [mask], deterministic=True)
        torch.testing.assert_close(before, c.rng_state()[2], rtol=0, atol=0)
        env.start_from_sac({m: np.full(env.act_dim, -1., np.float32) for m in env.regions})
        for m in env.regions:
            _, mask = env.observation(m, False)
            self.assertEqual(float(mask.sum()), 0.)

    def test_exact_learning_resume_validation_recovery_eval(self):
        uninterrupted = self.settings("uninterrupted")
        cfg, _ = common_config(uninterrupted)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(engine.train(uninterrupted, cfg), 0)
        paused = self.settings("paused")
        paused.MAX_NEW_EPISODES = 1
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(engine.train(paused, cfg), 75)
        self.assertEqual(read(paused.OUT / "latest.pt")["next_episode"], 1)
        paused.MAX_NEW_EPISODES = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(engine.train(paused, cfg), 0)
        a, b = read(uninterrupted.OUT / "latest.pt"), read(paused.OUT / "latest.pt")
        shape = SharedUAVRTVEnv(cfg)
        agents = [SACAgent(shape.obs_dim, shape.act_dim, paused) for _ in range(2)]
        for agent, saved in zip(agents, (a, b)):
            agent.restore(saved["agent"], training=True)
        self.assertEqual(policy_digest(agents[0]), policy_digest(agents[1]))
        self.assertEqual(a["sampler_rng"], b["sampler_rng"])
        for key in a["replay"]["data"]:
            np.testing.assert_array_equal(a["replay"]["data"][key], b["replay"]["data"][key])
        torch.testing.assert_close(a["rng"][2], b["rng"][2], rtol=0, atol=0)
        recovery = self.settings("validation_recovery")
        with contextlib.redirect_stdout(io.StringIO()), patch.object(engine, "validation", side_effect=RuntimeError("injected validation interruption")):
            with self.assertRaises(RuntimeError):
                engine.train(recovery, cfg)
        self.assertEqual(read(recovery.OUT / "latest.pt")["next_episode"], 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(engine.train(recovery, cfg), 0)
        self.assertEqual(len(read(recovery.OUT / "latest.pt")["validation"]), 1)
        before = (paused.OUT / "best.pt").read_bytes()
        paused.MAX_NEW_EPISODES = 1
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep(paused), 75)
        paused.MAX_NEW_EPISODES = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sweep(paused), 0)
            self.assertEqual(sweep(paused), 0)
        state = json.loads((paused.EVAL_OUT / "state.json").read_text())
        self.assertEqual(len(state["rows"]), 4)
        self.assertEqual((paused.OUT / "best.pt").read_bytes(), before)
        self.assertEqual(plot(paused), 0)
        from PIL import Image
        with Image.open(paused.OUT / "plots/learning.png") as im:
            im.verify()
        corrupted = copy.deepcopy(b)
        corrupted["source_sha256"][next(iter(corrupted["source_sha256"]))] = "0" * 64
        torch.save(corrupted, self.root / "bad_source.pt")
        with self.assertRaises(ValueError):
            read(self.root / "bad_source.pt")
        torch.save(dict(format="legacy_sac"), self.root / "old.pt")
        with self.assertRaises(ValueError):
            read(self.root / "old.pt")
        sidecar = paused.EVAL_OUT / state["rows"][0]["per_user_file"]
        sidecar.write_bytes(sidecar.read_bytes() + b" ")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            sweep(paused)

    def test_python310_and_standard_library_submission(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):
            if "runs" not in path.parts and "slurm_logs" not in path.parts:
                ast.parse(path.read_text(), feature_version=(3, 10))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(submit.main(["--mode", "train", "--dry-run"]), 0)
        self.assertIn("/data/surt321/anaconda3/envs/lab/bin/python", stream.getvalue())
        self.assertNotIn("export ", stream.getvalue())

    def test_smoke_then_fresh_training(self):
        s = self.settings("cli_sequence")
        with patch.object(cli, "load", return_value=s), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["smoke"]), 0)
            smoke = s.OUT.with_name(s.OUT.name + "_smoke")
            self.assertTrue((smoke / "latest.pt").is_file())
            self.assertFalse(s.OUT.exists())
            self.assertEqual(cli.main(["train"]), 0)
        self.assertEqual(read(s.OUT / "latest.pt")["next_episode"], 2)
        self.assertEqual(read(smoke / "latest.pt")["next_episode"], 2)


if __name__ == "__main__":
    unittest.main()
