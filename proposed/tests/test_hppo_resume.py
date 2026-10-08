"""Integration tests: original loop, interruption, RNG/buffers and legacy recovery."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from config_hppo import HPPOConfig
from hppo.logger import jsonable
from hppo.resume import load_source
from hppo.verify_trace import verify


class ResumeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.seed = cls.root / 'config'
        cls.seed.mkdir()
        cfg = HPPOConfig(num_regions=2, users_per_region=3, num_frames=3,
                         frame_slots=3, rollout_scenarios=2, train_episodes=4,
                         hidden_dims=(16, 16), ppo_minibatch_size=16,
                         slot_update_every_frames=2, device='cpu',
                         console_log_every_slots=0, save_every_episodes=2)
        cls.cfg = cfg
        (cls.seed / 'resolved_config.json').write_text(json.dumps(
            {'args': {'checkpoint_dir': 'checkpoints'}, 'config': jsonable(asdict(cfg))}))
        cls.run_cmd('full', cls.seed, '--fresh')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def command(cls, name, source, *extra):
        return [sys.executable, '-m', 'hppo.resume', '--source-run', str(source),
                '--output-dir', str(cls.root), '--run-name', name, '--device', 'cpu', *extra]

    @classmethod
    def run_cmd(cls, name, source, *extra):
        r = subprocess.run(cls.command(name, source, *extra), capture_output=True, text=True)
        if r.returncode not in (0, 75):
            raise AssertionError(r.stdout + r.stderr)
        return r.returncode

    def bundle(self, name):
        return torch.load(self.root / name / 'checkpoints/resume_latest.pt', weights_only=False)

    def same(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a: self.same(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.same(x, y)
        else:
            self.assertEqual(a, b)

    def same_final(self, name):
        full, actual = self.bundle('full'), self.bundle(name)
        for k in ('frame', 'slot', 'rng', 'next_episode', 'dual'):
            self.same(full[k], actual[k])
        self.assertEqual(sum(verify(self.root / name)[1].values()), 0)

    def test_odd_episode_preserves_pending_slow_buffer(self):
        self.assertEqual(self.run_cmd('odd', self.seed, '--fresh', '--stop-after-episodes', '1'), 75)
        b = self.bundle('odd')
        self.assertEqual(sum(len(v) for v in b['frame']['trajectories'].values()), 6)
        self.assertEqual(sum(verify(self.root / 'odd')[1].values()), 0)
        self.run_cmd('odd_done', self.root / 'odd')
        self.same_final('odd_done')

    def test_actual_usr1_and_kill(self):
        for label, sig, wanted_ep in [('usr1', signal.SIGUSR1, 0), ('killed', signal.SIGKILL, 1)]:
            log = (self.root / (label + '.out')).open('w')
            proc = subprocess.Popen(self.command(label, self.seed, '--fresh'), stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                trace = self.root / label / 'trace.jsonl'
                matched = False
                while time.monotonic() < deadline and proc.poll() is None:
                    if trace.exists():
                        for line in trace.read_text().splitlines():
                            try: row = json.loads(line)
                            except ValueError: continue
                            if row.get('event') == 'frame_start' and row['episode'] == wanted_ep:
                                matched = True
                                break
                    if matched: break
                    time.sleep(.02)
                self.assertTrue(matched)
                proc.send_signal(sig)
                proc.wait(timeout=60)
                self.assertEqual(proc.returncode, 75 if sig == signal.SIGUSR1 else -signal.SIGKILL)
                b = self.bundle(label)
                self.assertEqual(b['next_episode'], 1)
                self.run_cmd(label + '_done', self.root / label)
                self.same_final(label + '_done')
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()
                log.close()

    def test_legacy_and_original_loop(self):
        cli = [sys.executable, '-m', 'hppo.train', '--mode', 'train', '--output-dir', str(self.root), '--run-name', 'original']
        for k, v in asdict(self.cfg).items():
            value = ','.join(str(x) for x in v) if isinstance(v, tuple) else str(v)
            cli.append('--' + k.replace('_', '-') + '=' + value)
        r = subprocess.run(cli, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        full = self.bundle('full')
        for role in ('frame', 'slot'):
            old = torch.load(self.root / 'original' / 'checkpoints' / (role + '_latest.pt'), weights_only=False)
            self.same(old['model'], full[role]['model'])
            self.same(old['optimizer'], full[role]['optimizer'])
        oldrun = self.root / 'legacy'
        shutil.copytree(self.root / 'original', oldrun)
        for role in ('frame', 'slot'):
            (oldrun / 'checkpoints' / (role + '_ep00004.pt')).write_bytes(b'truncated')
        _, info = load_source(oldrun)
        self.assertEqual(info['next_episode'], 2)
        self.assertTrue(info['skipped_pairs'])
        self.run_cmd('legacy_done', oldrun)
        b = self.bundle('legacy_done')
        self.assertEqual(b['next_episode'], 4)
        self.assertEqual(b['frame']['update_count'], 2)
        self.assertEqual(b['slot']['update_count'], 8)
        self.assertEqual(sum(verify(self.root / 'legacy_done')[1].values()), 0)

    def test_budget_pause_before_episode(self):
        self.assertEqual(self.run_cmd('budget', self.seed, '--fresh', '--max-seconds', '.01'), 75)
        self.assertEqual(self.bundle('budget')['next_episode'], 0)
        self.assertEqual(sum(verify(self.root / 'budget')[1].values()), 0)
        self.run_cmd('budget_done', self.root / 'budget')
        self.same_final('budget_done')


if __name__ == '__main__': unittest.main()
