from __future__ import annotations

import unittest

from run.p3_snr_sweep import build_tasks, shifted_noise_psd_w_hz


class P3SNRSweepTests(unittest.TestCase):
    def test_30_db_is_exact_original_channel(self) -> None:
        base = 2e-20
        self.assertAlmostEqual(
            shifted_noise_psd_w_hz(base, 30.0, 30.0),
            base,
        )

    def test_20_and_40_db_apply_symmetric_10_db_shift(self) -> None:
        base = 2e-20
        self.assertAlmostEqual(
            shifted_noise_psd_w_hz(base, 20.0, 30.0),
            10.0 * base,
        )
        self.assertAlmostEqual(
            shifted_noise_psd_w_hz(base, 40.0, 30.0),
            0.1 * base,
        )

    def test_default_full_sweep_has_100_tasks(self) -> None:
        tasks = build_tasks(
            (20.0, 25.0, 30.0, 35.0, 40.0),
            (120026, 120027, 120028, 120029, 120030),
            ("proposed", "always_hire", "slow_ppo", "rsu_only"),
        )
        self.assertEqual(len(tasks), 100)


if __name__ == "__main__":
    unittest.main()
