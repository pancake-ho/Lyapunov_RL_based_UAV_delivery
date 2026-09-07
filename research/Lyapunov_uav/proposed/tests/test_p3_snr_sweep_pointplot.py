from __future__ import annotations

import unittest

import numpy as np

from run.p3_snr_sweep_aggregate import (
    DEFAULT_SHOW_CI,
    POLICY_LINESTYLES,
    _series,
    mean_ci95,
)


class P3SNRSweepPaperStyleTests(unittest.TestCase):
    def test_default_paper_style_hides_error_bars(self) -> None:
        self.assertFalse(DEFAULT_SHOW_CI)

    def test_every_policy_has_a_visible_line_style(self) -> None:
        for policy in (
            "proposed",
            "always_hire",
            "slow_ppo",
            "rsu_only",
        ):
            self.assertIn(policy, POLICY_LINESTYLES)
            self.assertNotEqual(
                POLICY_LINESTYLES[policy],
                "none",
            )

    def test_series_keeps_exact_snr_coordinates(self) -> None:
        rows = [
            {
                "policy": "proposed",
                "snr_db": 30.0,
                "metric_mean": 3.0,
                "metric_ci95": 0.3,
            },
            {
                "policy": "proposed",
                "snr_db": 20.0,
                "metric_mean": 1.0,
                "metric_ci95": 0.1,
            },
            {
                "policy": "proposed",
                "snr_db": 25.0,
                "metric_mean": 2.0,
                "metric_ci95": 0.2,
            },
        ]

        x, y, ci = _series(
            rows,
            "proposed",
            "metric",
        )

        np.testing.assert_array_equal(
            x,
            np.asarray([20.0, 25.0, 30.0]),
        )
        np.testing.assert_array_equal(
            y,
            np.asarray([1.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            ci,
            np.asarray([0.1, 0.2, 0.3]),
        )

    def test_five_seed_student_t_ci_is_preserved(self) -> None:
        mean, ci, n = mean_ci95(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )

        self.assertEqual(n, 5)
        self.assertAlmostEqual(mean, 3.0)
        self.assertGreater(ci, 0.0)


if __name__ == "__main__":
    unittest.main()
