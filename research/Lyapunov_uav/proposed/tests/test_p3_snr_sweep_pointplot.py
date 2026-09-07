from __future__ import annotations

import unittest

from run.p3_snr_sweep_aggregate import (
    DEFAULT_POINT_DODGE_SPAN_DB,
    mean_ci95,
    policy_point_offsets,
)


class P3SNRSweepPointPlotTests(unittest.TestCase):
    def test_four_policy_offsets_are_symmetric(self) -> None:
        policies = (
            "proposed",
            "always_hire",
            "slow_ppo",
            "rsu_only",
        )
        offsets = policy_point_offsets(
            policies,
            span_db=0.9,
        )
        self.assertAlmostEqual(offsets["proposed"], -0.45)
        self.assertAlmostEqual(offsets["always_hire"], -0.15)
        self.assertAlmostEqual(offsets["slow_ppo"], 0.15)
        self.assertAlmostEqual(offsets["rsu_only"], 0.45)
        self.assertAlmostEqual(sum(offsets.values()), 0.0)

    def test_zero_dodge_keeps_exact_snr_coordinates(self) -> None:
        policies = (
            "proposed",
            "always_hire",
            "slow_ppo",
            "rsu_only",
        )
        offsets = policy_point_offsets(
            policies,
            span_db=0.0,
        )
        self.assertEqual(
            offsets,
            {policy: 0.0 for policy in policies},
        )

    def test_default_dodge_is_small_relative_to_snr_spacing(self) -> None:
        self.assertGreaterEqual(DEFAULT_POINT_DODGE_SPAN_DB, 0.0)
        self.assertLess(DEFAULT_POINT_DODGE_SPAN_DB, 5.0)

    def test_five_seed_student_t_ci_remains_used(self) -> None:
        mean, ci, n = mean_ci95(
            [1.0, 2.0, 3.0, 4.0, 5.0]
        )
        self.assertEqual(n, 5)
        self.assertAlmostEqual(mean, 3.0)
        self.assertGreater(ci, 0.0)


if __name__ == "__main__":
    unittest.main()
