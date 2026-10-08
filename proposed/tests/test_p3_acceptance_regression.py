from __future__ import annotations

import json
import math
import unittest
from dataclasses import asdict

from config_p3 import P3Config
from run.p3_acceptance import json_safe, nonfinite_metric_paths


class P3AcceptanceRegressionTests(unittest.TestCase):
    def test_default_config_is_strict_json_serializable_after_sanitizing(self) -> None:
        payload = json_safe({"config": asdict(P3Config())})
        self.assertIsNone(payload["config"]["distance_bin_edges_m"][-1])
        json.dumps(payload, allow_nan=False)

    def test_runtime_nonfinite_metric_is_still_detected(self) -> None:
        rows = [{"stall_ratio": 0.0, "dpp_cost_per_user_slot": math.inf}]
        invalid = nonfinite_metric_paths(rows, "ppo")
        self.assertEqual(len(invalid), 1)
        self.assertIn("dpp_cost_per_user_slot", invalid[0])

    def test_finite_runtime_metrics_are_accepted(self) -> None:
        rows = [{"stall_ratio": 0.0, "dpp_cost_per_user_slot": -1.0}]
        self.assertEqual(nonfinite_metric_paths(rows, "ppo"), [])


if __name__ == "__main__":
    unittest.main()
