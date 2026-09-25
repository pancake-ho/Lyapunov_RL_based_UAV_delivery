from __future__ import annotations

import json
import unittest
from pathlib import Path

from run.p3_professor_eval_report import (
    average_video_bitrate_mbps,
    frame_path,
    paper_policy,
    raw_policy,
    reconstruct_return_distances,
)
from run.p3_eval_shard import build_eval_tasks


class P3ProfessorEvalReportTests(unittest.TestCase):
    def test_reduced_policy_matrix_has_five_tasks_per_seed(self) -> None:
        tasks = build_eval_tasks((120026,), ("dpp", "always_hire", "rsu_only"))
        self.assertEqual(len(tasks), 5)
        triples = {(task.group, task.policy, task.seed) for task in tasks}
        self.assertIn(("baselines", "dpp", 120026), triples)
        self.assertIn(("baselines", "always_hire", 120026), triples)
        self.assertIn(("baselines", "rsu_only", 120026), triples)
        self.assertIn(("ppo_best", "ppo", 120026), triples)
        self.assertIn(("ppo_latest", "ppo", 120026), triples)

    def test_dpp_is_paper_facing_proposed_but_raw_artifact_key_stays_dpp(self) -> None:
        self.assertEqual(paper_policy("dpp"), "proposed")
        self.assertEqual(raw_policy("proposed"), "dpp")
        path = frame_path(Path("/tmp/eval"), "proposed", 120026)
        self.assertEqual(
            path,
            Path("/tmp/eval/baselines/frames_dpp_seed120026.csv"),
        )

    def test_average_video_bitrate_is_not_network_throughput(self) -> None:
        # 8 users, b=1 chunk/slot. If aggregate network throughput is 8 Mbps
        # and each user receives 1 chunk/slot, mean delivered video bitrate is
        # 1 Mbps, not 8 Mbps.
        value = average_video_bitrate_mbps(
            8.0,
            1.0,
            num_users=8,
            playback_chunks_per_slot=1.0,
        )
        self.assertAlmostEqual(value, 1.0)

    def test_return_distance_reconstruction_matches_logged_events(self) -> None:
        # region_length=400 -> depots at x=200 and 600.
        rows = [
            {
                "frame": "0",
                "return_to_charge_events": "0",
                "actions_json": json.dumps(
                    [
                        {"region": 0, "mu": 1, "target_x": 260.0},
                        {"region": 1, "mu": 1, "target_x": 570.0},
                    ]
                ),
            },
            {
                "frame": "1",
                "return_to_charge_events": "2",
                "actions_json": json.dumps(
                    [
                        {"region": 0, "mu": 0, "target_x": 200.0},
                        {"region": 1, "mu": 0, "target_x": 600.0},
                    ]
                ),
            },
        ]
        distances, count = reconstruct_return_distances(
            rows,
            num_regions=2,
            region_length_m=400.0,
        )
        self.assertEqual(count, 2)
        self.assertEqual(sorted(distances), [30.0, 60.0])

    def test_return_distance_reconstruction_fails_on_inconsistent_log(self) -> None:
        rows = [
            {
                "frame": "0",
                "return_to_charge_events": "1",
                "actions_json": json.dumps(
                    [
                        {"region": 0, "mu": 0, "target_x": 200.0},
                        {"region": 1, "mu": 0, "target_x": 600.0},
                    ]
                ),
            }
        ]
        with self.assertRaises(RuntimeError):
            reconstruct_return_distances(
                rows,
                num_regions=2,
                region_length_m=400.0,
            )


if __name__ == "__main__":
    unittest.main()
