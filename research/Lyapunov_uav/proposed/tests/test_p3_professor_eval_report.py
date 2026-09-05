from __future__ import annotations

import json
import unittest

from run.p3_professor_eval_report import (
    average_video_bitrate_mbps,
    reconstruct_return_distances,
)


class P3ProfessorEvalReportTests(unittest.TestCase):
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
