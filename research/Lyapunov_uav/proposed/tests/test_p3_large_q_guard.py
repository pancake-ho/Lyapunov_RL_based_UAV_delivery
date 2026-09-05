from __future__ import annotations

import unittest

from agent.P3.exact_fast_controller import ExactFastController
from config_p3 import P3Config
from env.p3.environment import update_playback_queue


class P3LargeQAdmissibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = P3Config()
        self.fast = ExactFastController(self.cfg)

    def test_queue_safe_chunk_cap_near_qe(self) -> None:
        # Q=Qe-1, b=1 -> after departure Q=Qe-2, so at most 2 chunks.
        self.assertEqual(self.fast.max_queue_admissible_chunks(1.0), 2)
        # Q=Qe -> after one departure at most one chunk can refill.
        self.assertEqual(self.fast.max_queue_admissible_chunks(0.0), 1)
        # Recovery from an already-violated state must not add new chunks.
        self.assertEqual(self.fast.max_queue_admissible_chunks(-1.0), 0)

    def test_queue_update_never_exceeds_qe_for_admissible_cap(self) -> None:
        for z in (100.0, 10.0, 2.0, 1.0, 0.0, -1.0):
            q_before = max(0.0, self.cfg.large_queue_level - z)
            delivered = self.fast.max_queue_admissible_chunks(z)
            q_after, _, _, _ = update_playback_queue(
                q_before,
                delivered,
                self.cfg,
            )
            if q_before <= self.cfg.large_queue_level:
                self.assertLessEqual(
                    q_after,
                    self.cfg.large_queue_level + 1e-9,
                )
            else:
                self.assertLess(q_after, q_before + 1e-9)

    def test_rsu_solver_respects_queue_cap(self) -> None:
        z = 0.0
        option = self.fast.solve_rsu(
            z=z,
            horizontal_distance_m=0.0,
            fading=10.0,
        )
        self.assertLessEqual(
            option.chunks,
            self.fast.max_queue_admissible_chunks(z),
        )

    def test_uav_option_set_respects_queue_cap(self) -> None:
        z = 1.0
        options = self.fast.feasible_uav_options(
            z=z,
            horizontal_distance_m=0.0,
            fading=10.0,
            individual_power_cap_w=self.cfg.uav_max_total_power_w,
        )
        cap = self.fast.max_queue_admissible_chunks(z)
        self.assertTrue(all(option.chunks <= cap for option in options))


if __name__ == "__main__":
    unittest.main()
