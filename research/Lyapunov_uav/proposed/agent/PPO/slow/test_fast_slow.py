from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


PROPOSED_ROOT = Path(__file__).resolve().parents[3]
if str(PROPOSED_ROOT) not in sys.path:
    sys.path.insert(0, str(PROPOSED_ROOT))


from config import EnvConfig  # noqa: E402
from env.env import Env  # noqa: E402
from agent.PPO.slow.slow_matching import (  # noqa: E402
    PairEdge,
    select_dpp_max_weight_matching,
    solve_max_weight_b_matching,
)


class SlowMaximumWeightMatchingTest(
    unittest.TestCase
):
    def test_b_matching_respects_capacities_and_unique_users(
        self,
    ) -> None:
        edges = (
            PairEdge(0, 0, 10.0, 90.0),
            PairEdge(0, 1, 9.0, 91.0),
            PairEdge(0, 2, 1.0, 99.0),
            PairEdge(1, 0, 8.0, 92.0),
            PairEdge(1, 2, 7.0, 93.0),
            PairEdge(1, 3, 6.0, 94.0),
        )

        selected = solve_max_weight_b_matching(
            edges=edges,
            provider_count=2,
            user_count=4,
            capacities=(2, 1),
            min_weight=0.0,
        )

        provider_load = np.zeros(
            2,
            dtype=np.int32,
        )
        users = []

        for edge in selected:
            provider_load[
                int(edge.provider)
            ] += 1
            users.append(
                int(edge.user)
            )

        self.assertTrue(
            np.all(
                provider_load
                <= np.asarray(
                    [2, 1],
                    dtype=np.int32,
                )
            )
        )
        self.assertEqual(
            len(users),
            len(set(users)),
        )

        # Optimal value = 10 + 9 + 7 = 26.
        self.assertAlmostEqual(
            sum(
                float(edge.weight)
                for edge in selected
            ),
            26.0,
            places=6,
        )

    @staticmethod
    def _small_env() -> Env:
        base = EnvConfig()
        cfg = replace(
            base,
            num_user=4,
            num_rsu=2,
            num_uav=2,
            rsu_capacity=1,
            uav_user_cap=2,
            slow_T=2,
            episode_slots=2,
            mobility_mode="fsmc",
            move_prob=0.0,
            uav_hiring_cost=5.0,
            hire_weight=1.0,
            seed=2026,
        )

        env = Env(cfg)
        env.reset()

        env.user_region = np.asarray(
            [0, 0, 1, 1],
            dtype=np.int32,
        )
        env.requested_content = np.asarray(
            [0, 0, 0, 1],
            dtype=np.int32,
        )
        env.uav_cached_content = np.asarray(
            [0, 0],
            dtype=np.int32,
        )
        env._refresh_link_distances()
        return env

    def test_sequential_rsu_priority_residual_uav_matching(
        self,
    ) -> None:
        env = self._small_env()

        rsu_gain = {
            (0, 0): 10.0,
            (0, 1): 7.0,
            (1, 2): 8.0,
            (1, 3): 3.0,
        }
        uav_gain = {
            (0, 1): 9.0,
        }

        def evaluate(
            actions: Sequence[
                Dict[str, np.ndarray]
            ],
        ) -> list[float]:
            scores = []
            for action in actions:
                y = np.asarray(
                    action["rsu_scheduling"],
                    dtype=np.int32,
                )
                mu = np.asarray(
                    action["uav_hiring"],
                    dtype=np.int32,
                )
                phi = np.asarray(
                    action["uav_scheduling"],
                    dtype=np.int32,
                )

                cost = 100.0

                for provider, user in zip(
                    *np.nonzero(y)
                ):
                    cost -= rsu_gain.get(
                        (
                            int(provider),
                            int(user),
                        ),
                        0.0,
                    )

                for provider, user in zip(
                    *np.nonzero(phi)
                ):
                    cost -= uav_gain.get(
                        (
                            int(provider),
                            int(user),
                        ),
                        0.0,
                    )

                cost += (
                    5.0
                    * float(mu.sum())
                )
                scores.append(
                    float(cost)
                )
            return scores

        selected = (
            select_dpp_max_weight_matching(
                env=env,
                evaluate=evaluate,
                min_edge_weight=0.0,
                forbid_empty_hiring=True,
            )
        )

        action = selected.action
        y = action["rsu_scheduling"]
        mu = action["uav_hiring"]
        phi = action["uav_scheduling"]

        expected_y = np.zeros(
            (2, 4),
            dtype=np.int32,
        )
        expected_y[0, 0] = 1
        expected_y[1, 2] = 1

        expected_phi = np.zeros(
            (2, 4),
            dtype=np.int32,
        )
        expected_phi[0, 1] = 1

        np.testing.assert_array_equal(
            y,
            expected_y,
        )
        np.testing.assert_array_equal(
            mu,
            np.asarray(
                [1, 0],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            phi,
            expected_phi,
        )

        # 100 - 10 - 8 - 9 + 5 = 78.
        self.assertAlmostEqual(
            selected.predicted_round_cost,
            78.0,
            places=6,
        )
        self.assertEqual(
            selected.chosen_stage,
            "rsu_uav_matching",
        )

        applied = env.apply_slow_action(
            action
        )
        np.testing.assert_array_equal(
            applied.rsu_scheduling,
            expected_y,
        )
        np.testing.assert_array_equal(
            applied.uav_hiring,
            np.asarray(
                [1, 0],
                dtype=np.int32,
            ),
        )
        np.testing.assert_array_equal(
            applied.uav_scheduling,
            expected_phi,
        )

    def test_env_rejects_empty_hiring(
        self,
    ) -> None:
        env = self._small_env()
        action = {
            "rsu_scheduling": np.zeros(
                (2, 4),
                dtype=np.int32,
            ),
            "uav_hiring": np.asarray(
                [1, 0],
                dtype=np.int32,
            ),
            "uav_scheduling": np.zeros(
                (2, 4),
                dtype=np.int32,
            ),
        }

        with self.assertRaises(
            ValueError
        ):
            env.apply_slow_action(
                action
            )


if __name__ == "__main__":
    unittest.main()
