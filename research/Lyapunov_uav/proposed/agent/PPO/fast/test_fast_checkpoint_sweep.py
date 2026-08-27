from __future__ import annotations

import tempfile
import unittest

from pathlib import Path
from typing import (
    Dict,
    List,
)

from agent.PPO.fast.fast_checkpoint_sweep import (
    CheckpointSpec,
    _workload_matches,
    aggregate_results,
    discover_checkpoints,
)


def _row(
    *,
    cost_per_scheduled_slot: float,
    scheduled_slots: float = 1000.0,
    unscheduled_slots: float = 2000.0,
    hiring_cost: float = 100.0,
    outage_slots: float = 0.0,
    min_soc: float = 20.0,
    episode: int = 1,
) -> Dict[str, float]:
    fast_cost = (
        float(
            cost_per_scheduled_slot
        )
        * float(
            scheduled_slots
        )
    )

    return {
        "episode":
            float(
                episode
            ),

        "reward":
            -fast_cost,

        "fast_cost":
            fast_cost,

        "fast_cost_per_scheduled_user_slot":
            float(
                cost_per_scheduled_slot
            ),

        "scheduled_stall_rate":
            0.10,

        "quality_per_chunk":
            37.5,

        "quality_degradation_per_chunk":
            4.1,

        "delivery_per_scheduled_user_slot":
            0.8,

        "service_rate":
            0.9,

        "requested_chunks":
            4.5,

        "outage_slots":
            float(
                outage_slots
            ),

        "min_soc":
            float(
                min_soc
            ),

        "scheduled_user_slots":
            float(
                scheduled_slots
            ),

        "unscheduled_user_slots":
            float(
                unscheduled_slots
            ),

        "hiring_cost":
            float(
                hiring_cost
            ),
    }


class FastCheckpointSweepTest(
    unittest.TestCase
):
    def test_discover_checkpoints_requires_complete_grid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(
                temporary_directory
            )

            (
                checkpoint_dir
                / "fast_ppo_pretrain_initial.pt"
            ).touch()

            (
                checkpoint_dir
                / "fast_ppo_pretrain_ep25.pt"
            ).touch()

            discovered = discover_checkpoints(
                checkpoint_dir,
                (25,),
            )

            self.assertEqual(
                [
                    checkpoint.label
                    for checkpoint
                    in discovered
                ],
                [
                    "initial",
                    "ep25",
                ],
            )

    def test_workload_check_is_rowwise_exact_by_default(
        self,
    ) -> None:
        reference = [
            _row(
                cost_per_scheduled_slot=10.0,
                episode=1,
            ),
            _row(
                cost_per_scheduled_slot=10.0,
                episode=2,
            ),
        ]

        identical = [
            _row(
                cost_per_scheduled_slot=9.0,
                episode=1,
            ),
            _row(
                cost_per_scheduled_slot=9.0,
                episode=2,
            ),
        ]

        mismatched = [
            _row(
                cost_per_scheduled_slot=9.0,
                scheduled_slots=1001.0,
                episode=1,
            ),
            _row(
                cost_per_scheduled_slot=9.0,
                episode=2,
            ),
        ]

        matched, issues = _workload_matches(
            reference,
            identical,
            relative_tolerance=0.0,
        )

        self.assertTrue(
            matched
        )

        self.assertEqual(
            issues,
            [],
        )

        matched, issues = _workload_matches(
            reference,
            mismatched,
            relative_tolerance=0.0,
        )

        self.assertFalse(
            matched
        )

        self.assertTrue(
            any(
                (
                    "episode_row=1"
                    in item
                    and
                    "scheduled_user_slots"
                    in item
                )
                for item
                in issues
            )
        )

    def test_nonzero_tolerance_remains_available_for_diagnostics(
        self,
    ) -> None:
        reference = [
            _row(
                cost_per_scheduled_slot=10.0,
            )
        ]

        close = [
            _row(
                cost_per_scheduled_slot=9.0,
                scheduled_slots=1005.0,
            )
        ]

        matched, issues = _workload_matches(
            reference,
            close,
            relative_tolerance=0.02,
        )

        self.assertTrue(
            matched
        )

        self.assertEqual(
            issues,
            [],
        )

    def test_selection_requires_feasible_significant_fast_cost_improvement(
        self,
    ) -> None:
        checkpoints = [
            CheckpointSpec(
                "initial",
                0,
                Path(
                    "initial.pt"
                ),
            ),
            CheckpointSpec(
                "ep25",
                25,
                Path(
                    "ep25.pt"
                ),
            ),
            CheckpointSpec(
                "ep50",
                50,
                Path(
                    "ep50.pt"
                ),
            ),
        ]

        seeds = (
            2026,
            2027,
            2028,
        )

        results: Dict[
            tuple[str, int],
            List[
                Dict[str, float]
            ],
        ] = {}

        for seed in seeds:
            results[
                (
                    "initial",
                    seed,
                )
            ] = [
                _row(
                    cost_per_scheduled_slot=(
                        10.0
                        + 0.01
                        * index
                    ),
                    episode=index + 1,
                )
                for index
                in range(5)
            ]

            results[
                (
                    "ep25",
                    seed,
                )
            ] = [
                _row(
                    cost_per_scheduled_slot=(
                        9.5
                        + 0.01
                        * index
                    ),
                    episode=index + 1,
                )
                for index
                in range(5)
            ]

            results[
                (
                    "ep50",
                    seed,
                )
            ] = [
                _row(
                    cost_per_scheduled_slot=(
                        9.0
                        + 0.01
                        * index
                    ),
                    outage_slots=1.0,
                    episode=index + 1,
                )
                for index
                in range(5)
            ]

        (
            summary,
            deltas,
            decision,
        ) = aggregate_results(
            checkpoints=checkpoints,
            seeds=seeds,
            results=results,
            minimum_allowed_soc=19.95,
            workload_relative_tolerance=0.0,
        )

        self.assertEqual(
            len(summary),
            3,
        )

        self.assertEqual(
            len(deltas),
            2,
        )

        self.assertEqual(
            deltas[0][
                "n_pairs"
            ],
            3,
        )

        self.assertEqual(
            deltas[0][
                "pairing_unit"
            ],
            "seed_mean",
        )

        self.assertEqual(
            decision[
                "primary_metric"
            ],
            "fast_cost",
        )

        self.assertEqual(
            decision[
                "selected_label"
            ],
            "ep25",
        )

        self.assertTrue(
            decision[
                "slow_dpp_gate_passed"
            ]
        )

        self.assertEqual(
            decision[
                "best_mean_label"
            ],
            "ep25",
        )


if __name__ == "__main__":
    unittest.main()