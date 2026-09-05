from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config_p3 import P3Config
from run.p3_common import run_policy
from run.p3_eval_shard import (
    DEFAULT_BASELINE_POLICIES,
    DEFAULT_SEEDS,
    EvalTask,
    assigned_task_indices,
    artifact_paths,
    build_eval_tasks,
    is_complete,
    parse_list,
    task_fingerprint,
)
from run.p3_eval_aggregate import completed_payloads


class P3EvalPipelineTests(unittest.TestCase):
    def test_task_matrix_reuses_baselines_and_splits_checkpoints(self) -> None:
        tasks = build_eval_tasks(DEFAULT_SEEDS, DEFAULT_BASELINE_POLICIES)
        self.assertEqual(
            len(tasks),
            len(DEFAULT_SEEDS) * (len(DEFAULT_BASELINE_POLICIES) + 2),
        )
        triples = {(task.group, task.policy, task.seed) for task in tasks}
        self.assertEqual(len(triples), len(tasks))
        for seed in DEFAULT_SEEDS:
            self.assertIn(("ppo_best", "ppo", seed), triples)
            self.assertIn(("ppo_latest", "ppo", seed), triples)
            for policy in DEFAULT_BASELINE_POLICIES:
                self.assertIn(("baselines", policy, seed), triples)

    def test_two_long_lived_workers_cover_tasks_once_and_balance_policies(self) -> None:
        tasks = build_eval_tasks(DEFAULT_SEEDS, DEFAULT_BASELINE_POLICIES)
        left = assigned_task_indices(len(tasks), worker_index=0, worker_count=2)
        right = assigned_task_indices(len(tasks), worker_index=1, worker_count=2)
        self.assertEqual(len(left), 40)
        self.assertEqual(len(right), 40)
        self.assertEqual(set(left).intersection(right), set())
        self.assertEqual(set(left).union(right), set(range(80)))
        for policy in DEFAULT_BASELINE_POLICIES:
            self.assertEqual(sum(tasks[index].policy == policy for index in left), 5)
            self.assertEqual(sum(tasks[index].policy == policy for index in right), 5)

    def test_list_parser_supports_shell_safe_colon_lists(self) -> None:
        self.assertEqual(parse_list("3026:3027:3028", int), (3026, 3027, 3028))
        self.assertEqual(parse_list("dpp,rsu_only", str), ("dpp", "rsu_only"))
        with self.assertRaises(ValueError):
            parse_list("3026:3026", int)

    def test_progress_callback_and_deferred_outputs(self) -> None:
        cfg = P3Config(
            seed=17,
            num_frames=2,
            users_per_region=2,
            rollout_scenarios=1,
        )
        progress: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_policy(
                cfg,
                "rsu_only",
                root,
                progress_interval_frames=1,
                progress_callback=lambda count, row: progress.append(
                    (count, int(row["frame"]))
                ),
                write_outputs=False,
            )
            self.assertEqual(progress, [(1, 0), (2, 1)])
            self.assertEqual(len(result.frame_rows), 2)
            self.assertFalse((root / "frames_rsu_only_seed17.csv").exists())

    def test_parallel_rollout_preserves_actions_and_metrics(self) -> None:
        cfg = P3Config(
            seed=29,
            num_frames=2,
            users_per_region=2,
            rollout_scenarios=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequential = run_policy(
                cfg,
                "dpp",
                root,
                rollout_workers=1,
                write_outputs=False,
            )
            parallel = run_policy(
                cfg,
                "dpp",
                root,
                rollout_workers=2,
                write_outputs=False,
            )
        for left, right in zip(sequential.frame_rows, parallel.frame_rows):
            for key in left:
                if key != "selection_seconds":
                    self.assertEqual(left[key], right[key], key)
        for key in sequential.summary:
            if key != "runtime_seconds":
                self.assertEqual(sequential.summary[key], parallel.summary[key], key)

    def test_resume_requires_matching_summary_and_all_artifacts(self) -> None:
        task = EvalTask(group="baselines", policy="rsu_only", seed=3026)
        fingerprint = task_fingerprint(task, frames=400, rollouts=4, checkpoint_sha256=None)
        with tempfile.TemporaryDirectory() as directory:
            paths = artifact_paths(Path(directory), task)
            for name in ("frames", "distance", "points", "quality", "users"):
                paths[name].parent.mkdir(parents=True, exist_ok=True)
                paths[name].write_text("header\nvalue\n", encoding="utf-8")
            paths["summary"].parent.mkdir(parents=True, exist_ok=True)
            paths["summary"].write_text(
                json.dumps({"event": "completed", "fingerprint": fingerprint}),
                encoding="utf-8",
            )
            self.assertTrue(is_complete(paths, fingerprint))
            changed = dict(fingerprint, source_sha256="changed")
            self.assertFalse(is_complete(paths, changed))
            paths["users"].unlink()
            self.assertFalse(is_complete(paths, fingerprint))

    def test_aggregate_completion_probe_is_non_destructive(self) -> None:
        tasks = (
            EvalTask(group="baselines", policy="rsu_only", seed=3026),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payloads, incomplete = completed_payloads(root, tasks)
            self.assertEqual(payloads, [])
            self.assertEqual(incomplete, list(tasks))
            self.assertEqual(list(root.rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
