from __future__ import annotations

"""
File-only observation for scheduling-hppo-v2.

Outputs:
  plots/*.png
  frames/epXXXXXX/frameXXXXXX.png
  animations/epXXXXXX_partXXX.gif
  decisions/*.csv
  verification.json / verification.txt / status.json

No web server. No policy/environment/training modification.

Compatibility baseline:
  feat/hrl 019f474a761e8a3d2e7e1add0c4841c41fe91692

The original plotter and equation verifier are reused.
A private, append-only snapshot makes live verification independent of
the incomplete final line currently being written by the training process.
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Avoid writing Matplotlib caches under an unavailable HOME directory.
_MPL_CACHE = tempfile.TemporaryDirectory(prefix="hppo_export_mpl_")
os.environ["MPLCONFIGDIR"] = _MPL_CACHE.name

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from PIL import Image

from hppo.plot import plot_run
from hppo.trace_contract import check_structure
from hppo.verify_trace import load_config, _verify_physics


BASE_COMMIT = "a63b1da63addcb94e0f2e0bbe8b92dab5a9df318"
MAX_LINE_BYTES = 16 * 1024 * 1024

# Exact final-completion diagnostic in the verified baseline source.
# Only this diagnostic is treated as expected for a growing trace.
UNFINISHED_REPORT = "S0 FAIL: empty or unfinished run"

PROVIDER_COLORS = {
    0: "#8b98a5",
    1: "#276fd1",
    2: "#d17b17",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload):
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    atomic_bytes(path, (text + "\n").encode("utf-8"))


def save_figure(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        # Fixed canvas dimensions are important when assembling GIF frames.
        fig.savefig(temporary, format="png", dpi=dpi)
    finally:
        plt.close(fig)

    try:
        with Image.open(temporary) as image:
            image.verify()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class FileExporter:
    def __init__(self, run_dir: Path, output_dir: Path):
        self.run_dir = run_dir.resolve()
        self.source = self.run_dir / "trace.jsonl"
        self.config_source = self.run_dir / "resolved_config.json"
        self.output = output_dir.resolve()

        # Each invocation gets a new output directory.
        # This prevents accidental mixing or duplicate CSV append on restart.
        self.output.mkdir(parents=True, exist_ok=False)

        self.snapshot = self.output / "source_snapshot"
        self.snapshot.mkdir()
        self.snapshot_trace = self.snapshot / "trace.jsonl"
        self.snapshot_trace.touch()

        self.config_bytes = self.config_source.read_bytes()
        (self.snapshot / "resolved_config.json").write_bytes(
            self.config_bytes
        )
        self.cfg = load_config(self.snapshot)

        self.offset = 0
        self.lines = 0
        self.source_size = 0
        self.pending_bytes = 0

        self.first_event = None
        self.ended = False
        self.fatal = None

        self.frames = {}
        self.episode_count = 0
        self.numeric_failures = 0
        self.numeric_reports = []

        self.csv_files = {}
        self.csv_writers = {}
        self.gif_versions = {}

        atomic_json(
            self.output / "provenance.json",
            {
                "created_at": utc_now(),
                "source_run": str(self.run_dir),
                "compatible_commit": BASE_COMMIT,
                "source_schema": "scheduling-hppo-v2",
                "mode": "file_only_observer",
                "frame_image_phase": "last_slot_decision_positions_and_outcomes",
                "geometry": "physical_ground_projection_y0_with_labels",
            },
        )

    def close(self):
        for stream in self.csv_files.values():
            stream.close()

    def csv_row(self, name, row):
        # Keep list/dict cells explicit and correctly CSV-quoted.
        converted = {
            key: (
                json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (list, tuple, dict))
                else value
            )
            for key, value in row.items()
        }

        if name not in self.csv_writers:
            if name == "episode_summary":
                path = self.snapshot / "episode_summary.csv"
            else:
                path = self.output / "decisions" / f"{name}.csv"

            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                stream,
                fieldnames=list(converted),
                extrasaction="raise",
            )
            writer.writeheader()

            self.csv_files[name] = stream
            self.csv_writers[name] = writer

        self.csv_writers[name].writerow(converted)

    def flush_csv(self):
        for stream in self.csv_files.values():
            stream.flush()

    def record(self, rec, position):
        event = rec["event"]

        if self.first_event is None:
            self.first_event = event

        if event == "run_end":
            self.ended = True
            return

        if event == "frame_start":
            key = (rec["episode"], rec["frame"])

            if key in self.frames:
                raise ValueError(f"duplicate frame_start: {key}")

            self.frames[key] = {
                "start_position": position,
                "last_slot_position": None,
                "slots": 0,
                "complete": False,
                "fast_reward": 0.0,
                "slow_reward": None,
            }

            for region, rg in rec["regions"].items():
                completion = rg["completion"]

                self.csv_row(
                    "slow_decisions",
                    {
                        "episode": key[0],
                        "frame": key[1],
                        "region": int(region),
                        "members": rg["members"],
                        "proposal_rsu_users": rg["proposal_rsu_users"],
                        "proposal_uav_candidates":
                            rg["proposal_uav_candidates"],
                        "executed_rsu_users": rg["executed_rsu_users"],
                        "executed_uav_users": rg["executed_uav_users"],
                        "unserved_users": rg["unserved_users"],
                        "executed_hire": rg["executed_hire"],
                        "executed_point": rg["executed_point"],
                        "uav_x_before_m": rg["uav_x_before"],
                        "uav_x_after_m": rg["uav_x_after"],
                        "battery_before_j": rg["battery_before_j"],
                        "battery_after_relocation_j":
                            rg["battery_after_relocation_j"],
                        "hiring_cost_weighted":
                            rg["hiring_cost_weighted"],
                        "selected_candidate":
                            completion["selected_index"],
                        "selected_dpp": completion["selected_score"],
                        "fast_update_count":
                            completion["fast_update_count"],
                    },
                )

                for index, candidate in enumerate(
                    completion["candidates"]
                ):
                    self.csv_row(
                        "completion_candidates",
                        {
                            "episode": key[0],
                            "frame": key[1],
                            "region": int(region),
                            "candidate_index": index,
                            "selected":
                                index == completion["selected_index"],
                            "hired": candidate["hired"],
                            "point": candidate["point"],
                            "rsu_users": candidate["rsu_users"],
                            "uav_users": candidate["uav_users"],
                            "mean_dpp": candidate["mean_dpp"],
                            "sample_dpp": candidate["sample_dpp"],
                            "hiring_dpp": candidate["hiring_dpp"],
                            "mean_queue_drift": candidate.get("mean_queue_drift"),
                            "mean_quality_dpp": candidate.get("mean_quality_dpp"),
                            "delta_from_no_hire": candidate.get("delta_from_no_hire"),
                            "sample_components": candidate.get("sample_components"),
                        },
                    )

        elif event == "slot":
            key = (rec["episode"], rec["frame"])
            frame = self.frames[key]

            frame["last_slot_position"] = position
            frame["slots"] += 1
            frame["fast_reward"] += sum(
                reward["training"]
                for reward in rec["rewards"].values()
            )

            for region, rg in rec["regions"].items():
                for user in rg["users"]:
                    self.csv_row(
                        "fast_decisions",
                        {
                            "episode": key[0],
                            "frame": key[1],
                            "slot": rec["slot_in_frame"],
                            "global_slot": rec["global_slot"],
                            "region": int(region),
                            "user": user["user"],
                            "provider": user["provider"],
                            "req_chunks": user["req_chunks"],
                            "rsu_horizontal_distance_m": user["rsu_horizontal_distance_m"],
                            "uav_horizontal_distance_m": user["uav_horizontal_distance_m"],
                            "rsu_link_distance_m": user.get("rsu_link_distance_m"),
                            "uav_link_distance_m": user.get("uav_link_distance_m"),
                            "chunk_action_cap": user.get("chunk_action_cap"),
                            "transmission_failed": user.get("transmission_failed"),
                            "failure_reason": user.get("failure_reason"),
                            "delivery_mode": user.get("delivery_mode", "partial"),
                            "req_quality": user["req_quality"],
                            "req_power_level":
                                user["req_power_level"],
                            "req_power_w": user["req_power_w"],
                            "exec_power_w": user["exec_power_w"],
                            "delivered": user["delivered"],
                            "feasible_by_rate":
                                user["feasible_by_rate"],
                            "queue_admissible_cap":
                                user["queue_admissible_cap"],
                            "feasible_chunks":
                                user["feasible_chunks"],
                            "capacity_bps": user["capacity_bps"],
                            "q_before": user["q_before"],
                            "q_after": user["q_after"],
                            "z_before": user["z_before"],
                            "z_after": user["z_after"],
                            "x_before_m": user["x_m"],
                            "x_after_m":
                                rec["mobility"]["x_after"][
                                    user["user"]
                                ],
                            "stall": user["stall"],
                            "user_dpp": user["dpp_slot_cost"],
                            "region_training_reward":
                                rec["rewards"][region]["training"],
                            "region_power_cap_w": rg["p_eff_w"],
                            "region_battery_after_j":
                                rg["battery_after_j"],
                        },
                    )

        elif event == "frame_end":
            key = (rec["episode"], rec["frame"])
            frame = self.frames[key]

            frame["complete"] = True
            frame["slow_reward"] = sum(
                reward["training"]
                for reward in rec["rewards"].values()
            )

            self.csv_row(
                "frame_metrics",
                {
                    "episode": key[0],
                    "frame": key[1],
                    "slots": frame["slots"],
                    "slow_reward": frame["slow_reward"],
                    "fast_reward_sum": frame["fast_reward"],
                    "frame_dpp_total": rec["frame_dpp_cost_total"],
                    "original_cost_total": rec["original_cost_total"],
                    "mean_Q": rec["mean_Q"],
                    "mean_Z": rec["mean_Z"],
                },
            )

        elif event == "episode_end":
            self.episode_count += 1
            self.csv_row(
                "episode_summary",
                {
                    key: value
                    for key, value in rec.items()
                    if key != "event"
                },
            )

        elif event.endswith("_ppo_update"):
            actor = (
                "frame_ppo"
                if event.startswith("frame")
                else "slot_ppo"
            )
            metrics = {
                key: value
                for key, value in rec.items()
                if "/" in key
            }

            valid = bool(metrics) and all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in metrics.values()
            )

            if not valid:
                self.numeric_failures += 1
                if len(self.numeric_reports) < 100:
                    self.numeric_reports.append(
                        f"PPO numeric failure at line {self.lines}"
                    )

            names = (
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "grad_norm",
                "explained_variance",
                "transitions",
                "updates",
            )

            self.csv_row(
                "ppo_updates",
                {
                    "actor": actor,
                    "episode": rec.get("episode"),
                    "frame": rec.get("frame"),
                    **{
                        name: rec.get(f"{actor}/{name}")
                        for name in names
                    },
                },
            )

    def pull(self):
        """
        Copy only newly available newline-terminated records.

        A captured byte limit prevents following a continuously growing file
        forever during one refresh.
        """
        if self.fatal:
            return False

        if self.config_source.read_bytes() != self.config_bytes:
            raise ValueError(
                "resolved_config.json changed. "
                "Use a new output directory for the new run."
            )

        if not self.source.exists():
            raise FileNotFoundError(self.source)

        self.source_size = self.source.stat().st_size

        if self.source_size < self.offset:
            raise ValueError(
                "trace.jsonl was truncated. "
                "Restart the exporter with a new output directory."
            )

        if self.snapshot_trace.stat().st_size != self.offset:
            raise ValueError("The observer snapshot was modified.")

        old_offset = self.offset

        with self.source.open("rb") as source:
            # Detect common replacement/truncation cases.
            # This observer expects an append-only source run.
            if self.offset:
                with self.snapshot_trace.open("rb") as previous:
                    anchors = (
                        (0, min(4096, self.offset)),
                        (
                            max(0, self.offset - 512),
                            min(512, self.offset),
                        ),
                    )
                    for position, length in anchors:
                        source.seek(position)
                        previous.seek(position)
                        if source.read(length) != previous.read(length):
                            raise ValueError(
                                "Previously indexed trace bytes changed. "
                                "Use a new output directory."
                            )

            source.seek(self.offset)

            with self.snapshot_trace.open("ab") as snapshot:
                while self.offset < self.source_size:
                    position = self.offset
                    raw = source.readline(MAX_LINE_BYTES + 1)

                    if len(raw) > MAX_LINE_BYTES:
                        self.fatal = (
                            f"Line {self.lines + 1} exceeds "
                            f"{MAX_LINE_BYTES} bytes."
                        )
                        break

                    if (
                        not raw
                        or source.tell() > self.source_size
                        or not raw.endswith(b"\n")
                    ):
                        break

                    try:
                        rec = json.loads(raw)
                        if (
                            not isinstance(rec, dict)
                            or not isinstance(rec.get("event"), str)
                        ):
                            raise ValueError(
                                "record must contain a string event"
                            )
                    except (ValueError, UnicodeDecodeError) as exc:
                        self.fatal = (
                            f"Invalid committed JSON at line "
                            f"{self.lines + 1}, byte {position}: {exc}"
                        )
                        break

                    snapshot.write(raw)
                    self.offset += len(raw)
                    self.lines += 1

                    try:
                        self.record(rec, position)
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        IndexError,
                        OverflowError,
                    ) as exc:
                        self.fatal = (
                            f"Malformed record at line {self.lines}, "
                            f"byte {position}: {exc}"
                        )
                        break

        self.pending_bytes = self.source_size - self.offset
        self.flush_csv()
        return self.offset != old_offset

    def verify(self):
        if not self.lines and not self.fatal:
            return {
                "status": "WAITING",
                "passed": {},
                "failed": {},
                "reports": [],
            }

        passed, failed, reports = check_structure(
            self.snapshot,
            self.cfg,
            max_report=100,
        )

        # Missing final run_end is expected for a live prefix.
        # Do not suppress missing slots, malformed records, or any equation
        # or scheduling failure.
        if (
            not self.ended
            and self.first_event == "run_start"
            and UNFINISHED_REPORT in reports
        ):
            reports.remove(UNFINISHED_REPORT)
            failed["S0"] -= 1
            if failed["S0"] == 0:
                del failed["S0"]

        if self.fatal:
            failed["READ"] += 1
            reports.append(self.fatal)

        if self.ended and self.pending_bytes:
            failed["TRAILING"] += 1
            reports.append(
                "Unprocessed bytes exist after the recorded run_end."
            )

        # Reuse the original equation checks on the immutable prefix.
        if not failed:
            try:
                physical_passed, physical_failed, physical_reports = (
                    _verify_physics(self.snapshot, max_report=100)
                )
                passed += physical_passed
                failed += physical_failed
                reports.extend(physical_reports)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
                ZeroDivisionError,
                OverflowError,
            ) as exc:
                failed["PHYSICS_READ"] += 1
                reports.append(f"Physical verification failed: {exc}")

        if self.numeric_failures:
            failed["PPO_NUMERIC"] += self.numeric_failures
            reports.extend(self.numeric_reports)

        failed = Counter(
            {key: value for key, value in failed.items() if value > 0}
        )

        status = (
            "INVALID"
            if failed
            else "COMPLETE"
            if self.ended
            else "IN_PROGRESS"
        )

        return {
            "status": status,
            "passed": dict(passed),
            "failed": dict(failed),
            "reports": reports[:100],
            "scope": (
                "Original structural/physical checks on the committed "
                "prefix; final completion is required only for COMPLETE."
            ),
        }

    def read_at(self, position):
        with self.snapshot_trace.open("rb") as stream:
            stream.seek(position)
            return json.loads(stream.readline())

    def frame_png(self, episode, frame_index, dpi):
        key = (episode, frame_index)
        metadata = self.frames[key]

        output = (
            self.output
            / "frames"
            / f"ep{episode:06d}"
            / f"frame{frame_index:06d}.png"
        )

        if output.exists():
            return output

        start = self.read_at(metadata["start_position"])
        slot = self.read_at(metadata["last_slot_position"])
        cfg = self.cfg

        from hppo.topdown import frame_figure
        fig = frame_figure(start, slot, cfg, self.run_dir.name)

        save_figure(fig, output, dpi)
        return output

    def save_gif(self, paths, output, duration_ms):
        images = []
        temporary = output.with_name(output.name + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

            images[0].save(
                temporary,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
                optimize=False,
            )

            # Verify that every exported frame is represented.
            with Image.open(temporary) as result:
                if result.n_frames != len(paths):
                    raise ValueError(
                        "GIF frame count differs from the PNG sequence."
                    )

            os.replace(temporary, output)
        finally:
            for image in images:
                image.close()
            if temporary.exists():
                temporary.unlink()

    def render_episode(self, episode, args):
        keys = sorted(
            key
            for key, value in self.frames.items()
            if key[0] == episode and value["complete"]
        )

        if not keys:
            return

        paths = [
            self.frame_png(ep, frame, args.dpi)
            for ep, frame in keys
        ]

        # Split long episodes into bounded-size GIF segments.
        # No frames are silently dropped.
        for begin in range(0, len(paths), args.gif_chunk):
            block = paths[begin:begin + args.gif_chunk]
            part = begin // args.gif_chunk
            version_key = (episode, part)

            if self.gif_versions.get(version_key) == len(block):
                continue

            output = (
                self.output
                / "animations"
                / f"ep{episode:06d}_part{part:03d}.gif"
            )
            self.save_gif(block, output, args.duration_ms)
            self.gif_versions[version_key] = len(block)

        frame_numbers = [frame for _, frame in keys]
        slow = [self.frames[key]["slow_reward"] for key in keys]
        fast = [self.frames[key]["fast_reward"] for key in keys]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11, 6),
            sharex=True,
            layout="constrained",
        )
        axes[0].plot(
            frame_numbers, slow, marker="o", ms=3, color="#087f80"
        )
        axes[1].plot(
            frame_numbers, fast, marker="o", ms=3, color="#276fd1"
        )

        axes[0].set_ylabel("Slow training reward / frame")
        axes[1].set_ylabel("Fast training reward sum / frame")
        axes[1].set_xlabel("Completed frame index")

        for ax in axes:
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.suptitle(
            f"{self.run_dir.name} | episode {episode} | "
            "completed frames only"
        )

        save_figure(
            fig,
            self.output
            / "plots"
            / f"frame_rewards_ep{episode:06d}.png",
            args.dpi,
        )

    def render(self, args, verification):
        available = sorted(
            {
                episode
                for (episode, _), metadata in self.frames.items()
                if metadata["complete"]
            }
        )

        if args.all_episodes:
            selected = available
        elif args.episode is not None:
            selected = (
                [args.episode] if args.episode in available else []
            )
        else:
            selected = available[-1:]

        for episode in selected:
            self.render_episode(episode, args)

        # Reuse existing PNG generation instead of replacing it.
        if self.episode_count and selected:
            generated = plot_run(
                self.snapshot,
                episode=selected[-1],
                max_trace_slots=args.max_trace_slots,
                allow_partial=verification["status"] != "COMPLETE",
            )

            for name in (
                "training_overview.png",
                "trace_overview.png",
                "completion_example.png",
            ):
                path = generated / name
                if path.exists():
                    atomic_bytes(
                        self.output / "plots" / name,
                        path.read_bytes(),
                    )

        return selected

    def save_verification(self, result):
        payload = {
            **result,
            "checked_at": utc_now(),
            "source_run": str(self.run_dir),
            "committed_bytes": self.offset,
            "source_bytes_at_refresh": self.source_size,
            "pending_or_unprocessed_bytes": self.pending_bytes,
            "records": self.lines,
            "completed_episodes": self.episode_count,
        }

        atomic_json(
            self.output / "verification.json",
            payload,
        )

        codes = sorted(
            set(result["passed"]) | set(result["failed"])
        )

        lines = [
            f"status: {result['status']}",
            f"records: {self.lines}",
            f"committed bytes: {self.offset}",
            f"pending/unprocessed bytes: {self.pending_bytes}",
            "",
            f"{'check':<20}{'passed':>12}{'failed':>12}",
        ]

        for code in codes:
            lines.append(
                f"{code:<20}"
                f"{result['passed'].get(code, 0):>12}"
                f"{result['failed'].get(code, 0):>12}"
            )

        lines.extend(["", *result["reports"]])

        atomic_bytes(
            self.output / "verification.txt",
            ("\n".join(lines) + "\n").encode("utf-8"),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Save HPPO plots, frame PNGs, GIFs and CSVs."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="New output directory; must not already exist",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episode", type=int)
    selection.add_argument("--all-episodes", action="store_true")

    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="Repeat until run_end; 0 means export once",
    )
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--duration-ms", type=int, default=600)
    parser.add_argument("--gif-chunk", type=int, default=100)
    parser.add_argument("--max-trace-slots", type=int, default=1000)

    args = parser.parse_args(argv)

    if args.watch != 0 and args.watch < 5:
        parser.error("--watch must be 0 or at least 5 seconds")

    if not 60 <= args.dpi <= 200:
        parser.error("--dpi must be between 60 and 200")

    if not 1 <= args.gif_chunk <= 100:
        parser.error("--gif-chunk must be between 1 and 100")

    if args.duration_ms < 20 or args.max_trace_slots < 1:
        parser.error(
            "--duration-ms must be >= 20 and "
            "--max-trace-slots must be >= 1"
        )

    run_dir = args.run_dir.resolve()
    if not (run_dir / "resolved_config.json").is_file():
        parser.error("resolved_config.json was not found")
    if not (run_dir / "trace.jsonl").is_file():
        parser.error("trace.jsonl was not found")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.out or (
        run_dir / "artifacts" / f"files_{stamp}_{os.getpid()}"
    )

    exporter = None

    try:
        exporter = FileExporter(run_dir, output)
        print(f"[OUTPUT] {exporter.output}", flush=True)

        first = True
        while True:
            changed = exporter.pull()

            if first or changed or exporter.fatal:
                first = False
                result = exporter.verify()
                exporter.save_verification(result)

                if result["status"] == "INVALID":
                    atomic_json(
                        exporter.output / "status.json",
                        {
                            "updated_at": utc_now(),
                            "status": "INVALID_TRACE",
                            "message": (
                                "See verification.txt. "
                                "New images were not published."
                            ),
                        },
                    )
                    print(
                        "[INVALID] See verification.txt",
                        file=sys.stderr,
                        flush=True,
                    )
                    return 1

                selected = exporter.render(args, result)

                atomic_json(
                    exporter.output / "status.json",
                    {
                        "updated_at": utc_now(),
                        "status": result["status"],
                        "records": exporter.lines,
                        "completed_episodes":
                            exporter.episode_count,
                        "rendered_episodes": selected,
                        "frame_phase": "completed_frame_end",
                        "message": (
                            "Files updated"
                            if selected
                            else "Waiting for a completed selected frame"
                        ),
                    },
                )

                print(
                    f"[{result['status']}] "
                    f"records={exporter.lines} "
                    f"episodes={exporter.episode_count} "
                    f"rendered={selected}",
                    flush=True,
                )

                if result["status"] == "COMPLETE":
                    return 0

            if not args.watch:
                return 0

            time.sleep(args.watch)

    except KeyboardInterrupt:
        print(
            "\nObserver stopped. Training was not stopped.",
            flush=True,
        )
        return 0

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"

        if exporter is not None:
            try:
                atomic_json(
                    exporter.output / "status.json",
                    {
                        "updated_at": utc_now(),
                        "status": "EXPORT_ERROR",
                        "message": message,
                        "records": exporter.lines,
                    },
                )
            except OSError:
                pass

        print(f"[EXPORT_ERROR] {message}", file=sys.stderr)
        return 2

    finally:
        if exporter is not None:
            exporter.close()


if __name__ == "__main__":
    raise SystemExit(main())