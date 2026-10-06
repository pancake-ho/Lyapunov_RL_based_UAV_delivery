"""Independent reconstruction of trace QoE and physics verification."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from baseline.NDTVS.common.io import atomic, load_bundle
from baseline.NDTVS.common.config import read_config
from hppo.verify_trace import _verify_physics
from baseline.NDTVS.rewards.qoe import PSNR_DB, QOE_WEIGHTS, REWARD_SCALE, reward_spec


def audit(root):
    cfg = read_config(root / "resolved_config.json")
    records = [json.loads(x) for x in (root / "trace.jsonl").read_text().splitlines()]
    errors, active, frame, slot, ended = [], False, 0, 0, False
    episode_ends = 0
    previous_quality = {u: -1 for u in range(cfg.num_users)}
    received = {u: [] for u in range(cfg.num_users)}
    first_request = {u: None for u in range(cfg.num_users)}
    stall_seconds = {u: 0.0 for u in range(cfg.num_users)}
    algorithm = records[0].get("algorithm") if records else None
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(bool(records) and records[0]["event"] == "baseline_start", "missing baseline_start")
    check(algorithm in ("ndtvs", "hppo_rsu", "proposed"), "unknown algorithm")
    check(records[0].get("qoe_definition") == reward_spec() if records else False, "reward definition/version")
    episode_id = records[0].get("episode") if records else None
    qoe_total = 0.0
    for r in records[1:]:
        check(not ended, "event after end")
        ev = r["event"]
        if ev in ("frame_start", "slot", "frame_end", "episode_end"):
            check(r["episode"] == episode_id, "episode identity")
        if ev in ("frame_start", "slot", "frame_end"):
            check(set(r["regions"]) == {str(m) for m in range(cfg.num_regions)}, "region coverage")
        if ev == "frame_start":
            check(not active and r["frame"] == frame, "frame order")
            active, slot = True, 0
            for rg in r["regions"].values():
                if algorithm != "proposed":
                    check(rg["executed_hire"] == 0 and not rg["executed_uav_users"], "UAV enabled")
        elif ev == "slot":
            check(active and r["frame"] == frame and r["slot_in_frame"] == slot, "slot order")
            users = [u for rg in r["regions"].values() for u in rg["users"]]
            check(sorted(u["user"] for u in users) == list(range(cfg.num_users)), "user coverage")
            for m, rg in r["regions"].items():
                total = 0.0
                for u in rg["users"]:
                    check(u["last_quality_before"] == previous_quality[u["user"]], "quality continuity")
                    # Independently reconstruct the full received sequence; do not
                    # call the online QoEHistory implementation under test.
                    i = u["user"]
                    if u["req_chunks"] and first_request[i] is None:
                        first_request[i] = PSNR_DB[u["req_quality"]]
                    received[i].extend([u["req_quality"]] * int(u["delivered"]))
                    dt = max(cfg.playback_chunks_per_slot - u["q_before"], 0.0)
                    stall_seconds[i] += dt
                    seq = received[i]
                    if len(seq) >= 2:
                        vq = first_request[i] - np.mean([first_request[i] - PSNR_DB[k] for k in seq[1:]])
                        qv = np.mean(np.abs(np.diff(seq)))
                    else:
                        vq = PSNR_DB[seq[0]] if seq else 0.0
                        qv = 0.0
                    value = QOE_WEIGHTS[0]*vq - QOE_WEIGHTS[1]*qv - QOE_WEIGHTS[2]*stall_seconds[i]
                    for key, expected in (("paper_qoe", value), ("paper_vq_db", vq),
                                          ("paper_qv_index", qv), ("stall_duration_s", dt),
                                          ("rebuffer_s_cumulative", stall_seconds[i]),
                                          ("paper_reward_scaled", value*REWARD_SCALE)):
                        check(np.isclose(expected, u[key], rtol=0, atol=1e-9), key + " arithmetic")
                    check(u["received_segments"] == len(seq), "received segment count")
                    check(u["first_requested_psnr_db"] == first_request[i], "first requested PSNR")
                    check(u["delivered"] in (0, u["req_chunks"]), "atomic delivery violated")
                    check(0 <= u["q_after"] <= cfg.large_queue_level + 1e-8, "queue bound")
                    if algorithm != "proposed":
                        check(u["provider"] in (0, 1), "UAV provider")
                    if u["delivered"]:
                        previous_quality[u["user"]] = u["req_quality"]
                    total += value
                if algorithm == "ndtvs":
                    check(np.isclose(total, r["rewards"][m]["training"] / REWARD_SCALE), "NDTVS reward")
                qoe_total += total
            slot += 1
        elif ev == "frame_end":
            check(active and slot == cfg.frame_slots, "incomplete frame")
            active, frame = False, frame + 1
        elif ev == "episode_end":
            episode_ends += 1
            check(not active and frame == cfg.num_frames, "incomplete episode")
            check(np.isclose(r["paper_qoe_per_user_slot"],
                             qoe_total / (cfg.num_users * cfg.num_frames * cfg.frame_slots)), "QoE summary")
        elif ev == "baseline_end":
            check(not active and frame == cfg.num_frames and r["status"] == "complete", "incomplete run")
            ended = True
        elif ev not in ("slot_ppo_update", "frame_ppo_update"):
            check(False, "unknown event: " + ev)
    check(ended, "missing baseline_end")
    check(episode_ends == 1, "episode_end count")
    passed, failed, details = _verify_physics(root)
    errors.extend(details)
    result = {"valid": not errors and not failed, "physics_checks_passed": sum(passed.values()),
              "physics_checks_failed": dict(failed), "errors": errors[:50]}
    atomic(root / "audit.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


def audit_run(root):
    saved = load_bundle(root / "latest.pt")
    results = []
    for row in saved["rows"]:
        if not row["trace_dir"]:
            continue
        directory = root / row["trace_dir"]
        try:
            code = audit(directory)
            detail = json.loads((directory / "audit.json").read_text())
        except (OSError, ValueError, KeyError, IndexError) as exc:
            code, detail = 1, {"valid": False, "errors": [str(exc)]}
        results.append({"episode": row["episode"], "directory": row["trace_dir"],
                        "exit_code": code, **detail})
    result = {"valid": all(r["exit_code"] == 0 for r in results),
              "checkpoint_completed_episodes": saved["next_episode"],
              "traced_episodes_checked": len(results), "per_episode": results}
    atomic(root / "audit_run.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
    return 0 if result["valid"] else 1
