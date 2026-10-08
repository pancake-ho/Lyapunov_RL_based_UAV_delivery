"""Independently reconstruct UAVRTV reward and use the common physics verifier."""
import json
from pathlib import Path
import numpy as np
import baseline.NDTVS.api as c
from hppo.verify_trace import _verify_physics
from baseline.NDTVS.rewards.qoe import PSNR_DB


def audit(root):
    root = Path(root)
    data = json.loads((root / "resolved_config.json").read_text())
    cfg, reward = c.read_config(root / "resolved_config.json"), data["reward"]
    records = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    errors, boundaries, last_quality = [], {}, {u: -1 for u in range(cfg.num_users)}
    totals = dict(quality_gain=0., switch_penalty=0., rebuffer_penalty=0., energy_penalty=0., hiring_penalty=0.)
    episode_ends, ended, slots = 0, False, 0
    def check(ok, message):
        if not ok:
            errors.append(message)
    check(bool(records) and records[0].get("algorithm") == "uavrtv", "Missing UAVRTV start")
    rates = np.asarray(cfg.chunk_size_bits) * cfg.playback_chunks_per_slot / cfg.slot_duration_s
    for record in records:
        event = record["event"]
        if event == "frame_start":
            boundaries = record["regions"]
        elif event == "slot":
            slots += 1
            users = [u for rg in record["regions"].values() for u in rg["users"]]
            check(sorted(u["user"] for u in users) == list(range(cfg.num_users)), "Missing/duplicate user")
            for m, rg in record["regions"].items():
                expected = dict.fromkeys(totals, 0.)
                for u in rg["users"]:
                    i, k = u["user"], u["req_quality"]
                    check(last_quality[i] == u["last_quality_before"], "Quality history mismatch")
                    expected["rebuffer_penalty"] += reward["phi_per_second"] * max(
                        cfg.playback_chunks_per_slot - u["q_before"], 0.) * cfg.slot_duration_s / cfg.playback_chunks_per_slot
                    if u["delivered"]:
                        expected["quality_gain"] += reward["beta"] * PSNR_DB[k] / PSNR_DB[-1]
                        if last_quality[i] >= 0:
                            expected["switch_penalty"] += reward["delta_per_bps"] * abs(rates[k] - rates[last_quality[i]])
                        last_quality[i] = k
                first = record["slot_in_frame"] == 0
                move = boundaries[m]["relocation_energy_j"] if first else 0.
                expected["energy_penalty"] = reward["varsigma_per_joule"] * (
                    move + rg["hover_energy_j"] + rg["communication_energy_j"])
                expected["hiring_penalty"] = reward["lambda_h"] * reward["hiring_cost_per_frame"] * rg["hired"] if first else 0.
                raw = expected["quality_gain"] - sum(expected[k] for k in totals if k != "quality_gain")
                actual = record["rewards"][m]
                for key in totals:
                    check(np.isclose(actual[key], expected[key], rtol=0, atol=1e-8), key + " arithmetic")
                    totals[key] += expected[key]
                check(np.isclose(actual["training"], raw * reward["scale"], rtol=0, atol=1e-8), "Scaled SAC reward")
        elif event == "episode_end":
            episode_ends += 1
            for key, value in totals.items():
                check(np.isclose(record[key + "_total"], value, rtol=0, atol=1e-7), key + " summary")
        elif event == "baseline_end":
            ended = record.get("status") == "complete"
    check(ended and episode_ends == 1 and slots == cfg.num_frames * cfg.frame_slots, "Incomplete episode")
    passed, failed, details = _verify_physics(root)
    errors.extend(details)
    result = dict(valid=not errors and not failed, physics_checks_passed=sum(passed.values()),
                  physics_checks_failed=dict(failed), errors=errors[:50])
    c.atomic(root / "audit.json", result)
    if not result["valid"]:
        raise ValueError("UAVRTV trace audit failed: " + str(result))
    return result
