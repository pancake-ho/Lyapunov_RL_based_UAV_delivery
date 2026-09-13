"""Structural and scheduling-completion verification before physical checks."""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np


def check_structure(run_dir: Path, cfg, max_report=20):
    passed, failed, reports = Counter(), Counter(), []
    def check(ok, detail, code="S0"):
        (passed if ok else failed)[code] += 1
        if not ok and len(reports) < max_report:
            reports.append(f"{code} FAIL: {detail}")
    started = ended = False
    active = False
    completed_episodes = next_frame = next_slot = 0
    expected = offset = None
    expected_regions = {str(m) for m in range(cfg.num_regions)}
    frame_rewards = slot_rewards = 0.0
    try:
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                rec = json.loads(line)
                ev = rec["event"]
                check(not ended, f"line {line_no}: event after run_end")
                if ev == "run_start":
                    check(not started and line_no == 1, "duplicate/late run_start")
                    started = True
                    expected, offset = rec["expected_episodes"], rec["episode_offset"]
                    check(rec["schema"] == "scheduling-hppo-v2", "unsupported schema")
                    mode = rec["mode"]
                    args = json.loads((run_dir / "resolved_config.json").read_text())["args"]
                    check(mode == args.get("mode", mode), "mode differs from resolved args")
                    check(expected == (cfg.train_episodes if mode == "train" else cfg.eval_episodes)
                          and offset == cfg.episode_offset, "run budget differs from config")
                else:
                    check(started, f"line {line_no}: missing run_start")
                    if not started:
                        break
                if ev in ("frame_start", "slot", "frame_end"):
                    check(rec["episode"] == offset + completed_episodes and rec["frame"] == next_frame,
                          f"line {line_no}: episode/frame order")
                    check(set(rec["regions"]) == expected_regions, f"line {line_no}: region coverage")
                if ev == "frame_start":
                    check(not active and next_frame < cfg.num_frames, "frame started before previous frame ended")
                    active, next_slot = True, 0
                    all_members = [u for rg in rec["regions"].values() for u in rg["members"]]
                    check(sorted(all_members) == list(range(cfg.num_users)), "missing/duplicate membership")
                    check(len(rec["membership"]) == cfg.num_users, "membership vector length")
                    for m, rg in rec["regions"].items():
                        ids = [u["user"] for u in rg["user_state"]]
                        check(sorted(ids) == sorted(rg["members"]), "user snapshots missing/duplicated")
                        tokens = rg["raw_assoc"]
                        sr = [i for i, v in enumerate(tokens) if v == 1]
                        su = [i for i, v in enumerate(tokens) if v == 2]
                        check(len(tokens) == cfg.num_users and all(type(v) is int and v in (0, 1, 2) for v in tokens),
                              "proposal domain", "S1")
                        check(sr == rg["proposal_rsu_users"] and su == rg["proposal_uav_candidates"]
                              and len(sr) <= cfg.rsu_capacity and len(su) <= cfg.uav_capacity
                              and set(sr + su) <= set(rg["members"]), "proposal semantics/capacity", "S1")
                        check(rg["executed_rsu_users"] == sr
                              and rg["executed_uav_users"] == (su if rg["executed_hire"] else [])
                              and rg["unhired_uav_candidates"] == ([] if rg["executed_hire"] else su)
                              and rg["raw_action"] == tokens and not rg["projection_reasons"],
                              "completion changed scheduling", "S1")
                        d = rg["completion"]
                        rows = d["candidates"]
                        check([(c["hired"], c["point"]) for c in rows]
                              == [(0, -1)] + [(1, p) for p in rg["feasible_points"]],
                              "completion omitted/added a candidate", "S2")
                        check(d["scenarios"] == cfg.rollout_scenarios
                              and len(d["seed_domains"]) == cfg.rollout_scenarios, "MC sample count", "S2")
                        for c in rows:
                            samples = c["sample_dpp"]
                            check(len(samples) == cfg.rollout_scenarios and all(math.isfinite(v) for v in samples)
                                  and math.isclose(sum(samples)/len(samples), c["mean_dpp"], rel_tol=1e-9, abs_tol=1e-9),
                                  "candidate mean/sample mismatch", "S2")
                            check(c["rsu_users"] == sr and c["uav_users"] == (su if c["hired"] else []),
                                  "candidate resampled scheduling", "S2")
                            check(math.isclose(c["hiring_dpp"], cfg.lyapunov_v * cfg.lambda_h * cfg.hiring_cost_per_frame * c["hired"]),
                                  "candidate hiring cost", "S2")
                        for j, item in enumerate(d["seed_domains"]):
                            domain = [cfg.seed, cfg.completion_seed_offset, rec["episode"], rec["frame"], int(m), j]
                            seed = int(np.random.SeedSequence(domain).spawn(2)[0].generate_state(1, dtype=np.uint64)[0])
                            check(item["domain"] == domain and item["channel_seed"] == seed, "rollout seed domain", "S2")
                        best = min(range(len(rows)), key=lambda i: rows[i]["mean_dpp"])
                        check(d["selected_index"] == best and d["selected_score"] == rows[best]["mean_dpp"]
                              and (rg["executed_hire"], rg["executed_point"]) == (rows[best]["hired"], rows[best]["point"]),
                              "completion is not recorded argmin", "S2")
                elif ev == "slot":
                    check(active and rec["slot_in_frame"] == next_slot and next_slot < cfg.frame_slots,
                          f"line {line_no}: duplicate/missing/out-of-order slot")
                    check(rec["global_slot"] == next_frame * cfg.frame_slots + next_slot, "global slot sequence")
                    for rg in rec["regions"].values():
                        users = [u["user"] for u in rg["users"]]
                        check(len(set(users)) == len(users), "duplicate user slot records")
                        check(all(math.isfinite(u["q_after"]) and u["q_after"] >= 0 for u in rg["users"]), "invalid queue")
                        if cfg.enforce_queue_admissibility:
                            check(all(u["q_after"] <= cfg.large_queue_level + 1e-7 for u in rg["users"]), "queue guard violation")
                    if cfg.reward_mode == "dpp":
                        for m, rg in rec["regions"].items():
                            target = -cfg.ppo_reward_scale * rg["dpp_slot_cost"]
                            check(math.isclose(rec["rewards"][m]["training"], target, rel_tol=1e-9, abs_tol=1e-9),
                                  "slot DPP reward", "S3")
                    slot_rewards += sum(r["training"] for r in rec["rewards"].values())
                    next_slot += 1
                elif ev == "frame_end":
                    check(active and next_slot == cfg.frame_slots, "incomplete frame")
                    active = False
                    next_frame += 1
                    if cfg.reward_mode == "dpp":
                        for m, rg in rec["regions"].items():
                            target = -cfg.ppo_reward_scale * rg["frame_dpp_cost"]
                            check(math.isclose(rec["rewards"][m]["training"], target, rel_tol=1e-9, abs_tol=1e-9),
                                  "frame DPP reward", "S3")
                    frame_rewards += sum(r["training"] for r in rec["rewards"].values())
                elif ev == "episode_end":
                    check(not active and next_frame == cfg.num_frames and rec["frames"] == cfg.num_frames,
                          "incomplete episode")
                    check(rec["episode"] == offset + completed_episodes, "episode end index")
                    check(math.isclose(rec["slow_reward_sum"], frame_rewards, rel_tol=1e-8, abs_tol=1e-8)
                          and math.isclose(rec["fast_reward_sum"], slot_rewards, rel_tol=1e-8, abs_tol=1e-8),
                          "episode reward sums", "S3")
                    completed_episodes += 1
                    next_frame = next_slot = 0
                    frame_rewards = slot_rewards = 0.0
                elif ev == "run_end":
                    check(not active and next_frame == 0 and completed_episodes == expected
                          and rec["status"] == "complete", "incomplete run")
                    ended = True
                elif ev not in ("run_start", "frame_ppo_update", "slot_ppo_update"):
                    check(False, f"unknown event {ev}")
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError, OSError) as exc:
        check(False, f"malformed/missing trace: {exc}")
    check(started and ended and completed_episodes > 0 and completed_episodes == expected
          and not active and next_frame == 0, "empty or unfinished run")
    return passed, failed, reports
