"""Config comparability and paired single-SNR comparisons."""
from __future__ import annotations

import json
import numpy as np
from baseline.NDTVS.common.io import atomic, write_rows
from baseline.NDTVS.rewards.qoe import reward_spec


def comparable_config(c):
    ignore = {"seed", "episode_offset", "train_episodes", "eval_episodes", "save_every_episodes",
              "device", "torch_num_threads", "hidden_dims", "slot_update_every_frames",
              "frame_update_every_episodes", "write_jsonl_trace", "write_human_debug_log",
              "log_hidden_csi", "log_observation_vectors", "console_log_every_slots"}
    return {k: v for k, v in c.items() if k not in ignore and not k.startswith("ppo_")}


def compare(paths, out):
    data = [json.loads((p / "evaluation.json").read_text()) for p in paths]
    reference = data[0]
    if reference.get("qoe_definition") != reward_spec():
        raise ValueError("Re-evaluate every policy under the v2 QoE observer")
    if len({d["algorithm"] for d in data}) != len(data):
        raise ValueError("Use one evaluation per algorithm; repeat the comparison for each training seed")
    for d in data[1:]:
        for key in ("offset", "scenario_seed", "qoe_weights", "qoe_definition"):
            if d[key] != reference[key]:
                raise ValueError(f"Unpaired evaluations: {key}")
        if comparable_config(d["config"]) != comparable_config(reference["config"]):
            raise ValueError("Physical/control configuration differs")
        if [r["episode"] for r in d["per_episode"]] != [r["episode"] for r in reference["per_episode"]]:
            raise ValueError("Episode identities differ")
        # The adapter file is common too; differing source versions require a fresh evaluation.
        if d["source_sha256"] != reference["source_sha256"]:
            raise ValueError("Evaluation source versions differ")
    metrics = ("paper_qoe_per_user_slot", "stall_ratio", "average_quality_utility",
               "paper_qv_per_user_slot", "rebuffer_s_per_user", "stall_time_ratio", "average_received_psnr_db", "request_failure_ratio", "delivered_chunks_per_user_slot",
               "dpp_cost_per_user_slot", "original_cost_per_user_slot", "hire_rate", "energy_consumed_j")
    summaries = [{"algorithm": d["algorithm"], "episodes": len(d["per_episode"]),
                  **{k: float(np.mean([r[k] for r in d["per_episode"]])) for k in metrics}} for d in data]
    rng = np.random.default_rng(572913)
    pairs = []
    for i, a in enumerate(data):
        for b in data[i + 1:]:
            for key in metrics:
                delta = np.asarray([x[key] - y[key] for x, y in zip(a["per_episode"], b["per_episode"])])
                interval = [None, None]
                if len(delta) >= 2:
                    samples = rng.choice(delta, size=(5000, len(delta)), replace=True).mean(1)
                    interval = np.quantile(samples, [.025, .975]).tolist()
                pairs.append({"difference": a["algorithm"] + " - " + b["algorithm"], "metric": key,
                              "mean": float(delta.mean()), "ci95_low": interval[0], "ci95_high": interval[1]})
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "means.csv", summaries)
    write_rows(out / "paired_differences.csv", pairs)
    selected_episodes = {}
    for d in data:
        provenance = d["provenance"]
        count = provenance.get("trained_episodes")
        if count is None and provenance.get("pair"):
            last = provenance["pair"][0].get("episode")
            count = last + 1 if last is not None else None
        selected_episodes[d["algorithm"]] = count
    atomic(out / "comparison.json", {"means": summaries, "paired_differences": pairs,
           "selected_checkpoint_training_episodes": selected_episodes,
           "same_selected_training_episode": None not in selected_episodes.values()
                                             and len(set(selected_episodes.values())) == 1,
           "checkpoint_provenance": {d["algorithm"]: d["provenance"] for d in data},
           "uncertainty_scope": "Bootstrap across paired test scenarios, conditional on these trained checkpoints; not across independent training seeds.",
           "inputs": [str(p.resolve()) for p in paths]})
    print(json.dumps(summaries, indent=2))
    return 0
