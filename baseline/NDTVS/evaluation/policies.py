"""Load selected checkpoints; verify immutable evaluation policies."""
from __future__ import annotations

import math
from dataclasses import asdict, replace
import hashlib
from baseline.NDTVS.evaluation.checks import require, sha256, read_json, physical


def policy_digest(agents):
    h = hashlib.sha256()
    for agent in agents:
        h.update(str(agent.update_count).encode())
        h.update(str(agent.buffer_size()).encode())
        for name, tensor in sorted(agent.net.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_best(saved):
    points = [v for v in saved["validation"] if v["trained_episodes"] == saved["next_episode"]]
    require(len(points) == 1 and points[0]["selection_score"] == saved["best"],
            "Use validation-selected best.pt, with completed validation")


def load_policies(args, c):
    policies, provenance, configs = {}, {}, {}
    for name, path in (("ndtvs", args.ndtvs_checkpoint), ("hppo_rsu", args.rsu_checkpoint)):
        saved = c.load_bundle(path)
        verification = c.verify_checkpoint_source(saved, name)
        verify_best(saved)
        d = saved["spec"]["config"]
        cfg = c.HPPOConfig(**{k: tuple(math.inf if x is None and k == "distance_bin_edges_m"
                                    else x for x in v) if isinstance(v, list) else v
                             for k, v in d.items()})
        configs[name] = cfg
        policies[name] = c.make_agents(replace(cfg, device=args.device), name)
        c.restore_agents(policies[name], saved["policies"])
        provenance[name] = {"selected_training_episode": saved["next_episode"],
                            "source_verification": verification,
                            "training_reward_definition": saved["spec"].get("qoe_definition"),
                            "files": {str(path.resolve()): sha256(path)}}
    cfg = c.read_config(args.config)
    runtime_path = args.proposed_runtime or args.config.parent / "runtime.json"
    runtime = read_json(runtime_path)
    expected = runtime["code_sha256"]
    current = {k.removeprefix("proposed/"): v for k, v in c.source_hashes().items()
               if k.startswith("proposed/")}
    raw = read_json(args.config)
    explicit_config = raw.get("config", raw)
    source_verification = c.verify_shared_sources(
        {"proposed/"+k: v for k, v in expected.items()},
        {"proposed/"+k: v for k, v in current.items()}, explicit_config)
    pairs = [("proposed", args.frame_checkpoint, args.slot_checkpoint)]
    if args.frame_checkpoint_2 or args.slot_checkpoint_2:
        require(args.frame_checkpoint_2 and args.slot_checkpoint_2,
                "Second proposed policy requires both checkpoint files")
        pairs.append(("proposed_2", args.frame_checkpoint_2, args.slot_checkpoint_2))
    for name, frame, slot in pairs:
        agents = c.make_agents(replace(cfg, device=args.device), "proposed")
        extras = [a.load(p) for a, p in zip(agents, (frame, slot))]
        require(extras[0].get("pair_id") and extras[0] == extras[1],
                "Proposed checkpoint pair metadata mismatch")
        configs[name], policies[name] = cfg, agents
        provenance[name] = {"selected_training_episode": extras[0]["episode"] + 1,
                            "pair": extras, "runtime_sha256": sha256(runtime_path),
                            "source_verification": source_verification,
                            "files": {str(p.resolve()): sha256(p) for p in (frame, slot)}}
    reference = None
    for name, cfg in configs.items():
        require(cfg.episode_offset == 0, "Training configuration must use episode_offset=0")
        require(cfg.mask_queue_actions and cfg.enforce_queue_admissibility
                and cfg.delivery_mode == "all_or_nothing", "Atomic queue-masked delivery required")
        current = physical(c.jsonable(asdict(cfg)))
        require(reference is None or current == reference, "Algorithm physical/control configs differ")
        reference = current
        for agent in policies[name]:
            agent.net.eval()
            require(all(c.torch.isfinite(t).all() for t in agent.net.state_dict().values()),
                    "Nonfinite checkpoint parameters")
        provenance[name]["model_parameters"] = [sum(p.numel() for p in a.net.parameters())
                                                for a in policies[name]]
    return configs, policies, provenance
