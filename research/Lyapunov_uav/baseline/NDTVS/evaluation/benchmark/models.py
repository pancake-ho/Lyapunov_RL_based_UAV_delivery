"""Load each model with its own training config; verify physical comparability."""
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from dataclasses import asdict, replace
from baseline.NDTVS.evaluation.checks import require, read_json, sha256, physical
from baseline.NDTVS.evaluation.policies import verify_best, policy_digest


PATH_KEYS = ("config", "runtime", "frame_checkpoint", "slot_checkpoint", "checkpoint")


def freeze_models(s, c):
    """Freeze selected inputs once; resume always uses that same snapshot."""
    manifest = s.OUT / "inputs/manifest.json"
    requested = [{key:str(value) if isinstance(value,Path) else value
                  for key,value in item.items()} for item in s.MODELS]
    if manifest.exists():
        saved = read_json(manifest)
        require(saved["requested"] == requested, "Model selection changed; choose a new OUT")
        for item in saved["files"]:
            require(sha256(item["snapshot"]) == item["sha256"], "Frozen checkpoint/config changed")
        return saved["models"]
    originals = []
    for model in s.MODELS:
        if "completion_status" in model:
            status = read_json(Path(model["completion_status"]))
            require(status.get("status", "").lower() == "complete" and not status.get("final_validation_pending", False),
                    f"Training/final validation not complete: {model['name']}")
        for key in PATH_KEYS:
            if key in model:
                path = Path(model[key]).resolve()
                originals.append((model["name"], key, path, sha256(path)))
    root = s.OUT / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    frozen, files = [], []
    for model in s.MODELS:
        item = {k: v for k,v in model.items() if k not in PATH_KEYS and k != "completion_status"}
        for name, key, source, fingerprint in originals:
            if name != model["name"]:
                continue
            target = root/name/(key+source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name+"."+uuid.uuid4().hex+".tmp")
            try:
                shutil.copyfile(source, tmp)
                require(sha256(tmp) == fingerprint == sha256(source),
                        "Input changed during snapshot; retry after training stops")
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
            item[key] = str(target.resolve())
            files.append(dict(original=str(source), snapshot=str(target.resolve()), sha256=fingerprint))
        frozen.append(item)
    # Detect a training writer that advanced either member during the whole copy.
    require(all(sha256(path) == fp for _,_,path,fp in originals), "Inputs changed during snapshot")
    c.atomic(manifest, dict(requested=requested, models=frozen, files=files))
    return frozen


def configuration(d, c):
    converted = {k: tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
                 if isinstance(v, list) else v for k,v in d.items()}
    return c.HPPOConfig(**converted)


def capacity_report(agents, name):
    reports = []
    for agent in agents:
        groups = dict(actor_only=0, critic_only=0, shared=0)
        ndtvs = agent.name == "ndtvs_ppo"
        for key, p in agent.net.named_parameters():
            group = ("critic_only" if key.startswith(("critic.", "value_head.")) else "actor_only") if ndtvs else (
                "shared" if key.startswith("trunk.") else "critic_only" if key.startswith("critic.") else "actor_only")
            groups[group] += p.numel()
        total = sum(p.numel() for p in agent.net.parameters())
        require(sum(groups.values()) == total, "Parameter accounting mismatch")
        reports.append(dict(model=name, policy=agent.name, architecture=type(agent.net).__name__,
            hidden_dims=list(agent.cfg.hidden_dims), parameters=total,
            trained_ppo_updates=agent.update_count,
            trainable_parameters=sum(p.numel() for p in agent.net.parameters() if p.requires_grad),
            weight_bytes=sum(p.numel()*p.element_size() for p in agent.net.parameters()), **groups))
    return reports


def load_models(models, device, c):
    configs, agents, provenance, sizes = {}, {}, {}, []
    reference = None
    shared = {k:v for k,v in c.source_hashes().items() if k.startswith("proposed/")}
    for item in models:
        name, algorithm = item["name"], item["algorithm"]
        if algorithm == "proposed":
            cfg = c.read_config(Path(item["config"]))
            runtime = read_json(Path(item["runtime"]))
            raw = read_json(Path(item["config"]))
            expected = {(k if k.startswith("proposed/") else "proposed/"+k):v
                        for k,v in runtime["code_sha256"].items()}
            verification = c.verify_shared_sources(expected, shared, raw.get("config",raw))
            policies = c.make_agents(replace(cfg, device=device), algorithm)
            extras = [a.load(Path(item[key])) for a,key in zip(policies,("frame_checkpoint","slot_checkpoint"))]
            require(extras[0].get("pair_id") and extras[0] == extras[1], "Proposed checkpoint pair mismatch")
            trained, dual = int(extras[0]["episode"])+1, extras[0].get("dual_lambda_z", cfg.dual_init)
            metadata = {"pair":extras[0]}
        else:
            saved = c.load_bundle(Path(item["checkpoint"]))
            verification = c.verify_checkpoint_source(saved, algorithm)
            verify_best(saved)
            cfg = configuration(saved["spec"]["config"], c)
            policies = c.make_agents(replace(cfg,device=device), algorithm)
            c.restore_agents(policies, saved["policies"])
            trained, dual = saved["next_episode"], saved.get("dual",cfg.dual_init)
            metadata = {"training_reward_definition":saved["spec"].get("qoe_definition"),
                        "validation_best_score":saved["best"]}
        if "expected_v" in item:
            require(cfg.lyapunov_v == item["expected_v"], f"Wrong V in {name}")
        current = physical(c.jsonable(asdict(cfg)))
        current.pop("lyapunov_v", None)  # The intended treatment; preserve each policy's own V.
        if reference is not None:
            differences = sorted(k for k in reference.keys() | current.keys() if reference.get(k) != current.get(k))
            require(not differences, "Physical/control settings differ: "+", ".join(differences))
        reference = current
        require(cfg.num_quality_levels == 4, "Expected four PSNR quality levels")
        for a in policies:
            a.net.eval()
            require(all(c.torch.isfinite(v).all() for v in a.net.state_dict().values()), "Nonfinite weights")
        configs[name], agents[name] = cfg, policies
        sizes.extend(capacity_report(policies,name))
        provenance[name] = dict(algorithm=algorithm, training_v=cfg.lyapunov_v,
            selected_training_episode=trained, selection=item.get("selection","user-selected"),
            source_verification=verification, dual=dual,
            checkpoint_sha256={key:sha256(item[key]) for key in PATH_KEYS if key in item},
            policy_sha256=policy_digest(policies), **metadata)
    return configs, agents, provenance, sizes
