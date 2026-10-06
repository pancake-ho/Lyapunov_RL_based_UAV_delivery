"""Load each model with its own training config; verify physical comparability."""
import math
import os
import shutil
import uuid
from pathlib import Path
from dataclasses import asdict, replace

from baseline.NDTVS.evaluation.checks import (
    require, read_json, sha256, physical,
)
from baseline.NDTVS.evaluation.policies import verify_best, policy_digest


PATH_KEYS = (
    "config",
    "runtime",
    "resume_checkpoint",
    "frame_checkpoint",
    "slot_checkpoint",
    "checkpoint",
)


def freeze_models(s, c):
    """Freeze selected inputs once; resume always uses that same snapshot."""
    manifest = s.OUT / "inputs/manifest.json"
    requested = [
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in item.items()
        }
        for item in s.MODELS
    ]

    if manifest.exists():
        saved = read_json(manifest)
        require(
            saved["requested"] == requested,
            "Model selection changed; choose a new OUT",
        )
        for item in saved["files"]:
            require(
                sha256(item["snapshot"]) == item["sha256"],
                "Frozen checkpoint/config changed",
            )
        return saved["models"]

    originals = []
    for model in s.MODELS:
        if "completion_status" in model:
            status = read_json(Path(model["completion_status"]))
            require(
                status.get("status", "").lower() == "complete"
                and not status.get("final_validation_pending", False),
                f"Training/final validation not complete: {model['name']}",
            )

        for key in PATH_KEYS:
            if key in model:
                path = Path(model[key]).resolve()
                originals.append(
                    (model["name"], key, path, sha256(path))
                )

    root = s.OUT / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    frozen, files = [], []

    for model in s.MODELS:
        item = {
            k: v
            for k, v in model.items()
            if k not in PATH_KEYS and k != "completion_status"
        }

        for name, key, source, fingerprint in originals:
            if name != model["name"]:
                continue

            target = root / name / (key + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(
                target.name + "." + uuid.uuid4().hex + ".tmp"
            )

            try:
                shutil.copyfile(source, tmp)
                require(
                    sha256(tmp) == fingerprint == sha256(source),
                    "Input changed during snapshot; retry after training stops",
                )
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)

            item[key] = str(target.resolve())
            files.append(
                dict(
                    original=str(source),
                    snapshot=str(target.resolve()),
                    sha256=fingerprint,
                )
            )

        frozen.append(item)

    require(
        all(
            sha256(path) == fingerprint
            for _, _, path, fingerprint in originals
        ),
        "Inputs changed during snapshot",
    )
    c.atomic(
        manifest,
        dict(requested=requested, models=frozen, files=files),
    )
    return frozen


def configuration(d, c):
    converted = {
        k: tuple(
            math.inf if x is None and k == "distance_bin_edges_m" else x
            for x in v
        ) if isinstance(v, list) else v
        for k, v in d.items()
    }
    return c.HPPOConfig(**converted)


def proposed_sources(hashes, shared, raw_config, c):
    require(
        isinstance(hashes, dict) and hashes,
        "Missing proposed source hashes",
    )
    expected = {}

    for key, value in hashes.items():
        key = (
            key if key.startswith("proposed/")
            else "proposed/" + key
        )
        require(key not in expected, "Ambiguous proposed source paths")
        expected[key] = value

    return c.verify_shared_sources(expected, shared, raw_config)


def load_proposed(item, device, shared, c):
    """Read either an atomic resume bundle or a verified legacy policy pair."""
    from hppo.ppo import ARCHITECTURE_VERSION
    from hppo.resume import FORMAT

    run_cfg = c.read_config(Path(item["config"]))
    raw = read_json(Path(item["config"]))
    runtime = read_json(Path(item["runtime"]))

    verification = proposed_sources(
        runtime.get("code_sha256"),
        shared,
        raw.get("config", raw),
        c,
    )

    if "resume_checkpoint" not in item:
        policies = c.make_agents(
            replace(run_cfg, device=device), "proposed"
        )
        extras = [
            agent.load(Path(item[key]))
            for agent, key in zip(
                policies, ("frame_checkpoint", "slot_checkpoint")
            )
        ]
        require(
            extras[0].get("pair_id") and extras[0] == extras[1],
            "Proposed checkpoint pair mismatch",
        )
        return (
            run_cfg,
            policies,
            int(extras[0]["episode"]) + 1,
            extras[0].get("dual_lambda_z", run_cfg.dual_init),
            verification,
            {"pair": extras[0]},
        )

    bundle = c.torch.load(
        Path(item["resume_checkpoint"]),
        map_location="cpu",
        weights_only=False,
    )
    require(
        isinstance(bundle, dict) and bundle.get("format") == FORMAT,
        "Unsupported HPPO resume checkpoint format",
    )

    raw_cfg = bundle["config"]
    require(
        set(raw_cfg) == set(c.HPPOConfig.__dataclass_fields__),
        "Resume checkpoint must contain a complete explicit HPPO config",
    )
    cfg = configuration(raw_cfg, c)

    require(
        cfg.mask_queue_actions
        and cfg.enforce_queue_admissibility
        and cfg.delivery_mode == "all_or_nothing",
        "Incompatible resume queue/delivery settings",
    )

    saved_cfg = physical(c.jsonable(asdict(cfg)))
    resolved_cfg = physical(c.jsonable(asdict(run_cfg)))
    differences = sorted(
        key
        for key in saved_cfg.keys() | resolved_cfg.keys()
        if saved_cfg.get(key) != resolved_cfg.get(key)
    )
    require(
        not differences
        and tuple(cfg.hidden_dims) == tuple(run_cfg.hidden_dims),
        "Resume/resolved config mismatch: "
        + ", ".join(differences or ["hidden_dims"]),
    )

    bundle_verification = proposed_sources(
        bundle.get("code_sha256"), shared, raw_cfg, c
    )

    origin, target, trained = (
        bundle[key] for key in ("origin", "target", "next_episode")
    )
    require(
        all(type(value) is int for value in (origin, target, trained))
        and origin >= 0
        and target > 0
        and origin <= trained <= origin + target,
        "Invalid resume episode counters",
    )

    dual = float(bundle["dual"])
    require(
        math.isfinite(dual) and dual >= 0,
        "Invalid resume dual multiplier",
    )

    policies = c.make_agents(
        replace(cfg, device=device), "proposed"
    )
    pending = {}

    for agent, key in zip(policies, ("frame", "slot")):
        saved = bundle[key]
        require(
            saved.get("architecture") == ARCHITECTURE_VERSION
            and saved.get("name") == agent.name,
            "Resume architecture/policy role mismatch",
        )
        require(
            all(
                c.torch.isfinite(value).all()
                for value in saved["model"].values()
            ),
            "Nonfinite resume weights",
        )

        agent.net.load_state_dict(saved["model"], strict=True)

        require(
            type(saved["update_count"]) is int
            and saved["update_count"] >= 0,
            "Invalid resume PPO update count",
        )
        agent.update_count = saved["update_count"]
        pending[key] = sum(
            len(trajectory)
            for trajectory in saved.get("trajectories", {}).values()
        )

        # Evaluation uses fixed weights. Do not restore optimizers,
        # pending PPO trajectories or training RNG, and do not update.

    metadata = dict(
        resume=dict(
            format=bundle["format"],
            origin=origin,
            target=target,
            next_episode=trained,
            completed_since_origin=trained - origin,
            pending_training_transitions=pending,
            source_verification=bundle_verification,
        )
    )
    return cfg, policies, trained, dual, verification, metadata


def capacity_report(agents, name):
    reports = []

    for agent in agents:
        groups = dict(actor_only=0, critic_only=0, shared=0)
        ndtvs = agent.name == "ndtvs_ppo"

        for key, parameter in agent.net.named_parameters():
            if ndtvs:
                group = (
                    "critic_only"
                    if key.startswith(("critic.", "value_head."))
                    else "actor_only"
                )
            else:
                group = (
                    "shared" if key.startswith("trunk.")
                    else "critic_only" if key.startswith("critic.")
                    else "actor_only"
                )
            groups[group] += parameter.numel()

        total = sum(
            parameter.numel()
            for parameter in agent.net.parameters()
        )
        require(
            sum(groups.values()) == total,
            "Parameter accounting mismatch",
        )
        reports.append(
            dict(
                model=name,
                policy=agent.name,
                architecture=type(agent.net).__name__,
                hidden_dims=list(agent.cfg.hidden_dims),
                parameters=total,
                trained_ppo_updates=agent.update_count,
                trainable_parameters=sum(
                    parameter.numel()
                    for parameter in agent.net.parameters()
                    if parameter.requires_grad
                ),
                weight_bytes=sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in agent.net.parameters()
                ),
                **groups,
            )
        )

    return reports


def load_models(models, device, c):
    configs, agents, provenance, sizes = {}, {}, {}, []
    reference = None
    shared = {
        key: value
        for key, value in c.source_hashes().items()
        if key.startswith("proposed/")
    }

    for item in models:
        name, algorithm = item["name"], item["algorithm"]

        if algorithm == "proposed":
            (
                cfg, policies, trained, dual, verification, metadata
            ) = load_proposed(item, device, shared, c)
        else:
            saved = c.load_bundle(Path(item["checkpoint"]))
            verification = c.verify_checkpoint_source(saved, algorithm)
            verify_best(saved)
            cfg = configuration(saved["spec"]["config"], c)
            policies = c.make_agents(
                replace(cfg, device=device), algorithm
            )
            c.restore_agents(policies, saved["policies"])
            trained = saved["next_episode"]
            dual = saved.get("dual", cfg.dual_init)
            metadata = {
                "training_reward_definition": saved["spec"].get(
                    "qoe_definition"
                ),
                "validation_best_score": saved["best"],
            }

        if "expected_v" in item:
            require(
                cfg.lyapunov_v == item["expected_v"],
                f"Wrong V in {name}",
            )

        current = physical(c.jsonable(asdict(cfg)))
        current.pop("lyapunov_v", None)

        if reference is not None:
            differences = sorted(
                key
                for key in reference.keys() | current.keys()
                if reference.get(key) != current.get(key)
            )
            require(
                not differences,
                "Physical/control settings differ: "
                + ", ".join(differences),
            )
        reference = current

        require(
            cfg.num_quality_levels == 4,
            "Expected four PSNR quality levels",
        )
        for agent in policies:
            agent.net.eval()
            require(
                all(
                    c.torch.isfinite(value).all()
                    for value in agent.net.state_dict().values()
                ),
                "Nonfinite weights",
            )

        configs[name], agents[name] = cfg, policies
        sizes.extend(capacity_report(policies, name))
        provenance[name] = dict(
            algorithm=algorithm,
            training_v=cfg.lyapunov_v,
            selected_training_episode=trained,
            selection=item.get("selection", "user-selected"),
            source_verification=verification,
            dual=dual,
            checkpoint_sha256={
                key: sha256(item[key])
                for key in PATH_KEYS
                if key in item
            },
            policy_sha256=policy_digest(policies),
            **metadata,
        )

    return configs, agents, provenance, sizes