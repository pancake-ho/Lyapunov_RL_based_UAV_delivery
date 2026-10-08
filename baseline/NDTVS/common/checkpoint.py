"""Source compatibility and model/optimizer/rollout state."""
from __future__ import annotations

import hashlib
from copy import deepcopy
import json
from pathlib import Path
from baseline.NDTVS.common.config import HPPOConfig, VERSION
from baseline.NDTVS.common.paths import HERE, PROPOSED
from baseline.NDTVS.rewards.qoe import reward_spec
from hppo.logger import jsonable
LEGACY_ADAPTER_SHA256 = "e89913ead8344a29b6b7f559f73f99ce63acfa8240e2a1f7d0a04e53b1fd62d9"
REVIEWED_CONFIG_DEFAULT_HASHES = {'proposed/config_p3.py': ['140319f04ef5a98ffa47554e5f9191b6fbdc130a24b7b397318b22a2bf58c87a', '3330aa352060a8d0df49ffe79a78345d745b476e519376ce3d72cc549238f4c6'], 'proposed/config_hppo.py': ['c519fa627706dbd387a7d26efa3e70f3c09f78df4bd310aa5bd1f12108820511', 'a0c6577998a1505ce9b0bf852f82d28309bbd342f878a46e0a39924ed4570f96']}


CORE_SOURCE_FILES = ('__init__.py', '__main__.py', 'api.py', 'cli.py', 'ndtvs_common.py', 'common/__init__.py', 'common/paths.py', 'common/config.py', 'common/io.py', 'common/runtime.py', 'common/checkpoint.py', 'environment/__init__.py', 'environment/rsu.py', 'models/__init__.py', 'models/policy.py', 'rewards/__init__.py', 'rewards/qoe.py', 'metrics/__init__.py', 'metrics/observer.py', 'training/__init__.py', 'training/rollout.py', 'training/train.py', 'evaluation/__init__.py', 'evaluation/single.py')
EVALUATOR_SOURCE_FILES = ('snr_sweep_eval.py', 'ndtvs_analysis.py', 'evaluation/__init__.py', 'evaluation/__main__.py', 'evaluation/cli.py', 'evaluation/sweep.py', 'evaluation/checks.py', 'evaluation/policies.py', 'evaluation/scenario.py', 'analysis/__init__.py', 'analysis/__main__.py', 'analysis/cli.py', 'analysis/audit.py', 'analysis/compare.py', 'plot/__init__.py', 'plot/learning_diagnostics.py')
COMPAT_FILE = "common/source_compat.json"


def source_hashes():
    files = [HERE / name for name in (*CORE_SOURCE_FILES, COMPAT_FILE)]
    for pattern in ("config*.py", "hppo/*.py", "env/p3/*.py", "agent/P3/*.py"):
        files.extend(sorted(PROPOSED.glob(pattern)))
    return {str(p.relative_to(HERE.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def _parts(mapping):
    prefix = "baseline/NDTVS/"
    return ({k: v for k, v in mapping.items() if k.startswith(prefix)},
            {k: v for k, v in mapping.items() if not k.startswith(prefix)})


def _verified_modular_sources(current):
    # Only the attached, tested refactor is eligible for old-layout loading.
    # Source checks remain strict after any later edits to implementation files.
    registry = json.loads((HERE / COMPAT_FILE).read_text())
    local, _ = _parts(current)
    local.pop("baseline/NDTVS/" + COMPAT_FILE, None)
    if local != registry["modular_sources"]:
        raise ValueError("Modular implementation differs from the verified refactor")
    return registry


def _matches_previous_v2(expected, current):
    old_local, old_shared = _parts(expected)
    new_local, new_shared = _parts(current)
    if old_shared != new_shared:
        return False
    registry = _verified_modular_sources(current)
    return old_local in (registry["previous_v2_sources"], registry["previous_flat_sources"])


def verify_shared_sources(expected, current, config):
    if expected == current:
        return "exact"
    if set(config) != set(HPPOConfig.__dataclass_fields__):
        raise ValueError("Legacy default migration requires a complete explicit config")
    if set(expected) != set(current):
        raise ValueError("Physical/control source file sets differ")
    for key in expected:
        if expected[key] != current[key] and [expected[key], current[key]] != REVIEWED_CONFIG_DEFAULT_HASHES.get(key):
            raise ValueError("Unreviewed physical/control source change: " + key)
    return "reviewed-explicit-config-default-migration"


def verify_checkpoint_source(saved, algorithm):
    """Accept exact sources, the verified v2 module move, and known HPPO v1.

    The reward and shared physical/control sources retain their strict checks.
    NDTVS checkpoints trained on the old reward are refused.
    """
    spec = saved["spec"]
    if spec["algorithm"] != algorithm:
        raise ValueError("Checkpoint algorithm mismatch")
    def canonical(mapping):
        result = {}
        for key, value in mapping.items():
            if Path(key).name == "ndtvs_common.py":
                key = "baseline/NDTVS/ndtvs_common.py"
            if key in result:
                raise ValueError("Ambiguous checkpoint source paths")
            result[key] = value
        return result
    expected, current = canonical(spec["source_sha256"]), canonical(source_hashes())
    if algorithm == "ndtvs":
        if spec.get("version") != VERSION or spec.get("qoe_definition") != reward_spec():
            raise ValueError("Old/different NDTVS reward checkpoint: retrain from scratch")
        if expected == current:
            return "exact-v2"
        if _matches_previous_v2(expected, current):
            return "verified-v2-module-refactor"
        raise ValueError("NDTVS checkpoint source mismatch")
    if expected == current:
        return "exact-v2"
    if (spec.get("version") == VERSION and spec.get("qoe_definition") == reward_spec()
            and _matches_previous_v2(expected, current)):
        return "verified-v2-module-refactor"
    adapter = "baseline/NDTVS/ndtvs_common.py"
    reward = "baseline/NDTVS/ndtvs_qoe.py"
    if (algorithm != "hppo_rsu" or spec.get("version") != "ndtvs-common-v1"
            or expected.get(adapter) != LEGACY_ADAPTER_SHA256):
        raise ValueError("Checkpoint source mismatch; unrecognized historical adapter")
    _verified_modular_sources(current)
    old_local, old_shared = _parts(expected)
    _, new_shared = _parts(current)
    if old_local != {adapter: LEGACY_ADAPTER_SHA256}:
        raise ValueError("Unrecognized HPPO v1 source files")
    verify_shared_sources(old_shared, new_shared, spec.get("config", {}))
    return "verified-hppo-v1-observer-migration"


def policy_state(agents):
    return [{"model": a.net.state_dict(), "optimizer": a.opt.state_dict(),
             "trajectories": a.trajectories, "updates": a.update_count} for a in agents]


def restore_agents(agents, payload):
    if len(agents) != len(payload):
        raise ValueError("checkpoint policy count mismatch")
    for a, p in zip(agents, payload):
        a.net.load_state_dict(p["model"])
        a.opt.load_state_dict(p["optimizer"])
        a.trajectories = p["trajectories"]
        a.update_count = p["updates"]


def resume_spec(saved, algorithm):
    """Validate the known layout-only change; leave all learning state intact."""
    spec = dict(saved["spec"])
    if spec["source_sha256"] != source_hashes():
        migration = verify_checkpoint_source(saved, algorithm)
        if migration != "verified-v2-module-refactor":
            raise ValueError("Resume is permitted only for the same v2 reward and verified layout")
        spec["source_sha256"] = source_hashes()
    return spec


def evaluation_resume_spec(previous, current):
    """Normalize only verified file-move provenance; compare other fields unchanged."""
    result = deepcopy(previous)
    if result["source_sha256"] != current["source_sha256"]:
        if not _matches_previous_v2(result["source_sha256"], current["source_sha256"]):
            return result
        result["source_sha256"] = current["source_sha256"]
        old_provenance = result.get("provenance", {})
        new_provenance = current.get("provenance", {})
        if (old_provenance.get("source_verification") == "exact-v2"
                and new_provenance.get("source_verification") == "verified-v2-module-refactor"):
            old_provenance["source_verification"] = new_provenance["source_verification"]
    return result


def paired_resume_header(previous, current):
    """Permit the tested evaluator file move, without accepting new checkpoints/seeds."""
    result = deepcopy(previous)
    old_selection, new_selection = result["selection"], current["selection"]
    if old_selection["source_sha256"] != new_selection["source_sha256"]:
        if not _matches_previous_v2(old_selection["source_sha256"], new_selection["source_sha256"]):
            return result
        old_selection["source_sha256"] = new_selection["source_sha256"]
        for name, old_provenance in old_selection["checkpoints"].items():
            new_provenance = new_selection["checkpoints"].get(name, {})
            if (old_provenance.get("source_verification") == "exact-v2"
                    and new_provenance.get("source_verification") == "verified-v2-module-refactor"):
                old_provenance["source_verification"] = new_provenance["source_verification"]
    if result["evaluators"] != current["evaluators"]:
        registry = _verified_modular_sources(new_selection["source_sha256"])
        actual = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                  for name in registry["modular_evaluators"]}
        if (result["evaluators"] not in (registry["previous_evaluators"], registry["previous_flat_evaluators"])
                or current["evaluators"] != registry["modular_evaluators"]
                or actual != registry["modular_evaluators"]):
            return result
        result["evaluators"] = current["evaluators"]
    return result
