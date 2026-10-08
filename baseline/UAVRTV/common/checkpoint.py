"""Atomic episode recovery including replay, optimizers and all RNG states."""
import hashlib
import os
from pathlib import Path
import uuid
import torch
import baseline.NDTVS.api as c
from baseline.NDTVS.evaluation.checks import require, sha256

HERE = Path(__file__).resolve().parents[1]
FORMAT = "uavrtv-shared-sac-checkpoint-v1"


def source_hashes():
    current = c.source_hashes()
    for relative in ("metrics/service.py", "evaluation/scenario.py", "evaluation/checks.py", "evaluation/benchmark/radio.py"):
        p = c.HERE / relative
        current["baseline/NDTVS/" + relative] = sha256(p)
    patterns = ("*.py", "common/*.py", "environment/*.py", "models/*.py", "rewards/*.py",
                "training/*.py", "evaluation/*.py", "plot/*.py")
    for pattern in patterns:
        for p in sorted(HERE.glob(pattern)):
            if p == HERE / "config.py":
                continue
            current["baseline/UAVRTV/" + str(p.relative_to(HERE))] = sha256(p)
    return current


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def read(path, sources=True):
    b = torch.load(Path(path), map_location="cpu", weights_only=False)
    require(b.get("format") == FORMAT, "Old UAVRTV checkpoint: retrain in the shared environment")
    if sources:
        require(b["source_sha256"] == source_hashes(), "UAVRTV/shared source changed; do not bypass verification")
    return b


def policy_digest(agent):
    h = hashlib.sha256(str(agent.updates).encode())
    for name, net in agent.networks().items():
        h.update(name.encode())
        for key, tensor in sorted(net.state_dict().items()):
            h.update(key.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    h.update(agent.log_alpha.detach().cpu().numpy().tobytes())
    return h.hexdigest()
