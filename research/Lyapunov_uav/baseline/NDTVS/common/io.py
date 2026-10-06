"""Atomic JSON/checkpoint writes and CSV output."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import torch
from baseline.NDTVS.common.paths import HERE
from hppo.logger import jsonable


def atomic(path, obj, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb" if binary else "w") as f:
            if binary:
                torch.save(obj, f)
            else:
                json.dump(jsonable(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
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


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_bundle(path):
    return torch.load(path, map_location="cpu", weights_only=False)
