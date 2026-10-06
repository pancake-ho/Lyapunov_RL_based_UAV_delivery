"""Configuration loading and unchanged reward identity."""
from __future__ import annotations

import json
import math
from pathlib import Path
from baseline.NDTVS.common.paths import HERE, PROPOSED
from config_hppo import HPPOConfig
from baseline.NDTVS.rewards.qoe import REWARD_VERSION
VERSION = REWARD_VERSION
BASE_COMMIT = "40f0ba62526b19eb744579dc74389f84392cca3f"


def read_config(path):
    d = json.loads(Path(path).read_text())
    d = d.get("config", d)
    for k, v in list(d.items()):
        if isinstance(v, list):
            d[k] = tuple(math.inf if x is None and k == "distance_bin_edges_m" else x for x in v)
    required = {"num_regions", "users_per_region", "rsu_total_bandwidth_hz",
                "mask_queue_actions", "delivery_mode"}
    if not required <= d.keys():
        raise ValueError("Use a complete current proposed resolved_config.json")
    cfg = HPPOConfig(**d)
    if not (cfg.mask_queue_actions and cfg.enforce_queue_admissibility
            and cfg.delivery_mode == "all_or_nothing"):
        raise ValueError("This adaptation requires current queue-masked, atomic delivery")
    return cfg
