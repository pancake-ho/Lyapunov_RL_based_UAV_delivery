"""RSU-only constraints and no-UAV frame completion."""
from __future__ import annotations

import numpy as np
from baseline.NDTVS.common.paths import HERE
from hppo.env import P3HierarchicalEnv


class RSUEnv(P3HierarchicalEnv):
    def frame_action_masks(self, region):
        masks = super().frame_action_masks(region)
        for mask in masks:
            mask[2] = False
        return masks

    def proposal(self, region, raw):
        if np.any(np.asarray(raw) == 2):
            raise ValueError("RSU-only policy requested a UAV")
        return super().proposal(region, raw)

    def begin_frame(self, raw_actions, completed_actions, completion_info=None):
        if any(a.hired or a.uav_users for a in completed_actions.values()):
            raise ValueError("UAV service is disabled")
        return super().begin_frame(raw_actions, completed_actions, completion_info)


class NoUAVCompletion:
    def __init__(self, cfg):
        self.cfg = cfg

    def select_all(self, env, raw, fast_policy, deterministic=False):
        actions = {m: env.proposal(m, a).execute(0, -1) for m, a in raw.items()}
        detail = {m: {"reason": "UAV disabled", "runtime_s": 0.0, "scenarios": 0,
                      "fast_update_count": fast_policy.update_count,
                      "selected_index": None, "candidates": []} for m in raw}
        return actions, detail
