"""Decision adapter only. Every physical transition is the shared P3 code."""
import math
import numpy as np

import baseline.NDTVS.api  # Registers the existing proposed import root.
from hppo.env import P3HierarchicalEnv
from env.p3.types import RegionAction
from env.p3.radio import rsu_link_capacity_bps
from env.p3.battery import battery_power_cap_w
from baseline.NDTVS.evaluation.scenario import PairingObserver


def token(value, count):
    return min(count - 1, max(0, int(math.floor((float(value) + 1) * .5 * count))))


class SharedUAVRTVEnv(PairingObserver, P3HierarchicalEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.obs_dim = 9 + 12 * self.N
        self.act_dim = 2 + 3 * self.N

    def priorities(self, m):
        ordered = sorted(self.region_users[m], key=lambda u: (self.state.queue[u], u))
        return tuple(ordered[:self.cfg.rsu_capacity]), tuple(
            ordered[self.cfg.rsu_capacity:self.cfg.rsu_capacity + self.cfg.uav_capacity])

    def observation(self, m, boundary):
        c, state = self.cfg, self.state
        rsu, candidates = self.priorities(m) if boundary else (
            self.region_actions[m].rsu_users, self.region_actions[m].uav_users)
        possible = bool(self.feasible_hover_points(m)) if boundary else bool(self.region_actions[m].hired)
        candidates = candidates if possible else ()
        hired = 0 if boundary else self.region_actions[m].hired
        slot = 0 if boundary else self.local_slot
        remaining = self.T - slot
        battery = float(state.battery_j[m])
        power = battery_power_cap_w(battery, remaining, c) if hired else 0.
        head = [float(boundary), self.frame / c.num_frames, slot / self.T,
                float(hired), battery / c.battery_capacity_j, power / c.uav_max_total_power_w,
                (state.uav_x[m] - c.rsu_x(m)) / c.region_length_m,
                len(self.region_users[m]) / self.N, float(possible)]
        users = np.zeros((self.N, 12), dtype=np.float32)
        mask = np.zeros(self.act_dim, dtype=np.float32)
        if boundary and possible:
            mask[:2] = 1
        for u in self.region_users[m]:
            q = float(state.queue[u])
            cap = self.guard.max_queue_admissible_chunks(c.large_queue_level - q)
            users[u] = [1., float(u in rsu), float(u in candidates),
                q / c.large_queue_level, (c.large_queue_level - q) / c.large_queue_level,
                (state.user_x[u] - c.rsu_x(m)) / c.region_length_m,
                state.user_speed[u] / c.vehicle_speed_max_mps,
                (state.last_quality_index[u] + 1) / self.K,
                float(state.last_stalled[u]), cap / c.max_chunks_per_slot,
                abs(state.user_x[u] - c.rsu_x(m)) / c.region_length_m,
                abs(state.user_x[u] - state.uav_x[m]) / c.region_length_m]
            if u in candidates and cap > 0:
                mask[[2 + u, 2 + self.N + u, 2 + 2 * self.N + u]] = 1
        obs = np.concatenate((np.asarray(head, dtype=np.float32), users.ravel()))
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite SAC observation")
        return obs, mask

    def start_from_sac(self, actions):
        raw, completed = {}, {}
        for m in self.regions:
            rsu, residual = self.priorities(m)
            feasible = self.feasible_hover_points(m)
            candidates = residual if feasible else ()
            tokens = np.zeros(self.N, dtype=np.int64)
            tokens[list(rsu)] = 1
            tokens[list(candidates)] = 2
            want = bool(actions[m][0] > 0 and feasible)
            point = feasible[token(actions[m][1], len(feasible))] if want else -1
            raw[m] = tokens
            completed[m] = RegionAction(m, int(want), point, tuple(sorted(rsu)),
                                       tuple(sorted(candidates)) if want else ())
        return self.begin_frame(raw, completed)

    def requests(self, m, continuous):
        c, a = self.cfg, np.asarray(continuous)
        if a.shape != (self.act_dim,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1 + 1e-6):
            raise ValueError("Invalid SAC continuous action")
        raw = np.zeros(3 * self.N, dtype=np.int64)
        action = self.region_actions[m]
        for u in action.rsu_users:
            cap = self.guard.max_queue_admissible_chunks(c.large_queue_level - float(self.state.queue[u]))
            capacity = rsu_link_capacity_bps(abs(c.rsu_x(m) - self.state.user_x[u]), 1., c)
            for k in reversed(range(self.K)):
                count = min(cap, int(math.floor(capacity * c.slot_duration_s / c.chunk_size_bits[k] + 1e-9)))
                if count > 0:
                    raw[u], raw[self.N + u] = count, k
                    break
        for u in action.uav_users:
            cap = self.guard.max_queue_admissible_chunks(c.large_queue_level - float(self.state.queue[u]))
            chunks = token(a[2 + u], c.max_chunks_per_slot + 1)
            # Physical-state support only; never inspect the realized fading trace.
            raw[u] = min(chunks, cap)
            if raw[u]:
                raw[self.N + u] = token(a[2 + self.N + u], self.K)
                raw[2 * self.N + u] = token(a[2 + 2 * self.N + u], c.uav_power_levels)
        return raw
