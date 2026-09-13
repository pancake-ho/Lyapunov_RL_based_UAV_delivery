"""
Vehicular video-delivery environment (scenario fixed by the research guide)
with the decision interface of the reference paper
(Wu et al., "UAV-Assisted Real-Time Video Transmission for Vehicles: A SAC DRL
Approach", IEEE IoT-J 2024).

Scenario (fixed)
----------------
* M RSUs on a straight urban corridor, region m = [m*spacing, (m+1)*spacing).
* Each RSU serves at most J_R users with fixed reserved resource blocks
  (W_R/J_R, P_R/J_R).  A persistent UAV per region can be hired per frame
  (price c_H) and serves at most J_U users.
* Two time scales: frame r = T slots of length Delta.  User membership N_m(r),
  hiring and RSU/UAV association are fixed at frame start; chunk/quality/
  power decisions are updated every slot.
* Playback queue Q_n(t+1) = [Q_n(t) - b]^+ + d_n(t), stall = 1{Q_n(t) < b},
  virtual queue Z_n = Qe - Q_n.
* Physical UAV battery E_u(t) [J] with automatic-return reserve E_th,
  charging at the depot (= RSU position) when not hired.

Reference-paper decision interface (per region, per slot)
---------------------------------------------------------
action = [v_x, v_y, hire, bw_1..bw_Nmax, k_1..k_Nmax, l_1..l_Nmax]  in [-1, 1]
* v_x, v_y : UAV horizontal velocity (continuous trajectory, rotary-wing energy model)
* hire     : used only at the first slot of a frame (hire if > 0)
* bw_i     : soft-max bandwidth share among UAV-scheduled users (bw_mode=sac)
* k_i      : SVC layer / quality level of user slot i
* l_i      : number of chunks requested for user slot i (0..L_max)

Reward per region per slot (reference eq. 13-14 adapted to the scenario):
   r = beta * quality - delta * |bitrate switch| - phi * stall
       - varsigma * E_uav(t) - lambda_H * c_H / T * hired
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

C_LIGHT = 3e8


@dataclass
class Vehicle:
    vid: int
    x: float
    y: float
    v: float                # signed speed along x [m/s]
    Q: float                # playback buffer [chunks]
    region: int             # region membership (fixed during frame)
    last_k: int = 0         # last delivered quality level (0 = nothing yet)
    slot: int = -1          # observation/action slot index in region (-1: none)
    served_by: str = "none"  # 'rsu' | 'uav' | 'none'
    stalls: int = 0
    chunks_recv: int = 0
    last_stall: int = 0
    last_d: int = 0
    last_C: float = 0.0     # last link capacity [bps] of serving link

    def Z(self, Q_tilde):
        return Q_tilde - self.Q


@dataclass
class UAV:
    region: int
    pos: np.ndarray        # (x, y)
    vel: np.ndarray        # (vx, vy)
    E: float               # battery [J]
    hired: bool = False
    forced_return: bool = False
    users: List[int] = field(default_factory=list)
    prop_power: float = 0.0
    e_slot: float = 0.0     # energy consumed in the last slot [J]


@dataclass
class FrameState:
    slots: List[Optional[int]]     # slot index -> vid
    members: List[int]             # N_m(r)
    rsu_users: List[int]
    uav_users: List[int]
    overflow: List[int]            # members without a slot (> N_max)
    hire_feasible: bool = True
    hire_reason: str = ""


class VehicularVideoEnv:
    # ------------------------------------------------------------------ init
    def __init__(self, args, seed: int = 0):
        a = self.a = args
        self.rng = np.random.default_rng(seed)
        self.M = a.num_rsu
        self.spacing = a.rsu_spacing
        self.road_len = self.M * self.spacing
        self.rsu_xy = np.array([[(m + 0.5) * self.spacing, 0.0] for m in range(self.M)])
        self.Nmax = a.max_users_per_region
        self.T = a.slots_per_frame
        self.dt = a.slot_duration
        self.K = len(a.bitrates_mbps)
        self.R_bps = np.array(a.bitrates_mbps) * 1e6
        self.S = self.R_bps * a.chunk_duration           # chunk size [bits]
        self.U = np.array(a.utilities)
        self.Umax = float(self.U.max())
        self.Lmax = a.L_max
        self.b = a.b
        self.Qe = a.Q_tilde

        # radio
        self.N0 = 10 ** ((a.N0_dBm_Hz - 30) / 10)
        self.Gamma = 10 ** (a.snr_gap_dB / 10)
        fs = (C_LIGHT / (4 * math.pi * a.fc)) ** 2
        self.beta_R = fs if a.beta_R_dB is None else 10 ** (a.beta_R_dB / 10)
        self.beta_U = fs if a.beta_U_dB is None else 10 ** (a.beta_U_dB / 10)
        self.W_R_s = a.W_R / a.J_R      # per-session RSU bandwidth (fixed RB)
        self.P_R_s = a.P_R / a.J_R
        self.W_U_s = a.W_U / a.J_U      # fixed-RB fallback for UAV
        self.P_U_s = a.P_U_max / a.J_U

        # battery / energy
        self.E_max = a.E_max_Wh * 3600.0
        self.E_th = a.E_th_frac * self.E_max
        self.e_ch = a.eta_ch * a.P_ch * self.dt
        vs = np.linspace(0, a.v_max, 201)
        self.P_prop_max = float(max(self.propulsion_power(v) for v in vs))
        self.P_hover = self.propulsion_power(0.0)

        # dims
        self.act_dim = 3 + 3 * self.Nmax
        self.user_feat = 10
        self.obs_dim = 8 + self.user_feat * self.Nmax

        self.vehicles: Dict[int, Vehicle] = {}
        self.uavs: List[UAV] = []
        self.frame_state: List[FrameState] = []
        self._next_vid = 0
        self.t = 0
        self.frame = 0
        self.slot_in_frame = 0

    # --------------------------------------------------------------- models
    def propulsion_power(self, v: float) -> float:
        a = self.a
        blade = a.P0 * (1 + 3 * v ** 2 / a.U_tip ** 2)
        induced = a.Pi * math.sqrt(max(math.sqrt(1 + v ** 4 / (4 * a.v0 ** 4)) - v ** 2 / (2 * a.v0 ** 2), 0.0))
        parasite = 0.5 * a.d0 * a.rho * a.rotor_solidity * a.rotor_area * v ** 3
        return blade + induced + parasite

    def _fading(self):
        if self.a.fading == "rayleigh":
            return float(self.rng.exponential(1.0))
        return 1.0

    def gain_rsu(self, m, veh, fading=True):
        d = math.sqrt((self.a.rsu_height - self.a.user_height) ** 2 +
                      (self.rsu_xy[m, 0] - veh.x) ** 2 + (self.rsu_xy[m, 1] - veh.y) ** 2)
        xi = self._fading() if fading else 1.0
        return self.beta_R * xi / d ** self.a.alpha_R, d

    def gain_uav(self, uav, veh, fading=True):
        d = math.sqrt((self.a.uav_altitude - self.a.user_height) ** 2 +
                      (uav.pos[0] - veh.x) ** 2 + (uav.pos[1] - veh.y) ** 2)
        xi = self._fading() if fading else 1.0
        return self.beta_U * xi / d ** self.a.alpha_U, d

    def rate(self, W, P, g):
        if W <= 0 or P <= 0:
            return 0.0
        return W * math.log2(1 + P * g / (self.Gamma * self.N0 * W))

    def region_of(self, x):
        return int(min(max(x // self.spacing, 0), self.M - 1))

    # ---------------------------------------------------------------- reset
    def reset(self):
        a = self.a
        self.vehicles = {}
        self._next_vid = 0
        self.t = 0
        self.frame = 0
        self.slot_in_frame = 0
        for m in range(self.M):
            for _ in range(a.init_vehicles_per_region):
                x = float(self.rng.uniform(m * self.spacing, (m + 1) * self.spacing))
                self._spawn(x=x, direction=int(self.rng.choice([-1, 1])))
        self.uavs = [UAV(region=m, pos=self.rsu_xy[m].copy(), vel=np.zeros(2), E=self.E_max)
                     for m in range(self.M)]
        self.frame_state = [FrameState([None] * self.Nmax, [], [], [], []) for _ in range(self.M)]
        self._assign_membership()
        return self._observe()

    def _spawn(self, x, direction):
        a = self.a
        speed = float(self.rng.uniform(a.veh_speed_min, a.veh_speed_max))
        v = Vehicle(vid=self._next_vid, x=x, y=direction * a.lane_y, v=direction * speed,
                    Q=float(a.Q_init), region=self.region_of(x))
        self.vehicles[v.vid] = v
        self._next_vid += 1
        return v

    # ----------------------------------------------------------- frame logic
    def _assign_membership(self):
        """N_m(r) and observation slots, fixed for the coming frame."""
        for m in range(self.M):
            members = sorted(v.vid for v in self.vehicles.values() if self.region_of(v.x) == m)
            fs = self.frame_state[m]
            fs.members = members
            fs.slots = [None] * self.Nmax
            fs.overflow = []
            for v in self.vehicles.values():
                if v.vid in members:
                    v.region = m
            for i, vid in enumerate(members):
                if i < self.Nmax:
                    fs.slots[i] = vid
                    self.vehicles[vid].slot = i
                else:
                    fs.overflow.append(vid)
                    self.vehicles[vid].slot = -1

    def _frame_decisions(self, actions):
        """Hiring + association at the first slot of a frame (uses the action's hire component)."""
        a = self.a
        for m in range(self.M):
            fs = self.frame_state[m]
            uav = self.uavs[m]
            uav.forced_return = False
            slotted = [vid for vid in fs.slots if vid is not None and vid in self.vehicles]

            # --- hiring decision
            if a.hire_mode == "sac":
                want = bool(actions[m][2] > 0)
            elif a.hire_mode == "always":
                want = True
            elif a.hire_mode == "never":
                want = False
            else:  # threshold
                want = len(fs.members) > a.J_R
            need = self.E_th + self.T * self.P_prop_max * self.dt
            fs.hire_feasible = uav.E >= need
            if want and not fs.hire_feasible:
                fs.hire_reason = f"battery {uav.E:.0f}J < required {need:.0f}J -> charging"
                want = False
            elif want:
                fs.hire_reason = "hired"
            else:
                fs.hire_reason = "not hired"
            was_hired = uav.hired
            uav.hired = want
            if not uav.hired:
                if np.linalg.norm(uav.pos - self.rsu_xy[m]) > 1e-6:
                    uav.E = max(uav.E - a.e_rel, 0.0)
                    fs.hire_reason += " (returned to depot, e_rel charged)"
                uav.pos = self.rsu_xy[m].copy()
                uav.vel = np.zeros(2)
            elif not was_hired:
                uav.pos = self.rsu_xy[m].copy()   # takes off from the depot
                uav.vel = np.zeros(2)

            # --- association: most urgent (smallest Q) first; RSU takes J_R, UAV takes J_U
            order = sorted(slotted, key=lambda vid: (self.vehicles[vid].Q, vid))
            fs.rsu_users = order[:a.J_R]
            rest = order[a.J_R:]
            fs.uav_users = rest[:a.J_U] if uav.hired else []
            uav.users = list(fs.uav_users)
            for vid in fs.members:
                if vid in self.vehicles:
                    self.vehicles[vid].served_by = "none"
            for vid in fs.rsu_users:
                self.vehicles[vid].served_by = "rsu"
            for vid in fs.uav_users:
                self.vehicles[vid].served_by = "uav"

    # ------------------------------------------------------------- helpers
    def _decode_action(self, act):
        """Map [-1,1] action vector to physical decisions."""
        act = np.clip(np.asarray(act, dtype=float), -1.0, 1.0)
        vel = act[:2] * self.a.v_max
        n = np.linalg.norm(vel)
        if n > self.a.v_max:
            vel = vel * self.a.v_max / n
        bw_logits = act[3:3 + self.Nmax]
        k_raw = act[3 + self.Nmax:3 + 2 * self.Nmax]
        l_raw = act[3 + 2 * self.Nmax:3 + 3 * self.Nmax]
        k = np.clip(np.rint((k_raw + 1) / 2 * (self.K - 1)) + 1, 1, self.K).astype(int)
        l = np.clip(np.rint((l_raw + 1) / 2 * self.Lmax), 0, self.Lmax).astype(int)
        return vel, bw_logits, k, l

    def _room(self, veh):
        """Buffer room so that Q(t+1) <= Qe."""
        return max(0, int(self.Qe - max(veh.Q - self.b, 0)))

    def _veh_snapshot(self, veh, uav=None):
        s = {"vid": veh.vid, "x": round(veh.x, 1), "y": round(veh.y, 1), "v": round(veh.v, 1),
             "Q": veh.Q, "Z": veh.Z(self.Qe), "last_k": veh.last_k, "served_by": veh.served_by,
             "slot": veh.slot, "region_now": self.region_of(veh.x)}
        return s

    # ----------------------------------------------------------------- step
    def step(self, actions: Dict[int, np.ndarray]):
        a = self.a
        if self.slot_in_frame == 0:
            self._frame_decisions(actions)

        rewards = {}
        logs = {}
        delivered = {}        # vid -> (d, k)
        for m in range(self.M):
            uav = self.uavs[m]
            fs = self.frame_state[m]
            act = actions[m]
            vel_cmd, bw_logits, k_vec, l_vec = self._decode_action(act)
            region_vehicles = [self.vehicles[vid] for vid in fs.members if vid in self.vehicles]
            # vehicles that arrived mid-frame and are physically in this region (unserved this frame)
            newcomers = [v for v in self.vehicles.values()
                         if v.region == m and v.vid not in fs.members]
            before = [self._veh_snapshot(v) for v in region_vehicles + newcomers]

            log = {"region": m, "uav_before": {"hired": uav.hired, "pos": uav.pos.round(1).tolist(),
                                                "vel": uav.vel.round(2).tolist(), "E_J": round(uav.E, 1),
                                                "soc": round(uav.E / self.E_max, 3)},
                   "frame_state": {"members": list(fs.members), "rsu_users": list(fs.rsu_users),
                                    "uav_users": list(fs.uav_users), "overflow": list(fs.overflow),
                                    "hire_reason": fs.hire_reason},
                   "vehicles_before": before, "action_raw": np.asarray(act).round(3).tolist(),
                   "rsu_decisions": [], "uav_decisions": [], "events": []}

            energy_J = 0.0
            uav_dec = []
            # ---------------- UAV -----------------------------------------
            if uav.hired:
                remaining = self.T - self.slot_in_frame
                need = self.E_th + remaining * self.P_prop_max * self.dt
                if uav.E < need:
                    # automatic return: reserve would be violated
                    uav.hired = False
                    uav.forced_return = True
                    uav.E = max(uav.E - a.e_rel, 0.0)
                    uav.pos = self.rsu_xy[m].copy()
                    uav.vel = np.zeros(2)
                    for vid in fs.uav_users:
                        if vid in self.vehicles:
                            self.vehicles[vid].served_by = "none"
                    fs.uav_users = []
                    uav.users = []
                    log["events"].append(f"FORCED_RETURN: E={uav.E:.0f}J < reserve {need:.0f}J")

            if uav.hired:
                # move (reference: continuous trajectory), bounded flight zone
                new_pos = uav.pos + vel_cmd * self.dt
                xlo = m * self.spacing - a.flight_x_margin
                xhi = (m + 1) * self.spacing + a.flight_x_margin
                new_pos[0] = float(np.clip(new_pos[0], xlo, xhi))
                new_pos[1] = float(np.clip(new_pos[1], -a.flight_y_max, a.flight_y_max))
                uav.vel = (new_pos - uav.pos) / self.dt
                uav.pos = new_pos
                speed = float(np.linalg.norm(uav.vel))
                uav.prop_power = self.propulsion_power(speed)
                energy_J += uav.prop_power * self.dt

                alive = [vid for vid in fs.uav_users if vid in self.vehicles]
                if a.bw_mode == "sac" and alive:
                    logits = np.array([bw_logits[self.vehicles[vid].slot] for vid in alive]) * 3.0
                    frac = np.exp(logits - logits.max())
                    frac = frac / frac.sum()
                else:
                    frac = np.full(len(alive), 1.0 / a.J_U)
                p_sum = 0.0
                for vid, f in zip(alive, frac):
                    veh = self.vehicles[vid]
                    W = a.W_U * f if a.bw_mode == "sac" else self.W_U_s
                    P = a.P_U_max * f if a.bw_mode == "sac" else self.P_U_s
                    g, d = self.gain_uav(uav, veh)
                    C = self.rate(W, P, g)
                    k = int(k_vec[veh.slot])
                    l_req = int(l_vec[veh.slot])
                    l_max = int(C * self.dt // self.S[k - 1])
                    room = self._room(veh)
                    l = min(l_req, l_max, room)
                    p_sum += P
                    delivered[vid] = (l, k)
                    veh.last_C = C
                    uav_dec.append({"vid": vid, "bw_frac": round(float(f), 3), "W_MHz": round(W / 1e6, 3),
                                    "p_W": round(P, 3), "dist_m": round(d, 1),
                                    "snr_dB": round(10 * math.log10(max(P * g / (self.Gamma * self.N0 * max(W, 1)), 1e-30)), 1),
                                    "C_Mbps": round(C / 1e6, 2), "k": k, "l_req": l_req,
                                    "l_max": l_max, "room": room, "l_sent": l})
                e_com = self.dt / a.eta_PA * p_sum
                energy_J += e_com
                uav.E = max(uav.E - energy_J, 0.0)
                log["uav_move"] = {"vel_cmd": vel_cmd.round(2).tolist(), "vel": uav.vel.round(2).tolist(),
                                   "speed": round(speed, 2), "pos": uav.pos.round(1).tolist(),
                                   "P_prop_W": round(uav.prop_power, 1), "e_com_J": round(e_com, 2),
                                   "e_slot_J": round(energy_J, 1)}
            else:
                uav.prop_power = 0.0
                charge = min(self.e_ch, self.E_max - uav.E)
                uav.E += charge
                log["uav_move"] = {"charging_J": round(charge, 1), "pos": uav.pos.round(1).tolist()}
            uav.e_slot = energy_J
            log["uav_decisions"] = uav_dec

            # ---------------- RSU (rule-based, not the agent) -------------
            rsu_dec = []
            for vid in fs.rsu_users:
                if vid not in self.vehicles:
                    continue
                veh = self.vehicles[vid]
                g, d = self.gain_rsu(m, veh)
                C = self.rate(self.W_R_s, self.P_R_s, g)
                lmax_k = [int(C * self.dt // self.S[kk]) for kk in range(self.K)]
                room = self._room(veh)
                k, l = 0, 0
                if room > 0:
                    if a.rsu_rule == "maxq":
                        feas = [kk for kk in range(self.K) if lmax_k[kk] >= 1]
                        if feas:
                            k = max(feas) + 1
                            l = min(self.Lmax, lmax_k[k - 1], room)
                    else:  # maxchunks (ties -> higher quality)
                        best = max(range(self.K), key=lambda kk: (min(self.Lmax, lmax_k[kk]), kk))
                        if lmax_k[best] >= 1:
                            k = best + 1
                            l = min(self.Lmax, lmax_k[best], room)
                delivered[vid] = (l, k)
                veh.last_C = C
                rsu_dec.append({"vid": vid, "dist_m": round(d, 1), "C_Mbps": round(C / 1e6, 2),
                                "l_max_per_k": lmax_k, "room": room, "k": k, "l_sent": l})
            log["rsu_decisions"] = rsu_dec
            logs[m] = log

        # ---------------- playback queues, stalls, rewards ----------------
        comp = {m: {"quality": 0.0, "switch": 0.0, "stall": 0.0, "energy": 0.0, "hire": 0.0} for m in range(self.M)}
        after = {m: [] for m in range(self.M)}
        for veh in self.vehicles.values():
            m = veh.region
            l, k = delivered.get(veh.vid, (0, 0))
            stall = 1 if veh.Q < self.b else 0
            veh.stalls += stall
            veh.last_stall = stall
            veh.last_d = l
            veh.chunks_recv += l
            switch_bps = 0.0
            if l > 0:
                if veh.last_k > 0 and veh.last_k != k:
                    switch_bps = abs(self.R_bps[k - 1] - self.R_bps[veh.last_k - 1])
                veh.last_k = k
            in_scope = (self.a.reward_scope == "region") or (veh.served_by == "uav")
            if in_scope:
                if l > 0:
                    if self.a.quality_reward == "per_chunk":
                        comp[m]["quality"] += self.a.beta * l * self.U[k - 1]
                    else:
                        comp[m]["quality"] += self.a.beta * self.U[k - 1]
                comp[m]["switch"] += self.a.delta * switch_bps
                comp[m]["stall"] += self.a.phi * stall
            Q_prev = veh.Q
            veh.Q = max(veh.Q - self.b, 0) + l
            after[m].append({"vid": veh.vid, "Q_prev": Q_prev, "stall": stall, "d": l, "k": k,
                             "switch_bps": switch_bps, "Q_next": veh.Q, "Z_next": veh.Z(self.Qe),
                             "served_by": veh.served_by})
        for m in range(self.M):
            uav = self.uavs[m]
            comp[m]["energy"] = self.a.varsigma * uav.e_slot
            comp[m]["hire"] = (self.a.lambda_H * self.a.c_H / self.T) if uav.hired else 0.0
            r = comp[m]["quality"] - comp[m]["switch"] - comp[m]["stall"] - comp[m]["energy"] - comp[m]["hire"]
            rewards[m] = float(r)
            logs[m]["reward"] = {"total": round(r, 4), **{kk: round(vv, 4) for kk, vv in comp[m].items()}}
            logs[m]["vehicles_after"] = sorted(after[m], key=lambda d: d["vid"])
            logs[m]["uav_after"] = {"hired": uav.hired, "pos": uav.pos.round(1).tolist(),
                                     "E_J": round(uav.E, 1), "soc": round(uav.E / self.E_max, 3),
                                     "forced_return": uav.forced_return}

        # ---------------- mobility -----------------------------------------
        departed = []
        for veh in list(self.vehicles.values()):
            speed = veh.v
            if self.a.hotspot_region >= 0 and self.region_of(veh.x) == self.a.hotspot_region:
                speed = veh.v * self.a.hotspot_speed_factor
            veh.x += speed * self.dt
            if veh.x < 0 or veh.x > self.road_len:
                departed.append(veh.vid)
        for vid in departed:
            veh = self.vehicles.pop(vid)
            fs = self.frame_state[veh.region]
            if veh.slot >= 0 and fs.slots[veh.slot] == vid:
                fs.slots[veh.slot] = None
            logs[veh.region]["events"].append(f"DEPART vid={vid}")
        arrivals = []
        for direction, x0 in ((1, 0.0), (-1, self.road_len)):
            n_new = self.rng.poisson(self.a.arrival_rate * self.dt)
            for _ in range(n_new):
                v = self._spawn(x=x0 + direction * 1e-3, direction=direction)
                arrivals.append(v.vid)
                logs[v.region]["events"].append(f"ARRIVE vid={v.vid} (unserved until next frame)")
        for m in range(self.M):
            logs[m]["vehicles_moved"] = [{"vid": v.vid, "x": round(v.x, 1), "region_now": self.region_of(v.x)}
                                         for v in self.vehicles.values() if v.region == m]

        # ---------------- time bookkeeping -----------------------------------
        self.t += 1
        self.slot_in_frame += 1
        done = False
        if self.slot_in_frame == self.T:
            self.slot_in_frame = 0
            self.frame += 1
            if self.frame >= self.a.frames_per_episode:
                done = True
            else:
                self._assign_membership()      # new N_m(r), new slots (hire/association at next step)
        obs = self._observe()
        info = {"t": self.t, "frame": self.frame, "slot_in_frame": self.slot_in_frame, "region_logs": logs,
                "n_vehicles": len(self.vehicles)}
        return obs, rewards, done, info

    # ----------------------------------------------------------- observation
    def _observe(self):
        a = self.a
        obs = {}
        for m in range(self.M):
            uav = self.uavs[m]
            fs = self.frame_state[m]
            n_members = sum(1 for vid in fs.slots if vid is not None and vid in self.vehicles)
            head = [(uav.pos[0] - self.rsu_xy[m, 0]) / self.spacing,
                    uav.pos[1] / a.flight_y_max,
                    uav.vel[0] / a.v_max, uav.vel[1] / a.v_max,
                    uav.E / self.E_max, float(uav.hired),
                    self.slot_in_frame / self.T, n_members / self.Nmax]
            feats = np.zeros((self.Nmax, self.user_feat))
            for i, vid in enumerate(fs.slots):
                if vid is None or vid not in self.vehicles:
                    continue
                veh = self.vehicles[vid]
                g, _ = self.gain_uav(uav, veh, fading=False)
                feats[i] = [1.0, float(veh.served_by == "uav"), float(veh.served_by == "rsu"),
                            (veh.x - uav.pos[0]) / self.spacing, (veh.y - uav.pos[1]) / a.flight_y_max,
                            veh.v / a.veh_speed_max, veh.Q / self.Qe, veh.last_k / self.K,
                            (10 * math.log10(g) + 120) / 60.0, float(veh.last_stall)]
            obs[m] = np.concatenate([np.array(head, dtype=np.float32), feats.astype(np.float32).ravel()])
        return obs

    # --------------------------------------------------------------- metrics
    def episode_metrics(self):
        vs = list(self.vehicles.values())
        return {"n_vehicles_alive": len(vs)}
