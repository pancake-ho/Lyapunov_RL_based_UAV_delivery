from __future__ import annotations

"""Offline verification of a hierarchical-PPO history trace.

Reads ``trace.jsonl`` (+ ``resolved_config.json`` in the same directory) and
re-derives, for every frame and slot, the P3 equations from the logged
state/decision/update quantities.  A ``debug.log`` shows *what* happened; this
script proves the transitions are consistent with the formulation:

  Q1  Z_n(t) = Q^e - Q_n(t)                                   (3.12)
  Q2  Q(t+1) = Q(t) - min(Q(t), b) + d(t)                     (3.9)-(3.10)
  Q3  Q/Z continuity between consecutive slots and frames
  R1  d(t) <= l_req <= L_max, d(t) <= feasible, d(t)*S_k <= C*Delta  (3.1)-(3.2)
  R2  logged capacity == recomputed capacity from distance/fading/power (2.12), (2.15)
  R3  UAV: p_min(l,k) <= p_exec when d(t)>0                    (7.3)
  R4  unscheduled users deliver nothing; provider matches association
  A1  |RSU users| <= J^R, |UAV users| <= J^U, single provider, no UAV users if mu=0  (2.5)-(2.7)
  A2  membership fixed within a frame and equal to floor(x/L) at frame start
  P1  p_req = level/(levels-1)*P^U_max, p_exec = scale*p_req, sum p_exec <= P_eff  (3.3)
  P2  P_eff == min(P^U_max, eta*(E - E_th - R_t*e_hov)^+/Delta)   (4.9)-(4.10)
  B1  active: E' = E - e_hov - Delta/eta*sum p, E' >= E_th + (R_t-1)*e_hov  (4.6), (4.8)
  B2  inactive: E' = E + min(e_ch, E_max - E), E' <= E_max     (4.4), (4.6)
  B3  battery continuity across slots / relocation
  F1  relocation |dx| <= V_max*tau_c, e_rel charged iff moved, E' = E - e_rel  (2.4), (4.1), (4.5)
  F2  hired => E after relocation >= E_th + T*e_hov, point in feasible set  (4.11)
  F3  hiring cost = lambda_H * c^H * mu, exactly once per frame  (4.12)
  D1  J_F(t) = alpha_Z * Z(t)*(b_hat - d) + V*g^Q,  g^Q = d*(U_max - U_k)   (3.7), (6.12)
  D2  frame DPP = sum_t J_F + V*lambda_H*C^H; original = sum g^Q + lambda_H*C^H
  M1  x(t+1) = (x(t) + v*Delta) mod road_length

Usage:
    python -m hppo.verify_trace <run-dir> [--max-report 20]
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


from config_hppo import HPPOConfig
from env.p3.battery import battery_power_cap_w
from env.p3.radio import required_uav_power_w, rsu_link_capacity_bps
from hppo.env import uav_link_capacity_bps


class Verifier:
    def __init__(self, cfg: HPPOConfig, max_report: int = 20) -> None:
        self.cfg = cfg
        self.max_report = max_report
        self.passed: Counter = Counter()
        self.failed: Counter = Counter()
        self.reports: list[str] = []

    def check(self, code: str, ok: bool, where: str, detail: str = "") -> None:
        if ok:
            self.passed[code] += 1
        else:
            self.failed[code] += 1
            if len(self.reports) < self.max_report:
                self.reports.append(f"{code} FAIL @ {where}: {detail}")

    @staticmethod
    def close(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def load_config(run_dir: Path) -> HPPOConfig:
    with open(run_dir / "resolved_config.json", encoding="utf-8") as f:
        payload = json.load(f)["config"]
    cleaned = {}
    for k, v in payload.items():
        if isinstance(v, list):
            v = tuple(None if x is None else x for x in v)
            if k == "distance_bin_edges_m":
                v = tuple(math.inf if x is None else x for x in v)
            elif k == "hidden_dims":
                v = tuple(int(x) for x in v)
        cleaned[k] = v
    return HPPOConfig(**cleaned)


def _verify_physics(run_dir: Path, max_report: int = 20) -> tuple[Counter, Counter, list[str]]:
    cfg = load_config(run_dir)
    V = Verifier(cfg, max_report)
    Qe, b = cfg.large_queue_level, cfg.playback_chunks_per_slot

    last_q: dict[int, float] = {}                 # user -> Q after last update
    last_x: dict[int, float] = {}                 # user -> x after last mobility
    last_battery: dict[int, float] = {}           # region -> battery after last update
    frame_members: dict[int, set] = {}
    frame_hire: dict[int, int] = {}
    frame_users: dict[int, tuple[set, set]] = {}
    frame_hiring_cost: dict[int, float] = {}
    frame_dpp_sum: Counter = Counter()
    frame_deg_sum: Counter = Counter()
    frame_deliv_sum: Counter = Counter()
    frame_slots_seen = 0
    current_episode = None

    with open(run_dir / "trace.jsonl", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            ev = rec.get("event")
            if ev == "frame_start":
                if rec["episode"] != current_episode:
                    current_episode = rec["episode"]
                    last_q, last_x, last_battery = {}, {}, {}
                where = f"ep{rec['episode']} f{rec['frame']} frame_start"
                frame_dpp_sum.clear(); frame_deg_sum.clear(); frame_deliv_sum.clear(); frame_slots_seen = 0
                membership = rec["membership"]
                for m_str, rg in rec["regions"].items():
                    m = int(m_str)
                    members = set(rg["members"])
                    frame_members[m] = members
                    for us in rg["user_state"]:
                        u = us["user"]
                        V.check("A2", int(min(max(math.floor(us["x_m"] / cfg.region_length_m), 0), cfg.num_regions - 1)) == m,
                                where, f"user {u} x={us['x_m']} not in region {m}")
                        V.check("A2", membership[u] == m, where, f"membership[{u}]={membership[u]} != {m}")
                        if u in last_q:
                            V.check("Q3", V.close(us["Q"], last_q[u]), where, f"user {u} Q {us['Q']} vs carried {last_q[u]}")
                        if u in last_x:
                            V.check("M1", V.close(us["x_m"], last_x[u], 1e-6), where, f"user {u} x {us['x_m']} vs carried {last_x[u]}")
                        V.check("Q1", V.close(us["Z"], Qe - us["Q"]), where, f"user {u}")
                    hired = int(rg["executed_hire"])
                    frame_hire[m] = hired
                    rsu, uav = set(rg["executed_rsu_users"]), set(rg["executed_uav_users"])
                    frame_users[m] = (rsu, uav)
                    V.check("A1", len(rsu) <= cfg.rsu_capacity and len(uav) <= cfg.uav_capacity and not (rsu & uav)
                            and rsu <= members and uav <= members and (hired or not uav), where,
                            f"RSU={sorted(rsu)} UAV={sorted(uav)} hired={hired}")
                    dx = abs(rg["uav_x_after"] - rg["uav_x_before"])
                    moved = dx > 1e-9
                    V.check("F1", dx <= cfg.reachable_distance_m + 1e-9, where, f"relocation {dx} > {cfg.reachable_distance_m}")
                    V.check("F1", V.close(rg["relocation_energy_j"], cfg.relocation_energy_j if moved else 0.0), where, "e_rel")
                    V.check("F1", V.close(rg["battery_after_relocation_j"], rg["battery_before_j"] - rg["relocation_energy_j"]), where, "E - e_rel")
                    if m in last_battery:
                        V.check("B3", V.close(rg["battery_before_j"], last_battery[m]), where, f"battery_before {rg['battery_before_j']} vs {last_battery[m]}")
                    need = cfg.reserve_battery_j + cfg.frame_slots * cfg.hover_energy_per_slot_j
                    if hired:
                        V.check("F2", rg["battery_after_relocation_j"] + 1e-6 >= need, where, "activation reserve")
                        V.check("F2", rg["executed_point"] in rg["feasible_points"], where, "point not feasible")
                        V.check("F2", V.close(rg["uav_x_after"], cfg.candidate_points(m)[rg["executed_point"]]), where, "target x")
                    else:
                        V.check("F2", V.close(rg["uav_x_after"], cfg.depot_x(m)) and rg["executed_point"] == -1, where, "unhired UAV not at depot")
                    V.check("F3", V.close(rg["hiring_cost_weighted"], cfg.lambda_h * cfg.hiring_cost_per_frame * hired), where, "hiring cost")
                    frame_hiring_cost[m] = rg["hiring_cost_weighted"]
                    last_battery[m] = rg["battery_after_relocation_j"]
            elif ev == "slot":
                frame_slots_seen += 1
                R_t = rec["remaining_slots_including_current"]
                V.check("Q3", R_t == cfg.frame_slots - rec["slot_in_frame"], f"line {line_no}", "R_t")
                mob = rec["mobility"]
                for m_str, rg in rec["regions"].items():
                    m = int(m_str)
                    where = f"ep{rec['episode']} f{rec['frame']} t{rec['slot_in_frame']} R{m}"
                    rsu, uav = frame_users[m]
                    V.check("A2", set(rg["rsu_users"]) == rsu and set(rg["uav_users"]) == uav and rg["hired"] == frame_hire[m],
                            where, "frame action changed inside frame")
                    V.check("A2", {u["user"] for u in rg["users"]} == frame_members[m], where, "membership changed inside frame")
                    battery_before = rg["battery_before_j"]
                    if m in last_battery:
                        V.check("B3", V.close(battery_before, last_battery[m]), where, f"battery {battery_before} vs {last_battery[m]}")
                    # ---- power budget
                    if rg["hired"]:
                        p_eff_ref = battery_power_cap_w(battery_before, R_t, cfg)
                        V.check("P2", V.close(rg["p_eff_w"], p_eff_ref, 1e-6), where, f"P_eff {rg['p_eff_w']} vs {p_eff_ref}")
                    else:
                        V.check("P2", rg["p_eff_w"] == 0.0, where, "P_eff must be 0 when not hired")
                    tot_req = tot_exec = 0.0
                    region_dpp = region_deg = region_deliv = 0.0
                    for u in rg["users"]:
                        n = u["user"]
                        uw = f"{where} u{n}"
                        prov = u["provider"]
                        V.check("R4", (prov == 1) == (n in rsu) and (prov == 2) == (n in uav), uw, "provider/association mismatch")
                        if n in last_q:
                            V.check("Q3", V.close(u["q_before"], last_q[n]), uw, f"Q_before {u['q_before']} vs carried {last_q[n]}")
                        V.check("Q1", V.close(u["z_before"], Qe - u["q_before"]) and V.close(u["z_after"], Qe - u["q_after"]), uw, "Z identity")
                        dep = min(u["q_before"], b)
                        V.check("Q2", V.close(u["departure"], dep) and V.close(u["q_after"], u["q_before"] - dep + u["delivered"]), uw, "queue recurrence")
                        k = u["req_quality"]
                        S = cfg.chunk_size_bits[k]
                        d = u["delivered"]
                        V.check("R1", 0 <= d <= u["req_chunks"] <= cfg.max_chunks_per_slot and d <= u["feasible_chunks"]
                                and d <= u["queue_admissible_cap"], uw, f"d={d} req={u['req_chunks']} feas={u['feasible_chunks']}")
                        if d > 0:
                            V.check("R1", d * S <= u["capacity_bps"] * cfg.slot_duration_s + 1e-6, uw, "rate feasibility")
                        if prov == 0:
                            V.check("R4", d == 0 and u["exec_power_w"] == 0.0, uw, "unscheduled user delivered")
                        if u.get("fading") is not None and u["req_chunks"] > 0:
                            if prov == 1:
                                c_ref = rsu_link_capacity_bps(u["rsu_horizontal_distance_m"], u["fading"], cfg)
                            else:
                                c_ref, _ = uav_link_capacity_bps(u["uav_horizontal_distance_m"], u["fading"], u["exec_power_w"], cfg)
                            V.check("R2", V.close(u["capacity_bps"], c_ref, 1e-6), uw, f"C {u['capacity_bps']} vs {c_ref}")
                            V.check("R1", u["feasible_by_rate"] == int(math.floor(u["capacity_bps"] * cfg.slot_duration_s / S + 1e-9)), uw, "feasible_by_rate")
                        if prov == 2:
                            if u["req_chunks"] > 0:
                                V.check("P1", V.close(u["req_power_w"], cfg.power_level_to_w(u["req_power_level"])), uw, "p_req from level")
                                V.check("P1", V.close(u["exec_power_w"], u["req_power_w"] * rg["power_scale"]), uw, "p_exec = scale*p_req")
                            else:
                                V.check("P1", u["req_power_w"] == 0.0 and u["exec_power_w"] == 0.0, uw, "power with l=0")
                            if d > 0 and u.get("fading") is not None:
                                p_min = required_uav_power_w(d, k, u["uav_horizontal_distance_m"], u["fading"], cfg)
                                V.check("R3", p_min <= u["exec_power_w"] * (1 + 1e-6) + 1e-9, uw, f"p_min {p_min} > p_exec {u['exec_power_w']}")
                                V.check("R3", V.close(u["min_required_power_w"], p_min, 1e-6), uw, "logged p_min")
                            tot_req += u["req_power_w"]; tot_exec += u["exec_power_w"]
                        deg = d * (cfg.quality_max - cfg.quality_utility[k])
                        dpp = cfg.alpha_z * u["z_before"] * (dep - d) + cfg.lyapunov_v * deg
                        V.check("D1", V.close(u["degradation"], deg) and V.close(u["dpp_slot_cost"], dpp), uw, f"J_F {u['dpp_slot_cost']} vs {dpp}")
                        V.check("Q1", u["stall"] == int(u["q_before"] < b), uw, "stall indicator")
                        region_dpp += dpp; region_deg += deg; region_deliv += d
                        last_q[n] = u["q_after"]
                    V.check("P1", V.close(rg["total_requested_power_w"], tot_req) and V.close(rg["total_executed_power_w"], tot_exec)
                            and tot_exec <= rg["p_eff_w"] + 1e-9 and rg["p_eff_w"] <= cfg.uav_max_total_power_w + 1e-9, where,
                            f"sum p_exec {tot_exec} P_eff {rg['p_eff_w']}")
                    V.check("D1", V.close(rg["dpp_slot_cost"], region_dpp) and V.close(rg["degradation"], region_deg), where, "region J_F sum")
                    # ---- battery
                    if rg["hired"]:
                        comm = cfg.slot_duration_s * tot_exec / cfg.pa_efficiency
                        after = battery_before - cfg.hover_energy_per_slot_j - comm
                        V.check("B1", V.close(rg["battery_after_j"], after) and V.close(rg["communication_energy_j"], comm)
                                and V.close(rg["hover_energy_j"], cfg.hover_energy_per_slot_j), where, f"E' {rg['battery_after_j']} vs {after}")
                        V.check("B1", rg["battery_after_j"] + 1e-6 >= cfg.reserve_battery_j + (R_t - 1) * cfg.hover_energy_per_slot_j, where, "remaining-hover reserve")
                    else:
                        charge = max(0.0, min(cfg.charge_energy_per_slot_j, cfg.battery_capacity_j - battery_before))
                        V.check("B2", V.close(rg["charge_accepted_j"], charge) and V.close(rg["battery_after_j"], battery_before + charge)
                                and rg["battery_after_j"] <= cfg.battery_capacity_j + 1e-6, where, "charging")
                    last_battery[m] = rg["battery_after_j"]
                    frame_dpp_sum[m] += region_dpp; frame_deg_sum[m] += region_deg; frame_deliv_sum[m] += region_deliv
                for n, (xb, xa) in enumerate(zip(mob["x_before"], mob["x_after"])):
                    ref = math.fmod(xb + mob["speed"][n] * cfg.slot_duration_s, cfg.road_length_m)
                    V.check("M1", V.close(xa, ref, 1e-9) or V.close(abs(xa - ref), cfg.road_length_m, 1e-9), f"ep{rec['episode']} slot{rec['global_slot']} u{n}", f"x {xa} vs {ref}")
                    if n in last_x:
                        V.check("M1", V.close(xb, last_x[n], 1e-9), f"ep{rec['episode']} slot{rec['global_slot']} u{n}", "x continuity")
                    last_x[n] = xa
            elif ev == "frame_end":
                where = f"ep{rec['episode']} f{rec['frame']} frame_end"
                V.check("D2", frame_slots_seen == cfg.frame_slots, where, f"{frame_slots_seen} slots logged")
                for m_str, rg in rec["regions"].items():
                    m = int(m_str)
                    ref_dpp = frame_dpp_sum[m] + cfg.lyapunov_v * frame_hiring_cost[m]
                    ref_orig = frame_deg_sum[m] + frame_hiring_cost[m]
                    V.check("D2", V.close(rg["frame_dpp_cost"], ref_dpp) and V.close(rg["original_cost"], ref_orig)
                            and V.close(rg["delivered_chunks"], frame_deliv_sum[m]) and rg["hired"] == frame_hire[m],
                            where, f"R{m} frame DPP {rg['frame_dpp_cost']} vs {ref_dpp}")
                    V.check("F3", V.close(rg["hiring_cost_weighted"], frame_hiring_cost[m]), where, "hiring cost counted once")
                    V.check("B3", V.close(rg["battery_soc_end"] * cfg.battery_capacity_j, last_battery[m], 1e-6), where, "SoC end")
    return V.passed, V.failed, V.reports


def verify(run_dir: Path, max_report: int = 20):
    from hppo.trace_contract import check_structure
    try:
        cfg = load_config(run_dir)
        passed, failed, reports = check_structure(run_dir, cfg, max_report)
        if failed:
            return passed, failed, reports
        pp, ff, rr = _verify_physics(run_dir, max_report)
        return passed + pp, failed + ff, (reports + rr)[:max_report]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return Counter(), Counter({"S0": 1}), [f"S0 FAIL: {exc}"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--max-report", type=int, default=20)
    args = parser.parse_args(argv)
    passed, failed, reports = verify(Path(args.run_dir), args.max_report)
    codes = sorted(set(passed) | set(failed))
    print(f"trace verification for {args.run_dir}")
    print(f"{'check':<6}{'passed':>10}{'failed':>10}")
    for c in codes:
        print(f"{c:<6}{passed[c]:>10}{failed[c]:>10}")
    total_fail = sum(failed.values())
    print(f"TOTAL  passed={sum(passed.values())} failed={total_fail}")
    for r in reports:
        print("  " + r)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
