from __future__ import annotations

"""History logging (uav_hierarchical_ppo/logger.py design, extended).

Three outputs are written under ``<output-dir>/<run-name>/``:

* ``trace.jsonl``        one JSON record per event: ``frame_start``, ``slot``,
                         ``frame_end``, ``episode_end``, ``*_ppo_update``.
                         Every state -> decision -> update quantity is stored, so
                         ``hppo/verify_trace.py`` can re-derive all P3 equations
                         offline.
* ``debug.log``          human-readable text version of the same history: for
                         each slot and each user, the queue before, the raw and
                         executed decision, the realized channel, what was
                         delivered, the queue after and the check results.
* ``episode_summary.csv`` one row per episode.
"""

import csv
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from config_hppo import HPPOConfig


PROVIDER_NAME = {0: "---", 1: "RSU", 2: "UAV"}


def jsonable(x: Any):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _fmt_map(d: dict, fmt: str = "{:.3f}") -> str:
    return "{" + ", ".join(f"{k}:{fmt.format(v)}" for k, v in d.items()) + "}"


class HistoryLogger:
    def __init__(self, cfg: HPPOConfig, output_path: Path, resume: bool = False) -> None:
        self.cfg = cfg
        self.root = Path(output_path)
        self.root.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self.trace_f = open(self.root / "trace.jsonl", mode, encoding="utf-8") if cfg.write_jsonl_trace else None
        self.debug_f = open(self.root / "debug.log", mode, encoding="utf-8") if cfg.write_human_debug_log else None
        self.summary_path = self.root / "episode_summary.csv"
        self._header_written = resume and self.summary_path.exists() and self.summary_path.stat().st_size > 0
        if not resume and self.summary_path.exists():
            self.summary_path.unlink()

    def close(self) -> None:
        for f in (self.trace_f, self.debug_f):
            if f:
                f.close()

    # ------------------------------------------------------------------
    def event(self, name: str, payload: dict) -> None:
        if self.trace_f:
            rec = {"event": name, **jsonable({k: v for k, v in payload.items() if k != "event"})}
            self.trace_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.trace_f.flush()
        if name.endswith("ppo_update"):
            p = self.root / ("frame_updates.csv" if name.startswith("frame") else "slot_updates.csv")
            exists = p.exists() and p.stat().st_size > 0
            row = jsonable(payload)
            with p.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    def debug(self, text: str) -> None:
        if self.debug_f:
            self.debug_f.write(text.rstrip("\n") + "\n")
            self.debug_f.flush()

    # ------------------------------------------------------------------
    def log_frame_start(self, info: dict, obs: dict | None = None, masks: dict | None = None) -> None:
        payload = copy.deepcopy(info)
        if self.cfg.log_observation_vectors and obs is not None:
            payload["frame_observation"] = {m: np.asarray(o).tolist() for m, o in obs.items()}
        if masks is not None:
            payload["frame_masks"] = {m: [np.asarray(x).astype(int).tolist() for x in ms] for m, ms in masks.items()}
        self.event("frame_start", payload)
        if not self.debug_f:
            return
        cfg = self.cfg
        lines = ["=" * 100,
                 f"[FRAME START] ep={info['episode']} frame={info['frame']} global_slot={info['global_slot']}",
                 " membership (fixed for this frame): " + " ".join(
                     f"R{m}={list(rg['members'])}" for m, rg in info["regions"].items())]
        for m, rg in info["regions"].items():
            lines.append(f" --- Region {m} ---")
            lines.append(
                f"  state: UAV x={rg['uav_x_before']:.1f} battery={rg['battery_before_j']:.1f}J "
                f"(SoC={rg['battery_before_j']/cfg.battery_capacity_j:.4f}) feasible_points={rg['feasible_points']} "
                f"activation_req={rg['activation_energy_required_j']:.1f}J charging_needed={rg['charging_needed']}")
            for us in rg["user_state"]:
                lines.append(
                    f"    u{us['user']:<2d} x={us['x_m']:7.1f} v={us['speed_mps']:5.2f} Q={us['Q']:6.2f} Z={us['Z']:7.2f} "
                    f"dRSU={us['rsu_horizontal_distance_m']:6.1f} dCand={[round(d,1) for d in us['candidate_horizontal_distance_m']]} "
                    f"lastK={us['last_quality_index']}")
            assoc = {u: rg["raw_assoc"][u] for u in rg["members"]}
            lines.append(f"  PPO proposal   : assoc={assoc} UAV_candidates={rg['proposal_uav_candidates']}")
            detail = rg.get("completion")
            if detail:
                lines.append(f"  completion     : fast_update={detail['fast_update_count']} scenarios={detail['scenarios']} "
                             f"time={detail['runtime_s']:.4f}s selected={detail['selected_index']}")
                for i, row in enumerate(detail["candidates"]):
                    lines.append(f"    candidate {i}: hire={row['hired']} point={row['point']} "
                                 f"mean_DPP={row['mean_dpp']:.6f} samples={row['sample_dpp']}")
            lines.append(
                f"  executed       : hire={rg['executed_hire']} point={rg['executed_point']} (x={rg['executed_target_x']:.1f}) "
                f"RSU={rg['executed_rsu_users']} UAV={rg['executed_uav_users']} unserved={rg['unserved_users']}")
            lines.append(f"  projection     : {rg['projection_reasons'] if rg['projection_reasons'] else 'none'}")
            lines.append(
                f"  relocation     : {rg['uav_x_before']:.1f} -> {rg['uav_x_after']:.1f} "
                f"({rg['relocation_distance_m']:.1f}m <= {rg['reachable_distance_m']:.1f}m) e_rel={rg['relocation_energy_j']:.1f}J "
                f"battery {rg['battery_before_j']:.1f} -> {rg['battery_after_relocation_j']:.1f}J "
                f"activation_ok={rg['activation_ok']} return_to_charge={rg['return_to_charge']}")
            lines.append(f"  hiring cost    : lambda_H*c^H*mu = {rg['hiring_cost_weighted']:.3f}")
        self.debug("\n".join(lines))

    def log_slot(self, info: dict, rewards: dict | None = None, obs: dict | None = None,
                 masks: dict | None = None, dual: float | None = None) -> None:
        payload = copy.deepcopy(info)
        if rewards is not None:
            payload["rewards"] = rewards
        if dual is not None:
            payload["dual_lambda_z"] = float(dual)
        if self.cfg.log_observation_vectors and obs is not None:
            payload["slot_observation"] = {m: np.asarray(o).tolist() for m, o in obs.items()}
        if masks is not None and self.cfg.log_observation_vectors:
            payload["slot_masks"] = {m: [np.asarray(x).astype(int).tolist() for x in ms] for m, ms in masks.items()}
        if not self.cfg.log_hidden_csi:
            for rg in payload["regions"].values():
                for u in rg["users"]:
                    u.pop("fading", None); u.pop("gain", None)
        self.event("slot", payload)
        if not self.debug_f:
            return
        lines = [f"[SLOT] ep={info['episode']} frame={info['frame']} slot={info['slot_in_frame']}/{self.cfg.frame_slots} "
                 f"global_slot={info['global_slot']} R_t={info['remaining_slots_including_current']}"]
        ok_all = True
        for m, rg in info["regions"].items():
            lines.append(
                f" --- Region {m} --- hired={rg['hired']} point={rg['point_index']} x_uav={rg['uav_x']:.1f} "
                f"RSU={rg['rsu_users']} UAV={rg['uav_users']} P_eff=min(P_max={rg['p_max_w']:.2f}, P_bat)={rg['p_eff_w']:.3f}W")
            if rg["hired"]:
                req = {u["user"]: u["req_power_w"] for u in rg["users"] if u["provider"] == 2}
                ex = {u["user"]: u["exec_power_w"] for u in rg["users"] if u["provider"] == 2}
                lines.append(
                    f"  UAV power : requested={_fmt_map(req)} total={rg['total_requested_power_w']:.3f} "
                    f"scale={rg['power_scale']:.3f} executed={_fmt_map(ex)} total={rg['total_executed_power_w']:.3f} "
                    f"<= P_eff {rg['p_eff_w']:.3f} -> {'OK' if rg['total_executed_power_w'] <= rg['p_eff_w'] + 1e-9 else 'VIOLATION'}")
            for u in rg["users"]:
                prov = PROVIDER_NAME[u["provider"]]
                ok_all &= bool(u["identity_ok"])
                if u["provider"] == 0:
                    lines.append(
                        f"  u{u['user']:<2d} {prov} x={u['x_m']:6.1f} Q={u['q_before']:5.2f} Z={u['z_before']:6.2f} | unscheduled "
                        f"| dep={u['departure']:.1f} -> Q={u['q_after']:5.2f} Z={u['z_after']:6.2f} stall={u['stall']} "
                        f"| J_F={u['dpp_slot_cost']:8.2f} | Z-identity={'OK' if u['identity_ok'] else 'FAIL'}")
                    continue
                if u["provider"] == 2:
                    p_txt = f"lvl={u['req_power_level']}->p_req={u['req_power_w']:.3f}W p_exec={u['exec_power_w']:.3f}W"
                    d_txt = f"dUAV={u['uav_horizontal_distance_m']:.1f}m"
                else:
                    p_txt = f"p=P^R/J^R={u['exec_power_w']:.3f}W"
                    d_txt = f"dRSU={u['rsu_horizontal_distance_m']:.1f}m"
                fad = "hidden" if u.get("fading") is None else f"{u['fading']:.3f}"
                pmin = f" p_min(l,k)={u['min_required_power_w']:.3f}W" if u["provider"] == 2 and u["delivered"] > 0 else ""
                lines.append(
                    f"  u{u['user']:<2d} {prov} x={u['x_m']:6.1f} Q={u['q_before']:5.2f} Z={u['z_before']:6.2f} "
                    f"| req l={u['req_chunks']} k={u['req_quality']} {p_txt} | {d_txt} fading={fad} "
                    f"C={u['capacity_bps']/1e6:7.3f}Mbps feas_rate={u['feasible_by_rate']} Qcap={u['queue_admissible_cap']} "
                    f"-> feasible={u['feasible_chunks']} | delivered={u['delivered']} util={u['utility']:.2f} deg={u['degradation']:.2f}{pmin} "
                    f"| dep={u['departure']:.1f} -> Q={u['q_after']:5.2f} Z={u['z_after']:6.2f} stall={u['stall']} "
                    f"| J_F={u['dpp_slot_cost']:8.2f} | Z-identity={'OK' if u['identity_ok'] else 'FAIL'}")
            if rg["hired"]:
                lines.append(
                    f"  battery   : {rg['battery_before_j']:.1f} - hover {rg['hover_energy_j']:.1f} - comm {rg['communication_energy_j']:.3f} "
                    f"= {rg['battery_after_j']:.1f}J (SoC={rg['battery_soc_after']:.4f}) reserve_req={rg['reserve_required_after_j']:.1f}J "
                    f"-> {'OK' if rg['reserve_ok'] else 'VIOLATION'}")
            else:
                lines.append(
                    f"  battery   : {rg['battery_before_j']:.1f} + charge {rg['charge_accepted_j']:.1f} "
                    f"= {rg['battery_after_j']:.1f}J (SoC={rg['battery_soc_after']:.4f}) [depot charging]")
            r_txt = ""
            if rewards is not None and m in rewards:
                r_txt = f" | base_reward={rewards[m]['base']:.4f} training_reward={rewards[m]['training']:.4f}"
            lines.append(
                f"  region    : J_F={rg['dpp_slot_cost']:.2f} deg={rg['degradation']:.3f} delivered={rg['delivered_chunks']:.0f} "
                f"stalls={rg['stall_user_slots']}/{len(rg['users'])} Zcost={rg['constraint_cost_z_norm']:.3f}{r_txt}")
        mob = info["mobility"]
        moves = " ".join(f"u{n}:{a:.1f}->{b:.1f}" for n, (a, b) in enumerate(zip(mob["x_before"], mob["x_after"])))
        lines.append(f" mobility (x += v*Delta mod road): {moves}")
        lines.append(f" checks: Z-identity={'OK' if ok_all else 'FAIL'}" + (f" dual={dual:.3f}" if dual is not None else ""))
        self.debug("\n".join(lines))

    def log_frame_end(self, summary: dict, rewards: dict | None = None, dual: float | None = None) -> None:
        payload = dict(summary)
        if rewards is not None:
            payload["rewards"] = rewards
        if dual is not None:
            payload["dual_lambda_z"] = float(dual)
        self.event("frame_end", payload)
        if not self.debug_f:
            return
        lines = [f"[FRAME END] ep={summary['episode']} frame={summary['frame']} "
                 f"frame_DPP_total={summary['frame_dpp_cost_total']:.2f} original_total={summary['original_cost_total']:.3f} "
                 f"meanQ={summary['mean_Q']:.2f} meanZ={summary['mean_Z']:.2f}"]
        for m, rg in summary["regions"].items():
            r_txt = ""
            if rewards is not None and m in rewards:
                r_txt = f" base_reward={rewards[m]['base']:.4f} training_reward={rewards[m]['training']:.4f}"
            lines.append(
                f"  R{m}: hire={rg['hired']} frame_DPP=sum J_F {rg['dpp_slot_sum']:.2f} + V*hire {rg['hiring_cost_weighted']*self.cfg.lyapunov_v:.2f} "
                f"= {rg['frame_dpp_cost']:.2f} | original=deg {rg['degradation_sum']:.3f} + hire {rg['hiring_cost_weighted']:.2f} = {rg['original_cost']:.3f} "
                f"| delivered={rg['delivered_chunks']:.0f} stall_ratio={rg['stall_ratio']:.3f} Q>Qe={rg['q_gt_qe_user_slots']} "
                f"| E_used={rg['energy_consumed_j']:.1f}J E_chg={rg['energy_charged_j']:.1f}J SoC_end={rg['battery_soc_end']:.4f} "
                f"| violations reserve={rg['reserve_violations']} power={rg['power_violations']} projections={rg['projection_events']}{r_txt}")
        self.debug("\n".join(lines))

    def log_episode(self, summary: dict) -> None:
        summary = jsonable(summary)
        self.event("episode_end", summary)
        with open(self.summary_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow(summary)
        self.debug(
            f"[EPISODE END] ep={summary['episode']} DPP/user-slot={summary['dpp_cost_per_user_slot']:.3f} "
            f"original/user-slot={summary['original_cost_per_user_slot']:.4f} stall={summary['stall_ratio']:.4f} "
            f"served={summary['served_user_ratio']:.3f} hire_rate={summary['hire_rate']:.3f} "
            f"Zcost={summary['z_constraint_mean']:.3f} Q>Qe={summary['q_gt_qe_rate']:.4f} minSoC={summary['min_battery_soc']:.4f} "
            f"reserve_viol={summary['reserve_violations']} power_viol={summary['power_violations']} "
            f"projections={summary['projection_events']}" + (f" dual={summary['dual_lambda_z']:.3f}" if 'dual_lambda_z' in summary else ""))
        self.debug("#" * 100)
