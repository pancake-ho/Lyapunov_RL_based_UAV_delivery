"""
Logging utilities.

* ``run.log``          : human-readable run/episode log (python logging)
* ``slot_log.txt``     : per-slot, per-region debug log
                         (state before -> decisions -> state after -> reward)
* ``slot_log.jsonl``   : the same record as one JSON object per line
* ``episodes.csv``     : one row per episode with summary metrics
"""
import csv
import json
import logging
import os


def setup_run_logger(log_dir, level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("sac_baseline")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class SlotLogger:
    def __init__(self, log_dir, enabled=True, console=False):
        self.enabled = enabled
        self.console = console
        if enabled:
            self.txt = open(os.path.join(log_dir, "slot_log.txt"), "w")
            self.jsonl = open(os.path.join(log_dir, "slot_log.jsonl"), "w")

    def close(self):
        if self.enabled:
            self.txt.close()
            self.jsonl.close()

    @staticmethod
    def _fmt_before(v):
        return (f"veh{v['vid']}: x={v['x']:.0f} y={v['y']:+.1f} v={v['v']:+.1f} Q={v['Q']:.0f} Z={v['Z']:.0f} "
                f"lastk={v['last_k']} by={v['served_by']} slot={v['slot']} r_now={v['region_now']}")

    def log(self, episode, mode, info, actions=None):
        if not self.enabled:
            return
        t, fr, sl = info["t"] - 1, info["frame"], info["slot_in_frame"]
        # frame/slot of the slot that was just executed
        for m, L in info["region_logs"].items():
            rec = {"episode": episode, "mode": mode, "t": t, "region": m, **L}
            self.jsonl.write(json.dumps(rec, default=float) + "\n")

            ub, ua = L["uav_before"], L["uav_after"]
            lines = [f"[ep {episode} {mode} | t={t} | region {m}] "
                     f"UAV before: hired={int(ub['hired'])} pos=({ub['pos'][0]:.0f},{ub['pos'][1]:.0f}) "
                     f"vel=({ub['vel'][0]:.1f},{ub['vel'][1]:.1f}) E={ub['E_J']:.0f}J soc={ub['soc']:.3f} | "
                     f"members={L['frame_state']['members']} rsu={L['frame_state']['rsu_users']} "
                     f"uav={L['frame_state']['uav_users']} overflow={L['frame_state']['overflow']} "
                     f"({L['frame_state']['hire_reason']})"]
            lines.append("  STATE  : " + (" | ".join(self._fmt_before(v) for v in L["vehicles_before"]) or "(no vehicles)"))
            lines.append("  ACTION : raw=" + str(L["action_raw"]))
            mv = L["uav_move"]
            if "vel_cmd" in mv:
                lines.append(f"  UAV    : vel_cmd=({mv['vel_cmd'][0]:.1f},{mv['vel_cmd'][1]:.1f}) "
                             f"-> vel=({mv['vel'][0]:.1f},{mv['vel'][1]:.1f}) |v|={mv['speed']:.1f} "
                             f"pos=({mv['pos'][0]:.0f},{mv['pos'][1]:.0f}) P_prop={mv['P_prop_W']:.1f}W "
                             f"e_com={mv['e_com_J']:.2f}J e_slot={mv['e_slot_J']:.1f}J")
            else:
                lines.append(f"  UAV    : at depot, charging +{mv['charging_J']:.0f}J")
            if L["uav_decisions"]:
                lines.append("  UAV-TX : " + " | ".join(
                    f"veh{d['vid']} bw={d['bw_frac']:.2f} W={d['W_MHz']:.2f}MHz p={d['p_W']:.2f}W d={d['dist_m']:.0f}m "
                    f"snr={d['snr_dB']:.0f}dB C={d['C_Mbps']:.1f}Mbps k={d['k']} l_req={d['l_req']} "
                    f"l_max={d['l_max']} room={d['room']} -> l={d['l_sent']}" for d in L["uav_decisions"]))
            if L["rsu_decisions"]:
                lines.append("  RSU-TX : " + " | ".join(
                    f"veh{d['vid']} d={d['dist_m']:.0f}m C={d['C_Mbps']:.1f}Mbps lmax/k={d['l_max_per_k']} "
                    f"room={d['room']} -> k={d['k']} l={d['l_sent']}" for d in L["rsu_decisions"]))
            lines.append("  NEXT   : " + (" | ".join(
                f"veh{v['vid']} Q {v['Q_prev']:.0f}->{v['Q_next']:.0f} Z={v['Z_next']:.0f} d={v['d']} k={v['k']} "
                f"stall={v['stall']}" + (f" sw={v['switch_bps']/1e6:.1f}Mbps" if v['switch_bps'] else "")
                for v in L["vehicles_after"]) or "(none)"))
            lines.append("  MOVED  : " + ", ".join(f"veh{v['vid']}@{v['x']:.0f}(r{v['region_now']})"
                                                   for v in L["vehicles_moved"]))
            lines.append(f"  UAV after: hired={int(ua['hired'])} pos=({ua['pos'][0]:.0f},{ua['pos'][1]:.0f}) "
                         f"E={ua['E_J']:.0f}J soc={ua['soc']:.3f}"
                         + (" FORCED_RETURN" if ua["forced_return"] else ""))
            if L["events"]:
                lines.append("  EVENTS : " + "; ".join(L["events"]))
            r = L["reward"]
            lines.append(f"  REWARD : total={r['total']:+.3f} = quality {r['quality']:.3f} - switch {r['switch']:.3f} "
                         f"- stall {r['stall']:.1f} - energy {r['energy']:.3f} - hire {r['hire']:.3f}")
            block = "\n".join(lines) + "\n"
            self.txt.write(block)
            if self.console:
                print(block, end="")
        self.txt.flush()


class EpisodeCSV:
    def __init__(self, log_dir):
        self.path = os.path.join(log_dir, "episodes.csv")
        self.f = open(self.path, "w", newline="")
        self.w = None

    def write(self, row):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
