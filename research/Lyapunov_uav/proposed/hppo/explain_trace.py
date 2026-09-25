"""Read-only Korean slot/frame narratives for scheduling-hppo-v2.

Python standard library only. No training, policy loading, or GPU execution.
Compatible baseline: feat/hrl 0f4088e6d165e4bc7235921449900bb44bec94f8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
try:
    from hppo.completion_audit import candidate_lines
except ModuleNotFoundError:  # Preserve python hppo/explain_trace.py invocation.
    from completion_audit import candidate_lines

BASE_COMMIT = "a63b1da63addcb94e0f2e0bbe8b92dab5a9df318"
MAX_LINE = 64 * 1024 * 1024
LABELS = {
    "ORDER": "이벤트/슬롯 순서", "SETS": "집합 구성/고정 유지",
    "LINK": "이전 상태 -> 다음 상태 연결", "MOVE": "위치/재배치",
    "ENERGY": "배터리 에너지 수지", "POWER": "UAV 전력/예비 에너지",
    "DELIVERY": "chunk 요청/실제 수신", "QUEUE": "재생/버퍼/Z",
    "TOTAL": "프레임 합계", "CHOICE": "기록된 후보 비교/실행 일치",
    "DATA": "필수 데이터/형식",
}


def number(x):
    return f"{float(x):,.3f}".rstrip("0").rstrip(".")


def users(ids):
    return "{" + ", ".join(f"u{u}" for u in sorted(ids)) + "}"


def close(a, b):
    # Small communication-energy discrepancies must not vanish beside a 2 MJ battery.
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-10)


def strict_json(data):
    def reject(value):
        raise ValueError(f"비유한 JSON 숫자: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"비유한 JSON 숫자: {value}")
        return result
    return json.loads(data, parse_constant=reject, parse_float=finite_float)


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Narrator:
    def __init__(self, config, output, args):
        self.c, self.out, self.args = config, output, args
        self.passed, self.failed = Counter(), Counter()
        self.frame_pass, self.frame_fail = defaultdict(Counter), defaultdict(Counter)
        self.context, self.region = "초기화", "전체"
        self.frame_file = None
        self.summary = (output / "frames_summary.txt").open("w", encoding="utf-8")
        self.issues = (output / "issues.txt").open("w", encoding="utf-8")
        self.started = self.ended = False
        self.expected = self.offset = self.episodes = self.next_frame = self.next_slot = 0
        self.key, self.frame = None, None
        self.last_users, self.last_battery, self.last_uav, self.last_sets = {}, {}, {}, {}
        self.totals = {}
        self.rendered = 0
        self.scope_note = (
            "검사 범위: 읽은 전체 trace. --episode/--frame/--region은 출력만 선택합니다.\n"
            "PASS는 표시한 상태 전이/집합/합계 검사 통과이며 학습 수렴의 증명이 아닙니다.\n"
            "무선 채널 식/CSI 재계산과 PPO gradient 검증은 이 모듈의 범위 밖입니다.\n"
            "Q=재생 버퍼(chunk 수), Z=Qe-Q. k는 0부터 시작하는 quality index입니다.\n"
            "이 모델은 slot 시작 Q로 재생량과 stall을 정하고, 수신량을 더해 다음 Q를 만듭니다.\n"
            "수신 누계는 해당 frame 안의 개수입니다. video/chunk 고유 ID는 trace에 없습니다.\n"
            "association은 frame 안에서 고정됩니다. 실제 수신 0개가 집합 탈퇴를 뜻하지 않습니다.\n"
            "FRAME 종료 구간이 없으면 그 frame은 아직 완료로 판정하지 않습니다.\n"
        )
        self.summary.write(self.scope_note + "\n")

    def chosen(self, region=None):
        if self.key is None:
            return False
        ep, frame = self.key
        return ((not self.args.episode or ep in self.args.episode)
                and (not self.args.frame or frame in self.args.frame)
                and (region is None or not self.args.region or int(region) in self.args.region))

    def emit(self, text, summary=False):
        if self.chosen() and (self.region == "전체" or self.chosen(self.region)):
            if self.frame_file:
                self.frame_file.write(text + "\n")
            if summary:
                self.summary.write(text + "\n")

    def check(self, code, ok, detail):
        ok = bool(ok)
        (self.passed if ok else self.failed)[code] += 1
        (self.frame_pass if ok else self.frame_fail)[self.region][code] += 1
        if not ok:
            message = f"[FAIL:{code}] {self.context} R={self.region} | {detail}"
            if sum(self.failed.values()) <= 2000:
                self.issues.write(message + "\n")
            self.emit(message)
        return ok

    def flush(self):
        for stream in (self.frame_file, self.summary, self.issues):
            if stream:
                stream.flush()

    def dispose(self):
        for stream in (self.frame_file, self.summary, self.issues):
            if stream:
                stream.close()

    def delta_set(self, old, new):
        return f"{users(old)} -> {users(new)} (유입 {users(set(new)-set(old))}; 이탈 {users(set(old)-set(new))})"

    def process(self, rec, line):
        ev = rec["event"]
        if ev == "frame_start":
            self.frame_pass, self.frame_fail = defaultdict(Counter), defaultdict(Counter)
        self.region = "전체"
        self.context = f"line={line} ep={rec.get('episode', '-')} frame={rec.get('frame', '-')} {ev}"
        self.check("ORDER", not self.ended, "run_end 뒤에 이벤트가 추가됨")
        if ev == "run_start":
            self.check("ORDER", not self.started and line == 1, "run_start 위치/중복")
            self.check("DATA", rec["schema"] == "scheduling-hppo-v2", "지원 schema: scheduling-hppo-v2")
            self.expected, self.offset = rec["expected_episodes"], rec["episode_offset"]
            expected = self.c["train_episodes"] if rec["mode"] == "train" else self.c["eval_episodes"]
            self.check("ORDER", self.expected == expected and self.offset == self.c["episode_offset"], "설정과 실행 episode 수/offset 불일치")
            self.started = True
            return
        if not self.started:
            raise ValueError("run_start 없는 trace")
        if ev in ("frame_start", "slot", "frame_end"):
            self.check("ORDER", (rec["episode"], rec["frame"]) == (self.offset+self.episodes, self.next_frame), "episode/frame 순서 불일치")
            self.check("DATA", set(rec["regions"]) == {str(m) for m in range(self.c["num_regions"])}, "region 누락/중복")
        if ev == "frame_start":
            self.start_frame(rec)
        elif ev == "slot":
            self.slot(rec)
        elif ev == "frame_end":
            self.end_frame(rec)
        elif ev == "episode_end":
            self.check("ORDER", self.frame is None and self.next_frame == self.c["num_frames"] and rec["episode"] == self.offset+self.episodes, "불완전 episode/순서 오류")
            self.episodes += 1
            self.next_frame = 0
        elif ev == "run_end":
            self.check("ORDER", rec["status"] == "complete" and self.frame is None and self.next_frame == 0 and self.episodes == self.expected and self.episodes > 0, "run_end가 있지만 episode/frame 미완료")
            self.ended = True
        elif ev not in ("frame_ppo_update", "slot_ppo_update"):
            self.check("DATA", False, f"알 수 없는 이벤트 {ev}")

    def start_frame(self, rec):
        c = self.c
        self.check("ORDER", self.frame is None and rec["frame"] < c["num_frames"], "이전 frame_end 누락/범위 초과")
        if self.frame_file:
            self.frame_file.close()
            self.frame_file = None
        self.key = (rec["episode"], rec["frame"])
        self.frame, self.next_slot = rec, 0
        if rec["frame"] == 0:
            self.last_users, self.last_battery, self.last_uav, self.last_sets = {}, {}, {}, {}
        if self.chosen():
            folder = self.out / "episodes" / f"ep{self.key[0]:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents duplicate frame events overwriting evidence.
            self.frame_file = (folder / f"frame{self.key[1]:06d}.txt").open("x", encoding="utf-8")
            self.rendered += 1
            self.frame_file.write(self.scope_note + "\n")
        self.emit(f"--- FRAME 시작 | EP {self.key[0]} / FRAME {self.key[1]} | slot 0..{c['frame_slots']-1} ---", True)
        ids = [u for rg in rec["regions"].values() for u in rg["members"]]
        n = c["num_regions"]*c["users_per_region"]
        self.check("SETS", sorted(ids) == list(range(n)) and len(rec["membership"]) == n, "모든 user가 정확히 한 region의 member여야 함")
        self.totals = {}
        previous_owners = {u:m for m, sets in self.last_sets.items() for u in sets[0]}
        for m, rg in sorted(rec["regions"].items(), key=lambda p:int(p[0])):
            self.region = m
            members, rsu, uav = rg["members"], rg["executed_rsu_users"], rg["executed_uav_users"]
            old = self.last_sets.get(m)
            self.emit(f"--- R{m} | UAV{m} ---", True)
            self.emit("frame member: " + (self.delta_set(old[0], members) if old else users(members)+" (episode 초기 상태)"), True)
            self.emit(f"slow PPO 제안: RSU={users(rg['proposal_rsu_users'])}; UAV 후보={users(rg['proposal_uav_candidates'])}", True)
            choice = rg["completion"]
            rows = choice["candidates"]
            best = min(range(len(rows)), key=lambda i: rows[i]["mean_dpp"])
            selected = choice["selected_index"]
            self.check("CHOICE", selected == best and close(choice["selected_score"], rows[best]["mean_dpp"]), "기록된 후보 DPP 최소값과 선택 불일치")
            for i, row in enumerate(rows):
                self.check("CHOICE", row["rsu_users"] == rg["proposal_rsu_users"] and row["uav_users"] == (rg["proposal_uav_candidates"] if row["hired"] else []), "후보 비교 중 scheduling 변경")
                self.check("CHOICE", len(row["sample_dpp"]) == c["rollout_scenarios"] and close(sum(row["sample_dpp"])/len(row["sample_dpp"]), row["mean_dpp"]), "후보 평균/표본 개수 불일치")
                self.emit(f"  후보 {i}: hire={row['hired']} point={row['point']} -> 예상 DPP {number(row['mean_dpp'])}" + (" [선택]" if i==selected else ""))
            for line in candidate_lines(choice):
                self.emit(line)
            self.check("CHOICE", (rg["executed_hire"], rg["executed_point"], rsu, uav) == (rows[selected]["hired"], rows[selected]["point"], rows[selected]["rsu_users"], rows[selected]["uav_users"]), "선택된 후보와 실제 frame 동작 불일치")
            self.check("SETS", rg["executed_hire"] in (0,1) and len(rsu)==len(set(rsu)) and len(uav)==len(set(uav)) and not set(rsu)&set(uav) and set(rsu+uav)<=set(members) and len(rsu)<=c["rsu_capacity"] and len(uav)<=c["uav_capacity"] and (rg["executed_hire"]==1 or not uav), "집합 중복/용량/소속/고용 불일치")
            self.check("SETS", set(rg["unserved_users"]) == set(members)-set(rsu+uav), "미배정 집합 불일치")
            self.emit(f"실행 association: RSU{m} -> {users(rsu)}; UAV{m} -> {users(uav)}; 미배정 -> {users(rg['unserved_users'])}", True)
            if old:
                self.emit(f"RSU 집합 변화: {self.delta_set(old[1], rsu)}")
                self.emit(f"UAV 집합 변화: {self.delta_set(old[2], uav)}")
            self.emit(f"고용={rg['executed_hire']}; point={rg['executed_point']}; 위 집합/고용/point는 이 frame의 모든 slot에 고정", True)
            if not rg["executed_hire"] and rg["proposal_uav_candidates"]:
                self.emit(f"UAV 미고용 -> 후보 {users(rg['proposal_uav_candidates'])}는 이번 frame 미배정 (RSU로 자동 재배정하지 않음)")
            dx = abs(rg["uav_x_after"]-rg["uav_x_before"])
            rel = c["relocation_energy_j"] if dx>1e-9 else 0.0
            expected_x = (int(m)+0.5)*c["region_length_m"]
            if rg["executed_hire"]:
                expected_x += c["candidate_offsets_m"][rg["executed_point"]]
                self.check("MOVE", rg["executed_point"] in rg["feasible_points"], "실행 point가 feasible set 밖")
            else:
                self.check("MOVE", rg["executed_point"] == -1, "미고용 point는 -1이어야 함")
            self.check("MOVE", dx <= c["uav_max_speed_mps"]*c["control_interval_s"]+1e-6 and close(rg["uav_x_after"],expected_x), "이동 한계/목적 위치 불일치")
            self.check("ENERGY", close(rg["relocation_energy_j"],rel) and close(rg["battery_after_relocation_j"],rg["battery_before_j"]-rel), "재배치 에너지/배터리 차감 불일치")
            if m in self.last_battery:
                self.check("LINK", close(rg["battery_before_j"],self.last_battery[m]) and close(rg["uav_x_before"],self.last_uav[m]), "이전 frame 마지막 slot -> 현재 frame 재배치 전 배터리/위치 단절")
                self.emit(f"이전 frame 끝 배터리 {number(self.last_battery[m])} J -> 현재 재배치 전 {number(rg['battery_before_j'])} J")
            required = c["reserve_battery_j"]+c["frame_slots"]*c["hovering_power_w"]*c["slot_duration_s"]
            self.check("POWER", not rg["executed_hire"] or rg["battery_after_relocation_j"]+1e-6>=required, "고용 시작 예비 에너지 부족")
            self.emit(f"재배치: x {number(rg['uav_x_before'])} -> {number(rg['uav_x_after'])} m; 거리 {number(dx)} m")
            self.emit(f"배터리: {number(rg['battery_before_j'])} - 이동 {number(rg['relocation_energy_j'])} -> {number(rg['battery_after_relocation_j'])} J (slot 0 시작값)", True)
            snaps=rg["user_state"]
            self.check("SETS", sorted(u["user"] for u in snaps)==sorted(members), "frame user snapshot 누락/중복")
            for u in snaps:
                uid=u["user"]
                self.check("SETS", int(u["x_m"]//c["region_length_m"])==int(m) and rec["membership"][uid]==int(m), f"u{uid} frame 시작 위치와 membership 불일치")
                self.check("QUEUE", close(u["Z"],c["large_queue_level"]-u["Q"]), f"u{uid} 시작 Z 불일치")
                if uid in self.last_users:
                    old_u=self.last_users[uid]
                    self.check("LINK", all(close(a,b) for a,b in zip((u["Q"],u["Z"],u["x_m"]),old_u)), f"u{uid} 이전 frame 끝 -> 현재 frame 시작 Q/Z/x 단절")
                    self.emit(f"  u{uid} 이전 frame 끝 -> 현재 시작: Q {number(old_u[0])} -> {number(u['Q'])}; Z {number(old_u[1])} -> {number(u['Z'])}; x {number(old_u[2])} -> {number(u['x_m'])} m; 담당 R{previous_owners[uid]} -> R{m}")
                else:
                    self.emit(f"  u{uid} episode 초기 상태: Q={number(u['Q'])}개, Z={number(u['Z'])}, x={number(u['x_m'])} m")
                self.last_users[uid]=(u["Q"],u["Z"],u["x_m"])
            self.last_sets[m]=(list(members),list(rsu),list(uav))
            self.last_battery[m]=rg["battery_after_relocation_j"]
            self.last_uav[m]=rg["uav_x_after"]
            self.totals[m]={"hover":0.0,"comm":0.0,"charge":0.0,"dpp":0.0,"stall":0,"users":Counter(),"dep":Counter(),"delivery":0}
        self.region="전체"

    def slot(self, rec):
        if self.frame is None:
            raise ValueError("frame_start 없는 slot")
        c, t = self.c, rec["slot_in_frame"]
        self.context += f" slot={t}"
        self.check("ORDER", t==self.next_slot and t<c["frame_slots"] and rec["global_slot"]==rec["frame"]*c["frame_slots"]+t and rec["remaining_slots_including_current"]==c["frame_slots"]-t, "슬롯 누락/중복/순서/remaining_slots 불일치")
        self.emit(f"\n--- SLOT {t} / {c['frame_slots']-1} | global_slot={rec['global_slot']} ---")
        mob=rec["mobility"]
        n=c["num_regions"]*c["users_per_region"]
        self.check("DATA", all(len(mob[k])==n for k in ("x_before","x_after","speed")), "mobility user 수 불일치")
        for m, rg in sorted(rec["regions"].items(), key=lambda p:int(p[0])):
            self.region=m
            start=self.frame["regions"][m]
            total=self.totals[m]
            before_fails=sum(self.failed.values())
            self.check("SETS", rg["rsu_users"]==start["executed_rsu_users"] and rg["uav_users"]==start["executed_uav_users"] and rg["hired"]==start["executed_hire"] and rg["point_index"]==start["executed_point"] and close(rg["uav_x"],start["uav_x_after"]), "frame 도중 association/hire/point/UAV 위치 변경")
            self.check("SETS", sorted(u["user"] for u in rg["users"])==sorted(start["members"]), "frame 도중 member 누락/추가/중복")
            self.emit(f"--- R{m} | RSU{m} -> {users(rg['rsu_users'])}; UAV{m} -> {users(rg['uav_users'])} ---")
            self.check("LINK", close(rg["battery_before_j"],self.last_battery[m]), "직전 종료 배터리 != 현재 시작 배터리")
            self.emit(f"배터리 연결: 직전 {number(self.last_battery[m])} -> 현재 시작 {number(rg['battery_before_j'])} J")
            hover=c["hovering_power_w"]*c["slot_duration_s"] if rg["hired"] else 0.0
            comm=c["slot_duration_s"]*rg["total_executed_power_w"]/c["pa_efficiency"] if rg["hired"] else 0.0
            charge=0.0 if rg["hired"] else max(0.0,min(c["charging_efficiency"]*c["charging_power_w"]*c["slot_duration_s"],c["battery_capacity_j"]-rg["battery_before_j"]))
            expected=rg["battery_before_j"]-hover-comm+charge
            self.check("ENERGY", all(close(a,b) for a,b in [(rg["hover_energy_j"],hover),(rg["communication_energy_j"],comm),(rg["charge_accepted_j"],charge),(rg["battery_after_j"],expected)]), "hover/통신/충전/최종 배터리 수지 불일치")
            self.check("ENERGY", 0<=rg["battery_after_j"]<=c["battery_capacity_j"]+1e-6 and close(rg["battery_soc_after"],rg["battery_after_j"]/c["battery_capacity_j"]), "배터리 범위/SoC 불일치")
            self.emit(f"배터리: {number(rg['battery_before_j'])} - hover {number(rg['hover_energy_j'])} - 통신 {number(rg['communication_energy_j'])} + 충전 {number(rg['charge_accepted_j'])} -> {number(rg['battery_after_j'])} J")
            self.emit(f"SoC: {100*rg['battery_before_j']/c['battery_capacity_j']:.4f}% -> {100*rg['battery_after_j']/c['battery_capacity_j']:.4f}%")
            remain=c["frame_slots"]-t
            pcap=min(c["uav_max_total_power_w"],c["pa_efficiency"]*max(0.0,rg["battery_before_j"]-c["reserve_battery_j"]-remain*hover)/c["slot_duration_s"]) if rg["hired"] else 0.0
            reserve=c["reserve_battery_j"]+(remain-1)*hover if rg["hired"] else 0.0
            self.check("POWER", close(rg["p_eff_w"],pcap) and close(rg["reserve_required_after_j"],reserve) and rg["battery_after_j"]+1e-6>=reserve and rg["reserve_ok"], "슬롯 전력 cap/잔여 hover reserve 불일치")
            reqs=sum(u["req_power_w"] for u in rg["users"] if u["provider"]==2)
            execs=sum(u["exec_power_w"] for u in rg["users"] if u["provider"]==2)
            scale=1.0 if reqs<=pcap+1e-12 else pcap/max(reqs,1e-12)
            self.check("POWER", close(reqs,rg["total_requested_power_w"]) and close(execs,rg["total_executed_power_w"]) and close(scale,rg["power_scale"]) and execs<=pcap+1e-9, "UAV 전력 합/공통 scaling 불일치")
            self.emit(f"UAV 전력: 요청 합 {number(reqs)} W -> 실행 합 {number(execs)} W <= cap {number(pcap)} W; 종료 필요 reserve {number(reserve)} J")
            if rg["hired"]:
                self.emit("고용 중 -> 실제 수신이 0개여도 hover 비용 발생. 송신했으나 0개 수신한 경우에도 통신 에너지 소모.")
            else:
                self.emit("미고용 -> depot에서 충전; 만충이면 수용 충전량 0 J.")
            delivered_slot=stall_slot=0
            for u in rg["users"]:
                uid=u["user"]
                uf=sum(self.failed.values())
                p=1 if uid in start["executed_rsu_users"] else 2 if uid in start["executed_uav_users"] else 0
                self.check("SETS", u["provider"]==p, f"u{uid} provider가 frame 고정 집합과 다름")
                prev=self.last_users[uid]
                self.check("LINK", all(close(a,b) for a,b in zip((u["q_before"],u["z_before"],u["x_m"]),prev)) and close(mob["x_before"][uid],u["x_m"]), f"u{uid} Q/Z/x 연속성 불일치")
                xb,xa=mob["x_before"][uid],mob["x_after"][uid]
                road=c["num_regions"]*c["region_length_m"]
                self.check("MOVE", close(xa,(xb+mob["speed"][uid]*c["slot_duration_s"])%road), f"u{uid} 이동 식 불일치")
                k,l,d=u["req_quality"],u["req_chunks"],u["delivered"]
                if not (type(k) is int and 0<=k<len(c["chunk_size_bits"]) and type(l) is int and type(d) is int):
                    raise ValueError(f"u{uid} quality/chunk 정수 domain 오류")
                dep=min(u["q_before"],c["playback_chunks_per_slot"])
                q_expected=u["q_before"]-dep+d
                self.check("QUEUE", u["q_before"]>=0 and close(u["departure"],dep) and close(u["q_after"],q_expected) and close(u["z_before"],c["large_queue_level"]-u["q_before"]) and close(u["z_after"],c["large_queue_level"]-u["q_after"]) and u["stall"]==int(u["q_before"]<c["playback_chunks_per_slot"]), f"u{uid} Q/Z/재생/stall 식 불일치")
                room=max(0,min(c["max_chunks_per_slot"],math.floor(c["large_queue_level"]-(u["q_before"]-dep)+1e-9))) if c["enforce_queue_admissibility"] else c["max_chunks_per_slot"]
                rate=math.floor(u["capacity_bps"]*c["slot_duration_s"]/c["chunk_size_bits"][k]+1e-9) if p and l>0 else 0
                feasible=min(c["max_chunks_per_slot"],room,rate) if l>0 else 0
                expected_delivery = (l if l <= feasible else 0) if c.get('delivery_mode', 'partial') == 'all_or_nothing' else min(l, feasible)
                self.check("DELIVERY", 0<=l<=c["max_chunks_per_slot"] and u["capacity_bps"]>=0 and u["queue_admissible_cap"]==room and u["feasible_by_rate"]==rate and u["feasible_chunks"]==feasible and d==expected_delivery, f"u{uid} 요청/링크 cap/버퍼 cap/실제 수신 불일치")
                if c.get('mask_queue_actions', False) and c['enforce_queue_admissibility']:
                    self.check('DELIVERY', l <= room, f'u{uid} PPO queue mask 위반')
                if p==0:
                    self.check("DELIVERY", l==d==0 and u["exec_power_w"]==0, f"미배정 u{uid}가 송수신함")
                if p==2:
                    preq=c["uav_max_total_power_w"]*u["req_power_level"]/(c["uav_power_levels"]-1) if l>0 else 0.0
                    self.check("POWER", type(u["req_power_level"]) is int and 0<=u["req_power_level"]<c["uav_power_levels"] and close(u["req_power_w"],preq) and close(u["exec_power_w"],preq*scale), f"u{uid} UAV power level/scaling 불일치")
                source=f"RSU{m}" if p==1 else f"UAV{m}" if p==2 else "미배정"
                if not p:
                    reason="고정 집합에서 미배정"
                elif l==0:
                    reason="배정은 유지; fast 정책이 이번 slot 요청 0개 선택"
                elif d<l:
                    limits=[]
                    if rate<l: limits.append(f"링크 용량 최대 {rate}개")
                    if room<l: limits.append(f"버퍼 여유 최대 {room}개")
                    reason=" / ".join(limits) or "실제 수신량 불일치 확인 필요"
                else:
                    reason="요청량 모두 수신"
                total["users"][uid]+=d
                total["dep"][uid]+=dep
                delivered_slot+=d
                stall_slot+=u["stall"]
                verdict="PASS" if sum(self.failed.values())==uf else "FAIL"
                quality=f"k={k}({k+1}단계)" if l>0 else "quality 미사용(요청 0)"
                self.emit(f"  {source} -> u{uid}: 요청 {l}개, {quality} -> 실제 {d}개 [{reason}]")
                if c.get('delivery_mode') == 'all_or_nothing' and l > 0 and d == 0:
                    self.emit('    요청 전체 실패: 부분 chunk는 수신량에 반영하지 않음. UAV 송신 에너지는 소모.')
                self.emit(f"    거리: RSU 수평 {number(u['rsu_horizontal_distance_m'])} m; "
                          + (f"UAV 수평 {number(u['uav_horizontal_distance_m'])} m" if u.get('uav_horizontal_distance_m') is not None else 'UAV 미고용'))
                if 'rsu_link_distance_m' in u:
                    self.emit(f"    3D 링크거리: RSU {number(u['rsu_link_distance_m'])} m; "
                              + (f"UAV {number(u['uav_link_distance_m'])} m" if u.get('uav_link_distance_m') is not None else 'UAV 미고용'))
                if 'chunk_action_cap' in u:
                    self.emit(f"    PPO chunk 선택 범위: 0..{u['chunk_action_cap']} (slot 내 재생 후 여유 반영)")
                if l>0:
                    self.emit(f"    링크 {number(u['capacity_bps']/1e6)} Mbps / chunk {number(c['chunk_size_bits'][k]/1e6)} Mbit -> 링크 최대 {u['feasible_by_rate']}개; 버퍼 허용 {u['queue_admissible_cap']}개")
                else:
                    self.emit("    전송 요청 없음 -> 링크 용량 계산 생략 (기록된 0 bps를 채널 불량으로 해석하지 않음)")
                self.emit(f"    Q: {number(u['q_before'])} - 재생 {number(u['departure'])} + 수신 {d} -> {number(u['q_after'])}개; Z: {number(u['z_before'])} -> {number(u['z_after'])}; stall={'발생' if u['stall'] else '없음'}; frame 수신 누계 {total['users'][uid]}개 [{verdict}]")
                location=int(xa//c["region_length_m"])
                if location!=int(m):
                    self.emit(f"    이동 x {number(xb)} -> {number(xa)} m: 물리적 R{location}로 이동했지만 현재 frame 담당 R{m} 유지 -> 다음 frame에서 membership 재계산")
                if p==2:
                    self.emit(f"    UAV 송신전력 요청 {number(u['req_power_w'])} -> 실행 {number(u['exec_power_w'])} W")
                self.last_users[uid]=(u["q_after"],u["z_after"],xa)
            self.check("TOTAL", close(delivered_slot,rg["delivered_chunks"]) and stall_slot==rg["stall_user_slots"], "slot 지역 수신/stall 합계 불일치")
            total["delivery"]+=delivered_slot
            total["stall"]+=stall_slot
            for key,field in (("hover","hover_energy_j"),("comm","communication_energy_j"),("charge","charge_accepted_j"),("dpp","dpp_slot_cost")):
                total[key]+=rg[field]
            self.last_battery[m]=rg["battery_after_j"]
            self.last_uav[m]=rg["uav_x"]
            self.emit(f"  slot 정합성 [{'PASS' if sum(self.failed.values())==before_fails else 'FAIL'}]: 고정 집합/point, 상태 연결, 배터리, 전력, chunk, Q/Z 확인")
        self.region="전체"
        self.next_slot+=1

    def end_frame(self, rec):
        if self.frame is None:
            raise ValueError("frame_start 없는 frame_end")
        c=self.c
        self.check("ORDER", self.next_slot==c["frame_slots"], f"완료 slot {self.next_slot}/{c['frame_slots']}")
        self.emit(f"\n--- FRAME 종료 | EP {self.key[0]} / FRAME {self.key[1]} ---", True)
        for m, rg in sorted(rec["regions"].items(), key=lambda p:int(p[0])):
            self.region=m
            start,total=self.frame["regions"][m],self.totals[m]
            consumed=start["relocation_energy_j"]+total["hover"]+total["comm"]
            self.check("TOTAL", close(rg["energy_consumed_j"],consumed) and close(rg["energy_charged_j"],total["charge"]) and close(rg["delivered_chunks"],total["delivery"]) and rg["stall_user_slots"]==total["stall"], "frame 에너지/수신/stall 합계 불일치")
            end=start["battery_before_j"]-consumed+total["charge"]
            self.check("TOTAL", close(end,self.last_battery[m]) and close(rg["battery_soc_end"]*c["battery_capacity_j"],end) and close(rg["uav_x_end"],self.last_uav[m]), "frame 끝 배터리/SoC/위치 불일치")
            hiring=c["lambda_h"]*c["hiring_cost_per_frame"]*start["executed_hire"]
            self.check("TOTAL", rg["hired"]==start["executed_hire"] and close(rg["hiring_cost_weighted"],hiring) and close(rg["frame_dpp_cost"],total["dpp"]+c["lyapunov_v"]*hiring), "고용비 1회/DPP 합계 불일치")
            self.emit(f"R{m}: slot {self.next_slot}/{c['frame_slots']}개 기록; RSU={users(start['executed_rsu_users'])}; UAV={users(start['executed_uav_users'])}", True)
            self.emit(f"  배터리 {number(start['battery_before_j'])} - 이동 {number(start['relocation_energy_j'])} - hover 합 {number(total['hover'])} - 통신 합 {number(total['comm'])} + 충전 합 {number(total['charge'])} -> {number(self.last_battery[m])} J", True)
            self.emit("  사용자별 frame 총수신: " + ", ".join(f"u{u}={total['users'][u]}개" for u in start["members"]), True)
            self.emit(f"  frame DPP: slot 합 {number(total['dpp'])} + V×고용비 {number(c['lyapunov_v']*hiring)} -> {number(rg['frame_dpp_cost'])}")
            for code in LABELS:
                p=self.frame_pass[m][code]+self.frame_pass["전체"][code]
                f=self.frame_fail[m][code]+self.frame_fail["전체"][code]
                if p or f:
                    self.emit(f"  [{'FAIL' if f else 'PASS'}] {LABELS[code]}: 통과 {p}, 실패 {f}")
            failures=sum(self.frame_fail[m].values())+sum(self.frame_fail["전체"].values())
            self.emit(f"  frame 검사: {'FAIL' if failures else 'PASS'} (실패 {failures}건)", True)
        self.region="전체"
        self.emit("다음 frame이 있으면: 마지막 Q/Z/x/배터리를 넘겨받는지 다음 FRAME 시작에서 검사합니다.\n", True)
        if self.frame_file:
            self.frame_file.close()
            self.frame_file=None
        self.frame=None
        self.next_frame+=1
        self.next_slot=0


def main(argv=None):
    p=argparse.ArgumentParser(description="trace.jsonl을 직관적인 한국어 slot/frame 텍스트로 저장")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, help="새 출력 폴더; 기존 폴더 덮어쓰기 금지")
    p.add_argument("--episode", type=int, action="append", help="출력할 episode (반복 지정 가능)")
    p.add_argument("--frame", type=int, action="append", help="출력할 frame (반복 지정 가능)")
    p.add_argument("--region", type=int, action="append", help="출력할 region (반복 지정 가능)")
    p.add_argument("--watch", type=float, default=0, help="0: 현재 로그 1회 변환; 양수: 갱신 간격(초), 최소 1")
    a=p.parse_args(argv)
    if not math.isfinite(a.watch) or a.watch<0 or 0<a.watch<1:
        p.error("--watch는 0 또는 1초 이상")
    for field in (a.episode,a.frame,a.region):
        if field and any(i<0 for i in field): p.error("index는 0 이상")
    run=a.run_dir.resolve()
    src,config_path=run/"trace.jsonl",run/"resolved_config.json"
    if not src.is_file() or not config_path.is_file():
        p.error("run_dir에 trace.jsonl과 resolved_config.json이 필요합니다")
    try:
        config_bytes=config_path.read_bytes()
        cfg=strict_json(config_bytes)["config"]
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.error(f"설정 파일을 읽을 수 없습니다: {exc}")
    if a.region and any(m>=cfg["num_regions"] for m in a.region): p.error("region 범위 초과")
    if a.frame and any(f>=cfg["num_frames"] for f in a.frame): p.error("frame 범위 초과")
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out=(a.out or run/"artifacts"/f"readable_{stamp}_{os.getpid()}").resolve()
    try:
        out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        p.error(f"출력 폴더가 이미 있습니다. 새 --out을 지정하거나 --out을 생략하세요: {out}")
    atomic_json(out/"provenance.json",{"source_run":str(run),"compatible_commit":BASE_COMMIT,"config_sha256":hashlib.sha256(config_bytes).hexdigest(),"filters":{"episode":a.episode,"frame":a.frame,"region":a.region},"checker":"battery/delivery/continuity/association/frame totals; not radio or PPO verification"})
    n=Narrator(cfg,out,a)
    offset=lines=pending=0
    identity=None
    status="IN_PROGRESS"
    (out/"README.txt").write_text(n.scope_note+"\nframes_summary.txt: 프레임 요약\nepisodes/epXXXXXX/frameXXXXXX.txt: slot 상세\nissues.txt: 실패 위치(최대 2000건)\nstatus.json: COMPLETE / INVALID / INCOMPLETE / IN_PROGRESS / STOPPED\n읽기 완료와 정합성 PASS를 구분하세요. INCOMPLETE는 완료 판정이 아닙니다.\n",encoding="utf-8")
    print(f"[OUTPUT] {out}",flush=True)

    def save_status():
        n.flush()
        atomic_json(out/"status.json",{"status":status,"records":lines,"committed_bytes":offset,"pending_bytes":pending,"episodes_completed":n.episodes,"expected_episodes":n.expected,"frames_written":n.rendered,"passed":dict(n.passed),"failed":dict(n.failed),"failed_total":sum(n.failed.values()),"source_run":str(run),"checked_at":datetime.now(timezone.utc).isoformat(),"scope":n.scope_note})
        text=[f"상태: {status}",f"완료 episode: {n.episodes}/{n.expected}",f"처리 record: {lines}; 미처리 마지막 행 bytes: {pending}",f"출력 frame 파일: {n.rendered}",n.scope_note]
        text += [f"{LABELS[k]}: PASS {n.passed[k]} / FAIL {n.failed[k]}" for k in LABELS]
        (out/"checks_summary.txt").write_text("\n".join(text)+"\n",encoding="utf-8")

    try:
        while True:
            stat=src.stat()
            current=(stat.st_dev,stat.st_ino)
            if identity is None: identity=current
            if current!=identity or stat.st_size<offset:
                raise ValueError("source trace가 교체/축소됨; 새 출력 폴더로 다시 실행")
            if config_path.read_bytes()!=config_bytes:
                raise ValueError("실행 중 resolved_config.json 변경 감지")
            stop=stat.st_size
            pending=0
            with src.open("rb") as stream:
                stream.seek(offset)
                while stream.tell()<stop:
                    position=stream.tell()
                    data=stream.readline(min(MAX_LINE+1,stop-position))
                    if len(data)>MAX_LINE: raise ValueError("trace 한 행이 64 MiB 초과")
                    if not data.endswith(b"\n"):
                        pending=len(data)
                        break
                    rec=strict_json(data)
                    n.process(rec,lines+1)
                    offset=stream.tell()
                    lines+=1
            status="INVALID" if n.failed else "COMPLETE" if n.ended and not pending else "IN_PROGRESS" if a.watch else "INCOMPLETE"
            if n.ended and pending:
                n.check("ORDER",False,"run_end 이후 미완성 데이터 존재")
                status="INVALID"
            save_status()
            print(f"[{status}] records={lines} episodes={n.episodes}/{n.expected} failures={sum(n.failed.values())}",flush=True)
            if n.ended or not a.watch or n.failed:
                return 1 if n.failed else 0
            time.sleep(a.watch)
    except KeyboardInterrupt:
        status="STOPPED"
        save_status()
        return 130
    except (OSError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError,OverflowError) as exc:
        n.check("DATA",False,f"읽기/필수 필드 오류: {type(exc).__name__}: {exc}")
        status="INVALID"
        save_status()
        print(f"[INVALID] {exc}; see {out/'issues.txt'}",file=sys.stderr)
        return 1
    finally:
        n.dispose()


if __name__=="__main__":
    raise SystemExit(main())
