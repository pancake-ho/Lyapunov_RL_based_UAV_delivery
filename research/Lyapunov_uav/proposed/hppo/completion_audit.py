"""Human-readable candidate arithmetic; shared by live and exported logs."""
from __future__ import annotations


def candidate_lines(detail):
    if not detail or not detail.get('point_checks'):
        return []
    lines = [
        '  [DPP 후보 비교] slow PPO scheduling을 고정하고 no-hire 및 모든 feasible point를 비교',
        '  DPP_s = V*lambda_H*c_H*hire + sum_(t,n)[alpha_Z*Z*(departure-delivered) + V*delivered*(q_max-q_k)]',
        '  예상 DPP = scenario별 DPP의 평균; 현재 fast policy 고정, 독립 미래 표본, 후보 간 공통 난수',
        '  아래 slot/user 합계는 후보 rollout의 예측값이며 실제 미래 관측값이 아님.',
    ]
    for point in detail['point_checks']:
        status = '평가' if point['feasible'] else '제외: ' + ','.join(point['excluded_reasons'])
        lines.append(f"    point={point['point']} x={point['target_x_m']:.1f}m "
                     f"이동={point['distance_m']:.1f}m 이동에너지={point['relocation_energy_j']:.1f}J "
                     f"이동후배터리={point['battery_after_move_j']:.1f}J [{status}]")
    for i, row in enumerate(detail['candidates']):
        chosen = ' [선택]' if i == detail['selected_index'] else ''
        lines.append(f"    후보 {i}: hire={row['hired']} point={row['point']} "
                     f"RSU={row['rsu_users']} UAV={row['uav_users']}{chosen}")
        for sample in row['sample_components']:
            lines.append(f"      scenario {sample['scenario']}: queue {sample['queue_drift']:.6f} "
                         f"+ quality {sample['quality_dpp']:.6f} + hiring {sample['hiring_dpp']:.6f} "
                         f"= DPP {sample['frame_dpp']:.6f}")
            lines.append('        slot: queue + quality = J_F; delivery; failed requests')
            for term in sample['slots']:
                lines.append(f"        t={term['slot']}: {term['queue_drift']:.6f} + "
                             f"{term['quality_dpp']:.6f} = {term['dpp']:.6f}; "
                             f"d={term['delivered']} fail={term['failed_requests']}")
            for term in sample['users']:
                lines.append(f"        u{term['user']} frame합: queue={term['queue_drift']:.6f}, "
                             f"quality={term['quality_dpp']:.6f}, d={term['delivered']}, "
                             f"fail={term['failed_requests']}")
        lines.append(f"      평균: {row['mean_queue_drift']:.6f} + {row['mean_quality_dpp']:.6f} "
                     f"+ {row['hiring_dpp']:.6f} = {row['mean_dpp']:.6f}; "
                     f"미고용 대비={row['delta_from_no_hire']:+.6f}; 최소값 대비={row['delta_from_selected']:+.6f}")
    return lines
