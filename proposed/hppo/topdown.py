"""Ground projection of the existing 1-D physical model, one panel per region."""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {0: '#7b8794', 1: '#2166b1', 2: '#d17a14'}


def frame_figure(start, slot, cfg, run_name):
    columns = 2 if cfg.num_regions > 1 else 1
    rows = math.ceil(cfg.num_regions / columns)
    # Membership varies after handovers. Reserve table space for every user.
    table_heights = [.23 * (1 + max(len(slot['regions'][str(m)]['users'])
                     for m in range(r*columns, min((r+1)*columns, cfg.num_regions))))
                     for r in range(rows)]
    heights = [2.6 + height for height in table_heights]
    fig = plt.figure(figsize=(9 * columns, sum(heights) + 1.0), layout='constrained')
    grid = fig.add_gridspec(rows, columns, height_ratios=heights)
    for m in range(cfg.num_regions):
        cell = grid[m // columns, m % columns].subgridspec(2, 1, height_ratios=[2.6, table_heights[m // columns]])
        ax, table_ax = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])
        rg, initial = slot['regions'][str(m)], start['regions'][str(m)]
        members = rg['users']
        rx, ux = cfg.rsu_x(m), rg['uav_x']
        left, right = m * cfg.region_length_m, (m + 1) * cfg.region_length_m
        xs = [u['x_m'] for u in members] + [left, right, ux, initial['uav_x_before']]
        low, high = min(xs), max(xs)
        pad = max((high-low) * .06, 15.)
        ax.set_xlim(low-pad, high+pad)
        ax.set_ylim(-85, 130)
        ax.set_aspect('equal', adjustable='box')
        ax.axhspan(-5, 5, color='#e6eaee')
        ax.axhline(0, color='#929da7', lw=.8, ls='--')
        ax.axvline(left, color='#b7c1ca', lw=.8, ls=':')
        ax.axvline(right, color='#b7c1ca', lw=.8, ls=':')
        ax.scatter(cfg.candidate_points(m), [0]*cfg.num_candidate_points,
                   marker='D', s=22, color='#b9c3ca', zorder=3)
        ax.scatter(rx, 0, s=200, marker='^', color=COLORS[1], zorder=6)
        ax.annotate(f'RSU {m}', (rx, 0), xytext=(0, -43), textcoords='offset points',
                    ha='center', fontsize=9, color=COLORS[1], arrowprops={'arrowstyle':'-', 'color':COLORS[1]})
        ax.scatter(ux, 0, s=95, marker='X', color=COLORS[2] if rg['hired'] else '#657381',
                   edgecolors='white', linewidths=.7, zorder=7)
        ax.annotate(f"UAV {m} {'hired' if rg['hired'] else 'charging'}\nSoC {100*rg['battery_after_j']/cfg.battery_capacity_j:.1f}%",
                    (ux, 0), xytext=(0, 46), textcoords='offset points', fontsize=8,
                    ha='center', color=COLORS[2], arrowprops={'arrowstyle':'-', 'color':COLORS[2]})
        before = initial['uav_x_before']
        if abs(ux-before) > 1e-9:
            ax.annotate('', (ux, 105), (before, 105),
                        arrowprops={'arrowstyle':'->', 'color':COLORS[2], 'lw':2})
            ax.text((ux+before)/2, 115, f'relocate {abs(ux-before):.0f} m',
                    ha='center', fontsize=7, color=COLORS[2])
        label_ends = [-float('inf')] * 4
        for j, u in enumerate(sorted(members, key=lambda u: u['x_m'])):
            x, p = u['x_m'], u['provider']
            ax.scatter(x, 0, s=38, color=COLORS[p], edgecolors='white', zorder=8)
            lane = min(range(4), key=lambda k: label_ends[k])
            label_ends[lane] = x
            offset = [-14, 14, -28, 28][lane]
            ax.annotate(f"u{u['user']}", (x, 0), xytext=(0, offset), textcoords='offset points',
                        fontsize=7, ha='center', color=COLORS[p],
                        arrowprops={'arrowstyle':'-', 'lw':.4, 'color':'#98a5b0'})
            if p:
                source = rx if p == 1 else ux
                # Curved routes indicate association, not a physical flight path.
                ax.annotate('', (x, 0), (source, 0),
                            arrowprops={'arrowstyle':'->', 'lw':.8, 'alpha':.6,
                                        'color':COLORS[p], 'connectionstyle':f'arc3,rad={.22 if p == 1 else -.22}'})
        ax.set_yticks([])
        ax.set_xlabel('Ground x (m); all physical y = 0', fontsize=8)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.set_title(f"Region {m} | last-slot decision positions | RSU {len(rg['rsu_users'])} / UAV {len(rg['uav_users'])}",
                     loc='left', fontsize=10, weight='bold')
        table_ax.axis('off')
        entries = []
        for u in sorted(members, key=lambda u: u['user']):
            du = u.get('uav_horizontal_distance_m')
            du3 = math.hypot(du, cfg.uav_height_m-cfg.user_height_m) if du is not None else None
            dr = u['rsu_horizontal_distance_m']
            dr3 = math.hypot(dr, cfg.rsu_height_m-cfg.user_height_m)
            entries.append([f"u{u['user']}", ['none','RSU','UAV'][u['provider']],
                            f'{dr:.1f} / {dr3:.1f}', '--' if du is None else f'{du:.1f} / {du3:.1f}',
                            f"{u['req_chunks']} / {u['delivered']}", f"{u['q_before']:.0f} > {u['q_after']:.0f}"])
        if entries:
            table = table_ax.table(cellText=entries,
                                   colLabels=['User','Source','RSU h / 3D (m)','UAV h / 3D (m)','Req / Rx','Q before > after'],
                                   loc='upper center', cellLoc='center', colWidths=[.07,.09,.23,.23,.13,.25])
            table.auto_set_font_size(False); table.set_fontsize(7.5)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor('#dce3e8'); cell.set_linewidth(.4)
                if row == 0:
                    cell.set_facecolor('#eaf0f5'); cell.set_text_props(weight='bold')
                elif row % 2 == 0:
                    cell.set_facecolor('#f8fafb')
        else:
            table_ax.text(.5,.7,'No frame members',ha='center')
    fig.suptitle(f"{run_name} | episode {slot['episode']} / frame {slot['frame']} / slot {slot['slot_in_frame']}\n"
                 'TOP VIEW: physical ground projection. Label offsets and curved links are display only.\n'
                 'Arrows above the road: UAV relocation at frame start. Distances: last-slot START; Q and SoC: after that slot.',
                 fontsize=12)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[p],label=label)
                        for p,label in [(0,'Unassigned user'),(1,'RSU association'),(2,'UAV association')]],
               loc='outside lower center', ncol=3, fontsize=9)
    return fig
