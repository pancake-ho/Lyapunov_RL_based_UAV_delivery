"""Reproducible radio calibration; no training or default mutation.

Use the repository radio functions directly. Samples are paired across BW
and quality at each distance/power/seed. Outcomes are link feasibility only.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from config_hppo import HPPOConfig
from env.p3.radio import rsu_link_capacity_bps
from hppo.env import uav_link_capacity_bps
from hppo.logger import jsonable


def plot_comparison(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, provider in enumerate(['RSU', 'UAV']):
        for row, quality in enumerate([0, 3]):
            ax = axes[row, col]
            for bw in ([20, 5, 3, 1] if provider == 'RSU' else [5, 3, 1]):
                distances = [0,25,50,75,100,125,150,175,200,250,300]
                values = []
                for d in distances:
                    values.append(100 * np.mean([r['success_1_chunks'] for r in rows
                        if r['provider'] == provider and r['total_bw_mhz'] == bw
                        and r['quality_index'] == quality and r['horizontal_distance_m'] == d
                        and (provider == 'RSU' or r['power_per_user_w'] == 1.5)]))
                selected = (provider == 'RSU' and bw == 3) or (provider == 'UAV' and bw == 5)
                ax.plot(distances, values, 'o-', ms=3, label=f'{bw} MHz' + (' (selected)' if selected else ''))
            ax.set_title(f'{provider} | quality {quality+1} ({[.5,1,2,4][quality]} Mbit/chunk)')
            ax.set_xlabel('Horizontal distance (m)')
            ax.set_ylabel('P(one requested chunk succeeds) [%]')
            ax.set_ylim(-2,102); ax.grid(alpha=.2); ax.legend(fontsize=8)
    count = sum(r['samples'] for r in rows if r['provider'] == 'RSU' and r['total_bw_mhz'] == 20
                and r['quality_index'] == 0 and r['horizontal_distance_m'] == 0)
    fig.suptitle(f'BW calibration: same fading samples across candidates\n{count:,} samples per point; fixed per-user RBs; UAV 1.5 W per user', fontsize=13)
    fig.savefig(out/'bandwidth_comparison.png', dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/bw_calibration'))
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--seeds', default='2026,2027,2028')
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('--samples must be positive')
    seeds = [int(s) for s in args.seeds.split(',')]
    cfg = HPPOConfig(num_regions=10, users_per_region=5, rsu_total_bandwidth_hz=20e6)
    distances = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 1000, 3999]
    sweep = {'RSU': [20, 10, 5, 3, 2, 1], 'UAV': [5, 3, 2, 1]}
    rows = []
    for seed in seeds:
        fading = np.clip(np.random.default_rng(seed).exponential(size=args.samples), .05, 10.)
        for provider, bandwidths in sweep.items():
            powers = [cfg.rsu_total_power_w / cfg.rsu_capacity] if provider == 'RSU' else [.5, 1.5, 3.]
            for power in powers:
                for bw in bandwidths:
                    c = replace(cfg, **{('rsu' if provider == 'RSU' else 'uav') + '_total_bandwidth_hz': bw * 1e6})
                    for distance in distances:
                        if provider == 'RSU':
                            rates = np.fromiter((rsu_link_capacity_bps(distance, f, c) for f in fading), float)
                        else:
                            rates = np.fromiter((uav_link_capacity_bps(distance, f, power, c)[0] for f in fading), float)
                        for k, bits in enumerate(c.chunk_size_bits):
                            caps = np.floor(rates * c.slot_duration_s / bits + 1e-9)
                            row = dict(seed=seed, provider=provider, total_bw_mhz=bw,
                                       power_per_user_w=power, horizontal_distance_m=distance,
                                       quality_index=k, chunk_mbit=bits/1e6, samples=args.samples,
                                       rate_mbps_p10=float(np.quantile(rates, .1)/1e6),
                                       rate_mbps_p50=float(np.median(rates)/1e6),
                                       rate_mbps_p90=float(np.quantile(rates, .9)/1e6),
                                       chunks_p10=float(np.quantile(caps, .1)),
                                       chunks_p50=float(np.median(caps)),
                                       chunks_p90=float(np.quantile(caps, .9)))
                            for n in range(1, 4):
                                row[f'success_{n}_chunks'] = float(np.mean(caps >= n))
                            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'bandwidth_samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metadata = dict(config=asdict(cfg), seeds=seeds, samples_per_seed=args.samples,
                    distances_m=distances, bandwidths_mhz=sweep, python=platform.python_version(),
                    numpy=np.__version__, scope='Paired channel samples, not trained-policy performance',
                    note='UAV 3 W is a one-user upper bound; 1.5 W is equal sharing for two active users. Distances use abs(x_provider-x_user), as in the current environment.')
    (args.out/'metadata.json').write_text(json.dumps(jsonable(metadata), indent=2, allow_nan=False)+'\n')
    plot_comparison(rows, args.out)
    print(f'{len(rows)} rows -> {args.out / "bandwidth_samples.csv"}')


if __name__ == '__main__':
    main()
