"""Plot actual episode metrics and a bounded representative trace.

No smoothing or interpolation is used. Training plots describe the supplied
run only; a smoke run is not evidence of convergence.
"""
from __future__ import annotations
import argparse, csv, json, os, tempfile
from pathlib import Path
import numpy as np

def plot_run(run_dir: Path, episode=None, max_trace_slots=1000, allow_partial=False):
    from hppo.verify_trace import verify
    if not allow_partial:
        _, failed, reports=verify(run_dir)
        if failed:raise ValueError('incomplete/invalid run: '+'; '.join(reports[:3]))
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'hppo_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})
    with (run_dir/'episode_summary.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('no completed episodes to plot')
    x=np.array([int(r['episode']) for r in rows])
    series=lambda key:np.array([float(r[key]) for r in rows])
    if episode is None:episode=int(x[-1])
    out=run_dir/'plots';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained')
    axes[0,0].plot(x,series('slow_reward_sum'),label='Slow frame reward sum',color='#165d96',marker='o',ms=3)
    axes[0,0].plot(x,series('fast_reward_sum'),label='Fast slot reward sum',color='#db7c26',marker='o',ms=3)
    axes[0,0].set(title='Returns (raw episode sums)',ylabel='Scaled reward');axes[0,0].legend(fontsize=8)
    specifications=[(axes[0,1],'average_quality_utility','Delivered-chunk quality','Utility per delivered chunk'),
                    (axes[1,0],'stall_ratio','Playback stalls','Stalled user-slots / all user-slots'),
                    (axes[1,1],'hire_rate','UAV employment','Hired UAV-frames / all UAV-frames'),
                    (axes[2,0],'min_battery_soc','Minimum battery state','Minimum SoC'),
                    (axes[2,1],'completion_runtime_s','Hiring/location comparison time','Seconds per episode')]
    for ax,key,title,label in specifications:
        y=series(key)
        if key=='average_quality_utility':y=np.where(series('delivered_chunks_total')>0,y,np.nan)
        ax.plot(x,y,color='#165d96',marker='o',ms=3);ax.set(title=title,ylabel=label)
    for ax in axes.flat:
        ax.set_xlabel('Episode')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    # Display bounded fractions on their physical scale, without offset zoom.
    for ax in (axes[1,0],axes[1,1],axes[2,0]):ax.set_ylim(0,1.02)
    fig.suptitle(f'{run_dir.name} | completed episodes: {len(rows)}'+(' | PARTIAL LIVE VIEW' if allow_partial else ''),fontsize=15)
    fig.savefig(out/'training_overview.png',dpi=170);plt.close(fig)
    records=[]
    with (run_dir/'trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if allow_partial:break
                raise
            if r.get('event')=='slot' and r.get('episode')==episode:
                records.append(r)
                if len(records)>=max_trace_slots:break
    if not records:raise ValueError(f'no trace slots for episode {episode}')
    slots=np.array([r['global_slot'] for r in records]);region_ids=sorted(records[0]['regions'],key=int)
    user_ids=sorted({u['user'] for r in records for rg in r['regions'].values() for u in rg['users']})
    provider=np.zeros((len(user_ids),len(records)));queue=np.full_like(provider,np.nan,dtype=float)
    index={u:i for i,u in enumerate(user_ids)}
    fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained',sharex=True)
    for m in region_ids[:10]:
        axes[0].plot(slots,[r['regions'][m]['battery_soc_after'] for r in records],label=f'UAV {m}',lw=1.4)
    axes[0].set(title='Battery after each actual slot (up to 10 UAVs)',ylabel='SoC',ylim=(0,1.02));axes[0].legend(ncol=5,fontsize=8)
    for j,r in enumerate(records):
        for rg in r['regions'].values():
            for u in rg['users']:
                provider[index[u['user']],j]=u['provider'];queue[index[u['user']],j]=u['q_after']
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap(['#d7dde3','#2b6fbb','#ee9b42'])
    im=axes[1].imshow(provider,aspect='auto',interpolation='nearest',origin='lower',cmap=cmap,
                     norm=BoundaryNorm([-.5,.5,1.5,2.5],3),extent=[slots[0]-.5,slots[-1]+.5,-.5,len(user_ids)-.5])
    cb=fig.colorbar(im,ax=axes[1],ticks=[0,1,2],pad=.01);cb.ax.set_yticklabels(['Unserved','RSU','UAV'])
    axes[1].set(title='Executed provider by user',ylabel='User ID')
    axes[2].plot(slots,np.nanmean(queue,axis=0),label='Mean buffer',color='#165d96')
    axes[2].plot(slots,np.nanmin(queue,axis=0),label='Minimum buffer',color='#db7c26')
    axes[2].set(title='Playback queue after delivery',ylabel='Chunks',xlabel='Actual global slot within episode');axes[2].legend()
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f'{run_dir.name} | episode {episode} | first {len(records)} slots',fontsize=15)
    fig.savefig(out/'trace_overview.png',dpi=170);plt.close(fig)
    # Log DPP candidate scores for one real decision; only the selected candidate executes.
    decision=None
    with (run_dir/'trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:break
            if r.get('event')=='frame_start' and r.get('episode')==episode:
                decision=r;break
    if decision:
        m=next(iter(decision['regions']));d=decision['regions'][m]['completion']
        labels=['No hire' if c['hired']==0 else f'Point {c["point"]}' for c in d['candidates']]
        values=[c['mean_dpp'] for c in d['candidates']]
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        ax.bar(labels,values,color=['#e49335' if i==d['selected_index'] else '#406e9b' for i in range(len(labels))])
        ax.set(title=f'Frame {decision["frame"]}, region {m}: fixed scheduling candidates',ylabel='Mean rollout DPP (lower is better)')
        fig.savefig(out/'completion_example.png',dpi=170);plt.close(fig)
    return out

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path)
    p.add_argument('--episode',type=int);p.add_argument('--max-trace-slots',type=int,default=1000)
    p.add_argument('--allow-partial',action='store_true',help='explicitly label a live incomplete run')
    a=p.parse_args(argv)
    if a.max_trace_slots<=0:p.error('--max-trace-slots must be positive')
    print(plot_run(a.run_dir,a.episode,a.max_trace_slots,a.allow_partial))

if __name__=='__main__':main()
