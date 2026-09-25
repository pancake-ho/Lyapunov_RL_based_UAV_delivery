"""Evaluate a matched checkpoint pair using its saved experiment configuration."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
from hppo.verify_trace import load_config
from hppo.train import main as run

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-run',type=Path,required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--eval-episodes',type=int,default=5)
    p.add_argument('--episode-offset',type=int,default=1_000_000)
    p.add_argument('--device',default=None)
    p.add_argument('--output-dir',type=Path,default=None)
    a=p.parse_args(argv)
    cfg=asdict(load_config(a.train_run))
    saved=json.loads((a.train_run/'resolved_config.json').read_text(encoding='utf-8'))
    checkpoints=a.train_run/saved['args'].get('checkpoint_dir','checkpoints')
    cfg.update(eval_episodes=a.eval_episodes,episode_offset=a.episode_offset)
    if a.device is not None:cfg['device']=a.device
    cli=['--mode','eval','--run-name',a.run_name,'--output-dir',str(a.output_dir or a.train_run.parent),
         '--frame-checkpoint',str(checkpoints/'frame_latest.pt'),
         '--slot-checkpoint',str(checkpoints/'slot_latest.pt')]
    for key,value in cfg.items():
        text=','.join(str(v) for v in value) if isinstance(value,(tuple,list)) else str(value)
        cli.append('--'+key.replace('_','-')+'='+text)
    run(cli)

if __name__=='__main__':main()
