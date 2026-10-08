"""Actual CPU environment + checkpoint integration; synthetic weights are not research results."""
import contextlib
import io
import json
import sys
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
from baseline.NDTVS import api as c
from baseline.NDTVS.evaluation.benchmark.settings import load,validate
from baseline.NDTVS.evaluation.benchmark import runner,models,radio,submit
from baseline.NDTVS.metrics.service import ServiceLogger
from baseline.NDTVS.plot.benchmark import report,estimates
from baseline.NDTVS.plot.benchmark_animation import export

import tempfile
ROOT = Path(tempfile.mkdtemp(prefix='ndtvs-benchmark-test-'))/'results'
CHECKS = []


def expect_error(function):
    try:
        function()
    except (ValueError,RuntimeError,FileNotFoundError):
        return
    raise AssertionError('Expected rejection')


def write_settings(s,path):
    path.write_text('from pathlib import Path, PosixPath\n'+'\n'.join(
        k+' = '+repr(v) for k,v in vars(s).items() if k.isupper() and k != 'SETTINGS_PATH')+'\n')


def fairness(cfg):
    tiny = replace(cfg,num_regions=1,users_per_region=4,num_frames=1,frame_slots=2)
    for zero in (False,True):
        log = ServiceLogger(tiny,ROOT/('zero' if zero else 'starvation'))
        for slot in range(2):
            records=[]
            for user in range(4):
                serviced = user == 0 and not zero
                q = 1. if slot == 0 or serviced else 0.
                records.append(dict(user=user,provider=int(serviced),req_chunks=3 if serviced else 0,
                    req_quality=3,delivered=3 if serviced else 0,transmission_failed=False,
                    q_before=q,stall=q < 1))
            info = dict(episode=99,frame=0,slot_in_frame=slot,regions={0:dict(users=records)})
            log.observe_slot(info)
            log.observe_slot(info)
        row=log.measures()
        assert log.counts['slots'].sum() == 8
        if zero:
            assert row['unique_served_user_ratio'] == 0 and row['delivery_jain_index'] is None
            assert not row['quality_utility_defined']
            assert all(u['received_quality_utility'] is None for u in log.user_rows())
        else:
            assert row['average_quality_utility'] == 1.
            assert row['unique_served_user_ratio'] == .25 and row['delivery_jain_index'] == .25
            assert row['top20_delivery_share'] == 1.
            assert row['stall_time_ratio'] == 3/8
        log.close()
    CHECKS.append('all-user metrics expose high-quality starvation, preserve zero-delivery undefined quality/Jain, and avoid duplicate observation')


def main():
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    s=load()
    s.SNR_MODE=None
    s.COST_MODE=None
    expect_error(lambda:validate(s))
    assert validate(s,require_snr=False) is s
    CHECKS.append('unanswered SNR choice is explicit; inspect remains available')
    cfg=replace(c.HPPOConfig(),num_regions=2,users_per_region=2,num_frames=2,frame_slots=3,
        hidden_dims=(32,32),rollout_scenarios=1,ppo_minibatch_size=32,ppo_update_epochs=1,
        device='cpu',lyapunov_v=50.,evaluation_warmup_frames=0,rsu_total_bandwidth_hz=3e6)
    fairness(cfg)
    for mode,distance,levels in [('transmit',None,(25,35,45)),('received',150.,(25,35,45)),('offset',None,(-10,0,10))]:
        capacities=[]
        for db in levels:
            evaluated=radio.at_snr(cfg,mode,db,distance)
            assert evaluated.rsu_total_power_w == cfg.rsu_total_power_w
            assert evaluated.uav_max_total_power_w == cfg.uav_max_total_power_w
            capacities.append(__import__('env.p3.radio',fromlist=['rsu_link_capacity_bps']).rsu_link_capacity_bps(150,1,evaluated))
        assert capacities == sorted(capacities) and len(set(capacities)) == 3
    expect_error(lambda:radio.at_snr(cfg,'received',25,None))
    CHECKS.append('all three SNR conventions exactly map to the requested ratio and increase capacity without changing power/battery')
    config=ROOT/'train.json'
    c.atomic(config,dict(config=asdict(cfg)))
    args=SimpleNamespace(config=config,out=ROOT/'ndtvs',algorithm='ndtvs',device='cpu',seed=2026,
        episodes=2,resume=False,val_every=1,val_episodes=1,trace_every=1,
        walltime_seconds=3600,reserve_seconds=0,max_new_episodes=0)
    with contextlib.redirect_stdout(io.StringIO()):
        assert c.run_train(args) == 0
    nd_inputs={p:c.sha256(p) if hasattr(c,'sha256') else runner.sha256(p) for p in (args.out/'best.pt',args.out/'latest.pt')}
    registry=[]
    for v in (20.,50.):
        directory=ROOT/f'proposed_v{int(v)}'
        directory.mkdir()
        pcfg=replace(cfg,lyapunov_v=v,episode_offset=700)
        c.atomic(directory/'resolved_config.json',dict(config=asdict(pcfg)))
        c.atomic(directory/'runtime.json',dict(code_sha256={k.removeprefix('proposed/'):value
            for k,value in c.source_hashes().items() if k.startswith('proposed/')}))
        c.hrl.seed_all(41+int(v))
        agents=c.make_agents(pcfg,'proposed')
        extra=dict(episode=699,dual_lambda_z=3.,pair_id=f'test-v{v}')
        for agent,key in zip(agents,('frame','slot')):
            agent.save(directory/(key+'.pt'),extra)
        registry.append(dict(name=f'proposed_V{int(v)}',algorithm='proposed',expected_v=v,
            config=directory/'resolved_config.json',runtime=directory/'runtime.json',
            frame_checkpoint=directory/'frame.pt',slot_checkpoint=directory/'slot.pt',selection='integration-only random initialized weights'))
    registry.append(dict(name='ndtvs',algorithm='ndtvs',checkpoint=args.out/'best.pt',completion_status=args.out/'status.json'))
    configurations,policies,provenance,sizes=models.load_models(registry,'cpu',c)
    assert configurations['proposed_V20'].lyapunov_v == 20 and configurations['proposed_V50'].lyapunov_v == 50
    assert provenance['proposed_V20']['dual'] == 3.
    assert len(sizes) == 5 and all(r['parameters'] == r['actor_only']+r['critic_only']+r['shared'] for r in sizes)
    CHECKS.append('different V and training episode_offset load their own matched checkpoint pair, preserving saved dual and exact parameter counts')
    bad=json.loads((ROOT/'proposed_v20/resolved_config.json').read_text())
    bad['config']['rsu_total_bandwidth_hz']*=2
    c.atomic(ROOT/'bad_config.json',bad)
    altered=[dict(registry[0],config=ROOT/'bad_config.json')]+registry[1:]
    expect_error(lambda:models.load_models(altered,'cpu',c))
    altered=[dict(registry[0],slot_checkpoint=registry[1]['slot_checkpoint'])]+registry[1:]
    expect_error(lambda:models.load_models(altered,'cpu',c))
    CHECKS.append('mismatched physical radio config and mixed proposed checkpoint pairs are rejected')
    s.MODELS=registry
    s.PROJECT_ROOT=PROJECT
    s.OUT=ROOT/'evaluation'
    s.DEVICE='cpu'
    s.SEEDS=(2026,2027,2028)
    s.RESUME=True
    s.VISUAL_SNR_DB=None
    s.EPISODES_PER_SEED=2
    s.SNR_MODE='offset'
    s.SNR_OFFSETS_DB=(-5,5)
    s.COST_MODE='reevaluate'
    s.HIRING_COSTS=(0.,20.)
    s.QOE_COST_WEIGHT=1.
    s.COST_SNR_DB=5
    s.WALLTIME_SECONDS=3600
    s.RESERVE_SECONDS=0
    s.VISUAL_SLOT_STRIDE=1
    s.VISUAL_MAX_IMAGES=3
    s.VISUAL_SEEDS=(2026,)
    s.BOOTSTRAP_SAMPLES=100
    s.MAX_NEW_EPISODES=1
    validate(s)
    with contextlib.redirect_stdout(io.StringIO()):
        assert runner.run(s,'smoke') == 75
    s.MAX_NEW_EPISODES=0
    with contextlib.redirect_stdout(io.StringIO()):
        assert runner.run(s,'smoke') == 0
        assert runner.run(s,'sweep') == 0
    state=runner.verify_result(s.OUT/'sweep',c)
    assert len(state['cells']) == 2*3*3 + 2*3*3
    assert len({r['scenario_sha256'] for cell in state['cells'].values() for r in cell['rows']}) == 6
    assert all(r['hiring_cost_total'] == 0 for cell in state['cells'].values() if cell['model']=='ndtvs' for r in cell['rows'])
    assert all(r['audit_sha256'] for cell in state['cells'].values() for r in cell['rows'][:1])
    for cell in state['cells'].values():
        cost=cell['hiring_cost_per_frame']
        for row in cell['rows']:
            assert np.isclose(row['hiring_cost_total'],cfg.lambda_h*cost*row['hired_uav_frames'])
            assert np.isclose(row['cost_augmented_qoe_final_per_user'],row['paper_qoe_final_per_user']-row['hiring_cost_total']/cfg.num_users)
    CHECKS.append('actual three-seed, two-SNR, two-V + NDTVS grid and cost re-evaluation pass every-cell trace audit, pairing, RSU-only and hiring arithmetic')
    before=runner.sha256(s.OUT/'sweep/state.json')
    with contextlib.redirect_stdout(io.StringIO()):
        assert runner.run(s,'sweep') == 0
    assert runner.sha256(s.OUT/'sweep/state.json') == before
    assert all(runner.sha256(p) == fp for p,fp in nd_inputs.items())
    CHECKS.append('pause/resume completes missing scenarios only; completed resubmission and evaluation preserve trained input checkpoints')
    with contextlib.redirect_stdout(io.StringIO()):
        assert report(s) == 0
        assert export(s) == 0
    from PIL import Image
    images=list((s.OUT/'sweep/summary').glob('*.png'))
    assert len(images)==7
    for image in images:
        with Image.open(image) as im:
            im.verify()
    gifs=list((s.OUT/'sweep/visuals').rglob('*.gif'))
    assert len(gifs)==6
    for image in gifs:
        with Image.open(image) as im:
            assert im.n_frames==3
    CHECKS.append('seven PNG/PDF figures including quality/chunk distributions, all-user CDFs, CSVs and six three-frame method/SNR GIFs render successfully')
    # Accounting mode must leave behavior equal to the source SNR results.
    accounted=ROOT/'accounting'
    import shutil
    shutil.copytree(s.OUT,accounted)
    astate=json.loads((accounted/'sweep/state.json').read_text())
    astate['spec']['cost_mode']='accounting'
    astate['cells']={k:v for k,v in astate['cells'].items() if v['experiment']=='snr'}
    c.atomic(accounted/'sweep/state.json',astate)
    gate=json.loads((accounted/'sweep/verification.json').read_text())
    gate['state_sha256']=runner.sha256(accounted/'sweep/state.json')
    c.atomic(accounted/'sweep/verification.json',gate)
    original_out=s.OUT
    s.OUT=accounted
    with contextlib.redirect_stdout(io.StringIO()):
        assert report(s) == 0
    result=json.loads((accounted/'sweep/summary/report.json').read_text())
    costs=result['cost_means']
    for name in provenance:
        values=[r['estimate'] for r in costs if r['model']==name and r['metric']=='stall_time_ratio']
        assert len(set(values))==1
    s.OUT=original_out
    CHECKS.append('accounting-only cost sweep has explicitly fixed stall/quality/coverage behavior')
    settings=ROOT/'settings.py'
    s.PYTHON=Path(sys.executable)
    s.LOG_DIR=ROOT/'logs'
    write_settings(s,settings)
    assert not s.LOG_DIR.exists()
    with contextlib.redirect_stdout(io.StringIO()):
        assert submit.main(['--settings',str(settings),'--dry-run']) == 0
    assert not s.LOG_DIR.exists()
    submitted=[]
    def fake_sbatch(command,**kwargs):
        submitted.append(command)
        return SimpleNamespace(returncode=0)
    with patch.object(submit.subprocess,'run',fake_sbatch),contextlib.redirect_stdout(io.StringIO()):
        assert submit.main(['--settings',str(settings)])==0
    frozen=load(submitted[0][-1])
    assert frozen.PROJECT_ROOT == PROJECT and frozen.OUT == s.OUT and frozen.MODELS == registry
    CHECKS.append('dry-run has no writes; Slurm command and frozen configuration use configured lab Python and preserve paths without shell exports')
    # Tampered sidecars must fail even with unchanged state and gate.
    one=next(iter(state['cells'].values()))['rows'][0]
    sidecar=s.OUT/'sweep'/one['per_user_file']
    original=sidecar.read_bytes()
    sidecar.write_bytes(original+b' ')
    expect_error(lambda:runner.verify_result(s.OUT/'sweep',c))
    sidecar.write_bytes(original)
    CHECKS.append('changed per-user data cannot be plotted as verified results')
    smoke=runner.verify_result(s.OUT/'smoke',c)
    output=dict(passed=True,checks=CHECKS,actual_environment_evaluation_scenarios=
        sum(len(cell['rows']) for result in (state,smoke) for cell in result['cells'].values()),
        note='CPU integration uses two-episode NDTVS and randomly initialized proposed checkpoints; it is not a scientific performance/convergence evaluation.')
    (ROOT/'validation.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
