from __future__ import annotations
import json, random, tempfile, unittest, pickle
from pathlib import Path
from dataclasses import replace
from itertools import product
import numpy as np
from config_hppo import HPPOConfig
from hppo.env import P3HierarchicalEnv, slot_training_reward, frame_training_reward
from hppo.completion import FastPolicyCompletion
from hppo.train import RandomPolicy, main
from hppo.verify_trace import verify

def small(**kw):
    args = dict(num_regions=1, users_per_region=4, num_frames=2, frame_slots=3,
                hidden_dims=(16,16), rollout_scenarios=1, ppo_minibatch_size=16)
    args.update(kw)
    return HPPOConfig(**args)

class EnvironmentTests(unittest.TestCase):
    def test_original_claude_state_reward_and_physics_fixture(self):
        fixture=json.loads(Path(__file__).with_name('claude_physics_fixture.json').read_text())
        cfg=small(num_regions=2,num_frames=1)
        e=P3HierarchicalEnv(cfg);e.reset();obs=e.prepare_frame();raw={};completed={}
        for m in e.regions:
            np.testing.assert_array_equal(obs[m],fixture['frame_observations'][str(m)])
            a=np.zeros(cfg.num_users,dtype=np.int64);users=e.region_users[m]
            a[list(users[:2])]=1;a[users[2]]=2
            raw[m]=a;completed[m]=e.proposal(m,a).execute(1,2)
        e.begin_frame(raw,completed)
        for row in fixture['slots']:
            for m in e.regions:
                np.testing.assert_array_equal(e.get_slot_obs(m),row['observations'][str(m)])
            step=e.step_slot({int(m):np.asarray(a) for m,a in row['actions'].items()})
            for actual,key in [(e.state.queue,'Q'),(e.state.battery_j,'battery'),(e.state.user_x,'positions')]:
                np.testing.assert_array_equal(actual,row[key])
            for m in e.regions:
                self.assertEqual(slot_training_reward(cfg,step.metrics[m],1)[1],row['reward'][str(m)])
        for m in e.regions:
            self.assertEqual(frame_training_reward(cfg,step.info['frame_summary']['regions'][m],1)[1],fixture['frame_reward'][str(m)])
        self.assertEqual(e.episode_summary(),fixture['episode_summary'])

    def test_invalid_proposals_fail_instead_of_projection(self):
        e=P3HierarchicalEnv(small());e.reset();e.prepare_frame()
        for a in ([1,1,1,1],[2,2,2,0],[0,0,3,0],[0,0,-1,0],[0,.5,0,0],[0,float('nan'),0,0]):
            with self.assertRaises(ValueError):e.proposal(0,np.asarray(a))

    def test_unhired_candidates_are_not_reassigned(self):
        cfg=small(hiring_cost_per_frame=1e9);e=P3HierarchicalEnv(cfg);e.reset();e.prepare_frame()
        raw=np.asarray([1,0,2,0]);action,detail=FastPolicyCompletion(cfg).select(e,0,raw,RandomPolicy(10,cfg))
        self.assertEqual(action.hired,0)
        info=e.begin_frame({0:raw},{0:action},{0:detail})['regions'][0]
        self.assertEqual(info['proposal_uav_candidates'],[2]);self.assertEqual(info['executed_uav_users'],[])
        self.assertEqual(info['executed_rsu_users'],[0]);self.assertEqual(e.provider[2],0)

    def test_low_battery_and_empty_region(self):
        cfg=small(num_regions=2,initial_battery_j=0.20*548*3600+1000)
        e=P3HierarchicalEnv(cfg);e.reset();e.state.user_x[:]=500;e.prepare_frame()
        self.assertFalse(e.feasible_hover_points(0))
        self.assertTrue(all(np.array_equal(m,[True,False,False]) for m in e.frame_action_masks(0)))
        fp=RandomPolicy(1,cfg,True);sp=RandomPolicy(2,cfg)
        raw={m:fp.act(e.get_frame_obs(m),e.frame_action_masks(m))[0] for m in e.regions}
        actions,detail=FastPolicyCompletion(cfg).select_all(e,raw,sp)
        self.assertTrue(all(a.hired==0 for a in actions.values()))
        self.assertTrue(all(len(d['candidates'])==1 for d in detail.values()))
        before=e.state.battery_j.copy();e.begin_frame(raw,actions,detail)
        for _ in range(cfg.frame_slots):
            e.step_slot({m:sp.act(e.get_slot_obs(m),e.slot_action_masks(m))[0] for m in e.regions})
        self.assertTrue(np.all(e.state.battery_j>before))

    def test_local_rollout_matches_full_environment(self):
        from env.p3.environment import generate_frame_trace
        cfg=small(num_regions=2);full=P3HierarchicalEnv(cfg);full.reset();full.prepare_frame()
        trace=generate_frame_trace(cfg,987654321);trial=full.fork_for_rollout(0,trace);full.trace=trace
        raw={m:np.zeros(cfg.num_users,dtype=np.int64) for m in full.regions};raw[0][0]=1;raw[0][1]=2
        actions={m:full.proposal(m,raw[m]).execute(1,2) for m in full.regions}
        full.begin_frame(raw,actions);trial.begin_frame({0:raw[0]},{0:actions[0]})
        for _ in range(cfg.frame_slots):
            a={m:np.zeros(len(cfg.slot_action_nvec),dtype=np.int64) for m in full.regions}
            a[0][0]=a[0][1]=1;a[0][2*cfg.num_users+1]=3
            x=full.step_slot(a);y=trial.step_slot({0:a[0]});self.assertEqual(x.metrics[0],y.metrics[0])
        self.assertEqual(x.info['frame_summary']['regions'][0],y.info['frame_summary']['regions'][0])

try:
    import torch
    from hppo.ppo import PPOAgent
    TORCH=True
except ImportError:
    TORCH=False

@unittest.skipUnless(TORCH,'PyTorch required for learning verification')
class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(2026);np.random.seed(2026)

    def test_joint_probability_normalizes_and_likelihood_replays(self):
        cfg=small();e=P3HierarchicalEnv(cfg);e.reset();obs=e.prepare_frame()[0];masks=e.frame_action_masks(0)
        f=PPOAgent(cfg.frame_obs_dim,cfg.frame_action_nvec,cfg,'frame_ppo')
        candidates=[a for a in product(range(3),repeat=4) if a.count(1)<=3 and a.count(2)<=2]
        with torch.no_grad():
            lp,_,_=f.net.evaluate_actions(torch.tensor(np.stack([obs]*len(candidates))),torch.tensor(candidates),f._mask_tensors(masks,len(candidates)))
        self.assertAlmostEqual(float(lp.exp().sum()),1.0,places=5)
        for _ in range(25):
            a,lp,v,_=f.act(obs,masks);e.proposal(0,a)
            with torch.no_grad():
                replay,_,_=f.net.evaluate_actions(torch.tensor(obs[None]),torch.tensor(a[None]),f._mask_tensors(masks))
            self.assertAlmostEqual(lp,float(replay[0]),places=6)
            f.store(0,obs,masks,a,lp,v,.1,0,True)
        self.assertTrue(all(np.isfinite(v) for v in f.update().values()))

    def test_fast_vectorization_matches_independent_heads(self):
        cfg=small();f=PPOAgent(cfg.slot_obs_dim,cfg.slot_action_nvec,cfg,'slot_ppo')
        obs=torch.randn(4,cfg.slot_obs_dim);masks=[torch.ones(4,n,dtype=torch.bool) for n in cfg.slot_action_nvec];masks[0][:,1:]=False
        with torch.no_grad():
            old,value=f.net.distributions(obs,masks);joint,value2=f.net._joint(obs,masks)
            for i,d in enumerate(old):
                torch.testing.assert_close(d.probs,joint.probs[:,i,:cfg.slot_action_nvec[i]],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(value,value2)

    def test_completion_isolation_and_real_future_is_unused(self):
        cfg=small(rollout_scenarios=2);e=P3HierarchicalEnv(cfg);e.reset();e.prepare_frame()
        fast=PPOAgent(cfg.slot_obs_dim,cfg.slot_action_nvec,cfg,'slot_ppo');raw=np.array([1,2,0,0]);comp=FastPolicyCompletion(cfg)
        before=pickle.dumps(e.__dict__);np_rng=pickle.dumps(np.random.get_state());py_rng=random.getstate()
        torch_rng=torch.get_rng_state().clone();weights={k:v.clone() for k,v in fast.net.state_dict().items()}
        action,detail=comp.select(e,0,raw,fast)
        self.assertEqual(before,pickle.dumps(e.__dict__));self.assertEqual(np_rng,pickle.dumps(np.random.get_state()))
        self.assertEqual(py_rng,random.getstate());torch.testing.assert_close(torch_rng,torch.get_rng_state())
        self.assertEqual(fast.buffer_size(),0)
        for k,v in weights.items():torch.testing.assert_close(v,fast.net.state_dict()[k])
        e.trace.rsu_fading[:]=999;e.trace.uav_fading[:]=999
        action2,detail2=comp.select(e,0,raw,fast)
        self.assertEqual(action,action2);self.assertEqual(detail['candidates'],detail2['candidates'])
        self.assertEqual(detail['selected_index'],int(np.argmin([c['mean_dpp'] for c in detail['candidates']])))

    def test_train_save_load_eval_and_incomplete_trace_rejection(self):
        import contextlib,io
        common=['--num-regions','1','--num-frames','1','--frame-slots','2','--hidden-dims','16,16',
                '--rollout-scenarios','1','--ppo-minibatch-size','16','--frame-update-every-episodes','2']
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            main(['--mode','train','--train-episodes','3','--output-dir',tmp,'--run-name','train']+common)
            root=Path(tmp)/'train';cp=root/'checkpoints';payload=torch.load(cp/'frame_latest.pt',weights_only=False)
            self.assertEqual(payload['update_count'],2)
            main(['--mode','eval','--eval-episodes','1','--episode-offset','1000000',
                  '--frame-checkpoint',str(cp/'frame_latest.pt'),'--slot-checkpoint',str(cp/'slot_latest.pt'),
                  '--output-dir',tmp,'--run-name','eval']+common)
            self.assertFalse(verify(root)[1]);self.assertFalse(verify(Path(tmp)/'eval')[1])
            trace=root/'trace.jsonl';original=trace.read_text();lines=original.splitlines()
            for text in ('','\n'.join(lines[:2])+'\n','\n'.join(lines[:-1])+'\n'):
                trace.write_text(text);self.assertTrue(verify(root)[1])
            records=[json.loads(line) for line in lines];frame=next(r for r in records if r['event']=='frame_start')
            frame['regions']['0']['completion']['selected_index']=100
            trace.write_text('\n'.join(json.dumps(r) for r in records)+'\n');self.assertTrue(verify(root)[1]);trace.write_text(original)
            cfg=small(num_frames=1,frame_slots=2)
            agent=PPOAgent(cfg.frame_obs_dim,cfg.frame_action_nvec,cfg,'frame_ppo');agent.load(cp/'frame_latest.pt')
            e=P3HierarchicalEnv(cfg);e.reset();o=e.prepare_frame()[0];m=e.frame_action_masks(0);a=agent.act(o,m,True)[0]
            agent.save(Path(tmp)/'roundtrip.pt')
            again=PPOAgent(cfg.frame_obs_dim,cfg.frame_action_nvec,cfg,'frame_ppo');again.load(Path(tmp)/'roundtrip.pt')
            np.testing.assert_array_equal(a,again.act(o,m,True)[0])
            wrong=replace(cfg,hiring_cost_per_frame=cfg.hiring_cost_per_frame+1)
            with self.assertRaisesRegex(RuntimeError,'config mismatch'):
                PPOAgent(wrong.frame_obs_dim,wrong.frame_action_nvec,wrong,'frame_ppo').load(cp/'frame_latest.pt')

if __name__=='__main__':unittest.main()
