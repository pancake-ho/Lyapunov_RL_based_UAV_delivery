"""Guards for known layout-only checkpoint/evaluation compatibility."""
import copy
import json
import unittest
from dataclasses import asdict, replace
from unittest.mock import patch
import baseline.NDTVS.api as n
import baseline.NDTVS.common.checkpoint as cp


class ModularCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.current = n.source_hashes()
        self.registry = json.loads((n.HERE / cp.COMPAT_FILE).read_text())
        self.previous = {k:v for k,v in self.current.items() if not k.startswith('baseline/NDTVS/')}
        self.previous.update(self.registry['previous_v2_sources'])
        self.spec = {'algorithm':'ndtvs','version':n.VERSION,'qoe_definition':n.reward_spec(),
                     'source_sha256':self.previous,'config':asdict(replace(n.HPPOConfig(),device='cpu'))}

    def test_known_previous_v2_checkpoint_is_accepted(self):
        self.assertEqual(n.verify_checkpoint_source({'spec':self.spec},'ndtvs'),
                         'verified-v2-module-refactor')
        self.spec['algorithm']='hppo_rsu'
        self.assertEqual(n.verify_checkpoint_source({'spec':self.spec},'hppo_rsu'),
                         'verified-v2-module-refactor')

    def test_unknown_previous_source_and_changed_physics_are_rejected(self):
        for key in ('baseline/NDTVS/ndtvs_common.py','proposed/hppo/env.py'):
            spec=copy.deepcopy(self.spec)
            spec['source_sha256'][key]='not-the-reviewed-source'
            with self.assertRaises(ValueError):n.verify_checkpoint_source({'spec':spec},'ndtvs')

    def test_later_modular_implementation_edits_are_not_whitelisted(self):
        current=dict(self.current)
        current['baseline/NDTVS/models/policy.py']='modified-after-refactor'
        with patch.object(cp,'source_hashes',return_value=current):
            with self.assertRaisesRegex(ValueError,'verified refactor'):
                n.verify_checkpoint_source({'spec':self.spec},'ndtvs')

    def test_resume_normalizes_source_only_without_mutating_checkpoint(self):
        saved={'spec':copy.deepcopy(self.spec)}
        original=copy.deepcopy(saved)
        result=cp.resume_spec(saved,'ndtvs')
        self.assertEqual(saved,original)
        self.assertEqual(result['source_sha256'],self.current)
        result.pop('source_sha256')
        self.assertEqual(result,{k:v for k,v in self.spec.items() if k!='source_sha256'})

    def test_single_eval_checkpoint_identity_is_not_changed_by_migration(self):
        old={'source_sha256':self.previous,'provenance':{'sha256':'old-model','source_verification':'exact-v2'},
             'scenario_seed':2026}
        new={'source_sha256':self.current,'provenance':{'sha256':'old-model','source_verification':'verified-v2-module-refactor'},
             'scenario_seed':2026}
        self.assertEqual(cp.evaluation_resume_spec(old,new),new)
        changed=copy.deepcopy(new)
        changed['provenance']['sha256']='different-model'
        self.assertNotEqual(cp.evaluation_resume_spec(old,changed),changed)

    def test_paired_resume_preserves_checkpoint_and_scenario_guard(self):
        old={'selection':{'source_sha256':self.previous,'checkpoints':{
             'ndtvs':{'files':{'best.pt':'same-model'},'source_verification':'exact-v2'}}},
             'evaluators':self.registry['previous_evaluators'],'scenario_ids':[7000000]}
        new={'selection':{'source_sha256':self.current,'checkpoints':{
             'ndtvs':{'files':{'best.pt':'same-model'},'source_verification':'verified-v2-module-refactor'}}},
             'evaluators':self.registry['modular_evaluators'],'scenario_ids':[7000000]}
        self.assertEqual(cp.paired_resume_header(old,new),new)
        changed=copy.deepcopy(new)
        changed['scenario_ids']=[7000001]
        self.assertNotEqual(cp.paired_resume_header(old,changed),changed)
        changed=copy.deepcopy(new)
        changed['selection']['checkpoints']['ndtvs']['files']['best.pt']='different-model'
        self.assertNotEqual(cp.paired_resume_header(old,changed),changed)


    def test_previous_flat_checkpoint_keeps_same_reward_and_learning_spec(self):
        saved={'spec':copy.deepcopy(self.spec)}
        saved['spec']['source_sha256']={k:v for k,v in self.current.items()
                                      if not k.startswith('baseline/NDTVS/')}
        saved['spec']['source_sha256'].update(self.registry['previous_flat_sources'])
        self.assertEqual(n.verify_checkpoint_source(saved,'ndtvs'),'verified-v2-module-refactor')
        result=cp.resume_spec(saved,'ndtvs')
        self.assertEqual(result['source_sha256'],self.current)
        self.assertEqual(result['qoe_definition'],saved['spec']['qoe_definition'])
        bad=copy.deepcopy(saved)
        bad['spec']['source_sha256']['baseline/NDTVS/ndtvs_model.py']='unreviewed'
        with self.assertRaises(ValueError):n.verify_checkpoint_source(bad,'ndtvs')

    def test_previous_flat_paired_evaluator_resume_keeps_scenario_identity(self):
        source={k:v for k,v in self.current.items() if not k.startswith('baseline/NDTVS/')}
        source.update(self.registry['previous_flat_sources'])
        old={'selection':{'source_sha256':source,'checkpoints':{
             'ndtvs':{'files':{'best.pt':'same-model'},'source_verification':'exact-v2'}}},
             'evaluators':self.registry['previous_flat_evaluators'],'scenario_ids':[7000000]}
        new={'selection':{'source_sha256':self.current,'checkpoints':{
             'ndtvs':{'files':{'best.pt':'same-model'},'source_verification':'verified-v2-module-refactor'}}},
             'evaluators':self.registry['modular_evaluators'],'scenario_ids':[7000000]}
        self.assertEqual(cp.paired_resume_header(old,new),new)
        changed=copy.deepcopy(new)
        changed['scenario_ids']=[7000001]
        self.assertNotEqual(cp.paired_resume_header(old,changed),changed)

    def test_legacy_entry_point_and_package_share_policy_class_identity(self):
        import importlib.util
        from baseline.NDTVS.models.policy import NDTVSNet
        spec=importlib.util.spec_from_file_location('compat_entry_point',n.HERE/'ndtvs_common.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIs(module.NDTVSNet,NDTVSNet)
        self.assertIs(module.episode,n.episode)


if __name__ == '__main__':
    unittest.main(verbosity=2)
