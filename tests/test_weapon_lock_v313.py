"""v3.13 single-variable combat contract and real CUDA subprocess continuation."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
import env.mavuav as module
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv as Env, RED_IDS, BLUE_IDS, ENTITY_IDS, OBS_DIM, GLOBAL_STATE_DIM
from env.vector_env import _environment_state, _restore_environment_state
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.train_happo import _evaluation_row
ROOT=Path(__file__).resolve().parents[1]


def cfg(version=313): return module.load_environment_config(ROOT/f'configs/env_v{version}.yaml')


def scene(version=313):
    e=Env(cfg(version),randomize=False);e.reset(seed=1)
    for a in ENTITY_IDS:
        s=e.entities[a].state;s.alive=a in ('MAV','UAV1','Blue1','Blue2','Blue4')
        s.x=20000.;s.y=0.;s.h=6000.;s.psi=0.;s.theta=0.;s.v=275.
    for a,x in [('MAV',-15000.),('UAV1',0.),('Blue1',2000.),('Blue2',2500.)]: e.entities[a].state.x=x
    return e


def assert_single(e):
    assert e.weapon_lock_target['MAV'] is None
    for a in ENTITY_IDS:
        positive=[b for (attacker,b),v in e._attack_streak.items() if attacker==a and v>0]
        assert len(positive)<=1
        if positive: assert positive==[e.weapon_lock_target[a]]


def test_legacy_multi_candidate_and_new_zero_preaccumulation():
    old=scene(312);old._attack_streak['UAV1','Blue1']=old._attack_streak['UAV1','Blue2']=2
    events,_=old._resolve_attacks()
    assert [v for v in events if v['attacker']=='UAV1']==[{'attacker':'UAV1','target':b} for b in ('Blue1','Blue2')]
    new=scene();new._attack_streak['UAV1','Blue1']=new._attack_streak['UAV1','Blue2']=2
    events,_=new._resolve_attacks()
    assert not events and new._attack_streak['UAV1','Blue1']==1
    assert new._attack_streak['UAV1','Blue2']==0
    assert_single(new)


@pytest.mark.parametrize('a,b',[
    ((.1,.8,2800.),(.2,.1,1200.)),
    ((.1,.1,2800.),(.1,.2,1200.)),
    ((.1,.1,1200.),(.1,.1,2800.)),
    ((.1,.1,2000.),(.1,.1,2000.)),
])
def test_acquisition_priority_ATA_AA_distance_canonical_ID(a,b,monkeypatch):
    e=scene();s=e.entities['UAV1'].state
    fake={id(e.entities[k].state):SimpleNamespace(ata=v[0],aa=v[1],distance=v[2]) for k,v in [('Blue1',a),('Blue2',b)]}
    real=module.compute_pairwise_geometry
    monkeypatch.setattr(module,'compute_pairwise_geometry',lambda x,y:fake[id(y)] if x is s and id(y) in fake else real(x,y))
    e.entities=dict(reversed(list(e.entities.items())))
    assert e._acquire_weapon_lock('UAV1',BLUE_IDS)=='Blue1'


def test_hold_three_and_kill_cleanup_no_same_boundary_reacquisition():
    e=scene()
    for t in (1,2):
        events,_=e._resolve_attacks()
        assert not events and e._attack_streak['UAV1','Blue1']==t
        assert_single(e)
    events,_=e._resolve_attacks()
    assert events==[dict(attacker='UAV1',target='Blue1')]
    assert e.entities['Blue2'].state.alive and e.weapon_lock_target['UAV1'] is None
    assert all(v==0 for (a,b),v in e._attack_streak.items() if a=='UAV1')
    e._resolve_attacks()
    assert e.weapon_lock_target['UAV1']=='Blue2' and e._attack_streak['UAV1','Blue2']==1


def test_persistence_not_reselection_of_better_target():
    e=scene();e._resolve_attacks()
    e.entities['Blue2'].state.x=1500.
    e._resolve_attacks()
    assert e.weapon_lock_target['UAV1']=='Blue1' and e._attack_streak['UAV1','Blue1']==2


def test_gate_loss_switch_same_boundary_from_one_and_no_candidates():
    e=scene();e._resolve_attacks();e.entities['Blue1'].state.y=4000.
    e._attack_streak['UAV1','Blue2']=2
    events,_=e._resolve_attacks()
    assert not events and e.weapon_lock_target['UAV1']=='Blue2'
    assert e._attack_streak['UAV1','Blue1']==0 and e._attack_streak['UAV1','Blue2']==1
    e.entities['Blue2'].state.y=4000.
    e._resolve_attacks()
    assert e.weapon_lock_target['UAV1'] is None
    assert all(v==0 for (a,b),v in e._attack_streak.items() if a=='UAV1')


@pytest.mark.parametrize('shared',[False,True])
def test_synchronous_attackers_and_shared_victim(shared):
    e=scene();e.entities['UAV2'].state.alive=True
    e.entities['UAV2'].state.x=0.;e.entities['UAV2'].state.y=0.
    for a,b in [('UAV1','Blue1'),('UAV2','Blue1' if shared else 'Blue2')]:
        e.weapon_lock_target[a]=b;e._attack_streak[a,b]=2
    events,deaths=e._resolve_attacks()
    assert len(events)==2 and len(deaths)==(1 if shared else 2)
    assert len(e._red_attack_kills)==len(deaths)
    assert all(e.weapon_lock_target[a] is None for a in ('UAV1','UAV2'))
    assert_single(e)


def test_target_killed_by_another_attacker_clears_lock_without_reacquire():
    e=scene();e.entities['UAV2'].state.alive=True;e.entities['UAV2'].state.x=0.
    e.weapon_lock_target['UAV1']=e.weapon_lock_target['UAV2']='Blue1'
    e._attack_streak['UAV1','Blue1']=2;e._attack_streak['UAV2','Blue1']=0
    e._resolve_attacks()
    assert e.weapon_lock_target['UAV2'] is None and e._attack_streak['UAV2','Blue1']==0
    assert e._attack_streak['UAV2','Blue2']==0


def test_blue_symmetry_mav_unarmed_and_navigation_reward_targets_not_used():
    e=scene();e.entities['UAV2'].state.alive=True;e.entities['UAV2'].state.x=4500.
    e.entities['MAV'].state.x=4000.
    e._reward_target_previous['UAV1']='Blue2'
    for _ in range(3): events,_=e._resolve_attacks()
    assert not any(v['attacker']=='MAV' for v in events)
    assert sum(v['attacker']=='Blue1' for v in events)==1
    assert e.weapon_lock_target['MAV'] is None
    assert_single(e)


def test_mutual_kill_is_synchronous(monkeypatch):
    e=scene();e.weapon_lock_target['UAV1']='Blue1';e.weapon_lock_target['Blue1']='UAV1'
    e._attack_streak['UAV1','Blue1']=e._attack_streak['Blue1','UAV1']=2
    monkeypatch.setattr(e,'_weapon_gate_geometry',lambda a,b:SimpleNamespace(ata=0.,aa=0.,distance=2000.) if (a,b) in [('UAV1','Blue1'),('Blue1','UAV1')] else None)
    events,deaths=e._resolve_attacks()
    assert len(events)==2 and set(deaths)=={'UAV1','Blue1'}
    assert e.weapon_lock_target['UAV1'] is e.weapon_lock_target['Blue1'] is None


def test_boundary_death_reset_and_state_roundtrip():
    e=scene();e._resolve_attacks();state=_environment_state(e)
    other=scene();_restore_environment_state(other,state)
    assert other.weapon_lock_target==e.weapon_lock_target and other._attack_streak==e._attack_streak
    assert other._resolve_attacks()==e._resolve_attacks()
    e.entities['UAV1'].state.alive=False;e._resolve_attacks()
    assert e.weapon_lock_target['UAV1'] is None
    e.reset(seed=2);assert all(v is None for v in e.weapon_lock_target.values())
    missing=deepcopy(state);missing.pop('weapon_lock_target')
    with pytest.raises(ValueError,match='weapon_lock_target'):_restore_environment_state(other,missing)
    state['attack_streak']['UAV1','Blue2']=1
    with pytest.raises(ValueError,match='streak'):_restore_environment_state(other,state)


def test_config_exact_diff_initial_obs_state_and_dimensions():
    old,new=cfg(312),cfg()
    new['environment_version']=old['environment_version']
    assert new['combat'].pop('weapon_engagement_mode')=='single_target_lock'
    assert new==old
    assert yaml.safe_load((ROOT/'configs/happo_v312_baseline.yaml').read_text())==yaml.safe_load((ROOT/'configs/happo_v313_baseline.yaml').read_text())
    a=Env(cfg(312),randomize=False);b=Env(cfg(),randomize=False)
    ao,_=a.reset(seed=1);bo,info=b.reset(seed=1)
    assert OBS_DIM==100 and GLOBAL_STATE_DIM==117
    for aid in RED_IDS:np.testing.assert_array_equal(ao[aid],bo[aid])
    np.testing.assert_array_equal(a.global_state(),b.global_state())
    assert info['weapon_engagement_mode']=='single_target_lock'


def test_single_target_invariant_on_real_physics_steps():
    e=Env(cfg(),profile='main');e.reset(seed=1000)
    rng=np.random.default_rng(2000)
    for _ in range(30):
        _,_,term,trunc,info=e.step(rng.uniform(-1,1,(4,3)))
        assert_single(e)
        for a in ENTITY_IDS:
            assert sum(v['attacker']==a for v in info['attack_events'])<=1
        assert info['weapon_lock_target']==e.weapon_lock_target
        if term or trunc:e.reset(seed=1001)


@pytest.mark.parametrize('field,value,inside',[
    ('distance',1000.,True),('distance',3000.,True),('distance',999.,False),
    ('distance',3001.,False),('ata',np.deg2rad(30.),False),('aa',np.deg2rad(90.),False),
])
def test_unchanged_gate_inclusive_range_strict_angles(field,value,inside,monkeypatch):
    e=scene();g=SimpleNamespace(distance=2000.,ata=0.,aa=0.);setattr(g,field,value)
    monkeypatch.setattr(module,'compute_pairwise_geometry',lambda a,b:g)
    assert (e._weapon_gate_geometry('UAV1','Blue1') is not None)==inside


@pytest.mark.parametrize('bad',[{'actor_variant':'tam','critic_variant':'tam_attention'},
    {'method_variant':'rgaa'},{'actor_variant':'pcta'},{'actor_variant':'recurrent'},
    {'critic_variant':'relational'},{'method_variant':'agp'},
    {'actor_variant':'entity_recurrent','critic_variant':'entity_attention_recurrent'}])
def test_no_historical_algorithm_opened(bad):
    with pytest.raises(ValueError):HAPPOTrainer(cfg(),dict(device='cuda',num_envs=1,**bad))


def test_reward_regression_real_steps_without_parallel_competition():
    old,new=scene(312),scene()
    for e in (old,new):e.entities['Blue2'].state.x=18000.
    for _ in range(3):
        x=old.step(np.zeros((4,3)));y=new.step(np.zeros((4,3)))
        assert x[1:4]==y[1:4]
        for key in ('event_reward','terminal_reward','team_reward','mav_process_reward','uav1_process_reward','uav2_process_reward','uav3_process_reward'):
            assert x[-1][key]==y[-1][key]
        for aid in RED_IDS:np.testing.assert_array_equal(x[0][aid],y[0][aid])
    assert old._red_attack_kills==new._red_attack_kills=={'Blue1'}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required; no CPU smoke fallback')
def test_cuda_subprocess_checkpoint_exact_continuation_and_evaluator(tmp_path,monkeypatch):
    training=yaml.safe_load((ROOT/'configs/happo_v313_baseline.yaml').read_text())['training']
    training.update(num_envs=2,rollout_steps=1,hidden_dim=8,ppo_epochs=1,minibatch_size=2,randomization_curriculum_enabled=False)
    a=HAPPOTrainer(cfg(),training);b=None
    try:
        assert a.device.type=='cuda' and a.vector_env.parallel
        assert len(set(a.vector_env.worker_pids))==2 and os.getpid() not in a.vector_env.worker_pids
        e=scene();e.entities['Blue2'].state.x=18000.;e._resolve_attacks()
        a.vector_env.set_env_states([_environment_state(e)]*2,a.vector_env.reset_counts,a.vector_env.base_seed)
        a.observations=np.stack([np.stack([e._observations()[aid] for aid in RED_IDS])]*2)
        a.global_states=np.stack([e.global_state()]*2);a.active_masks=np.stack([e.active_masks]*2)
        a.collect_rollout();metrics=a.update()
        assert all(np.isfinite(v) for k,v in metrics.items() if isinstance(v,(float,int)))
        saved=a.vector_env.get_env_states()
        assert all(s['weapon_lock_target']['UAV1']=='Blue1' for s in saved)
        assert all(s['attack_streak']['UAV1','Blue1']==2 for s in saved)
        checkpoint=tmp_path/'checkpoint_final.pt';a.save_checkpoint(checkpoint)
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint_contract(data,cfg());assert data['sampled_steps']==2
        bad=deepcopy(data);bad['weapon_engagement_mode']='all_pair'
        with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,cfg())
        b=HAPPOTrainer(cfg(),training);assert b.load_checkpoint(checkpoint)==2
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
        ra=a.collect_rollout();ma=a.update()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
        rb=b.collect_rollout();mb=b.update()
        assert ra==rb and ma==mb
        for field in ('actions','values','returns','advantages','rewards'):np.testing.assert_array_equal(getattr(a.buffer,field),getattr(b.buffer,field))
        for field in ('observations','global_states','active_masks'):np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
        for aa,bb in [(a.actors,b.actors),(a.critic,b.critic)]:
            for key,v in aa.state_dict().items():assert torch.equal(v,bb.state_dict()[key])
        for x,y in zip(a.vector_env.get_env_states(),b.vector_env.get_env_states()):
            assert x['weapon_lock_target']==y['weapon_lock_target'] and x['attack_streak']==y['attack_streak']
            for aid in ENTITY_IDS:assert vars(x['entities'][aid].state)==vars(y['entities'][aid].state)
        row=_evaluation_row(b,1,'main',1,'cuda','stochastic',2000)
        assert row['weapon_engagement_mode']=='single_target_lock' and row['blue_target_strategy']=='coordinated_assignment'
        from algorithm import evaluate_happo
        import sys
        monkeypatch.setattr(sys,'argv',['evaluate_happo',str(checkpoint),'--episodes','1','--device','cuda','--action-mode','stochastic'])
        evaluate_happo.main()
        result=json.loads((tmp_path/'evaluation_final_stochastic_summary.json').read_text(encoding='utf-8'))
        assert result['weapon_engagement_mode']=='single_target_lock' and result['environment_version'].endswith('v3_13')
        assert result['blue_target_strategy']=='coordinated_assignment'
        # Weights-only and exact resume both reject mismatched v3.13 metadata.
        badpath=tmp_path/'bad.pt';torch.save(bad,badpath)
        with pytest.raises(RuntimeError):b.load(badpath)
        with pytest.raises(RuntimeError):b.load_checkpoint(badpath)
    finally:
        a.close()
        if b is not None:b.close()
