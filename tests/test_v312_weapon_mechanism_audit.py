"""Audit-only checks: exact timing, interventions and frozen CUDA replay."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tools import audit_v312_weapon_mechanisms as audit


def cfg():
    return audit.prior.load_environment_config(audit.ROOT/'configs/env_v312.yaml')


def synthetic_env():
    env=audit.base.Env(cfg(),profile='main');env.reset(seed=1000)
    for a in audit.ENTITY_IDS:
        env.entities[a].state.alive=a in ('MAV','UAV1','Blue1','Blue2')
    for a,x,y in [('UAV1',0.,0.),('Blue1',2000.,0.),('Blue2',2500.,0.),('MAV',-15000.,0.)]:
        s=env.entities[a].state;s.x=x;s.y=y;s.h=6000.;s.psi=0.;s.theta=0.;s.v=275.
    return env


def test_boundary_hold_ignores_substep_excursion():
    result=audit.hold_experiment(cfg())
    assert result['first_gate_seconds']==1.
    assert result['kill_seconds']==3.
    assert result['sample_span_seconds']==2.
    assert result['transient_substep_exit_did_not_reset']


def test_w0_simultaneous_candidates_and_other_survivor_preload():
    env=synthetic_env()
    env._attack_streak['UAV1','Blue1']=2
    env._attack_streak['UAV1','Blue2']=1
    events,_=env._resolve_attacks()
    assert dict(attacker='UAV1',target='Blue1') in events
    assert env._attack_streak['UAV1','Blue2']==2


def test_reset_only_clears_killer_other_streak_not_simultaneous_batch():
    env=synthetic_env();audit.install_intervention(env,'R1')
    env._attack_streak['UAV1','Blue1']=2;env._attack_streak['UAV1','Blue2']=1
    ev,_=env._resolve_attacks()
    assert len(ev)==1 and env._attack_streak['UAV1','Blue2']==0
    env=synthetic_env();audit.install_intervention(env,'R1')
    env._attack_streak['UAV1','Blue1']=env._attack_streak['UAV1','Blue2']=2
    ev,_=env._resolve_attacks();assert len(ev)==2


@pytest.mark.parametrize('mode',['L1','L2'])
def test_single_lock_only_one_pair_and_switch_resets(mode):
    env=synthetic_env();env._reward_target_previous['UAV1']='Blue1'
    original=audit.base.Env._resolve_attacks
    audit.install_intervention(env,mode)
    env._resolve_attacks()
    assert env._attack_streak['UAV1','Blue1']==1
    assert env._attack_streak['UAV1','Blue2']==0
    env.entities['Blue1'].state.alive=False
    env._reward_target_previous['UAV1']='Blue2'
    env._attack_streak['UAV1','Blue2']=2
    ev,_=env._resolve_attacks()
    assert env._attack_streak['UAV1','Blue2']==1 and not ev
    assert audit.base.Env._resolve_attacks is original


def test_l1_no_reward_target_no_fallback():
    env=synthetic_env();audit.install_intervention(env,'L1')
    ev,_=env._resolve_attacks()
    assert not ev and all(env._attack_streak['UAV1',b]==0 for b in audit.BLUE)


def test_carry_uses_next_actual_victim_not_unrelated_preload():
    ep=dict(events=[dict(attacker='UAV1',target=b,step=s) for b,s in [('Blue1',3),('Blue2',6),('Blue3',9)]],
        pairs=[dict(agent='UAV1',target='Blue4',step=3,streak_evaluated=2),
               dict(agent='UAV1',target='Blue2',step=3,streak_evaluated=0,streak_before=0),
               dict(agent='UAV1',target='Blue3',step=6,streak_evaluated=0,streak_before=0)])
    assert audit.classify_carry(ep,'UAV1')['classification']=='TYPE-SERIAL'
    ep['pairs'][1]['streak_evaluated']=1
    c=audit.classify_carry(ep,'UAV1')
    assert c['classification']=='TYPE-MIXED' and c['preaccumulated_intervals']==1


def test_simultaneous_and_serial_classification_not_conflated():
    ep=dict(events=[dict(attacker='UAV1',target=b,step=3) for b in audit.BLUE],pairs=[])
    assert audit.classify_carry(ep,'UAV1')['classification']=='TYPE-SIMULTANEOUS'
    ep['events'][-1]['step']=6
    assert audit.classify_carry(ep,'UAV1')['classification']=='TYPE-MIXED'


def test_missing_intervals_not_zero_and_exact_distributions():
    assert audit.distribution([])['minimum'] is None
    d=audit.distribution([0,1,2,3,5])
    assert d['p_eq_0']==.2 and d['p_le_2']==.6


def test_output_refuses_existing_and_outside_directory(tmp_path):
    with pytest.raises(ValueError):audit.exclusive_output(tmp_path/'audit')
    with pytest.raises(FileExistsError):audit.exclusive_output(audit.PRIOR)


@pytest.mark.parametrize('mode',['W0','L1','L2','R1'])
def test_cuda_frozen_replay_invariants(mode):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    path=audit.ROOT/'outputs/happo_v312_cap_seed1_2m/checkpoint_final.pt'
    if not path.exists():pytest.skip('local frozen checkpoint required')
    digest=audit.base.sha(path)
    data=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        actors=audit.base.IndependentActors(hidden_dim=data['trainer_config']['hidden_dim']).cuda().eval()
        actors.load_state_dict(data['actors'])
        params={k:v.clone() for k,v in actors.state_dict().items()}
        rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
        cls=audit.base.Env;resolver=cls._resolve_attacks
        ep=audit.diagnostic_replay(actors,data['environment_config'],1000,2000,mode)
        assert all(np.isfinite(float(ep['result'][k])) for k in ('episode_return','episode_length','red_attack_kills'))
        assert cls._resolve_attacks is resolver and audit.base.Env is cls
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert all(torch.equal(v,actors.state_dict()[k]) for k,v in params.items())
        if mode=='W0':
            official=audit.official_readonly(actors,data['environment_config'],1000,2000)
            assert ep['result']==official
            assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    assert audit.base.sha(path)==digest


def test_l2_retains_visible_lock_when_it_leaves_attack_range():
    env=synthetic_env();audit.install_intervention(env,'L2')
    env._resolve_attacks()
    env.entities['Blue1'].state.x=4000.
    env._resolve_attacks()
    assert env._attack_streak['UAV1','Blue1']==0
    assert env._attack_streak['UAV1','Blue2']==0


@pytest.mark.parametrize('mode',['L1','L2','R1'])
def test_interventions_preserve_blue_attack_on_unarmed_mav(mode):
    env=synthetic_env()
    # Blue heading +x, MAV ahead with same heading: Blue full gate, no MAV fire.
    env.entities['MAV'].state.x=4500.
    env.entities['Blue1'].state.x=2500.
    env._attack_streak['Blue1','MAV']=2
    audit.install_intervention(env,mode)
    events,deaths=env._resolve_attacks()
    assert dict(attacker='Blue1',target='MAV') in events
    assert deaths['MAV']=='blue_attack'
    assert not any(e['attacker']=='MAV' for e in events)
