from copy import deepcopy
from itertools import product
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv as Env, load_environment_config, RED_IDS, BLUE_IDS
from env.blue_policy import BluePolicy
from env.cap_blue_policy import CAPBluePolicy, balanced_assignment
from env.vector_env import _environment_state, _restore_environment_state, MAVUAVVectorEnv
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.evaluate_happo import validate_checkpoint_contract
ROOT=Path(__file__).resolve().parents[1]


def cfg(): return load_environment_config(ROOT/'configs/env_v312.yaml')
def env():
    e=Env(cfg(),randomize=False); e.reset(seed=1); return e
def mappings(e): return ({b:e.entities[b] for b in BLUE_IDS},{r:e.entities[r] for r in RED_IDS})
def prepare(e,step):
    b,r=mappings(e); e.blue_policy.prepare_step(b,r,step); return b,r


@pytest.mark.parametrize('m,n,loads',[(4,4,[1,1,1,1]),(3,4,[0,1,1,1]),(4,3,[1,1,2]),(4,2,[2,2]),(4,1,[4])])
def test_balanced_cases(m,n,loads):
    e=env(); b,r=mappings(e)
    for i,a in enumerate(BLUE_IDS): b[a].state.alive=i<m
    for i,a in enumerate(RED_IDS): r[a].state.alive=i<n
    assigned=balanced_assignment(b,r); c=Counter(assigned.values())
    assert len(assigned)==m and sorted(c[a] for a in RED_IDS[:n])==loads


def test_exact_not_greedy_and_3D_distance():
    e=env(); b,r=mappings(e)
    for a in BLUE_IDS: b[a].state.alive=a in BLUE_IDS[:2]
    for a in RED_IDS: r[a].state.alive=a in RED_IDS[:2]
    # Greedy B1->MAV costs 1+5=6; global B1->UAV1 costs 3+1=4.
    for a,x in [('Blue1',1),('Blue2',-1),('MAV',0),('UAV1',4)]:
        st=e.entities[a].state; st.x=x; st.y=0; st.h=6000
    assert balanced_assignment(b,r)=={'Blue1':'UAV1','Blue2':'MAV'}
    r['UAV1'].state.h=6002
    assigned=balanced_assignment(b,r)
    costs=[]
    for t in product(RED_IDS[:2],repeat=2):
        if len(set(t))!=2: continue
        cost=sum(np.linalg.norm(np.array([b[bid].state.x-r[aid].state.x,0,b[bid].state.h-r[aid].state.h])) for bid,aid in zip(BLUE_IDS[:2],t))
        costs.append((cost,t))
    assert tuple(assigned.values())==min(costs)[1]


def test_tie_deterministic_ID_order():
    e=env(); b,r=mappings(e)
    for a in e.entities.values(): a.state.x=a.state.y=0; a.state.h=6000
    expected=dict(zip(sorted(BLUE_IDS),sorted(RED_IDS)))
    for _ in range(5): assert balanced_assignment(dict(reversed(list(b.items()))),dict(reversed(list(r.items()))))==expected


def test_refresh_hold_death_and_Blue_exit():
    e=env(); p=e.blue_policy
    state=deepcopy(p.state_dict())
    e.entities['UAV1'].state.x+=1000
    prepare(e,1)
    assert p.state_dict()['guidance_state']==state['guidance_state']
    prepare(e,2); assert all(s.last_refresh_step==2 for s in p._guidance_state.values())
    e.entities['UAV1'].state.alive=False
    prepare(e,3)
    assert all(s.last_refresh_step==3 for s in p._guidance_state.values())
    assert all(s.target_id!='UAV1' for s in p._guidance_state.values())
    e.entities['Blue4'].state.alive=False
    prepare(e,5)
    assert p._guidance_state['Blue4'].target_id is None
    assert len(set(p._guidance_state[b].target_id for b in BLUE_IDS[:3]))==3


@pytest.mark.parametrize('recovery',[False,True])
def test_action_call_order_invariant(recovery):
    e=env(); b,r=prepare(e,0); p=e.blue_policy
    if recovery:
        b['Blue1'].state.h=1010; b['Blue1'].state.theta=-.5
    before=deepcopy(p.state_dict())
    forward={bid:p.action(b[bid],r,0) for bid in BLUE_IDS}
    p.load_state_dict(before)
    reverse={bid:p.action(b[bid],r,0) for bid in reversed(BLUE_IDS)}
    for bid in BLUE_IDS: np.testing.assert_array_equal(forward[bid],reverse[bid])


@pytest.mark.parametrize('case',['normal','altitude','horizontal'])
def test_existing_controller_exact_same_actions(case):
    e=env(); b,r=prepare(e,1); cap=e.blue_policy
    legacy=BluePolicy(cap.decision_dt,cap.physics_dt,cap.battlefield,2)
    legacy.load_state_dict({k:v for k,v in cap.state_dict().items() if k in ('guidance_mode','target_refresh_steps','guidance_state')})
    blue=b['Blue1']
    if case=='altitude': blue.state.h=1010; blue.state.theta=-.5
    if case=='horizontal': blue.state.x=99990; blue.state.psi=0
    np.testing.assert_array_equal(cap.action(blue,r,1),legacy.action(blue,r,1))
    if case!='normal': assert cap._guidance_state['Blue1'].force_refresh
    prepare(e,3)
    assert cap._last_assignment_step==(0 if case=='normal' else 3)


def test_state_roundtrip_exact_next_action():
    e=env(); e.step(np.zeros((4,3))); state=deepcopy(_environment_state(e))
    other=env(); _restore_environment_state(other,state)
    assert other.blue_policy.state_dict()==e.blue_policy.state_dict()
    b,r=prepare(e,1); bb,rr=prepare(other,1)
    for bid in BLUE_IDS: np.testing.assert_array_equal(e.blue_policy.action(b[bid],r,1),other.blue_policy.action(bb[bid],rr,1))
    assert other.blue_policy.state_dict()==e.blue_policy.state_dict()


def test_config_exact_two_fields_and_training_identical():
    old=load_environment_config(ROOT/'configs/env_v311.yaml'); new=cfg()
    assert new['environment_version']=='heterogeneous_mavuav_4v4_v3_12'
    assert new['blue_policy']['target_strategy']=='coordinated_assignment'
    restored=deepcopy(new); restored['environment_version']=old['environment_version']; restored['blue_policy']['target_strategy']=old['blue_policy']['target_strategy']
    assert restored==old
    with (ROOT/'configs/happo_v311_baseline.yaml').open() as f: t1=yaml.safe_load(f)
    with (ROOT/'configs/happo_v312_baseline.yaml').open() as f: t2=yaml.safe_load(f)
    assert t1==t2
    obs,_=Env(old,randomize=False).reset(seed=1); e=env()
    for a in RED_IDS: np.testing.assert_array_equal(obs[a],e._observations()[a])
    np.testing.assert_array_equal(Env(old,randomize=False).reset(seed=1)[0]['MAV'],obs['MAV'])


@pytest.mark.parametrize('config',[{'actor_variant':'tam','critic_variant':'tam_attention'},
    {'method_variant':'rgaa'},{'method_variant':'tacm_rgaa'},
    {'actor_variant':'entity_recurrent','critic_variant':'entity_attention_recurrent'}])
def test_historical_algorithm_contract_not_opened(config):
    with pytest.raises(ValueError): HAPPOTrainer(cfg(),dict(device='cuda',num_envs=1,**config))


def test_cap_diagnostics_are_pure():
    e=env(); p=e.blue_policy; b,r=mappings(e); state=deepcopy(p.state_dict())
    for bid in BLUE_IDS:
        d=p.diagnostics(b[bid],r,0)
        assert d['assigned_target_id'] in RED_IDS and d['assignment_distance']>0
    assert p.team_diagnostics(b,r,0)['max_target_load']==1
    assert p.state_dict()==state


def test_other_target_death_requests_cohort_diagnostics():
    e=env(); b,r=mappings(e); p=e.blue_policy
    r[p._guidance_state['Blue1'].target_id].state.alive=False
    assert all(p.diagnostics(b[bid],r,1)['guidance_refresh_due'] for bid in BLUE_IDS)
    prepare(e,1)
    assert all(not p.diagnostics(b[bid],r,1)['guidance_refresh_due'] for bid in BLUE_IDS)


def test_evaluation_log_reports_actual_CAP_strategy():
    from algorithm.train_happo import _evaluation_lines
    row=dict(sampled_steps=2,blue_target_strategy='coordinated_assignment',red_win_rate=0,
             blue_win_rate=0,draw_rate=1,mean_episode_return=0,mean_red_attack_kills=0,MAV_survival_rate=1)
    assert 'coordinated_assignment' in _evaluation_lines('eval',row)
    row.pop('blue_target_strategy')
    assert 'nearest_red_aircraft' in _evaluation_lines('eval',row)


def test_vector_subprocess_exact_resume():
    a=MAVUAVVectorEnv(num_envs=2,config_path=cfg()); b=MAVUAVVectorEnv(num_envs=2,config_path=cfg())
    try:
        a.reset(seed=312); a.step(np.zeros((2,4,3)))
        state=a.get_env_states(); b.set_env_states(state,a.reset_counts,a.base_seed)
        x=a.step(np.ones((2,4,3))*.1); y=b.step(np.ones((2,4,3))*.1)
        for i in range(6): np.testing.assert_array_equal(x[i],y[i])
        sa=a.get_env_states(); sb=b.get_env_states()
        assert [s['blue_policy_state'] for s in sa]==[s['blue_policy_state'] for s in sb]
    finally: a.close(); b.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_vanilla_CUDA_tiny_checkpoint_resume(tmp_path,monkeypatch):
    with (ROOT/'configs/happo_v312_baseline.yaml').open() as f: training=yaml.safe_load(f)['training']
    training.update(num_envs=1,rollout_steps=2,hidden_dim=8,ppo_epochs=1,minibatch_size=2)
    a=HAPPOTrainer(cfg(),training); b=None
    try:
        a.collect_rollout(); path=tmp_path/'checkpoint_final.pt'; a.save_checkpoint(path)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint_contract(payload,cfg())
        assert payload['reward_mode']=='heterogeneous_role_coupled_gate_v1'
        b=HAPPOTrainer(cfg(),training); assert b.load_checkpoint(path)==2
        cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all()
        a.collect_rollout()
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
        b.collect_rollout()
        np.testing.assert_array_equal(a.observations,b.observations)
        from tools import audit_cap_blue
        import sys
        monkeypatch.setattr(sys,'argv',['audit_cap_blue','--run-dir',str(tmp_path),'--episodes','1','--output',str(tmp_path/'audit')])
        audit_cap_blue.main()
        import json
        audit=json.loads((tmp_path/'audit/summary.json').read_text())
        assert audit['metrics']['fraction_unnecessary_duplicate_assignment']==0
        assert audit['actor_parameters_unchanged'] and audit['input_SHA_unchanged']
    finally:
        a.close()
        if b: b.close()
