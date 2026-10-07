"""Third-stage observer contracts; no trainer or production mutations."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tools import audit_v311_environment_rationality as a


@pytest.fixture
def config():
    return a.load_environment_config(a.ROOT/'configs/env_v311.yaml')


def test_cached_snapshot_is_pure_and_distinct(config):
    env=a.Env(config); env.reset(seed=1011)
    env.blue_policy._guidance_state['Blue1'].target_id='MAV'
    blue=env.entities['Blue1'].state; uav=env.entities['UAV1'].state
    uav.x,uav.y,uav.h=blue.x+1,blue.y,blue.h
    before=deepcopy(env.blue_policy.state_dict())
    row=a.navigation_snapshot(env)[0]
    assert row['cached_target']=='MAV' and row['instantaneous_nearest']=='UAV1'
    assert env.blue_policy.state_dict()==before


def test_concentration_counts():
    c=a.concentration(['UAV1','UAV1','UAV1','UAV2'])
    assert c['max_same']==3 and c['pressure_UAV1']==3 and c['pressure_UAV2']==1
    assert a.concentration([])['max_same']==0


def test_source_all_four_can_select_same_target(config):
    env=a.Env(config); env.reset(seed=1011)
    # Four live Red choices: unlike a lone survivor, duplicates are not necessary.
    for aid in a.RED_IDS:
        st=env.entities[aid].state; st.alive=True
        st.x,st.y,st.h=(0 if aid=='UAV1' else -20000),0,6000
    for bid in a.BLUE_IDS:
        st=env.entities[bid].state; st.x,st.y,st.h=1000,0,6000
    red={aid:env.entities[aid] for aid in a.RED_IDS}
    assert [env.blue_policy.select_target(env.entities[b],red).aircraft_id for b in a.BLUE_IDS]==['UAV1']*4


def test_kill_unique_attribution_and_multi_kill():
    events=[dict(attacker='UAV1',target='Blue1'),dict(attacker='UAV2',target='Blue1'),dict(attacker='UAV1',target='Blue2')]
    ks=a.kill_sets(events)
    assert ks['UAV1']=={'Blue1','Blue2'} and ks['UAV2']=={'Blue1'} and len(set.union(*ks.values()))==2
    d=a.dominance([4,0,0]); assert d['max_agent_kills']==4 and d['dominance_share']==1 and d['kill_HHI']==1
    assert a.dominance([0,0,0])['dominance_share'] is None


def test_shared_gap_is_per_unique_target_noncontributor(config):
    info=dict(uav2_process_reward=-.1,event_reward=90)
    pair=dict(full_gate=0,distance_m=9000,ATA_deg=150,direct_visible=0)
    row=a.contribution_gap('UAV2','Blue1',{'UAV1'},pair,info,config)
    assert row['non_contributor']==1 and row['shared_kill_component']==100 and row['shared_event_reward']==90
    pair['full_gate']=1
    assert a.contribution_gap('UAV2','Blue1',{'UAV1'},pair,info,config)['non_contributor']==0


def test_MAV_centroid_and_visibility(config):
    env=a.Env(config); env.reset(seed=1011)
    mav=env.entities['MAV'].state; mav.x,mav.y,mav.h=0,0,6000
    for u in a.UAVS:
        s=env.entities[u].state; s.x,s.y,s.h=3000,4000,6000
    for b in a.BLUE_IDS: env.entities[b].state.alive=False
    b=env.entities['Blue1'].state; b.alive=True; b.x,b.y,b.h=4000,4000,6000
    # Dedicated geometry tests isolate sensing predicates without mutating production.
    env.direct_visible=lambda own,target: target=='Blue1' and own=='UAV1'
    env.team_visible=lambda target: target=='Blue1'
    env.datalink_visible=lambda own,target: target=='Blue1' and own!='UAV1'
    row=a.support_snapshot(env,dict(mav_process_reward=.12,mav_R_threat=0,mav_R_aspect=0,mav_R_aware=.3))
    assert row['MAV_UAV_centroid_m']==5000
    assert row['MAV_direct_count']==0 and row['team_visible_count']==1 and row['UAV_datalink_only_total']==2
    assert row['MAV_exclusive_information_count']==0


def test_real_team_only_awareness_without_MAV_direct(config):
    env=a.Env(config); env.reset(seed=1011)
    for b in a.BLUE_IDS: env.entities[b].state.alive=b=='Blue1'
    mav=env.entities['MAV'].state; mav.x,mav.y,mav.h,mav.psi,mav.theta=0,0,6000,0,0
    b=env.entities['Blue1'].state; b.x,b.y,b.h,b.psi,b.theta=16000,0,6000,0,0
    u=env.entities['UAV1'].state; u.x,u.y,u.h=15500,0,6000
    assert not env.direct_visible('MAV','Blue1') and env.team_visible('Blue1')
    _,info=env._role_process_rewards()
    assert a.verify_awareness_component(env,info)==pytest.approx(.3)
    assert info['mav_process_reward']>0


@pytest.mark.parametrize('mode',['B1','B2'])
def test_blue_variants_no_config_write(config,mode):
    env=a.Env(config); env.reset(seed=1011); before=deepcopy(env.config)
    a.install_blue_variant(env,mode)
    red={aid:env.entities[aid] for aid in a.RED_IDS}
    for b in a.BLUE_IDS: env.blue_policy.action(env.entities[b],red,0)
    assert env.config==before
    if mode=='B1': assert len({s.target_id for s in env.blue_policy._guidance_state.values()})==4


@pytest.mark.parametrize('mode,limit',[('W1',2),('W2',1),('W3',4)])
def test_weapon_variants_resources_and_config(config,mode,limit):
    env=a.Env(config); env.reset(seed=1011); before=deepcopy(env.config)
    for aid,e in env.entities.items():
        e.state.alive=aid in ('MAV','UAV1')+a.BLUE_IDS
        st=e.state; st.x=2000 if aid in a.BLUE_IDS else 0; st.y=0; st.h=6000; st.psi=0; st.theta=0
    a.install_weapon_variant(env,mode)
    for i in range(3):
        env.step_count=i; events,deaths=env._resolve_attacks()
    assert len(a.kill_sets(events)['UAV1'])==limit
    assert env.config==before
    if mode=='W3':
        # New live target at k+1; no progress for five steps after k.
        b=env.entities['Blue1'].state; b.alive=True
        for i in range(3,8):
            env.step_count=i; events,_=env._resolve_attacks()
            assert not events and env._attack_streak['UAV1','Blue1']==0
        env.step_count=8; env._resolve_attacks()
        assert env._attack_streak['UAV1','Blue1']==1


def test_historical_duplicate_credits_not_guessed():
    prior=[dict(training_seed=1,episode=0,sampled_steps=2000000,outcome='draw',red_attack_kills=1)]
    agents=[dict(training_seed=1,episode=0,sampled_steps=2000000,agent=u,streak3=int(u!='UAV3')) for u in a.UAVS]
    row=a.population_dominance(prior,agents)[0]
    assert not row['unique_attribution_exact'] and row['unique_attribution_missing']


def test_kill_windows_no_duplicate_step_denominators():
    ss=[dict(step=k,alive_blue_pre=4,attack_events='[{"attacker":"UAV1","target":"Blue1"}]' if k in (7,9) else '[]') for k in range(1,12)]
    scopes=a.scopes(ss)
    assert len(scopes['kill_before5'])==7
    assert len({r['step'] for r in scopes['kill_before5']})==7


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA policy replay smoke requires GPU')
def test_CUDA_matched_scenario_exact_evaluator_SHA_parameters_RNG(config):
    path=a.ROOT/'outputs/happo_v311_seed1_2m/checkpoint_final.pt'
    if not path.exists(): pytest.skip('local research checkpoint absent')
    before=a.sha(path); payload=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        tc=payload['trainer_config']; actors=a.IndependentActors(hidden_dim=tc['hidden_dim']).cuda().eval(); actors.load_state_dict(payload['actors'])
    params={k:v.clone() for k,v in actors.state_dict().items()}
    rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state_all()
    ep=a.replay(actors,config,1011,2011)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        official=a.evaluate_actors(actors,config,1,'main',1011,'cuda',deterministic=False,action_seed=2011)[0]
    assert ep['result']==official and ep['result']['red_attack_kills']==4
    assert all(torch.equal(v,params[k]) for k,v in actors.state_dict().items()) and a.sha(path)==before
    assert all(np.isfinite(r['team_reward']) for r in ep['steps'])
