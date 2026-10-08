"""Frozen shared reward contract, legacy isolation and production tiny smoke."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

import env.mavuav as em
import algorithm.happo.trainer as tm
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv as Env, RED_IDS, BLUE_IDS
from env.vector_env import MAVUAVVectorEnv, _environment_state, _restore_environment_state
from env.reward_chen_v316 import CONFIG, METADATA, shared_reward
from env.reward_chen_v315 import angle_reward
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.evaluate_happo import validate_checkpoint_contract
from test_chen_baseline_v315 import continuation_check

ROOT = Path(__file__).resolve().parents[1]


def config(version=316):
    return em.load_environment_config(ROOT / f"configs/env_v{version}.yaml")


def training(**changes):
    cfg = yaml.safe_load((ROOT / 'configs/happo_v316_baseline.yaml').read_text())['training']
    cfg.update(device='cpu', num_envs=1, hidden_dim=8, rollout_steps=4, ppo_epochs=1, minibatch_size=4)
    cfg.update(changes)
    return cfg


def dense(value=0.):
    result = {'mav_r_safety': .1}
    for aid in RED_IDS[1:]:
        for key in ('speed', 'angle', 'distance'):
            result[f'{aid.lower()}_r_{key}'] = value
    return result


def reward(d=None, deaths=None, seen=None):
    return shared_reward(d or dense(), deaths or {}, set() if seen is None else seen, CONFIG)


@pytest.fixture
def trainers(monkeypatch):
    monkeypatch.setattr(tm, 'MAVUAVVectorEnv', lambda *a, **k: MAVUAVVectorEnv(*a, parallel=False, **k))
    opened = []
    def make(ec=None, tc=None):
        t = HAPPOTrainer(ec or config(), tc or training())
        opened.append(t)
        return t
    yield make
    for t in opened:
        t.close()


def test_frozen_environment_and_optimizer_inheritance():
    old, new = config(315), config()
    for key in old:
        if key not in ('environment_version', 'chen_reward'):
            assert old[key] == new[key]
    a = yaml.safe_load((ROOT/'configs/happo_v315_baseline.yaml').read_text())
    b = yaml.safe_load((ROOT/'configs/happo_v316_baseline.yaml').read_text())
    assert a == b
    assert new['chen_reward'] == CONFIG


@pytest.mark.parametrize('components', [(1,1,1),(0,0,0),(-1,-1,-1),(.2,-.8,.7)])
def test_raw_and_normalized_process(components):
    d=dense()
    for key, value in zip(('speed','angle','distance'),components):
        d[f'uav1_r_{key}']=value
    _,info,_=reward(d)
    expected=10*components[0]+15*components[1]+10*components[2]
    assert info['uav1_raw_process']==expected
    assert info['uav1_q_process']==expected/35
    assert -1 <= info['uav1_q_process'] <= 1


def test_scale_invariant_and_negative_angle_unclipped():
    _,info,_=reward(dense(1))
    assert info['mav_q_process']==.1
    assert info['shared_process_reward']==.775
    assert 150*info['shared_process_reward']==116.25
    assert 116.25 < CONFIG['mav']['death_penalty']
    assert 116.25 < CONFIG['uav']['kill_reward']
    assert angle_reward(np.pi,np.pi)==-1
    d=dense(1)
    d['uav1_r_angle']=-1
    assert reward(d)[1]['uav1_raw_process']==5


def test_mav_actual_no_direct_visible_safe_process(monkeypatch):
    e=Env(config());e.reset(seed=1)
    monkeypatch.setattr(e,'direct_visible',lambda *args:False)
    d=e._chen_dense_rewards()
    assert d['mav_r_dist']==.2 and d['mav_r_aspect']==0
    assert reward(d)[1]['mav_q_process']==.1


@pytest.mark.parametrize('dead', [False,True])
def test_no_target_or_dead_uav_zero(monkeypatch,dead):
    e=Env(config());e.reset(seed=1)
    if dead:e.entities['UAV1'].state.alive=False
    else:monkeypatch.setattr(e,'team_visible',lambda bid: False)
    _,info,_=reward(e._chen_dense_rewards())
    assert info['uav1_raw_process']==info['uav1_q_process']==0


def test_fixed_denominator():
    d=dense(1)
    for key in ('speed','angle','distance'):d[f'uav1_r_{key}']=0
    assert reward(d)[1]['shared_process_reward']==(.1+0+1+1)/4


@pytest.mark.parametrize('deaths,expected', [
    ({'Blue1':'red_attack'},200),
    ({'Blue1':'red_attack','Blue2':'red_attack'},400),
    ({'UAV1':'blue_attack'},-200),
    ({'UAV2':'boundary'},-100),
    ({'MAV':'boundary'},-200),
    ({'MAV':'blue_attack'},-200),
    ({'MAV':'blue_attack','Blue1':'red_attack'},0),
    ({'UAV1':'blue_attack','Blue1':'red_attack'},0),
    ({'MAV':'boundary','Blue1':'red_attack','Blue2':'red_attack'},200),
    ({'Blue1':'boundary'},0),
])
def test_event_combinations(deaths,expected):
    total, info, _=reward(deaths=deaths)
    assert info['shared_event_reward']==expected
    assert total==expected+info['shared_process_reward']


def test_unique_kill_deduplication():
    seen=set()
    assert reward(deaths={'Blue1':'red_attack'},seen=seen)[1]['shared_event_reward']==200
    assert reward(deaths={'Blue1':'red_attack'},seen=seen)[1]['shared_event_reward']==0


@pytest.mark.parametrize('attackers',[1,2,3])
def test_actual_step_cokill_broadcast_accounting_and_diagnostic_only(monkeypatch,attackers):
    def run(contribution):
        e=Env(config());e.reset(seed=1);e.step_count=149
        e._chen_mav_contribution=contribution
        monkeypatch.setattr(e.blue_policy,'action',lambda *a:np.zeros(3))
        def resolve():
            e.entities['Blue1'].state.alive=False;e._red_attack_kills.add('Blue1')
            return ([{'attacker':aid,'target':'Blue1'} for aid in RED_IDS[1:1+attackers]]*2,
                    {'Blue1':'red_attack'})
        monkeypatch.setattr(e,'_resolve_attacks',resolve)
        _,r,term,trunc,info=e.step(np.zeros((4,3)))
        assert trunc and not term
        assert info['shared_event_reward']==200
        assert info['terminal_reward']==0
        assert all(r[aid]==info['shared_reward'] for aid in RED_IDS)
        assert e.episode_return==info['shared_reward']
        summary=info['episode_summary']
        assert summary['team_reward_sum']==summary['episode_return']==info['shared_reward']
        assert summary['shared_event_reward_sum']==200
        assert summary['shared_process_reward_sum']==info['shared_process_reward']
        assert 'chen_local_reward_sums' not in summary
        restored=Env(config());restored.reset(seed=9)
        _restore_environment_state(restored,_environment_state(e))
        assert restored._chen_shared_sums==e._chen_shared_sums
        assert restored._chen_seen_blue_kills==e._chen_seen_blue_kills
        return r,info['mav_team_contribution_diagnostic']
    a,ca=run(0);b,cb=run(200)
    assert a==b and ca==50 and cb==200


def test_situation_score_selection_only(monkeypatch):
    e=Env(config());e.reset(seed=1)
    monkeypatch.setattr(e,'team_visible',lambda bid:bid=='Blue1')
    monkeypatch.setattr(em,'situation_score',lambda *a:1.)
    a=e._chen_dense_rewards();ra=reward(a)[0]
    monkeypatch.setattr(em,'situation_score',lambda *a:10000.)
    b=e._chen_dense_rewards();rb=reward(b)[0]
    assert a['uav1_situation_score']!=b['uav1_situation_score']
    assert ra==rb


def test_cpu_rollout_checkpoint_exact_resume_and_contract(trainers,tmp_path):
    a=trainers();a.collect_rollout();metrics=a.update()
    assert all(np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))
    path=tmp_path/'checkpoint_final.pt';a.save_checkpoint(path)
    data=torch.load(path,weights_only=False)
    validate_checkpoint_contract(data,config())
    for k,v in METADATA.items():assert data[k]==v
    b=trainers();assert b.load_checkpoint(path)==4
    continuation_check(a,b)
    weights=tmp_path/'weights.pt';a.save(weights);b.load(weights)
    for k in METADATA:
        bad=deepcopy(data);bad.pop(k)
        broken=tmp_path/'bad.pt';torch.save(bad,broken)
        with pytest.raises(RuntimeError):b.load_checkpoint(broken)
        with pytest.raises(RuntimeError):b.load(broken)
        with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,config())
        bad=deepcopy(data);bad[k]='wrong'
        torch.save(bad,broken)
        with pytest.raises(RuntimeError):b.load_checkpoint(broken)
        with pytest.raises(RuntimeError):b.load(broken)
        with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,config())
    legacy=trainers(config(315));legacy.save_checkpoint(tmp_path/'v315.pt')
    for load in (b.load_checkpoint,b.load):
        with pytest.raises(RuntimeError):load(tmp_path/'v315.pt')
    for load in (legacy.load_checkpoint,legacy.load):
        with pytest.raises(RuntimeError):load(path)
    with pytest.raises(RuntimeError):validate_checkpoint_contract(data,config(315))


@pytest.mark.parametrize('outcome',['red','blue'])
def test_terminal_events_no_extra_bonus(monkeypatch,outcome):
    e=Env(config());e.reset(seed=1)
    monkeypatch.setattr(e.blue_policy,'action',lambda *args:np.zeros(3))
    def resolve():
        if outcome=='red':
            for bid in BLUE_IDS:e.entities[bid].state.alive=False
            e._red_attack_kills.update(BLUE_IDS)
            return ([],{bid:'red_attack' for bid in BLUE_IDS})
        e.entities['MAV'].state.alive=False
        return ([],{'MAV':'blue_attack'})
    monkeypatch.setattr(e,'_resolve_attacks',resolve)
    _,rewards,terminated,truncated,info=e.step(np.zeros((4,3)))
    assert terminated and not truncated and info['outcome']==outcome
    assert info['terminal_reward']==info['episode_summary']['terminal_reward_sum']==0
    assert info['shared_event_reward']==(800 if outcome=='red' else -200)
    assert all(r==info['shared_event_reward']+info['shared_process_reward'] for r in rewards.values())


def test_resolved_metadata_and_persisted_process_metrics(trainers):
    from algorithm.train_happo import _episode_metrics, _initial_resolved
    from algorithm.happo.evaluation import summarize_records
    from types import SimpleNamespace
    t=trainers()
    args=SimpleNamespace(profile='main',seed=1,device='cpu',num_envs=1,steps=16,
        checkpoint_interval=16,eval_interval=0,log_interval=16,eval_episodes=1,
        final_eval_episodes=1,eval_action_mode='stochastic',eval_action_seed=2000)
    resolved=_initial_resolved(args,t.environment_config,t,'cpu',None)
    for k,v in METADATA.items():assert resolved[k]==v
    e=Env(config());e.reset(seed=1);e.step_count=149
    info=e.step(np.zeros((4,3)))[-1];record=info['episode_summary']
    for fn in (_episode_metrics,summarize_records):
        m=fn([record]);assert m['mean_shared_process_reward_sum']==record['shared_process_reward_sum']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
def test_cuda_16env_resume_evaluator(tmp_path,monkeypatch):
    a=HAPPOTrainer(config(),training(device='cuda',num_envs=16,rollout_steps=2,minibatch_size=16));b=None
    try:
        assert a.vector_env.parallel and len(set(a.vector_env.worker_pids))==16
        assert os.getpid() not in a.vector_env.worker_pids
        a.collect_rollout();m=a.update();assert a.env_steps==32
        assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
        path=tmp_path/'checkpoint_final.pt';a.save_checkpoint(path)
        b=HAPPOTrainer(config(),training(device='cuda',num_envs=16,rollout_steps=2,minibatch_size=16))
        assert b.load_checkpoint(path)==32;continuation_check(a,b)
        for model in (a.actors,a.critic):assert all(torch.isfinite(p).all() for p in model.parameters())
        from algorithm import evaluate_happo
        monkeypatch.setattr(sys,'argv',['evaluate_happo',str(path),'--episodes','1','--device','cuda','--action-mode','stochastic'])
        evaluate_happo.main()
        summary=json.loads((tmp_path/'evaluation_final_stochastic_summary.json').read_text())
        assert summary['reward_mode']=='chen_shared_event_dominant_v1'
        for k,v in METADATA.items():assert summary[k]==v
    finally:
        a.close()
        if b is not None:b.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
def test_production_entry_tiny_smoke(tmp_path,monkeypatch):
    from algorithm import train_happo
    run=tmp_path/'run'
    def new_run(args):run.mkdir();return run
    monkeypatch.setattr(train_happo,'_new_run_dir',new_run)
    monkeypatch.setattr(sys,'argv',['train_happo','--steps','16','--device','cuda','--num-envs','16',
        '--config',str(ROOT/'configs/happo_v316_baseline.yaml'),'--env-config',str(ROOT/'configs/env_v316.yaml'),
        '--checkpoint-interval','16','--eval-interval','0','--final-eval-episodes','1'])
    train_happo.main()
    resolved=yaml.safe_load((run/'resolved_config.yaml').read_text())
    summary=json.loads((run/'summary.json').read_text())
    for k,v in METADATA.items():assert summary[k]==resolved[k]==v
    assert summary['sampled_steps']==16 and summary['status']=='complete'
