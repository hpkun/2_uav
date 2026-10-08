"""Frozen reward, latent probability, scale-safe critic and legacy isolation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import json

import numpy as np
import pytest
import torch
import yaml

import env.mavuav as env_module
import algorithm.happo.trainer as trainer_module
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv as Env, RED_IDS, BLUE_IDS, ENTITY_IDS
from env.vector_env import MAVUAVVectorEnv, _environment_state, _restore_environment_state
from env.reward_chen_v315 import angle_reward, distance_reward, speed_reward, mav_distance_reward
from algorithm.common.networks import GaussianActor
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.happo.v315_protocol import ValueNorm, V315_DEFAULTS, clipped_huber_value_loss
from algorithm.evaluate_happo import validate_checkpoint_contract

ROOT = Path(__file__).resolve().parents[1]


def config():
    return env_module.load_environment_config(ROOT / 'configs/env_v315.yaml')


def training(**overrides):
    c = yaml.safe_load((ROOT / 'configs/happo_v315_baseline.yaml').read_text())['training']
    c.update(device='cpu', num_envs=1, hidden_dim=8, rollout_steps=4, ppo_epochs=1, minibatch_size=4)
    c.update(overrides)
    return c


@pytest.fixture
def trainers(monkeypatch):
    monkeypatch.setattr(trainer_module, 'MAVUAVVectorEnv', lambda *a, **k: MAVUAVVectorEnv(*a, parallel=False, **k))
    opened = []
    def create(env=None, cfg=None):
        t = HAPPOTrainer(env or config(), cfg or training())
        opened.append(t)
        return t
    yield create
    for t in opened:
        t.close()


def scene():
    e = Env(config()); e.reset(seed=1)
    for aid in ENTITY_IDS:
        s = e.entities[aid].state
        s.x, s.y, s.h, s.v, s.psi, s.theta = 0., 0., 6000., 250., 0., 0.
    for bid in BLUE_IDS:
        e.entities[bid].state.x = 8000.
    e.entities['MAV'].state.x = -15000.
    return e


def test_environment_and_optimization_contract():
    old = env_module.load_environment_config(ROOT / 'configs/env_v314.yaml')
    c = config()
    for key in ('simulation','battlefield','aircraft_specs','scenario','randomization_profiles','sensing','normalization','safety','combat','blue_policy'):
        assert c[key] == old[key]
    assert not any(c['reward'].values())
    cfg = yaml.safe_load((ROOT / 'configs/happo_v315_baseline.yaml').read_text())['training']
    assert all(cfg[k] == v for k,v in V315_DEFAULTS.items())
    assert (cfg['actor_learning_rate'],cfg['critic_learning_rate'],cfg['entropy_coef'],cfg['max_grad_norm']) == (.0005,.0005,.01,10.)
    assert (cfg['num_envs'],cfg['rollout_steps'],cfg['ppo_epochs'],cfg['minibatch_size'],cfg['hidden_dim']) == (16,128,4,256,128)
    assert cfg['actor_log_std_init'] == -.25 and not cfg['randomization_curriculum_enabled']


@pytest.mark.parametrize('ata,aa', [(0,0),(.2,.9),(np.pi,np.pi),(0,np.pi)])
def test_angle_exact_unclipped(ata,aa):
    assert angle_reward(ata,aa) == 1-(ata+aa)/np.pi


@pytest.mark.parametrize('d,expected', [(0,1),(5000,1),(5000.1,np.exp(-.921*.0001)),(8000,np.exp(-.921*3)),(9999,np.exp(-.921*4.999)),(10000,-1),(15000,-1)])
def test_distance_segments(d,expected):
    assert distance_reward(d) == pytest.approx(expected)


@pytest.mark.parametrize('vb,expected', [(124,1),(125,1),(250,0),(375,-1),(376,-1)])
def test_speed_segments(vb,expected):
    assert speed_reward(250,vb) == expected


def test_no_visible_target_dense_and_disabled_zero(monkeypatch):
    e = scene(); monkeypatch.setattr(e,'team_visible',lambda bid: False)
    d = e._chen_dense_rewards()
    for aid in RED_IDS[1:]:
        p=aid.lower()
        assert d[f'{p}_reward_target'] is None
        assert all(d[f'{p}_r_{component}']==0 for component in ('speed','angle','distance','height','dodge'))
    assert all(d[f'mav_r_{component}']==0 for component in ('threat','pos','aware','support'))
    assert d['mav_r_dist'] == .2


def test_max_situation_score_tie_and_not_reward(monkeypatch):
    e=scene(); monkeypatch.setattr(e,'team_visible',lambda bid: bid in ('Blue1','Blue2'))
    e.entities['Blue2'].state.x=12000
    # A farther but better-angle enemy can beat the nearer one.
    def geometry(a,b):
        good=b is e.entities['Blue2'].state
        return SimpleNamespace(distance=12000. if good else 8000.,ata=0. if good else np.pi,
                               aa=0. if good else np.pi,relative_velocity=np.zeros(3))
    monkeypatch.setattr(env_module,'compute_pairwise_geometry',geometry)
    d=e._chen_dense_rewards(); assert d['uav1_reward_target']=='Blue2'
    local,_=e._chen_rewards(d,[],{},True)
    assert local['UAV1']==10*d['uav1_r_speed']+15*d['uav1_r_angle']+10*d['uav1_r_distance']
    assert d['uav1_situation_score']==.35
    monkeypatch.setattr(env_module,'compute_pairwise_geometry',lambda *args: SimpleNamespace(distance=8000.,ata=0.,aa=0.,relative_velocity=np.zeros(3)))
    assert e._chen_dense_rewards()['uav1_reward_target']=='Blue1'
    e.entities['Blue1'].state.alive=False
    assert e._chen_dense_rewards()['uav1_reward_target']=='Blue2'


@pytest.mark.parametrize('aid,cause,value', [('UAV1','boundary',-100),('UAV2','blue_attack',-200),('MAV','boundary',-200),('MAV','blue_attack',-200)])
def test_loss_event_sources(aid,cause,value):
    e=scene(); d=e._chen_dense_rewards()
    _,info=e._chen_rewards(d,[],{aid:cause},aid!='MAV')
    assert info[f'{aid.lower()}_r_event']==value
    for other in RED_IDS:
        if other!=aid: assert info[f'{other.lower()}_r_event']==0
    if aid!='MAV':
        assert info[f'{aid.lower()}_boundary_loss_event']==(value if cause=='boundary' else 0)
        assert info[f'{aid.lower()}_combat_loss_event']==(value if cause=='blue_attack' else 0)


def test_unique_kills_cap_duplicates_and_no_future_dead_mav_contribution():
    e=scene(); d=e._chen_dense_rewards()
    for index,bid in enumerate(BLUE_IDS):
        events=[{'attacker':'UAV1','target':bid}]*2
        _,info=e._chen_rewards(d,events,{bid:'red_attack'},True)
        assert info['uav1_kill_event']==200
        assert info['mav_r_event']==50
        assert info['mav_team_contribution_cumulative']==50*(index+1)
        _,again=e._chen_rewards(d,events,{bid:'red_attack'},True)
        assert again['mav_r_event']==again['uav1_r_event']==0
    assert e._chen_mav_contribution==200
    e=scene(); d=e._chen_dense_rewards()
    _,info=e._chen_rewards(d,[{'attacker':'UAV2','target':'Blue1'}],{'Blue1':'red_attack'},False)
    assert info['mav_r_event']==info['mav_team_contribution_cumulative']==0
    assert info['uav2_r_event']==200


def test_simultaneous_own_kill_and_own_combat_loss():
    e=scene();d=e._chen_dense_rewards()
    _,info=e._chen_rewards(d,[{'attacker':'UAV1','target':'Blue1'}],{'Blue1':'red_attack','UAV1':'blue_attack'},True)
    assert info['uav1_kill_event']==200 and info['uav1_combat_loss_event']==-200
    assert info['uav1_r_event']==0


@pytest.mark.parametrize('distance,expected', [(0,-1),(2500,-.5),(4999,-.0002),(5000,-.5),(7500,-.25),(10000,.2),(None,.2)])
def test_mav_distance_segments(distance,expected):
    assert mav_distance_reward(distance,5000,10000)==pytest.approx(expected)


def test_mav_direct_only_reverse_ATA(monkeypatch):
    e=scene();e.entities['Blue1'].state.x=-13000.;e.entities['Blue1'].state.psi=np.pi
    monkeypatch.setattr(e,'direct_visible',lambda aid,bid: bid=='Blue1')
    d=e._chen_dense_rewards()
    assert d['mav_r_dist']==pytest.approx(-.6)
    assert d['mav_r_aspect']==-1
    assert d['mav_r_safety']==pytest.approx(-.5)
    monkeypatch.setattr(e,'direct_visible',lambda *args: False)
    d=e._chen_dense_rewards();assert d['mav_r_dist']==.2 and d['mav_r_aspect']==0


def test_shared_divide4_no_terminal_and_environment_state_restore(monkeypatch):
    e=scene();e.step_count=149
    e.entities['Blue4'].state.alive=False
    e._red_attack_kills.add('Blue4');e._chen_seen_blue_kills.add('Blue4')
    e._chen_mav_contribution=50.
    monkeypatch.setattr(e.blue_policy,'action',lambda *args: np.zeros(3))
    monkeypatch.setattr(e,'_resolve_attacks',lambda: ([],{}))
    _,rewards,term,trunc,info=e.step(np.zeros((4,3)))
    assert trunc and not term and info['terminal_reward']==0
    expected=sum(info[f'{aid.lower()}_reward_local'] for aid in RED_IDS)/4
    assert all(rewards[aid]==expected for aid in RED_IDS)
    assert info['team_reward']==expected
    state=_environment_state(e); other=scene();_restore_environment_state(other,state)
    assert other._chen_local_sums==e._chen_local_sums
    assert other._chen_seen_blue_kills==e._chen_seen_blue_kills
    assert other._chen_mav_contribution==e._chen_mav_contribution


@pytest.mark.parametrize('raw', [0.,2.,-2.,5.,-5.,8.,-8.,20.,-20.])
def test_unchanged_actor_latent_ratio_identity(raw):
    actor=GaussianActor(hidden_dim=8); obs=torch.zeros((5,100));u=torch.full((5,3),raw)
    old=actor._distribution(obs).log_prob(u).sum(-1)
    new,entropy=actor.evaluate_raw_actions(obs,u)
    assert torch.equal((new-old).exp(),torch.ones(5))
    assert torch.isfinite(entropy).all()
    # Historical path remains an inverse-tanh/Jacobian density.
    legacy,_=actor.evaluate_actions(obs,u.tanh())
    if abs(raw)==20: assert not torch.allclose(legacy,old)


def test_raw_buffer_PPO_and_preceding_factor_without_reconstruction(trainers,monkeypatch):
    t=trainers();t.collect_rollout()
    assert t.buffer.raw_actions.shape==t.buffer.actions.shape
    np.testing.assert_allclose(np.tanh(t.buffer.raw_actions),t.buffer.actions,rtol=1e-6,atol=1e-7)
    for actor in t.actors.actors:
        monkeypatch.setattr(actor,'evaluate_actions',lambda *args: pytest.fail('historical inverse path used'))
    calls=[]
    original=trainer_module.preceding_factor_update
    def spy(factor,old,new,active):
        if not calls: assert torch.equal(factor,torch.ones_like(factor))
        expected=factor*torch.where(active>.5,(new-old).exp(),torch.ones_like(factor))
        result=original(factor,old,new,active)
        torch.testing.assert_close(result,expected)
        calls.append((factor.clone(),result.clone()))
        return result
    monkeypatch.setattr(trainer_module,'preceding_factor_update',spy)
    m=t.update();assert len(calls)==4 and all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
    for previous,next_call in zip(calls,calls[1:]):assert torch.equal(previous[1],next_call[0])


def test_inactive_factor_identity():
    f=torch.tensor([1.,2.,3.]);old=torch.zeros(3);new=torch.tensor([.1,.2,.3]);mask=torch.tensor([1.,0.,0.])
    out=trainer_module.preceding_factor_update(f,old,new,mask)
    assert torch.equal(out[1:],f[1:])


def test_valuenorm_roundtrip_and_raw_GAE(trainers):
    t=trainers();t.value_normalizer.update(torch.tensor([100.,200.]))
    x=torch.tensor([-1000.,0.,1234.]);torch.testing.assert_close(t.value_normalizer.denormalize(t.value_normalizer.normalize(x)),x)
    with torch.no_grad():
        for p in t.critic.parameters():p.zero_()
    t.collect_rollout();np.testing.assert_allclose(t.buffer.values,150.)
    next_value=np.full(1,150.,dtype=np.float32);gae=np.zeros(1,dtype=np.float32)
    for step in reversed(range(t.buffer.horizon)):
        continuation=1-np.logical_or(t.buffer.terminated[step],t.buffer.truncated[step]).astype(np.float32)
        gae=t.buffer.rewards[step]+.99*next_value*continuation-t.buffer.values[step]+.99*.95*continuation*gae
        np.testing.assert_allclose(t.buffer.advantages[step],gae,rtol=1e-6)
        next_value=t.buffer.values[step]


def test_huber_delta_and_clipping():
    # error20 Huber10 =150; clipping produces error19.8 =>148 (max is150).
    assert clipped_huber_value_loss(torch.tensor([0.]),torch.tensor([0.]),torch.tensor([20.])).item()==150
    # new2 perfectly fits target2, but old0 permits only .2: max loss=.5*1.8².
    assert clipped_huber_value_loss(torch.tensor([2.]),torch.tensor([0.]),torch.tensor([2.])).item()==pytest.approx(1.62)


def test_orthogonal_and_log_std(trainers):
    t=trainers()
    for actor in t.actors.actors:
        assert torch.equal(actor.log_std,torch.full((3,),-.25))
    for model,gain in [(a.network,.01) for a in t.actors.actors]+[(t.critic.network,1.)]:
        linear=[m for m in model if isinstance(m,torch.nn.Linear)]
        for i,m in enumerate(linear):
            w=m.weight;gram=w@w.T if w.shape[0]<=w.shape[1] else w.T@w
            torch.testing.assert_close(gram,torch.eye(len(gram))*(gain**2 if i==len(linear)-1 else 2),rtol=1e-4,atol=1e-6)
            assert torch.count_nonzero(m.bias)==0


def continuation_check(a,b):
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    a.collect_rollout();ma=a.update()
    torch.set_rng_state(cpu)
    if cuda is not None:torch.cuda.set_rng_state_all(cuda)
    b.collect_rollout();mb=b.update();assert ma==mb
    for field in ('actions','raw_actions','log_probs','values','returns','advantages','rewards'):
        np.testing.assert_array_equal(getattr(a.buffer,field),getattr(b.buffer,field))
        assert np.isfinite(getattr(a.buffer,field)).all()
    for left,right in ((a.actors,b.actors),(a.critic,b.critic),(a.value_normalizer,b.value_normalizer)):
        for k,v in left.state_dict().items():assert torch.equal(v,right.state_dict()[k])


def test_exact_resume_weights_metadata_and_cross_version_rejection(trainers,tmp_path):
    a=trainers();a.collect_rollout();a.update();path=tmp_path/'checkpoint.pt';a.save_checkpoint(path)
    b=trainers();assert b.load_checkpoint(path)==4;continuation_check(a,b)
    a.save(tmp_path/'weights.pt');b.load(tmp_path/'weights.pt')
    assert all(torch.equal(v,b.value_normalizer.state_dict()[k]) for k,v in a.value_normalizer.state_dict().items())
    data=torch.load(path,weights_only=False);validate_checkpoint_contract(data,config())
    old=env_module.load_environment_config(ROOT/'configs/env_v314.yaml')
    oldcfg=training();[oldcfg.pop(k) for k in V315_DEFAULTS]
    legacy=trainers(old,oldcfg)
    with pytest.raises(RuntimeError):legacy.load_checkpoint(path)
    with pytest.raises(RuntimeError):legacy.load(path)
    legacy.save_checkpoint(tmp_path/'old.pt')
    with pytest.raises(RuntimeError):a.load_checkpoint(tmp_path/'old.pt')
    with pytest.raises(RuntimeError):a.load(tmp_path/'old.pt')
    for key in V315_DEFAULTS:
        bad=deepcopy(data);bad.pop(key)
        with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,config())
        bad=deepcopy(data);bad['trainer_config'].pop(key)
        with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,config())
    bad=deepcopy(data);bad.pop('value_normalizer')
    with pytest.raises(RuntimeError):validate_checkpoint_contract(bad,config())


def test_legacy_uses_original_sample_and_evaluation(trainers,monkeypatch):
    old=env_module.load_environment_config(ROOT/'configs/env_v314.yaml')
    c=training();[c.pop(k) for k in V315_DEFAULTS];t=trainers(old,c)
    assert t.value_normalizer is None and not hasattr(t.buffer,'raw_actions')
    for actor in t.actors.actors:
        monkeypatch.setattr(actor,'sample_with_raw',lambda *args: pytest.fail('legacy raw sample'))
        monkeypatch.setattr(actor,'evaluate_raw_actions',lambda *args: pytest.fail('legacy raw evaluate'))
    t.collect_rollout();t.update()


def test_critic_loss_inputs_share_current_normalized_scale(trainers,monkeypatch):
    t=trainers();t.value_normalizer.update(torch.tensor([-100.,300.]))
    t.collect_rollout();raw_old=t.buffer.values.copy();raw_returns=t.buffer.returns.copy()
    calls=[];original=trainer_module.clipped_huber_value_loss
    def spy(new,old,target,clip,delta):
        expected_old=t.value_normalizer.normalize(torch.as_tensor(raw_old.reshape(-1)))
        expected_return=t.value_normalizer.normalize(torch.as_tensor(raw_returns.reshape(-1)))
        # A full minibatch is shuffled together; match paired old/target values.
        pairs=list(zip(expected_old.tolist(),expected_return.tolist()))
        for o,r in zip(old.tolist(),target.tolist()):
            assert any(abs(o-a)<1e-5 and abs(r-b)<1e-5 for a,b in pairs)
        assert clip==.2 and delta==10
        calls.append(1)
        return original(new,old,target,clip,delta)
    monkeypatch.setattr(trainer_module,'clipped_huber_value_loss',spy)
    t.update();assert calls


def test_resolved_config_metadata_and_weights_save(trainers,tmp_path):
    from algorithm import train_happo
    from types import SimpleNamespace
    t=trainers()
    args=SimpleNamespace(profile='main',seed=1,device='cpu',num_envs=1,steps=32,
                         checkpoint_interval=16,eval_interval=0,log_interval=16,
                         eval_episodes=1,final_eval_episodes=1,eval_action_mode='stochastic',eval_action_seed=2000)
    resolved=train_happo._initial_resolved(args,t.environment_config,t,'cpu',None)
    assert all(resolved[k]==v for k,v in V315_DEFAULTS.items())
    t.save(tmp_path/'weights.pt');data=torch.load(tmp_path/'weights.pt',weights_only=False)
    assert all(data[k]==v for k,v in V315_DEFAULTS.items())
    validate_checkpoint_contract(data,config())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
def test_cuda_16env_smoke_resume_evaluator(tmp_path,monkeypatch):
    a=HAPPOTrainer(config(),training(device='cuda',num_envs=16,rollout_steps=2,minibatch_size=16));b=None
    try:
        assert a.vector_env.parallel and len(set(a.vector_env.worker_pids))==16
        assert os.getpid() not in a.vector_env.worker_pids
        obs=torch.zeros((2,100),device='cuda');raw=torch.tensor([[20.]*3,[-20.]*3],device='cuda')
        actor=a.actors.actors[0]
        old=actor._distribution(obs).log_prob(raw).sum(-1)
        new,_=actor.evaluate_raw_actions(obs,raw)
        assert torch.equal((new-old).exp(),torch.ones(2,device='cuda'))
        a.collect_rollout();m=a.update();assert a.env_steps==32
        assert all(np.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
        path=tmp_path/'checkpoint_final.pt';a.save_checkpoint(path)
        b=HAPPOTrainer(config(),training(device='cuda',num_envs=16,rollout_steps=2,minibatch_size=16))
        assert b.load_checkpoint(path)==32;continuation_check(a,b)
        assert all(torch.isfinite(p).all() for model in (a.actors,a.critic) for p in model.parameters())
        from algorithm import evaluate_happo
        monkeypatch.setattr(sys,'argv',['evaluate_happo',str(path),'--episodes','1','--device','cuda','--action-mode','stochastic'])
        evaluate_happo.main()
        summary=json.loads((tmp_path/'evaluation_final_stochastic_summary.json').read_text())
        assert summary['reward_mode']=='chen_heterogeneous_v1' and summary['use_valuenorm'] is True
    finally:
        a.close()
        if b is not None:b.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires real CUDA')
def test_actual_v315_config_training_entry_tiny_smoke(tmp_path,monkeypatch):
    from algorithm import train_happo
    run=tmp_path/'run'
    def new_run(args):
        run.mkdir();return run
    monkeypatch.setattr(train_happo,'_new_run_dir',new_run)
    monkeypatch.setattr(sys,'argv',['train_happo','--steps','16','--profile','main',
                                  '--device','cuda','--num-envs','16',
                                  '--config',str(ROOT/'configs/happo_v315_baseline.yaml'),
                                  '--env-config',str(ROOT/'configs/env_v315.yaml'),
                                  '--checkpoint-interval','16','--eval-interval','0',
                                  '--final-eval-episodes','1','--eval-action-mode','stochastic'])
    train_happo.main()
    summary=json.loads((run/'summary.json').read_text())
    resolved=yaml.safe_load((run/'resolved_config.yaml').read_text())
    assert summary['sampled_steps']==16 and summary['status']=='complete'
    assert summary['device']=='cuda'
    for key,value in V315_DEFAULTS.items():
        assert summary[key]==resolved[key]==value
    payload=torch.load(run/'checkpoint_final.pt',weights_only=False)
    validate_checkpoint_contract(payload,config())
    assert payload['actor_architecture']['hidden_dim']==128
    assert torch.isfinite(payload['value_normalizer']['running_mean']).all()
