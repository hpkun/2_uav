"""Frozen clean baseline contract, legacy isolation and real CUDA continuation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys
import numpy as np
import pytest
import torch
import yaml
import env.mavuav as module
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv as Env, RED_IDS, BLUE_IDS, ENTITY_IDS
from env.vector_env import _environment_state, _restore_environment_state
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.evaluate_happo import validate_checkpoint_contract

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return module.load_environment_config(ROOT / 'configs/env_v314.yaml')


def scene():
    e = Env(cfg(), randomize=False)
    e.reset(seed=1)
    for aid in ENTITY_IDS:
        s = e.entities[aid].state
        s.x, s.y, s.h, s.psi, s.theta, s.v = 20000., 0., 6000., 0., 0., 250.
        s.alive = aid in ('MAV', 'UAV1', 'Blue1', 'Blue2', 'Blue4')
    for aid, x in [('MAV', -15000.), ('UAV1', 0.), ('Blue1', 2000.), ('Blue2', 2500.)]:
        e.entities[aid].state.x = x
    return e


def test_version_config_and_only_curriculum_training_difference():
    c = cfg()
    assert c['environment_version'] == 'heterogeneous_mavuav_4v4_v3_14'
    assert c['simulation'] == dict(decision_dt=1., physics_dt=.1, max_decision_steps=150)
    assert c['combat'] == dict(distance=(0., 5000.), ata_deg=30., aa_deg=90., hold_steps=1, mav_can_attack=False, weapon_engagement_mode='single_target_lock')
    old = module.load_environment_config(ROOT / 'configs/env_v313.yaml')
    for key in ('battlefield', 'sensing', 'normalization', 'blue_policy'):
        assert c[key] == old[key]
    for aid in ENTITY_IDS:
        assert c['scenario']['initial'][aid]['speed'] == 250.
        for key in ('position', 'heading_deg'):
            assert c['scenario']['initial'][aid][key] == old['scenario']['initial'][aid][key]
    for kind in ('MAV', 'UAV', 'Blue'):
        assert (c['aircraft_specs'][kind]['v_min'], c['aircraft_specs'][kind]['v_max']) == (150., 300.)
        for key in ('nx', 'ny', 'nz'):
            assert c['aircraft_specs'][kind][key] == old['aircraft_specs'][kind][key]
    assert all(v == 0. for p in c['randomization_profiles'].values() for v in p.values())
    a = yaml.safe_load((ROOT / 'configs/happo_v313_baseline.yaml').read_text())
    b = yaml.safe_load((ROOT / 'configs/happo_v314_baseline.yaml').read_text())
    assert b['training']['randomization_curriculum_enabled'] is False
    b['training']['randomization_curriculum_enabled'] = True
    assert a == b
    assert 'role_reward' not in c and 'shaping' not in c
    assert c['reward'] == dict(blue_kill=10., uav_loss=-10., mav_loss=-10., terminal_red_win=20., terminal_blue_win=-20., terminal_draw=0.)


def test_reset_dimensions_and_deterministic_initialization():
    e = Env(cfg())
    a, _ = e.reset(seed=1)
    b, _ = e.reset(seed=999)
    for aid in RED_IDS:
        np.testing.assert_array_equal(a[aid], b[aid])
        assert a[aid].shape == (100,)
    assert e.global_state().shape == (117,)
    assert e.blue_policy.TARGET_STRATEGY == 'coordinated_assignment'
    assert all(e.entities[aid].state.v == 250. for aid in ENTITY_IDS)


@pytest.mark.parametrize('distance,ata,aa,inside', [
    (0., 0., 0., False), (.01, 0., 0., True), (5000., 0., 0., True),
    (5000.01, 0., 0., False), (2000., 30., 0., False), (2000., 0., 90., False),
])
def test_gate_limits(distance, ata, aa, inside, monkeypatch):
    e = scene()
    monkeypatch.setattr(module, 'compute_pairwise_geometry', lambda a, b: SimpleNamespace(distance=distance, ata=np.deg2rad(ata), aa=np.deg2rad(aa)))
    assert (e._weapon_gate_geometry('UAV1', 'Blue1') is not None) == inside


def test_real_geometry_one_boundary_single_candidate_no_reacquire():
    e = scene()
    events, deaths = e._resolve_attacks()
    assert [v for v in events if v['attacker'] == 'UAV1'] == [{'attacker': 'UAV1', 'target': 'Blue1'}]
    assert deaths == {'Blue1': 'red_attack'}
    assert e.entities['Blue2'].state.alive
    assert e.weapon_lock_target['UAV1'] is None
    assert all(v == 0 for (a, b), v in e._attack_streak.items() if a == 'UAV1')
    events, _ = e._resolve_attacks()
    assert {'attacker': 'UAV1', 'target': 'Blue2'} in events
    assert not any(v['attacker'] == 'MAV' for v in events)


@pytest.mark.parametrize('a,b', [((.1,.8,4000.),(.2,.1,1000.)), ((.1,.1,4000.),(.1,.2,1000.)), ((.1,.1,1000.),(.1,.1,4000.)), ((.1,.1,2000.),(.1,.1,2000.))])
def test_acquisition_priority(a, b, monkeypatch):
    e = scene()
    g = {'Blue1': a, 'Blue2': b}
    monkeypatch.setattr(e, '_weapon_gate_geometry', lambda aid, bid: SimpleNamespace(ata=g[bid][0], aa=g[bid][1], distance=g[bid][2]) if bid in g else None)
    assert e._acquire_weapon_lock('UAV1', BLUE_IDS) == 'Blue1'


@pytest.mark.parametrize('distance,ata,aa,reward', [(5001.,0.,180.,.01), (5000.,0.,180.,.02), (2000.,30.,0.,0.), (2000.,29.,100.,.02)])
def test_guide_distance_angles_and_AA_ignored(distance, ata, aa, reward, monkeypatch):
    e = scene()
    monkeypatch.setattr(e, 'team_visible', lambda bid: bid == 'Blue1')
    monkeypatch.setattr(module, 'compute_pairwise_geometry', lambda a, b: SimpleNamespace(distance=distance, ata=np.deg2rad(ata), aa=np.deg2rad(aa)))
    r = e._clean_guide_rewards()
    assert r['UAV1'] == dict(target='Blue1', reward=reward)
    assert r['UAV2'] == dict(target=None, reward=0.)
    assert 'MAV' not in r


def test_guide_nearest_tie_and_visibility(monkeypatch):
    e = scene()
    e.entities['Blue2'].state.x = 2000.
    monkeypatch.setattr(e, 'team_visible', lambda bid: bid in ('Blue1', 'Blue2'))
    assert e._clean_guide_rewards()['UAV1']['target'] == 'Blue1'
    e.entities['Blue2'].state.x = 1500.
    assert e._clean_guide_rewards()['UAV1']['target'] == 'Blue2'
    monkeypatch.setattr(e, 'team_visible', lambda bid: False)
    assert all(v['reward'] == 0. and v['target'] is None for v in e._clean_guide_rewards().values())


def test_dead_closer_blue_excluded_and_dead_UAV_zero(monkeypatch):
    e = scene()
    monkeypatch.setattr(e, 'team_visible', lambda bid: True)
    e.entities['Blue1'].state.alive = False
    assert e._clean_guide_rewards()['UAV1']['target'] == 'Blue2'
    e.entities['UAV1'].state.alive = False
    assert e._clean_guide_rewards()['UAV1'] == dict(target=None, reward=0.)


def test_synchronous_mutual_kill_and_different_attackers(monkeypatch):
    e = scene()
    e.entities['UAV2'].state.alive = True
    pairs = {('UAV1', 'Blue1'), ('Blue1', 'UAV1'), ('UAV2', 'Blue2')}
    monkeypatch.setattr(e, '_weapon_gate_geometry', lambda a, b: SimpleNamespace(ata=0., aa=0., distance=2000.) if (a,b) in pairs else None)
    events, deaths = e._resolve_attacks()
    assert {(v['attacker'], v['target']) for v in events} == pairs
    assert set(deaths) == {'UAV1', 'Blue1', 'Blue2'}
    assert all(v == 0 for v in e._attack_streak.values())
    assert all(e.weapon_lock_target[a] is None for a in ('UAV1','UAV2','Blue1'))


def test_new_clean_section_rejected_by_legacy_and_old_role_rejected_by_clean():
    old = module.load_environment_config(ROOT / 'configs/env_v313.yaml')
    old['clean_reward'] = cfg()['clean_reward']
    with pytest.raises(ValueError): module.validate_config(old)
    new = cfg()
    new['role_reward'] = old['role_reward']
    with pytest.raises(ValueError): module.validate_config(new)


def test_friendly_proximity_has_no_penalty(monkeypatch):
    e = Env(cfg()); e.reset(seed=1)
    monkeypatch.setattr(e, '_minimum_friendly_red_distance', lambda: 1.)
    monkeypatch.setattr(e, '_resolve_attacks', lambda: ([], {}))
    _, rewards, _, _, info = e.step(np.zeros((4,3)))
    assert info['red_safe_distance_violation'] and info['safety_reward'] == 0.
    assert set(rewards.values()) == {info['team_guide_reward']}


def test_reward_precombat_fixed_denominator_no_legacy_formula(monkeypatch):
    e = scene()
    monkeypatch.setattr(e.blue_policy, 'action', lambda *args: np.zeros(3))
    monkeypatch.setattr(e, '_team_situation_reward', lambda: pytest.fail('legacy situation called'))
    monkeypatch.setattr(e, '_role_process_rewards', lambda: pytest.fail('legacy role reward called'))
    _, rewards, _, _, info = e.step(np.zeros((4, 3)))
    assert info['uav1_guide_target'] == 'Blue1'  # computed BEFORE Blue1 dies
    assert info['uav1_guide_reward'] == .02
    assert info['team_guide_reward'] == .02 / 3
    assert info['event_reward'] == 10.
    assert len(set(rewards.values())) == 1
    assert rewards['MAV'] == 10. + .02 / 3
    assert info['safety_reward'] == info['absolute_situation'] == info['potential_shaping_reward'] == 0.
    s = _environment_state(e)
    other = scene()
    _restore_environment_state(other, s)
    assert other._clean_guide_sum == e._clean_guide_sum


@pytest.mark.parametrize('aid,cause,event,terminal', [('UAV1','boundary',-10.,0.), ('UAV2','blue_attack',-10.,0.), ('MAV','boundary',-10.,-20.), ('MAV','blue_attack',-10.,-20.)])
def test_loss_once_shared_reward(aid, cause, event, terminal, monkeypatch):
    e = Env(cfg()); e.reset(seed=1)
    monkeypatch.setattr(e.blue_policy, 'action', lambda *args: np.zeros(3))
    monkeypatch.setattr(e, '_clean_guide_rewards', lambda: {u: dict(target=None,reward=0.) for u in RED_IDS[1:]})
    def die():
        d = {}; e._deactivate(aid, cause, d); return d
    monkeypatch.setattr(e, '_apply_boundaries', die if cause == 'boundary' else lambda: {})
    monkeypatch.setattr(e, '_resolve_attacks', (lambda: ([], die())) if cause == 'blue_attack' else lambda: ([], {}))
    _, rewards, _, _, info = e.step(np.zeros((4, 3)))
    assert info['event_reward'] == event and info['terminal_reward'] == terminal
    assert set(rewards.values()) == {event + terminal}


def test_timeout_at_150_and_draw_reward(monkeypatch):
    e = Env(cfg()); e.reset(seed=1)
    monkeypatch.setattr(e.blue_policy, 'action', lambda *args: np.zeros(3))
    monkeypatch.setattr(e, '_resolve_attacks', lambda: ([], {}))
    e.step_count = 149
    _, _, term, trunc, info = e.step(np.zeros((4, 3)))
    assert not term and trunc and info['outcome'] == 'draw'
    assert info['terminal_reward'] == 0.
    assert info['episode_summary']['episode_length'] == 150


def test_red_terminal_and_no_sensor_weapon_dependency(monkeypatch):
    e = scene()
    for bid in BLUE_IDS[1:]: e.entities[bid].state.alive = False
    e._red_attack_kills.update(BLUE_IDS[1:])
    monkeypatch.setattr(e, 'team_visible', lambda bid: False)
    monkeypatch.setattr(e.blue_policy, 'action', lambda *args: np.zeros(3))
    _, rewards, term, trunc, info = e.step(np.zeros((4, 3)))
    assert term and not trunc and info['outcome'] == 'red'
    assert info['terminal_reward'] == 20. and info['event_reward'] == 10.
    assert set(rewards.values()) == {30.}


@pytest.mark.parametrize('bad', [{'method_variant':'rgaa'}, {'actor_variant':'pcta'}, {'actor_variant':'recurrent'}, {'critic_variant':'relational'}])
def test_only_vanilla_baseline_allowed(bad):
    with pytest.raises(ValueError): HAPPOTrainer(cfg(), dict(device='cpu', num_envs=1, **bad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke only')
def test_cuda_16env_exact_resume_and_standalone_eval(tmp_path, monkeypatch):
    training = yaml.safe_load((ROOT / 'configs/happo_v314_baseline.yaml').read_text())['training']
    training.update(rollout_steps=1, hidden_dim=8, ppo_epochs=1, minibatch_size=16)
    a = HAPPOTrainer(cfg(), training); b = None
    try:
        assert a.device.type == 'cuda' and a.vector_env.parallel
        assert len(set(a.vector_env.worker_pids)) == 16 and os.getpid() not in a.vector_env.worker_pids
        a.collect_rollout(); metrics = a.update()
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (float, int)))
        path = tmp_path / 'checkpoint_final.pt'; a.save_checkpoint(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert payload['sampled_steps'] == 16 and payload['reward_mode'] == 'clean_combat_v1'
        assert payload['reward_shaping_mode'] is None
        validate_checkpoint_contract(payload, cfg())
        old = module.load_environment_config(ROOT / 'configs/env_v313.yaml')
        with pytest.raises(RuntimeError): validate_checkpoint_contract(payload, old)
        bad = deepcopy(payload); bad['environment_version'] = old['environment_version']
        with pytest.raises(RuntimeError): validate_checkpoint_contract(bad, cfg())
        badpath = tmp_path / 'bad.pt'; torch.save(bad, badpath)
        with pytest.raises(RuntimeError): a.load_checkpoint(badpath)
        with pytest.raises(RuntimeError): a.load(badpath)
        b = HAPPOTrainer(cfg(), training); assert b.load_checkpoint(path) == 16
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        a.collect_rollout(); ma = a.update()
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda)
        b.collect_rollout(); mb = b.update()
        assert ma == mb
        for field in ('actions', 'values', 'returns', 'advantages', 'rewards'):
            np.testing.assert_array_equal(getattr(a.buffer, field), getattr(b.buffer, field))
            assert np.all(np.isfinite(getattr(a.buffer, field)))
        for left, right in ((a.actors, b.actors), (a.critic, b.critic)):
            for key, value in left.state_dict().items(): assert torch.equal(value, right.state_dict()[key])
        from algorithm import evaluate_happo
        monkeypatch.setattr(sys, 'argv', ['evaluate_happo', str(path), '--episodes', '1', '--device', 'cuda', '--action-mode', 'stochastic'])
        evaluate_happo.main()
        result = json.loads((tmp_path / 'evaluation_final_stochastic_summary.json').read_text())
        assert result['reward_mode'] == 'clean_combat_v1' and result['environment_version'].endswith('v3_14')
    finally:
        a.close()
        if b is not None: b.close()
