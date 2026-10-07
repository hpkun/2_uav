"""Audit-only regression: exact replay protocol, denominators and config fence."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tools import audit_v312_cap_mechanisms as audit
from algorithm.happo.evaluation import evaluate_actors
from env.mavuav import load_environment_config


def test_environment_only_two_explicit_differences():
    a=load_environment_config(audit.ROOT/'configs/env_v311.yaml')
    b=load_environment_config(audit.ROOT/'configs/env_v312.yaml')
    audit.compare_environments(a,b)
    before=deepcopy(b)
    b['combat']['hold_steps']+=1
    with pytest.raises(ValueError,match='identical'): audit.compare_environments(a,b)
    assert before['combat']['hold_steps']==3


def test_switch_not_inferred_to_reset_streak():
    rows=[]
    for step,target,sw in [(1,'UAV1',0),(2,'UAV1',0),(3,'UAV2',1),(4,'UAV2',0)]:
        rows.append(dict(environment_seed=1000,blue='Blue1',step=step,target=target,target_switch=sw,
                         old_distance_pre=2000,old_full_gate_pre=1,old_streak_pre=1 if sw else 0,old_streak_evaluated=2 if sw else 0,old_target_alive_pre=1))
    s=audit.switching(rows)
    assert s['target_switch_rate']==1/3
    assert s['mean_target_dwell_steps']==2
    assert s['switch_positive_old_streak_lost']==0
    assert s['switch_positive_old_streak_retained']==1


def test_contribution_real_event_identity_and_simultaneous_targets():
    e=[dict(attacker='UAV1',target='Blue1'),dict(attacker='UAV2',target='Blue1'),dict(attacker='UAV1',target='Blue2')]
    k=audit.kill_sets(e)
    assert len(k['UAV1'])==2 and len(k['UAV2'])==1
    # Credits can exceed unique team deaths; no arbitrary attribution.
    assert sum(map(len,k.values()))==3


def test_training_phase_uses_completed_counter_differences(monkeypatch,tmp_path):
    metrics=['mean_episode_return','red_win_rate','blue_win_rate','draw_rate','MAV_survival_rate','mean_UAV_survivors',
             'mean_red_attack_kills','mean_blue_attack_kills','mean_episode_length','entropy','critic_loss']+[f'actor_{i}_loss' for i in range(4)]
    rows=[dict(sampled_steps='2048',completed_episodes='10',**{m:'1' for m in metrics}),
          dict(sampled_steps='4096',completed_episodes='30',**{m:'4' for m in metrics})]
    monkeypatch.setattr(audit,'read_csv',lambda _:rows)
    r=audit.training_phases(tmp_path,'test')[0]
    assert r['completed_episodes']==30 and r['red_win_rate']==3


@pytest.mark.parametrize('version',['v311','v312'])
def test_cuda_replay_matches_official_evaluator_without_rng_parameter_mutation(version):
    if not torch.cuda.is_available(): pytest.skip('real CUDA audit smoke requires CUDA')
    run='happo_v311_seed1_2m' if version=='v311' else 'happo_v312_cap_seed1_2m'
    path=audit.ROOT/'outputs'/run/'checkpoint_final.pt'
    if not path.exists(): pytest.skip('local completed checkpoint required')
    digest=audit.sha(path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        actors=audit.IndependentActors(hidden_dim=128).cuda().eval(); actors.load_state_dict(payload['actors'])
        params={k:v.clone() for k,v in actors.state_dict().items()}
        cpu_rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state().clone()
        ep=audit.replay(actors,payload['environment_config'],1000,2000)
        assert torch.equal(torch.get_rng_state(),cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(),cuda_rng)
        official=evaluate_actors(actors,payload['environment_config'],1,'main',1000,'cuda',deterministic=False,action_seed=2000)[0]
        assert ep['result']==official
        assert all(torch.equal(v,actors.state_dict()[k]) for k,v in params.items())
        assert not any(r['agent']=='MAV' for r in ep['pairs'])
        assert all(np.isfinite(r['distance_m']) for r in ep['pairs'])
        assert all(r['streak_evaluated']>=3 for r in ep['pairs'] if r['kill_event'])
    assert audit.sha(path)==digest
