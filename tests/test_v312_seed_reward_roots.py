"""Audit-only regressions: time alignment, RNG invariance and tied quantiles."""
import pytest
import torch
import gzip
import json
import numpy as np
from tools import audit_v312_seed_reward_roots as audit


def test_atomic_cache_serializes_numpy_without_changing_values(tmp_path):
    path=tmp_path/'episode.json.gz'
    audit.write_gzip(path,[dict(array=np.array([1.,2.]),scalar=np.int64(3),missing_distance=float('inf'))])
    with gzip.open(path,'rt') as stream: assert json.load(stream)==[dict(array=[1.,2.],scalar=3,missing_distance=None)]
    assert not path.with_suffix('.gz.tmp').exists()


def test_early_geometry_uses_first_actual_gate_and_cap_conditional_counts():
    ep=dict(steps=[dict(environment_seed=1000)],pairs=[dict(agent=a,step=t,direct_visible=t>=2,distance_m=6000-1500*t,full_gate=t==3)
        for a in audit.UAVS for t in (1,2,3)],nav=[dict(blue='Blue1',step=t,target=a) for t,a in [(1,'UAV1'),(2,'UAV1'),(3,'UAV2')]])
    early,transitions=audit.early_geometry_and_cap({1:[ep]})
    assert len(early)==3
    assert all(r['first_direct_visibility']==2 and r['first_full_gate']==3 for r in early)
    assert sum(r['count'] for r in transitions)==2
    assert sum(r['conditional_probability'] for r in transitions)==1


def test_weapon_blocked_streak_is_not_geometrically_reconstructed():
    ep=dict(weapon='W1',events=[],pairs=[dict(agent='UAV1',full_gate=1,streak_evaluated=1,streak_stored_post=0)],
        result={**{f'{a.lower()}_process_reward_sum':0. for a in audit.UAVS},'event_reward_sum':0.,'terminal_reward_sum':0.})
    row=audit.contributions(ep)[0]
    assert row['gate_steps']==1 and row['streak1']==0


def test_future_window_strict_and_episode_censored():
    ep=dict(result=dict(episode_length=8),pairs=[dict(agent='UAV1',step=2,full_gate=1,streak_evaluated=1)],
            events=[dict(attacker='UAV1',target='Blue1',step=6)])
    assert audit.future_labels(ep,'UAV1',1,5)==dict(gate=1,streak=1,kill=1)
    assert audit.future_labels(ep,'UAV1',2,5)==dict(gate=0,streak=0,kill=1)
    assert audit.future_labels(ep,'UAV1',6,5) is None


def test_quintiles_keep_reward_ties_unsplit():
    rows=[dict(process=-.5) for _ in range(9)]+[dict(process=.2)]
    bins,edges=audit.quintile_groups(rows)
    assert len(bins[0])==9 and len(bins[4])==1
    assert sum(map(len,bins))==10


def test_cache_only_never_replays(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'CACHE_ONLY',True)
    with pytest.raises(FileNotFoundError,match='postprocess requires'): audit.cached_replays(None,None,tmp_path,'missing',20)


@pytest.mark.parametrize('mode',['W0','W1','W2'])
def test_cuda_enriched_replay_is_readonly_and_weapon_limits(mode):
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    path=audit.ROOT/'outputs/happo_v312_cap_seed1_2m/checkpoint_final.pt'
    if not path.exists(): pytest.skip('completed local model required')
    d=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        actors=audit.base.IndependentActors(hidden_dim=128).cuda().eval(); actors.load_state_dict(d['actors'])
        params={k:v.clone() for k,v in actors.state_dict().items()}; original=audit.base.Env
        cpu=torch.get_rng_state().clone(); gpu=torch.cuda.get_rng_state().clone()
        ep=audit.enriched_replay(actors,d['environment_config'],1000,2000,mode)
        assert audit.base.Env is original
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
        assert all(torch.equal(v,actors.state_dict()[k]) for k,v in params.items())
        if mode=='W0':
            official=audit.evaluate_actors(actors,d['environment_config'],1,'main',1000,'cuda',deterministic=False,action_seed=2000)[0]
            assert ep['result']==official
            tables=audit.reward_analyses({1:[ep]},d['environment_config'])
            own=[r for r in tables['reward_target_alignment'] if r['agent']=='UAV3'][0]
            assert own['own_kill_credits']==4 and own['kill_matches_post_selector']==0
            assert own['kill_matches_prior_selector']>0
        if mode=='W1':
            for a in audit.UAVS:
                times=sorted({e['step'] for e in ep['events'] if e['attacker']==a})
                assert all(b-a>5 for a,b in zip(times,times[1:]))
        if mode=='W2': assert max(len(v) for v in audit.base.kill_sets(ep['events']).values())<=2
