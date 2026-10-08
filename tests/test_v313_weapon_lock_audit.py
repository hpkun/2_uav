"""Focused read-only audit accounting/timestamp regressions (no training)."""
from copy import deepcopy
import numpy as np
from tools.audit_v313_weapon_lock import (CombatObserver, UAVS, alignment,
    conditional_streaks, last_kill, tables, validate_formal_replay)
from env.mavuav import HeterogeneousMAVUAVAirCombatEnv


def row(agent='UAV1',lock='Blue1',streak=1,reward='Blue1',kills=()):
    return dict(agent=agent,alive=True,pre_combat_alive=True,weapon_lock_target=lock,
        weapon_lock_streak=streak,reward_target=reward,evaluated_lock_target=lock,
        evaluated_streak=streak,kills=list(kills),full_gate_targets=['Blue1'],
        geometry={'Blue1':dict(distance=2000,ATA_deg=0,AA_deg=0,full_gate=True)})


def episode(rows):
    return dict(episode=0,environment_seed=1000,events=[],kill_times=[],
        summary=dict(red_attack_kills=0,outcome='draw'),
        steps=[dict(step=j+1,rows=[r]+[row(a,None,0,None) for a in UAVS[1:]],
            alive_blue_pre=['Blue1']) for j,r in enumerate(rows)])


def test_alignment_streaks_and_terminal_censoring():
    ep=episode([row(streak=1),row(streak=2,reward='Blue2')])
    result=alignment([ep])[-1]
    assert result['streak_ge1_match_rate']==.5
    assert result['streak_ge2_match_rate']==0
    assert result['conflict_episodes']==1
    assert result['reward_switched_away_during_lock']==1
    buckets=conditional_streaks([ep])
    assert buckets[0]['p_streak_increases']==1
    assert buckets[3]['terminal_censored']==1
    assert buckets[3]['p_streak_reset'] is None


def test_successful_kill_cleanup_is_not_streak_reset():
    killed=row(lock=None,streak=0,reward='Blue2',kills=['Blue1'])
    killed.update(evaluated_lock_target='Blue1',evaluated_streak=3)
    ep=episode([row(streak=2),killed])
    bucket=conditional_streaks([ep])[2]
    assert bucket['p_streak_increases']==1
    assert bucket['p_streak_reset']==0
    assert bucket['p_own_same_target_kill_next1']==1


def test_candidate_credit_is_not_unique_kill():
    ep=episode([row()]); ep['summary']['red_attack_kills']=1
    ep['events']=[dict(attacker=a,target='Blue1',step=1) for a in UAVS[:2]]
    ep['kill_times']=[1]
    result,_=tables([ep],75)
    assert result['total_unique_team_kills']==1
    assert result['total_candidate_credits']==2
    assert result['attacker_participation']['2']['count']==1
    assert result['uav_contributions']['UAV1']['contribution_share']==.5


def test_endgame_classification_and_overlay_not_exclusive():
    ep=episode([row(streak=1),row(streak=2)])
    ep['kill_times']=[1,1,1]; ep['summary']['red_attack_kills']=3
    rows,trace=last_kill([ep],75)
    assert rows[0]['category']=='D_max_streak2'
    assert rows[0]['F_horizon_forming_attack_overlay']
    assert rows[0]['remaining_horizon_steps']==74
    assert len(trace)==3


def test_no_locked_samples_is_missing_not_zero():
    result=alignment([episode([row(lock=None,streak=0,reward=None)])])[-1]
    assert result['lock_exists_n']==0
    assert result['lock_exists_match_rate'] is None


def test_observer_does_not_change_state_or_result():
    a=HeterogeneousMAVUAVAirCombatEnv('configs/env_v313.yaml',profile='main')
    b=HeterogeneousMAVUAVAirCombatEnv('configs/env_v313.yaml',profile='main')
    a.reset(seed=1000); b.reset(seed=1000); obs=CombatObserver(b)
    for _ in range(5):
        ra=a.step(np.zeros((4,3))); rb=b.step(np.zeros((4,3)))
        assert ra[1:4]==rb[1:4]
        assert ra[4]['attack_events']==rb[4]['attack_events']
        assert a.weapon_lock_target==b.weapon_lock_target
        assert a._attack_streak==b._attack_streak
        np.testing.assert_array_equal(a.global_state(),b.global_state())
        for key in ra[0]: np.testing.assert_array_equal(ra[0][key],rb[0][key])
    assert obs.evaluated is not None


def test_observer_captures_precleanup_killing_lock():
    env=HeterogeneousMAVUAVAirCombatEnv('configs/env_v313.yaml',randomize=False)
    env.reset(seed=1)
    for aid,entity in env.entities.items():
        s=entity.state
        s.alive=aid in ('MAV','UAV1','Blue1','Blue2')
        s.x={'MAV':-15000.,'UAV1':0.,'Blue1':2000.,'Blue2':2500.}.get(aid,20000.)
        s.y=0.; s.h=6000.; s.theta=0.; s.psi=0.
    observer=CombatObserver(env)
    for _ in range(3): env._resolve_attacks()
    assert observer.evaluated['UAV1']==dict(target='Blue1',streak=3)
    assert env.weapon_lock_target['UAV1'] is None
    assert not env.entities['Blue1'].state.alive


def test_reward_roundoff_tolerated_but_actual_outcome_change_rejected():
    keys=('red_win_rate','blue_win_rate','draw_rate','mean_episode_return','mean_red_attack_kills',
          'mean_blue_attack_kills','MAV_survival_rate','mean_UAV_survivors','mean_episode_length')
    formal={k:0. for k in keys}; formal['mean_episode_return']=239.24266853033348
    actual=dict(formal); actual['mean_episode_return']=239.24266852534078
    validation=validate_formal_replay(actual,formal)
    assert validation['passed']
    assert validation['metrics']['mean_episode_return']['difference']!=0
    actual['red_win_rate']=.005
    assert not validate_formal_replay(actual,formal)['passed']
    actual=dict(formal); actual['mean_episode_return']+=1e-4
    assert not validate_formal_replay(actual,formal)['passed']
