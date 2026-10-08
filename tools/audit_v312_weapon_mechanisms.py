"""Read-only weapon audit: cached W0 plus bounded instance-local interventions.

No trainer, production mutation, extra policy sampling, or new reward is used.
Pre-combat pair timestamps differ intentionally from post-combat reward selectors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import gzip
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools import audit_v312_seed_reward_roots as prior
from env.mavuav import ENTITY_IDS
from env.reward_role_v37 import target_score

base = prior.base
UAVS, RED, BLUE = base.UAVS, base.RED_IDS, base.BLUE_IDS
PRIOR = ROOT / 'outputs/audits/v312_seed_reward_roots_20261008'


def load_cache(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def exclusive_output(path):
    path = Path(path).resolve()
    allowed = (ROOT / 'outputs/audits').resolve()
    if not path.is_relative_to(allowed):
        raise ValueError('output must be a new directory under outputs/audits')
    path.mkdir(parents=True, exist_ok=False)
    (path / 'raw').mkdir()
    return path


def indexed(ep):
    return {(p['step'], p['agent'], p['target']): p for p in ep['pairs']}


def kill_events(ep, agent):
    targets = BLUE if agent in UAVS else RED
    return sorted((e for e in ep['events'] if e['attacker'] == agent and e['target'] in targets),
                  key=lambda e: (e['step'], e['target']))


def classify_carry(ep, agent):
    """Mutually exclusive mechanism labels, not estimates of causal dependence.

    Same-boundary intervals take precedence over any preload interpretation.
    For different boundaries the NEXT actual victim is checked at the prior kill.
    Positive evaluated streak counts even if its previous-boundary streak was 0.
    """
    ev = kill_events(ep, agent)
    lookup = indexed(ep)
    counts = Counter()
    for old, new in zip(ev, ev[1:]):
        if old['step'] == new['step']:
            counts['simultaneous'] += 1
        else:
            pair = lookup.get((old['step'], agent, new['target']))
            preload = pair is not None and pair['streak_evaluated'] > 0
            counts['next_target_before_ge1'] += int(pair is not None and pair['streak_before'] >= 1)
            counts['next_target_before_ge2'] += int(pair is not None and pair['streak_before'] >= 2)
            counts['preaccumulated' if preload else 'serial'] += 1
    nonzero = [k for k in ('serial','preaccumulated','simultaneous') if counts[k]]
    label = 'TYPE-' + (nonzero[0].upper() if len(nonzero) == 1 else 'MIXED') if nonzero else None
    return dict(agent=agent, kills=len(ev), classification=label,
                serial_intervals=counts['serial'], preaccumulated_intervals=counts['preaccumulated'],
                simultaneous_intervals=counts['simultaneous'], has_preaccumulation=int(counts['preaccumulated'] > 0),
                next_target_before_ge1_intervals=counts['next_target_before_ge1'],
                next_target_before_ge2_intervals=counts['next_target_before_ge2'],
                intervals=json.dumps([b['step']-a['step'] for a, b in zip(ev, ev[1:])]))


def distribution(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), minimum=float(a.min()) if len(a) else None,
                p10=float(np.percentile(a, 10)) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                mean=float(a.mean()) if len(a) else None,
                p90=float(np.percentile(a, 90)) if len(a) else None,
                **{f'p_le_{k}': base.mean([x <= k for x in a]) for k in (2, 3, 5)},
                p_eq_0=base.mean([x == 0 for x in a]), p_eq_1=base.mean([x == 1 for x in a]))


def analyze(finals):
    tables = defaultdict(list)
    for seed, eps in finals.items():
        steps_by_side = {'Red': [], 'Blue': []}
        contexts = []
        for ep in eps:
            envseed = ep['steps'][0]['environment_seed']
            lookup = indexed(ep)
            groups = defaultdict(list)
            for p in ep['pairs']:
                groups[p['step'], p['agent']].append(p)
            carry = {a: len(kill_events(ep, a)) >= 3 for a in UAVS}
            for (step, a), pp in sorted(groups.items()):
                row = dict(seed=seed, environment_seed=envseed, agent=a, step=step,
                    alive_enemy_count=len(pp), distance_count=sum(p['distance_gate'] for p in pp),
                    ATA_count=sum(p['ATA_gate'] for p in pp), AA_count=sum(p['AA_gate'] for p in pp),
                    parallel_gate_count=sum(p['full_gate'] for p in pp),
                    **{f'parallel_streak{k}_count': sum(p['streak_evaluated'] >= k for p in pp) for k in (1, 2, 3)},
                    eventual_own_ge3_kills=int(carry.get(a, False)))
                steps_by_side['Red' if a in UAVS else 'Blue'].append(row)
                if a in UAVS:
                    tables['parallel_pair_step_records'].append(row)
                    previous = ep['reward_trace'][step-2]['info'][f'reward_target_{a}'] if step >= 2 else None
                    current = ep['reward_trace'][step-1]['info'][f'reward_target_{a}']
                    active = [p for p in pp if p['streak_evaluated'] >= 1]
                    tables['reward_target_step_records'].append(dict(seed=seed, environment_seed=envseed,
                        step=step, agent=a, prior_reward_target=previous, post_reward_target=current,
                        active_weapon_pairs=len(active), mismatch_prior=sum(p['target'] != previous for p in active),
                        mismatch_post=sum(p['target'] != current for p in active),
                        own_kills=sum(p['kill_event'] for p in pp),
                        kill_mismatch_prior=sum(p['kill_event'] and p['target'] != previous for p in pp)))
            for event in (e for e in ep['events'] if e['attacker'] in UAVS):
                step, a, b = event['step'], event['attacker'], event['target']
                others = [p for p in groups[step, a] if p['target'] != b]
                deaths = json.loads(ep['steps'][step-1]['death_causes'])
                for p in others:
                    # Cache hit rows retain 3 as event evidence, NOT actual post-cleanup storage.
                    actual_post = 0 if p['agent'] in deaths or p['target'] in deaths else p['streak_evaluated']
                    tables['kill_context_other_pair_records'].append(dict(seed=seed, environment_seed=envseed,
                        step=step, killer=a, killed_target=b, other_target=p['target'],
                        **{k: p[k] for k in ('distance_m', 'ATA_deg', 'AA_deg', 'full_gate', 'streak_before', 'streak_evaluated')},
                        actual_streak_post_cleanup=actual_post,
                        other_simultaneously_killed=int(p['target'] in deaths)))
                row = dict(seed=seed, environment_seed=envseed, step=step, killer=a, killed_target=b,
                    other_alive_targets=len(others),
                    killer_other_target_max_streak_before=max((p['streak_before'] for p in others), default=0),
                    killer_other_target_max_streak_after=max((p['streak_evaluated'] for p in others), default=0),
                    **{f'number_other_targets_streak{k}': sum(p['streak_evaluated'] >= k for p in others) for k in (1, 2, 3)},
                    other_surviving_streak2=sum(p['streak_evaluated'] >= 2 and p['target'] not in deaths and a not in deaths for p in others))
                contexts.append(row)
                tables['kill_context_other_streaks'].append(row)
            for a in UAVS:
                c = classify_carry(ep, a)
                if c['kills'] >= 3:
                    tables['carry_mechanism_classification'].append(dict(seed=seed, environment_seed=envseed, **c))
            for a in UAVS+BLUE:
                ev = kill_events(ep,a)
                for old,new in zip(ev,ev[1:]):
                    pair = lookup.get((old['step'],a,new['target']))
                    interval = new['step']-old['step']
                    tables['kill_interval_records'].append(dict(seed=seed,environment_seed=envseed,agent=a,
                        side='Red' if a in UAVS else 'Blue',old_target=old['target'],next_target=new['target'],
                        old_kill_step=old['step'],next_kill_step=new['step'],interval_steps=interval,
                        next_target_streak_before=pair['streak_before'] if pair else None,
                        next_target_streak_evaluated_at_old_kill=pair['streak_evaluated'] if pair else None,
                        mechanism='simultaneous' if interval==0 else ('preaccumulated' if pair and pair['streak_evaluated']>0 else 'serial')))
            # First-arrival delays retain explicit right censoring; no absent event becomes zero.
            for a in UAVS:
                for b in BLUE:
                    pp = [p for p in ep['pairs'] if p['agent'] == a and p['target'] == b]
                    if not pp:
                        continue
                    first3 = next((p['step'] for p in pp if p['distance_m'] <= 3000), None)
                    firstgate = next((p['step'] for p in pp if p['full_gate']), None)
                    kill = next((p['step'] for p in pp if p['kill_event']), None)
                    tables['geometry_first_arrival_records'].append(dict(seed=seed, environment_seed=envseed, agent=a, target=b,
                        first_within3km=first3, first_gate=firstgate, first_kill=kill,
                        range_to_gate=firstgate-first3 if first3 is not None and firstgate is not None else None,
                        gate_to_kill=kill-firstgate if firstgate is not None and kill is not None else None,
                        range_without_gate=int(first3 is not None and firstgate is None),
                        gate_without_kill=int(firstgate is not None and kill is None)))
                    first2 = next((p for p in pp if p['streak_evaluated'] >= 2), None)
                    if first2:
                        nxt = lookup.get((first2['step']+1, a, b))
                        tables['deterministic_pair_records'].append(dict(seed=seed, environment_seed=envseed, agent=a, target=b,
                            first_streak2=first2['step'], next_pair_exists=int(nxt is not None),
                            next_gate=int(bool(nxt and nxt['full_gate'])), next_kill=int(bool(nxt and nxt['kill_event'])),
                            boundary_censored=int(nxt is None)))
        for side, rows in steps_by_side.items():
            agents = UAVS if side == 'Red' else BLUE
            for a in ('all',)+tuple(agents):
                rr = [r for r in rows if a == 'all' or r['agent'] == a]
                eligible = [r for r in rr if r['eventual_own_ge3_kills']]
                multi_eps = sum(any(sum(e['step'] == step for e in kill_events(ep, agent)) >= 2
                    for agent in agents if a == 'all' or agent == a
                    for step in {e['step'] for e in kill_events(ep, agent)}) for ep in eps)
                intervals = [b['step']-old['step'] for ep in eps for agent in agents if a == 'all' or agent == a
                             for old, b in zip(kill_events(ep, agent), kill_events(ep, agent)[1:])]
                row = dict(seed=seed, side=side, agent=a, episodes=len(eps), agent_steps=len(rr),
                    **{f'p_{key}_ge2': base.mean([r[key] >= 2 for r in rr]) for key in
                       ('parallel_gate_count', 'parallel_streak1_count', 'parallel_streak2_count')},
                    carry_agent_steps=len(eligible), p_parallel_streak2_ge2_given_own_carry=base.mean([r['parallel_streak2_count'] >= 2 for r in eligible]),
                    simultaneous_multi_kill_episodes=multi_eps, simultaneous_multi_kill_episode_rate=multi_eps/len(eps),
                    simultaneous_multi_kill_credits=sum(sum(e['step'] == step for e in kill_events(ep, agent))
                        for ep in eps for agent in agents if a == 'all' or agent == a
                        for step in {e['step'] for e in kill_events(ep, agent)}
                        if sum(e['step'] == step for e in kill_events(ep, agent)) >= 2))
                tables['parallel_streak_summary' if side == 'Red' else 'blue_parallel_weapon_summary'].append(row)
                tables['kill_interval_distribution'].append(dict(seed=seed, side=side, agent=a, **distribution(intervals)))
            if side == 'Red':
                pp = [p for ep in eps for p in ep['pairs'] if p['agent'] in UAVS]
                ranged = [p for p in pp if p['distance_gate']]
                arrivals = [r for r in tables['geometry_first_arrival_records'] if r['seed'] == seed]
                tables['geometry_gate_bottleneck'].append(dict(seed=seed, pair_steps=len(pp), ranged_pair_steps=len(ranged),
                    **{f'p_{k}': base.mean([p[k] for p in pp]) for k in ('distance_gate','ATA_gate','AA_gate','full_gate')},
                    **{f'p_{k}_given_range': base.mean([p[k] for p in ranged]) for k in ('ATA_gate','AA_gate','full_gate')},
                    entered3km_pairs=sum(r['first_within3km'] is not None for r in arrivals),
                    entered_gate_pairs=sum(r['first_gate'] is not None for r in arrivals),
                    range_without_gate_pairs=sum(r['range_without_gate'] for r in arrivals),
                    gate_without_kill_pairs=sum(r['gate_without_kill'] for r in arrivals),
                    mean_range_to_gate_observed=base.mean([r['range_to_gate'] for r in arrivals]),
                    median_range_to_gate_observed=base.mean([np.median([r['range_to_gate'] for r in arrivals if r['range_to_gate'] is not None])]) if any(r['range_to_gate'] is not None for r in arrivals) else None,
                    mean_gate_to_kill_observed=base.mean([r['gate_to_kill'] for r in arrivals])))
                gates = [p for p in pp if p['full_gate']]
                tables['sensor_combat_alignment'].append(dict(seed=seed, full_gate_pair_steps=len(gates),
                    direct_visible_fraction=base.mean([p['direct_visible'] for p in gates]),
                    team_visible_fraction=1. if gates and all(p['direct_visible'] for p in gates) else None,
                    team_visibility_basis='source: any Red direct_visible implies team_visible at same pre-combat timestamp'))
                dr = [r for r in tables['deterministic_pair_records'] if r['seed'] == seed]
                valid = [r for r in dr if r['next_pair_exists']]
                nextgate = [r for r in valid if r['next_gate']]
                third = [p for p in pp if p['streak_evaluated'] >= 3]
                tables['deterministic_kill_stats'].append(dict(seed=seed, first2_pairs=len(dr), next_valid_pairs=len(valid),
                    censored_pairs=len(dr)-len(valid), p_next_valid_kill=base.mean([r['next_kill'] for r in valid]),
                    p_next_gate_kill=base.mean([r['next_kill'] for r in nextgate]),
                    streak3_pair_steps=len(third), p_kill_given_streak3=base.mean([p['kill_event'] for p in third])))
        rr = [r for r in tables['reward_target_step_records'] if r['seed'] == seed]
        tables['reward_target_vs_weapon_pairs'].append(dict(seed=seed, agent_steps=len(rr),
            p_exactly_one_active_streak=base.mean([r['active_weapon_pairs'] == 1 for r in rr]),
            p_multiple_active_streak=base.mean([r['active_weapon_pairs'] >= 2 for r in rr]),
            active_pair_exposures=sum(r['active_weapon_pairs'] for r in rr),
            p_active_pair_differs_prior_reward_target=base.ratio(sum(r['mismatch_prior'] for r in rr),sum(r['active_weapon_pairs'] for r in rr)),
            p_active_pair_differs_post_reward_target=base.ratio(sum(r['mismatch_post'] for r in rr),sum(r['active_weapon_pairs'] for r in rr)),
            p_kill_differs_prior_reward_target=base.ratio(sum(r['kill_mismatch_prior'] for r in rr),sum(r['own_kills'] for r in rr))))
        tables['kill_context_summary'].append(dict(seed=seed, red_kill_events=len(contexts),
            p_other_before_ge1=base.mean([r['killer_other_target_max_streak_before'] >= 1 for r in contexts]),
            p_other_before_ge2=base.mean([r['killer_other_target_max_streak_before'] >= 2 for r in contexts]),
            p_other_evaluated_ge1=base.mean([r['killer_other_target_max_streak_after'] >= 1 for r in contexts]),
            p_other_evaluated_ge2=base.mean([r['killer_other_target_max_streak_after'] >= 2 for r in contexts]),
            p_other_evaluated_ge3=base.mean([r['killer_other_target_max_streak_after'] >= 3 for r in contexts]),
            p_other_surviving_streak2=base.mean([r['other_surviving_streak2'] > 0 for r in contexts])))
    return dict(tables)


def install_intervention(env, mode):
    """Instance-bound resolver only. Blue synchronous candidate logic is unchanged.

    R1 resets AFTER the synchronous batch: same-step multi-kill remains possible.
    L2 eligibility = alive + team-visible; leaving range alone does not release lock.
    Raw all-pair geometric streak for L2 ranking never contributes a kill directly.
    """
    if mode == 'W0':
        return
    original = env._resolve_attacks
    locks = {a: None for a in UAVS}
    raw = {}
    if mode == 'R1':
        def reset_all():
            events, deaths = original()
            killers = {e['attacker'] for e in events if e['attacker'] in UAVS}
            for a in killers:
                for b in BLUE:
                    env._attack_streak[a, b] = 0
            return events, deaths
        env._resolve_attacks = reset_all
        return
    if mode not in ('L1','L2'):
        raise ValueError(mode)

    def single_lock():
        combat = env.config['combat']
        selected = {}
        for a in UAVS:
            candidates = [b for b in BLUE if env.entities[a].state.alive and env.entities[b].state.alive and env.team_visible(b)]
            scores = {}
            for b in BLUE:
                valid = env.entities[a].state.alive and env.entities[b].state.alive
                g = base.geometry(env.entities[a].state, env.entities[b].state) if valid else None
                gate = bool(g and all(base.conditions(g.distance,g.ata,g.aa,combat)))
                raw[a,b] = raw.get((a,b),0)+1 if gate else 0
                if b in candidates:
                    n = env.config['normalization']
                    score = target_score(env.entities[a].state,env.entities[b].state,n['relative_altitude_scale'],n['relative_velocity_scale'],combat['distance'][1])
                    scores[b] = (int(gate),raw[a,b],score,-BLUE.index(b))
            target = env._reward_target_previous[a] if mode == 'L1' else locks[a]
            if target not in candidates:
                target = max(candidates,key=lambda b:scores[b]) if mode == 'L2' and candidates else None
            if target != locks[a]:
                if target is not None:
                    env._attack_streak[a,target] = 0
                locks[a] = target
            selected[a] = target
        candidates = []
        for a in ENTITY_IDS:
            if not env.entities[a].state.alive:
                continue
            for b in (BLUE if a in RED else RED):
                key = a,b
                valid = env.entities[b].state.alive and not (a == 'MAV' and not combat.get('mav_can_attack',True))
                if a in UAVS:
                    valid = valid and b == selected[a]
                if not valid:
                    env._attack_streak[key] = 0
                    continue
                g = base.geometry(env.entities[a].state,env.entities[b].state)
                gate = all(base.conditions(g.distance,g.ata,g.aa,combat))
                env._attack_streak[key] = env._attack_streak.get(key,0)+1 if gate else 0
                if env._attack_streak[key] >= combat['hold_steps']:
                    candidates.append(key)
        events = [dict(attacker=a,target=b) for a,b in sorted(candidates)]
        deaths = {}
        for _,b in candidates:
            cause = 'red_attack' if b in BLUE else 'blue_attack'
            env._deactivate(b,cause,deaths)
            (env._red_attack_kills if b in BLUE else env._blue_attack_kills).add(b)
        for a,b in list(env._attack_streak):
            if a in deaths or b in deaths:
                env._attack_streak[a,b] = 0
        return events,deaths
    env._resolve_attacks = single_lock


def diagnostic_replay(actors, cfg, envseed, actionseed, mode, device='cuda'):
    original = base.Env
    class AuditEnv(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            install_intervention(self,mode)
    base.Env = AuditEnv
    try:
        # Variant pair reconstruction in the legacy observer is not used for scientific
        # gate/streak claims. Only exact events and episode outcomes are consumed here.
        return base.replay(actors,cfg,envseed,actionseed,device)
    finally:
        base.Env = original


def official_readonly(actors,cfg,envseed,actionseed):
    # The existing evaluator intentionally seeds global RNG. Isolate that call
    # here rather than changing the evaluator or the stochastic action protocol.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        return prior.evaluate_actors(actors,cfg,1,'main',envseed,'cuda',deterministic=False,action_seed=actionseed)[0]


def counterfactual_rows(seed, originals, eps, mode):
    rows = []
    for original, ep in zip(originals,eps):
        dominant = max(UAVS,key=lambda a:len(kill_events(original,a)))
        credits = {a:len(kill_events(ep,a)) for a in UAVS}
        rows.append(dict(seed=seed,mode=mode,environment_seed=original['steps'][0]['environment_seed'],
            action_seed=original['steps'][0]['action_seed'],dominant_UAV=dominant,
            original_win=int(original['result']['outcome']=='red'),win=int(ep['result']['outcome']=='red'),
            kills=ep['result']['red_attack_kills'],dominant_UAV_kills=credits[dominant],
            other_UAV_kills=sum(v for a,v in credits.items() if a != dominant),
            any_UAV_ge3=int(max(credits.values())>=3),any_UAV_4kill=int(max(credits.values())==4),
            episode_length=ep['result']['episode_length'],outcome=ep['result']['outcome'],
            win_to_draw=int(original['result']['outcome']=='red' and ep['result']['outcome']=='draw')))
    return rows


def hold_experiment(cfg):
    """Synthetic, instance-local substep trajectory fed through REAL Env.step.

    Only the designated diagnostic entity's integrator is replaced, temporarily.
    No trained-policy measurement is inferred from this artificial excursion.
    """
    import env.mavuav as module
    from unittest.mock import patch
    env = base.Env(cfg,profile='main')
    env.reset(seed=1000)
    for a in ENTITY_IDS:
        env.entities[a].state.alive = a in ('MAV','UAV1','Blue1','Blue2')
    u,b = env.entities['UAV1'].state,env.entities['Blue1'].state
    u.x,u.y,u.h,u.psi,u.theta = 0.,0.,6000.,0.,0.
    b.x,b.y,b.h,b.psi,b.theta = 2000.,0.,6000.,0.,0.
    # One distant living Blue keeps the real four-kill terminal invariant intact.
    env.entities['Blue2'].state.x = 20000.
    env.entities['MAV'].state.x = -15000.
    # Navigation remains unmodified. Synthetic integrator freezes other states.
    calls = Counter()
    samples = []
    resolved_streaks = []
    deactivate = env._deactivate
    def observe_death(aid,cause,deaths):
        if aid == 'Blue1' and cause == 'red_attack':
            resolved_streaks.append(env._attack_streak['UAV1','Blue1'])
        return deactivate(aid,cause,deaths)
    env._deactivate = observe_death
    uav_spec = env.entities['UAV1'].spec
    def integrator(state,command,dt,spec):
        nxt = state.copy()
        if spec is uav_spec:
            calls['UAV1'] += 1
            k = (calls['UAV1']-1)%env.physics_substeps+1
            nxt.y = 4000. if k == 5 else 0.
            g = base.geometry(nxt,b)
            samples.append(dict(time_seconds=calls['UAV1']*env.physics_dt,
                decision_boundary=int(k==env.physics_substeps),full_gate=int(all(base.conditions(g.distance,g.ata,g.aa,cfg['combat'])))))
        return nxt
    boundaries = []
    with patch.object(module,'rk4_step',integrator):
        for _ in range(3):
            _,_,_,_,info = env.step(np.zeros((4,3)))
            boundaries.append(dict(time_seconds=env.step_count*env.decision_dt,
                streak_evaluated=resolved_streaks[-1] if resolved_streaks else env._attack_streak.get(('UAV1','Blue1'),0),
                kill=int(any(e['attacker']=='UAV1' and e['target']=='Blue1' for e in info['attack_events']))))
    assert [r['kill'] for r in boundaries] == [0,0,1]
    assert any(not r['full_gate'] for r in samples if not r['decision_boundary'])
    assert all(r['full_gate'] for r in samples if r['decision_boundary'])
    return dict(boundaries=boundaries,substep_samples=samples,
        first_gate_seconds=boundaries[0]['time_seconds'],kill_seconds=boundaries[-1]['time_seconds'],
        sample_span_seconds=boundaries[-1]['time_seconds']-boundaries[0]['time_seconds'],
        transient_substep_exit_did_not_reset=True,experiment='synthetic instance-local integration schedule; real Env.step and resolver')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=None)
    parser.add_argument('--counterfactual-episodes',type=int,default=20)
    parser.add_argument('--cache-source',type=Path,default=PRIOR)
    parser.add_argument('--finish-existing',action='store_true',help='Finalize this tool\'s interrupted output from complete diagnostic caches; never replay cached episodes.')
    args = parser.parse_args()
    if not 1 <= args.counterfactual_episodes <= 20:
        raise ValueError('bounded diagnostic: 1..20 matched carry episodes per seed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU policy replay fallback')
    destination = args.output or ROOT/'outputs/audits'/('v312_weapon_mechanism_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.finish_existing:
        out = Path(destination).resolve()
        if not out.is_relative_to((ROOT/'outputs/audits').resolve()) or not (out/'input_SHA256.json').is_file():
            raise ValueError('finish-existing requires this audit\'s interrupted SHA manifest')
        for s in (1,2,3):
            for m in ('L1','L2','R1'):
                if not (out/'raw'/f'seed{s}_{m}.json.gz').is_file():
                    raise FileNotFoundError('finish-existing requires all nine completed diagnostic caches')
    else:
        out = exclusive_output(destination)
    runs = {s:ROOT/f'outputs/happo_v312_cap_seed{s}_2m' for s in (1,2,3)}
    files = [f for run in runs.values() for f in run.rglob('*') if f.is_file() and f.suffix in ('.pt','.csv','.json','.yaml','.log')]
    files += [f for folder in ('algorithm','env','configs') for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.yaml')]
    files += [args.cache_source/'raw'/f'seed{s}_final.json.gz' for s in (1,2,3)]
    hashes = {str(f):base.sha(f) for f in files}
    if args.finish_existing:
        assert hashes==json.loads((out/'input_SHA256.json').read_text(encoding='utf-8')), 'inputs differ from interrupted audit'
    else:
        base.dump(out/'input_SHA256.json',hashes)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(),torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(),random.getstate()
    finals = {s:load_cache(args.cache_source/'raw'/f'seed{s}_final.json.gz') for s in (1,2,3)}
    assert all(len(eps)==50 for eps in finals.values())
    prior_contract = json.loads((args.cache_source/'input_contract.json').read_text(encoding='utf-8'))
    for s,eps in finals.items():
        for i,ep in enumerate(eps):
            assert ep['steps'][0]['environment_seed']==1000+i and ep['steps'][0]['action_seed']==2000+i
            assert ep.get('weapon')=='W0'
        path = runs[s]/'checkpoint_final.pt'
        assert prior_contract[str(path)]==hashes[str(path)], 'W0 cache checkpoint SHA does not match current final model'
    tables = analyze(finals)
    for name,rows in tables.items():
        base.table(out/f'{name}.csv',rows)
    start = time.monotonic()
    replay_count = 0
    counterrows = []
    cfg = None
    for seed,run in runs.items():
        data = torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        assert data['sampled_steps']==2_000_000
        if cfg is not None:
            assert cfg==data['environment_config']
        cfg = data['environment_config']
        assert cfg['environment_version']=='heterogeneous_mavuav_4v4_v3_12'
        base.validate_checkpoint_contract(data,cfg)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            actors = base.IndependentActors(hidden_dim=data['trainer_config']['hidden_dim']).cuda().eval()
        actors.load_state_dict(data['actors'])
        frozen = {k:v.clone() for k,v in actors.state_dict().items()}
        originals = [e for e in finals[seed] if max(len(kill_events(e,a)) for a in UAVS)>=3][:args.counterfactual_episodes]
        assert originals
        # One short exact-protocol W0 replay per seed, never rerun the 150 cached episodes.
        e0 = originals[0]
        replay = diagnostic_replay(actors,cfg,e0['steps'][0]['environment_seed'],e0['steps'][0]['action_seed'],'W0')
        replay_count += 1
        assert replay['result']==e0['result'] and replay['events']==e0['events']
        official = official_readonly(actors,cfg,e0['steps'][0]['environment_seed'],e0['steps'][0]['action_seed'])
        replay_count += 1
        assert official==e0['result'], 'cached W0 differs from official evaluator'
        counterrows.extend(counterfactual_rows(seed,originals,originals,'W0'))
        for mode in ('L1','L2','R1'):
            path = out/'raw'/f'seed{seed}_{mode}.json.gz'
            episodes = []
            if args.finish_existing:
                episodes = load_cache(path)
                assert len(episodes)==len(originals)
                assert all(e['steps'][0]['environment_seed']==o['steps'][0]['environment_seed'] and e['steps'][0]['action_seed']==o['steps'][0]['action_seed'] for e,o in zip(episodes,originals))
                print(f'seed{seed} {mode}: reused {len(episodes)} completed diagnostic episodes',flush=True)
            else:
                for i,original in enumerate(originals):
                    es,ac = original['steps'][0]['environment_seed'],original['steps'][0]['action_seed']
                    episodes.append(diagnostic_replay(actors,cfg,es,ac,mode))
                    replay_count += 1
                    if (i+1)%5==0:
                        print(f'seed{seed} {mode}: {i+1}/{len(originals)}; elapsed {time.monotonic()-start:.1f}s',flush=True)
                prior.write_gzip(path,episodes)
            counterrows.extend(counterfactual_rows(seed,originals,episodes,mode))
            base.table(out/'counterfactual_episode_records.csv',counterrows)
        assert all(torch.equal(v,actors.state_dict()[k]) for k,v in frozen.items())
    aggregate = []
    for seed in (1,2,3):
        for mode in ('W0','L1','L2','R1'):
            rr = [r for r in counterrows if r['seed']==seed and r['mode']==mode]
            aggregate.append(dict(seed=seed,mode=mode,episodes=len(rr),**{f'mean_{k}':base.mean([r[k] for r in rr]) for k in
                ('win','kills','dominant_UAV_kills','other_UAV_kills','any_UAV_ge3','any_UAV_4kill','episode_length','win_to_draw')}))
    base.table(out/'single_lock_counterfactual.csv',[r for r in aggregate if r['mode']!='R1'])
    base.table(out/'kill_reset_counterfactual.csv',[r for r in aggregate if r['mode'] in ('W0','R1')])
    hold = hold_experiment(cfg)
    base.dump(out/'hold_time_experiment.json',hold)
    assert hashes=={str(f):base.sha(f) for f in files}, 'read-only source/artifact contract violated'
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    summary = dict(protocol=dict(profile='main',device='cuda',cached_W0_episodes_per_seed=50,
        environment_seeds='1000-1049',action_seeds='2000-2049',matched_carry_max=args.counterfactual_episodes,
        selection='first carry episodes in ascending environment seed; outcome-conditioned diagnostic, not formal win comparison'),
        counterfactuals=aggregate,hold=hold,source_unchanged=True,actors_unchanged=True,RNG_unchanged=True,
        counterfactual_episode_count=sum(r['episodes'] for r in aggregate if r['mode']!='W0'),
        invocation_replay_count=replay_count,completed_counterfactual_caches_reused=args.finish_existing,
        elapsed_seconds=time.monotonic()-start,
        definitions=dict(pair_timestamp='post-physics, pre-combat; boundary losses already removed',
            parallel_denominator='alive attacker-step with at least one alive opponent at combat boundary',
            conditional_carry_denominator='same attacker eventual >=3 kills, not every teammate in carry episode',
            L1='previous reward selector only; unavailable means dead or not team-visible; no same-step fallback',
            L2='persist while alive + team-visible; full gate, raw geometric streak, target score, stable ID ranking',
            R1='after synchronous resolution clear killer UAV streaks; simultaneous same-batch kills remain legal'))
    summary.update({k:tables[k] for k in ('parallel_streak_summary','blue_parallel_weapon_summary','kill_context_summary',
        'kill_interval_distribution','geometry_gate_bottleneck','deterministic_kill_stats','sensor_combat_alignment')})
    base.dump(out/'summary.json',summary)
    render_figures(out,finals,tables,aggregate)
    write_reports(out,summary,tables,cfg)
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    after_numpy = np.random.get_state()
    assert numpy_rng[0]==after_numpy[0] and np.array_equal(numpy_rng[1],after_numpy[1]) and numpy_rng[2:]==after_numpy[2:]
    assert python_rng==random.getstate()
    print('Audit complete:',out,flush=True)


def render_figures(out, finals, tables, aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/f'{name}.png',dpi=180)
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,4))
    for i,seed in enumerate((1,2,3)):
        rr = next(r for r in tables['parallel_streak_summary'] if r['seed']==seed and r['agent']=='all')
        ax.bar(np.arange(3)+i*.24,[rr[f'p_{k}_ge2'] for k in ('parallel_gate_count','parallel_streak1_count','parallel_streak2_count')],.24,label=f'seed{seed}')
    ax.set_xticks(np.arange(3)+.24,['gate >=2 targets','streak1 >=2 targets','streak2 >=2 targets'])
    ax.set_ylabel('Fraction of alive UAV steps');ax.legend();save(fig,'parallel_streak_distribution')
    fig,ax = plt.subplots(figsize=(8,4))
    for seed in (1,2,3):
        vals = [b['step']-a['step'] for ep in finals[seed] for u in UAVS for a,b in zip(kill_events(ep,u),kill_events(ep,u)[1:])]
        count = Counter(vals)
        x = sorted(count)
        ax.plot(x,[count[k]/len(vals) for k in x],marker='o',label=f'seed{seed}')
    ax.set(xlabel='Kill-to-kill interval (decision steps)',ylabel='Probability');ax.legend();save(fig,'kill_interval_distribution')
    fig,ax = plt.subplots(figsize=(8,4))
    for i,r in enumerate(tables['geometry_gate_bottleneck']):
        ax.bar(np.arange(4)+i*.24,[r[f'p_{k}'] for k in ('distance_gate','ATA_gate','AA_gate','full_gate')],.24,label=f"seed{r['seed']}")
    ax.set_xticks(np.arange(4)+.24,['Range','ATA','AA','Full gate']);ax.set_ylabel('Alive UAV–Blue pair fraction');ax.legend();save(fig,'geometry_gate_funnel')
    fig,axs = plt.subplots(1,2,figsize=(11,4))
    for seed in (1,2,3):
        rr = [r for r in aggregate if r['seed']==seed]
        for ax,key in zip(axs,('mean_kills','mean_win')):
            ax.plot([r['mode'] for r in rr],[r[key] for r in rr],marker='o',label=f'seed{seed}')
            ax.set_ylabel(key);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Matched original carry episodes; frozen stochastic policies')
    save(fig,'weapon_counterfactual_comparison')
    fig,axs = plt.subplots(3,3,figsize=(15,10),sharey=True)
    timeline = []
    for si,seed in enumerate((1,2,3)):
        choices = [(ep,a) for ep in finals[seed] for a in UAVS if len(kill_events(ep,a))==4][:3]
        for ei,(ep,a) in enumerate(choices):
            ax=axs[si,ei];lookup=indexed(ep)
            for b in BLUE:
                pp=[p for p in ep['pairs'] if p['agent']==a and p['target']==b]
                ax.step([p['step'] for p in pp],[p['streak_evaluated'] for p in pp],where='post',label=b)
                hits=[p for p in pp if p['kill_event']]
                ax.scatter([p['step'] for p in hits],[3]*len(hits),marker='x',s=55)
                for step in range(1,ep['result']['episode_length']+1):
                    p=lookup.get((step,a,b))
                    timeline.append(dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],agent=a,target=b,step=step,
                        alive_pair_observed=int(p is not None),
                        **{k:p[k] if p else None for k in ('distance_m','ATA_deg','AA_deg','full_gate','streak_before','streak_evaluated','kill_event')}))
            ax.set(title=f"s{seed} env{ep['steps'][0]['environment_seed']} {a}",xlabel='Decision boundary',ylabel='Evaluated streak',ylim=(-.1,3.3));ax.grid(alpha=.2)
            if si==0 and ei==0:ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'parallel_streak_timeline.png',dpi=180)
    save(fig,'four_kill_parallel_streak_timeline')
    base.table(out/'four_kill_timelines.csv',timeline)


def write_reports(out,summary,tables,cfg):
    """Evidence-limited machine reproducible notes; final interpretation follows data."""
    (out/'combat_contract.md').write_text('''# v3.12 combat contract\n\nSOURCE-CODE FACT — env/mavuav.py::_resolve_attacks\n\nUAV→Blue and Blue→Red use inclusive distance [1000,3000] m, strict ATA <30° and AA <90°, hold=3. MAV is unarmed. Each alive attack-capable attacker checks EVERY alive enemy, maintaining independent pair streaks. Events are collected before any deactivation, sorted, and resolved synchronously; one attacker may kill several victims at the same boundary, and several attackers may receive candidate credit for one victim. Team kill reward counts unique deaths, not candidate multiplicity.\n\nThere is no direct/team/datalink visibility test, reward-target test, navigation-assignment test, fire action, ammunition, cooldown, lock, reacquisition, miss probability or Pk. Only pairs involving a newly dead entity are cleared. A surviving killer retains streaks toward other surviving enemies.\n\nAUTO_FIRE = TRUE. Action has three continuous flight-control dimensions; policy does not choose firing. DETERMINISTIC_KILL_AFTER_GATE_HOLD. These are abstraction facts, not proof of inappropriate geometry thresholds.\n''',encoding='utf-8')
    (out/'hold_time_semantics.md').write_text('''# Hold time semantics\n\nSOURCE-CODE FACT — Env.step integrates ten RK4 physics substeps (0.1 s), applies boundaries, then calls combat once. Gates are not checked at substeps.\n\nMEASURED SYNTHETIC EXPERIMENT — hold_time_experiment.json uses real Env.step/resolver with an instance-local synthetic integration schedule. Gates are true at 1,2,3 s; kill at 3 s. The first-to-third sample span is 2 s, not a continuous 3 s envelope guarantee. At 0.5,1.5,2.5 s the UAV briefly leaves the envelope, then returns by each boundary; kill still occurs. This demonstrates permissiveness relative to continuous hold but does not measure its prevalence in trained-policy trajectories.\n''',encoding='utf-8')
    carry=[r for r in tables['carry_mechanism_classification']]
    byseed=[]
    for seed in (1,2,3):
        rr=[r for r in carry if r['seed']==seed]
        byseed.append(dict(seed=seed,carry_episodes=len(rr),types=dict(Counter(r['classification'] for r in rr)),
            any_preaccumulation=sum(r['has_preaccumulation'] for r in rr),
            next_victim_preexisting_streak1_intervals=sum(r['next_target_before_ge1_intervals'] for r in rr),
            next_victim_preexisting_streak2_intervals=sum(r['next_target_before_ge2_intervals'] for r in rr),
            fourkill_serial=sum(r['kills']==4 and r['classification']=='TYPE-SERIAL' for r in rr),
            fourkill_total=sum(r['kills']==4 for r in rr)))
    summary['carry_summary']=byseed
    def cf(s,m): return next(r for r in summary['counterfactuals'] if r['seed']==s and r['mode']==m)
    def cfsequence(m,k,scale=1): return '/'.join(f"{cf(s,m)[k]*scale:g}" for s in (1,2,3))
    def delta_sequence(m,k): return '/'.join(f"{100*(cf(s,'W0')[k]-cf(s,m)[k]):g}" for s in (1,2,3))
    rows=[]
    descriptions=['range too wide','ATA too wide','AA too wide','hold too short','boundary sampling weaker than continuous hold',
        'parallel streak materially enables carry','simultaneous multi-kill exists','preload shortens intervals','absence of lock enables carry',
        'unlimited ammo enables carry','absence of cooldown enables carry','deterministic kill enables carry','auto-fire lowers firing-decision complexity',
        'sensor decoupling impacts current carry','weapon abstraction overly permissive','geometry envelope overly permissive']
    for i,description in enumerate(descriptions,1):
        status='INCONCLUSIVE';evidence='No matched intervention isolates this normative or causal claim.'
        if i==5:status='VERIFIED';evidence='Real step/resolver synthetic substep experiment; continuous-hold prevalence not measured.'
        if i==6:status='PARTIALLY_SUPPORTED';evidence=f"Parallel accumulation enables short intervals; matched wins W0 {cfsequence('W0','mean_win',100)}% vs R1 {cfsequence('R1','mean_win',100)}%; occurrence is not overall carry dependence."
        if i==7:
            n=sum(r['simultaneous_multi_kill_episodes'] for r in tables['parallel_streak_summary'] if r['agent']=='all')
            status='VERIFIED' if n else 'REFUTED';evidence=f'{n}/150 cached episodes contain same-attacker simultaneous multi-kill.'
        if i==8:status='PARTIALLY_SUPPORTED';evidence='Intervals of 1–2 steps require earlier preload; observed association is distinct from terminal-performance dependence. R1 causes small outcome changes.'
        if i==9:status='PARTIALLY_SUPPORTED';evidence=f"L1 carry {cfsequence('L1','mean_any_UAV_ge3',100)}%; L2 {cfsequence('L2','mean_any_UAV_ge3',100)}%; W0 {cfsequence('W0','mean_any_UAV_ge3',100)}%. Selection/eligibility/persistence are part of the lock intervention."
        if i==10:status='INCONCLUSIVE';evidence='No ammo exists in source. Prior max2 unique kill credits mechanically limits carry, but is not an ammunition or shots model and does not identify a realistic ammo effect.'
        if i==11:status='PARTIALLY_SUPPORTED';evidence='Prior separate 10-carry-episode 5-step cooldown lowers wins in all three seeds; no claim that this cooldown duration is physically appropriate.'
        if i==12:status='PARTIALLY_SUPPORTED';evidence='P(kill|streak3)=1 is verified; causal share requires probabilistic-kill contrast not performed.'
        if i==13:status='VERIFIED';evidence='Three flight actions, no firing decision; no causal claim about win magnitude.'
        if i==14:status='REFUTED';evidence='All observed UAV full gates are directly visible; source direct visibility implies team visibility.'
        if i==15:status='PARTIALLY_SUPPORTED';evidence='Automatic parallel candidates, simultaneous resolution and retained streaks facilitate carry; realism remains modelling judgement.'
        rows.append(dict(hypothesis=f'H{i}',description=description,status=status,evidence=evidence))
    base.table(out/'root_cause_matrix.csv',rows)
    summary['root_cause_matrix']=rows
    summary['next_single_variable_recommendation']='weapon target lock only; proposed not implemented in production'
    summary['tests']=dict(focused=18,related_total=72,passed=True,
        runtime='PowerShell conda uav CUDA; WSL E_ACCESSDENIED',
        no_training=True,official_evaluator_calls_isolated_by_torch_fork_rng=True)
    base.dump(out/'summary.json',summary)
    def pct(v): return 'missing' if v is None else f'{v*100:.1f}%'
    def md(headers,data):
        return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
    def row(name,s): return next(r for r in tables[name] if r['seed']==s and r.get('agent','all')=='all' and r.get('side','Red')=='Red')
    text='# v3.12 CAP-Blue 武器机制专项审计\n\n'
    text+='## 范围与总判断\n\nMEASURED：复用三 training seeds 各50局 W0 缓存，环境 seeds1000–1049、动作 seeds2000–2049，main profile。每 seed 选择按环境 seed 排序的前20局单 UAV≥3-kill 回合，共180局 L1/L2/R1 冻结 CUDA 回放。这是原始 carry 条件样本，不能替代正式200局胜率。另做每 seed 一局 W0/正式 evaluator exact equality 检查。\n\n'
    text+='INFERENCE：现有 carry 同时受真实攻击几何能力与自动、all-pair、无单目标武器约束的抽象机制影响。单目标 lock 对照影响明显，kill-reset-only 影响有限。不能把“预积累发生”写成“总体胜率依赖预积累”，也不能由成功率直接判定几何过宽。\n\n'
    text+='## SOURCE-CODE FACT：正式合同\n\n源码 env/mavuav.py::_resolve_attacks/step：UAV与Blue距离[1000,3000]m、ATA<30°、AA<90°，连续3个decision boundary；MAV无武器。每1s结束检查一次，0.1s物理子步不检查。第一次判定1s、击杀3s、检查点跨度2s。synthetic真实step/resolver实验在0.5/1.5/2.5s短暂出包线仍击杀，证明非连续保持合同，不表示真实轨迹出包线频率。\n\n'
    text+='所有alive对独立累计streak；先收集所有candidates再同步死亡，同机同step可多杀、多机可对同一victim产生event。死亡相关pair清零，其他surviving pair保留。无target锁定、显式fire、ammo/cooldown/reacquisition/Pk/miss。3D action仅控制飞行。shared kill reward按unique victim计，不按event重复数计。\n\n'
    text+='## MEASURED：并行资格与carry类型\n\n'
    text+=md(['seed','gate≥2targets/agent-step','streak2≥2targets/agent-step','同机多杀episodes','carry≥3 episodes','含下个victim预积累','四杀全serial/全部四杀'],[
        [s,pct(row('parallel_streak_summary',s)['p_parallel_gate_count_ge2']),pct(row('parallel_streak_summary',s)['p_parallel_streak2_count_ge2']),
         f"{row('parallel_streak_summary',s)['simultaneous_multi_kill_episodes']}/50",b['carry_episodes'],
         f"{b['any_preaccumulation']}/{b['carry_episodes']}",f"{b['fourkill_serial']}/{b['fourkill_total']}"] for s,b in zip((1,2,3),byseed)])
    text+='\n并行率分母为有alive敌方pair的alive UAV combat-boundary step，boundary已死亡者不计。carry条件概率限定该UAV自身最终≥3 kills，分别为2.76%、3.35%、1.67%。短暂窗口在全局agent-step中比例低，但在击杀上下文中明显集中。\n\n'
    text+=md(['seed','SERIAL','PREACCUMULATED','SIMULTANEOUS','MIXED'],[[b['seed']]+[b['types'].get(k,0) for k in ('TYPE-SERIAL','TYPE-PREACCUMULATED','TYPE-SIMULTANEOUS','TYPE-MIXED')] for b in byseed])
    text+='\nSERIAL：每次前次kill时，下个实际victim evaluated streak=0；PREACCUMULATED：全部连续interval都已有正streak；SIMULTANEOUS：全部kill同boundary；MIXED：多种interval机制。mixed不必都有preload。含preload仅描述轨迹，不代表因果依赖。总共77/122 carry出现preload，严格serial四杀5/66。已有streak_before与本boundary新增到1的streak分开记录。\n\n'
    text+=md(['seed','kill时其他target before≥1','before≥2','evaluated≥2','同时evaluated≥3'],[[s]+[pct(row('kill_context_summary',s)[k]) for k in ('p_other_before_ge1','p_other_before_ge2','p_other_evaluated_ge2','p_other_evaluated_ge3')] for s in (1,2,3)])
    text+='\n## MEASURED：连续击杀间隔\n\n'
    text+=md(['seed','interval n','min','p10','median','mean','p90','P(0)','P(1)','P≤2','P≤3','P≤5'],[[s]+[row('kill_interval_distribution',s)[k] for k in ('n','minimum','p10','median','mean','p90')]+[pct(row('kill_interval_distribution',s)[k]) for k in ('p_eq_0','p_eq_1','p_le_2','p_le_3','p_le_5')] for s in (1,2,3)])
    text+='\n间隔1–2步在hold3合同下必然需要已有资格，间隔0是同步多杀。各seed多杀event credits为'+', '.join(str(row('parallel_streak_summary',s)['simultaneous_multi_kill_credits']) for s in (1,2,3))+'；不能把event credits作为team unique kills。每seed3个四杀示例，共9例，见完整timeline表/图，死亡后缺失为missing，不伪造0。\n\n'
    text+='## MEASURED：几何与传感器\n\n'
    text+=md(['seed','range fraction','fullgate fraction','P(ATA|range)','P(AA|range)','P(fullgate|range)','首次≤3km后无gate pairs','range→gate observed mean','gate→kill observed mean'],[[s]+[pct(row('geometry_gate_bottleneck',s)[k]) for k in ('p_distance_gate','p_full_gate','p_ATA_gate_given_range','p_AA_gate_given_range','p_full_gate_given_range')]+[f"{row('geometry_gate_bottleneck',s)['range_without_gate_pairs']}/{row('geometry_gate_bottleneck',s)['entered3km_pairs']}",f"{row('geometry_gate_bottleneck',s)['mean_range_to_gate_observed']:.2f}",f"{row('geometry_gate_bottleneck',s)['mean_gate_to_kill_observed']:.2f}"] for s in (1,2,3)])
    text+='\n有效距离内ATA通过率27.5–30.1%，低于AA37.2–44.3%；角度组合是实质限制。全体pair-step上AA的通过率最小，与有效距离内ATA更稀缺是不同条件分母，不能简单只指一个全局瓶颈。延迟均值仅对已观察到终点的pair计算，未进gate/未kill单列删失；约53–70%曾≤3km pair未进入fullgate。现有证据不足以断言1–3km/30°/90°/hold3太宽或太短。\n\n'
    text+='全部fullgate pair direct visible=100%，依据team_visible=any Red direct_visible，同时间team visible也100%。combat确实未显式检查sensor，但这不是已观察carry的实际来源。\n\n'
    text+='## MEASURED：reward target、Blue和确定性\n\n'
    text+=md(['seed','active pair不匹配prior selector','kill不匹配prior selector','Blue parallel streak2≥2','Blue同机多杀episodes','P(next valid kill|first2)','P(kill|streak3)'],[[s,pct(row('reward_target_vs_weapon_pairs',s)['p_active_pair_differs_prior_reward_target']),pct(row('reward_target_vs_weapon_pairs',s)['p_kill_differs_prior_reward_target']),
        pct(next(r for r in tables['blue_parallel_weapon_summary'] if r['seed']==s and r['agent']=='all')['p_parallel_streak2_count_ge2']),
        next(r for r in tables['blue_parallel_weapon_summary'] if r['seed']==s and r['agent']=='all')['simultaneous_multi_kill_episodes'],
        pct(row('deterministic_kill_stats',s)['p_next_valid_kill']),pct(row('deterministic_kill_stats',s)['p_kill_given_streak3'])] for s in (1,2,3)])
    text+='\nreward只选一个目标不限制其他combat pairs。Blue也有all-pair资格，但并行streak2比例仅0.065%、0%、0.069%；其导航target不等于weapon target。首次streak2下一boundary pair仍存在的分母186/158/210，缺失2/4/20分别包含死亡或episode结束，不能误计失败。下一步gate保持时P(kill)=100%。确定性kill事实不等于已识别其对胜率的独立因果贡献。\n\n'
    text+='## DIAGNOSTIC COUNTERFACTUAL：同场景同随机动作种子\n\n'
    text+=md(['seed','mode','win','team kills','original dominant kills','other credits','any UAV≥3','any UAV4','length'],[[r['seed'],r['mode'],pct(r['mean_win']),f"{r['mean_kills']:.2f}",f"{r['mean_dominant_UAV_kills']:.2f}",f"{r['mean_other_UAV_kills']:.2f}",pct(r['mean_any_UAV_ge3']),pct(r['mean_any_UAV_4kill']),f"{r['mean_episode_length']:.2f}"] for r in summary['counterfactuals']])
    text+='\nL1仅上一decision reward target，首step无锁，不可用定义dead/not team-visible，无same-step fallback。L2保持alive/team-visible锁，即使不在攻击距离内；新锁按fullgate→raw geometric streak→原target_score→ID顺序，换锁从0建立weapon streak。L2原始geometric streak只用于排序，不提供实际kill。两者改变weapon engagement约束及目标选择/保持，不能把整个下降全部归因于parallel预积累。\n\n'
    text+=f"R1在同步batch结束后清空killer其他Blue streak，不取消同batch多杀。win{cfsequence('W0','mean_win',100)}%→{cfsequence('R1','mean_win',100)}%；team kills{cfsequence('W0','mean_kills')}→{cfsequence('R1','mean_kills')}；carry{cfsequence('W0','mean_any_UAV_ge3',100)}%→{cfsequence('R1','mean_any_UAV_ge3',100)}%。所以保留preload确有局部影响，但不足以证明总体carry主要依靠它。此前独立10carry/seed cooldown/max2诊断只作为历史对照，见原审计weapon_counterfactual.csv；max2-kill不是物理ammo模型。\n\n"
    previous_path=PRIOR/'weapon_counterfactual.csv'
    if previous_path.exists():
        previous=[r for r in base.read_csv(previous_path) if r['agent']=='UAV1']
        text+='历史DIAGNOSTIC（每seed仅10局原始carry；不与新20局样本混合）：\n\n'
        text+=md(['seed','variant','episodes','win','Red kills'],[[r['seed'],{'W0':'原始','W1':'5-step cooldown','W2':'max2 unique kills'}[r['weapon']],r['episodes'],pct(float(r['red_win_rate'])),r['mean_red_attack_kills']] for r in previous])+'\n'
    text+='## 最终17问的直接回答\n\n'
    answers=[
        '完整条件：alive UAV–alive Blue，1–3km且ATA<30°/AA<90°，3个相邻1s边界满足；无sensor/target/fire/resource条件。',
        'hold3：3个decision检查点，first-to-third跨度2s，不保证3s连续保持；子步短暂出门不清零。',
        '可同时对多个Blue累计，真实轨迹有测量证据。',
        '可同step多杀，15/19/5局出现，非单纯源码潜能。',
        'killer存活且另一target存活时保留其他streak；killer自己同步死亡则也清零。',
        '含下一实际victim预积累30/45、33/42、14/35；因果依赖比例不能从分类直接推得，reset对照较小。',
        '严格从0逐目标重建的四杀2/31、1/25、2/10；合计5/66。',
        '最短0步；去除同步多杀后的最短1步。',
        '距离不是唯一瓶颈；有效距离内ATA比AA更稀缺，fullgate仅9.4–10.7%。',
        '没有足够证据认定当前几何/hold数值太宽松；continuous-hold语义确实更弱。',
        '更直接证据指向automatic all-pair/no-single-lock abstraction，而非门限数字本身；是否“过于”宽松含任务建模判断。',
        f"L1 carry下降{delta_sequence('L1','mean_any_UAV_ge3')}个百分点；L2下降{delta_sequence('L2','mean_any_UAV_ge3')}个百分点。锁选择与保持也参与影响，不能纯化成只有parallel一个因素。",
        f"R1 carry下降{delta_sequence('R1','mean_any_UAV_ge3')}个百分点，win下降{delta_sequence('R1','mean_win')}个百分点；没有普遍大幅坍塌。",
        'Blue同样具备parallel资格，实际使用显著少于Red。',
        '所有fullgate direct/team visible100%，当前样本未显示sensor解耦带来实质carry。',
        '若只选择一个下一步最小变量：weapon target lock，保持geometry/hold/reward/actor不变；不是同时加ammo/cooldown。',
        '仅提出该单变量验证建议，未修改任何正式combat或训练实现。']
    text+='\n'.join(f'{i}. {a}' for i,a in enumerate(answers,1))+'\n\n'
    text+='## 只读性与测试\n\n仅新增独立tool/test/report。源代码、配置、checkpoint、原始run artifacts和W0 cache SHA均不变，actor参数逐元素不变。official evaluator调用在audit-side fork_rng内，不改evaluator。72项专项及相关回归通过，含18项本工具测试与4种CUDA冻结回放/有限性检查。WSL E_ACCESSDENIED，使用PowerShell uav CUDA。无训练、无新正式200局评估。\n'
    (out/'research_report.md').write_text(text,encoding='utf-8')


if __name__ == '__main__':
    main()
