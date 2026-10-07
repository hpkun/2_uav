"""Read-only stage-three audit. Frozen actors; no trainer or production writes.

Sampling: pre-action navigation snapshot; post-physics/pre-combat pair geometry;
post-combat process rewards/support state. All timestamps are decision steps (1-based).
Counterfactuals are instance-local and explicitly diagnostic, not new environments.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithm.happo.evaluation import evaluate_actors, summarize_records
from env.blue_policy import BluePolicy
from tools.audit_v311_attack_geometry import conditions, boundary_margin, read_numeric_csv
from tools.audit_v311_happo_failure import (
    Env, RED_IDS, BLUE_IDS, geometry, table, dump, sha, IndependentActors,
    validate_checkpoint_contract, load_environment_config,
)
UAVS = RED_IDS[1:]
PRIOR = ROOT / 'outputs/audits/v311_happo_failure_audit_200ep_20261007_141809'


def mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def ratio(a, b):
    return a / b if b else None


def position(state):
    return np.array([state.x, state.y, state.h], dtype=float)


def distance(a, b):
    return float(np.linalg.norm(position(a) - position(b)))


def navigation_snapshot(env):
    """Diagnostics are pure: never call action/refresh to read cached navigation."""
    red = {a: env.entities[a] for a in RED_IDS}
    rows = []
    for bid in BLUE_IDS:
        blue = env.entities[bid]; st = blue.state
        d = env.blue_policy.diagnostics(blue, red, env.step_count)
        nearest = env.blue_policy.select_target(blue, red)
        row = dict(blue=bid, alive=int(st.alive), x=st.x, y=st.y, h=st.h,
                   speed=st.v, heading_deg=float(np.rad2deg(st.psi)),
                   cached_target=d['blue_target_id'],
                   instantaneous_nearest=nearest.aircraft_id if st.alive and nearest else None,
                   refresh_due=d['blue_guidance_refresh_due'], guidance_age=d['blue_guidance_age'],
                   desired_heading=d['blue_desired_heading'], desired_pitch=d['blue_desired_pitch'],
                   boundary_recovery_active=bool(d['blue_boundary_recovery_active'] or d['blue_horizontal_recovery_active']))
        for aid in RED_IDS:
            row[f'distance_{aid}_m'] = distance(st, red[aid].state) if st.alive and red[aid].state.alive else None
        rows.append(row)
    return rows


def concentration(targets):
    c = Counter(t for t in targets if t in RED_IDS)
    return dict(max_same=max(c.values(), default=0), **{f'pressure_{a}': c[a] for a in RED_IDS})


def support_snapshot(env, info):
    """Post-combat state matches the real process reward's state, not pre-action."""
    mav = env.entities['MAV'].state
    uu = [env.entities[a].state for a in UAVS if env.entities[a].state.alive]
    bb = [env.entities[b].state for b in BLUE_IDS if env.entities[b].state.alive]
    def centroid(states):
        return np.mean([position(s) for s in states], axis=0) if states else None
    row = dict(MAV_alive=int(mav.alive), alive_blue_post=len(bb),
               nearest_UAV_m=min((distance(mav, s) for s in uu), default=None),
               nearest_Blue_m=min((distance(mav, s) for s in bb), default=None))
    for label, ss in [('UAV', uu), ('Blue', bb), ('engagement', uu+bb)]:
        c = centroid(ss)
        row[f'MAV_{label}_centroid_m'] = float(np.linalg.norm(position(mav)-c)) if c is not None else None
        for i, ax in enumerate(('x', 'y', 'h')): row[f'{label}_centroid_{ax}'] = float(c[i]) if c is not None else None
    row['MAV_direct_count'] = sum(env.direct_visible('MAV', b) for b in BLUE_IDS)
    row['team_visible_count'] = sum(env.team_visible(b) for b in BLUE_IDS)
    for aid in UAVS:
        row[f'{aid}_datalink_only_count'] = sum(env.datalink_visible(aid, b) and not env.direct_visible(aid, b) for b in BLUE_IDS)
    row['UAV_datalink_only_total'] = sum(row[f'{a}_datalink_only_count'] for a in UAVS)
    row['MAV_exclusive_information_count'] = sum(env.direct_visible('MAV', b) and not any(env.direct_visible(a, b) for a in UAVS) for b in BLUE_IDS)
    for key in ('mav_process_reward', 'mav_R_threat', 'mav_R_aspect', 'mav_R_aware'):
        row[key] = float(info[key])
    # Independent literal check against actual saved components; no second reward call.
    assert np.isclose(row['mav_process_reward'], .3*row['mav_R_threat']+.2*row['mav_R_aspect']+.4*row['mav_R_aware'])
    return row


def kill_sets(events):
    return {a: {e['target'] for e in events if e['attacker'] == a and e['target'] in BLUE_IDS} for a in UAVS}


def dominance(counts):
    counts = list(counts); total = sum(counts)
    return dict(max_agent_kills=max(counts, default=0), attacking_agents=sum(c > 0 for c in counts),
                dominance_share=ratio(max(counts, default=0), total),
                kill_HHI=sum((c/total)**2 for c in counts) if total else None)


def contribution_gap(aid, target, killers, pair, info, config):
    return dict(agent=aid, target=target, killer=int(aid in killers), full_gate=pair['full_gate'],
                distance_m=pair['distance_m'], ATA_deg=pair['ATA_deg'], direct_visible=pair['direct_visible'],
                process_reward=info[f'{aid.lower()}_process_reward'], shared_event_reward=info['event_reward'],
                shared_kill_component=config['reward']['blue_kill'],
                non_contributor=int(aid not in killers and not pair['full_gate']))


def verify_awareness_component(env, info):
    """Independently recompute awareness from TEAM visibility without reward calls."""
    mav=env.entities['MAV'].state
    alive=[env.entities[b].state for b in BLUE_IDS if env.entities[b].state.alive]
    raw=sum(.3*max(1-geometry(mav,env.entities[b].state).ata/(np.pi/2),0)
            for b in BLUE_IDS if env.entities[b].state.alive and env.team_visible(b)) if mav.alive else 0.
    expected=raw/len(alive) if alive else 0.
    if not np.isclose(info['mav_R_aware'],expected,rtol=0,atol=1e-12):
        raise AssertionError('actual awareness disagrees with team-visible formula')
    return expected


def assign_targets(env, mode, occupancy_penalty_m=5000.):
    """DIAGNOSTIC ONLY distance assignment; ties resolved by canonical ID order."""
    blues = [b for b in BLUE_IDS if env.entities[b].state.alive]
    reds = [a for a in RED_IDS if env.entities[a].state.alive]
    if not reds: return {}
    distances = {(b, a): distance(env.entities[b].state, env.entities[a].state) for b in blues for a in reds}
    if mode == 'B1':
        # Injection if possible; otherwise every remaining Red receives pressure.
        choices = (c for c in product(reds, repeat=len(blues)) if len(set(c)) == min(len(blues), len(reds)))
        best = min(choices, key=lambda c: sum(distances[b, a] for b, a in zip(blues, c)))
        return dict(zip(blues, best))
    if mode != 'B2': raise ValueError(mode)
    loads = Counter(); result = {}
    for b in blues:
        a = min(reds, key=lambda a: distances[b, a]+occupancy_penalty_m*loads[a])
        result[b] = a; loads[a] += 1
    return result


def install_blue_variant(env, mode):
    if mode == 'B0': return
    class DiagnosticBluePolicy(BluePolicy):
        def _refresh_guidance(self, blue, red, decision_step):
            # A single synchronized cohort refresh avoids sequential-slot bias.
            # Invalid-target / boundary-forced refresh also refreshes the cohort.
            if getattr(self, '_audit_refresh_step', None) != decision_step:
                assignment = assign_targets(env, mode)
                for bid, aid in assignment.items():
                    st = self._guidance_state[bid]
                    st.target_id = aid
                    st.desired_heading, st.desired_pitch = self._guidance_angles(env.entities[bid], red[aid])
                    st.last_refresh_step = decision_step; st.force_refresh = False
                self._audit_refresh_step = decision_step
            target = self._guidance_state[blue.aircraft_id].target_id
            return red.get(target)
    p = env.blue_policy
    env.blue_policy = DiagnosticBluePolicy(p.decision_dt, p.physics_dt, p.battlefield, p.target_refresh_steps)


def install_weapon_variant(env, mode):
    """Instance-only resolver: only UAV resource admissibility is changed.

    W3 blocks k+1..k+5 after a kill at k; blocked streaks reset. Simultaneous
    multiple target kills remain possible for W3; W1/2 allocate by canonical pair
    order with strict per-UAV unique-target budgets. Blue rules are unchanged.
    """
    if mode == 'W0': return
    if mode not in ('W1', 'W2', 'W3'): raise ValueError(mode)
    credits = {a: set() for a in UAVS}; last_kill = {}
    def resolve():
        c = env.config['combat']; pairs = []; step = env.step_count+1
        for aid in RED_IDS+BLUE_IDS:
            a = env.entities[aid]
            if not a.state.alive: continue
            unavailable = aid in UAVS and (
                (mode in ('W1', 'W2') and len(credits[aid]) >= (2 if mode == 'W1' else 1)) or
                (mode == 'W3' and step-last_kill.get(aid, -100) <= 5))
            for bid in (BLUE_IDS if a.team == 'red' else RED_IDS):
                key = aid, bid; b = env.entities[bid]
                if unavailable or not b.state.alive or (aid == 'MAV' and not c.get('mav_can_attack', True)):
                    env._attack_streak[key] = 0; continue
                g = geometry(a.state, b.state)
                env._attack_streak[key] = env._attack_streak.get(key, 0)+1 if all(conditions(g.distance, g.ata, g.aa, c)) else 0
                if env._attack_streak[key] >= c['hold_steps']: pairs.append(key)
        accepted = []; batch = defaultdict(set)
        for aid, bid in sorted(pairs):
            if aid in UAVS and mode in ('W1', 'W2') and len(credits[aid] | batch[aid]) >= (2 if mode == 'W1' else 1):
                env._attack_streak[aid, bid] = 0; continue
            accepted.append((aid, bid))
            if aid in UAVS: batch[aid].add(bid)
        deaths = {}
        for aid, bid in accepted:
            cause = 'red_attack' if bid in BLUE_IDS else 'blue_attack'
            env._deactivate(bid, cause, deaths)
            (env._red_attack_kills if bid in BLUE_IDS else env._blue_attack_kills).add(bid)
        for aid, targets in batch.items():
            credits[aid].update(targets); last_kill[aid] = step
        for key in list(env._attack_streak):
            if key[0] in deaths or key[1] in deaths: env._attack_streak[key] = 0
        return [dict(attacker=a, target=b) for a, b in accepted], deaths
    env._resolve_attacks = resolve


def replay(actors, cfg, env_seed, action_seed, device='cuda', blue_mode='B0', weapon_mode='W0'):
    env = Env(cfg, profile='main'); obs, _ = env.reset(seed=env_seed)
    config_before = deepcopy(env.config)
    install_blue_variant(env, blue_mode); install_weapon_variant(env, weapon_mode)
    original = env._resolve_attacks
    captures = []; steps = []; blue_rows = []; pairs = []; support = []; events_all = []; gaps = []; drift = []
    def observer():
        pp = {}
        for aid in RED_IDS+BLUE_IDS:
            if not env.entities[aid].state.alive: continue
            for bid in (BLUE_IDS if aid in RED_IDS else RED_IDS):
                if not env.entities[bid].state.alive: continue
                g = geometry(env.entities[aid].state, env.entities[bid].state)
                pp[aid, bid] = dict(agent=aid, target=bid, distance_m=g.distance,
                    ATA_deg=float(np.rad2deg(g.ata)), AA_deg=float(np.rad2deg(g.aa)),
                    full_gate=int(all(conditions(g.distance, g.ata, g.aa, cfg['combat']))),
                    direct_visible=int(env.direct_visible(aid, bid)) if aid in RED_IDS else None,
                    streak_before=env._attack_streak.get((aid, bid), 0))
        # Navigation command held during this transition; do not refresh diagnostics.
        nav = {b: deepcopy(env.blue_policy._guidance_state[b]) for b in BLUE_IDS}
        events, deaths = original()
        for key, r in pp.items():
            r['kill_event'] = int(any(e['attacker']==key[0] and e['target']==key[1] for e in events))
            r['streak_evaluated'] = (r['streak_before']+1 if r['full_gate'] else 0)
            if key[0]=='MAV' and not cfg['combat'].get('mav_can_attack', True): r['streak_evaluated'] = 0
            if weapon_mode != 'W0':
                r['streak_evaluated'] = cfg['combat']['hold_steps'] if r['kill_event'] else env._attack_streak.get(key, 0)
            r['navigation_target'] = nav[key[0]].target_id if key[0] in BLUE_IDS else nav[key[1]].target_id
        captures.append((pp, nav, events)); return events, deaths
    env._resolve_attacks = observer
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(action_seed)
        if torch.device(device).type == 'cuda': torch.cuda.manual_seed_all(action_seed)
        while True:
            nav_pre = navigation_snapshot(env)
            pre = {a: e.state.copy() for a, e in env.entities.items()}
            with torch.no_grad():
                actions = np.stack([actor.sample(torch.as_tensor(obs[aid], device=device).unsqueeze(0), deterministic=False)[0].squeeze(0).cpu().numpy()
                                    for actor, aid in zip(actors.actors, RED_IDS)])
            obs, _, term, trunc, info = env.step(actions)
            verify_awareness_component(env,info)
            step = env.step_count; pp, nav, events = captures[-1]
            tag = dict(environment_seed=env_seed, action_seed=action_seed, step=step)
            row = dict(**tag, alive_blue_pre=sum(pre[b].alive for b in BLUE_IDS), alive_blue_post=sum(env.entities[b].state.alive for b in BLUE_IDS),
                       team_reward=info['team_reward'], event_reward=info['event_reward'], attack_events=json.dumps(events))
            target_ids = []
            for nr in nav_pre:
                b = nr['blue']; state = nav[b]
                nr.update(tag); nr['target_used'] = state.target_id
                nr['used_guidance_age'] = step-1-state.last_refresh_step if state.target_id else None
                nr['used_desired_heading'] = state.desired_heading; nr['used_desired_pitch'] = state.desired_pitch
                nr['alive_post'] = int(env.entities[b].state.alive)
                blue_rows.append(nr)
                if pre[b].alive: target_ids.append(state.target_id)
            row.update(concentration(target_ids))
            bp = [distance(pre[a], pre[b]) for i, a in enumerate(BLUE_IDS) for b in BLUE_IDS[i+1:] if pre[a].alive and pre[b].alive]
            row['mean_Blue_separation_m'] = mean(bp)
            for aid, e in env.entities.items():
                for ax, val in [('x', e.state.x), ('y', e.state.y), ('h', e.state.h), ('alive', int(e.state.alive))]: row[f'{aid}_{ax}'] = val
            sr = dict(**tag, **support_snapshot(env, info)); support.append(sr)
            for r in pp.values(): pairs.append(dict(**tag, **r))
            for ev in events: events_all.append(dict(**tag, **ev, navigation_target=nav[ev['target']].target_id if ev['target'] in BLUE_IDS else nav[ev['attacker']].target_id))
            killed = {e['target'] for e in events if e['target'] in BLUE_IDS}
            for b in killed:
                killers = {e['attacker'] for e in events if e['target']==b}
                for a in UAVS:
                    if env.entities[a].state.alive and (a,b) in pp:
                        gaps.append(dict(**tag, **contribution_gap(a,b,killers,pp[a,b],info,cfg)))
            for i, a in enumerate(UAVS, 1):
                rr = [r for (agent,b),r in pp.items() if agent==a]
                if not rr: continue
                d = min(r['distance_m'] for r in rr)
                uc = np.mean([position(env.entities[u].state) for u in UAVS if env.entities[u].state.alive],axis=0) if any(env.entities[u].state.alive for u in UAVS) else None
                dr = dict(**tag, agent=a, minimum_distance_m=d, any_full_gate=int(any(r['full_gate'] for r in rr)),
                    any_ATA_gate=int(any(r['ATA_deg']<cfg['combat']['ata_deg'] for r in rr)),
                    any_direct=int(any(r['direct_visible'] for r in rr)),
                    attack_credits=sum(e['attacker']==a for e in events),
                    boundary_margin_m=boundary_margin(env.entities[a].state,cfg),
                    UAV_centroid_distance_m=float(np.linalg.norm(position(env.entities[a].state)-uc)) if uc is not None else None,
                    action_saturation=float(np.mean(np.abs(actions[i])>=.95)))
                dr['non_engagement'] = int(d>8000 and not dr['any_full_gate'] and not dr['attack_credits'])
                drift.append(dr)
            steps.append(row)
            if term or trunc:
                assert env.config == config_before, 'diagnostic variant modified environment config'
                result = dict(info['episode_summary'])
                for rows in (steps,blue_rows,pairs,support,events_all,gaps,drift):
                    for r in rows: r['outcome']=result['outcome']
                return dict(result=result,steps=steps,blue=blue_rows,pairs=pairs,support=support,events=events_all,gaps=gaps,drift=drift)


def scopes(steps):
    kills = {r['step'] for r in steps if any(e['target'] in BLUE_IDS for e in json.loads(r['attack_events']))}
    return {'all': steps, **{f'alive_Blue_{n}':[r for r in steps if r['alive_blue_pre']==n] for n in (4,3,2,1)},
            'kill_before5':[r for r in steps if any(k-5<=r['step']<k for k in kills)],
            'kill_after5':[r for r in steps if any(k<r['step']<=k+5 for k in kills)]}


def summarize_concentration(episodes, tag):
    rows = []
    for scope in ('all','alive_Blue_4','alive_Blue_3','alive_Blue_2','alive_Blue_1','kill_before5','kill_after5'):
        ss = [r for ep in episodes for r in scopes(ep['steps'])[scope]]
        rows.append(dict(**tag,scope=scope,decision_steps=len(ss),alive_Blue_exposures=sum(r['alive_blue_pre'] for r in ss),
            max_same_ge2_fraction=mean([r['max_same']>=2 for r in ss]),max_same_ge3_fraction=mean([r['max_same']>=3 for r in ss]),
            max_same_eq4_fraction=mean([r['max_same']==4 for r in ss]),mean_Blue_separation_m=mean([r['mean_Blue_separation_m'] for r in ss]),
            **{f'mean_pressure_{a}':mean([r[f'pressure_{a}'] for r in ss]) for a in RED_IDS}))
    return rows


def event_context(ep, tag):
    result=[]
    for e in ep['events']:
        if e['target'] not in BLUE_IDS: continue
        prior=[r for r in ep['blue'] if r['blue']==e['target'] and e['step']-10<=r['step']<=e['step']]
        pair=next(r for r in ep['pairs'] if r['step']==e['step'] and r['agent']==e['attacker'] and r['target']==e['target'])
        sr=next(r for r in ep['steps'] if r['step']==e['step'])
        chased=e['navigation_target']
        killer_chased_distance=None
        if chased in RED_IDS:
            k=np.array([sr[f'{e["attacker"]}_{a}'] for a in ('x','y','h')]); t=np.array([sr[f'{chased}_{a}'] for a in ('x','y','h')])
            killer_chased_distance=float(np.linalg.norm(k-t))
        result.append(dict(**tag,**e,killer_is_chased_red=int(chased==e['attacker']),third_party=int(chased!=e['attacker']),
            full_gate=pair['full_gate'],streak=pair['streak_evaluated'],distance_m=pair['distance_m'],ATA_deg=pair['ATA_deg'],AA_deg=pair['AA_deg'],
            killer_to_chased_Red_m=killer_chased_distance,other_alive_Blue=sr['alive_blue_pre']-1,
            preceding10_and_kill_step_target_histogram=json.dumps(dict(Counter(r['target_used'] for r in prior))),
            preceding10_and_kill_step_targets=json.dumps([{k:r[k] for k in ('step','target_used','instantaneous_nearest')} for r in prior])))
    return result


def population_dominance(prior, agent_rows):
    rows=[]
    index={(int(r['training_seed']),int(r['episode']),r['agent']):r for r in agent_rows if int(r['sampled_steps'])==2_000_000}
    for r in prior:
        if int(r['sampled_steps'])!=2_000_000: continue
        seed=int(r['training_seed']); ep=int(r['episode'])
        counts=[int(index[seed,ep,a]['streak3']) for a in UAVS]
        unique=int(r['red_attack_kills']); exact=sum(counts)==unique
        rows.append(dict(training_seed=seed,episode=ep,outcome=r['outcome'],population='prior_200_episodes',
            unique_Blue_kills=unique,**dominance(counts),
            **{f'{a}_attack_credits':c for a,c in zip(UAVS,counts)},
            unique_attribution_exact=exact,unique_attribution_missing=None if exact else 'prior aggregates omit target identity; simultaneous credits ambiguous'))
    return rows


def drift_summary(episodes, tag):
    output=[]
    for aid in UAVS:
        rr=[r for ep in episodes for r in ep['drift'] if r['agent']==aid]
        intervals=[]
        for ep in episodes:
            length=0; previous=None
            for r in [r for r in ep['drift'] if r['agent']==aid]:
                if r['non_engagement'] and (previous is None or r['step']==previous+1): length+=1
                else:
                    if length>=10: intervals.append(length)
                    length=1 if r['non_engagement'] else 0
                previous=r['step']
            if length>=10: intervals.append(length)
        unique= sum(len(kill_sets(ep['events'])[aid]) for ep in episodes)
        output.append(dict(**tag,agent=aid,alive_combat_decision_steps=len(rr),
            **{f'within_{k}km_fraction':mean([r['minimum_distance_m']<=k*1000 for r in rr]) for k in (3,5,8)},
            **{f'beyond_{k}km_fraction':mean([r['minimum_distance_m']>k*1000 for r in rr]) for k in (8,12)},
            full_gate_fraction=mean([r['any_full_gate'] for r in rr]),ATA_gate_fraction=mean([r['any_ATA_gate'] for r in rr]),direct_visible_fraction=mean([r['any_direct'] for r in rr]),
            attack_credits=sum(r['attack_credits'] for r in rr),unique_target_credits=unique,
            mean_centroid_distance_m=mean([r['UAV_centroid_distance_m'] for r in rr]),mean_boundary_margin_m=mean([r['boundary_margin_m'] for r in rr]),
            action_saturation=mean([r['action_saturation'] for r in rr]),non_engagement_intervals_ge10=len(intervals),
            non_engagement_interval_steps=sum(intervals),longest_non_engagement_interval=max(intervals,default=0)))
    return output


def support_summary(episodes, tag):
    rr=[r for ep in episodes for r in ep['support'] if r['MAV_alive'] and r['alive_blue_post']>0]
    masks={'all':lambda r:True,'MAV_direct_zero':lambda r:r['MAV_direct_count']==0,
           'MAV_direct_zero_team_positive':lambda r:r['MAV_direct_count']==0 and r['team_visible_count']>0,
           'MAV_direct_positive':lambda r:r['MAV_direct_count']>0}
    for label in ('UAV','engagement'):
        for k in (5,8,12): masks[f'{label}_centroid_beyond_{k}km']=lambda r,label=label,k=k:r[f'MAV_{label}_centroid_m'] is not None and r[f'MAV_{label}_centroid_m']>k*1000
    output=[]
    for key,predicate in masks.items():
        ss=[r for r in rr if predicate(r)]
        output.append(dict(**tag,state=key,steps=len(ss),fraction_of_alive_combat_steps=ratio(len(ss),len(rr)),
            mean_process_reward=mean([r['mav_process_reward'] for r in ss]),median_process_reward=float(np.median([r['mav_process_reward'] for r in ss])) if ss else None,
            positive_reward_fraction=mean([r['mav_process_reward']>0 for r in ss]),nonnegative_reward_fraction=mean([r['mav_process_reward']>=0 for r in ss]),
            mean_awareness=mean([r['mav_R_aware'] for r in ss]),mean_aspect=mean([r['mav_R_aspect'] for r in ss]),mean_threat=mean([r['mav_R_threat'] for r in ss]),
            mean_team_visible=mean([r['team_visible_count'] for r in ss]),mean_datalink_only=mean([r['UAV_datalink_only_total'] for r in ss]),
            mean_MAV_exclusive_information=mean([r['MAV_exclusive_information_count'] for r in ss])))
    return output


def cf_record(ep, tag, mode):
    ks=kill_sets(ep['events']); cc=summarize_concentration([ep],{})[0]
    return dict(**tag,condition=mode,evidence='DIAGNOSTIC COUNTERFACTUAL',**ep['result'],
        killer_distribution=json.dumps({a:sorted(v) for a,v in ks.items()}),
        **dominance([len(ks[a]) for a in UAVS]),P2_steps=sum(r['alive_blue_pre']==1 for r in ep['steps']),
        P2_entered=any(r['alive_blue_pre']==1 for r in ep['steps']),
        max_same_ge3_fraction=cc['max_same_ge3_fraction'],max_same_eq4_fraction=cc['max_same_eq4_fraction'],
        Blue_target_distribution=json.dumps({a:sum(r[f'pressure_{a}'] for r in ep['steps']) for a in RED_IDS}))


def make_plots(ep, seed, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=['#555555','#2474b7','#d98b2b','#459b68','#dddddd']; ids=list(RED_IDS)+['dead']
    steps=ep['steps']; x=[r['step'] for r in steps]
    matrix=[]
    for b in BLUE_IDS:
        rr=[r for r in ep['blue'] if r['blue']==b]
        matrix.append([ids.index(r['target_used']) if r['alive'] and r['target_used'] in RED_IDS else 4 for r in rr])
    fig,ax=plt.subplots(figsize=(10,3)); im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap=ListedColormap(colors),vmin=-.5,vmax=4.5,extent=(.5,len(x)+.5,3.5,-.5))
    ax.set_yticks(range(4),BLUE_IDS); ax.set_xlabel('Decision step (target used during transition)'); ax.set_title(f'Seed {seed} / scenario 1011 / {ep["result"]["outcome"]}')
    cb=fig.colorbar(im,ax=ax,ticks=range(5)); cb.ax.set_yticklabels(ids); fig.tight_layout(); fig.savefig(out/f'blue_target_timeline_seed{seed}.png',dpi=180); plt.close(fig)
    specs=[('blue_target_concentration',steps,['max_same']+[f'pressure_{a}' for a in RED_IDS],'Blue target pressure'),
           ('mav_support_distance',ep['support'],['MAV_UAV_centroid_m','MAV_engagement_centroid_m','nearest_Blue_m'],'Distance (m)'),
           ('mav_visibility',ep['support'],['MAV_direct_count','team_visible_count','UAV_datalink_only_total','MAV_exclusive_information_count'],'Target count'),
           ('mav_reward',ep['support'],['mav_process_reward','mav_R_threat','mav_R_aspect','mav_R_aware'],'Reward / raw normalized component')]
    for name,rr,fields,ylabel in specs:
        fig,ax=plt.subplots(figsize=(9,3.3))
        for f in fields: ax.plot([r['step'] for r in rr],[r[f] for r in rr],label=f)
        ax.set_xlabel('Decision step'); ax.set_ylabel(ylabel); ax.set_title(f'Seed {seed}: scenario 1011'); ax.legend(fontsize=8,ncol=2); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/f'{name}_seed{seed}.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6))
    for i,aid in enumerate(RED_IDS+BLUE_IDS):
        rr=[r for r in steps if r[f'{aid}_alive'] or r==steps[0]]
        ax.plot([r[f'{aid}_x']/1000 for r in rr],[r[f'{aid}_y']/1000 for r in rr],label=aid,linestyle='-' if aid in RED_IDS else '--')
        if aid in BLUE_IDS:
            last=None
            for r in [r for r in ep['blue'] if r['blue']==aid and r['alive']]:
                if r['target_used']!=last:
                    ax.scatter(r['x']/1000,r['y']/1000,s=14); ax.annotate(f'{r["step"]}:{r["target_used"]}',(r['x']/1000,r['y']/1000),fontsize=6)
                last=r['target_used']
    for key in ('UAV','engagement'):
        ax.plot([r[f'{key}_centroid_x']/1000 if r[f'{key}_centroid_x'] is not None else np.nan for r in ep['support']],
                [r[f'{key}_centroid_y']/1000 if r[f'{key}_centroid_y'] is not None else np.nan for r in ep['support']],':',lw=2,label=f'{key} centroid')
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_aspect('equal',adjustable='datalim'); ax.legend(fontsize=8,ncol=3); ax.set_title(f'Seed {seed}: cached-target switches (step:target)'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/f'xy_target_switch_seed{seed}.png',dpi=180); plt.close(fig)


SOURCE_FACTS = '''# SOURCE-CODE FACT: v3.11 contracts

- env/blue_policy.py::select_target: independent nearest alive Red, squared 3D distance; no occupancy, assignment, threat priority or coordinated pincer. Four Blue may choose one Red. target_refresh_steps=2; heading/pitch and target are held between refreshes, except dead target / boundary forced refresh.
- env/mavuav.py::_resolve_attacks: no ammunition, missile entities, cooldown or per-UAV kill limit. Every living attack-capable aircraft tests ALL opposite aircraft, not just its navigation target. UAV may kill all four Blue, including synchronous multi-target events. Killing resets pairs involving the dead aircraft, not other living targets' progression.
- _reward: +100 per UNIQUE killed Blue is shared by all four Red (even a non-killer); own process rewards differ. Team reward is mean of four rewards; common/buffer.py stores their mean and team GAE feeds active-agent HAPPO updates. An inactive agent does not gain its own policy update merely because shared reward exists.
- _role_process_rewards / mav_normalized_role_reward: P_M=.3 R_threat+.2 mean(R_aspect)+.4 mean(R_aware). Threat=-1 if alive Blue attack streak against MAV>0. Aspect=-max(1-Blue ATA/(pi/4),0), all alive Blue. Awareness=.3 max(1-MAV ATA/(pi/2),0), TEAM-visible alive Blue; both means denominator=all alive Blue. MAV death zeros own process reward; shared MAV-loss and terminal penalties remain separate.
- MAV own direct visibility: NOT REQUIRED for awareness. MAV-UAV distance, engagement-center distance, exclusive MAV datalink contribution: NOT REPRESENTED IN REWARD. Team sensing and MAV/Blue orientation do enter. Survival/threat enter; this is not proof that spatial drift causes failure.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,nargs=3,default=[ROOT/f'outputs/happo_v311_seed{s}_2m' for s in (1,2,3)])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/audits/v311_environment_rationality')
    parser.add_argument('--episodes',type=int,choices=(20,200),default=20)
    parser.add_argument('--allow-population-replay',action='store_true')
    parser.add_argument('--device',default='cuda',choices=('cuda',))
    args=parser.parse_args()
    if args.episodes!=20 and not args.allow_population_replay: parser.error('200 episodes requires explicit --allow-population-replay')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    if args.output.exists(): raise FileExistsError(f'fresh audit output required: {args.output}')
    torch.set_num_threads(1)
    cfg=load_environment_config(ROOT/'configs/env_v311.yaml')
    prior=read_numeric_csv(PRIOR/'replay_episodes.csv'); agents=read_numeric_csv(PRIOR/'replay_agents.csv')
    protected=[p for folder in ('algorithm','env','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')]
    protected += [p for run in args.run_dir for p in run.rglob('*') if p.is_file()]
    protected += [p for p in PRIOR.rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    args.output.mkdir(parents=True); out=args.output
    (out/'source_contract.md').write_text(SOURCE_FACTS,encoding='utf-8')
    all_conc=[];contexts=[];all_gap=[];all_support=[];all_drift=[];weapons=[];blues=[];population=[];support_examples=[];blue_attacks=[];kill_intervals=[];checks={};cases={}
    global_cpu_rng=torch.get_rng_state().clone(); global_cuda_rng=torch.cuda.get_rng_state_all()
    for run in args.run_dir:
        payload=torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False)
        validate_checkpoint_contract(payload,cfg)
        if payload['environment_config']!=cfg or int(payload['sampled_steps'])!=2_000_000: raise ValueError('exact2M env_v311 required')
        if (payload['actor_variant'],payload['critic_variant'],payload['method_variant'])!=('vanilla','mlp','baseline'): raise ValueError('Vanilla baseline required')
        tc=payload['trainer_config']; seed=int(tc['seed']); tag=dict(run=run.name,training_seed=seed)
        if seed not in (1,2,3) or seed in cases: raise ValueError('three distinct seeds1-3 required')
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            actor=IndependentActors(hidden_dim=tc['hidden_dim'],log_std_init=tc.get('actor_log_std_init',-.5)).cuda().eval()
        actor.load_state_dict(payload['actors']); params={k:v.clone() for k,v in actor.state_dict().items()}
        episodes=[]
        for i in range(args.episodes):
            ep=replay(actor,cfg,1000+i,2000+i); episodes.append(ep)
            old=next(r for r in prior if int(r['training_seed'])==seed and int(r['sampled_steps'])==2_000_000 and int(r['environment_seed'])==1000+i)
            for k in ('outcome','episode_length','red_attack_kills','blue_attack_kills','red_uav_survivors','episode_return'):
                if ep['result'][k]!=old[k]: raise AssertionError(f'prior replay mismatch {seed}/{i}/{k}')
            print(f'Seed{seed} mechanism episode {i+1}/{args.episodes}: {ep["result"]["outcome"]} kills={ep["result"]["red_attack_kills"]}',flush=True)
            population.append(dict(**tag,environment_seed=1000+i,action_seed=2000+i,**ep['result']))
        case=episodes[11]; cases[seed]=case
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            official=evaluate_actors(actor,cfg,1,'main',1011,'cuda',deterministic=False,action_seed=2011)[0]
        assert case['result']==official
        checks[f'seed{seed}_official_evaluator_exact_match']=True
        for population_name,ee in [('mechanism_20_episodes' if args.episodes==20 else 'explicit_population_200',episodes),('matched_1011',[case])]:
            tt=dict(**tag,population=population_name)
            all_conc.extend(summarize_concentration(ee,tt)); all_support.extend(support_summary(ee,tt)); all_drift.extend(drift_summary(ee,tt))
        for ep in episodes:
            contexts.extend(event_context(ep,tag)); all_gap.extend(dict(**tag,**r) for r in ep['gaps'])
            for a in UAVS:
                kk=sorted({e['step'] for e in ep['events'] if e['attacker']==a})
                kill_intervals.extend(dict(**tag,environment_seed=ep['result'].get('seed',ep['steps'][0]['environment_seed']),agent=a,first_kill_step=k,next_kill_step=l,time_between_kills_same_uav=l-k) for k,l in zip(kk,kk[1:]))
                simultaneous=Counter(e['step'] for e in ep['events'] if e['attacker']==a)
                kill_intervals.extend(dict(**tag,environment_seed=ep['steps'][0]['environment_seed'],agent=a,first_kill_step=k,next_kill_step=k,time_between_kills_same_uav=0) for k,n in simultaneous.items() for _ in range(n-1))
            for a in RED_IDS:
                pp=[r for r in ep['pairs'] if r['agent'] in BLUE_IDS and r['target']==a]
                blue_attacks.append(dict(**tag,environment_seed=ep['steps'][0]['environment_seed'],target=a,alive_pair_exposures=len(pp),full_gate_pair_steps=sum(r['full_gate'] for r in pp),streak_ge1=sum(r['streak_evaluated']>=1 for r in pp),streak_ge2=sum(r['streak_evaluated']>=2 for r in pp),attack_event_credits=sum(r['kill_event'] for r in pp),unique_death=int(any(r['kill_event'] for r in pp)),navigation_target_pair_steps=sum(r['navigation_target']==a for r in pp),full_gate_while_navigation_target=sum(r['full_gate'] and r['navigation_target']==a for r in pp)))
        for typ,predicate in [('A_MAV_direct',lambda r:r['MAV_direct_count']>0),('B_team_only',lambda r:r['MAV_direct_count']==0 and r['team_visible_count']>0)]:
            choices=[r for ep in episodes for r in ep['support'] if predicate(r) and r['MAV_alive'] and r['alive_blue_post']>0]
            if choices: support_examples.append(dict(**tag,case=typ,selection='highest awareness among qualifying recorded states',**max(choices,key=lambda r:r['mav_R_aware'])))
        table(out/f'blue_target_timeline_seed{seed}.csv',case['blue']); table(out/f'mav_support_steps_seed{seed}.csv',case['support']); table(out/f'case_steps_seed{seed}.csv',case['steps']); table(out/f'case_pairs_seed{seed}.csv',case['pairs'])
        make_plots(case,seed,out)
        for mode in ('W0','W1','W2','W3'):
            ep=case if mode=='W0' else replay(actor,cfg,1011,2011,weapon_mode=mode)
            weapons.append(cf_record(ep,tag,mode)); print(f'Seed{seed} {mode}: {ep["result"]["outcome"]}, kills{ep["result"]["red_attack_kills"]}',flush=True)
        for mode in ('B0','B1','B2'):
            ep=case if mode=='B0' else replay(actor,cfg,1011,2011,blue_mode=mode)
            blues.append(cf_record(ep,tag,mode)); print(f'Seed{seed} {mode}: {ep["result"]["outcome"]}, kills{ep["result"]["red_attack_kills"]}',flush=True)
        assert all(torch.equal(v,params[k]) for k,v in actor.state_dict().items())
        checks[f'seed{seed}_actor_parameters_unchanged']=True
    checks['Torch_CPU_RNG_restored']=torch.equal(global_cpu_rng,torch.get_rng_state())
    checks['Torch_CUDA_RNG_restored']=all(torch.equal(a,b) for a,b in zip(global_cuda_rng,torch.cuda.get_rng_state_all()))
    assert all(checks.values())
    assert all(sha(Path(p))==h for p,h in hashes.items()); checks['all_input_SHA256_unchanged']=True
    dom=population_dominance(prior,agents)
    for name,rows in [('blue_target_concentration_summary',all_conc),('kill_context_by_blue_target',contexts),('single_uav_kill_dominance',dom),('shared_reward_contribution_gap',all_gap),('weapon_counterfactual',weapons),('mav_support_summary',all_support),('mav_reward_vs_support_state',support_examples),('uav_non_engagement_drift',all_drift),('blue_policy_counterfactual',blues),('blue_attack_target_summary',blue_attacks),('time_between_kills_same_uav',kill_intervals),('mechanism_episode_summary',population)]:
        table(out/f'{name}.csv',rows)
    summary=dict(device='cuda',device_name=torch.cuda.get_device_name(),protocol=dict(profile='main',environment_seeds=[1000,999+args.episodes],action_seeds=[2000,1999+args.episodes],episodes_per_policy=args.episodes,matched_environment_seed=1011,matched_action_seed=2011,deterministic=False),
        case_results={str(s):ep['result'] for s,ep in cases.items()},checks=checks,input_sha256=hashes,
        limitations=['20 mechanism episodes per policy are not population200; historical200 credited kills lack Blue target identities in multi-attacker episodes','B1/B2 synchronized cohort refresh; B2 occupancy penalty5000m, diagnostic only','Weapon resources change later trajectories; frozen-policy counterfactuals are not training causality'],
        concentration=all_conc,mav_support=all_support,uav_drift=all_drift,weapon_counterfactual=weapons,blue_counterfactual=blues)
    dump(out/'summary.json',summary)
    write_report(out,summary,dom,contexts,all_gap)
    finish_saved_analysis(out)
    print(f'Audit complete: {out}',flush=True)


def write_report(out, summary, dom, contexts, gaps):
    report=['# v3.11 第三阶段环境合理性审计','',
        'SOURCE-CODE FACT 详见 source_contract.md；MEASURED 为固定20局/策略与既有200局；DIAGNOSTIC COUNTERFACTUAL 只为1011冻结策略。INFERENCE 不等于因果。',
        '导航快照为pre-action，target_used为本步实际保持的导航目标；gate为post-physics/pre-combat；MAV support/reward为post-combat。比例均有独立分母。',
        'kill context按真实attacker-target事件，shared kill gap按唯一被击杀Blue×存活UAV，避免同步攻击重复奖励。旧200局仅attack credits可以完全恢复；多attacker归因缺失显式标记。','',
        '## MEASURED: matched scenario1011','|Seed|Outcome|Kills|Length|UAV survivors|','|---|---|---|---|---|']
    for seed,r in summary['case_results'].items(): report.append(f'|{seed}|{r["outcome"]}|{r["red_attack_kills"]}|{r["episode_length"]}|{r["red_uav_survivors"]}|')
    report+=['','## MEASURED: 既有200局击杀集中','|Seed|Max credits≥2|≥3|4|One attacker|Ambiguous unique attribution|','|---|---|---|---|---|---|']
    for seed in (1,2,3):
        rr=[r for r in dom if r['training_seed']==seed]
        report.append(f'|{seed}|{mean([r["max_agent_kills"]>=2 for r in rr]):.1%}|{mean([r["max_agent_kills"]>=3 for r in rr]):.1%}|{mean([r["max_agent_kills"]==4 for r in rr]):.1%}|{mean([r["attacking_agents"]==1 for r in rr]):.1%}|{sum(not r["unique_attribution_exact"] for r in rr)}|')
    report+=['','## DIAGNOSTIC COUNTERFACTUAL: scenario1011','|Seed|Variant|Outcome|Kills|UAV kill distribution|','|---|---|---|---|---|']
    for r in summary['weapon_counterfactual']+summary['blue_counterfactual']: report.append(f'|{r["training_seed"]}|{r["condition"]}|{r["outcome"]}|{r["red_attack_kills"]}|{r["killer_distribution"]}|')
    report+=['','## Hypothesis matrix','|Hypothesis|Evidence|Status|Severity|','|---|---|---|---|']
    matrix=[('H1 multiple Blue target same Red','concentration CSV / independent nearest source','VERIFIED','medium'),
        ('H2 dogpile creates convergence','Blue separation co-occurs with cached-target concentration; B1/B2 trajectory changes','PARTIALLY_SUPPORTED','medium'),
        ('H3 dogpile enables third-party farming','kill_context third_party plus target histories; frozen replay not causal training','PARTIALLY_SUPPORTED','medium'),
        ('H4 sufficiently realistic opponent','No empirical real-combat reference; simple pursuit source only','INCONCLUSIVE','medium'),
        ('H5 unlimited resources enable role collapse','Single-UAV four kills allowed; credit concentration measured, training cause unknown','PARTIALLY_SUPPORTED','high'),
        ('H6 seed1 performance depends on one UAV','matched case weapon budgets + prior200 dominance; cannot generalize one counterfactual','PARTIALLY_SUPPORTED','high'),
        ('H7 shared reward benefits non-contributors','+100 shared event and per-kill non-contributor gap','VERIFIED','medium'),
        ('H8 shared reward causes inactivity','No retraining intervention','INCONCLUSIVE','medium'),
        ('H9 MAV reward fails spatial anchoring','No formation/engagement-distance variable in process reward; far-state reward CSV','VERIFIED','high'),
        ('H10 reward without own direct sensing','team-visible awareness source + recorded B states','VERIFIED','high'),
        ('H11 MAV drift reduces UAV information support','MAV-exclusive and datalink-only counts are associations, no isolated intervention','INCONCLUSIVE','medium'),
        ('H12 UAV non-engagement drift','>8km/no gate/no kill intervals>=10; per-agent CSV','VERIFIED','medium'),
        ('H13 three issues explain seed sensitivity','No training counterfactuals','INCONCLUSIVE','high'),
        ('H14 algorithm primary cause','Observed local policies differ; root-cause priority not identified','INCONCLUSIVE','high')]
    report += [f'|{h}|{e}|{status}|{severity}|' for h,e,status,severity in matrix]
    report+=['','## INFERENCE: 模块分级与下一步',
        'Blue: SIMPLISTIC_BUT_USABLE。冻结策略对简化nearest仍能获胜；单局分配反事实显示几何敏感，但不足以判定总体MATERIALLY_DISTORTING。',
        'Weapon: OVERLY_PERMISSIVE。无资源约束与单机多目标收割为可验证抽象，不意味着combat程序错误。',
        'MAV reward: PARTIALLY_ROLE_ALIGNED。存活/threat/aspect合理进入，但没有空间支援锚定，而且awareness可来自UAV的探测。',
        '唯一优先建议：下一轮单变量测试MAV awareness资格由team-visible改为MAV direct-visible；这是角色定义与源码最直接的错配。此处没有实施，也不能保证提高胜率。',
        '暂不优先扩展算法；先验证明确的环境角色建模假设，不能宣称已找到seed sensitivity因果根因。',
        '原始正式数据/actor/checkpoint SHA、RNG恢复及逐参数不变检查均见summary.json。']
    (out/'research_report.md').write_text('\n'.join(report),encoding='utf-8')


def finish_saved_analysis(out):
    """Postprocess our new outputs only; no actor/environment replay calls."""
    out=Path(out); summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p))==h for p,h in summary['input_sha256'].items())
    dom=read_numeric_csv(out/'single_uav_kill_dominance.csv')
    contexts=read_numeric_csv(out/'kill_context_by_blue_target.csv')
    gaps=read_numeric_csv(out/'shared_reward_contribution_gap.csv')
    blue_attempts=read_numeric_csv(out/'blue_attack_target_summary.csv')
    intervals=read_numeric_csv(out/'time_between_kills_same_uav.csv')
    stats=[]; gap_stats=[]; target_stats=[]; blue_stats=[]; cases={}
    for seed in (1,2,3):
        dd=[r for r in dom if r['training_seed']==seed]
        for outcome in ('all','red','draw','blue'):
            rr=[r for r in dd if outcome=='all' or r['outcome']==outcome]
            stats.append(dict(training_seed=seed,outcome=outcome,episodes=len(rr),
                max_credit_ge2_fraction=mean([r['max_agent_kills']>=2 for r in rr]),
                max_credit_ge3_fraction=mean([r['max_agent_kills']>=3 for r in rr]),
                max_credit_eq4_fraction=mean([r['max_agent_kills']==4 for r in rr]),
                **{f'{n}_attacker_fraction':mean([r['attacking_agents']==n for r in rr]) for n in (1,2,3)},
                mean_HHI=mean([r['kill_HHI'] for r in rr]),mean_dominance_share=mean([r['dominance_share'] for r in rr]),
                ambiguous_unique_attribution=sum(r['unique_attribution_exact']=='False' for r in rr),
                **{f'{a}_mean_attack_credits':mean([r[f'{a}_attack_credits'] for r in rr]) for a in UAVS}))
        for population in ('matched_1011','mechanism_sample'):
            gg=[r for r in gaps if r['training_seed']==seed and (population!='matched_1011' or r['environment_seed']==1011)]
            for aid in UAVS:
                rr=[r for r in gg if r['agent']==aid]
                gap_stats.append(dict(training_seed=seed,population=population,agent=aid,shared_kill_exposures=len(rr),
                    non_contributor_exposures=sum(r['non_contributor'] for r in rr),
                    beyond5km_fraction=mean([r['distance_m']>5000 for r in rr]),beyond8km_fraction=mean([r['distance_m']>8000 for r in rr]),
                    ATA_beyond90_fraction=mean([r['ATA_deg']>90 for r in rr])))
        bb=read_numeric_csv(out/f'blue_target_timeline_seed{seed}.csv'); cc=read_numeric_csv(out/f'case_steps_seed{seed}.csv')
        support=read_numeric_csv(out/f'mav_support_steps_seed{seed}.csv')
        for b in BLUE_IDS:
            rr=[r for r in bb if r['blue']==b and r['alive']]
            counts=Counter(r['target_used'] for r in rr)
            target_stats.append(dict(training_seed=seed,blue=b,alive_decision_steps=len(rr),main_target=counts.most_common(1)[0][0],
                **{f'target_{a}_steps':counts[a] for a in RED_IDS},cached_vs_instantaneous_mismatch_steps=sum(r['target_used']!=r['instantaneous_nearest'] for r in rr)))
        for aid in RED_IDS:
            rr=[r for r in blue_attempts if r['training_seed']==seed and r['target']==aid]
            blue_stats.append(dict(training_seed=seed,target=aid,
                **{f:sum(r[f] for r in rr) for f in ('alive_pair_exposures','full_gate_pair_steps','streak_ge1','streak_ge2','attack_event_credits','unique_death','navigation_target_pair_steps','full_gate_while_navigation_target')}))
        ee=[r for r in contexts if r['training_seed']==seed and r['environment_seed']==1011]
        ks={aid:sorted({r['target'] for r in ee if r['attacker']==aid}) for aid in UAVS}
        longest=0; current=0
        for r in cc:
            current=current+1 if r['max_same']>=3 else 0; longest=max(longest,current)
        cases[str(seed)]=dict(kill_distribution=ks,
            kill_chasing_killer=sum(r['killer_is_chased_red'] for r in ee),third_party_kill_credits=sum(r['third_party'] for r in ee),
            longest_ge3_concentration_steps=longest,ge3_steps=sum(r['max_same']>=3 for r in cc),eq4_steps=sum(r['max_same']==4 for r in cc))
    table(out/'single_uav_kill_dominance_summary.csv',stats)
    table(out/'shared_reward_contribution_gap_summary.csv',gap_stats)
    table(out/'blue_target_assignment_summary.csv',target_stats)
    table(out/'blue_attack_summary.csv',blue_stats)
    summary['population_kill_dominance_summary']=stats
    summary['gap_summary']=gap_stats; summary['matched_case_mechanisms']=cases
    summary['blue_target_assignment_summary']=target_stats; summary['blue_attack_summary']=blue_stats
    summary['time_between_kills_summary']=dict(intervals=len(intervals),within_3_to_5_steps=sum(3<=r['time_between_kills_same_uav']<=5 for r in intervals),
        within5_including_simultaneous=sum(r['time_between_kills_same_uav']<=5 for r in intervals),simultaneous=sum(r['time_between_kills_same_uav']==0 for r in intervals))
    summary['module_ratings']=dict(Blue_policy='SIMPLISTIC_BUT_USABLE',weapon_abstraction='OVERLY_PERMISSIVE',MAV_reward='PARTIALLY_ROLE_ALIGNED')
    summary['diagnostic_definitions']=dict(
        navigation='cached_target is pre-action; target_used is actual command during transition; nearest is instantaneous pre-action',
        concentration='fraction denominator=decision steps, not alive-Blue exposure; exposure separately recorded',
        datalink='raw current datalink_visible interface flags for all three UAV slots, including inactive slots; NOT evidence of active UAV support or exclusive MAV support',
        information_support='MAV_exclusive_information_count excludes Blue already directly sensed by any alive UAV; no exclusive contribution when MAV_direct_count=0',
        kill_credit='actual attacker-target event credits; multiple attackers may share one unique Blue death')
    for r in summary['concentration']:
        if r['decision_steps']:
            g2,g3,g4=r['max_same_ge2_fraction'],r['max_same_ge3_fraction'],r['max_same_eq4_fraction']
            for n,f in ((1,1-g2),(2,g2-g3),(3,g3-g4),(4,g4)):
                r[f'max_same_{n}_steps']=round(r['decision_steps']*f)
    table(out/'blue_target_concentration_summary.csv',summary['concentration'])
    dump(out/'summary.json',summary)
    report=(out/'research_report.md').read_text(encoding='utf-8')
    marker='\n## 关键定量结果（MEASURED）'
    report=report.split(marker)[0]+marker+'\n'
    report+='\n|Seed|Blue|主追击目标|MAV/UAV1/UAV2/UAV3 steps|\n|---|---|---|---|\n'
    for r in target_stats: report+=f'|{r["training_seed"]}|{r["blue"]}|{r["main_target"]}|'+ '/'.join(str(r[f'target_{a}_steps']) for a in RED_IDS)+'|\n'
    for seed,c in cases.items():
        report+=f'\nseed{seed} case: >=3架同追 {c["ge3_steps"]}steps，最长连续{c["longest_ge3_concentration_steps"]}steps；4架同追{c["eq4_steps"]}steps。真实kill target sets={c["kill_distribution"]}；追击killer的kill credits={c["kill_chasing_killer"]}，third-party={c["third_party_kill_credits"]}。\n'
    report+='\n### MAV support-stratum measurements\n|Seed|Population|Condition|Steps|Mean process|Positive fraction|Mean awareness|Mean datalink-only|\n|---|---|---|---|---|---|---|---|\n'
    for r in summary['mav_support']:
        if r['state'] in ('MAV_direct_zero_team_positive','UAV_centroid_beyond_8km','engagement_centroid_beyond_8km'):
            report+=f'|{r["training_seed"]}|{r["population"]}|{r["state"]}|{r["steps"]}|{r["mean_process_reward"]}|{r["positive_reward_fraction"]}|{r["mean_awareness"]}|{r["mean_datalink_only"]}|\n'
    report+='\n### 解释边界\n单机击杀率是所有200局的比例，不仅获胜局；同时给出red/draw/blue分层。旧数据同时多人击杀缺唯一归因，不把attack-credit数误当额外死亡数。\n'
    report+='本轮60局与case反事实不能识别训练因果；不能因同追概率高便断言Blue不真实，也不能把zero-own-direct的正奖励等同于已证明MAV漂移导致失败。\n'
    report+='Datalink-only counts遵循当前接口flag（包括inactive UAV slot）；不是alive UAV实际支援量，更不等于MAV贡献。判断MAV独有信息贡献应使用MAV_exclusive_information_count；本轮不据此推断H11因果。\n'
    (out/'research_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__': main()
