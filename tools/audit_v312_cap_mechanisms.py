"""Frozen-policy CAP/legacy diagnostic cross replay; no training or source mutation.

Geometry is sampled post-physics/pre-combat. Navigation is the command actually
used, not a fresh selection. Streaks come from the real resolver, including its
kill event before dead-pair cleanup. 4vN means N alive Red; P2 means one Blue.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from tools.audit_v311_environment_rationality import (
    Env, RED_IDS, BLUE_IDS, UAVS, geometry, conditions, mean, ratio, distance,
    position, dominance, kill_sets, table, dump, sha, IndependentActors,
    validate_checkpoint_contract,
)
from algorithm.happo.evaluation import summarize_records


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def compare_environments(a, b):
    a, b = deepcopy(a), deepcopy(b)
    assert a.pop('environment_version').endswith('v3_11')
    assert b.pop('environment_version').endswith('v3_12')
    assert a['blue_policy'].pop('target_strategy') == 'nearest_red_aircraft'
    assert b['blue_policy'].pop('target_strategy') == 'coordinated_assignment'
    if a != b:
        raise ValueError('cross evaluation requires identical configs except version and Blue strategy')


def velocity(s):
    return s.v*np.array([np.cos(s.theta)*np.cos(s.psi), np.cos(s.theta)*np.sin(s.psi), np.sin(s.theta)])


def replay(actors, cfg, env_seed, action_seed, device='cuda'):
    """One action sample per actor/step, exact official evaluator ordering."""
    env = Env(cfg, profile='main'); obs, _ = env.reset(seed=env_seed)
    original = env._resolve_attacks
    pairs, steps, navrows, events = [], [], [], []
    captures = []
    pre = {}; previous_targets = {}; previous_alive = {}
    def observe():
        snapshot = []
        nav = {b: deepcopy(env.blue_policy._guidance_state[b]) for b in BLUE_IDS}
        for a in RED_IDS+BLUE_IDS:
            sa = env.entities[a].state
            if not sa.alive or a == 'MAV': continue
            for b in (BLUE_IDS if a in RED_IDS else RED_IDS):
                sb = env.entities[b].state
                if not sb.alive: continue
                g = geometry(sa, sb); dg, ag, bg = conditions(g.distance,g.ata,g.aa,cfg['combat'])
                r = dict(agent=a,target=b,distance_m=float(g.distance),ATA_deg=float(np.rad2deg(g.ata)),
                         AA_deg=float(np.rad2deg(g.aa)),distance_gate=int(dg),ATA_gate=int(ag),AA_gate=int(bg),
                         full_gate=int(dg and ag and bg),streak_before=int(env._attack_streak.get((a,b),0)),
                         navigation_target=nav[a].target_id if a in BLUE_IDS else nav[b].target_id,
                         direct_visible=int(env.direct_visible(a,b)) if a in UAVS else None,
                         closure_mps=float(np.dot(velocity(sa)-velocity(sb), (position(sb)-position(sa))/max(g.distance,1e-12))),
                         attacker_speed=float(sa.v),target_speed=float(sb.v))
                snapshot.append(r)
        ev, deaths = original()
        for r in snapshot:
            hit = any(e['attacker']==r['agent'] and e['target']==r['target'] for e in ev)
            r['kill_event']=int(hit)
            r['streak_evaluated']=int(cfg['combat']['hold_steps'] if hit else env._attack_streak.get((r['agent'],r['target']),0))
            # A simultaneous unrelated death can clear a non-killing pair. Preserve
            # its pre-cleanup geometric hold while retaining the actual stored value.
            r['streak_stored_post']=r['streak_evaluated']
            if not hit and (r['agent'] in deaths or r['target'] in deaths):
                r['streak_evaluated']=r['streak_before']+1 if r['full_gate'] else 0
        captures.append((snapshot,nav,ev,deaths)); return ev,deaths
    env._resolve_attacks = observe
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(action_seed)
        if torch.device(device).type=='cuda': torch.cuda.manual_seed_all(action_seed)
        while True:
            pre = {a:e.state.copy() for a,e in env.entities.items()}
            pre_streak = dict(env._attack_streak)
            with torch.no_grad():
                actions=np.stack([actor.sample(torch.as_tensor(obs[a],device=device).unsqueeze(0),deterministic=False)[0].squeeze(0).cpu().numpy()
                                  for actor,a in zip(actors.actors,RED_IDS)])
            obs,_,term,trunc,info=env.step(actions)
            pp,nav,ev,deaths=captures[-1]; step=env.step_count
            tag=dict(environment_seed=env_seed,action_seed=action_seed,step=step,
                     alive_red_pre=sum(pre[a].alive for a in RED_IDS),alive_blue_pre=sum(pre[b].alive for b in BLUE_IDS))
            lookup={(r['agent'],r['target']):r for r in pp}
            row=dict(**tag,alive_red_post=sum(e.state.alive for a,e in env.entities.items() if a in RED_IDS),
                     alive_blue_post=sum(env.entities[b].state.alive for b in BLUE_IDS),
                     event_reward=info['event_reward'],team_reward=info['team_reward'],death_causes=json.dumps(info['death_causes']),
                     attack_events=json.dumps(ev),winner=info['episode_summary']['outcome'] if term or trunc else '')
            pressure=Counter()
            for a,e in env.entities.items():
                for k,v in [('x',e.state.x),('y',e.state.y),('h',e.state.h),('speed',e.state.v),
                            ('heading_deg',np.rad2deg(e.state.psi)),('alive',int(e.state.alive))]: row[f'{a}_{k}']=float(v)
            for b in BLUE_IDS:
                if not pre[b].alive: continue
                target=nav[b].target_id; old=previous_targets.get(b)
                switch=int(previous_alive.get(b,False) and old is not None and old!=target)
                pressure[target]+=1
                r=lookup.get((b,target),{})
                oldr=lookup.get((b,old),{})
                oldg=geometry(pre[b],pre[old]) if old in RED_IDS and pre[old].alive else None
                nr=dict(**tag,blue=b,target=target,target_switch=switch,guidance_age=step-1-nav[b].last_refresh_step,
                        old_target=old,old_target_alive_pre=int(pre[old].alive) if old in RED_IDS else None,
                        old_distance_pre=oldg.distance if oldg else None,old_full_gate_pre=int(all(conditions(oldg.distance,oldg.ata,oldg.aa,cfg['combat']))) if oldg else None,
                        old_streak_pre=pre_streak.get((b,old),0),old_streak_evaluated=oldr.get('streak_evaluated'),
                        distance_decreasing=int(r.get('distance_m',float('inf')) < distance(pre[b],pre[target])) if target in RED_IDS and pre[target].alive and r else None,
                        **{k:r.get(k) for k in ('distance_m','ATA_deg','AA_deg','full_gate','streak_evaluated','closure_mps','attacker_speed','target_speed','kill_event')})
                navrows.append(nr)
                for k in ('target','target_switch','guidance_age','distance_m','full_gate','streak_evaluated'): row[f'{b}_{k}']=nr[k]
                previous_targets[b]=target
            previous_alive={b:bool(env.entities[b].state.alive) for b in BLUE_IDS}
            for a in RED_IDS: row[f'pressure_{a}']=pressure[a]
            row['load_1111']=int(tag['alive_red_pre']==4 and tag['alive_blue_pre']==4 and all(pressure[a]==1 for a in RED_IDS))
            for a in UAVS:
                candidates=[r for r in pp if r['agent']==a]
                nearest=min(candidates,key=lambda r:r['distance_m']) if candidates else {}
                row[f'{a}_nearest_Blue']=nearest.get('target')
                for k in ('distance_m','ATA_deg','AA_deg','full_gate','streak_evaluated'): row[f'{a}_nearest_{k}']=nearest.get(k)
                row[f'{a}_kill_contribution']=sum(e['attacker']==a for e in ev)
            steps.append(row)
            pairs.extend(dict(**tag,**r) for r in pp)
            events.extend(dict(**tag,**e,navigation_target=nav[e['attacker']].target_id if e['attacker'] in BLUE_IDS else None) for e in ev)
            if term or trunc:
                return dict(result=dict(info['episode_summary']),steps=steps,pairs=pairs,nav=navrows,events=events)


def dwell_lengths(rows):
    """Right-censored episode/phase boundary spells retained and labelled."""
    lengths=[]; current=0; prev=None
    for r in rows:
        key=(r['environment_seed'],r['blue'],r['target'])
        if prev!=key:
            if current: lengths.append(current)
            current=0
        current+=1; prev=key
    if current: lengths.append(current)
    return lengths


def switching(rows):
    rows=sorted(rows,key=lambda r:(r['environment_seed'],r['blue'],r['step']))
    ds=dwell_lengths(rows); switches=[r for r in rows if r['target_switch']]
    # Only pairs consecutive within the selected phase contribute denominator.
    valid=sum(a['environment_seed']==b['environment_seed'] and a['blue']==b['blue'] and a['step']+1==b['step'] for a,b in zip(rows,rows[1:]))
    eligible={(b['environment_seed'],b['blue'],b['step']) for a,b in zip(rows,rows[1:]) if a['environment_seed']==b['environment_seed'] and a['blue']==b['blue'] and a['step']+1==b['step']}
    return dict(target_switch_count=len(switches),valid_consecutive_steps=valid,
        target_switch_rate=ratio(sum((r['environment_seed'],r['blue'],r['step']) in eligible for r in switches),valid),
        mean_target_dwell_steps=mean(ds),median_target_dwell_steps=float(np.median(ds)) if ds else None,
        p10_dwell=float(np.percentile(ds,10)) if ds else None,p90_dwell=float(np.percentile(ds,90)) if ds else None,
        dwell_eq2_fraction=mean([d==2 for d in ds]),dwell_le4_fraction=mean([d<=4 for d in ds]),dwell_ge10_fraction=mean([d>=10 for d in ds]),
        switches_distance_lt5km=sum(r['old_distance_pre'] is not None and r['old_distance_pre']<5000 for r in switches),
        switches_distance_lt3km=sum(r['old_distance_pre'] is not None and r['old_distance_pre']<3000 for r in switches),
        switches_full_gate=sum(bool(r['old_full_gate_pre']) for r in switches),
        switches_streak1=sum(r['old_streak_pre']>=1 for r in switches),switches_streak2=sum(r['old_streak_pre']>=2 for r in switches),
        switch_positive_old_streak_lost=sum(r['old_streak_pre']>0 and r['old_streak_evaluated']==0 for r in switches),
        switch_positive_old_streak_retained=sum(r['old_streak_pre']>0 and r['old_streak_evaluated'] is not None and r['old_streak_evaluated']>0 for r in switches),
        switches_old_target_dead=sum(r['old_target_alive_pre']==0 for r in switches),
        switches_old_target_alive=sum(r['old_target_alive_pre']==1 for r in switches))


def funnel(pp):
    return dict(pair_steps=len(pp),distance_gate_fraction=mean([r['distance_gate'] for r in pp]),
        ATA_gate_fraction=mean([r['ATA_gate'] for r in pp]),AA_gate_fraction=mean([r['AA_gate'] for r in pp]),
        full_gate_fraction=mean([r['full_gate'] for r in pp]),
        **{f'streak_ge{k}_fraction':mean([r['streak_evaluated']>=k for r in pp]) for k in (1,2,3)},
        kill_credits=sum(r['kill_event'] for r in pp))


def summarize_mechanisms(episodes,group):
    ss=[r for e in episodes for r in e['steps']]; nn=[r for e in episodes for r in e['nav']]
    pp=[r for e in episodes for r in e['pairs']]; bp=[r for r in pp if r['agent'] in BLUE_IDS]
    stats={}; pressure=[]; mav=[]; switches=[]; combat=[]; red=[]; p2=[]; outcomes=[]
    for scope,chosen in [('all',ss),('4v4',[r for r in ss if r['alive_red_pre']==4 and r['alive_blue_pre']==4])]:
        pressure.append(dict(group=group,scope=scope,steps=len(chosen),load_1111_fraction=mean([r['load_1111'] for r in chosen]),
                             **{f'mean_pressure_{a}':mean([r[f'pressure_{a}'] for r in chosen]) for a in RED_IDS},
                             **{f'targeted_step_fraction_{a}':mean([r[f'pressure_{a}']>0 for r in chosen]) for a in RED_IDS},
                             dogpile_ge3_fraction=mean([max(r[f'pressure_{a}'] for a in RED_IDS)>=3 for r in chosen])))
    for target in RED_IDS:
        x=[r for r in nn if r['target']==target and r['distance_m'] is not None]
        mav.append(dict(group=group,target=target,pair_steps=len(x),alive_blue_step_fraction=ratio(len([r for r in nn if r['target']==target]),len(nn)),
            mean_distance_m=mean([r['distance_m'] for r in x]),median_distance_m=float(np.median([r['distance_m'] for r in x])) if x else None,
            distance_decreasing_fraction=mean([r['distance_decreasing'] for r in x]),mean_closure_mps=mean([r['closure_mps'] for r in x]),
            p10_closure_mps=float(np.percentile([r['closure_mps'] for r in x],10)) if x else None,p90_closure_mps=float(np.percentile([r['closure_mps'] for r in x],90)) if x else None,
            within_5km_fraction=mean([r['distance_m']<5000 for r in x]),within_3km_fraction=mean([r['distance_m']<3000 for r in x]),
            ATA_lt30_fraction=mean([r['ATA_deg']<30 for r in x]),AA_lt90_fraction=mean([r['AA_deg']<90 for r in x]),full_gate_fraction=mean([r['full_gate'] for r in x]),
            mean_Blue_speed=mean([r['attacker_speed'] for r in x]),mean_target_speed=mean([r['target_speed'] for r in x]),
            **{f'streak_ge{k}_fraction':mean([r['streak_evaluated']>=k for r in x]) for k in (1,2,3)},kill_credits=sum(r['kill_event'] for r in x)))
    for scope in ('all','4v4','4v3','4v2','4v1','P2'):
        subset=nn if scope=='all' else [r for r in nn if (r['alive_blue_pre']==1 if scope=='P2' else r['alive_red_pre']==int(scope[-1]))]
        for blue in ('all',)+BLUE_IDS:
            x=subset if blue=='all' else [r for r in subset if r['blue']==blue]
            switches.append(dict(group=group,scope=scope,blue=blue,**switching(x)))
    for target in ('all',)+RED_IDS:
        x=bp if target=='all' else [r for r in bp if r['navigation_target']==target]
        combat.append(dict(group=group,navigation_target=target,**funnel(x),
                           gate_on_navigation_fraction=ratio(sum(r['full_gate'] and r['target']==r['navigation_target'] for r in x),sum(r['full_gate'] for r in x)),
                           kill_on_navigation_fraction=ratio(sum(r['kill_event'] and r['target']==r['navigation_target'] for r in x),sum(r['kill_event'] for r in x))))
    cs=[list(map(len,kill_sets(e['events']).values())) for e in episodes]
    for a in UAVS:
        x=[r for r in pp if r['agent']==a]
        astep={}
        for r in x: astep.setdefault((r['environment_seed'],r['step']),[]).append(r)
        red.append(dict(group=group,agent=a,episodes=len(episodes),mean_unique_kill_credits=mean([len(kill_sets(e['events'])[a]) for e in episodes]),
                        contribution_share=ratio(sum(len(kill_sets(e['events'])[a]) for e in episodes),sum(sum(c) for c in cs)),
                        within_3km_fraction=mean([min(r['distance_m'] for r in rr)<3000 for rr in astep.values()]),
                        within_5km_fraction=mean([min(r['distance_m'] for r in rr)<5000 for rr in astep.values()]),
                        direct_visible_fraction=mean([any(r['direct_visible'] for r in rr) for rr in astep.values()]),**funnel(x)))
    stats['red_team']=dict(**{f'one_UAV_ge{k}_kills':mean([max(c)>=k for c in cs]) for k in (2,3,4)},
        **{f'attacking_UAVs_{k}_fraction':mean([sum(v>0 for v in c)==k for c in cs]) for k in (1,2,3)},
        mean_dominance_share=mean([dominance(c)['dominance_share'] for c in cs]),mean_kill_HHI=mean([dominance(c)['kill_HHI'] for c in cs]))
    entered=[e for e in episodes if any(r['alive_blue_pre']==1 for r in e['steps'])]
    stats['p2']=dict(episodes=len(episodes),entered=len(entered),win_fraction=mean([e['result']['outcome']=='red' for e in entered]),
        draw_fraction=mean([e['result']['outcome']=='draw' for e in entered]),
        mean_duration=mean([sum(r['alive_blue_pre']==1 for r in e['steps']) for e in entered]),
        mean_remaining_steps=mean([76-next(r['step'] for r in e['steps'] if r['alive_blue_pre']==1) for e in entered]),
        wins_skipping_P2=sum(e['result']['outcome']=='red' and not any(r['alive_blue_pre']==1 for r in e['steps']) for e in episodes))
    for a in UAVS:
        x=[r for r in pp if r['agent']==a and r['alive_blue_pre']==1]
        p2.append(dict(group=group,agent=a,**stats['p2'],**funnel(x)))
    for outcome in ('red','draw','blue'):
        for n in range(5):
            x=[e for e in episodes if e['result']['outcome']==outcome and e['steps'][-1]['alive_blue_post']==n]
            outcomes.append(dict(group=group,outcome=outcome,Blue_survivors=n,episodes=len(x),fraction=len(x)/len(episodes)))
    for k in range(1,5):
        kill_steps=[]
        for e in episodes:
            first={}
            for ev in e['events']:
                if ev['target'] in BLUE_IDS: first.setdefault(ev['target'],ev['step'])
            ordered=sorted(first.values())
            if len(ordered)>=k: kill_steps.append(ordered[k-1])
        stats[f'kill_{k}']=dict(reached=len(kill_steps),mean_step=mean(kill_steps))
    stats['blue_episode_gate_fraction']=mean([any(r['full_gate'] for r in e['pairs'] if r['agent'] in BLUE_IDS) for e in episodes])
    stats['blue_episode_streak2_fraction']=mean([any(r['streak_evaluated']>=2 for r in e['pairs'] if r['agent'] in BLUE_IDS) for e in episodes])
    stats['blue_episode_kill_fraction']=mean([e['result']['blue_attack_kills']>0 for e in episodes])
    p2_blue=[]
    for e in episodes:
        ss=e['steps']
        for i,r in enumerate(ss):
            if r['alive_blue_pre']!=1 or i==0: continue
            for b in BLUE_IDS:
                if ss[i-1][f'{b}_alive']:
                    delta=(r[f'{b}_heading_deg']-ss[i-1][f'{b}_heading_deg']+180)%360-180
                    p2_blue.append(abs(delta))
    stats['p2_Blue_mean_absolute_heading_change_deg']=mean(p2_blue)
    return stats,dict(blue_target_pressure=pressure,blue_mav_pursuit=mav,blue_target_switching=switches,blue_combat_effectiveness=combat,
                      red_uav_contribution_comparison=red,p2_comparison=p2,outcome_Blue_survivors=outcomes)


def training_phases(run,label):
    rows=read_csv(run/'training.csv'); result=[]
    previous=0
    for r in rows:
        cumulative=int(r['completed_episodes'])
        r['_episode_weight']=cumulative-previous
        if r['_episode_weight']<0: raise ValueError('completed episode counter reversed')
        previous=cumulative
    metrics=['mean_episode_return','red_win_rate','blue_win_rate','draw_rate','MAV_survival_rate','mean_UAV_survivors',
             'mean_red_attack_kills','mean_blue_attack_kills','mean_episode_length','entropy','critic_loss']+[f'actor_{i}_loss' for i in range(4)]
    for lo in range(0,2_000_000,400_000):
        selected=[r for r in rows if lo<int(r['sampled_steps'])<=lo+400_000]
        w=[r['_episode_weight'] for r in selected]; n=sum(w)
        result.append(dict(run=label,start=lo,end=lo+400_000,completed_episodes=n,windows=len(selected),
                           **{m:(sum(float(r[m])*v for r,v in zip(selected,w) if v)/n if n else None) for m in metrics}))
    return result


def recover_existing_v311_200(out,groups):
    prior=ROOT/'outputs/audits/v311_happo_failure_audit_200ep_20261007_141809'
    if not (prior/'replay_episodes.csv').exists(): return {'status':'missing'}
    select=lambda r:r['run']=='happo_v311_seed1_2m' and int(r['sampled_steps'])==2_000_000
    er=[r for r in read_csv(prior/'replay_episodes.csv') if select(r)]
    ar=[r for r in read_csv(prior/'replay_agents.csv') if select(r)]
    assert len(er)==200 and len(ar)==600
    lookup={int(r['environment_seed']):r for r in er}
    for e in groups['A']:
        r=lookup[e['steps'][0]['environment_seed']]
        for k in ('episode_return','episode_length','red_attack_kills','blue_attack_kills','red_uav_survivors'):
            assert float(r[k])==float(e['result'][k]),f'existing200 mismatch: {k}'
        assert r['outcome']==e['result']['outcome']
    table(out/'existing_v311_formal200_episode_records.csv',er)
    table(out/'existing_v311_formal200_agent_records.csv',ar)
    rows=[]
    for a in UAVS:
        rr=[r for r in ar if r['agent']==a]; n=sum(int(r['geometry_samples']) for r in rr)
        rows.append(dict(run='v311',source='existing200',agent=a,episodes=200,
            mean_kill_credits=mean([float(r['streak3']) for r in rr]),geometry_agent_steps=n,
            gate_any_Blue_fraction=sum(float(r['fraction_full_gate'])*int(r['geometry_samples']) for r in rr)/n if n else None,
            within3km_fraction=sum(float(r['fraction_within3km'])*int(r['geometry_samples']) for r in rr)/n if n else None,
            within5km_fraction=sum(float(r['fraction_within5km'])*int(r['geometry_samples']) for r in rr)/n if n else None,
            **{f'streak{k}_pair_steps':sum(int(r[f'streak{k}']) for r in rr) for k in (1,2,3)},direct_visible_fraction='missing'))
    table(out/'existing_v311_formal200_contribution.csv',rows)
    return dict(status='recovered',episodes=200,agent_rows=600,A50_exactly_matches_existing200=True,
                v312_per_agent200='missing',source_SHA256={str(prior/f):sha(prior/f) for f in ('replay_episodes.csv','replay_agents.csv')})


def matched_cases(groups,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    amap={e['steps'][0]['environment_seed']:e for e in groups['A']}; dmap={e['steps'][0]['environment_seed']:e for e in groups['D']}
    cases=[]
    for label,aout,dout in [('A','draw','red'),('B','red','red'),('C','draw','draw')]:
        candidates=[s for s in amap if amap[s]['result']['outcome']==aout and dmap[s]['result']['outcome']==dout]
        exact=bool(candidates)
        seed=min(candidates) if candidates else min(amap,key=lambda s:(int(amap[s]['result']['outcome']!=aout)+int(dmap[s]['result']['outcome']!=dout),s))
        dest=out/'matched_cases'/f'case_{label}'; dest.mkdir(parents=True,exist_ok=True)
        timelines=[]
        for version,ep in [('v311',amap[seed]),('v312',dmap[seed])]:
            table(dest/f'trajectory_{version}.csv',ep['steps'])
            timelines.extend(dict(version=version,**r) for r in ep['nav'])
            fig,ax=plt.subplots(figsize=(9,7)); ss=ep['steps']
            for a in RED_IDS+BLUE_IDS:
                ax.plot([r[f'{a}_x']/1000 for r in ss],[r[f'{a}_y']/1000 for r in ss],label=a,ls='-' if a in RED_IDS else '--')
                ax.scatter(ss[0][f'{a}_x']/1000,ss[0][f'{a}_y']/1000,s=20)
            for r in ep['nav']:
                if r['target_switch']:
                    sr=ss[r['step']-1]; ax.scatter(sr[f'{r["blue"]}_x']/1000,sr[f'{r["blue"]}_y']/1000,marker='|',c='grey',s=40)
            for e in ep['events']:
                r=ss[e['step']-1]; ax.scatter(r[f'{e["target"]}_x']/1000,r[f'{e["target"]}_y']/1000,marker='x',c='black',s=65)
            p2=next((r for r in ss if r['alive_blue_pre']==1),None)
            if p2:
                b=next(b for b in BLUE_IDS if (ss[p2['step']-2] if p2['step']>1 else p2)[f'{b}_alive'])
                ax.scatter(p2[f'{b}_x']/1000,p2[f'{b}_y']/1000,marker='s',s=70,facecolors='none',edgecolors='purple',label='P2 entry')
            ax.set(xlabel='x (km)',ylabel='y (km)',title=f'{version} seed {seed}: {ep["result"]["outcome"]}; | switch, x kill, square P2')
            ax.legend(ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(dest/f'XY_{version}.png',dpi=180); plt.close(fig)
        table(dest/'target_timeline.csv',timelines)
        fig,axes=plt.subplots(6,2,figsize=(14,16),sharex='col')
        for col,(version,e) in enumerate([('v311',amap[seed]),('v312',dmap[seed])]):
            for b in BLUE_IDS:
                nn=[r for r in e['nav'] if r['blue']==b]
                axes[0,col].step([r['step'] for r in nn],[RED_IDS.index(r['target']) if r['target'] in RED_IDS else -1 for r in nn],where='post',label=b)
                dwell=[]; age=0; old=None
                for r in nn:
                    age=age+1 if r['target']==old else 1
                    old=r['target']; dwell.append(age)
                axes[1,col].plot([r['step'] for r in nn],dwell,label=b)
                sw=[r['step'] for r in nn if r['target_switch']]
                axes[1,col].scatter(sw,[0]*len(sw),marker='|')
                bp=[r for r in e['pairs'] if r['agent']==b]
                bystep={r['step']:max(p['streak_evaluated'] for p in bp if p['step']==r['step']) for r in bp}
                axes[4,col].step(list(bystep),list(bystep.values()),label=b)
            ss=e['steps']; axes[2,col].step([r['step'] for r in ss],[r['alive_blue_post'] for r in ss])
            kills=[v for v in e['events'] if v['target'] in BLUE_IDS]
            axes[3,col].scatter([v['step'] for v in kills],[UAVS.index(v['attacker']) for v in kills])
            for (step,actor),count in Counter((v['step'],v['attacker']) for v in kills).items():
                if count>1: axes[3,col].annotate(f'x{count}',(step,UAVS.index(actor)),xytext=(4,5),textcoords='offset points')
            for a in UAVS:
                cum=np.cumsum([r[f'{a}_kill_contribution'] for r in ss]); axes[5,col].step([r['step'] for r in ss],cum,label=a)
            axes[0,col].set_title(version); axes[0,col].set_yticks(range(4),RED_IDS); axes[3,col].set_yticks(range(3),UAVS)
            for i,title in enumerate(['Blue navigation target','Target dwell age (markers: switch)','Alive Blue','Red kill events','Max Blue pair streak','Cumulative UAV kill credits']):
                axes[i,col].set_ylabel(title); axes[i,col].grid(alpha=.2)
            axes[5,col].set_xlabel('decision step'); axes[0,col].legend(); axes[5,col].legend()
        fig.tight_layout(); fig.savefig(dest/'mechanism_timelines.png',dpi=170); plt.close(fig)
        info=dict(case=label,environment_seed=seed,action_seed=seed+1000,exact_category=exact,
                  v311=amap[seed]['result'],v312=dmap[seed]['result'])
        dump(dest/'case_summary.json',info)
        p2notes=[]
        for v,e in [('v311',amap[seed]),('v312',dmap[seed])]:
            for a in UAVS:
                ps=[r for r in e['pairs'] if r['agent']==a and r['alive_blue_pre']==1]
                p2notes.append(f'- {v} {a}: P2 pair-steps={len(ps)}, full-gate steps={sum(r["full_gate"] for r in ps)}, mean distance={mean([r["distance_m"] for r in ps])}.')
        (dest/'case_report.md').write_text(f'# Matched case {label}\n\nEnv seed {seed}; action seed {seed+1000}; exact category: {exact}.\n\n'+
            '\n'.join(f'- {v}: outcome={e["result"]["outcome"]}, length={e["result"]["episode_length"]}, kills={e["result"]["red_attack_kills"]}; '+
                      'kill events '+str([(x['step'],x['attacker'],x['target']) for x in e['events'] if x['target'] in BLUE_IDS])
                      for v,e in [('v311',amap[seed]),('v312',dmap[seed])])+ '\n\n'+ '\n'.join(p2notes)+
            '\n\nOnly observed geometry and actual event credit are used; no explicit Red target action is inferred.\n',encoding='utf-8')
        cases.append(info)
    return cases


def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=50); p.add_argument('--env-seed',type=int,default=1000)
    p.add_argument('--action-seed',type=int,default=2000); p.add_argument('--device',default='cuda'); p.add_argument('--output',required=True)
    p.add_argument('--postprocess-only',action='store_true'); args=p.parse_args()
    out=Path(args.output)
    runs=[ROOT/'outputs/happo_v311_seed1_2m',ROOT/'outputs/happo_v312_cap_seed1_2m']
    if args.postprocess_only:
        groups={g:json.loads((out/f'raw_{g}.json').read_text()) for g in 'ABCD'}
        summary=json.loads((out/'summary.json').read_text())
    else:
        if args.device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no CPU fallback')
        out.mkdir(parents=True,exist_ok=False)
        source_paths=[r/f for r in runs for f in ('checkpoint_final.pt','training.csv','evaluations.csv','summary.json','resolved_config.yaml','run.log') if (r/f).exists()]
        hashes={str(f):sha(f) for f in source_paths}
        payloads=[torch.load(r/'checkpoint_final.pt',map_location='cpu',weights_only=False) for r in runs]
        cfgs=[x['environment_config'] for x in payloads]; compare_environments(*cfgs)
        if payloads[0]['trainer_config']!=payloads[1]['trainer_config']: raise ValueError('trainer configs differ')
        actors=[]
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for d,cfg in zip(payloads,cfgs):
                validate_checkpoint_contract(d,cfg)
                if int(d['sampled_steps'])!=2_000_000 or (d['actor_variant'],d['critic_variant'],d['method_variant'])!=('vanilla','mlp','baseline'): raise ValueError('expected exact 2M vanilla baseline')
                net=IndependentActors(hidden_dim=d['trainer_config']['hidden_dim'],log_std_init=d['trainer_config'].get('actor_log_std_init',-.5)).cuda().eval()
                net.load_state_dict(d['actors']); actors.append(net)
        frozen=[{k:v.clone() for k,v in a.state_dict().items()} for a in actors]
        groups={}; start=time.monotonic()
        for g,ai,ci in [('A',0,0),('B',0,1),('C',1,0),('D',1,1)]:
            eps=[]
            for i in range(args.episodes):
                eps.append(replay(actors[ai],cfgs[ci],args.env_seed+i,args.action_seed+i,args.device))
                if (i+1)%10==0: print(f'{g}: {i+1}/{args.episodes}, elapsed={time.monotonic()-start:.1f}s',flush=True)
            groups[g]=eps; dump(out/f'raw_{g}.json',eps)
        assert hashes=={str(f):sha(f) for f in source_paths},'source artifacts changed'
        assert all(torch.equal(a.state_dict()[k],v) for a,ss in zip(actors,frozen) for k,v in ss.items()),'actor parameters changed'
        formal=[read_csv(r/'evaluations.csv')[-1] for r in runs]
        for f in formal:
            assert int(f['episodes'])==200 and f['action_mode']=='stochastic' and int(f['action_seed'])==2000 and int(f['evaluation_environment_seed_start'])==1000 and f['evaluation_profile']=='main'
        summary=dict(protocol=dict(episodes_per_group=args.episodes,environment_seed_start=args.env_seed,action_seed_start=args.action_seed,profile='main',mode='stochastic',diagnostic_only=True),
            formal200=formal,source_SHA256=hashes,source_unchanged=True,actors_unchanged=True,training_configs_identical=True,
            environment_difference_only=['environment_version','blue_policy.target_strategy'],elapsed_seconds=time.monotonic()-start,
            missing=['v3.12 formal200 per-episode/per-UAV artifacts; D50 substitutes are labelled, not 200-episode statistics'])
        dump(out/'summary.json',summary)
    cross=[]; tables={}; mechanism={}
    for g,eps in groups.items():
        cross.append(dict(group=g,policy='P11' if g in 'AB' else 'P12',Blue='B11' if g in 'AC' else 'B12',**summarize_records([e['result'] for e in eps])))
        mechanism[g],tt=summarize_mechanisms(eps,g)
        for k,v in tt.items(): tables.setdefault(k,[]).extend(v)
        table(out/f'episode_records_{g}.csv',[dict(environment_seed=e['steps'][0]['environment_seed'],action_seed=e['steps'][0]['action_seed'],**e['result']) for e in eps])
    table(out/'cross_eval_2x2.csv',cross)
    for k,v in tables.items(): table(out/f'{k}.csv',v)
    table(out/'training_phase_comparison.csv',[r for run,label in zip(runs,['v311','v312']) for r in training_phases(run,label)])
    summary.update(cross_eval=cross,mechanisms=mechanism,matched_cases=matched_cases(groups,out),
                   existing200_recovery=recover_existing_v311_200(out,groups))
    # Interpretations are specific to these frozen seed1 policies, not a causal
    # estimator of opponent difficulty or a claim of cross-training-seed stability.
    conclusions=[
        ('H1','CAP removes pathological Blue dogpile','VERIFIED','medium','Exact 4v4 load 1/1/1/1; compare blue_target_pressure.csv.'),
        ('H2','CAP reduces Blue effective combat pressure','INCONCLUSIVE','high','A/D reduction is confounded by policy; fixed-policy B/A and D/C Blue kills increase.'),
        ('H3','one Blue is frequently wasted pursuing MAV','PARTIALLY_SUPPORTED','medium','Assigned-MAV gate zero for D; off-navigation UAV kills prevent claiming wholly wasted capacity.'),
        ('H4','MAV pursuit is combat-effective','REFUTED','medium','D assigned-MAV full gate, streak and MAV kill are zero; restricted to this replay.'),
        ('H5','CAP causes excessive target switching','REFUTED','low','See switching rate and long dwell distribution, not refresh frequency.'),
        ('H6','CAP destroys active Blue attack streaks','REFUTED','low','Resolver checks all pairs independently of assignment. Rare switch/break co-occurrence is not reset causality.'),
        ('H7','v3.12 Red policy generalizes back to legacy Blue','PARTIALLY_SUPPORTED','high','C retains nonzero wins but is below both D and A; no evidence of superior transfer.'),
        ('H8','v3.12 improvement is mostly environment-easiness','INCONCLUSIVE','high','B versus A refutes unconditional ease, but the causal share of learning versus policy-specific ease is not identifiable.'),
        ('H9','v3.12 improvement is mostly better learned Red policy','PARTIALLY_SUPPORTED','high','D over B supports better CAP-adapted policy, not overall transferable superiority (C below A).'),
        ('H10','v3.12 reduces single-UAV carry','PARTIALLY_SUPPORTED','medium','Dominance falls slightly but D remains 95% UAV3 kill credits.'),
        ('H11','v3.12 improves multi-UAV attack participation','PARTIALLY_SUPPORTED','medium','More two-contributor episodes, zero three-contributor episodes; small subordinate shares.'),
        ('H12','v3.12 specifically improves P2 last-Blue clearance','VERIFIED','high','Higher P2 conditional win and full gate; shorter P2, also more episodes reach P2.'),
        ('H13','CAP is a more reasonable fixed opponent than legacy nearest','PARTIALLY_SUPPORTED','medium','Balanced pressure fixes dogpile; mandatory MAV allocation efficiency remains limited.'),
        ('H14','CAP is too weak for formal future experiments','INCONCLUSIVE','high','Two fixed policies and one training seed do not identify general opponent strength.'),
    ]
    hypotheses=[dict(hypothesis=h,description=d,status=s,severity=v,evidence=e) for h,d,s,v,e in conclusions]
    table(out/'hypothesis_matrix.csv',hypotheses)
    summary['interpretation']=dict(overall_classification='MIXED',hypotheses=hypotheses,
                                   scope='seed1 fixed-policy diagnostic evidence; not cross-training-seed causality')
    dump(out/'summary.json',summary)
    print(json.dumps(cross,indent=2),flush=True)


if __name__=='__main__': main()
