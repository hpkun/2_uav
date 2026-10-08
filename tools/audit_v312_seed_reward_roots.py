"""Read-only seed/credit/geometry audit of frozen v3.12 vanilla checkpoints.

No trainer is constructed. Rewards are read at their real post-combat timestamp;
kill geometry is read at the real pre-combat boundary. Policy permutation and
weapon limits are labelled diagnostic interventions, never production edits.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
from itertools import permutations
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import audit_v312_cap_mechanisms as base
from tools.audit_v311_environment_rationality import install_weapon_variant
from algorithm.happo.evaluation import evaluate_actors,summarize_records
from env.mavuav import load_environment_config
UAVS=base.UAVS; RED=base.RED_IDS; BLUE=base.BLUE_IDS
CACHE_ONLY=False


def enriched_replay(actors,cfg,env_seed,action_seed,weapon='W0',device='cuda'):
    """Instance-local observer; no additional sample, reward or policy calls."""
    original_class=base.Env; instances=[]
    class TraceEnv(original_class):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw); self.audit_info=[]; instances.append(self)
            if weapon!='W0': install_weapon_variant(self,'W3' if weapon=='W1' else 'W1')
        def step(self,actions):
            pre_alive={a:bool(self.entities[a].state.alive) for a in RED}
            result=super().step(actions); info=result[-1]; extras={}
            for a in UAVS:
                pairs=[]
                if self.entities[a].state.alive:
                    for b in BLUE:
                        if self.entities[b].state.alive:
                            g=base.geometry(self.entities[a].state,self.entities[b].state)
                            pairs.append(dict(target=b,distance=g.distance,ATA=np.rad2deg(g.ata),AA=np.rad2deg(g.aa),
                                gate=int(all(base.conditions(g.distance,g.ata,g.aa,self.config['combat']))),
                                direct=int(self.direct_visible(a,b)),streak=self._attack_streak.get((a,b),0)))
                extras[a]=dict(pre_alive=pre_alive[a],post_alive=bool(self.entities[a].state.alive),post_pairs=pairs)
            self.audit_info.append(dict(info={k:v for k,v in info.items() if k!='episode_summary'},extra=extras,
                                        agent_rewards=result[1]))
            return result
    base.Env=TraceEnv
    try:
        ep=base.replay(actors,cfg,env_seed,action_seed,device)
        ep['reward_trace']=instances[0].audit_info
        ep['weapon']=weapon
        return ep
    finally: base.Env=original_class


def contributions(ep):
    counts={a:len(base.kill_sets(ep['events'])[a]) for a in UAVS}
    rows=[]
    for a in UAVS:
        pp=[r for r in ep['pairs'] if r['agent']==a]
        # Resource-limited diagnostic pairs must not reconstruct a geometric
        # increment for a blocked pair cleared by a simultaneous death.
        streak=lambda r:r['streak_stored_post'] if ep.get('weapon','W0')!='W0' else r['streak_evaluated']
        rows.append(dict(agent=a,kill_credits=counts[a],pair_steps=len(pp),gate_steps=sum(r['full_gate'] for r in pp),
            streak1=sum(streak(r)>=1 for r in pp),streak2=sum(streak(r)>=2 for r in pp),
            streak3=sum(streak(r)>=3 for r in pp),process_sum=ep['result'][f'{a.lower()}_process_reward_sum'],
            shared_event_sum=ep['result']['event_reward_sum'],shared_terminal_sum=ep['result']['terminal_reward_sum']))
    return rows


def aggregate_contributions(eps,tag):
    stats=summarize_records([e['result'] for e in eps]); rows=[]
    counts={a:sum(len(base.kill_sets(e['events'])[a]) for e in eps) for a in UAVS}
    total=sum(counts.values()); dominant=max(UAVS,key=lambda a:counts[a]) if total else None
    for a in UAVS:
        rr=[r for e in eps for r in contributions(e) if r['agent']==a]; exposures=sum(r['pair_steps'] for r in rr)
        active_shared=sum(trace['info']['event_reward']+trace['info']['terminal_reward']+trace['info']['safety_reward']
            for ep in eps for trace in ep['reward_trace'] if trace['extra'][a]['pre_alive'])
        active_kill_exposures=sum(1 for ep in eps for step,trace in enumerate(ep['reward_trace'],1) if trace['extra'][a]['pre_alive']
            for target in {e['target'] for e in ep['events'] if e['step']==step and e['target'] in BLUE})
        rows.append(dict(**tag,agent=a,episodes=len(eps),dominant_agent=dominant,contribution_share=base.ratio(counts[a],total),
            mean_kill_credits=counts[a]/len(eps),full_gate_fraction=base.ratio(sum(r['gate_steps'] for r in rr),exposures),
            streak1_fraction=base.ratio(sum(r['streak1'] for r in rr),exposures),streak2_fraction=base.ratio(sum(r['streak2'] for r in rr),exposures),
            mean_process_sum=base.mean([r['process_sum'] for r in rr]),mean_shared_event_sum=stats['mean_event_reward_sum'],
            mean_active_shared_sum=active_shared/len(eps),active_shared_kill_exposures=active_kill_exposures,
            **{k:stats[k] for k in ['red_win_rate','draw_rate','mean_red_attack_kills','mean_episode_length']}))
    return rows


def future_labels(ep,a,step,horizon):
    """Strict future t+1..t+h, episode-safe; missing horizon is right-censored."""
    length=ep['result']['episode_length']; end=step+horizon
    if end>length: return None
    pp=[r for r in ep['pairs'] if r['agent']==a and step<r['step']<=end]
    return dict(gate=int(any(r['full_gate'] for r in pp)),streak=int(any(r['streak_evaluated']>=1 for r in pp)),
                kill=int(any(e['attacker']==a and e['target'] in BLUE and step<e['step']<=end for e in ep['events'])))


def quintile_groups(rows):
    """Value-cut bins: tied reward values never split to manufacture monotonicity."""
    values=np.asarray([r['process'] for r in rows]); edges=np.quantile(values,[.2,.4,.6,.8])
    result=[[] for _ in range(5)]
    for r in rows: result[int(np.searchsorted(edges,r['process'],side='left'))].append(r)
    return result,edges.tolist()


def reward_analyses(finals,cfg):
    decomposition=[]; free=[]; alignment=[]; futures=[]; behaviours=[]; attributed=[]; carry=[]; observations=[]; process_steps=[]
    for seed,eps in finals.items():
        for a in UAVS:
            exposure=Counter(); alignment_count=Counter(); future_rows=[]; behaviour_rows=[]
            for ep in eps:
                kills=[e for e in ep['events'] if e['target'] in BLUE]; bystep=defaultdict(dict)
                for p in ep['pairs']: bystep[p['step']][p['agent'],p['target']]=p
                last_target=None
                sums=Counter(); actual=0.; r1=0.; r2=0.
                for i,trace in enumerate(ep['reward_trace']):
                    step=i+1; info=trace['info']; ex=trace['extra'][a]; post=ex['post_pairs']
                    process=float(info[f'{a.lower()}_process_reward']); actual+=trace['agent_rewards'][a]
                    selected=info[f'reward_target_{a}']; postgates=[p['target'] for p in post if p['gate']]
                    positive=max(trace['agent_rewards'][a],0)
                    if ex['pre_alive']: exposure['total_positive_reward']+=positive
                    for p in post:
                        if p['gate']:
                            alignment_count['post_gate_pairs']+=1; alignment_count['post_gate_selected']+=p['target']==selected
                    if postgates:
                        alignment_count['steps_any_post_gate']+=1; alignment_count['missed_gate']+=selected not in postgates
                    if info[f'reward_target_switch_{a}']:
                        alignment_count['switches']+=1
                        old=bystep[step].get((a,last_target))
                        alignment_count['switch_streak_break_cooccurrence']+=bool(old and old['streak_before']>0 and old['streak_evaluated']==0)
                    nowkill={e['target'] for e in kills if e['step']==step}
                    own={e['target'] for e in kills if e['step']==step and e['attacker']==a}
                    alignment_count['own_kill_credits']+=len(own)
                    alignment_count['kill_matches_prior_selector']+=sum(b==last_target for b in own)
                    alignment_count['kill_matches_post_selector']+=sum(b==selected for b in own)
                    noncontrib=[]
                    for b in nowkill:
                        pair=bystep[step].get((a,b)); killer=b in own
                        gate=bool(pair and pair['full_gate'])
                        non=int(not killer and not gate)
                        observations.append(dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],step=step,agent=a,killed_Blue=b,
                            alive_pre=int(ex['pre_alive']),killer=int(killer),full_gate=int(gate),distance_m=pair['distance_m'] if pair else None,
                            ATA_deg=pair['ATA_deg'] if pair else None,direct_visible=pair['direct_visible'] if pair else None,
                            own_process_reward=process,shared_event_reward=info['event_reward'],team_reward=info['team_reward'],
                            shared_kill_component=cfg['reward']['blue_kill'],non_contributor=non))
                        if ex['pre_alive']:
                            exposure['kill_exposures']+=1; exposure['non_contributor_exposures']+=non
                            exposure['non_contributor_kill_reward']+=non*cfg['reward']['blue_kill']
                        noncontrib.append(non)
                    # R1 splits simultaneous shared credit equally among actual killers
                    # per unique Blue. Shared losses/terminal/safety remain untouched.
                    credited=sum(cfg['reward']['blue_kill']/len({e['attacker'] for e in kills if e['step']==step and e['target']==b}) for b in own)
                    virtualgate=sum(p['gate'] for p in post)*.5
                    r1+=trace['agent_rewards'][a]-len(nowkill)*cfg['reward']['blue_kill']+credited
                    r2+=trace['agent_rewards'][a]+virtualgate
                    for label,val in [('process',process),('event',info['event_reward']),('terminal',info['terminal_reward']),('safety',info['safety_reward'])]:
                        sums[label]+=val; sums[label+'_absolute']+=abs(val)
                        if ex['pre_alive']:
                            sums['active_'+label]+=val; sums['active_'+label+'_absolute']+=abs(val)
                    if ex['pre_alive'] and ex['post_alive'] and ep['steps'][i]['alive_blue_post']>0:
                        selected_pair=next((p for p in post if p['target']==selected),None)
                        closest=min(post,key=lambda p:p['distance']) if post else None
                        state='A' if postgates else ('B' if closest and closest['distance']<=3000 else ('C' if any(p['direct'] for p in post) else 'D'))
                        item=dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],step=step,agent=a,process=process,
                            dense=info[f'{a.lower()}_R_AD'],gate_reward=info[f'{a.lower()}_R_gate'],selected_target=selected,
                            distance_m=selected_pair['distance'] if selected_pair else None,ATA_deg=selected_pair['ATA'] if selected_pair else None,
                            AA_deg=selected_pair['AA'] if selected_pair else None,full_gate=int(bool(selected_pair and selected_pair['gate'])),
                            attack_streak=selected_pair['streak'] if selected_pair else 0,behaviour=state,
                            future5=future_labels(ep,a,step,5),future10=future_labels(ep,a,step,10))
                        future_rows.append(item); behaviour_rows.append(item)
                        process_steps.append({k:json.dumps(v) if isinstance(v,dict) else v for k,v in item.items()})
                    last_target=selected
                attributed.append(dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],agent=a,
                    actual_return=actual,R1_contribution_attributed_return=r1,R2_actual_plus_virtual_gate_return=r2,
                    R2_virtual_gate_bonus_unit=.5,**dict(sums)))
                times=sorted(e['step'] for e in kills if e['attacker']==a)
                carry.append(dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],agent=a,kill_credits=len(times),
                    first_kill=times[0] if times else None,second_kill=times[1] if len(times)>1 else None,
                    third_kill=times[2] if len(times)>2 else None,fourth_kill=times[3] if len(times)>3 else None,
                    kill_intervals=json.dumps(np.diff(times).tolist())))
            rr=[r for r in attributed if r['seed']==seed and r['agent']==a]
            total_abs=sum(sum(r[k+'_absolute'] for k in ['process','event','terminal','safety']) for r in rr)
            decomposition.append(dict(seed=seed,agent=a,episodes=len(eps),**{f'mean_{k}_sum':base.mean([r[k] for r in rr]) for k in ['process','event','terminal','safety']},
                **{f'mean_{k}_absolute_sum':base.mean([r[k+'_absolute'] for r in rr]) for k in ['process','event','terminal','safety']},
                **{f'mean_active_{k}_absolute_sum':base.mean([r.get('active_'+k+'_absolute',0.) for r in rr]) for k in ['process','event','terminal','safety']},
                process_absolute_fraction=base.ratio(sum(r['process_absolute'] for r in rr),total_abs)))
            free.append(dict(seed=seed,agent=a,**dict(exposure),non_contributor_fraction=base.ratio(exposure['non_contributor_exposures'],exposure['kill_exposures']),
                non_contributor_shared_kill_over_total_positive=base.ratio(exposure['non_contributor_kill_reward'],exposure['total_positive_reward'])))
            alignment.append(dict(seed=seed,agent=a,**dict(alignment_count),
                selected_given_post_gate_fraction=base.ratio(alignment_count['post_gate_selected'],alignment_count['post_gate_pairs']),
                missed_gate_step_fraction=base.ratio(alignment_count['missed_gate'],alignment_count['steps_any_post_gate']),
                kill_prior_selector_match_fraction=base.ratio(alignment_count['kill_matches_prior_selector'],alignment_count['own_kill_credits'])))
            bins,edges=quintile_groups(future_rows)
            for q,rr in enumerate(bins):
                futures.append(dict(seed=seed,agent=a,quintile=q+1,value_cut_edges=json.dumps(edges),samples=len(rr),mean_process=base.mean([r['process'] for r in rr]),
                    **{f'future{h}_{metric}_probability':base.mean([r[f'future{h}'][metric] for r in rr if r[f'future{h}'] is not None]) for h in (5,10) for metric in ('gate','streak','kill')},
                    **{f'future{h}_eligible_samples':sum(r[f'future{h}'] is not None for r in rr) for h in (5,10)}))
            for state in 'ABCD':
                rr=[r for r in behaviour_rows if r['behaviour']==state]
                behaviours.append(dict(seed=seed,agent=a,behaviour=state,samples=len(rr),mean_process=base.mean([r['process'] for r in rr]),
                                       positive_process_fraction=base.mean([r['process']>0 for r in rr]),team_advantage_proxy='DATA_UNAVAILABLE'))
        mav_sums=[]
        for ep in eps:
            totals=Counter()
            for trace in ep['reward_trace']:
                info=trace['info']
                for key,value in [('process',info['mav_process_reward']),('event',info['event_reward']),('terminal',info['terminal_reward']),('safety',info['safety_reward'])]:
                    totals[key]+=value; totals[key+'_absolute']+=abs(value)
            mav_sums.append(totals)
        decomposition.append(dict(seed=seed,agent='MAV',episodes=len(eps),
            **{f'mean_{k}_sum':base.mean([r[k] for r in mav_sums]) for k in ['process','event','terminal','safety']},
            **{f'mean_{k}_absolute_sum':base.mean([r[k+'_absolute'] for r in mav_sums]) for k in ['process','event','terminal','safety']}))
    return dict(reward_decomposition=decomposition,free_rider_exposure=free,process_reward_future_success=futures,reward_target_alignment=alignment,
        diagnostic_reward_attribution=attributed,weapon_carry_analysis=carry,behaviour_reward=behaviours,kill_credit_step_records=observations,process_reward_step_records=process_steps)


def early_geometry_and_cap(finals):
    early=[]; counts=Counter(); totals=Counter()
    for seed,eps in finals.items():
        for ep in eps:
            for a in UAVS:
                pp=[r for r in ep['pairs'] if r['agent']==a]
                first=lambda predicate: min((r['step'] for r in pp if predicate(r)),default=None)
                early.append(dict(seed=seed,environment_seed=ep['steps'][0]['environment_seed'],agent=a,
                    first_direct_visibility=first(lambda r:r['direct_visible']),first_within5km=first(lambda r:r['distance_m']<=5000),
                    first_within3km=first(lambda r:r['distance_m']<=3000),first_full_gate=first(lambda r:r['full_gate'])))
            for b in BLUE:
                rows=sorted((r for r in ep['nav'] if r['blue']==b and r['step']<=10),key=lambda r:r['step'])
                for old,new in zip(rows,rows[1:]):
                    if new['step']==old['step']+1:
                        counts[seed,b,old['target'],new['target']]+=1; totals[seed,b,old['target']]+=1
    transition=[dict(seed=s,Blue=b,from_Red=a,to_Red=z,count=n,source_exposures=totals[s,b,a],conditional_probability=n/totals[s,b,a])
                for (s,b,a,z),n in sorted(counts.items())]
    return early,transition


def root_cause_rows():
    """Evidence classification, not a causal estimator or algorithm proposal."""
    hypotheses=[
        ('H1','Numeric seed ID is the cause','REFUTED','SOURCE-CODE FACT','Seed labels identify random streams; no physical ordering or seed-number correlation.'),
        ('H2','Random streams trigger different early trajectories','PARTIALLY_SUPPORTED','MEASURED / INFERENCE','Attack discovery differs strongly; initialization vs rollout contributions cannot be isolated retrospectively.'),
        ('H3','Shared team credit supports kill non-contributor exposure','PARTIALLY_SUPPORTED','SOURCE-CODE FACT / MEASURED','Identical shared kill components dominate non-attacker rewards; original actor gradients and causal free-riding unavailable.'),
        ('H4','Unlimited weapon abstraction enables single-UAV carry','PARTIALLY_SUPPORTED','SOURCE-CODE FACT / DIAGNOSTIC COUNTERFACTUAL','Unlimited resolver and multi-kill episodes verified; resource limits reveal frozen-policy dependence, not retraining outcomes.'),
        ('H5','Process reward is sufficiently aligned with future attack','PARTIALLY_SUPPORTED','MEASURED','Strong quantile alignment in attackers, no gate/kill in seed2 inactive attackers; conditional alignment does not ensure discovery.'),
        ('H6','Reward selector has meaningful target mismatch','PARTIALLY_SUPPORTED','MEASURED','Surviving-gate and prior-selector mismatch measured; post-kill selector comparison is invalid and selector does not reset streak.'),
        ('H7','Independent actors spontaneously break symmetry','PARTIALLY_SUPPORTED','SOURCE-CODE FACT / MEASURED','Learned policy specialization verified by actor permutation; slots and ordered observations are not exactly exchangeable.'),
        ('H8','Fixed slot geometry creates systematic combat advantage','PARTIALLY_SUPPORTED','MEASURED','Middle and outer initial distributions differ; outer UAV1/UAV3 are close, no universal UAV3 initial advantage established.'),
        ('H9','CAP assignment creates slot-specific exposure','PARTIALLY_SUPPORTED','MEASURED','Strong Blue-slot identity association with equal initial loads; attack-learning causal bias not established.'),
        ('H10','First training kill creates a reinforcing loop','INCONCLUSIVE','INFERENCE','First-kill training episode +/-50 trajectories unavailable; seed3 checkpoint dominance changes.'),
        ('H11','Multiple learned attractor basins exist','PARTIALLY_SUPPORTED','MEASURED / INFERENCE','Distinct one- and two-attacker policies observed; true stable optimization attractors not established.'),
        ('H12','Late discovery seed is not mature at 2M','PARTIALLY_SUPPORTED','MEASURED','Seed2 discovers later and improves through 2M; convergence or eventual catch-up not measured.'),
        ('H13','Optimization is the primary cause','INCONCLUSIVE','INFERENCE','No isolated optimization intervention; finite losses and declining std do not identify primary cause.'),
        ('H14','Environment/reward interaction is the primary cause','INCONCLUSIVE','INFERENCE','Joint amplification is supported but cannot rank causal importance against optimization without controlled training.'),
    ]
    return [dict(hypothesis_id=i,hypothesis=h,status=s,evidence_type=t,evidence=e) for i,h,s,t,e in hypotheses]


def slot_audit(cfg,count):
    env=base.Env(cfg,profile='main'); records=[]; counts=Counter()
    for i in range(count):
        env.reset(seed=1000+i)
        for b in BLUE: counts[b,env.blue_policy._guidance_state[b].target_id]+=1
        for a in UAVS:
            s=env.entities[a].state; assigned=next(b for b in BLUE if env.blue_policy._guidance_state[b].target_id==a)
            g=base.geometry(s,env.entities[assigned].state)
            records.append(dict(environment_seed=1000+i,agent=a,x=s.x,y=s.y,h=s.h,speed=s.v,heading_deg=np.rad2deg(s.psi),assigned_Blue=assigned,
                assigned_distance_m=g.distance,ATA_deg=np.rad2deg(g.ata),AA_deg=np.rad2deg(g.aa),
                closure_mps=np.dot(base.velocity(s)-base.velocity(env.entities[assigned].state),(base.position(env.entities[assigned].state)-base.position(s))/g.distance),
                **{f'distance_{b}_m':base.distance(s,env.entities[b].state) for b in BLUE}))
    matrix=[dict(Blue=b,Red=a,resets=count,assignments=counts[b,a],fraction=counts[b,a]/count) for b in BLUE for a in RED]
    stats=[]
    for a in UAVS:
        rr=[r for r in records if r['agent']==a]
        for k in ['x','y','h','speed','heading_deg','assigned_distance_m','ATA_deg','AA_deg','closure_mps']:
            values=[r[k] for r in rr]
            stats.append(dict(agent=a,feature=k,resets=count,mean=float(np.mean(values)),std=float(np.std(values)),p10=float(np.percentile(values,10)),p50=float(np.median(values)),p90=float(np.percentile(values,90))))
    return records,matrix,stats


def learning_data(runs):
    milestones=[]; phases=[]
    for seed,run in runs.items():
        rows=base.read_csv(run/'training.csv')
        for metric,thresholds in [('mean_red_attack_kills',[.1,.5,1,2,3]),('red_win_rate',[.05,.2,.4,.6])]:
            for t in thresholds:
                hits=[i for i,r in enumerate(rows) if float(r[metric])>=t]
                sustained=[i for i in hits if i+1<len(rows) and float(rows[i+1][metric])>=t]
                milestones.append(dict(seed=seed,metric=metric,threshold=t,first_sampled_steps=int(rows[hits[0]]['sampled_steps']) if hits else None,
                    sustained_two_steps=int(rows[sustained[0]]['sampled_steps']) if sustained else None))
        phases.extend(base.training_phases(run,f'seed{seed}'))
    return milestones,phases


def write_gzip(path,data):
    def clean(value):
        if isinstance(value,np.ndarray): return clean(value.tolist())
        if isinstance(value,np.generic): return clean(value.item())
        if isinstance(value,float) and not np.isfinite(value): return None
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v) for v in value]
        return value
    temporary=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=2) as f:
        # Some unused environment diagnostic distances use inf as "no target".
        # JSON null preserves missingness; actions/rewards are never synthesized.
        json.dump(clean(data),f,allow_nan=False)
    temporary.replace(path)


def cached_replays(actors,cfg,out,label,n,weapon='W0',seeds=None):
    path=out/'raw'/f'{label}.json.gz'
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as f: return json.load(f)
    if CACHE_ONLY: raise FileNotFoundError(f'postprocess requires existing cache: {path}')
    eps=[]; start=time.monotonic()
    for i in range(n):
        s=seeds[i] if seeds else 1000+i
        eps.append(enriched_replay(actors,cfg,s,s+1000,weapon))
        if (i+1)%10==0: print(f'{label}: {i+1}/{n}, {time.monotonic()-start:.1f}s',flush=True)
    write_gzip(path,eps); return eps


def main():
    global CACHE_ONLY
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--postprocess-only',action='store_true'); args=p.parse_args()
    CACHE_ONLY=args.postprocess_only
    out=Path(args.output); runs={i:ROOT/f'outputs/happo_v312_cap_seed{i}_2m' for i in (1,2,3)}
    if not args.postprocess_only and not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    out.mkdir(parents=True,exist_ok=True); (out/'raw').mkdir(exist_ok=True)
    files=[f for r in runs.values() for f in r.iterdir() if f.suffix in ('.pt','.csv','.json','.yaml','.log')]
    files.extend(f for folder in ['algorithm','env','configs'] for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.yaml'))
    hashes={str(f):base.sha(f) for f in files}; meta=out/'input_contract.json'
    if meta.exists() and json.loads(meta.read_text())!=hashes: raise ValueError('input artifacts differ from cached audit contract')
    if not meta.exists(): base.dump(meta,hashes)
    finals={}; checkpoints=[]; permutation_rows=[]; weapons=[]; weapon_episodes=[]; formal=[]; cfg=None; reference_training=None; reference_environment=None
    start=time.monotonic()
    for seed,run in runs.items():
        d=torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False); cfg=d['environment_config']
        assert cfg['environment_version']=='heterogeneous_mavuav_4v4_v3_12' and d['sampled_steps']==2_000_000
        base.validate_checkpoint_contract(d,cfg)
        public_training={k:v for k,v in d['trainer_config'].items() if k!='seed'}
        if reference_training is None: reference_training=public_training; reference_environment=cfg
        assert public_training==reference_training,'cross-seed training contract drift'
        assert cfg==reference_environment,'cross-seed environment contract drift'
        formal.append({**base.read_csv(run/'evaluations.csv')[-1],'seed':seed})
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))): net=base.IndependentActors(hidden_dim=d['trainer_config']['hidden_dim']).cuda().eval()
        net.load_state_dict(d['actors']); frozen={k:v.clone() for k,v in net.state_dict().items()}
        finals[seed]=cached_replays(net,cfg,out,f'seed{seed}_final',50)
        for filename in ['checkpoint_501760.pt','checkpoint_1001472.pt','checkpoint_1501184.pt','checkpoint_final.pt']:
            pd=torch.load(run/filename,map_location='cpu',weights_only=False)
            assert pd['environment_config']==cfg and pd['trainer_config']==d['trainer_config']
            net.load_state_dict(pd['actors']); steps=int(pd['sampled_steps'])
            eps=finals[seed][:20] if steps==2_000_000 else cached_replays(net,cfg,out,f'seed{seed}_{steps}',20)
            checkpoints.extend(aggregate_contributions(eps,dict(seed=seed,sampled_steps=steps)))
        net.load_state_dict(d['actors']); original=list(net.actors)
        for perm in permutations((1,2,3)):
            for slot,source in enumerate(perm,1): net.actors[slot]=original[source]
            eps=finals[seed][:20] if perm==(1,2,3) else cached_replays(net,cfg,out,f'seed{seed}_perm_'+''.join(map(str,perm)),20)
            permutation_rows.extend(aggregate_contributions(eps,dict(seed=seed,permutation=''.join(map(str,perm)),actor_source_by_slot=json.dumps(dict(zip(UAVS,perm))))))
        for i,actor in enumerate(original): net.actors[i]=actor
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in frozen.items())
        candidates=[e for e in finals[seed] if max(len(s) for s in base.kill_sets(e['events']).values())>=3][:10]
        for mode in ('W0','W1','W2'):
            eps=candidates if mode=='W0' else cached_replays(net,cfg,out,f'seed{seed}_{mode}',len(candidates),mode,[e['steps'][0]['environment_seed'] for e in candidates])
            if eps:
                weapons.extend(aggregate_contributions(eps,dict(seed=seed,weapon=mode)))
                for original_episode,episode in zip(candidates,eps):
                    weapon_episodes.append(dict(seed=seed,weapon=mode,environment_seed=episode['steps'][0]['environment_seed'],
                        action_seed=episode['steps'][0]['action_seed'],original_outcome=original_episode['result']['outcome'],
                        win_to_draw=int(original_episode['result']['outcome']=='red' and episode['result']['outcome']=='draw'),
                        **episode['result']))
        assert all(torch.equal(v,net.state_dict()[k]) for k,v in frozen.items())
        print(f'seed{seed} complete; cumulative elapsed {time.monotonic()-start:.1f}s',flush=True)
    # --postprocess-only requires the cache; cached calls above load, never replay.
    slots,matrix,slotstats=slot_audit(cfg,1000)
    milestones,phases=learning_data(runs)
    tables=reward_analyses(finals,cfg)
    early,transitions=early_geometry_and_cap(finals)
    tables.update(early_geometry_times=early,cap_first10_assignment_transition=transitions,root_cause_matrix=root_cause_rows())
    tables.update(seed_learning_milestones=milestones,training_phases=phases,checkpoint_uav_contributions=checkpoints,
        actor_permutation=permutation_rows,slot_geometry_bias=slotstats,slot_reset_records=slots,cap_assignment_matrix=matrix,weapon_counterfactual=weapons,
        weapon_counterfactual_episodes=weapon_episodes,formal_existing200=formal)
    for name,rows in tables.items(): base.table(out/f'{name}.csv',rows)
    final_stats={seed:summarize_records([e['result'] for e in es]) for seed,es in finals.items()}
    assert hashes=={str(f):base.sha(f) for f in files},'source files changed'
    base.dump(out/'summary.json',dict(protocol=dict(final50=50,checkpoint20=20,permutation20=20,reset_only=1000,weapon_matched_max=10,
        profile='main',environment_seed_start=1000,action_seed_start=2000,device='cuda'),final50=final_stats,
        source_SHA256=hashes,source_unchanged=True,actors_unchanged=True,elapsed_seconds=time.monotonic()-start,
        invocation_mode='cache_only_postprocess' if args.postprocess_only else 'collect_or_reuse',
        data_unavailable=['Training first-kill episode +/-50 trajectories','Actual training advantage on replay states'],
        reward_timing='process/selector post-combat; combat gates pre-combat; kill alignment uses prior selector',
        weapon_mapping={'W0':'current','W1':'5-step cooldown (k+1..k+5 blocked)','W2':'max2 unique kills per UAV'},
        R2='diagnostic metric only: actual individual return +0.5 per own surviving-Blue full-gate pair-step',
        training_contract_equal_except_seed=True,environment_contract_equal=True,
        final_uav_contributions=[r for s,es in finals.items() for r in aggregate_contributions(es,dict(seed=s))],
        next_single_variable_diagnostic='Seed2 training budget 2M -> 3M only; proposed, not executed',
        root_cause_matrix=root_cause_rows(),
        limitations=['50/20-episode mechanism samples are not replacement formal200 evaluations',
            'Future windows right-censored at episode boundary; overlapping windows are not independent',
            'Actor permutations change ordered teammate observation meaning and team context',
            'Main resets do not reproduce the earlier interpolated training curriculum',
            'Kill non-contribution does not exclude indirect support or decoy contribution',
            'Checkpoint replay is not original first-kill training history']))
    figures(out,tables)
    print(json.dumps(final_stats,indent=2),flush=True)


def figures(out,tables):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    def save(fig,name): fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for seed in (1,2,3):
        rows=base.read_csv(ROOT/f'outputs/happo_v312_cap_seed{seed}_2m/training.csv')
        for ax,metric in zip(axs,['mean_red_attack_kills','red_win_rate']):
            x=[int(r['sampled_steps'])/1e6 for r in rows]; y=[float(r[metric]) for r in rows]
            smooth=np.convolve(y,np.ones(25)/25,mode='valid'); ax.plot(x[24:],smooth,label=f'seed{seed}')
            ax.set(xlabel='sampled steps (M)',ylabel=metric); ax.legend(); ax.grid(alpha=.2)
    save(fig,'seed_attack_discovery')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,seed in zip(axs,(1,2,3)):
        for a in UAVS:
            rr=[r for r in tables['checkpoint_uav_contributions'] if r['seed']==seed and r['agent']==a]
            ax.plot([r['sampled_steps']/1e6 for r in rr],[r['mean_kill_credits'] for r in rr],marker='o',label=a)
        ax.set(title=f'seed{seed}',xlabel='sampled steps (M)',ylabel='mean kill credits (20 episodes)'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'uav_contribution_over_training')
    fig,ax=plt.subplots(figsize=(11,4)); rows=tables['reward_decomposition']; xx=np.arange(len(rows))
    ax.bar(xx-.2,[r['mean_event_absolute_sum']+r['mean_terminal_absolute_sum'] for r in rows],.4,label='shared event + terminal (raw absolute)')
    ax.bar(xx+.2,[r['mean_process_absolute_sum'] for r in rows],.4,label='own process absolute')
    ax.set_xticks(xx,[f's{r["seed"]}/{r["agent"]}' for r in rows]); ax.legend(); save(fig,'shared_vs_process_reward')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,seed in zip(axs,(1,2,3)):
        for a in UAVS:
            rr=[r for r in tables['process_reward_future_success'] if r['seed']==seed and r['agent']==a]
            ax.plot([r['quintile'] for r in rr],[r['future10_kill_probability'] if r['future10_kill_probability'] is not None else np.nan for r in rr],marker='o',label=a)
        ax.set(title=f'seed{seed}',xlabel='process reward quintile (ties unsplit)',ylabel='P(own kill in next 10 steps)'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'process_reward_vs_future_kill')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,seed in zip(axs,(1,2,3)):
        rr=[r for r in tables['actor_permutation'] if r['seed']==seed and r['agent']=='UAV1']
        ax.bar([r['permutation'] for r in rr],[r['red_win_rate'] for r in rr]); ax.set(title=f'seed{seed}',ylabel='win (20 episodes)',xlabel='source actors in UAV1/2/3 slots')
    save(fig,'actor_permutation_effect')
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,metric in zip(axs,['assigned_distance_m','ATA_deg','closure_mps']):
        values=[[r[metric] for r in tables['slot_reset_records'] if r['agent']==a] for a in UAVS]
        ax.boxplot(values,showfliers=False); ax.set_xticks(range(1,4),UAVS); ax.set_ylabel(metric)
    save(fig,'slot_geometry_distributions')


if __name__=='__main__': main()
