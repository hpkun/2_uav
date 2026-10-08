"""Read-only final-checkpoint audit; one frozen stochastic 200-episode replay.

Instance-local observers capture resolver inputs/evaluated locks without changing
production code, state, RNG order or sampling. Rewards select targets AFTER the
synchronous deaths; primary alignment uses surviving post-cleanup locks. Killing
boundary pre-cleanup alignment is separately labelled, never a streak failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.evaluation import summarize_records
from algorithm.happo.networks import IndependentActors
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, compute_pairwise_geometry

UAVS = RED_IDS[1:]


def rate(n, d):
    return n / d if d else None


def mean(xs):
    return float(np.mean(xs)) if xs else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_formal_replay(actual, formal):
    """Counts/rates remain strict; only floating reward sum permits roundoff."""
    differences={}
    for k in ('red_win_rate','blue_win_rate','draw_rate','mean_episode_return','mean_red_attack_kills',
              'mean_blue_attack_kills','MAV_survival_rate','mean_UAV_survivors','mean_episode_length'):
        tolerance=1e-7 if k=='mean_episode_return' else 1e-12
        difference=float(actual[k])-float(formal[k])
        differences[k]=dict(actual=float(actual[k]),formal=float(formal[k]),difference=difference,
            absolute_tolerance=tolerance,passed=bool(np.isfinite(actual[k]) and abs(difference)<=tolerance))
    return dict(passed=all(r['passed'] for r in differences.values()),metrics=differences)


class CombatObserver:
    """Only read state around the ORIGINAL resolver/deactivator calls."""
    def __init__(self, env):
        self.env = env
        self.resolve = env._resolve_attacks
        self.deactivate = env._deactivate
        self.in_combat = False
        env._resolve_attacks = self.observed_resolve
        env._deactivate = self.observed_deactivate

    def snapshot_locks(self):
        self.evaluated = {a: dict(target=self.env.weapon_lock_target[a],
            streak=self.env._attack_streak.get((a, self.env.weapon_lock_target[a]), 0)) for a in UAVS}

    def observed_deactivate(self, *args, **kwargs):
        if self.in_combat and self.evaluated is None:
            self.snapshot_locks()  # before FIRST synchronous death; all candidates already formed
        return self.deactivate(*args, **kwargs)

    def observed_resolve(self):
        self.evaluated = None
        self.pairs = {}
        self.pre_alive = {a: self.env.entities[a].state.alive for a in UAVS}
        self.pre_blue = [b for b in BLUE_IDS if self.env.entities[b].state.alive]
        for a in UAVS:
            self.pairs[a] = {}
            if self.pre_alive[a]:
                for b in self.pre_blue:
                    g = compute_pairwise_geometry(self.env.entities[a].state, self.env.entities[b].state)
                    self.pairs[a][b] = dict(distance=g.distance, ATA_deg=float(np.rad2deg(g.ata)),
                        AA_deg=float(np.rad2deg(g.aa)), full_gate=self.env._weapon_gate_geometry(a, b) is not None)
        self.in_combat = True
        try:
            result = self.resolve()
            if self.evaluated is None:
                self.snapshot_locks()
            return result
        finally:
            self.in_combat = False


def replay(actors, config, episodes=200, device='cuda'):
    env = HeterogeneousMAVUAVAirCombatEnv(config, profile='main')
    observer = CombatObserver(env)
    results = []
    for episode in range(episodes):
        observations, _ = env.reset(seed=1000 + episode)
        torch.manual_seed(2000 + episode)
        torch.cuda.manual_seed_all(2000 + episode)
        steps = []; kills = set(); times = []; events = []
        while True:
            actions = []
            with torch.no_grad():
                for i, a in enumerate(RED_IDS):
                    action, _ = actors.actors[i].sample(torch.as_tensor(observations[a], device=device).unsqueeze(0), deterministic=False)
                    actions.append(action.squeeze(0).cpu().numpy())
            prior_reward = dict(env._reward_target_previous)
            observations, _, terminated, truncated, info = env.step(np.asarray(actions))
            step = env.step_count
            red_events = [e for e in info['attack_events'] if e['attacker'] in UAVS]
            for e in red_events:
                events.append(dict(step=step, **e))
                if e['target'] not in kills:
                    kills.add(e['target']); times.append(step)
            rows = []
            for a in UAVS:
                ev = observer.evaluated[a]
                target = info['weapon_lock_target'][a]
                reward = info[f'reward_target_{a}']
                rows.append(dict(agent=a, alive=bool(env.entities[a].state.alive),
                    pre_combat_alive=bool(observer.pre_alive[a]), weapon_lock_target=target,
                    weapon_lock_streak=info['weapon_lock_streak'][a], reward_target=reward,
                    reward_target_switch=info[f'reward_target_switch_{a}'], prior_reward_target=prior_reward[a],
                    evaluated_lock_target=ev['target'], evaluated_streak=ev['streak'],
                    full_gate_targets=[b for b, g in observer.pairs[a].items() if g['full_gate']],
                    geometry=observer.pairs[a], kills=[e['target'] for e in red_events if e['attacker'] == a]))
            steps.append(dict(step=step, rows=rows, unique_kills=len(kills),
                alive_blue_pre=observer.pre_blue, alive_blue_post=[b for b in BLUE_IDS if env.entities[b].state.alive],
                terminated=bool(terminated), truncated=bool(truncated)))
            if terminated or truncated:
                assert len(kills) == info['episode_summary']['red_attack_kills']
                results.append(dict(episode=episode, environment_seed=1000+episode, action_seed=2000+episode,
                    summary=info['episode_summary'], events=events, kill_times=times, steps=steps))
                break
        if (episode+1) % 20 == 0:
            print(f'Replay {episode+1}/{episodes}', flush=True)
    return results


def alignment(episodes):
    output = []
    for agent in (*UAVS, 'overall'):
        rows = [r for ep in episodes for s in ep['steps'] for r in s['rows']
                if r['alive'] and (agent == 'overall' or r['agent'] == agent)]
        conflicts = lambda r: r['weapon_lock_target'] is not None and r['reward_target'] is not None and r['weapon_lock_target'] != r['reward_target']
        row = dict(agent=agent, alive_uav_steps=len(rows), conflict_steps=sum(map(conflicts, rows)))
        row['conflict_step_fraction'] = rate(row['conflict_steps'], len(rows))
        conflict_episodes = sum(any(conflicts(r) for s in ep['steps'] for r in s['rows']
                              if r['alive'] and (agent == 'overall' or r['agent'] == agent)) for ep in episodes)
        row.update(conflict_episodes=conflict_episodes, conflict_episode_fraction=rate(conflict_episodes, len(episodes)))
        for name, cutoff in [('lock_exists', 0), ('streak_ge1', 1), ('streak_ge2', 2)]:
            selected = [r for r in rows if r['weapon_lock_target'] is not None and r['weapon_lock_streak'] >= cutoff]
            matches = sum(r['reward_target'] == r['weapon_lock_target'] for r in selected)
            row.update({name+'_n': len(selected), name+'_match_rate': rate(matches, len(selected)),
                        name+'_mismatch_rate': rate(len(selected)-matches, len(selected))})
        switches = []
        for ep in episodes:
            for before, after in zip(ep['steps'], ep['steps'][1:]):
                for old, new in zip(before['rows'], after['rows']):
                    if agent != 'overall' and old['agent'] != agent: continue
                    lock = old['weapon_lock_target']
                    if lock and new['weapon_lock_target'] == lock and new['weapon_lock_streak'] == old['weapon_lock_streak']+1:
                        switches.append(old['reward_target'] is not None and new['reward_target'] is not None
                                        and old['reward_target'] != new['reward_target'] and new['reward_target'] != lock)
        row.update(continuous_lock_pairs=len(switches), reward_switched_away_during_lock=sum(switches))
        output.append(row)
    return output


def conditional_streaks(episodes):
    buckets = {(streak, group): [] for streak in (1, 2) for group in ('MATCH', 'CONFLICT')}
    for ep in episodes:
        for j, s in enumerate(ep['steps']):
            for i, r in enumerate(s['rows']):
                st = r['weapon_lock_streak']; target = r['weapon_lock_target']
                if st not in (1, 2) or not target or r['reward_target'] is None or not r['alive']: continue
                group = 'MATCH' if r['reward_target'] == target else 'CONFLICT'
                later = ep['steps'][j+1:j+4]
                if not later:
                    buckets[st, group].append(dict(censored=True)); continue
                nxt = later[0]['rows'][i]
                keep = nxt['evaluated_lock_target'] == target and nxt['evaluated_streak'] == st+1
                kill1 = target in nxt['kills']
                kill3 = any(target in step['rows'][i]['kills'] for step in later)
                buckets[st, group].append(dict(censored=False, keep=keep,
                    reset=not keep, kill1=kill1, kill3=kill3, full3=len(later)==3))
    rows = []
    for (st, group), observations in buckets.items():
        eligible = [o for o in observations if not o['censored']]
        full3 = [o for o in eligible if o['full3']]
        rows.append(dict(streak=st, group=group, events=len(observations), next_step_observed=len(eligible),
            terminal_censored=len(observations)-len(eligible), p_streak_increases=rate(sum(o['keep'] for o in eligible),len(eligible)),
            p_streak_reset=rate(sum(o['reset'] for o in eligible),len(eligible)),
            p_own_same_target_kill_next1=rate(sum(o['kill1'] for o in eligible),len(eligible)),
            next3_fully_observed=len(full3), p_own_same_target_kill_next3=rate(sum(o['kill3'] for o in full3),len(full3))))
    return rows


def last_kill(episodes, horizon):
    rows = []; traces = []
    for ep in episodes:
        if len(ep['kill_times']) < 3: continue
        third = ep['kill_times'][2]
        endgame = [s for s in ep['steps'] if s['step'] > third and len(s['alive_blue_pre']) == 1]
        all_rows = [r for s in endgame for r in s['rows'] if r['pre_combat_alive']]
        near = any(g['distance'] <= 3000 for r in all_rows for g in r['geometry'].values())
        gate = any(r['full_gate_targets'] for r in all_rows)
        maximum = max((r['evaluated_streak'] for r in all_rows), default=0)
        conflicts = [r for r in all_rows if r['weapon_lock_target'] and r['reward_target']]
        conflict_fraction = rate(sum(r['weapon_lock_target'] != r['reward_target'] for r in conflicts),len(conflicts))
        completed = len(ep['kill_times']) >= 4
        horizon_forming = ep['summary']['outcome'] == 'draw' and any(r['weapon_lock_streak'] in (1,2) for r in ep['steps'][-1]['rows'])
        category = ('completed_fourth' if completed else 'A_no_UAV_within_3km' if not near else
                    'B_within_3km_no_full_gate' if not gate else 'C_max_streak1' if maximum <= 1 else
                    'D_max_streak2' if maximum == 2 else 'G_other')
        rows.append(dict(episode=ep['episode'], environment_seed=ep['environment_seed'],
            outcome=ep['summary']['outcome'], red_attack_kills=ep['summary']['red_attack_kills'],
            third_kill_step=third, remaining_horizon_steps=horizon-third,
            fourth_kill_step=ep['kill_times'][3] if completed else None, category=category,
            endgame_uav_steps=len(all_rows), endgame_lock_steps=len(conflicts), conflict_fraction=conflict_fraction,
            E_long_conflict_overlay=bool(len(conflicts)>=3 and conflict_fraction is not None and conflict_fraction>=0.5),
            F_horizon_forming_attack_overlay=horizon_forming))
        for s in endgame:
            b = s['alive_blue_pre'][0]
            for r in s['rows']:
                traces.append(dict(episode=ep['episode'],step=s['step'],agent=r['agent'],last_blue=b,
                    alive=r['pre_combat_alive'], **r['geometry'].get(b,{}),
                    weapon_lock=r['evaluated_lock_target'], weapon_streak=r['evaluated_streak'],
                    post_cleanup_lock=r['weapon_lock_target'],reward_target=r['reward_target']))
    return rows, traces


def tables(episodes, horizon):
    distributions = []
    for k in range(5):
        eps = [e for e in episodes if e['summary']['red_attack_kills']==k]
        distributions.append(dict(red_attack_kills=k,count=len(eps),percentage=100*len(eps)/len(episodes),
            red=sum(e['summary']['outcome']=='red' for e in eps), blue=sum(e['summary']['outcome']=='blue' for e in eps),
            draw=sum(e['summary']['outcome']=='draw' for e in eps)))
    contributions = []; participation = Counter(); maxima = Counter(); total = Counter()
    for ep in episodes:
        credits = Counter(e['attacker'] for e in ep['events'])
        total.update(credits); participation[sum(credits[a]>0 for a in UAVS)] += 1
        mx = max(credits.values(), default=0)
        maxima['single_UAV_ge2'] += mx>=2; maxima['single_UAV_ge3'] += mx>=3; maxima['single_UAV_eq4'] += mx==4
        for a in UAVS:
            contributions.append(dict(episode=ep['episode'],environment_seed=ep['environment_seed'],agent=a,
                kill_candidate_credits=credits[a],unique_team_kills=ep['summary']['red_attack_kills']))
    totals = {a: dict(kill_credits=total[a],contribution_share=rate(total[a],sum(total.values()))) for a in UAVS}
    last, trace = last_kill(episodes,horizon)
    summary = dict(kill_distribution=distributions,uav_contributions=totals,
        attacker_participation={str(n):dict(count=participation[n],fraction=rate(participation[n],len(episodes))) for n in range(4)},
        single_UAV_thresholds=dict(maxima),total_unique_team_kills=sum(e['summary']['red_attack_kills'] for e in episodes),
        total_candidate_credits=sum(total.values()),dominant_UAV=max(UAVS,key=lambda a:total[a]),
        entered_three_kills=len(last),completed_fourth=sum(r['fourth_kill_step'] is not None for r in last),
        p_fourth_given_entered_three=rate(sum(r['fourth_kill_step'] is not None for r in last),len(last)),
        p_draw_given_exact_three=rate(distributions[3]['draw'],distributions[3]['count']),
        mean_remaining_steps_after_third=mean([r['remaining_horizon_steps'] for r in last]),
        mean_kill_times={str(n):mean([e['kill_times'][n-1] for e in episodes if len(e['kill_times'])>=n]) for n in range(1,5)},
        last_kill_categories=dict(Counter(r['category'] for r in last)),
        last_kill_overlays={k:sum(r[k] for r in last) for k in ('E_long_conflict_overlay','F_horizon_forming_attack_overlay')})
    return summary,dict(kill_distribution=distributions,uav_kill_contributions=contributions,
        reward_lock_alignment=alignment(episodes),streak_match_vs_conflict=conditional_streaks(episodes),
        last_kill_failure=last,last_kill_step_records=trace)


def complete_report(output):
    """Small cache-only reporting pass; NEVER samples actions or steps an env."""
    output=Path(output)
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with gzip.open(output/'replay_cache.json.gz','rt',encoding='utf-8') as f: episodes=json.load(f)
    old_path=ROOT/'outputs/audits/v312_seed_reward_roots_20261008/raw/seed3_final.json.gz'
    if old_path.exists():
        with gzip.open(old_path,'rt',encoding='utf-8') as f: old=json.load(f)
        comparison={}
        for name,eps in [('v312_existing50',old),('v313_first50',episodes[:len(old)])]:
            credits=Counter(); participants=Counter()
            for ep in eps:
                c=Counter(e['attacker'] for e in ep['events'] if e['attacker'] in UAVS)
                credits.update(c); participants[sum(c[a]>0 for a in UAVS)]+=1
            comparison[name]=dict(episodes=len(eps),credits=dict(credits),
                shares={a:rate(credits[a],sum(credits.values())) for a in UAVS},
                episodes_two_or_more_killing_UAVs=participants[2]+participants[3],
                fraction_two_or_more_killing_UAVs=rate(participants[2]+participants[3],len(eps)))
        summary['cached_v312_same_seed_range_comparison']=comparison
    killing=[r for ep in episodes for st in ep['steps'] for r in st['rows'] if r['evaluated_streak']>=3]
    summary['killing_boundary_timing_diagnostic']=dict(n=len(killing),
        matches_post_reward=sum(r['evaluated_lock_target']==r['reward_target'] for r in killing),
        note='Reward is selected after target death. This is a timestamp/mechanical mismatch, not evidence of ongoing lock conflict.')
    count_pairs=[]
    for ep in episodes:
        c=Counter(e['attacker'] for e in ep['events'])
        nonzero=[a for a in UAVS if c[a]>0]
        mx=max((c[a] for a in UAVS),default=0)
        count_pairs.append(dict(episode=ep['episode'],dominant_agents=';'.join(a for a in nonzero if c[a]==mx),
            dominant_contribution_share=rate(mx,sum(c.values())),killing_UAV_count=len(nonzero)))
    write_csv(output/'episode_dominance.csv',count_pairs)
    formal=summary['formal_evaluation']; overall=summary['alignment'][-1]
    last=list(csv.DictReader((output/'last_kill_failure.csv').open(encoding='utf-8')))
    def pct(x): return 'missing' if x is None else f'{100*x:.2f}%'
    text='# v3.13 seed3 单目标武器锁定向审计\n\n'
    text+='## 正式结果与回放合同\n\n'
    text+=f"正式 stochastic 200 局：Red {pct(formal['red_win_rate'])}，Blue {pct(formal['blue_win_rate'])}，Draw {pct(formal['draw_rate'])}；Red kills {formal['mean_red_attack_kills']:.3f}，Blue kills {formal['mean_blue_attack_kills']:.3f}，MAV survival {pct(formal['MAV_survival_rate'])}，UAV survivors {formal['mean_UAV_survivors']:.3f}，length {formal['mean_episode_length']:.2f}，return {formal['mean_episode_return']:.3f}。\n\n"
    text+='CUDA final checkpoint 单次冻结回放：main，env seeds 1000–1199，action seeds 2000–2199。九个核心结果与原正式评估一致。原 run/算法/环境文件 SHA-256 未变，actor 参数未变，无训练。\n\n'
    text+='## 1. 击杀分布\n\n|Red kills|回合数|比例|Red / Blue / Draw|\n|---|---:|---:|---|\n'
    for r in summary['kill_distribution']:
        text+=f"|{r['red_attack_kills']}|{r['count']}|{r['percentage']:.1f}%|{r['red']} / {r['blue']} / {r['draw']}|\n"
    text+=f"\nP(draw | exactly 3 kills)={pct(summary['p_draw_given_exact_three'])}；进入第三杀 {summary['entered_three_kills']} 局，其中完成第四杀 {summary['completed_fourth']} 局，条件完成率 {pct(summary['p_fourth_given_entered_three'])}。第三杀后平均剩余 {summary['mean_remaining_steps_after_third']:.2f} 个 decision steps。\n"
    text+='\n平均第1/2/3/4杀时间（仅达到对应杀数的局，单位 decision step）：'+str(summary['mean_kill_times'])+'。\n'
    text+='\n## 2. UAV 贡献\n\n|UAV|candidate kill credits|share|\n|---|---:|---:|\n'
    for a,r in summary['uav_contributions'].items(): text+=f"|{a}|{r['kill_credits']}|{pct(r['contribution_share'])}|\n"
    text+=f"\n唯一 team kills={summary['total_unique_team_kills']}，candidate credits={summary['total_candidate_credits']}，二者不混用。有效攻击 UAV 定义为至少取得一个 kill candidate credit。0/1/2/3 个 UAV 有贡献的回合分布：{summary['attacker_participation']}。单 UAV ≥2 / ≥3 / =4 credits 回合数：{summary['single_UAV_thresholds']}。全局 dominant UAV={summary['dominant_UAV']}。\n"
    if 'cached_v312_same_seed_range_comparison' in summary:
        text+='\nv3.12 旧50局缓存与 v3.13 相同前50个 seeds 的贡献比较（不新增旧版回放；不同训练 run，不构成武器锁改动的因果实验）：\n\n```json\n'+json.dumps(summary['cached_v312_same_seed_range_comparison'],ensure_ascii=False,indent=2)+'\n```\n'
    text+='\n## 3. Reward / weapon lock 对齐\n\n|UAV|lock samples|lock match|streak≥1 match|streak≥2 samples|streak≥2 match|conflict episodes|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in summary['alignment']:
        text+=f"|{r['agent']}|{r['lock_exists_n']}|{pct(r['lock_exists_match_rate'])}|{pct(r['streak_ge1_match_rate'])}|{r['streak_ge2_n']}|{pct(r['streak_ge2_match_rate'])}|{r['conflict_episodes']}|\n"
    text+=f"\nOverall conflict / all alive UAV steps={pct(overall['conflict_step_fraction'])}，conflict episode fraction={pct(overall['conflict_episode_fraction'])}。连续保持锁的相邻 pair={overall['continuous_lock_pairs']}，reward selector 在期间换向其他 Blue 的 pair={overall['reward_switched_away_during_lock']}。\n"
    text+='\nPrimary 对齐口径：post-combat reward selector 与 surviving post-cleanup weapon lock。击杀 boundary 上 victim 已死亡，因此 reward 转向别的 Blue 不代表正在持续的锁冲突；其 pre-cleanup lock/streak=3 单独保留在 cache。\n'
    text+='\n## 4. MATCH / CONFLICT 后续统计\n\n|streak|组|events|next-step n|继续|reset|next1 own kill|next3 n|next3 own kill|\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in summary['conditional_streaks']:
        text+=f"|{r['streak']}|{r['group']}|{r['events']}|{r['next_step_observed']}|{pct(r['p_streak_increases'])}|{pct(r['p_streak_reset'])}|{pct(r['p_own_same_target_kill_next1'])}|{r['next3_fully_observed']}|{pct(r['p_own_same_target_kill_next3'])}|\n"
    text+='\n终点无下一步的事件 censored，不计为 reset。成功 kill 使用 pre-cleanup evaluated streak 判定，避免 kill 清锁伪造失败。next3 分母只使用完整三个未来边界；未观察到的概率写 missing。条件关联不证明因果。\n'
    text+='\n## 5. 最后一杀\n\n主分类（A/B/C/D/G 互斥，completed 单列）：'+str(summary['last_kill_categories'])+'。\n\n'
    text+='E/F 是可重叠描述，不将 E 直接称为失败原因。E：≥3 个可比较锁 steps 且冲突占比≥50%；F：horizon draw 的最后边界仍有 streak1/2。计数：'+str(summary['last_kill_overlays'])+'。\n'
    text+='\n末架 Blue 的逐 UAV distance / ATA / AA / full gate / lock / streak / reward target 在 last_kill_step_records.csv，逐局类别在 last_kill_failure.csv。\n'
    text+='\n## 限制\n\n单 training seed、有限观测时域，仅描述可观测行为。不能由此证明改 reward selector 或延长 horizon 会改善胜率，不实施任何科研修改。\n'
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (output/'research_report.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    run=args.run_dir.resolve(); checkpoint=run/'checkpoint_final.pt'
    files=[p for p in run.rglob('*') if p.is_file()]+list((ROOT/'env').rglob('*.py'))+list((ROOT/'algorithm').rglob('*.py'))
    before={str(p):sha(p) for p in files}
    payload=torch.load(checkpoint,map_location='cuda',weights_only=False)
    config=payload['environment_config']; validate_checkpoint_contract(payload,config)
    if payload['sampled_steps']!=2000000 or payload['environment_version']!='heterogeneous_mavuav_4v4_v3_13':
        raise RuntimeError('Requires exact v3.13 2M final checkpoint')
    c=payload.get('trainer_config',payload.get('config',{}))
    if int(c['seed'])!=3: raise RuntimeError('Requires training seed3')
    original=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    formal=[r for r in original['final_evaluations'] if r['episodes']==200 and r['action_mode']=='stochastic'
            and r['evaluation_profile']=='main' and r['evaluation_environment_seed_start']==1000 and r['action_seed']==2000]
    if len(formal)!=1: raise RuntimeError('Formal protocol missing or ambiguous')
    output=(args.output or ROOT/'outputs/audits'/('v313_lock_seed3_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not output.is_relative_to((ROOT/'outputs/audits').resolve()): raise ValueError('Output must be under outputs/audits')
    output.mkdir(parents=True,exist_ok=False)
    cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all()
    try:
        actors=IndependentActors(hidden_dim=int(c['hidden_dim']),log_std_init=float(c.get('actor_log_std_init',-.5))).to('cuda')
        actors.load_state_dict(payload['actors']); actors.eval()
        actor_before={k:v.clone() for k,v in actors.state_dict().items()}
        episodes=replay(actors,config)
        # Persist the complete observation evidence BEFORE any aggregate assertion.
        # A protocol/numerical mismatch must never discard an expensive replay.
        with gzip.open(output/'replay_cache.json.gz','wt',encoding='utf-8') as f: json.dump(episodes,f)
        actual=summarize_records([e['summary'] for e in episodes])
        validation=validate_formal_replay(actual,formal[0])
        (output/'replay_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        if not validation['passed']:
            raise AssertionError('Frozen replay differs from formal evaluation; raw cache and replay_validation.json retained')
        assert all(torch.equal(v,actor_before[k]) for k,v in actors.state_dict().items())
        summary, data=tables(episodes,int(config['simulation']['max_decision_steps']))
        summary.update(formal_evaluation=formal[0],replay_evaluation=actual,formal_replay_match=True,
            replay_validation=validation,
            checkpoint=str(checkpoint),checkpoint_sha256=before[str(checkpoint)],
            protocol=dict(episodes=200,profile='main',action_mode='stochastic',environment_seeds=[1000,1199],action_seeds=[2000,2199],device='cuda'),
            alignment=data['reward_lock_alignment'],conditional_streaks=data['streak_match_vs_conflict'],
            definitions=dict(alignment='Post-combat reward selector vs surviving post-cleanup lock; kill-step evaluated lock separately in raw cache.',
                contributions='One credit per synchronous attacker/target candidate; team kills are unique victims.',
                next3='Own same-target kill; denominator requires 3 observed future boundaries. Terminal censoring reported.',
                endgame='Transitions AFTER third unique Red kill with exactly one pre-combat Blue alive.',
                categories='A/B/C/D/G exclusive geometry/streak classes. E >=3 comparable lock steps and >=50% conflict; F timeout with streak1/2 are overlays, not causal labels.'))
        comparison=ROOT/'outputs/audits/v312_seed_reward_roots_20261008/checkpoint_uav_contributions.csv'
        if comparison.exists():
            with comparison.open(encoding='utf-8') as f:
                summary['v312_existing_contribution_comparison']=[r for r in csv.DictReader(f) if r['seed']=='3' and r['sampled_steps']=='2000000']
        for name,rows in data.items(): write_csv(output/(name+'.csv'),rows)
        episode_rows=[]
        for ep in episodes:
            row=dict(episode=ep['episode'],environment_seed=ep['environment_seed'],action_seed=ep['action_seed'],**ep['summary'])
            row.update({f'time_to_kill_{n}':ep['kill_times'][n-1] if len(ep['kill_times'])>=n else None for n in range(1,5)})
            episode_rows.append(row)
        write_csv(output/'episode_records.csv',episode_rows)
        assert before=={str(p):sha(p) for p in files}, 'Original files changed'
        summary['source_sha256_unchanged']=True; summary['actor_parameters_unchanged']=True
        (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        report='# v3.13 seed3 单目标武器锁定向审计\n\n'
        report+='正式200局冻结回放指标与原始正式评估一致；无训练、干预或原文件修改。\n\n'
        report+='## 击杀分布\n\n|kills|count|percentage|draw|\n|---|---|---|---|\n'
        for r in data['kill_distribution']: report+=f"|{r['red_attack_kills']}|{r['count']}|{r['percentage']:.1f}%|{r['draw']}|\n"
        report+='\n## 击杀贡献与最后一杀\n\n```json\n'+json.dumps({k:summary[k] for k in ('uav_contributions','attacker_participation','single_UAV_thresholds','p_fourth_given_entered_three','mean_remaining_steps_after_third','mean_kill_times','last_kill_categories','last_kill_overlays')},indent=2,ensure_ascii=False)+'\n```\n'
        report+='\n## Reward / lock 对齐与下一步条件统计\n\n```json\n'+json.dumps(dict(alignment=summary['alignment'],conditional_streaks=summary['conditional_streaks']),indent=2,ensure_ascii=False)+'\n```\n'
        report+='\n## 口径与限制\n\n'+ '\n'.join('- '+v for v in summary['definitions'].values())+'\n\n仅单 training seed；条件统计不证明因果。v3.12 比较仅复用旧缓存，样本数见 summary，不能当 paired 200 局实验。\n'
        (output/'research_report.md').write_text(report,encoding='utf-8')
        complete_report(output)
        print('Audit complete:',output,flush=True)
        print(json.dumps({k:summary[k] for k in ('kill_distribution','uav_contributions','last_kill_categories','mean_remaining_steps_after_third')},indent=2))
    finally:
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)


if __name__=='__main__': main()
