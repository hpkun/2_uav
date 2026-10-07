"""Read-only CAP verification. CUDA only; no trainer or training calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algorithm.happo.evaluation import summarize_records
from tools.audit_v311_environment_rationality import replay, kill_sets, dominance
from tools.audit_v311_happo_failure import Env, RED_IDS, BLUE_IDS, IndependentActors, load_environment_config, validate_checkpoint_contract, sha, table, dump


def rule_smoke(config):
    env=Env(config,randomize=False); env.reset(seed=1)
    def diag():
        p=env.blue_policy
        blue={b:env.entities[b] for b in BLUE_IDS}; red={r:env.entities[r] for r in RED_IDS}
        p.prepare_step(blue,red,env.step_count)
        return p.team_diagnostics(blue,red,env.step_count)
    nominal=diag(); assert nominal['max_target_load']==1
    rows=[]
    for n in (4,3,2,1):
        env.reset(seed=1)
        for i,r in enumerate(RED_IDS): env.entities[r].state.alive=i<n
        env.step_count=1
        d=diag(); loads=sorted(d[f'target_load_{r}'] for r in RED_IDS if env.entities[r].state.alive)
        assert loads=={4:[1,1,1,1],3:[1,1,2],2:[2,2],1:[4]}[n]
        rows.append(dict(alive_red=n,**d))
    for seed in range(30):
        env=Env(config,profile='main'); env.reset(seed=seed)
        assert diag()['max_target_load']==1
    summaries=[]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(312)
        for seed in range(5):
            env=Env(config,profile='main'); env.reset(seed=1000+seed)
            while True:
                actions=(2*torch.rand((4,3),device='cuda')-1).cpu().numpy()
                obs,reward,term,trunc,info=env.step(actions)
                assert all(np.isfinite(o).all() for o in obs.values()) and all(np.isfinite(list(reward.values())))
                if term or trunc: summaries.append(info['episode_summary']); break
    # Rule-level Blue attack test, independent of randomly discovering a kill.
    env=Env(config,randomize=False); env.reset(seed=1)
    for i,(aid,e) in enumerate(env.entities.items()):
        e.state.x=-50000+7000*i; e.state.y=30000; e.state.h=6000; e.state.psi=0; e.state.theta=0
    env.entities['Blue1'].state.x=0; env.entities['Blue1'].state.y=0
    env.entities['UAV1'].state.x=2000; env.entities['UAV1'].state.y=0
    for i in range(3): events,deaths=env._resolve_attacks()
    assert dict(attacker='Blue1',target='UAV1') in events and deaths['UAV1']=='blue_attack'
    return dict(status='PASS',nominal=nominal,load_cases=rows,main_resets=30,random_episodes=summaries,
                Blue_rule_attack_kill=True,device='cuda',configuration_unchanged=env.config==config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--env-seed',type=int,default=1000)
    parser.add_argument('--action-seed',type=int,default=2000)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rule-smoke',action='store_true')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    if args.output.exists(): raise FileExistsError('refusing to overwrite existing audit')
    torch.set_num_threads(1)
    config=load_environment_config(ROOT/'configs/env_v312.yaml')
    if args.rule_smoke:
        result=rule_smoke(config)
        args.output.mkdir(parents=True); dump(args.output/'summary.json',result)
        print(json.dumps(result,indent=2),flush=True); return
    if not args.run_dir or args.episodes<=0: parser.error('--run-dir and positive --episodes required')
    protected=[p for p in args.run_dir.rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in protected}
    payload=torch.load(args.run_dir/'checkpoint_final.pt',map_location='cpu',weights_only=False)
    validate_checkpoint_contract(payload,config)
    if payload.get('environment_config')!=config or (payload['actor_variant'],payload['critic_variant'],payload['method_variant'])!=('vanilla','mlp','baseline'):
        raise RuntimeError('CAP audit requires exact resolved v3.12 vanilla baseline contract')
    tc=payload['trainer_config']
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        actors=IndependentActors(hidden_dim=tc['hidden_dim'],log_std_init=tc.get('actor_log_std_init',-.5)).cuda().eval()
    actors.load_state_dict(payload['actors']); parameters={k:v.clone() for k,v in actors.state_dict().items()}
    episodes=[]; cap_steps=[]; contributions=[]
    for i in range(args.episodes):
        ep=replay(actors,config,args.env_seed+i,args.action_seed+i)
        episodes.append(dict(episode=i,environment_seed=args.env_seed+i,action_seed=args.action_seed+i,**ep['result']))
        sets=kill_sets(ep['events']); counts=[len(sets[a]) for a in RED_IDS[1:]]
        contributions.append(dict(episode=i,**{f'{a}_attack_contribution':len(sets[a]) for a in RED_IDS[1:]},**dominance(counts)))
        for r in ep['steps']:
            # alive Red pre-action is the contract used by assignment.
            # For n>=4 iff all Red survived previous transition; initial n=4.
            previous=ep['steps'][r['step']-2] if r['step']>1 else None
            n=sum(previous[f'{a}_alive'] for a in RED_IDS) if previous else 4
            m=r['alive_blue_pre']; load=r['max_same']
            assert load<=int(np.ceil(m/n)), 'unbalanced CAP load'
            cap_steps.append(dict(episode=i,step=r['step'],alive_Red=n,alive_Blue=m,max_target_load=load,
                                  unnecessary_duplicate=int(n==4 and m==4 and load>1)))
        print(f'{i+1}/{args.episodes}: {ep["result"]["outcome"]} kills={ep["result"]["red_attack_kills"]}',flush=True)
    s=summarize_records(episodes)
    s.update(win_rate=s['red_win_rate'],mean_return=s['mean_episode_return'],mean_red_kills=s['mean_red_attack_kills'],MAV_survival=s['MAV_survival_rate'])
    for n in (1,2,3,4): s[f'fraction_max_target_load_{n}']=float(np.mean([r['max_target_load']==n for r in cap_steps]))
    four=[r for r in cap_steps if r['alive_Red']==4 and r['alive_Blue']==4]
    s['fraction_unnecessary_duplicate_assignment']=float(np.mean([r['unnecessary_duplicate'] for r in four])) if four else None
    for a in RED_IDS[1:]: s[f'{a}_attack_contribution']=sum(r[f'{a}_attack_contribution'] for r in contributions)
    s['dominant_UAV_share']=float(np.mean([r['dominance_share'] for r in contributions if r['dominance_share'] is not None])) if any(r['dominance_share'] is not None for r in contributions) else None
    s['single_UAV_ge3_kill_contribution_episode_rate']=float(np.mean([r['max_agent_kills']>=3 for r in contributions]))
    s['single_UAV_4kill_contribution_episode_rate']=float(np.mean([r['max_agent_kills']==4 for r in contributions]))
    assert all(torch.equal(parameters[k],v) for k,v in actors.state_dict().items())
    assert all(sha(Path(p))==h for p,h in hashes.items())
    args.output.mkdir(parents=True)
    table(args.output/'episodes.csv',episodes); table(args.output/'cap_steps.csv',cap_steps); table(args.output/'UAV_contributions.csv',contributions)
    dump(args.output/'summary.json',dict(metrics=s,sampled_steps=payload['sampled_steps'],protocol=dict(profile='main',deterministic=False,environment_seeds=[args.env_seed,args.env_seed+args.episodes-1],action_seeds=[args.action_seed,args.action_seed+args.episodes-1]),input_sha256=hashes,actor_parameters_unchanged=True,input_SHA_unchanged=True))
    print(json.dumps(s,indent=2),flush=True)


if __name__=='__main__': main()
