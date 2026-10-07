"""Read-only v3.11 audit. No trainer construction, updates, or config writes.

Replay observes the real resolver once, at its post-physics/pre-kill boundary.
All policy actions follow evaluate_actors' exact stochastic sampling order.
Outputs are diagnostic small-N evidence, never replacement formal evaluations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from algorithm.happo.networks import IndependentActors
from algorithm.happo.evaluation import summarize_records
from algorithm.evaluate_happo import validate_checkpoint_contract
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv as Env, load_environment_config
from env.geometry import compute_pairwise_geometry as geometry
from env.reward_role_v37 import uav_distance_reward
from env.reward_role_v39 import attack_gate_indicator, uav_angle_quality, uav_coupled_dense_reward, uav_gate_reward


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def table(path, rows):
    if not rows:
        raise ValueError(f"refusing empty audit table: {path}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(g, cfg):
    c = cfg["combat"]
    return bool(attack_gate_indicator(g.distance, g.ata, g.aa, *c["distance"],
                                     np.deg2rad(c["ata_deg"]), np.deg2rad(c["aa_deg"])))


def attack_gap(g, cfg):
    """Diagnostic normalized envelope violation; not a policy target action."""
    lo, hi = cfg["combat"]["distance"]
    return max(lo-g.distance, 0)/lo + max(g.distance-hi, 0)/hi + max(g.ata-np.pi/6, 0)/(np.pi/6) + max(g.aa-np.pi/2, 0)/(np.pi/2)


def install_observer(env):
    original = env._resolve_attacks
    snapshots = []
    def observed():
        pairs = {}
        for aid in RED_IDS[1:]:
            if env.entities[aid].state.alive:
                for bid in BLUE_IDS:
                    if env.entities[bid].state.alive:
                        g = geometry(env.entities[aid].state, env.entities[bid].state)
                        pairs[(aid, bid)] = (g, inside(g, env.config), env._attack_streak.get((aid, bid), 0))
        targets = [env.blue_policy._guidance_state[b].target_id for b in BLUE_IDS if env.entities[b].state.alive]
        events, deaths = original()
        event_pairs = {(e["attacker"], e["target"]) for e in events}
        # Kill-transition counters are cleared by the real resolver: use its actual
        # event to identify the third hold, not a reconstructed pseudo streak.
        after = {p: (3 if p in event_pairs else env._attack_streak.get(p, 0)) for p in pairs}
        snapshots.append(dict(pairs=pairs, after=after, events=events, targets=targets,
                              blue_mav_streak=(3 if any(e["target"]=="MAV" and e["attacker"] in BLUE_IDS for e in events)
                                               else max((env._attack_streak.get((b, "MAV"), 0) for b in BLUE_IDS), default=0))))
        return events, deaths
    env._resolve_attacks = observed
    return snapshots


def replay(actors, config, episodes, profile, env_seed, action_seed, device="cuda", random=False):
    records, agent_rows, alignments, examples = [], [], [], []
    env = Env(config, profile=profile)
    snaps = install_observer(env)
    for episode in range(episodes):
        obs, _ = env.reset(seed=env_seed+episode)
        torch.manual_seed(action_seed+episode)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(action_seed+episode)
        random_rng = np.random.default_rng(action_seed+episode)
        per_agent = {a: dict(geoms=[], p2_geoms=[], actions=[], first_direct=None, first_3km=None,
                            first_gate=None, first_kill=None, switches=0, streak1=0, streak2=0,
                            streak3=0, death="alive", death_step=None) for a in RED_IDS[1:]}
        counts = Counter(); aligned = {a: Counter() for a in RED_IDS[1:]}
        previous = {a: None for a in RED_IDS[1:]}
        first_kill = None; kill_steps = []; discounted = 0.; steps = 0
        mav_death_cause = "alive"; mav_death_step = None
        while True:
            steps += 1
            p2_pre_action=sum(env.entities[b].state.alive for b in BLUE_IDS)==1
            pre_active = {a: env.entities[a].state.alive for a in RED_IDS}
            for a in RED_IDS[1:]:
                if pre_active[a] and any(env.direct_visible(a, b) for b in BLUE_IDS) and per_agent[a]["first_direct"] is None:
                    per_agent[a]["first_direct"] = steps-1
            if random:
                actions = random_rng.uniform(-1, 1, (4, 3))
            else:
                with torch.no_grad():
                    actions = np.stack([actors.actors[i].sample(torch.as_tensor(obs[a], device=device).unsqueeze(0), deterministic=False)[0].squeeze(0).cpu().numpy() for i, a in enumerate(RED_IDS)])
            obs, rewards, terminated, truncated, info = env.step(actions)
            discounted += .99**(steps-1)*np.mean(list(rewards.values()))
            snap = snaps[-1]
            if "MAV" in info["death_causes"]:
                mav_death_cause, mav_death_step = info["death_causes"]["MAV"], steps
            for target in snap["targets"]:
                counts[f"blue_target_{target}"] += 1
            counts["blue_target_exposures"] += len(snap["targets"])
            counts["multi_blue_MAV_steps"] += snap["targets"].count("MAV") >= 2
            counts["blue_mav_streak_steps"] += snap["blue_mav_streak"] > 0
            counts["P2_transitions"] += p2_pre_action
            new_kills = sum(b in BLUE_IDS and cause == "red_attack" for b, cause in info["death_causes"].items())
            if new_kills:
                first_kill = first_kill or steps
                kill_steps.extend([steps]*new_kills)
            for i, aid in enumerate(RED_IDS[1:], 1):
                stats = per_agent[aid]
                if pre_active[aid]: stats["actions"].append(actions[i])
                pairs = {b: v for (a, b), v in snap["pairs"].items() if a == aid}
                if pairs:
                    # Closest living opponent defines per-step nearest geometry.
                    closest = min(pairs, key=lambda b: pairs[b][0].distance)
                    g = pairs[closest][0]
                    stats["geoms"].append((g.distance, g.ata, g.aa, any(v[1] for v in pairs.values())))
                    if p2_pre_action:
                        stats["p2_geoms"].append((g.distance,g.ata,g.aa,any(v[1] for v in pairs.values()),
                                                   max(snap["after"][(aid,b)] for b in pairs)))
                    if g.distance <= 3000 and stats["first_3km"] is None: stats["first_3km"] = steps
                    if any(v[1] for v in pairs.values()) and stats["first_gate"] is None: stats["first_gate"] = steps
                    for b in pairs:
                        s = snap["after"][(aid, b)]
                        for n in (1, 2, 3): stats[f"streak{n}"] += s >= n
                    counts["gate_pair_steps"] += sum(v[1] for v in pairs.values())
                    counts["pair_exposures"] += len(pairs)
                if any(e["attacker"] == aid and e["target"] in BLUE_IDS for e in snap["events"]):
                    stats["first_kill"] = stats["first_kill"] or steps
                if aid in info["death_causes"]:
                    stats["death"], stats["death_step"] = info["death_causes"][aid], steps
                target = info.get(f"reward_target_{aid}")
                if env.entities[aid].state.alive:
                    alive = [b for b in BLUE_IDS if env.entities[b].state.alive]
                    visible = [b for b in alive if env.team_visible(b)]
                    if target and alive:
                        gs = {b: geometry(env.entities[aid].state, env.entities[b].state) for b in alive}
                        nearest = min(alive, key=lambda b: gs[b].distance)
                        best = min(alive, key=lambda b: attack_gap(gs[b], config))
                        gates = [b for b in alive if inside(gs[b], config)]
                        streaks = [b for b in alive if env._attack_streak.get((aid, b), 0) > 0]
                        ac = aligned[aid]; ac["samples"] += 1
                        ac["selector_nearest"] += target == nearest
                        ac["selector_attackable"] += target == best
                        ac["active_streak_samples"] += bool(streaks)
                        ac["selector_active_streak"] += target in streaks
                        ac["gate_available_samples"] += bool(gates)
                        ac["selector_misses_gate"] += bool(gates) and target not in gates
                        ac["visible_samples"] += bool(visible)
                        ac["direct_samples"] += env.direct_visible(aid, target)
                        switched = previous[aid] is not None and previous[aid] != target
                        stats["switches"] += switched
                        old = previous[aid]
                        broken = switched and (aid, old) in snap["pairs"] and snap["pairs"][(aid, old)][2] > 0 and snap["after"][(aid, old)] == 0
                        ac["switch_streak_break_cooccurrence"] += broken
                        if gates and target not in gates and len(examples) < 12:
                            examples.append(dict(episode=episode, step=steps, agent=aid, selected=target,
                                                 gate_targets="|".join(gates), selected_distance=gs[target].distance,
                                                 selected_ata_deg=np.rad2deg(gs[target].ata), selected_aa_deg=np.rad2deg(gs[target].aa),
                                                 gate_distances="|".join(str(round(gs[b].distance, 2)) for b in gates),
                                                 state_timing="post-resolver; surviving targets only"))
                        previous[aid] = target
            if terminated or truncated:
                record = dict(info["episode_summary"])
                record.update(episode=episode, environment_seed=env_seed+episode, action_seed=action_seed+episode,
                              first_kill_step=first_kill, kill_steps="|".join(map(str, kill_steps)),
                              three_kill_draw=int(record["outcome"] == "draw" and record["red_attack_kills"] == 3),
                              MAV_death_cause=mav_death_cause, MAV_death_step=mav_death_step,
                              discounted_return=discounted, **dict(counts))
                record["gate_fraction"] = counts["gate_pair_steps"]/max(counts["pair_exposures"], 1)
                record["team_process_sum"] = sum(record[f"{a.lower()}_process_reward_sum"] for a in RED_IDS)/4
                assert np.isclose(record["episode_return"], record["team_process_sum"]+record["event_reward_sum"]+record["terminal_reward_sum"]+record["safety_reward_sum"], atol=1e-9)
                records.append(record)
                for aid, stats in per_agent.items():
                    ga = np.asarray(stats.pop("geoms"), float).reshape(-1, 4)
                    p2 = np.asarray(stats.pop("p2_geoms"),float).reshape(-1,5)
                    aa = np.asarray(stats.pop("actions"), float).reshape(-1, 3)
                    row = dict(episode=episode, agent=aid, **stats, geometry_samples=len(ga), action_samples=len(aa))
                    if len(ga):
                        d, ata, angle = ga[:, 0], ga[:, 1], ga[:, 2]
                        row.update(minimum_distance=float(d.min()), mean_ATA_deg=float(np.rad2deg(ata).mean()), mean_AA_deg=float(np.rad2deg(angle).mean()))
                        for label, mask in dict(within12km=d<=12000, within5km=d<=5000, within3km=d<=3000,
                                               below1km=d<1000, distance_gate=(d>=1000)&(d<=3000),
                                               ata_gate=ata<np.pi/6, aa_gate=angle<np.pi/2,
                                               angle_gate=(ata<np.pi/6)&(angle<np.pi/2), full_gate=ga[:, 3]>0).items():
                            row[f"fraction_{label}"] = float(mask.mean())
                    if len(aa):
                        for dim in range(3):
                            row[f"action{dim}_mean"] = float(aa[:, dim].mean())
                            row[f"action{dim}_std"] = float(aa[:, dim].std())
                            row[f"action{dim}_saturation"] = float((np.abs(aa[:, dim])>=.95).mean())
                    row["P2_geometry_samples"]=len(p2)
                    if len(p2):
                        row.update(P2_min_distance=float(p2[:,0].min()),P2_mean_distance=float(p2[:,0].mean()),
                                   P2_min_ATA_deg=float(np.rad2deg(p2[:,1]).min()),P2_mean_ATA_deg=float(np.rad2deg(p2[:,1]).mean()),
                                   P2_min_AA_deg=float(np.rad2deg(p2[:,2]).min()),P2_mean_AA_deg=float(np.rad2deg(p2[:,2]).mean()),
                                   P2_distance_gate_fraction=float(((p2[:,0]>=1000)&(p2[:,0]<=3000)).mean()),
                                   P2_angle_gate_fraction=float(((p2[:,1]<np.pi/6)&(p2[:,2]<np.pi/2)).mean()),
                                   P2_full_gate_fraction=float(p2[:,3].mean()),P2_max_real_streak=int(p2[:,4].max()))
                    agent_rows.append(row)
                    ac = aligned[aid]
                    alignments.append(dict(episode=episode, agent=aid, **dict(ac), target_switches=row["switches"]))
                break
    return records, agent_rows, alignments, examples


def phase_rows(rows, label):
    previous = 0; weighted = []
    for r in rows:
        now = int(r["completed_episodes"])
        if now < previous: raise ValueError("completed episodes regressed")
        weighted.append((r, now-previous)); previous = now
    results = []
    edges = [0, 200000, 400000, 800000, 1200000, 1600000, 2000000]
    metrics = [k for k in rows[0] if k.startswith("mean_") or k.endswith("_rate") or k == "MAV_survival_rate"]
    for lo, hi in zip(edges, edges[1:]):
        selected = [(r, w) for r, w in weighted if lo<int(r["sampled_steps"])<=hi]
        n = sum(w for _, w in selected)
        result = dict(run=label, phase=f"{lo}-{hi}", episodes=n, log_windows=len(selected), weighting="completed episode increments; endpoint bucket")
        for k in metrics: result[k] = sum(float(r[k])*w for r, w in selected)/n if n else None
        for k in ("entropy", "critic_loss", "actor_0_loss", "actor_1_loss", "actor_2_loss", "actor_3_loss", "curriculum_alpha"):
            result[k] = float(np.mean([float(r[k]) for r, _ in selected])) if selected else None
        results.append(result)
    milestones = {}
    for metric, thresholds in {"mean_red_attack_kills": [.1, .5, 1, 2, 3], "red_win_rate": [.01, .1, .2, .5]}.items():
        for t in thresholds:
            ids = [i for i, r in enumerate(rows) if float(r[metric])>=t]
            sustained = [i for i in ids if i+1<len(rows) and float(rows[i+1][metric])>=t]
            milestones[f"{metric}>={t}"] = dict(first=int(rows[ids[0]]["sampled_steps"]) if ids else None,
                                               two_records=int(rows[sustained[0]]["sampled_steps"]) if sustained else None)
    return results, milestones


def range_time(position, velocity, radius):
    p, v = np.asarray(position), np.asarray(velocity)
    if p@p <= radius**2: return 0.
    a, b, c = v@v, 2*(p@v), p@p-radius**2
    disc = b*b-4*a*c
    if a<=0 or disc<0: return None
    roots = [x for x in ((-b-np.sqrt(disc))/(2*a), (-b+np.sqrt(disc))/(2*a)) if x>=0]
    return float(min(roots)) if roots else None


def static_geometry(config, n=3000):
    env = Env(config, profile="main")
    nominal = []
    env.reset(seed=0, options={"randomization_override": {k: 0 for k in config["randomization_profiles"]["main"]}})
    for a in RED_IDS:
        for b in BLUE_IDS:
            red, blue = env.entities[a].state, env.entities[b].state
            g = geometry(red, blue); p = blue.position()-red.position() if hasattr(red, "position") else np.array([blue.x-red.x, blue.y-red.y, blue.h-red.h])
            v = blue.velocity_vector()-red.velocity_vector()
            nominal.append(dict(agent=a, blue=b, distance_m=g.distance, ata_deg=np.rad2deg(g.ata), aa_deg=np.rad2deg(g.aa),
                                relative_velocity_m_s=v.tolist(), straight_line_5km_s=range_time(p,v,5000),
                                straight_line_3km_s=range_time(p,v,3000), aa_change_to_threshold_deg=max(np.rad2deg(g.aa)-90, 0)))
    env.set_randomization_override(None)
    samples = []
    for seed in range(n):
        env.reset(seed=seed)
        for a in RED_IDS:
            gs = [geometry(env.entities[a].state, env.entities[b].state) for b in BLUE_IDS]
            samples.append(dict(agent=a, min_distance=min(g.distance for g in gs), mean_distance=np.mean([g.distance for g in gs]),
                                direct=sum(env.direct_visible(a,b) for b in BLUE_IDS),
                                datalink_only=sum(env.team_visible(b) and not env.direct_visible(a,b) for b in BLUE_IDS),
                                ata_mean_deg=np.mean([np.rad2deg(g.ata) for g in gs]), aa_mean_deg=np.mean([np.rad2deg(g.aa) for g in gs]),
                                best_aa_change_deg=min(max(np.rad2deg(g.aa)-90,0) for g in gs),
                                best_range_change_m=min(max(g.distance-3000,0) for g in gs)))
    summary = {}
    for a in RED_IDS:
        summary[a] = {}
        for k in samples[0]:
            if k != "agent":
                values = np.array([r[k] for r in samples if r["agent"]==a])
                summary[a][k] = dict(mean=float(values.mean()), p01=float(np.quantile(values,.01)), p99=float(np.quantile(values,.99)), min=float(values.min()), max=float(values.max()))
    return dict(reset_count=n, nominal=nominal, main_reset_summary=summary,
                range_time_caveat="Straight constant-velocity crossings, NOT attack gate entry times; AA must also change.")


def landscape():
    rows = []
    for d in sorted(set(np.arange(500,12001,500).tolist()+[4900,5000,5100,999,1000,1001,2999,3000,3001])):
        for ata in sorted(set(range(0,181,10))|{29,30,31}):
            for aa in sorted(set(range(0,181,10))|{89,90,91}):
                a,b=np.deg2rad([ata,aa]); q=uav_angle_quality(a,b); dq=uav_distance_reward(d,1000,3000)
                dense=uav_coupled_dense_reward(q,dq); gate=attack_gate_indicator(d,a,b,1000,3000,np.pi/6,np.pi/2)
                rows.append(dict(distance_m=d, ATA_deg=ata, AA_deg=aa, uav_angle_quality=q,uav_distance_quality=dq,
                                 uav_dense_reward=dense, gate_indicator=gate,gate_reward=uav_gate_reward(gate),uav_process_reward=dense+uav_gate_reward(gate)))
    return rows


def scripted(config, targets=1, loss=None):
    env=Env(config, profile="learnability"); env.reset(seed=77)
    for i, a in enumerate(RED_IDS):
        s=env.entities[a].state; s.x=-20000.; s.y=i*3000.; s.h=6000.; s.v=275.; s.theta=0.; s.psi=0.
    env.entities["UAV1"].state.x=0.; env.entities["UAV1"].state.y=0.
    for i,b in enumerate(BLUE_IDS):
        s=env.entities[b].state; s.x=2000. if i<targets else 25000.; s.y=(i-.5*(targets-1))*250. if i<targets else 10000.+i*2000.
        s.h=6000.; s.v=275.; s.theta=0.; s.psi=0.
    if loss:
        env.entities[loss].state.h=900.
    snaps=install_observer(env); records=[]
    for step in range(3):
        _,_,terminated,truncated,info=env.step(np.zeros((4,3)))
        records.append(dict(step=step+1, real_streak=snaps[-1]["after"].get(("UAV1","Blue1")),
                            attack_events=info["attack_events"], deaths=info["death_causes"],
                            event=info["event_reward"],terminal=info["terminal_reward"],safety=info["safety_reward"],
                            process={a:info[f"{a.lower()}_process_reward"] for a in RED_IDS}, team_reward=info["team_reward"]))
        if terminated or truncated: break
    return records


def preferences(config):
    r=config["reward"]; g=.99
    # Explicit hypothetical event timing; process/safety fixed to zero for a
    # matched-component comparison, NOT asserted to be feasible safe trajectories.
    kill=r["blue_kill"]
    scenarios={"A":([],75),"B":([(20,kill)],75),"C":([(20,kill),(35,kill),(50,kill)],75),
               "D":([(20,kill),(35,kill),(50,kill),(65,kill+r["terminal_red_win"])],65),
               "E":([(20,kill),(35,kill),(50,kill),(60,r["uav_loss"]),(65,kill+r["terminal_red_win"])],65),
               "F":([(20,kill),(35,kill),(50,r["mav_loss"]+r["terminal_blue_win"])],50)}
    rows=[dict(scenario=k, horizon=n, process_baseline=0, event_timing=json.dumps(events),
               undiscounted=sum(v for _,v in events), discounted=sum(g**(t-1)*v for t,v in events)) for k,(events,n) in scenarios.items()]
    discount_sum=(1-g**75)/(1-g)
    return dict(gamma=g, rows=rows, pointwise_draw_process_bounds=[-.5*75,.78*75],
                discounted_pointwise_draw_bounds=[-.5*discount_sum,.78*discount_sum],
                never_gate_draw_upper_bound=.405*75,
                disclaimer="Pointwise bounds need not be jointly trajectory-achievable. Scenario ordering uses matched zero process/safety and specified event timing.")


def diff(a,b,prefix=""):
    result=[]
    for k in sorted(set(a)|set(b)):
        path=f"{prefix}.{k}" if prefix else k
        if isinstance(a.get(k),dict) and isinstance(b.get(k),dict): result+=diff(a[k],b[k],path)
        elif a.get(k)!=b.get(k): result.append(dict(field=path,v310=a.get(k),v311=b.get(k)))
    return result


def visibility_boundary(config):
    env=Env(config); env.reset(seed=0); rows=[]
    for distance in (4900.,5000.,5100.):
        red=env.entities["UAV1"].state; blue=env.entities["Blue1"].state; mav=env.entities["MAV"].state
        red.x=0.; red.y=0.; red.h=6000.; red.psi=0.; red.theta=0.
        blue.x=distance; blue.y=0.; blue.h=6000.; blue.psi=0.; blue.theta=0.
        mav.x=-1000.; mav.y=0.; mav.h=6700.
        for b in BLUE_IDS[1:]: env.entities[b].state.alive=False
        process,diag=env._role_process_rewards()
        obs=env._observations()["UAV1"]; block=obs[44:58]
        rows.append(dict(distance=distance,direct=env.direct_visible("UAV1","Blue1"),
                         datalink=env.team_visible("Blue1") and not env.direct_visible("UAV1","Blue1"),
                         target=diag["reward_target_UAV1"],process=process["UAV1"],enemy_geometry=block[:9].tolist()))
    return rows


def finite_tree(value):
    if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
    if isinstance(value,dict): return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(finite_tree(v) for v in value)
    return True


def same_tree(a, b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return isinstance(b,type(a)) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b


def artifact_checks(run):
    """Independent artifact-only checks; no environment advance or policy call."""
    resolved=yaml.safe_load((run/"resolved_config.yaml").read_text(encoding="utf-8"))
    a=torch.load(run/"checkpoint_2000000.pt",map_location="cpu",weights_only=False)
    b=torch.load(run/"checkpoint_final.pt",map_location="cpu",weights_only=False)
    evaluations=read_csv(run/"evaluations.csv")
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    formal=summary["final_evaluations"][0]
    for key in ("red_win_rate","draw_rate","blue_win_rate","mean_red_attack_kills","mean_episode_return"):
        assert float(evaluations[-1][key])==formal[key]
    # YAML sequences are lists; the canonical validator resolves range tuples.
    assert load_environment_config(resolved["environment"])==a["environment_config"]
    assert resolved["total_steps"]==a["sampled_steps"]==b["sampled_steps"]==2000000
    stale=[]
    for path in run.glob("checkpoint_*.pt"):
        p=torch.load(path,map_location="cpu",weights_only=False)
        for ei,state in enumerate(p["rollout_state"]["environment_states"]):
            for (attacker,target),n in state["attack_streak"].items():
                if n and (not state["entities"][attacker].state.alive or not state["entities"][target].state.alive):
                    stale.append(dict(checkpoint=path.name,env_index=ei,attacker=attacker,target=target,streak=n))
    return dict(resume_history=resolved.get("resume_history","missing"),requested_device=resolved.get("requested_device","missing"),
                resolved_device=resolved.get("resolved_device","missing"),device_fallback=resolved.get("device_fallback_reason","missing"),
                formal_summary_matches_evaluations_csv=True,resolved_environment_matches_checkpoint=True,
                final_equivalence={k:same_tree(a[k],b[k]) for k in ("actors","critic","actor_optimizer_states","critic_optimizer_state","rollout_state","trainer_numpy_rng","torch_rng","cuda_rng","sampled_steps")},
                vector_base_seed=a["rollout_state"]["vector_base_seed"],stale_dead_pair_streaks_in_saved_checkpoints=stale)


def markdown_table(rows, fields):
    lines=["|"+"|".join(fields)+"|","|"+"|".join("---" for _ in fields)+"|"]
    for row in rows:
        values=[]
        for key in fields:
            v=row.get(key)
            if v in (None,""): text="missing"
            elif isinstance(v,(float,np.floating)): text=f"{v:.4f}"
            else: text=str(v)
            values.append(text.replace("|",","))
        lines.append("|"+"|".join(values)+"|")
    return "\n".join(lines)


def postprocess(output):
    """Artifact-only secondary analysis, safe to run after a completed audit."""
    output=Path(output); summary=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    feasibility=json.loads((output/"environment_feasibility.json").read_text(encoding="utf-8"))
    episodes=read_csv(output/"replay_episodes.csv"); agents=read_csv(output/"replay_agents.csv"); align=read_csv(output/"target_selector_alignment.csv")
    runs=[Path(p) for p in summary["protocol"]["run_dir"]]
    checks={r.name:artifact_checks(r) for r in runs}
    outcome_rows=[]; geometry_rows=[]; alignment_rows=[]; blue_rows=[]; p2_rows=[]
    for run in runs:
        er=[r for r in episodes if r["run"]==run.name and int(r["sampled_steps"])==2000000]
        ar=[r for r in agents if r["run"]==run.name and int(r["sampled_steps"])==2000000]
        al=[r for r in align if r["run"]==run.name and int(r["sampled_steps"])==2000000]
        for outcome in ("red","draw","blue"):
            group=[r for r in er if r["outcome"]==outcome]
            result=dict(run=run.name,outcome=outcome,episodes=len(group))
            for key in ("episode_return","team_process_sum","event_reward_sum","terminal_reward_sum","safety_reward_sum","mav_process_reward_sum",*[f"{a.lower()}_process_reward_sum" for a in RED_IDS[1:]]):
                result[key]=float(np.mean([float(r[key]) for r in group])) if group else None
            outcome_rows.append(result)
        for aid in RED_IDS[1:]:
            group=[r for r in ar if r["agent"]==aid]; n=sum(int(r["geometry_samples"]) for r in group)
            result=dict(run=run.name,agent=aid,geometry_samples=n,kill_event_credit=sum(int(r["streak3"]) for r in group),
                        boundary_losses=sum(r["death"]=="boundary" for r in group),blue_attack_losses=sum(r["death"]=="blue_attack" for r in group))
            for key in ("fraction_within5km","fraction_distance_gate","fraction_ata_gate","fraction_aa_gate","fraction_angle_gate","fraction_full_gate"):
                result[key]=sum(float(r[key])*int(r["geometry_samples"]) for r in group if r.get(key))/n if n else None
            geometry_rows.append(result)
        totals={k:sum(int(r.get(k) or 0) for r in al) for k in ("samples","selector_nearest","selector_attackable","active_streak_samples","selector_active_streak","gate_available_samples","selector_misses_gate","target_switches","switch_streak_break_cooccurrence")}
        alignment_rows.append(dict(run=run.name,**totals,
                                   selector_nearest_fraction=totals["selector_nearest"]/max(totals["samples"],1),
                                   selector_attackable_fraction=totals["selector_attackable"]/max(totals["samples"],1),
                                   selector_active_streak_fraction=totals["selector_active_streak"]/max(totals["active_streak_samples"],1),
                                   selector_misses_available_gate_fraction=totals["selector_misses_gate"]/max(totals["gate_available_samples"],1)))
        exposures=sum(int(r.get("blue_target_exposures") or 0) for r in er)
        blue_rows.append(dict(run=run.name,episodes=len(er),blue_target_exposures=exposures,
                             **{f"target_{aid}_fraction":sum(int(r.get(f"blue_target_{aid}") or 0) for r in er)/max(exposures,1) for aid in RED_IDS},
                             multi_blue_MAV_steps=sum(int(r["multi_blue_MAV_steps"]) for r in er),
                             MAV_streak_positive_steps=sum(int(r["blue_mav_streak_steps"]) for r in er)))
        for r in er:
            ks=[int(s) for s in r["kill_steps"].split("|") if s]
            entry=ks[2] if len(ks)>=3 else None
            p2_rows.append(dict(run=run.name,episode=r["episode"],outcome=r["outcome"],kills=r["red_attack_kills"],
                                P2_posttransition_entry_step=entry,P2_following_transition_count=int(r["episode_length"])-entry if entry is not None else 0,
                                final_elimination=r["outcome"]=="red",three_kill_draw=int(r["three_kill_draw"])))
    for name,rows in {"replay_reward_by_outcome.csv":outcome_rows,"final_agent_geometry_summary.csv":geometry_rows,
                      "final_target_alignment_summary.csv":alignment_rows,"final_Blue_target_summary.csv":blue_rows,"final_endgame_timing.csv":p2_rows}.items(): table(output/name,rows)
    summary["artifact_checks"]=checks
    summary["input_sha256_current"]={str(p):sha(p) for r in runs for p in r.rglob("*") if p.is_file()}
    # The interpretation below is the reviewed fixed-protocol 5-episode audit.
    # A larger user-run must NEVER inherit its numeric claims automatically.
    protocol=summary["protocol"]
    reviewed=(protocol["episodes"]==5 and protocol["env_seed"]==1000 and protocol["action_seed"]==2000
              and protocol["profile"]=="main" and protocol["static_resets"]==3000
              and protocol["counterfactual_episodes"]==5 and protocol["random_episodes"]==10
              and protocol["checkpoint"]==["checkpoint_501760.pt","checkpoint_1001472.pt","checkpoint_1501184.pt","checkpoint_final.pt"]
              and [r.name for r in runs]==[f"happo_v311_seed{s}_2m" for s in (1,2,3)])
    if not reviewed:
        summary["classification"]="INCONCLUSIVE"
        summary["interpretation_status"]="New protocol requires review; no copied small-N conclusions."
        dump(output/"summary.json",summary)
        report="\n\n".join(["# v3.11 checkpoint机制审计：新协议数值结果（待研究者解读）",
                              "MEASURED: "+json.dumps(protocol,ensure_ascii=False),
                              "原始run SHA与actor参数不变；未训练。分类INCONCLUSIVE，不自动继承固定5局的假设矩阵。",
                              markdown_table(read_csv(output/"checkpoint_mechanism_summary.csv"),["run","sampled_steps","red_win_rate","mean_red_attack_kills","gate_fraction"]),
                              markdown_table(geometry_rows,["run","agent","geometry_samples","fraction_distance_gate","fraction_full_gate","kill_event_credit"]),
                              markdown_table(alignment_rows,["run","samples","selector_misses_available_gate_fraction"]),
                              "见同目录其余CSV与environment_feasibility.json；后续报告应基于此次实际数值解释。"])
        (output/"research_report.md").write_text(report+"\n",encoding="utf-8")
        return
    summary["static_findings"]={"boundary_dead_attacker_streak_cleanup":{
        "status":"VERIFIED", "reproduction":"Set UAV1->Blue1 real streak=2, UAV1 altitude=900; _apply_boundaries then _resolve_attacks leaves streak=2, emits no attack event.",
        "scope":"Environment state-cleanup defect, NOT a demonstrated HAPPO update defect or cause of seed divergence.",
        "saved_checkpoint_occurrences":sum(len(c["stale_dead_pair_streaks_in_saved_checkpoints"]) for c in checks.values())}}
    summary["classification"]="MIXED"
    summary["hypotheses"]=[
        {"id":"H1","hypothesis":"Attack envelope too sparse","status":"PARTIALLY_SUPPORTED","severity":"high","evidence":"Random 10 episodes: zero gate / 5584 alive UAV-Blue pair steps; early weak checkpoints zero gate despite approaching. Population rarity and necessity not established."},
        {"id":"H2","hypothesis":"75-step horizon too short","status":"INCONCLUSIVE","severity":"medium","evidence":"Actual final replay wins at 34/38; seed1 three-kill draws retain 41/55 steps after third kill. Fundamental horizon insufficiency refuted; more-time rescue untested."},
        {"id":"H3","hypothesis":"5km sensing is primary bottleneck","status":"INCONCLUSIVE","severity":"medium","evidence":"Initial datalink ~3.71 Blue/UAV; 8km counterfactual helps seed1, worsens seeds2/3; no uniform rescue."},
        {"id":"H4","hypothesis":"v3.11 geometry increases difficulty","status":"INCONCLUSIVE","severity":"medium","evidence":"Restoring v3.10 geometry does not consistently rescue policies; these are only five paired episodes, not retraining."},
        {"id":"H5","hypothesis":"Dense reward favors safe draw","status":"PARTIALLY_SUPPORTED","severity":"medium","evidence":"Risk/delayed-gradient plateau possible; positive dense-return hacking NOT observed. Every final replay draw has negative team process sum; wins driven by shared kill+terminal."},
        {"id":"H6","hypothesis":"MAV loss penalty drives excessive avoidance","status":"PARTIALLY_SUPPORTED","severity":"medium","evidence":"MAV death costs -200 shared event+terminal; random Blue wins occurred with zero Blue attack kills, so boundary exploration risk is real. Avoidance causation unproven."},
        {"id":"H7","hypothesis":"Dense reward poorly aligned with kill progress","status":"PARTIALLY_SUPPORTED","severity":"medium","evidence":"Flat range quality within1-3km; high non-gate dense values exist; process-kill correlations mixed in five-episode groups; task-return event ranking correct."},
        {"id":"H8","hypothesis":"Reward selector mismatched with attack target","status":"VERIFIED","severity":"medium","evidence":"17/162 surviving post-step gate-available samples select a non-gate target across milestone replay. 1235 switches, zero observed switch/streak-break cooccurrences; causal role unresolved."},
        {"id":"H9","hypothesis":"Curriculum transition creates failure basin","status":"INCONCLUSIVE","severity":"medium","evidence":"Continuous linear alpha, no400k step switch; seed2 discovery~1.26M/seed3~.71M. Final weak policies not strongly rescued by learnability profile."},
        {"id":"H10","hypothesis":"Blue pathological behavior","status":"INCONCLUSIVE","severity":"low","evidence":"Actual cached nearest target, not distance-inferred. MAV lock exposures differ, no verified pathological exploit in inspected samples."},
        {"id":"H11","hypothesis":"HAPPO implementation defect in audited path","status":"REFUTED","severity":"low","evidence":"Audited vanilla mean reward, GAE boundaries, masks, squashed actions/logprob, sequential ratios, RNG seeds and artifact contracts consistent. Separate environment stale counter issue disclosed."},
        {"id":"H12","hypothesis":"Ordinary optimization seed sensitivity is sufficient explanation","status":"PARTIALLY_SUPPORTED","severity":"high","evidence":"Contracts identical and discovery times diverge markedly; seed sensitivity measured, but its sufficiency independent of task/reward is not proven."}]
    dump(output/"summary.json",summary)
    final=[]
    for name,s in summary["formal_existing_summaries"].items():
        r=s["final_evaluations"][0]
        final.append(dict(seed=s["seed"],**{k:r[k] for k in ("red_win_rate","blue_win_rate","draw_rate","mean_red_attack_kills","mean_episode_return","MAV_survival_rate","mean_UAV_survivors","mean_episode_length")}))
    phases=read_csv(output/"seed_phase_summary.csv")
    for r in phases:
        for key in ("red_win_rate","draw_rate","mean_red_attack_kills","mean_episode_return","MAV_survival_rate"):
            r[key]=float(r[key])
    report=["# v3.11 Vanilla HAPPO 三种子科研审计", "",
        "## 0. 范围与证据等级", "",
        "SOURCE-CODE FACT=源码事实；MEASURED=现有产物/实测；DIAGNOSTIC COUNTERFACTUAL=内存反事实；INFERENCE=推断；UNRESOLVED=尚无法判断。",
        "未训练、未改正式源码或配置、未重复正式200局胜率评估。12个milestone×5局，final内存反事实各5局、随机10局；原始A与已有final5局重复验证，不能算独立样本。CUDA策略前向；WSL拒绝访问，使用PowerShell uav。",
        "现有正式final200是stochastic、main、环境1000–1199、action seed2000，role=execution_smoke；不是新增完整机制200局。",
        "", "## 1. MEASURED：正式已有结果与合同", "", markdown_table(final,["seed","red_win_rate","blue_win_rate","draw_rate","mean_red_attack_kills","mean_episode_return","MAV_survival_rate","mean_UAV_survivors","mean_episode_length"]),
        "三个环境与checkpoint完全相同合同：v3.11 / main / vanilla / mlp / baseline / CUDA /16env /128rollout /2M；奖励coupled_gate_v1，entropy=.001，log_std_init=-.25。trainer_config除seed完全相同。resume_history=[]；requested/resolved均cuda。所有actor/critic/optimizer tensor finite；final与2M的actor、critic、optimizer、rollout/RNG均逐元素等价（详见summary）。",
        "", "## 2. SOURCE-CODE FACT + MEASURED：初始几何/可见性", "",
        "nominal所有UAV/Blue275m/s，Red+X、Blue−X，相对速度(-550,0,0)m/s。UAV到Blue8.022–8.544km，MAV10.042–10.185km。按直线常速度：UAV5km约5.52–7.27s，3km约9.20–14.55s；MAV5km9.25–9.80s，3km12.99–14.01s。MAV实际12km传感器nominal在t=0已全部direct可见。**距离交叉时间不是完整gate时间**。AA按Blue前向与Red→Blue视线定义，初始AA≈159–176°；直接对头接近不满足AA<90°。",
        "3000 main resets：UAV初始最近Blue均值7.93–7.96km，pair均值8.21–8.30km；MAV pair均值10.19km。每UAV直接可见Blue均值仅.001–.00233，datalink-only约3.71。各UAV平均AA166–168°，最有利目标AA仍平均需改变66.6–68.9°；最近距离仍需缩短约4.93–4.96km。分位数及全部16个nominal pair见environment_feasibility.json。",
        "UAV奖励候选基于team_visible，非own direct；datalink保留全部9个几何字段。4.9/5.0/5.1km direct=true/true/false，后一项转datalink；同一目标reward .03082/.01342/−.00341，连续差值来自距离指数，未发现额外5km断崖。",
        "", "## 3. SOURCE-CODE FACT + MEASURED：combat", "",
        "range[1000,3000]包含端点，ATA<30°/AA<90°严格不等；连续3步，失败归零；UAV-only12个Red攻击pair。每个pair独立，允许同一UAV同一步攻击多个Blue，所有达标pair同步结算。多个攻击者同杀一Blue只有一次death/+100，但可以多个attack events。",
        "同一alive pair/同一时间点的reward gate公式与combat一致。**reward只选择一个存活/可见目标、combat检查全部pair**，且reward发生在kill之后：因此整体reward gate为0但其他pair真实streak增加、或杀死后原gate消失，不能误报成几何公式bug。",
        "脚本合法追尾状态：真实streak1→2→第三步真实kill事件；单杀event100；四杀event400+terminal100，team_reward500；一个UAV越界另扣10，总事件+terminal490。完整逐步process分解见feasibility JSON。该脚本不是nominal任务oracle，也不是训练性能。",
        "**发现环境状态清理瑕疵**：boundary-dead attacker的既有streak可残留，dead attacker不会产生攻击事件但残留会出现在state/obs；源码复现streak2保持2。本次所有保存checkpoint的16env快照未发现该残留。其训练因果影响UNRESOLVED，未修环境。",
        "", "## 4. SOURCE-CODE FACT：完整reward与数值尺度", "",
        "对MAV和每UAV：r_i=P_i+E+T+S；HAPPO buffer使用mean_i(r_i)=mean_i(P_i)+E+T+S，不是4倍shared事件。E=100×Blue独立击杀数−10×UAV失效数−100×MAV失效；T=+100win/−100Bluewin/0draw；S=−1当任意alive Red距离<100m，否则0。",
        "MAV P=.3 threat+.2 aspect_mean+.4 aware_mean；threat为Blue→MAV真实streak>0时−1；aspect=−(1−BlueATA/45°)若ATA<45；aware=.3(1−MAVATA/90°)若可见且ATA<90；两者以alive Blue数量为分母。alive时P范围[−.5,.12]，不是单独奖励远离Blue。",
        "UAV P=Q_A×Q_D−.5+.5gate，Q_A=1−(ATA+AA)/(2π)；Q_D在1–3km为1，低于1km为exp((d−1000)/1000)，高于3km为exp((3000−d)/3000)。目标无可见但有存活Blue则−.5；自身死亡或无Blue则0。旧R_V仅诊断，不进入此process。",
        "75步0kill/0loss draw的pointwise宽界[−37.5,58.5]；从不进入gate的上界30.375。γ=.99折扣宽界[−26.4707,41.2942]。这些非共同可实现轨迹保证。一次UAV-loss的事件成本10远小于一次kill100；MAV-loss+Blue-terminal成本200。安全draw梯度局部盆地理论可能，但不是已验证的正dense reward hacking。",
        "按匹配process=0/safety=0和明确event时刻的诊断比较：", "", markdown_table(feasibility["reward_preferences"]["rows"],["scenario","horizon","event_timing","undiscounted","discounted"]),
        "D>E>C>B>A成立。F两杀再MAV死的undiscounted=0，discounted31.4487（早获奖后受罚）；不可比较不同未知过程的真实路径优劣。",
        "", "## 5. MEASURED：奖励landscape与target alignment", "",
        "reward_landscape.csv扫描.5–12km、ATA/AA0–180°，另加gate/sensing边界点。固定角度下8→5→3km的reward递增；3→2→1km distance quality完全平坦，只有角度/gate改变；<1km指数下降并失去gatebonus，但999m/Q=1仍有约.499dense。Q=1时8km−.3111、5km+.0134、3/2/1km gate时1.0、.5km+.1065（非gate）。对头Q=.5时8km−.4056、5km−.2433、3km0。超过5.079km时即便Q=1 dense也为负，因此不能概括5–12km shaping均正。",
        "非gate角度状态可有较高dense（例如ATA=30°,AA=0°,d=2km得.4167，严格ATA失败）；角度有连续梯度、距离1–3km无连续梯度，gate边缘跳跃.5。无法不指定角度/距离单位就比较二者梯度强弱。",
        "selector score=.35Q+.25[d<=3km]+.20Δh/10000+.20||Δv||/800；distance项没有1km下界，速度项偏好较大相对速度。nearest≠selector并非必然错误；most_attackable定义为归一化包线超限最小，仅诊断proxy。",
        "整个60局milestone：162个存活post-step有gate样本中17次selector漏选（10.49%）；1235次selector switch，0次观察到切换与原streak中断共现。真实例：seed1@1.001M，env1001第28步UAV1：Blue1可gate、距离1737m，selector却Blue3距离2748m、ATA99.1°。combat仍可处理Blue1，故不能把selector当显式policy目标/kill switch。",
        "", markdown_table(alignment_rows,["run","samples","selector_nearest_fraction","selector_attackable_fraction","selector_active_streak_fraction","selector_misses_available_gate_fraction","target_switches","switch_streak_break_cooccurrence"]),
        "", "## 6. MEASURED：学习阶段（按完成回合增量加权）", "",
        "episode指标按每行completed_episodes的增量加权；loss/entropy按update窗口平均。跨分段回合归到完成行endpoint；没有伪称按逐transition精确切分。", "",
        markdown_table(phases,["run","phase","episodes","red_win_rate","draw_rate","mean_red_attack_kills","MAV_survival_rate","mean_episode_return"]),
        "持续两个记录的kills≥.1：seed1 .239616M、seed2 1.255424M、seed3 .710656M；≥.5：.251904/1.372160/.864256M；≥1：.284672/1.460224/1.173504M；≥2：.415744/1.763328/missing；≥3：.903168/missing/missing。两记录仍短、需结合阶段平均，不代表收敛。",
        "curriculum alpha=clip(sampled_steps/400000,0,1)，jitter线性插值；每rollout开始传入vector env，作用于随后reset，不搬移当前飞机。因此400k无跳变且有reset时延。成功seed1在curriculum结束前已发现攻击，seed2长期main下才发现，seed3结束后发现；不证明curriculum导致失败。",
        "0–200k MAV loss：s1约7.9%、s2约13.1%、s3约9.4%，不支持‘成功seed更高早期MAV风险’；200–400k s1约6.2%但同时已学攻击，时间关联非因果。",
        "", "## 7. MEASURED：几何链条和endgame", "",
        "详表checkpoint_mechanism_summary.csv/replay_agents.csv覆盖4milestone×3seed。nearest距离/ATA/AA指标以当步最近alive Blue为准；full_gate表示任一alive Blue完整gate，故两种指标不能逐行相乘，分母仅alive UAV可比较步。", "",
        markdown_table(geometry_rows,["run","agent","geometry_samples","fraction_distance_gate","fraction_angle_gate","fraction_full_gate","kill_event_credit","boundary_losses","blue_attack_losses"]),
        "seed2@.502M/1.001M：能接近距离区间却所有UAV完整gate/真实streak/kill都0，是角度几何discover失败，不是单纯看不见。final分配UAV1/2/3真实kill event=3/2/6，开始多机攻击但覆盖和completion不足。",
        "seed3@final：UAV2贡献5个kill，UAV1/3为0；UAV2可连续3步并击杀，其他机的角度/gate支持不足。主要是局部攻击者形成但多目标清场能力不足，而非全部streak无法保持。",
        "seed1@final：15个kill全部来自UAV1，另两机非攻击支持；正式41%胜率不等于已可靠多机协同。两局3kill draw分别第三杀在step34和20，剩41/55步仍未完成末杀；有真实win在38步。seed2另一3kill draw第三杀step34，同样剩41步。seed3此final样本未进P2。75s不是普遍物理不可行，追加时间能否救末杀UNRESOLVED。",
        "随机10局：0gate/5584pair steps、0Red/Blue attack kill、70%Bluewin，指向边界探索损失而非Blue攻击必杀。小样本零观测不能声称总体概率严格0，pair samples强相关不能当5584独立试验。",
        "", "## 8. MEASURED：真实reward/Blue机制", "", markdown_table(outcome_rows,["run","outcome","episodes","episode_return","team_process_sum","event_reward_sum","terminal_reward_sum","mav_process_reward_sum"]),
        "final所有draw team过程累积均负，胜局约495.63/499.34主要来自event400+terminal100。不支持高process safe-draw赢过真正win的hacking。五局process–kill Pearson s1+.739/s2+.235/s3−.735，不能用于因果或稳健统计；说明不是所有seed过程奖励都随kill单调增加。",
        "探索尺度：checkpoint .502M→final的global geometric pre-tanh std，s1 .675→.414、s2 .730→.502、s3 .759→.458；三个seed总体收缩而非log_std inflation。训练phase entropy也下降，不能把失败直接解释为熵持续膨胀。动作饱和逐UAV/维度见replay_agents.csv；高饱和本身不证明数值错误。",
        "实际Blue缓存目标而非最近距离推测：", "", markdown_table(blue_rows,["run","blue_target_exposures","target_MAV_fraction","target_UAV1_fraction","target_UAV2_fraction","target_UAV3_fraction","multi_blue_MAV_steps","MAV_streak_positive_steps"]),
        "s2有一局全UAV失效，Blue48步多机指向MAV但无MAV gate/streak、最终MAV活着draw；targeting≠attack threat。已有正式200局MAV生存全为98.5%，所以不能用MAV死亡差异解释41/7/2%win。MAV逐回合death cause若旧版本次replay记录缺失，不凭距离猜；后续工具已增加该字段，未为补字段重复回放。",
        "", "## 9. DIAGNOSTIC COUNTERFACTUAL：版本差异与profile", "",
        "v310→v311只有version、UAV sensing8→5km、MAV(-5000,0,5000)→(-6000,0,6700)、其余飞机h5000→6000。dynamics/Blue/combat/reward/75horizon/profile随机化一致，字段实际在reset/sensing消费。内存B/C/D仅改这些，不写配置。", "",
        markdown_table(read_csv(output/"counterfactual_summary.csv"),["run","condition","diagnostic_profile","red_win_rate","mean_red_attack_kills","MAV_survival_rate","mean_UAV_survivors"]),
        "8km并非普遍救援：s1 kills3→3.4，s2 2.2→1.8，s3 1→.2。geometry恢复也无统一提升。weak final在learnability kills2.4/1.6且均0win，不能归类为只在main泛化失败。每条件5局、不含重新训练，是冻结policy的OOD输入/轨迹干预，对原训练因果或重新训练后的难度不可外推。",
        "", "## 10. SOURCE-CODE FACT：HAPPO排除性检查", "",
        "buffer四agent mean；terminated或truncated都阻断bootstrap与GAE传播（75步正式任务终止语义）；active masks用pre-action，死agent不进入actor PPO；action=tanh Gaussian sample，evaluate_actions用atanh与相同Jacobian；random sequential agent order；preceding factor乘已更新actor new/old ratio、dead为1，detach后进入当前actor advantage；team critic仍team return。trim mapping/RK4正常。vector seed=training_seed+env_index+1000003×reset_count；checkpoint base seeds1/2/3不同，未见config drift。",
        "**NO IMPLEMENTATION DEFECT FOUND IN AUDITED HAPPO PATH**。这是有限静态/产物检查，不是形式化证明整个项目无bug；上文环境stale-counter瑕疵另列。",
        "", "## 11. 诊断矩阵", "", markdown_table(summary["hypotheses"],["id","hypothesis","status","severity","evidence"]),
        "", "## 12. INFERENCE：结论与后续优先级", "",
        "总分类 **MIXED**：明确的attack discovery/后续几何completion seed sensitivity；严格包线与reward选目标/局部shape可能放大它，但没有足够证据把单一环境、reward或curriculum认定为根因。",
        "当前最直接瓶颈：s1单攻击者+末杀几何；s2长时间angle discovery迟滞，final仍清场弱；s3单UAV局部攻击、覆盖不足。排除：配置漂移、NaN、错shared平均、5km完全看不见、400k硬切换、75步普遍不可四杀、正dense奖励安全draw高过win。",
        "不能据5局批准改环境/算法。先用完整机制样本确认；若必须选择一个待验证修改方向，优先reward selector对齐真实全pair attack progress（INFERENCE），而非放松attack/加horizon/加新网络；其因果收益尚未证实。boundary stale counter作为独立bug需另行授权修复，不应混作科研调参。",
        "", "## 13. 复现与限制", "",
        "新增工具tools/audit_v311_happo_failure.py，测试tests/test_v311_happo_failure_audit.py。新增16项专项通过，v310/v31120项通过，总36；CUDA tiny observer与官方stochastic evaluator逐字段完全一致，actor参数不变；checkpoint/原run全文件SHA不变。仅新audit CSV/JSON/本报告写入。",
        "完整200局机制审计（用户后续自行执行，输出新目录，**不是训练**）：", "", "```bash",
        "python -u tools/audit_v311_happo_failure.py \\",
        "  --run-dir outputs/happo_v311_seed1_2m outputs/happo_v311_seed2_2m outputs/happo_v311_seed3_2m \\",
        "  --episodes 200 --env-seed 1000 --action-seed 2000 --profile main \\",
        "  --static-resets 3000 --random-episodes 10 --counterfactual-episodes 5 \\",
        "  --output outputs/audits/v311_happo_failure_audit_200ep", "```",
        "UNRESOLVED：full200机制总体分布、真正末杀失败逐transition几何、改变包线/horizon/reward的因果收益、curriculum分布导致失败盆地的反事实训练。此次不运行这些正式实验。"]
    (output/"research_report.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")


def checkpoint_info(path,payload):
    config=payload["trainer_config"]
    return dict(file=str(path), sha256=sha(path), sampled_steps=payload["sampled_steps"],
                environment_version=payload["environment_version"],environment_profile=payload["environment_profile"],
                actor_variant=payload["actor_variant"],critic_variant=payload["critic_variant"],method_variant=payload["method_variant"],
                tensors_finite=finite_tree({k:payload[k] for k in ("actors","critic","actor_optimizer_states","critic_optimizer_state")}),
                seed=config["seed"],torch_rng_sha256=hashlib.sha256(payload["torch_rng"].cpu().numpy().tobytes()).hexdigest(),
                log_std={a:payload["actors"][f"actors.{i}.log_std"].cpu().tolist() for i,a in enumerate(RED_IDS)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",nargs="+",type=Path,required=True)
    parser.add_argument("--checkpoint",nargs="+",default=["checkpoint_501760.pt","checkpoint_1001472.pt","checkpoint_1501184.pt","checkpoint_final.pt"])
    parser.add_argument("--episodes",type=int,default=5); parser.add_argument("--env-seed",type=int,default=1000)
    parser.add_argument("--action-seed",type=int,default=2000); parser.add_argument("--profile",choices=("main","learnability"),default="main")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/audits/v311_happo_failure_audit")
    parser.add_argument("--static-resets",type=int,default=3000); parser.add_argument("--random-episodes",type=int,default=10)
    parser.add_argument("--counterfactual-episodes",type=int,default=5)
    args=parser.parse_args()
    if min(args.episodes,args.static_resets,args.counterfactual_episodes,args.random_episodes)<1: parser.error("sample counts must be positive")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU policy fallback")
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(f"choose a new audit output directory: {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    watched={p:sha(p) for run in args.run_dir for p in run.rglob("*") if p.is_file()}
    cfg=load_environment_config(ROOT/"configs/env_v311.yaml"); old=load_environment_config(ROOT/"configs/env_v310.yaml")
    feasibility=static_geometry(cfg,args.static_resets)
    feasibility.update(config_diff=diff(old,cfg),reward_preferences=preferences(cfg),visibility_boundary=visibility_boundary(cfg),
                       scripted_one_kill=scripted(cfg),scripted_four_kills=scripted(cfg,4),scripted_four_kills_uav_loss=scripted(cfg,4,"UAV2"))
    table(out/"reward_landscape.csv",landscape()); dump(out/"environment_feasibility.json",feasibility)
    all_episodes=[]; all_agents=[]; alignments=[]; examples=[]; checkpoints=[]; phases=[]; mechanisms=[]; counterfactual=[]
    summaries={}; milestones={}; contract_reference=None
    for run in args.run_dir:
        label=run.name; summary=json.loads((run/"summary.json").read_text(encoding="utf-8")); summaries[label]=summary
        rows=read_csv(run/"training.csv"); phase,mile=phase_rows(rows,label); phases+=phase; milestones[label]=mile
        steps=[int(r["sampled_steps"]) for r in rows]
        assert all(a<b for a,b in zip(steps,steps[1:])) and steps[-1]==2000000
        for filename in dict.fromkeys(args.checkpoint+["checkpoint_2000000.pt","checkpoint_final.pt"]):
            path=run/filename
            if not path.exists(): raise FileNotFoundError(path)
            payload=torch.load(path,map_location="cpu",weights_only=False)
            validate_checkpoint_contract(payload,cfg)
            assert payload["environment_config"]==cfg
            tc=payload["trainer_config"]; normalized={k:v for k,v in tc.items() if k!="seed"}
            if contract_reference is None: contract_reference=normalized
            assert normalized==contract_reference, "training contract drift"
            checkpoints.append(checkpoint_info(path,payload))
            if filename not in args.checkpoint: continue
            actors=IndependentActors(hidden_dim=tc["hidden_dim"],log_std_init=tc["actor_log_std_init"]).to("cuda")
            actors.load_state_dict(payload["actors"]); actors.eval()
            before={k:v.clone() for k,v in actors.state_dict().items()}
            print(f"Replay {label}/{filename}: {args.episodes} stochastic {args.profile} episodes",flush=True)
            records,agents,align,ex=replay(actors,cfg,args.episodes,args.profile,args.env_seed,args.action_seed)
            tag=dict(run=label,training_seed=tc["seed"],checkpoint=filename,sampled_steps=payload["sampled_steps"],profile=args.profile)
            all_episodes += [dict(**tag,**r) for r in records]; all_agents += [dict(**tag,**r) for r in agents]
            alignments += [dict(**tag,**r) for r in align]; examples += [dict(**tag,**r) for r in ex]
            mechanisms.append(dict(**tag,episodes=len(records),**summarize_records(records),
                                   gate_fraction=sum(r.get("gate_pair_steps",0) for r in records)/max(sum(r.get("pair_exposures",0) for r in records),1),
                                   three_kill_draw_rate=np.mean([r["three_kill_draw"] for r in records]),
                                   mean_first_kill_step=float(np.mean([r["first_kill_step"] for r in records if r["first_kill_step"] is not None])) if any(r["first_kill_step"] is not None for r in records) else None))
            if payload["sampled_steps"]==2000000:
                for condition in ("A_original","B_sensor8","C_geometry310","D_both","learnability"):
                    cc=deepcopy(cfg)
                    if condition in ("B_sensor8","D_both"): cc["sensing"]["UAV_range"]=8000.
                    if condition in ("C_geometry310","D_both"): cc["scenario"]["initial"]=deepcopy(old["scenario"]["initial"])
                    profile="learnability" if condition=="learnability" else args.profile
                    print(f"Diagnostic counterfactual {label}: {condition}",flush=True)
                    rec,_,_,_=replay(actors,cc,args.counterfactual_episodes,profile,args.env_seed,args.action_seed)
                    counterfactual.append(dict(**tag,condition=condition,diagnostic_profile=profile,evidence="DIAGNOSTIC COUNTERFACTUAL",**summarize_records(rec)))
            assert all(torch.equal(before[k],v) for k,v in actors.state_dict().items()), "actor changed during audit"
    print("Random-action geometry diagnostic",flush=True)
    random_records,random_agents,_,_=replay(None,cfg,args.random_episodes,args.profile,args.env_seed,args.action_seed,random=True)
    feasibility["random_policy"]={**summarize_records(random_records),"episodes":args.random_episodes,
                                  "attack_gate_pair_fraction":sum(r.get("gate_pair_steps",0) for r in random_records)/max(sum(r.get("pair_exposures",0) for r in random_records),1)}
    dump(out/"environment_feasibility.json",feasibility)
    for filename,records in {"checkpoint_mechanism_summary.csv":mechanisms,"seed_phase_summary.csv":phases,
                             "target_selector_alignment.csv":alignments,"replay_episodes.csv":all_episodes,
                             "replay_agents.csv":all_agents,"counterfactual_summary.csv":counterfactual,
                             "random_episodes.csv":random_records,"random_agents.csv":random_agents}.items(): table(out/filename,records)
    if examples: table(out/"target_mismatch_examples.csv",examples)
    for path,before in watched.items(): assert sha(path)==before, f"input changed: {path}"
    dump(out/"summary.json",dict(device="cuda",cuda_name=torch.cuda.get_device_name(0),protocol=vars(args)|{"run_dir":[str(p) for p in args.run_dir],"output":str(out)},
                                checkpoints=checkpoints,formal_existing_summaries=summaries,learning_milestones=milestones,
                                input_SHA256_unchanged=True,actor_parameters_unchanged=True,
                                limitations="Small-N replay, no training, no repeated formal200 evaluation, counterfactuals not formal performance results."))
    postprocess(out)
    print(f"Audit complete: {out}",flush=True)


if __name__=="__main__": main()
