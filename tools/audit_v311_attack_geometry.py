"""Read-only second-stage combat audit; default: ONE matched scenario per policy.

No trainer is constructed. Full 200-episode pair-step statistics require an explicit
--full-replay-episodes 200 invocation because the first audit did not save raw pairs.
Offline envelope sensitivity never changes motion, deaths, or rewards. It measures
opportunities before actual deaths censor the recorded trajectories, NOT win rates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithm.happo.evaluation import evaluate_actors
from tools.audit_v311_happo_failure import (
    Env, RED_IDS, BLUE_IDS, geometry, inside, attack_gap, read_csv, table, dump,
    sha, IndependentActors, validate_checkpoint_contract, load_environment_config,
)

DIAGNOSTIC = "DIAGNOSTIC ONLY"
UAVS = RED_IDS[1:]


def phase(alive_blue):
    return "P2" if alive_blue == 1 else "P0" if alive_blue == 4 else "P1"


def conditions(distance, ata_rad, aa_rad, combat):
    """Independent literal copy of REAL resolver comparisons, not reward helper."""
    lo, hi = combat["distance"]
    return (lo <= distance <= hi, ata_rad < np.deg2rad(combat["ata_deg"]),
            aa_rad < np.deg2rad(combat["aa_deg"]))


def envelope_specs(combat):
    specs = [("official", "none", None, deepcopy(combat))]
    for field, values in [("ata_deg", [30, 40, 45, 60, 90]),
                          ("aa_deg", [90, 105, 120, 135, 150]),
                          ("distance_max", [3000, 3500, 4000, 5000]),
                          ("distance_min", [500, 750, 1000]),
                          ("hold_steps", [1, 2, 3])]:
        for value in values:
            c = deepcopy(combat)
            if field.startswith("distance_"):
                distances = list(c["distance"])
                distances[0 if field.endswith("min") else 1] = value
                c["distance"] = tuple(distances)
            else:
                c[field] = value
            specs.append((f"{field}_{value}", field, value, c))
    c = deepcopy(combat); c["ata_deg"] = 45; c["aa_deg"] = 120
    specs.append(("supplement_ATA45_AA120", "supplement_combination", None, c))
    return specs


def diagnostic_config(config, name):
    result = deepcopy(config)
    if name == "hold1": result["combat"]["hold_steps"] = 1
    elif name == "ATA45": result["combat"]["ata_deg"] = 45
    elif name == "AA120": result["combat"]["aa_deg"] = 120
    elif name == "range4km": result["combat"]["distance"] = (result["combat"]["distance"][0], 4000)
    elif name == "horizon100": result["simulation"]["max_decision_steps"] = 100
    elif name == "sensor8km": result["sensing"]["UAV_range"] = 8000
    else: raise ValueError(name)
    return result


def choose_case(rows):
    groups = defaultdict(dict)
    for r in rows:
        if int(r["sampled_steps"]) == 2_000_000:
            key = int(r["environment_seed"])
            seed = int(r["training_seed"])
            if seed in groups[key]: raise ValueError("duplicate final scenario")
            groups[key][seed] = r
    candidates = []
    for env_seed, trio in groups.items():
        if set(trio) != {1, 2, 3} or trio[1]["outcome"] != "red": continue
        if len({r["action_seed"] for r in trio.values()}) != 1:
            raise ValueError("matched scenario has unequal action seeds")
        draws = [r for s, r in trio.items() if s != 1 and r["outcome"] == "draw"]
        p2 = [r for r in draws if int(r.get("P2_transitions") or 0) > 0]
        # Prefer BOTH weak policies having long P2 draws, then total exposure.
        score = (len(p2), min((int(r["P2_transitions"]) for r in p2), default=0),
                 sum(int(r["P2_transitions"]) for r in p2), len(draws), -env_seed)
        candidates.append((score, env_seed, trio))
    if not candidates: raise RuntimeError("no seed1-win matched scenario in existing audit")
    score, env_seed, trio = max(candidates, key=lambda c: c[0])
    return dict(environment_seed=env_seed, action_seed=int(trio[1]["action_seed"]),
                original_episode=int(trio[1]["episode"]), score=list(score),
                selection="maximize weak P2-draw count, minimum P2 duration, total P2 duration; deterministic tie-break",
                original_records={str(s): r for s, r in trio.items()})


def boundary_margin(state, config):
    bounds = config["battlefield"]
    return min(state.x-bounds["x"][0], bounds["x"][1]-state.x,
               state.y-bounds["y"][0], bounds["y"][1]-state.y,
               state.h-bounds["altitude"][0], bounds["altitude"][1]-state.h)


def replay_episode(actors, config, env_seed, action_seed, device="cuda"):
    """Observe resolver exactly once; sample exactly once per actor per decision.

    Pair geometry: post-physics/post-boundary/pre-combat. Phase: pre-action alive
    Blue count, so the transition killing the third Blue is NOT retrospectively P2.
    Trajectory aircraft states: post-combat; initial/pre-action coordinates retained.
    """
    env = Env(config, profile="main")
    obs, _ = env.reset(seed=env_seed)
    rows, pairs, snapshots = [], [], []
    original = env._resolve_attacks

    def observer():
        before = {}
        for aid in UAVS:
            if not env.entities[aid].state.alive: continue
            for bid in BLUE_IDS:
                if not env.entities[bid].state.alive: continue
                a, b = env.entities[aid].state, env.entities[bid].state
                g = geometry(a, b)
                flags = conditions(g.distance, g.ata, g.aa, env.config["combat"])
                assert all(flags) == inside(g, env.config), "reward/combat gate mismatch"
                line = g.relative_position/max(g.distance, 1e-9)
                before[(aid, bid)] = dict(
                    distance_m=g.distance, ATA_deg=float(np.rad2deg(g.ata)), AA_deg=float(np.rad2deg(g.aa)),
                    distance_gate=int(flags[0]), ATA_gate=int(flags[1]), AA_gate=int(flags[2]),
                    full_gate=int(all(flags)), streak_before=env._attack_streak.get((aid, bid), 0),
                    UAV_speed=a.v, Blue_speed=b.v,
                    closure_mps=float(np.dot(a.velocity_vector()-b.velocity_vector(), line)),
                    direct_visible=int(env.direct_visible(aid, bid)),
                    datalink_visible=int(env.datalink_visible(aid, bid)), team_visible=int(env.team_visible(bid)),
                    boundary_margin_m=boundary_margin(a, env.config),
                    heading_difference_deg=float(abs(np.rad2deg((a.psi-b.psi+np.pi) % (2*np.pi)-np.pi))),
                )
        events, deaths = original()
        hits = {(e["attacker"], e["target"]) for e in events}
        for key, record in before.items():
            expected = record["streak_before"]+1 if record["full_gate"] else 0
            # Actual kill clears counters, including simultaneous attacker death.
            record["attack_streak"] = expected  # resolver-evaluated progression BEFORE synchronous cleanup
            record["persisted_streak_after_cleanup"] = env._attack_streak.get(key, 0)
            record["kill_event"] = int(key in hits)
            record["combat_cleanup"] = int(key[0] in deaths or key[1] in deaths)
            assert (key in hits) == (expected >= int(env.config["combat"]["hold_steps"]))
            if not record["combat_cleanup"]:
                assert record["attack_streak"] == expected, "offline/real streak mismatch"
        snapshots.append((before, events, deaths))
        return events, deaths

    env._resolve_attacks = observer
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.manual_seed(action_seed)
        if torch.device(device).type == "cuda": torch.cuda.manual_seed_all(action_seed)
        while True:
            pre = {a: e.state.copy() for a, e in env.entities.items()}
            nblue = sum(pre[b].alive for b in BLUE_IDS)
            with torch.no_grad():
                actions = np.stack([actors.actors[i].sample(torch.as_tensor(obs[a], device=device).unsqueeze(0),
                                                           deterministic=False)[0].squeeze(0).cpu().numpy()
                                    for i, a in enumerate(RED_IDS)])
            obs, _, terminated, truncated, info = env.step(actions)
            captured, events, _ = snapshots[-1]
            row = dict(episode_step=env.step_count, phase=phase(nblue), alive_blue_pre_action=nblue,
                       alive_Blue_count=sum(env.entities[b].state.alive for b in BLUE_IDS),
                       alive_UAV_count=sum(env.entities[a].state.alive for a in UAVS),
                       event_reward=info["event_reward"], terminal_reward=info["terminal_reward"],
                       team_reward=info["team_reward"], winner=info["outcome"],
                       killed_ids="|".join(info["killed_ids"]), death_causes=json.dumps(info["death_causes"]),
                       attack_events=json.dumps(events), environment_seed=env_seed, action_seed=action_seed)
            for aid, entity in env.entities.items():
                st = entity.state
                for field, value in dict(x=st.x, y=st.y, altitude=st.h, speed=st.v,
                                         heading_deg=np.rad2deg(st.psi), pitch_deg=np.rad2deg(st.theta), alive=int(st.alive),
                                         pre_x=pre[aid].x, pre_y=pre[aid].y, pre_altitude=pre[aid].h,
                                         boundary_margin_m=boundary_margin(st, env.config)).items():
                    row[f"{aid}_{field}"] = float(value)
                row[f"{aid}_death_cause"] = info["death_causes"].get(aid)
                if aid in RED_IDS: row[f"{aid}_process_reward"] = info[f"{aid.lower()}_process_reward"]
                if aid in BLUE_IDS:
                    guidance = env.blue_policy._guidance_state[aid]
                    row[f"{aid}_target"] = guidance.target_id
                    row[f"{aid}_heading_command_deg"] = float(np.rad2deg(guidance.desired_heading))
                    row[f"{aid}_heading_change_deg"] = float(np.rad2deg((st.psi-pre[aid].psi+np.pi) % (2*np.pi)-np.pi))
            for i, aid in enumerate(UAVS, 1):
                for dim in range(3): row[f"{aid}_action_{dim}"] = float(actions[i, dim])
                row[f"{aid}_action_saturation"] = int(np.any(np.abs(actions[i]) >= .95))
                choices = {b: r for (a, b), r in captured.items() if a == aid}
                nearest = min(choices, key=lambda b: choices[b]["distance_m"]) if choices else None
                best = min(choices, key=lambda b: attack_gap(geometry(env.entities[aid].state, env.entities[b].state), env.config)) if choices else None
                # Reward selector is post-combat, not guessed from pre-combat geometry.
                selected = info.get(f"reward_target_{aid}")
                for label, target in [("nearest", nearest), ("most_attackable", best), ("reward_selected", selected)]:
                    row[f"{aid}_{label}_target"] = target
                    if target:
                        g = geometry(env.entities[aid].state, env.entities[target].state)
                        for key, value in dict(distance_m=g.distance, ATA_deg=np.rad2deg(g.ata), AA_deg=np.rad2deg(g.aa),
                                               direct_visible=int(env.direct_visible(aid, target)), datalink_visible=int(env.datalink_visible(aid, target))).items():
                            row[f"{aid}_{label}_{key}"] = float(value)
                for bid in BLUE_IDS:
                    r = captured.get((aid, bid))
                    if r is None: continue
                    for key in ("distance_m", "ATA_deg", "AA_deg", "full_gate", "attack_streak"):
                        row[f"{aid}_{bid}_{key}"] = r[key]
                    guidance = env.blue_policy._guidance_state[bid]
                    pair = dict(**r, agent=aid, blue=bid, episode_step=env.step_count,
                                environment_seed=env_seed, action_seed=action_seed, phase=row["phase"],
                                action_saturation=int(np.any(np.abs(actions[i]) >= .95)),
                                Blue_target=guidance.target_id, Blue_heading_deg=row[f"{bid}_heading_deg"],
                                UAV_heading_deg=row[f"{aid}_heading_deg"],
                                Blue_heading_change_deg=row[f"{bid}_heading_change_deg"],
                                Blue_relative_bearing_deg=180-r["AA_deg"],
                                remaining_steps=env.max_decision_steps-env.step_count,
                                optimistic_range_time_s=max(r["distance_m"]-env.config["combat"]["distance"][1], 0)/
                                (env.entities[aid].spec.v_max+env.entities[bid].spec.v_max))
                    pair.update({f"action_{d}": float(actions[i, d]) for d in range(3)})
                    pairs.append(pair)
            rows.append(row)
            if terminated or truncated:
                result = dict(info["episode_summary"])
                for r in pairs: r["outcome"] = result["outcome"]
                return result, rows, pairs


def scopes(pairs):
    return {"all": pairs, "P2": [r for r in pairs if r["phase"] == "P2"],
            "P2_draw": [r for r in pairs if r["phase"] == "P2" and r["outcome"] == "draw"],
            "P2_win": [r for r in pairs if r["phase"] == "P2" and r["outcome"] == "red"]}


def ratio(count, denominator):
    return count/denominator if denominator else None


def offline_streaks(rows, combat):
    result = []; previous = {}; lengths = {}
    for r in sorted(rows, key=lambda r: (r["environment_seed"], r["episode_step"], r["agent"], r["blue"])):
        key = (r["environment_seed"], r["agent"], r["blue"])
        gate = all(conditions(r["distance_m"], np.deg2rad(r["ATA_deg"]), np.deg2rad(r["AA_deg"]), combat))
        consecutive = previous.get(key) == r["episode_step"]-1
        streak = (lengths.get(key, 0) if consecutive else 0)+1 if gate else 0
        previous[key], lengths[key] = r["episode_step"], streak
        result.append((r, gate, streak))
    return result


def compute_statistics(pairs, combat, tag):
    funnel, angles, sensitivity, resets, environment = [], [], [], [], []
    reconstructed_all = {name: offline_streaks(pairs, c) for name, _, _, c in envelope_specs(combat)}
    for scope, rows in scopes(pairs).items():
        n = len(rows)
        D = sum(r["distance_gate"] for r in rows); A = sum(r["ATA_gate"] for r in rows); B = sum(r["AA_gate"] for r in rows)
        DA = sum(r["distance_gate"] and r["ATA_gate"] for r in rows)
        DB = sum(r["distance_gate"] and r["AA_gate"] for r in rows)
        AB = sum(r["ATA_gate"] and r["AA_gate"] for r in rows)
        F = sum(r["full_gate"] for r in rows)
        counts = [n, D, A, B, DA, DB, AB, F]+[sum(r["attack_streak"] >= k for r in rows) for k in (1, 2, 3)]
        labels = ["alive_pair", "distance", "ATA", "AA", "distance_AND_ATA", "distance_AND_AA", "ATA_AND_AA", "full_gate", "streak_ge1", "streak_ge2", "streak_ge3"]
        # Parallel predicates L1-L6 are NOT a sequential funnel. Explicit denominators.
        parents = [n, n, n, n, D, D, A, D, F, counts[8], counts[9]]
        parent_labels = ["alive_pair"]*4+["distance", "distance", "ATA", "distance", "full_gate", "streak_ge1", "streak_ge2"]
        for level, (label, count, den, parent) in enumerate(zip(labels, counts, parents, parent_labels)):
            funnel.append(dict(**tag, scope=scope, level=f"L{level}", predicate=label, count=count,
                               denominator=n, fraction=ratio(count, n), conditional_denominator=den,
                               conditional_on=parent, conditional_pass_rate=ratio(count, den),
                               streak_definition="real resolver; pair-step counts, not episode conversion"))
        if scope in ("P2_draw", "P2_win"):
            for aid in ("ALL_UAV", *UAVS):
                rr = rows if aid == "ALL_UAV" else [r for r in rows if r["agent"] == aid]
                for condition in ("all_distances", "distance_1_3km"):
                    selected = rr if condition == "all_distances" else [r for r in rr if r["distance_gate"]]
                    for name, cutoffs in [("ATA", [30, 45, 60, 90]), ("AA", [90, 120, 150])]:
                        v = np.asarray([r[f"{name}_deg"] for r in selected])
                        row = dict(**tag, scope=scope, agent=aid, condition=condition, angle=name, samples=len(v))
                        row.update({f"p{q}": float(np.percentile(v, q)) if len(v) else None for q in (10, 25, 50, 75, 90)})
                        row["minimum"] = float(v.min()) if len(v) else None
                        row.update({f"fraction_lt_{x}": float((v < x).mean()) if len(v) else None for x in cutoffs})
                        angles.append(row)
        for name, field, value, c in envelope_specs(combat):
            # Preserve P1->P2 streak continuity; do not restart a pair at scope entry.
            reconstructed = [(r, g, s) for r, g, s in reconstructed_all[name]
                             if scope == "all" or (r["phase"] == "P2" and
                                 (scope == "P2" or r["outcome"] == ("draw" if scope == "P2_draw" else "red")))]
            maxima = defaultdict(int); first = {}; full = set()
            for r, gate, streak in reconstructed:
                ep = r["environment_seed"]
                maxima[ep] = max(maxima[ep], streak)
                if gate: full.add(ep)
                if streak >= c["hold_steps"]: first.setdefault(ep, r["episode_step"])
            episodes = len({r["environment_seed"] for r in rows})
            sensitivity.append(dict(**tag, scope=scope, evidence=DIAGNOSTIC, setting=name, changed_field=field,
                                    changed_value=value, pair_exposures=n, full_gate_pair_steps=sum(g for _, g, _ in reconstructed),
                                    full_gate_occupancy=ratio(sum(g for _, g, _ in reconstructed), n), episodes=episodes,
                                    episodes_with_gate=len(full), episodes_streak_ge2=sum(v >= 2 for v in maxima.values()),
                                    episodes_streak_ge3=sum(v >= 3 for v in maxima.values()),
                                    episodes_with_theoretical_opportunity=len(first), theoretical_opportunity_fraction=ratio(len(first), episodes),
                                    longest_gate_length_distribution=json.dumps(dict(sorted(Counter(maxima.values()).items()))),
                                    first_qualifying_step_mean=float(np.mean(list(first.values()))) if first else None,
                                    first_qualifying_steps=json.dumps(first, sort_keys=True),
                                    episode_ge1_to_ge2=ratio(sum(v >= 2 for v in maxima.values()), len(full)),
                                    episode_ge2_to_ge3=ratio(sum(v >= 3 for v in maxima.values()), sum(v >= 2 for v in maxima.values()))))
        reset = Counter()
        for r in rows:
            if r["streak_before"] > 0 and not r["full_gate"]:
                failed = [key for key in ("distance", "ATA", "AA") if not r[f"{key}_gate"]]
                reset[failed[0] if len(failed) == 1 else "multiple_conditions"] += 1
        resets.append(dict(**tag, scope=scope, **{f"reset_by_{key}": reset[key] for key in ("distance", "ATA", "AA", "multiple_conditions")},
                           total_geometric_resets=sum(reset.values()),
                           excluded="kill/death cleanup, pair disappearance, episode end are censored, not geometric reset"))
        if scope.startswith("P2"):
            for aid in ("ALL_UAV", *UAVS):
                rr = rows if aid == "ALL_UAV" else [r for r in rows if r["agent"] == aid]
                row = dict(**tag, scope=scope, agent=aid, samples=len(rr))
                for key in ("UAV_speed", "Blue_speed", "closure_mps", "boundary_margin_m"):
                    v = np.asarray([r[key] for r in rr])
                    row.update({f"{key}_p{q}": float(np.percentile(v, q)) if len(v) else None for q in (10, 50, 90)})
                for key in ("direct_visible", "datalink_visible", "action_saturation"):
                    row[f"{key}_fraction"] = float(np.mean([r[key] for r in rr])) if rr else None
                unique = {(r["environment_seed"], r["episode_step"], r["blue"]): r for r in rr}
                row["team_invisible_fraction_unique_steps"] = ratio(sum(not r["team_visible"] for r in unique.values()), len(unique))
                row["nonclosing_fraction"] = ratio(sum(r["closure_mps"] <= 0 for r in rr), len(rr))
                row["tail_chase_nonclosing_fraction"] = ratio(sum(r["ATA_deg"] < 30 and r["AA_deg"] < 30 and r["closure_mps"] <= 0 for r in rr), len(rr))
                row["saturated_high_ATA_fraction"] = ratio(sum(r["action_saturation"] and r["ATA_deg"] >= 30 for r in rr), len(rr))
                row["optimistic_range_time_within_remaining_fraction"] = ratio(sum(r["optimistic_range_time_s"] <= r["remaining_steps"] for r in rr), len(rr))
                row["range_time_note"] = "lower bound (d-3000)+/(UAV vmax+Blue vmax); ignores heading, acceleration, vertical/boundary constraints; not a feasible intercept guarantee"
                row["Blue_target_counts"] = json.dumps(dict(Counter(r["Blue_target"] for r in unique.values())))
                environment.append(row)
    return funnel, sensitivity, resets, angles, environment


def closest_states(pairs, combat, tag):
    candidates = []
    lo, hi = combat["distance"]
    for r in pairs:
        if r["phase"] != "P2" or r["outcome"] != "draw": continue
        row = dict(**tag, **r)
        row.update(distance_violation_m=max(lo-r["distance_m"], r["distance_m"]-hi, 0),
                   ATA_excess_deg=r["ATA_deg"]-combat["ata_deg"], AA_excess_deg=r["AA_deg"]-combat["aa_deg"])
        row["normalized_violation_score"] = (max(lo-r["distance_m"], 0)/lo + max(r["distance_m"]-hi, 0)/hi +
                                              max(row["ATA_excess_deg"], 0)/combat["ata_deg"] + max(row["AA_excess_deg"], 0)/combat["aa_deg"])
        candidates.append(row)
    return sorted(candidates, key=lambda r: (r["normalized_violation_score"], r["episode_step"], r["agent"]))[:10]


def plot_case(rows, output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip((*RED_IDS, *BLUE_IDS), ["black", "tab:blue", "tab:orange", "tab:green", "#a33030", "#d45757", "#bb6ba5", "#884f66"]))
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    event_labels = []
    p2 = next((r for r in rows if r["phase"] == "P2"), None)
    for aid in (*RED_IDS, *BLUE_IDS):
        # Include death position, never continue a frozen dead aircraft's path.
        rr = [r for r in rows if r[f"{aid}_alive"] or r[f"{aid}_death_cause"]]
        if not rr: continue
        x = [rr[0][f"{aid}_pre_x"]]+[r[f"{aid}_x"] for r in rr]
        y = [rr[0][f"{aid}_pre_y"]]+[r[f"{aid}_y"] for r in rr]
        ax.plot(np.asarray(x)/1000, np.asarray(y)/1000, color=colors[aid], label=aid)
        ax.scatter(x[0]/1000, y[0]/1000, color=colors[aid], marker="o", s=20)
        ax.scatter(x[-1]/1000, y[-1]/1000, color=colors[aid], marker="s", s=25)
        if p2 and (p2[f"{aid}_alive"] or p2[f"{aid}_death_cause"]):
            ax.scatter(p2[f"{aid}_pre_x"]/1000, p2[f"{aid}_pre_y"]/1000, color=colors[aid], marker="D", s=35)
        death = next((r for r in rr if r[f"{aid}_death_cause"]), None)
        if death:
            ax.scatter(death[f"{aid}_x"]/1000, death[f"{aid}_y"]/1000, color=colors[aid], marker="x", s=60)
            event_labels.append(f"{aid}: {death[f'{aid}_death_cause']}, t={int(death['episode_step'])}")
    if event_labels:
        ax.text(.02, .98, "Death events (x markers):\n"+"\n".join(event_labels), transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    ax.set(xlabel="x (km)", ylabel="y (km)", title=f"Same scenario {rows[0]['environment_seed']}; training seed {seed}\n{rows[-1]['winner']}; o=start, square=end, diamond=P2 entry, x=death")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(output/f"xy_seed{seed}.png", dpi=180); plt.close(fig)
    pp = [r for r in rows if r["phase"] == "P2"]
    for metric in ("distance", "ata", "aa", "actions", "streak", "heading"):
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
        for aid, ax in zip(UAVS, axes):
            units = dict(distance="Distance (km)", ata="ATA (deg)", aa="AA (deg)", actions="Normalized action",
                         streak="Gate / streak", heading="Heading (deg)")
            ax.set_ylabel(f"{aid}\n{units[metric]}")
            if not pp:
                ax.text(.5, .5, "No actual P2 transition (possibly simultaneous final kills)", ha="center", transform=ax.transAxes)
                continue
            for dim in (range(3) if metric == "actions" else [0]):
                xx, yy = [], []
                for r in pp:
                    bids = [b for b in BLUE_IDS if r.get(f"{aid}_{b}_distance_m") is not None]
                    if metric == "actions": value = r[f"{aid}_action_{dim}"] if r[f"{aid}_alive"] or r[f"{aid}_death_cause"] else None
                    elif metric == "heading": value = r[f"{aid}_heading_deg"] if bids else None
                    elif not bids: value = None
                    else:
                        suffix = dict(distance="distance_m", ata="ATA_deg", aa="AA_deg", streak="attack_streak")[metric]
                        value = r[f"{aid}_{bids[0]}_{suffix}"]
                        if metric == "distance": value /= 1000
                    xx.append(r["episode_step"]); yy.append(np.nan if value is None else value)
                ax.plot(xx, yy, label=f"action {dim}" if metric == "actions" else metric, color=None if metric == "actions" else colors[aid])
            if metric == "distance": ax.axhspan(1, 3, color="green", alpha=.1)
            if metric in ("ata", "aa", "streak"): ax.axhline(dict(ata=30, aa=90, streak=3)[metric], color="gray", ls="--")
            if metric == "actions": ax.axhline(.95, color="gray", ls="--"); ax.axhline(-.95, color="gray", ls="--")
            if metric == "heading":
                vals = []
                for r in pp:
                    b = next((b for b in BLUE_IDS if r.get(f"{aid}_{b}_distance_m") is not None), None)
                    vals.append(r[f"{b}_heading_deg"] if b else np.nan)
                ax.plot([r["episode_step"] for r in pp], vals, ls="--", color="red", label="last Blue heading")
            if metric == "streak":
                gate = [max((r[f"{aid}_{b}_full_gate"] for b in BLUE_IDS
                             if r.get(f"{aid}_{b}_full_gate") is not None), default=np.nan) for r in pp]
                ax.step([r["episode_step"] for r in pp], gate, where="mid", ls=":", label="full gate")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        axes[-1].set_xlabel("Decision step (1 s); P2 phase; geometry pre-combat")
        fig.suptitle(f"Training seed {seed}: P2 {metric}")
        fig.savefig(output/f"p2_{metric}_seed{seed}.png", dpi=170); plt.close(fig)


COMBAT_TEXT = """# Real combat contract (source-verified)

Sources: env/mavuav.py::step/_apply_boundaries/_resolve_attacks/_termination;
env/geometry.py::compute_pairwise_geometry; env/reward_role_v39.py::attack_gate_indicator.

LOS = (target position - attacker position) / distance.
ATA = acos(attacker 3D velocity unit vector dot LOS).
AA = acos(target 3D velocity unit vector dot LOS), NOT target-to-attacker LOS.
Thus ideal parallel tail chase has ATA=AA=0; head-on has ATA=0, AA=180 deg.
Distances: metres. Internal angles: radians; CSV/plots: degrees.

```text
sample four Red actors once each (including inactive slots, evaluator order)
Blue guidance/action updated from pre-action states (refresh every 2 steps or invalid target)
map actions once; advance 10 RK4 physics substeps of 0.1 s
apply Red boundary deaths; check Blue boundary invariant
for each living attacker and opposite-team target:
    MAV unarmed -> streak[MAV,target] = 0; skip
    dead target -> streak = 0; skip
    compute post-physics/post-boundary geometry
    gate = 1000 <= d <= 3000 AND ATA < radians(30) AND AA < radians(90)
    streak = previous_streak + 1 if gate else 0
    if streak >= 3: append attacker-target kill candidate
after ALL candidate collection: deactivate each candidate target synchronously
multiple attackers can have events for one target; shared kill reward counts target ONCE
one attacker can progress against multiple targets independently
clear counters with attacker/target killed in this combat resolution
increment decision step; MAV death has precedence over all-Blue-dead Red win
compute shared event/terminal reward, then process rewards on surviving visible targets
```

Hold=3 means three consecutive 1 s decision-end checks (two seconds between
first and third observation); it does NOT guarantee continuous geometry within substeps.
Blue motion/turn during the current second is included BEFORE streak update.
Boundary-dead attackers are skipped; historical stale dictionary entries for such
attackers are not exposures/attack opportunities and must not be counted.
Reward gate and combat geometric predicates agree; reward target population/timing
differs intentionally: post-combat surviving team-visible targets vs pre-combat pairs.
No sensing requirement in real combat. Kill events, death cleanup and horizon censor
recorded sequences; counterfactual occupancy is not counterfactual actual combat.
"""


def case_report(cases, selection, output, closest, counterfactual):
    lines = ["# 同场景三策略轨迹诊断", "", f"环境 seed={selection['environment_seed']}，action seed={selection['action_seed']}。",
             "MEASURED 为真实记录；INFERENCE 为解释。仅一个 matched scenario，不能代替200局结果。", ""]
    for seed, (result, rows, pairs) in cases.items():
        first = next((r["episode_step"] for r in rows if r["attack_events"] != "[]" and any(e["target"] in BLUE_IDS for e in json.loads(r["attack_events"]))), None)
        p2_start = next((r["episode_step"] for r in rows if r["phase"] == "P2"), None)
        lines += [f"## Seed{seed}", "", f"MEASURED: outcome={result['outcome']}; kills={result['red_attack_kills']}; return={result['episode_return']:.3f}; length={result['episode_length']}; MAV={result['mav_survived']}; UAV survivors={result['red_uav_survivors']}.",
                  f"Phase A 初始接敌: reset→首次攻击；Phase B 首次击杀 decision step={first}。",
                  f"Phase C 中期多目标交战: 首杀→P2；Phase D 实际P2首个transition={p2_start}；Phase E 结束={result['episode_length']}。", ""]
        for aid in UAVS:
            rr = [r for r in pairs if r["phase"] == "P2" and r["agent"] == aid]
            if not rr:
                lines.append(f"- {aid}: 无有效P2 pair样本（已死亡或本局未经历P2），不可用0替代缺失几何。")
                continue
            failures = Counter(k for r in rr for k in ("distance", "ATA", "AA") if not r[f"{k}_gate"])
            milestones = {key: next((r["episode_step"] for r in rr if r[key]), None) for key in ("distance_gate", "ATA_gate", "AA_gate", "full_gate")}
            sat = np.mean([r["action_saturation"] for r in rr]); tail = sum(r["ATA_deg"] < 30 and r["AA_deg"] < 30 and r["closure_mps"] <= 0 for r in rr)
            death = next((r for r in rows if r[f"{aid}_death_cause"]), None)
            lines.append(f"- MEASURED {aid}: P2有效样本={len(rr)}；min distance={min(r['distance_m'] for r in rr):.1f}m；min ATA={min(r['ATA_deg'] for r in rr):.2f}°；min AA={min(r['AA_deg'] for r in rr):.2f}°；gate steps={sum(r['full_gate'] for r in rr)}；max real streak={max(r['attack_streak'] for r in rr)}。")
            lines.append(f"  条件失败次数（可重叠）={dict(failures)}；首次满足step={milestones}；任一动作维饱和比例={sat:.3f}；定义的tail-chase/nonclosing样本={tail}；min boundary margin={min(r['boundary_margin_m'] for r in rr):.1f}m；death={death[f'{aid}_death_cause'] if death else 'alive'}。")
            overshoot = sum(r["distance_m"] < 1000 and r["closure_mps"] <= 0 for r in rr)
            lines.append(f"  MEASURED: 距离<1km且非闭合样本={overshoot}（近距离离开标志，不自动等同于overshoot因果）；Blue heading change最大绝对值={max(abs(r['Blue_heading_change_deg']) for r in rr):.2f}°/step。")
            lines.append("  INFERENCE: 以同时条件与时间序列判断瓶颈；各自最小ATA/AA/距离可能来自不同step，不能拼成一个可攻击状态。Blue转向、饱和或边界的同现不能证明因果。")
        lines.append("")
    lines += ["## 局部反事实与限制", "", "见上一级 counterfactual_case.csv：DIAGNOSTIC COUNTERFACTUAL，冻结policy、改变内存单一环境因素；并非正式性能结果。",
              "closest_gate_states.csv按明确归一化违反程度排序；这是诊断排序，不是policy target/action。",
              "最短距离闭合时间=(d-3000)+/600为极乐观下界，不保证受转弯、加速度和边界约束的真实拦截可行。",
              "本轮没有200局原始pair轨迹；本工具的精确漏斗/角度/敏感性为所选案例，不能外推为200局总体统计。"]
    (output/"trajectory_case_report.md").write_text("\n".join(lines), encoding="utf-8")


def recover_population_p2(prior_dir, output):
    """Recover exact integer counts from stored n and full-precision fractions.

    One Blue in P2 makes old nearest geometry exactly the unique pair. No means or
    marginal probabilities are multiplied to invent unavailable intersections.
    """
    episodes = [r for r in read_csv(prior_dir/"replay_episodes.csv") if int(r["sampled_steps"]) == 2_000_000]
    agents = [r for r in read_csv(prior_dir/"replay_agents.csv") if int(r["sampled_steps"]) == 2_000_000]
    result = []
    for seed in (1, 2, 3):
        for scope in ("P2", "P2_draw", "P2_win"):
            ep = [r for r in episodes if int(r["training_seed"]) == seed and int(r["P2_transitions"]) > 0
                  and (scope == "P2" or r["outcome"] == ("draw" if scope == "P2_draw" else "red"))]
            keys = {(r["run"], r["episode"]) for r in ep}
            aa = [r for r in agents if (r["run"], r["episode"]) in keys]
            counts = Counter()
            maxima = defaultdict(int)
            for r in aa:
                n = int(r["P2_geometry_samples"])
                counts["L0_alive_pair"] += n
                maxima[(r["run"], r["episode"])] = max(maxima[(r["run"], r["episode"])], int(r["P2_max_real_streak"] or 0))
                if not n: continue
                for label, field in [("L1_distance", "P2_distance_gate_fraction"),
                                     ("L6_ATA_AND_AA", "P2_angle_gate_fraction"),
                                     ("L7_full_gate", "P2_full_gate_fraction")]:
                    value = float(r[field])*n
                    if abs(value-round(value)) > 1e-7: raise AssertionError("cannot exactly recover count from fraction")
                    counts[label] += round(value)
            row = dict(training_seed=seed, scope=scope, population="prior_200_episode",
                       episodes=len(ep), P2_transitions=sum(int(r["P2_transitions"]) for r in ep), **dict(counts),
                       episodes_max_streak_ge1=sum(v >= 1 for v in maxima.values()),
                       episodes_max_streak_ge2=sum(v >= 2 for v in maxima.values()),
                       episodes_max_streak_ge3=sum(v >= 3 for v in maxima.values()),
                       mean_P2_transitions=float(np.mean([int(r["P2_transitions"]) for r in ep])) if ep else None,
                       missing="separate ATA/AA marginals and conditional intersections/quantiles; raw pair time sequences")
            for key in ("L1_distance", "L6_ATA_AND_AA", "L7_full_gate"):
                row[f"{key}_fraction"] = ratio(counts[key], counts["L0_alive_pair"])
            result.append(row)
    table(output/"prior_population_p2_reference.csv", result)
    return result


def read_numeric_csv(path):
    rows = read_csv(path)
    for r in rows:
        for k, v in r.items():
            if v == "": r[k] = None; continue
            try: r[k] = float(v)
            except ValueError: pass
    return rows


def finish_saved_analysis(output, prior_dir):
    """Analysis-only refresh of OUR generated audit outputs; no policy calls."""
    output = Path(output)
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    for filename, expected in summary["input_sha256"].items():
        if sha(filename) != expected: raise AssertionError(f"protected input changed: {filename}")
    selection = json.loads((output/"trajectory_case/case_selection.json").read_text(encoding="utf-8"))
    cases = {}
    for seed in (1, 2, 3):
        trajectory = read_numeric_csv(output/f"trajectory_case/seed{seed}_trajectory.csv")
        pairs = read_numeric_csv(output/f"trajectory_case/seed{seed}_pair_steps.csv")
        cases[seed] = summary["case_results"][str(seed)], trajectory, pairs
        plot_case(trajectory, output/"trajectory_case", seed)
    cfg = load_environment_config(ROOT/"configs/env_v311.yaml")
    # Recompute ONLY saved case statistics, preserving full200 rows if explicitly run.
    combined = [[] for _ in range(5)]
    for seed, (_, _, pairs) in cases.items():
        tag = dict(run=f"happo_v311_seed{seed}_2m", training_seed=seed, population="matched_case_1_episode")
        for dest, source in zip(combined, compute_statistics(pairs, cfg["combat"], tag)): dest.extend(source)
    names = ("gate_funnel.csv", "gate_sensitivity.csv", "streak_reset_causes.csv", "p2_angle_distribution.csv", "p2_environment_diagnostics.csv")
    for name, rows in zip(names, combined):
        rows += [r for r in read_csv(output/name) if r["population"] != "matched_case_1_episode"]
        table(output/name, rows)
    recovered = recover_population_p2(prior_dir, output)
    case_report(cases, selection, output/"trajectory_case", [], summary["counterfactual_results"])
    text = ["# v3.11 第二阶段专项审计", "", "## 范围与证据", "",
            f"同场景 environment seed={selection['environment_seed']}；action seed={selection['action_seed']}；3个final policy。",
            f"MEASURED：3局与上一轮、官方stochastic evaluator逐字段精确一致；{len(summary['counterfactual_results'])}局单变量内存反事实；未训练。",
            "已复用上轮600局final数据；没有重复3×200。原始200局缺少全pair轨迹，因此无法恢复全量ATA/AA分位数与单变量敏感性。",
            "gate_funnel.csv等精确逐pair统计population=matched_case_1_episode；prior_population_p2_reference.csv为原200局可精确恢复的计数。",
            "所有反事实为DIAGNOSTIC ONLY；不能视为正式评估或重新训练效果。", "",
            "## 同场景结果", "", "|Training seed|Outcome|Kills|Return|Length|UAV survivors|", "|---|---|---|---|---|---|"]
    for seed, (r, _, _) in cases.items():
        text.append(f"|{seed}|{r['outcome']}|{r['red_attack_kills']}|{r['episode_return']:.3f}|{r['episode_length']}|{r['red_uav_survivors']}|")
    text += [""]
    for seed, (_, rows, pairs) in cases.items():
        p2rows = [r for r in rows if r["phase"] == "P2"]
        text.append(f"MEASURED seed{seed}: P2 transitions={len(p2rows)}；首个P2 step={p2rows[0]['episode_step'] if p2rows else 'n.a.'}；P2 gate pair-steps={sum(r['full_gate'] for r in pairs if r['phase']=='P2')}；最终transition前alive Blue={rows[-1]['alive_blue_pre_action']}。无P2不等于0攻击能力。")
    text += ["",
             "## P2距离条件下的角度分布", "", "|Seed|Distance samples|ATA min / median|AA min / median|P(ATA<30 given distance)|P(AA<90 given distance)|",
             "|---|---|---|---|---|---|"]
    for seed in (2, 3):
        rr = [r for r in cases[seed][2] if r["phase"] == "P2" and r["distance_gate"]]
        ata = [r["ATA_deg"] for r in rr]; aa = [r["AA_deg"] for r in rr]
        if rr:
            text.append(f"|{seed}|{len(rr)}|{min(ata):.2f} / {np.median(ata):.2f}|{min(aa):.2f} / {np.median(aa):.2f}|{np.mean(np.asarray(ata)<30):.1%}|{np.mean(np.asarray(aa)<90):.1%}|")
        else:
            text.append(f"|{seed}|0|n.a.|n.a.|n.a.|n.a.|")
    text += ["", "## 原轨迹单变量P2敏感性", "", "|Seed|Setting|Gate pair-steps|Episodes ge2|Episodes ge3|", "|---|---|---|---|---|"]
    for r in combined[1]:
        if r["scope"] == "P2_draw" and r["training_seed"] in (2, 3) and r["setting"] in (
                "official", "ata_deg_45", "ata_deg_60", "ata_deg_90", "aa_deg_120",
                "distance_max_4000", "distance_max_5000", "distance_min_500", "hold_steps_1", "hold_steps_2"):
            text.append(f"|{r['training_seed']}|{r['setting']}|{r['full_gate_pair_steps']}|{r['episodes_streak_ge2']}|{r['episodes_streak_ge3']}|")
    text += ["", "离线表只重判定状态合格程度，未替换死亡或改变轨迹；单局敏感性不能证明全局应修改某个门限。", "",
             "## 环境因素", ""]
    for seed in (2, 3):
        rr = [r for r in cases[seed][2] if r["phase"] == "P2"]
        unique = {r["episode_step"]: r for r in rr}
        if not rr:
            text.append(f"- seed{seed}: 无有效P2 pair样本，相关统计n.a.。")
            continue
        text.append(f"- MEASURED seed{seed}: P2 team-invisible={np.mean([not r['team_visible'] for r in unique.values()]):.1%}；非闭合pair-step={np.mean([r['closure_mps']<=0 for r in rr]):.1%}；严格定义的tail-chase/nonclosing={sum(r['ATA_deg']<30 and r['AA_deg']<30 and r['closure_mps']<=0 for r in rr)}；Blue targets={dict(Counter(r['Blue_target'] for r in unique.values()))}。")
        for aid in UAVS:
            ar = [r for r in rr if r["agent"] == aid]
            if ar:
                text.append(f"  {aid}速度median={np.median([r['UAV_speed'] for r in ar]):.1f}；Blue速度median={np.median([r['Blue_speed'] for r in ar]):.1f}；closure median={np.median([r['closure_mps'] for r in ar]):.1f}m/s。")
        for row in cases[seed][1]:
            if row["phase"] != "P2": continue
            for aid in UAVS:
                if row[f"{aid}_death_cause"]:
                    text.append(f"  {aid} death={row[f'{aid}_death_cause']}，step={row['episode_step']}，位置=({row[f'{aid}_x']:.1f},{row[f'{aid}_y']:.1f},{row[f'{aid}_altitude']:.1f})m。")
    text += ["- UAV/Blue v_min/v_max和nx/ny/nz完全一致；实际速度分布见上，不能用相同速度限替代实际速度。",
             "- Boundary减少平台数量，但不能仅由末杀失败与boundary同现证明它是主要原因。",
             "- horizon100和sensor8km结果见下表；单场景不能证明总体horizon/sensing合理性；team-invisible应保留为次生候选。",
             "- Blue最近目标与转向是MEASURED；它是否造成病态几何是INCONCLUSIVE，没有隔离Blue policy的实验。",
             "- 受动力学限制的最短拦截时间未求解；(d-3km)+/600仅是乐观距离下界，不能用于证明一定来得及攻击。", "",
             "## 单episode环境反事实", "", "|Seed|Condition|Outcome|Kills|Length|", "|---|---|---|---|---|"]
    for r in summary["counterfactual_results"]:
        text.append(f"|{r['training_seed']}|{r['condition']}|{r['outcome']}|{r['red_attack_kills']}|{r['episode_length']}|")
    text += ["", "提前击杀会改变后续目标/Blue转向/观测，即使相同RNG也不再是相同运动；因此hold1可能比原policy更差，不代表hold1本身难度更高。", "",
             "combat配置由双方共享：内存hold/ATA/AA/range反事实同时影响Red与Blue判定，不能解释成只放宽Red门限。离线gate_sensitivity.csv则只审计已记录UAV→Blue状态，不改变双方运动。", "",
             "## Combat gate分类", "", "ATTACK_CONDITION_STRICT_BUT_LEARNABLE。seed1正式200局41%win、3.035kills证明联合gate和hold3可实现；随机10局gate0证明初始发现不容易，但不足以证明病态。",
             "REFUTED（当前P2 draw主要瓶颈）：HOLD_STEPS_IS_PRIMARY_BOTTLENECK。原200局P2 draw中58/67、68/69、22/22从未进入gate，减少hold无法救活这些原轨迹。", "",
             "## Hypothesis matrix", "", "|Hypothesis|Evidence|Status|Severity|", "|---|---|---|---|"]
    hypotheses = [
        ("H1 distance restrictive", "case有37/25个有效距离样本但ATA全部失败；放宽range不救原P2", "PARTIALLY_SUPPORTED", "medium"),
        ("H2 ATA restrictive", "case距离内ATA minimum62.00/45.45；60/90离线才恢复机会；是否应改阈值尚未证明", "PARTIALLY_SUPPORTED", "high"),
        ("H3 AA restrictive", "距离内AA pass67.6%/64%；AA120单独仍gate0", "PARTIALLY_SUPPORTED", "medium"),
        ("H4 hold primary", "原200局大多数P2 draw没有第一步gate；case hold1/2无机会", "REFUTED", "low"),
        ("H5 combined sparse", "all alivepair gate低但seed1能学会；病态性质未证明", "PARTIALLY_SUPPORTED", "high"),
        ("H6 horizon primary", "两弱case P2有55步；100步反事实仍不胜；总体延时效果未测", "INCONCLUSIVE", "medium"),
        ("H7 sensing P2 primary", "seed3case60%不可见，sensor8km仍不胜；无法确定主因", "PARTIALLY_SUPPORTED", "medium"),
        ("H8 equal-speed tail lock", "相同速度限但实际150vs277；case严格tail-chase样本0", "INCONCLUSIVE", "medium"),
        ("H9 Blue pathological", "Blue持续追UAV2/切至UAV1；只有同现，没有Blue-policy干预", "INCONCLUSIVE", "medium"),
        ("H10 authority insufficient", "同控制限下seed1可胜；case饱和与高ATA同现不证明物理不可达", "INCONCLUSIVE", "medium"),
        ("H11 boundary contributes", "case各损失一架support UAV但主攻击者存活；原200局boundary重要", "PARTIALLY_SUPPORTED", "medium"),
        ("H12 initial headon longterm cause", "相同环境初态seed1胜、弱seed均17步首杀；差异主要出现在后续几何", "INCONCLUSIVE", "medium"),
        ("H13 reward alignment insufficient", "reward/combat gate一致，但reward postkill population与角度连续分数不同；未隔离因果", "PARTIALLY_SUPPORTED", "medium"),
        ("H14 optimization/local optimum primary", "同合同同初态policy产生显著不同转向/几何；具体优化原因未证明", "PARTIALLY_SUPPORTED", "high"),
    ]
    for h, evidence, status, severity in hypotheses: text.append(f"|{h}|{evidence}|{status}|{severity}|")
    text += ["", "## 下一步优先项（INFERENCE）", "",
             "若必须在给定类别中选一个，优先algorithm侧再接敌/优化机制的最小验证，而非直接放宽正式combat条件。",
             "依据：同环境同初态存在成功policy，弱policy长期ATA远超阈值；45度微放宽、延时或扩大传感未恢复成功。",
             "这不证明必须增加新网络或模块；本任务未实施任何算法、reward或环境修改。",
             "全量条件敏感性仍缺失，不能从这一局决定全局阈值合理性或因果归因。"]
    (output/"research_report.md").write_text("\n".join(text), encoding="utf-8")
    summary["recovered_population_p2"] = recovered
    summary["hypothesis_matrix"] = [dict(hypothesis=h, evidence=e, status=s, severity=v) for h,e,s,v in hypotheses]
    summary["next_priority"] = dict(category="algorithm", epistemic_status="INFERENCE", scope="minimal re-engagement/optimization validation; not new architecture or a proved cause")
    dump(output/"summary.json", summary)
    print(f"Saved-output analysis complete (no replay): {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-audit", type=Path, default=ROOT/"outputs/audits/v311_happo_failure_audit_200ep_20261007_141809")
    parser.add_argument("--run-dir", type=Path, nargs=3, default=[ROOT/f"outputs/happo_v311_seed{s}_2m" for s in (1, 2, 3)])
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/audits/v311_attack_geometry_audit")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--full-replay-episodes", type=int, default=0, choices=[0, 200], help="EXPLICIT opt-in for missing population pair trajectories; default never reruns 600 episodes")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument("--postprocess-existing", type=Path, help="only recompute our saved audit tables/report; no actor loading or replay")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU fallback")
    if args.postprocess_existing:
        finish_saved_analysis(args.postprocess_existing, args.prior_audit)
        return
    if args.output.exists(): raise FileExistsError(f"audit output exists; choose a new --output: {args.output}")
    cfg = load_environment_config(ROOT/"configs/env_v311.yaml")
    prior = read_csv(args.prior_audit/"replay_episodes.csv")
    for seed in (1, 2, 3):
        rr = [r for r in prior if int(r["training_seed"]) == seed and int(r["sampled_steps"]) == 2_000_000]
        if len(rr) != 200 or {int(r["environment_seed"]) for r in rr} != set(range(1000, 1200)):
            raise ValueError("prior final audit must have seeds1000-1199 exactly once per training seed")
        if any(r["profile"] != "main" or int(r["action_seed"]) != int(r["environment_seed"])+1000 for r in rr):
            raise ValueError("prior stochastic main seed contract mismatch")
    selection = choose_case(prior)
    paths = sorted((ROOT/"algorithm").rglob("*.py"))+sorted((ROOT/"env").rglob("*.py"))+sorted((ROOT/"configs").glob("*.yaml"))
    paths += sorted(args.prior_audit.glob("*"))
    paths += [p for run in args.run_dir for p in run.iterdir() if p.is_file()]
    hashes = {str(p): sha(p) for p in paths if p.is_file()}
    args.output.mkdir(parents=True); case_dir = args.output/"trajectory_case"; case_dir.mkdir()
    dump(case_dir/"case_selection.json", selection)
    (args.output/"combat_contract.md").write_text(COMBAT_TEXT, encoding="utf-8")
    combined = [[] for _ in range(5)]; closest = []; cases = {}; counterfactual = []; checks = {}
    for run in args.run_dir:
        payload = torch.load(run/"checkpoint_final.pt", map_location="cpu", weights_only=False)
        validate_checkpoint_contract(payload, cfg)
        if payload["environment_config"] != cfg or int(payload["sampled_steps"]) != 2_000_000:
            raise ValueError("checkpoint must be exact2M matching frozen env_v311")
        if (payload["actor_variant"], payload["critic_variant"], payload["method_variant"]) != ("vanilla", "mlp", "baseline"):
            raise ValueError("vanilla baseline contract required")
        tc = payload["trainer_config"]; seed = int(tc["seed"])
        if seed in cases or seed not in (1, 2, 3): raise ValueError("expected three distinct training seeds1-3")
        actors = IndependentActors(hidden_dim=tc["hidden_dim"], log_std_init=tc.get("actor_log_std_init", -.5)).cuda().eval()
        actors.load_state_dict(payload["actors"])
        params = {k: v.clone() for k, v in actors.state_dict().items()}
        print(f"Case replay seed{seed}: env={selection['environment_seed']} action={selection['action_seed']}", flush=True)
        result, rows, pairs = replay_episode(actors, cfg, selection["environment_seed"], selection["action_seed"])
        old = selection["original_records"][str(seed)]
        for key in ("outcome", "episode_length", "red_attack_kills", "blue_attack_kills", "red_uav_survivors", "episode_return"):
            expected = old[key] if key == "outcome" else float(old[key])
            if result[key] != expected: raise AssertionError(f"prior replay mismatch seed{seed} {key}: {result[key]} vs {expected}")
        # Official evaluator equality: ONE episode only, not a repeated formal200 evaluation.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            formal = evaluate_actors(actors, cfg, 1, "main", selection["environment_seed"], "cuda", deterministic=False, action_seed=selection["action_seed"])[0]
        if result != formal: raise AssertionError("observer changed official stochastic episode summary")
        cases[seed] = result, rows, pairs
        table(case_dir/f"seed{seed}_trajectory.csv", rows)
        table(case_dir/f"seed{seed}_pair_steps.csv", pairs)
        tag = dict(run=run.name, training_seed=seed, population="matched_case_1_episode")
        for dest, source in zip(combined, compute_statistics(pairs, cfg["combat"], tag)): dest.extend(source)
        closest.extend(closest_states(pairs, cfg["combat"], tag))
        if not args.skip_counterfactual:
            for condition in ("hold1", "ATA45", "AA120", "range4km", "horizon100", "sensor8km"):
                cc = diagnostic_config(cfg, condition)
                cr, _, _ = replay_episode(actors, cc, selection["environment_seed"], selection["action_seed"])
                counterfactual.append(dict(**tag, evidence="DIAGNOSTIC COUNTERFACTUAL", condition=condition, **cr))
                print(f"  {condition}: {cr['outcome']}, kills={cr['red_attack_kills']}", flush=True)
        if args.full_replay_episodes:
            population = []
            for ep in range(200):
                print(f"EXPLICIT full pair replay seed{seed} episode{ep+1}/200", flush=True)
                _, _, pp = replay_episode(actors, cfg, 1000+ep, 2000+ep)
                population.extend(pp)
            table(args.output/f"seed{seed}_population_pair_steps.csv", population)
            tag = dict(run=run.name, training_seed=seed, population="full_200_episode_replay")
            for dest, source in zip(combined, compute_statistics(population, cfg["combat"], tag)): dest.extend(source)
        checks[f"seed{seed}_actor_parameters_unchanged"] = all(torch.equal(params[k], v) for k, v in actors.state_dict().items())
        checks[f"seed{seed}_official_evaluator_exact_equality"] = result == formal
    for path, before in hashes.items():
        if sha(path) != before: raise AssertionError(f"read-only source changed: {path}")
    checks["all_input_SHA256_unchanged"] = True
    for filename, values in zip(("gate_funnel.csv", "gate_sensitivity.csv", "streak_reset_causes.csv", "p2_angle_distribution.csv", "p2_environment_diagnostics.csv"), combined):
        table(args.output/filename, values)
    if closest: table(args.output/"closest_gate_states.csv", closest)
    if counterfactual: table(args.output/"counterfactual_case.csv", counterfactual)
    # Reuse ONLY L0/L7 from prior raw counts. Other predicates are genuinely missing.
    reused = []
    for seed in (1, 2, 3):
        rr = [r for r in prior if int(r["sampled_steps"]) == 2_000_000 and int(r["training_seed"]) == seed]
        n = sum(int(r["pair_exposures"]) for r in rr); f = sum(int(r["gate_pair_steps"]) for r in rr)
        reused.append(dict(training_seed=seed, population="prior_200_episode", alive_pair_steps=n, full_gate_pair_steps=f,
                           full_gate_fraction=f/n, missing="raw ATA/AA/distance predicates, angles, closure and consecutive pair trajectories"))
    table(args.output/"prior_population_gate_reference.csv", reused)
    case_report(cases, selection, case_dir, closest, counterfactual)
    summary = dict(evidence=DIAGNOSTIC, classification="ATTACK_CONDITION_STRICT_BUT_LEARNABLE",
                   classification_basis="Prior seed1 200-episode win41%, kills3.035 proves realizable learned combat; case sensitivity does not establish population pathologies",
                   device="cuda", protocol=dict(profile="main", deterministic=False, case_environment_seed=selection["environment_seed"], case_action_seed=selection["action_seed"]),
                   case_results={str(s): r[0] for s, r in cases.items()}, checks=checks,
                   global_pair_statistics_available=bool(args.full_replay_episodes),
                   missing_population_statistics="none" if args.full_replay_episodes else "prior audit aggregated away raw all-pair trajectories; exact200 funnel/quantiles/sensitivity unavailable without explicit new replay",
                   input_sha256=hashes, counterfactual_results=counterfactual)
    dump(args.output/"summary.json", summary)
    report = ["# v3.11 第二阶段 attack geometry audit", "", f"DIAGNOSTIC ONLY；matched environment seed={selection['environment_seed']}。",
              "正式环境、算法、reward、配置及checkpoint未修改；见summary SHA和逐actor检查。",
              "本轮精确pair统计默认仅为3个单局，population字段明确区分；不能标成200局总体漏斗。",
              "上一轮200局原始pair缺失；本轮复用L0/L7和episode结果，生成选定场景的完整pair时间序列。",
              "分类：ATTACK_CONDITION_STRICT_BUT_LEARNABLE。依据是seed1正式200局41%win，而非单局反事实。",
              "详见combat_contract.md、trajectory_case/trajectory_case_report.md、gate_sensitivity.csv。",
              "假设矩阵：VERIFIED仅表示直接可验证事实；阈值、Blue策略、reward或优化的因果根因不能从单局冻结policy证明。", "",
              "|Hypothesis|Evidence|Status|Severity|", "|---|---|---|---|"]
    hypotheses = [("H1 distance restrictive", "case distance sensitivity; missing200 raw distribution", "INCONCLUSIVE", "medium"),
                  ("H2 ATA restrictive", "prior P2 draw withinrange but gate0; inspect conditional ATA distributions", "PARTIALLY_SUPPORTED", "high"),
                  ("H3 AA restrictive", "case conditional AA distribution; not isolated causal proof", "INCONCLUSIVE", "medium"),
                  ("H4 hold primary", "prior P2 draw maxstreak0:58/67,68/69,22/22; hold2 cannot help these", "REFUTED", "low"),
                  ("H5 combined sparse", "formal alive-pair gate sparse yet seed1 learned kills3.035", "PARTIALLY_SUPPORTED", "high"),
                  ("H6 horizon primary", "prior P2 draw has mean41-51 transitions without gate; singlecase horizon100 cannot prove population causality", "INCONCLUSIVE", "medium"),
                  ("H7 sensing primary", "case direct/datalink; population P2 visibility missing", "INCONCLUSIVE", "medium"),
                  ("H8 equal speed tail-lock", "equal150-300 limits verified; realized speed/closure only case", "INCONCLUSIVE", "medium"),
                  ("H9 Blue pathological geometry", "case Blue heading/target cooccurrence not causal", "INCONCLUSIVE", "medium"),
                  ("H10 control insufficient", "UAV/Blue controls equal; successfulseed disproves universal infeasibility", "INCONCLUSIVE", "medium"),
                  ("H11 boundary contributes", "prior200 boundary losses91/77/52; does not prove re-engagement cause", "PARTIALLY_SUPPORTED", "medium"),
                  ("H12 initial headon longterm failure", "same starting contract admits successful seed1", "INCONCLUSIVE", "medium"),
                  ("H13 reward alignment insufficient", "same predicate but postdeath surviving-target process; selector mismatch; no causal isolation", "PARTIALLY_SUPPORTED", "medium"),
                  ("H14 optimization/local optimum primary", "same contract highly seed-sensitive; causal priority not established", "PARTIALLY_SUPPORTED", "high")]
    for h, evidence, status, severity in hypotheses: report.append(f"|{h}|{evidence}|{status}|{severity}|")
    (args.output/"research_report.md").write_text("\n".join(report), encoding="utf-8")
    finish_saved_analysis(args.output, args.prior_audit)
    print(f"Audit complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
