"""Evaluate zero-action and pure-pursuit rule baselines without learning."""
import argparse,json
from pathlib import Path
import numpy as np
from uav_combat.environment import HomogeneousAirCombatEnv
from uav_combat.mappo.trainer import finish_funnel,new_funnel,summarize_records,update_funnel
from uav_combat.rule_policy import PurePursuitPolicy


def rule_action(mode,own,target,pursuit):
    return np.zeros(3,dtype=float) if mode=="zero" else pursuit.action(own,target)


def run_matchup(env_config,red_mode,blue_mode,episodes_per_scenario=100,seed=700000):
    templates=("tail_chase","offset_head_on","crossing"); records=[]
    for scenario_index,scenario in enumerate(templates):
        for episode in range(episodes_per_scenario):
            env=HomogeneousAirCombatEnv(env_config); env.reset(seed+scenario_index*10000+episode,scenario,"red" if scenario=="tail_chase" else None); cfg=env.config["action"]; pursuit=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]); total=0.0; funnel=new_funnel()
            while True:
                red,blue=env.aircraft; actions={"red_0":rule_action(red_mode,red,blue,pursuit),"blue_0":rule_action(blue_mode,blue,red,pursuit)}; _,rewards,terminated,truncated,info=env.step(actions); total+=rewards["red_0"]; update_funnel(funnel,info["geometries"]["red_0"],env.config["combat"],info["attacks"]["red_0"])
                if terminated or truncated: break
            result="win" if info["outcome"]=="red" else ("draw" if info["outcome"]=="draw" else "loss"); finish_funnel(funnel,info["termination_reason"],result); records.append({"scenario":scenario,"result":result,"reason":info["termination_reason"],"return":total,"length":info["step_count"],"funnel":funnel})
    return {"red_policy":red_mode,"blue_policy":blue_mode,"seed_set":"seedset0","overall":summarize_records(records),"by_scenario":{scenario:summarize_records([row for row in records if row["scenario"]==scenario]) for scenario in templates}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--env-config",default="configs/homogeneous_1v1.yaml"); parser.add_argument("--episodes-per-scenario",type=int,default=100); parser.add_argument("--output",default="outputs/baselines/rule_baselines.json"); args=parser.parse_args(); combinations=(("pursuit","zero"),("zero","pursuit"),("pursuit","pursuit"),("zero","zero")); results={f"red_{red}_vs_blue_{blue}":run_matchup(args.env_config,red,blue,args.episodes_per_scenario) for red,blue in combinations}; output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"{'matchup':34} {'W':>4} {'L':>4} {'D':>4} {'win%':>7} {'full%':>7} {'kill%':>7}")
    for name,result in results.items():
        overall=result["overall"]; print(f"{name:34} {overall['wins']:4d} {overall['losses']:4d} {overall['draws']:4d} {100*overall['win_rate']:7.2f} {100*overall['full_attack_envelope_entry_rate']:7.2f} {100*overall['kill_rate']:7.2f}")
    print(f"saved: {output}")


if __name__=="__main__": main()
