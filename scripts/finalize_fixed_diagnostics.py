"""Finalize a failed fixed-opponent gate and collect reproducible diagnostics."""
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from uav_combat.environment import HomogeneousAirCombatEnv
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import resolve_device
from uav_combat.rule_policy import PurePursuitPolicy


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",default="outputs/mappo/checkpoints/fixed_best.pt"); parser.add_argument("--evaluation",default="outputs/mappo/evaluation_fixed_best_red_vs_pursuit_all_seedset0.json"); parser.add_argument("--env-config",default="configs/homogeneous_1v1.yaml"); parser.add_argument("--episodes",type=int,default=300); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    device=resolve_device(args.device); evaluation=json.loads(Path(args.evaluation).read_text(encoding="utf-8")); gate=evaluation["overall"]["win_rate"]>=0.70
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False); config=checkpoint["config"]; actor=GaussianActor(14,3,config["network"]["hidden_dim"],config["network"]["log_std_init"]).to(device); actor.load_state_dict(checkpoint["red_actor"]); actor.eval()
    action_values={name:[] for name in ("yaw","pitch","speed")}; control_values={key:[] for key in ("delta_yaw","delta_pitch","delta_speed","yaw_error","pitch_error","speed_error","unclipped_yaw_rate","unclipped_pitch_rate","unclipped_acceleration","clipped_yaw_rate","clipped_pitch_rate","clipped_acceleration","actual_yaw_rate","actual_pitch_rate","actual_acceleration","nx","nz","phi")}; tracking={label:[] for label in ("acceleration","pitch_rate","yaw_rate")}; saturation={key:[] for key in ("yaw_rate","pitch_rate","acceleration","nx","nz","phi")}; rewards={key:0.0 for key in ("reward_terminal","reward_boundary","reward_guide","reward_position","reward_threat","reward_total")}; reasons={}; steps=0; templates=("tail_chase","offset_head_on","crossing")
    for episode in range(args.episodes):
        env=HomogeneousAirCombatEnv(args.env_config); observations,_=env.reset(checkpoint["seed"]+200000+episode,templates[episode%3],"red" if templates[episode%3]=="tail_chase" else None); cfg=env.config["action"]; pursuit=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"])
        while True:
            with torch.no_grad(): red_action=actor.deterministic_action(torch.as_tensor(observations["red_0"],dtype=torch.float32,device=device)[None]).squeeze().cpu().numpy()
            red,blue=env.aircraft; blue_action=pursuit.action(blue,red); observations,_,terminated,truncated,info=env.step({"red_0":red_action,"blue_0":blue_action}); steps+=1
            for index,name in enumerate(("yaw","pitch","speed")): action_values[name].append(float(red_action[index]))
            diagnostics=info["control_diagnostics"]["red_0"]
            for key in control_values: control_values[key].append(float(diagnostics[key]))
            for label in tracking: tracking[label].append(float(diagnostics[f"{label}_tracking_absolute_error"]))
            for key in saturation: saturation[key].append(bool(diagnostics[f"{key}_saturated"]))
            for key in rewards: rewards[key]+=float(info["reward_terms"]["red_0"][key])
            if terminated or truncated: reasons[info["termination_reason"]]=reasons.get(info["termination_reason"],0)+1; break
    result={"checkpoint":"fixed_best","actor":"red","opponent":"pursuit","scenario":"all","seed_set":"seedset0","episodes":args.episodes,"environment_steps_observed":steps,"gate_win_rate":evaluation["overall"]["win_rate"],"gate_pass":gate,"reward_component_mean_per_step":{key:value/steps for key,value in rewards.items()},"actions":{name:{"mean":float(np.mean(values)),"std":float(np.std(values)),"min":float(np.min(values)),"max":float(np.max(values))} for name,values in action_values.items()},"mean_control_diagnostics":{key:float(np.mean(values)) for key,values in control_values.items()},"control_saturation_rates":{key:float(np.mean(values)) for key,values in saturation.items()},"tracking_absolute_errors":{label:{"mae":float(np.mean(values)),"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":float(np.max(values))} for label,values in tracking.items()},"attack_funnel":{key:evaluation["overall"][key] for key in ("within_4000_rate","attack_distance_entry_rate","ata_gate_rate","aa_gate_rate","distance_and_ata_entry_rate","distance_and_aa_entry_rate","ata_and_aa_entry_rate","full_attack_envelope_entry_rate","attack_to_kill_conversion_rate","kill_rate","mean_minimum_distance_violation","mean_minimum_ata_violation","mean_minimum_aa_violation","mean_minimum_combined_violation")},"termination_rates":{key:evaluation["overall"][key] for key in ("win_rate","loss_rate","draw_rate","boundary_rate","max_steps_rate","collision_rate")},"termination_reason_counts":reasons}
    output=Path(config["experiment"]["output_dir"]); output.mkdir(parents=True,exist_ok=True); (output/"fixed_best_diagnostics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    flat={"episodes":args.episodes,"environment_steps_observed":steps,"gate_win_rate":result["gate_win_rate"],"gate_pass":gate}
    for section in ("reward_component_mean_per_step","mean_control_diagnostics","control_saturation_rates","attack_funnel","termination_rates"): flat.update({f"{section}.{key}":value for key,value in result[section].items()})
    for label,stats in result["tracking_absolute_errors"].items(): flat.update({f"tracking_absolute_errors.{label}.{key}":value for key,value in stats.items()})
    for name,stats in result["actions"].items(): flat.update({f"action_{name}_{key}":value for key,value in stats.items()})
    with (output/"fixed_best_diagnostics.csv").open("w",newline="",encoding="utf-8") as file: writer=csv.DictWriter(file,fieldnames=flat); writer.writeheader(); writer.writerow(flat)
    for filename,title,data in (("fixed_best_attack_funnel.png","Fixed-best attack funnel (project diagnostics)",result["attack_funnel"]),("fixed_best_control_saturation.png","Fixed-best controller saturation rates",result["control_saturation_rates"])):
        fig,axis=plt.subplots(figsize=(10,5)); axis.bar(range(len(data)),list(data.values())); axis.set_xticks(range(len(data)),list(data),rotation=30,ha="right"); axis.set_ylim(0,1); axis.set_title(title); axis.grid(True,axis="y"); fig.tight_layout(); fig.savefig(output/filename,dpi=150); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
