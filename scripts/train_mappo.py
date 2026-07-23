"""Run the project-defined three-stage fixed-opponent training protocol."""
import argparse,csv,json,random,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch,yaml
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import MAPPOTrainer,evaluate_actor


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument("--env-config",default="configs/homogeneous_1v1.yaml"); parser.add_argument("--train-config",default="configs/mappo_1v1.yaml"); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--total-env-steps",type=int); parser.add_argument("--num-envs",type=int); parser.add_argument("--seed",type=int); parser.add_argument("--device"); parser.add_argument("--resume"); return parser.parse_args()


def load_config(args):
    with open(args.train_config,encoding="utf-8") as file: config=yaml.safe_load(file)
    if config["training"].get("training_mode")!="paper_staged": raise ValueError("train_mappo.py only supports training_mode=paper_staged")
    if args.smoke:
        config["training"].update(total_env_steps=8192,num_envs=2,rollout_steps=64,straight_tail_chase_env_steps=2048,pursuit_tail_chase_env_steps=4096,fixed_training_env_steps=8192,ppo_epochs=2,minibatch_size=128)
        config["evaluation"]["episodes"]=6; config["experiment"]["output_dir"]="outputs/mappo_smoke"
    for value,section,key in ((args.total_env_steps,"training","total_env_steps"),(args.num_envs,"training","num_envs"),(args.seed,"experiment","seed"),(args.device,"experiment","device")):
        if value is not None: config[section][key]=value
    return config


def phase_spec(phase):
    return {"straight_tail_chase":("zero","tail_chase","straight_best.pt","straight_final.pt"),"pursuit_tail_chase":("pursuit","tail_chase","pursuit_tail_best.pt","pursuit_tail_final.pt"),"pursuit_all_scenarios":("pursuit","all","fixed_best.pt","fixed_final.pt")}[phase]


def checkpoint_evaluation(path,env_config,device,opponent,scenario,episodes,seed_offset=200000):
    checkpoint=torch.load(path,map_location=device,weights_only=False); config=checkpoint["config"]; actor=GaussianActor(14,3,config["network"]["hidden_dim"],config["network"]["log_std_init"]).to(device); actor.load_state_dict(checkpoint["red_actor"])
    return evaluate_actor(actor,env_config,episodes,device,opponent,"red",scenario,checkpoint["seed"]+seed_offset)


def diagnostic_row(trainer,episodes,metrics,phase,opponent,evaluation):
    nan=float("nan"); count=len(episodes); row={"update":trainer.update_count,"env_steps":trainer.env_steps,"phase":phase,"fixed_opponent_mode":opponent,**metrics}
    row["red_mean_episode_return"]=float(np.mean([episode["returns"][0] for episode in episodes])) if count else nan; row["blue_mean_episode_return"]=float(np.mean([episode["returns"][1] for episode in episodes])) if count else nan; row["mean_episode_length"]=float(np.mean([episode["length"] for episode in episodes])) if count else nan
    for outcome in ("red","blue","draw"): row[f"{outcome}_win_rate" if outcome!="draw" else "draw_rate"]=sum(episode["outcome"]==outcome for episode in episodes)/count if count else nan
    for reason in ("collision","max_steps","red_kill","blue_kill"): row[f"{reason}_rate"]=sum(episode["reason"]==reason for episode in episodes)/count if count else nan
    row["boundary_rate"]=sum(episode["reason"] in {"boundary","xy_boundary","altitude_boundary"} for episode in episodes)/count if count else nan; funnels=[episode["funnel"] for episode in episodes]
    for source,target in (("ever_within_4000m","within_4000_rate"),("ever_within_attack_distance","attack_distance_entry_rate"),("ever_satisfy_ata","ata_gate_rate"),("ever_satisfy_aa","aa_gate_rate"),("ever_distance_and_ata","distance_and_ata_entry_rate"),("ever_distance_and_aa","distance_and_aa_entry_rate"),("ever_ata_and_aa","ata_and_aa_entry_rate"),("ever_full_attack_envelope","full_attack_envelope_entry_rate")):
        row[target]=sum(funnel[source] for funnel in funnels)/count if count else nan
    for source,target in (("minimum_distance_violation","mean_minimum_distance_violation"),("minimum_ata_violation","mean_minimum_ata_violation"),("minimum_aa_violation","mean_minimum_aa_violation"),("minimum_combined_violation","mean_minimum_combined_violation")):
        row[target]=float(np.mean([funnel[source] for funnel in funnels])) if count else nan
    envelopes=sum(funnel["ever_full_attack_envelope"] for funnel in funnels); row["attack_to_kill_conversion_rate"]=sum(funnel["kill"] for funnel in funnels)/envelopes if envelopes else (0.0 if count else nan)
    for key in ("reward_terminal","reward_boundary","reward_guide","reward_position","reward_threat","reward_total"): row[f"mean_{key}"]=float(np.mean([episode["reward_components"].get(key,0.0) for episode in episodes])) if count else nan
    diagnostics=trainer.last_control_diagnostics
    for action in ("yaw","pitch","speed"):
        values=np.asarray([entry[f"action_{action}"] for entry in diagnostics]); row[f"action_{action}_mean"]=float(values.mean()); row[f"action_{action}_std"]=float(values.std()); row[f"action_{action}_min"]=float(values.min()); row[f"action_{action}_max"]=float(values.max())
    for key in ("yaw_rate","pitch_rate","acceleration","nx","nz","phi"): row[f"{key}_saturation_rate"]=float(np.mean([entry[f"{key}_saturated"] for entry in diagnostics]))
    for label,key in (("acceleration","acceleration_tracking_absolute_error"),("pitch_rate","pitch_rate_tracking_absolute_error"),("yaw_rate","yaw_rate_tracking_absolute_error")):
        values=np.asarray([entry[key] for entry in diagnostics]); row[f"{label}_tracking_mae"]=float(values.mean()); row[f"{label}_tracking_median"]=float(np.median(values)); row[f"{label}_tracking_p95"]=float(np.percentile(values,95)); row[f"{label}_tracking_max"]=float(values.max())
    for key in ("delta_yaw","delta_pitch","delta_speed","yaw_error","pitch_error","speed_error","unclipped_yaw_rate","unclipped_pitch_rate","unclipped_acceleration","clipped_yaw_rate","clipped_pitch_rate","clipped_acceleration","actual_yaw_rate","actual_pitch_rate","actual_acceleration","nx","nz","phi"): row[f"mean_{key}"]=float(np.mean([entry[key] for entry in diagnostics]))
    overall=evaluation["overall"]; row["eval_win_rate"]=overall["win_rate"]; row["eval_mean_return"]=overall["mean_return"]; row["eval_full_attack_envelope_entry_rate"]=overall["full_attack_envelope_entry_rate"]
    return row


def write_metrics(rows,path):
    with path.open("w",newline="",encoding="utf-8") as file: writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_series(rows,output,boundaries):
    x=[row["env_steps"] for row in rows]
    groups={"training_curves.png":(("red_mean_episode_return",),("red_win_rate","blue_win_rate","draw_rate"),("red_policy_loss","value_loss"),("mean_reward_guide","mean_reward_position","mean_reward_threat")),"attack_funnel.png":(("attack_distance_entry_rate","ata_gate_rate","aa_gate_rate"),("distance_and_ata_entry_rate","distance_and_aa_entry_rate","ata_and_aa_entry_rate","full_attack_envelope_entry_rate")),"control_saturation.png":(("yaw_rate_saturation_rate","pitch_rate_saturation_rate","acceleration_saturation_rate"),("nx_saturation_rate","nz_saturation_rate","phi_saturation_rate")),"control_tracking_error.png":(("acceleration_tracking_mae","pitch_rate_tracking_mae","yaw_rate_tracking_mae"),)}
    for filename,panels in groups.items():
        fig,axes=plt.subplots(len(panels),1,figsize=(11,4*len(panels)),squeeze=False)
        for axis,keys in zip(axes[:,0],panels):
            for key in keys: axis.plot(x,[row.get(key,np.nan) for row in rows],label=key)
            for boundary in boundaries: axis.axvline(boundary,color="k",linestyle="--",alpha=.6)
            axis.grid(True); axis.legend(fontsize=7); axis.set_xlabel("environment steps")
        fig.tight_layout(); fig.savefig(output/filename,dpi=150); plt.close(fig)


def main():
    args=parse_args(); config=load_config(args); training=config["training"]; seed=config["experiment"]["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); trainer=MAPPOTrainer(args.env_config,config)
    if args.resume: trainer.load_checkpoint(args.resume)
    output=Path(config["experiment"]["output_dir"]); checkpoints=output/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True); trainer.save_checkpoint(checkpoints/"initial.pt"); torch.cuda.reset_peak_memory_stats() if trainer.device.type=="cuda" else None
    print(f"device={trainer.device} gpu={torch.cuda.get_device_name(0) if trainer.device.type=='cuda' else None} torch={torch.__version__} cuda={torch.version.cuda}",flush=True)
    boundaries={"straight_tail_chase":training["straight_tail_chase_env_steps"],"pursuit_tail_chase":training["pursuit_tail_chase_env_steps"],"pursuit_all_scenarios":training["fixed_training_env_steps"]}; rows=[]; best_scores={phase:(-1.0,-1.0,-np.inf) for phase in boundaries}; stage_starts={"straight_tail_chase":time.perf_counter()}; stage_summaries={}; start=time.perf_counter(); current="straight_tail_chase"; trainer.save_checkpoint(checkpoints/phase_spec(current)[2]); evaluation=evaluate_actor(trainer.red_actor,args.env_config,config["evaluation"]["episodes"],trainer.device,"zero","red","tail_chase",seed+100000)
    while trainer.env_steps<training["fixed_training_env_steps"]:
        phase=trainer.phase(); opponent,scenario,best_name,final_name=phase_spec(phase); boundary=boundaries[phase]; completed=trainer.collect_rollout(boundary-trainer.env_steps); metrics=trainer.update("red")
        if trainer.update_count%training["eval_interval_updates"]==0 or trainer.env_steps==boundary:
            evaluation=evaluate_actor(trainer.red_actor,args.env_config,config["evaluation"]["episodes"],trainer.device,opponent,"red",scenario,seed+100000); overall=evaluation["overall"]; candidate=(overall["win_rate"],overall["full_attack_envelope_entry_rate"],overall["mean_return"])
            if candidate>best_scores[phase]: best_scores[phase]=candidate; trainer.save_checkpoint(checkpoints/best_name)
        row=diagnostic_row(trainer,completed,metrics,phase,opponent,evaluation); rows.append(row); trainer.save_checkpoint(checkpoints/"latest.pt"); write_metrics(rows,output/"training_metrics.csv")
        print(f"update={trainer.update_count} steps={trainer.env_steps} phase={phase} opponent={opponent} return={row['red_mean_episode_return']} win={row['red_win_rate']} full={row['full_attack_envelope_entry_rate']} yaw_track={row['yaw_rate_tracking_mae']:.4f}",flush=True)
        if trainer.env_steps==boundary:
            trainer.save_checkpoint(checkpoints/final_name); now=time.perf_counter(); stage_summaries[phase]={"environment_steps":boundary-(0 if phase=="straight_tail_chase" else (training["straight_tail_chase_env_steps"] if phase=="pursuit_tail_chase" else training["pursuit_tail_chase_env_steps"])),"updates":sum(row["phase"]==phase for row in rows),"elapsed_seconds":now-stage_starts[phase]}
            if trainer.env_steps<training["fixed_training_env_steps"]:
                trainer.reset_environments(); next_phase=trainer.phase(); stage_starts[next_phase]=time.perf_counter(); trainer.save_checkpoint(checkpoints/phase_spec(next_phase)[2])
    fixed_elapsed=time.perf_counter()-start; formal_episodes=3 if args.smoke else None; checkpoint_names=("initial","straight_best","straight_final","pursuit_tail_best","pursuit_tail_final","fixed_best","fixed_final"); formal_results={}
    for checkpoint_name in checkpoint_names:
        formal_results[checkpoint_name]={}
        for opponent,scenario,episodes in (("zero","tail_chase",formal_episodes or 100),("pursuit","tail_chase",formal_episodes or 100),("zero","all",formal_episodes or 300),("pursuit","all",formal_episodes or 300)):
            result=checkpoint_evaluation(checkpoints/f"{checkpoint_name}.pt",args.env_config,trainer.device,opponent,scenario,episodes); key=f"{opponent}_{scenario}"; formal_results[checkpoint_name][key]=result; (output/f"evaluation_{checkpoint_name}_red_vs_{opponent}_{scenario}_seedset0.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    gate=formal_results["fixed_best"]["pursuit_all"]["overall"]; trainer.fixed_gate_result=gate["win_rate"]>=training["fixed_opponent_gate_win_rate"]; trainer.save_checkpoint(checkpoints/"fixed_final.pt"); plot_series(rows,output,(training["straight_tail_chase_env_steps"],training["pursuit_tail_chase_env_steps"]))
    summary={"device":str(trainer.device),"gpu":torch.cuda.get_device_name(0) if trainer.device.type=="cuda" else None,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"actual_environment_steps":trainer.env_steps,"fixed_elapsed_seconds":fixed_elapsed,"total_elapsed_seconds":time.perf_counter()-start,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated() if trainer.device.type=="cuda" else 0,"stages":stage_summaries,"fixed_gate_win_rate":gate["win_rate"],"fixed_gate_pass":trainer.fixed_gate_result,"competitive_executed":False}; (output/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__": main()
