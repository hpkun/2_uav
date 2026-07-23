"""按固定对手优先协议训练，并在论文门槛后才进入交替冻结。"""
import argparse,csv,json,random,time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch,yaml
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import MAPPOTrainer,evaluate_actor


def args_parser():
    p=argparse.ArgumentParser(); p.add_argument("--env-config",default="configs/homogeneous_1v1.yaml"); p.add_argument("--train-config",default="configs/mappo_1v1.yaml"); p.add_argument("--smoke",action="store_true"); p.add_argument("--total-env-steps",type=int); p.add_argument("--num-envs",type=int); p.add_argument("--seed",type=int); p.add_argument("--device"); p.add_argument("--resume"); return p.parse_args()


def load_config(args):
    with open(args.train_config,encoding="utf-8") as f: c=yaml.safe_load(f)
    if args.smoke:
        c["training"].update(total_env_steps=4096,num_envs=2,rollout_steps=64,fixed_tail_chase_env_steps=2048,fixed_training_env_steps=4096,competitive_training_env_steps=4096,alternating_block_env_steps=1024,ppo_epochs=2,minibatch_size=128)
        c["evaluation"]["episodes"]=6; c["experiment"]["output_dir"]="outputs/mappo_smoke"
    for value,section,key in ((args.total_env_steps,"training","total_env_steps"),(args.num_envs,"training","num_envs"),(args.seed,"experiment","seed"),(args.device,"experiment","device")):
        if value is not None: c[section][key]=value
    return c


def diagnostic_row(trainer,episodes,metrics,evaluations,phase=None):
    nan=float("nan"); row={"update":trainer.update_count,"env_steps":trainer.env_steps,"phase":phase or trainer.phase(),**metrics}; count=len(episodes)
    for side,index in (("red",0),("blue",1)): row[f"{side}_mean_episode_return"]=float(np.mean([e["returns"][index] for e in episodes])) if count else nan
    row["mean_episode_length"]=float(np.mean([e["length"] for e in episodes])) if count else nan
    for outcome in ("red","blue","draw"): row[f"{outcome}_win_rate" if outcome!="draw" else "draw_rate"]=sum(e["outcome"]==outcome for e in episodes)/count if count else nan
    for reason in ("collision","max_steps","red_kill","blue_kill"): row[f"{reason}_rate"]=sum(e["reason"]==reason for e in episodes)/count if count else nan
    row["boundary_rate"]=sum(e["reason"] in {"boundary","xy_boundary","altitude_boundary"} for e in episodes)/count if count else nan
    funnels=[e["funnel"] for e in episodes]
    for source,target in (("ever_within_4000m","within_4000_rate"),("ever_within_attack_distance","attack_distance_entry_rate"),("ever_satisfy_ata","ata_gate_rate"),("ever_satisfy_aa","aa_gate_rate"),("ever_satisfy_attack_envelope","attack_envelope_entry_rate")):
        row[target]=sum(f[source] for f in funnels)/count if count else nan
    envelope=sum(f["ever_satisfy_attack_envelope"] for f in funnels); row["attack_to_kill_conversion_rate"]=sum(f["kill"] for f in funnels)/envelope if envelope else (0.0 if count else nan)
    for key in ("reward_terminal","reward_boundary","reward_guide","reward_position","reward_threat","reward_total"): row[f"mean_{key}"]=float(np.mean([e["reward_components"].get(key,0) for e in episodes])) if count else nan
    diagnostics=trainer.last_control_diagnostics
    for action in ("yaw","pitch","speed"):
        values=np.asarray([d[f"action_{action}"] for d in diagnostics]); row[f"action_{action}_mean"]=float(values.mean()); row[f"action_{action}_std"]=float(values.std()); row[f"action_{action}_min"]=float(values.min()); row[f"action_{action}_max"]=float(values.max())
    for key in ("yaw_rate","pitch_rate","acceleration","nx","nz","phi"): row[f"{key}_saturation_rate"]=float(np.mean([d[f"{key}_saturated"] for d in diagnostics]))
    for key in ("delta_yaw","delta_pitch","delta_speed","yaw_error","pitch_error","speed_error","unclipped_yaw_rate","unclipped_pitch_rate","unclipped_acceleration","clipped_yaw_rate","clipped_pitch_rate","clipped_acceleration","nx","nz","phi"):
        row[f"mean_{key}"]=float(np.mean([d[key] for d in diagnostics]))
    for opponent in ("zero","pursuit"):
        overall=evaluations[opponent]["overall"]; row[f"eval_red_{opponent}_win_rate"]=overall["win_rate"]; row[f"eval_red_{opponent}_mean_return"]=overall["mean_return"]; row[f"eval_red_{opponent}_attack_envelope_entry_rate"]=overall["attack_envelope_entry_rate"]
    return row


def plots(rows,path,tail_end):
    x=[r["env_steps"] for r in rows]; fig,axes=plt.subplots(3,2,figsize=(12,12)); series=(("red_mean_episode_return",),("red_win_rate","blue_win_rate","draw_rate"),("red_policy_loss","value_loss"),("mean_reward_guide","mean_reward_position","mean_reward_threat"),("attack_distance_entry_rate","attack_envelope_entry_rate"),("yaw_rate_saturation_rate","pitch_rate_saturation_rate","acceleration_saturation_rate"))
    for axis,keys in zip(axes.flat,series):
        for key in keys: axis.plot(x,[r.get(key,np.nan) for r in rows],label=key)
        axis.axvline(tail_end,linestyle="--",color="k"); axis.grid(True); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def diagnostic_plots(rows,output):
    x=[r["env_steps"] for r in rows]
    for filename,keys,title in (("attack_funnel.png",("within_4000_rate","attack_distance_entry_rate","ata_gate_rate","aa_gate_rate","attack_envelope_entry_rate","attack_to_kill_conversion_rate"),"Attack funnel (project diagnostics)"),("control_saturation.png",("yaw_rate_saturation_rate","pitch_rate_saturation_rate","acceleration_saturation_rate","nx_saturation_rate","nz_saturation_rate","phi_saturation_rate"),"Controller saturation rates")):
        fig,axis=plt.subplots(figsize=(10,5))
        for key in keys: axis.plot(x,[r.get(key,np.nan) for r in rows],label=key)
        axis.grid(True); axis.legend(fontsize=7); axis.set_title(title); axis.set_xlabel("environment steps"); fig.tight_layout(); fig.savefig(output/filename,dpi=150); plt.close(fig)


def save_eval(result,path): path.write_text(json.dumps(result,indent=2),encoding="utf-8")


def evaluate_checkpoint(path,env_config,device,opponent,episodes=300):
    checkpoint=torch.load(path,map_location=device,weights_only=False); config=checkpoint["config"]
    actor=GaussianActor(14,3,config["network"]["hidden_dim"],config["network"]["log_std_init"]).to(device); actor.load_state_dict(checkpoint["red_actor"])
    return evaluate_actor(actor,env_config,episodes,device,opponent,"red","all",checkpoint["seed"]+200000)


def main():
    args=args_parser(); c=load_config(args); seed=c["experiment"]["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); trainer=MAPPOTrainer(args.env_config,c)
    if args.resume: trainer.load_checkpoint(args.resume)
    output=Path(c["experiment"]["output_dir"]); checkpoints=output/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True); trainer.save_checkpoint(checkpoints/"initial.pt"); torch.cuda.reset_peak_memory_stats() if trainer.device.type=="cuda" else None
    print(f"device={trainer.device} gpu={torch.cuda.get_device_name(0) if trainer.device.type=='cuda' else None} torch={torch.__version__} cuda={torch.version.cuda}"); rows=[]; best=(-1.0,-1.0,-np.inf); start=time.perf_counter(); target=int(c["training"]["total_env_steps"])
    evaluations={o:evaluate_actor(trainer.red_actor,args.env_config,c["evaluation"]["episodes"],trainer.device,o,"red","all",seed+100000) for o in ("zero","pursuit")}
    trainer.save_checkpoint(checkpoints/"fixed_best.pt")
    while trainer.env_steps<target:
        stage_before=trainer.phase(); next_boundary=min(target,c["training"]["fixed_tail_chase_env_steps"] if trainer.env_steps<c["training"]["fixed_tail_chase_env_steps"] else target)
        completed=trainer.collect_rollout(next_boundary-trainer.env_steps); metrics=trainer.update("red")
        if trainer.phase()!=stage_before and trainer.env_steps<target: trainer.reset_environments()
        if trainer.update_count%c["training"]["eval_interval_updates"]==0 or trainer.env_steps>=target:
            evaluations={o:evaluate_actor(trainer.red_actor,args.env_config,c["evaluation"]["episodes"],trainer.device,o,"red","all",seed+100000) for o in ("zero","pursuit")}; pursuit=evaluations["pursuit"]["overall"]; candidate=(pursuit["win_rate"],pursuit["attack_envelope_entry_rate"],pursuit["mean_return"])
            if candidate>best: best=candidate; trainer.save_checkpoint(checkpoints/"fixed_best.pt")
        row=diagnostic_row(trainer,completed,metrics,evaluations,stage_before); rows.append(row); trainer.save_checkpoint(checkpoints/"latest.pt"); print(f"update={trainer.update_count} steps={trainer.env_steps} stage={stage_before} red_return={row['red_mean_episode_return']} win={row['red_win_rate']} loss={row['blue_win_rate']} draw={row['draw_rate']} envelope={row['attack_envelope_entry_rate']} sat={row['yaw_rate_saturation_rate']:.3f}/{row['pitch_rate_saturation_rate']:.3f}/{row['acceleration_saturation_rate']:.3f}")
    trainer.save_checkpoint(checkpoints/"fixed_final.pt"); fixed_elapsed=time.perf_counter()-start
    fixed_evaluations={}
    protocol_eval_episodes=6 if args.smoke else 300
    for checkpoint_name in ("initial","fixed_best","fixed_final"):
        fixed_evaluations[checkpoint_name]={}
        for opponent in ("zero","pursuit"):
            result=evaluate_checkpoint(checkpoints/f"{checkpoint_name}.pt",args.env_config,trainer.device,opponent,protocol_eval_episodes)
            fixed_evaluations[checkpoint_name][opponent]=result
            save_eval(result,output/f"evaluation_{checkpoint_name}_red_vs_{opponent}_all_seedset0.json")
    gate=fixed_evaluations["fixed_best"]["pursuit"]; gate_pass=gate["overall"]["win_rate"]>=c["training"]["fixed_opponent_gate_win_rate"]; trainer.fixed_gate_result=gate_pass; trainer.save_checkpoint(checkpoints/"fixed_final.pt")
    if gate_pass:
        trainer.copy_red_to_blue_for_competition(); trainer.reset_environments(); trainer.save_checkpoint(checkpoints/"competitive_initial.pt"); trainer.save_checkpoint(checkpoints/"competitive_best.pt"); competitive_target=trainer.env_steps+c["training"]["competitive_training_env_steps"]; competitive_best=(-1.0,-1.0,-np.inf)
        while trainer.env_steps<competitive_target:
            active_before=trainer.active_side(); block_steps=c["training"]["alternating_block_env_steps"]; offset=trainer.env_steps-c["training"]["fixed_training_env_steps"]; block_boundary=c["training"]["fixed_training_env_steps"]+(offset//block_steps+1)*block_steps; next_boundary=min(competitive_target,block_boundary)
            completed=trainer.collect_rollout(next_boundary-trainer.env_steps); metrics=trainer.update(active_before)
            if trainer.update_count%c["training"]["eval_interval_updates"]==0 or trainer.env_steps>=competitive_target:
                evaluations={o:evaluate_actor(trainer.red_actor,args.env_config,c["evaluation"]["episodes"],trainer.device,o,"red","all",seed+400000) for o in ("zero","pursuit")}; pursuit=evaluations["pursuit"]["overall"]; candidate=(pursuit["win_rate"],pursuit["attack_envelope_entry_rate"],pursuit["mean_return"])
                if candidate>competitive_best: competitive_best=candidate; trainer.save_checkpoint(checkpoints/"competitive_best.pt")
            rows.append(diagnostic_row(trainer,completed,metrics,evaluations))
        trainer.save_checkpoint(checkpoints/"competitive_final.pt")
        for checkpoint_name in ("competitive_best","competitive_final"):
            for opponent in ("zero","pursuit"):
                result=evaluate_checkpoint(checkpoints/f"{checkpoint_name}.pt",args.env_config,trainer.device,opponent,protocol_eval_episodes)
                save_eval(result,output/f"evaluation_{checkpoint_name}_red_vs_{opponent}_all_seedset0.json")
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with (output/"training_metrics.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    plots(rows,output/"training_curves.png",c["training"]["fixed_tail_chase_env_steps"]); diagnostic_plots(rows,output)
    summary={"device":str(trainer.device),"gpu":torch.cuda.get_device_name(0) if trainer.device.type=="cuda" else None,"torch":torch.__version__,"torch_cuda":torch.version.cuda,"fixed_environment_steps":target,"actual_environment_steps":trainer.env_steps,"fixed_elapsed_seconds":fixed_elapsed,"total_elapsed_seconds":time.perf_counter()-start,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated() if trainer.device.type=="cuda" else 0,"fixed_gate_win_rate":gate["overall"]["win_rate"],"fixed_gate_pass":gate_pass,"competitive_executed":gate_pass}
    (output/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
