"""Train the color-independent alternating-freeze v6 baseline."""
import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from uav_combat.mappo.trainer import MAPPOTrainer, POLICIES, SCENARIOS, competitive_score, evaluate_competitive_match, summarize_competitive_records


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-config",default="configs/homogeneous_1v1.yaml")
    parser.add_argument("--train-config",default="configs/mappo_1v1.yaml")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--total-env-steps",type=int); parser.add_argument("--num-envs",type=int)
    parser.add_argument("--seed",type=int); parser.add_argument("--device"); parser.add_argument("--output-dir"); parser.add_argument("--resume")
    return parser.parse_args()


def load_config(args):
    with open(args.train_config,encoding="utf-8") as f: config=yaml.safe_load(f)
    if config["training"].get("training_mode")!="alternating_self_play": raise ValueError("train_mappo.py requires alternating_self_play")
    config["experiment"]["output_dir"]=config["experiment"].get("output_dir","outputs/mappo_v6")
    config["training"].setdefault("opponent_history_latest_probability",.7)
    if args.smoke:
        config["training"].update(total_env_steps=8192,num_envs=2,rollout_steps=64,alternating_block_env_steps=2048,ppo_epochs=2,minibatch_size=128)
        config["evaluation"]["episodes"]=12; config["experiment"]["output_dir"]="outputs/mappo_v6_smoke"
    for value,section,key in ((args.total_env_steps,"training","total_env_steps"),(args.num_envs,"training","num_envs"),(args.seed,"experiment","seed"),(args.device,"experiment","device"),(args.output_dir,"experiment","output_dir")):
        if value is not None: config[section][key]=value
    return config


def write_metrics(rows,path):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def episode_summary(episodes):
    return summarize_competitive_records(episodes) if episodes else {}


def diagnostic_row(trainer,episodes,metrics,evaluation,active,block):
    local=episode_summary(episodes); overall=evaluation["overall"]
    row={"update":trainer.update_count,"env_steps":trainer.env_steps,"active_policy":active,"block_index":block,**metrics}
    for key in ("red_kill_rate","blue_kill_rate","combat_decisive_rate","red_boundary_loss_rate","blue_boundary_loss_rate","boundary_rate","collision_rate","max_steps_rate","mean_episode_length"):
        row[key]=local.get(key,np.nan); row[f"eval_{key}"]=overall[key]
    for policy in POLICIES:
        for key in ("kill_rate","boundary_loss_rate","as_red_kill_rate","as_blue_kill_rate","role_kill_gap"):
            row[f"eval_policy_{policy}_{key}"]=overall[f"policy_{policy}_{key}"]
    row["eval_min_policy_kill_rate"]=overall["min_policy_kill_rate"]
    row["eval_policy_kill_imbalance"]=overall["policy_kill_imbalance"]
    row["eval_worst_scenario_combat_decisive_rate"]=overall["worst_scenario_combat_decisive_rate"]
    for team in ("red","blue"): row[f"active_team_{team}_count"]=trainer.active_team_counts[team]
    for key,value in trainer.tail_combo_counts.items(): row[f"tail_combo_{key}_count"]=value
    return row


def score(evaluation):
    """Return the v6 score; accept legacy diagnostic dictionaries for old rule tests."""
    if "min_policy_kill_rate" in evaluation.get("overall", {}):
        return competitive_score(evaluation)
    overall = evaluation["overall"]
    return (overall["combat_decisive_rate"], -(overall["boundary_rate"] + overall["collision_rate"]), -overall["mean_episode_length"])

def _state_bytes(value):
    stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()


def main():
    args=parse_args();config=load_config(args);t=config["training"];seed=config["experiment"]["seed"]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    total,num_envs,block_size=int(t["total_env_steps"]),int(t["num_envs"]),int(t["alternating_block_env_steps"])
    if total%num_envs or block_size%num_envs: raise ValueError("step counts must be divisible by num_envs")
    trainer=MAPPOTrainer(args.env_config,config)
    if args.resume: trainer.load_checkpoint(args.resume)
    else:
        trainer.configure_block_opponent(0,"a",force=True)
    output=Path(config["experiment"]["output_dir"]);checkpoints=output/"checkpoints";candidates=checkpoints/"candidates"
    candidates.mkdir(parents=True,exist_ok=True)
    if not args.resume: trainer.save_checkpoint(checkpoints/"initial.pt")
    print(f"device={trainer.device} gpu={torch.cuda.get_device_name(0) if trainer.device.type=='cuda' else None}",flush=True)
    start=time.perf_counter();rows=[];blocks=[]
    evaluation=evaluate_competitive_match(trainer.policy_a_actor,trainer.policy_b_actor,args.env_config,config["evaluation"]["episodes"],trainer.device,seed=seed+100000)
    if trainer.quick_best_score is None: trainer.quick_best_score=competitive_score(evaluation)
    initial_quick=evaluation
    while trainer.env_steps<total:
        block=trainer.block_index();active=trainer.active_policy();end=min(total,(block+1)*block_size)
        if trainer.current_block_index!=block:
            trainer.configure_block_opponent(block,active);trainer.reset_environments()
        if args.smoke:
            frozen=next(p for p in POLICIES if p!=active)
            frozen_bytes=_state_bytes({"actor":trainer._actor(frozen).state_dict(),"critic":trainer._critic(frozen).state_dict(),"ao":getattr(trainer,f"policy_{frozen}_actor_optimizer").state_dict(),"co":getattr(trainer,f"policy_{frozen}_critic_optimizer").state_dict(),"behavior":trainer._actor(frozen,True).state_dict()})
            active_actor_before=_state_bytes(trainer._actor(active).state_dict());active_critic_before=_state_bytes(trainer._critic(active).state_dict())
        completed=trainer.collect_rollout(end-trainer.env_steps)
        if not np.isfinite(trainer.buffer.rewards).all():raise FloatingPointError("non-finite rollout reward")
        metrics=trainer.update(active)
        if args.smoke:
            now=_state_bytes({"actor":trainer._actor(frozen).state_dict(),"critic":trainer._critic(frozen).state_dict(),"ao":getattr(trainer,f"policy_{frozen}_actor_optimizer").state_dict(),"co":getattr(trainer,f"policy_{frozen}_critic_optimizer").state_dict(),"behavior":trainer._actor(frozen,True).state_dict()})
            if now!=frozen_bytes:raise AssertionError("frozen policy state changed")
            if _state_bytes(trainer._actor(active).state_dict())==active_actor_before or _state_bytes(trainer._critic(active).state_dict())==active_critic_before:raise AssertionError("active policy failed to update")
        finished=trainer.env_steps==end
        if finished:metrics.update(trainer.finish_block(active,block))
        if trainer.update_count%t["eval_interval_updates"]==0 or finished:
            evaluation=evaluate_competitive_match(trainer.policy_a_actor,trainer.policy_b_actor,args.env_config,config["evaluation"]["episodes"],trainer.device,seed=seed+100000)
            candidate=competitive_score(evaluation)
            if candidate>tuple(trainer.quick_best_score):
                trainer.quick_best_score=candidate
                path=candidates/f"candidate_update_{trainer.update_count:04d}.pt"
                trainer.candidate_checkpoints.append(str(path.relative_to(output)));trainer.save_checkpoint(path)
        row=diagnostic_row(trainer,completed,metrics,evaluation,active,block);rows.append(row)
        finite=[v for k,v in row.items() if isinstance(v,(int,float,np.number)) and not (isinstance(v,float) and np.isnan(v))]
        if args.smoke and not np.isfinite(finite).all():raise FloatingPointError("non-finite smoke diagnostics")
        trainer.save_checkpoint(checkpoints/"latest.pt");write_metrics(rows,output/"training_metrics.csv")
        print(f"update={trainer.update_count} steps={trainer.env_steps} block={block} active_policy={active} min_policy_kill={row['eval_min_policy_kill_rate']:.3f} opponent={trainer.current_opponent_policy}:{trainer.current_opponent_generation}",flush=True)
        if finished:
            trainer.save_checkpoint(checkpoints/f"block_{block:03d}.pt");blocks.append(deepcopy_safe(trainer.block_history[-1]))
    trainer.save_checkpoint(checkpoints/"final.pt")
    restored_ok=None
    if args.smoke:
        restored=MAPPOTrainer(args.env_config,config);restored.load_checkpoint(checkpoints/"final.pt");restored.collect_rollout()
        restored_ok=restored.env_steps==trainer.env_steps+restored.rollout_steps*restored.num_envs
        if not restored_ok:raise AssertionError("v6 smoke continuation failed")
    final_eval=evaluate_competitive_match(trainer.policy_a_actor,trainer.policy_b_actor,args.env_config,config["evaluation"]["episodes"],trainer.device,seed=seed+100000)
    summary={"version":6,"device":str(trainer.device),"actual_environment_steps":trainer.env_steps,"updates":trainer.update_count,"elapsed_seconds":time.perf_counter()-start,"block_order":[r["active_policy"] for r in trainer.block_history],"blocks":trainer.block_history,"scenario_counts":trainer.scenario_counts,"active_team_counts":trainer.active_team_counts,"tail_combo_counts":trainer.tail_combo_counts,"history_selection_counts":trainer.history_selection_counts,"quick_best_score":list(trainer.quick_best_score),"candidate_checkpoints":trainer.candidate_checkpoints,"initial_quick_evaluation":initial_quick,"final_quick_evaluation":final_eval,"smoke_v6_restore_and_continue_ok":restored_ok}
    (output/"run_summary.json").write_text(json.dumps(summary,indent=2,default=json_default),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=json_default),flush=True)


def deepcopy_safe(value):
    return json.loads(json.dumps(value,default=json_default))


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,tuple):return list(value)
    raise TypeError(type(value).__name__)


if __name__=="__main__":main()
