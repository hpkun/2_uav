"""Formally select the strongest v6 competitive checkpoint without training."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import CHECKPOINT_VERSION,competitive_score,evaluate_competitive_match,resolve_device

def load(path,device):
    c=torch.load(path,map_location=device,weights_only=False)
    if c.get("checkpoint_version")!=CHECKPOINT_VERSION:raise RuntimeError(f"{path} is not a v6 checkpoint")
    n=c["config"]["network"];a=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(device);b=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(device)
    a.load_state_dict(c["policy_a_actor"]);b.load_state_dict(c["policy_b_actor"]);return a,b,c

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",default="outputs/mappo_v6");p.add_argument("--env-config",default="configs/homogeneous_1v1.yaml");p.add_argument("--episodes",type=int,default=300);p.add_argument("--device",default="auto");args=p.parse_args()
    if args.episodes<=0 or args.episodes%2:raise ValueError("--episodes must be a positive even number")
    output=Path(args.output_dir);checkpoints=output/"checkpoints";device=resolve_device(args.device)
    paths=[p for p in (checkpoints/"initial.pt",checkpoints/"final.pt") if p.exists()]+sorted((checkpoints/"candidates").glob("candidate_update_*.pt"))
    if not paths:raise FileNotFoundError("no initial, final, or candidate checkpoints found")
    rows=[];evaluations={}
    for path in paths:
        a,b,c=load(path,device);seed=c["config"]["experiment"]["seed"]+300000
        result=evaluate_competitive_match(a,b,args.env_config,args.episodes,device,seed=seed);score=competitive_score(result);o=result["overall"]
        evaluations[path.name]=result
        rows.append({"checkpoint":str(path.relative_to(output)),"quick_score":json.dumps(c.get("quick_best_score")),"formal_score":json.dumps(score),"worst_scenario_combat_decisive_rate":score[0],"min_policy_kill_rate":o["min_policy_kill_rate"],"paired_combat_decisive_rate":o["paired_combat_decisive_rate"],"policy_a_boundary_loss_rate":o["policy_a_boundary_loss_rate"],"policy_b_boundary_loss_rate":o["policy_b_boundary_loss_rate"],"policy_a_role_kill_gap":o["policy_a_role_kill_gap"],"policy_b_role_kill_gap":o["policy_b_role_kill_gap"],"collision_rate":o["collision_rate"]})
    rows.sort(key=lambda r:tuple(json.loads(r["formal_score"])),reverse=True)
    for rank,row in enumerate(rows,1):row["rank"]=rank
    best=output/rows[0]["checkpoint"];destination=checkpoints/"competitive_best.pt"
    if best.resolve()!=destination.resolve():shutil.copy2(best,destination)
    with (output/"candidate_selection.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary_path=output/"run_summary.json";summary=json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    final_key="final.pt";initial_key="initial.pt"
    best_a,best_b,best_c=load(destination,device);best_eval=evaluate_competitive_match(best_a,best_b,args.env_config,args.episodes,device,seed=best_c["config"]["experiment"]["seed"]+300000)
    best_overall = best_eval["overall"]
    summary.update({"formal_best_score":json.loads(rows[0]["formal_score"]),"formal_best_checkpoint":str(destination.relative_to(output)),"initial_formal_evaluation":evaluations.get(initial_key),"competitive_best_formal_evaluation":best_eval,"final_formal_evaluation":evaluations.get(final_key),"formal_policy_a_kill_rate":best_overall["policy_a_kill_rate"],"formal_policy_b_kill_rate":best_overall["policy_b_kill_rate"],"formal_policy_a_boundary_loss_rate":best_overall["policy_a_boundary_loss_rate"],"formal_policy_b_boundary_loss_rate":best_overall["policy_b_boundary_loss_rate"],"formal_policy_a_role_kill_gap":best_overall["policy_a_role_kill_gap"],"formal_policy_b_role_kill_gap":best_overall["policy_b_role_kill_gap"],"formal_worst_scenario_combat_decisive_rate":best_overall["worst_scenario_combat_decisive_rate"]})
    summary_path.write_text(json.dumps(summary,indent=2,default=lambda x:x.tolist() if isinstance(x,(np.ndarray,)) else x.item() if isinstance(x,np.generic) else x),encoding="utf-8")
    print(json.dumps({"formal_best_checkpoint":str(destination),"formal_best_score":summary["formal_best_score"],"evaluated_checkpoints":len(rows)},indent=2))
if __name__=="__main__":main()
