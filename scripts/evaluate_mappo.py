"""Evaluate a v6 policy checkpoint with paired color swaps."""
import argparse,json
from pathlib import Path
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import CHECKPOINT_VERSION,evaluate_matchup,resolve_device

def load_actors(path,device):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get("checkpoint_version")!=CHECKPOINT_VERSION:raise RuntimeError("evaluate_mappo.py requires a v6 checkpoint; v5 and earlier are incompatible")
    n=checkpoint["config"]["network"]
    a=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(device);b=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(device)
    a.load_state_dict(checkpoint["policy_a_actor"]);b.load_state_dict(checkpoint["policy_b_actor"])
    return a,b,checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--env-config",default="configs/homogeneous_1v1.yaml")
    p.add_argument("--matchup",required=True,choices=("a_vs_b","a_vs_zero","a_vs_pursuit","b_vs_zero","b_vs_pursuit"))
    p.add_argument("--episodes",type=int,default=300);p.add_argument("--scenario",choices=("all","tail_chase","offset_head_on","crossing"),default="all")
    p.add_argument("--device",default="auto");p.add_argument("--seedset",default="seedset0");args=p.parse_args()
    device=resolve_device(args.device);a,b,checkpoint=load_actors(args.checkpoint,device)
    result=evaluate_matchup(a,b,args.env_config,args.episodes,device,args.matchup,args.scenario,checkpoint["config"]["experiment"]["seed"]+200000)
    output=Path(checkpoint["config"]["experiment"]["output_dir"])/f"evaluation_{Path(args.checkpoint).stem}_{args.matchup}_{args.scenario}_{args.seedset}.json"
    output.write_text(json.dumps(result,indent=2,default=lambda x:x.tolist()),encoding="utf-8");print(json.dumps(result,indent=2,default=lambda x:x.tolist()));print(f"saved: {output}")
if __name__=="__main__":main()
