"""评估 v2 检查点中的红方或蓝方 Actor。"""
import argparse,json
from pathlib import Path
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.trainer import evaluate_actor,resolve_device


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--env-config",default="configs/homogeneous_1v1.yaml"); parser.add_argument("--actor",choices=("red","blue"),required=True); parser.add_argument("--episodes",type=int,default=90); parser.add_argument("--opponent",choices=("zero","pursuit"),default="zero"); parser.add_argument("--side",choices=("red","blue","both"),default="both"); parser.add_argument("--scenario",choices=("all","tail_chase","offset_head_on","crossing"),default="all"); parser.add_argument("--device",default="auto"); args=parser.parse_args()
    device=resolve_device(args.device); checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    if checkpoint.get("checkpoint_version",0)<2: raise RuntimeError("旧共享Actor检查点不能用于双Actor评估")
    config=checkpoint["config"]; actor=GaussianActor(14,3,config["network"]["hidden_dim"],config["network"]["log_std_init"]).to(device); actor.load_state_dict(checkpoint[f"{args.actor}_actor"])
    result=evaluate_actor(actor,args.env_config,args.episodes,device,args.opponent,args.side,args.scenario,checkpoint["seed"]+200000)
    output=Path(config["experiment"]["output_dir"])/f"evaluation_{args.actor}_{args.opponent}.json"; output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); print(f"saved: {output}")


if __name__=="__main__": main()
