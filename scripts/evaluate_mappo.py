"""评估训练后的共享 MAPPO Actor。"""
import argparse
import json
from pathlib import Path

import torch

from uav_combat.mappo.networks import SharedActor
from uav_combat.mappo.trainer import evaluate_policy, resolve_device


def main() -> None:
    """加载检查点，以 zero 或 pursuit 对手评估并保存 JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", default="configs/homogeneous_1v1.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--opponent", choices=("zero", "pursuit"), default="zero")
    parser.add_argument("--side", choices=("red", "blue", "both"), default="both")
    parser.add_argument("--scenario", choices=("all", "tail_chase", "offset_head_on", "crossing"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    actor = SharedActor(13, 3, config["network"]["hidden_dim"], config["network"]["log_std_init"]).to(device)
    actor.load_state_dict(checkpoint["actor"])
    result = evaluate_policy(actor, args.env_config, args.episodes, device, args.opponent, args.side, args.scenario, checkpoint["seed"] + 200000)
    output = Path(config["experiment"]["output_dir"]) / f"evaluation_{args.opponent}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
