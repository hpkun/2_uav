"""Evaluate a 3v3 MAPPO checkpoint with parallel vector env."""
import argparse, json, time
from pathlib import Path
import torch
from uav_combat.mappo.networks import GaussianActor
from uav_combat.mappo.evaluation_3v3 import evaluate_mappo_fixed_blue_3v3
from uav_combat.mappo.trainer_3v3 import CHECKPOINT_FAMILY, OBS_DIM, resolve_device

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--env-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if ckpt.get("checkpoint_family") != CHECKPOINT_FAMILY:
        raise RuntimeError(f"Expected {CHECKPOINT_FAMILY}")
    n_cfg = ckpt["config"]["network"]
    actor = GaussianActor(OBS_DIM, 3, n_cfg["hidden_dim"], n_cfg["log_std_init"]).to(device)
    actor.load_state_dict(ckpt["shared_red_actor"])

    t0 = time.perf_counter()
    result = evaluate_mappo_fixed_blue_3v3(actor, args.env_config, args.episodes,
                                             args.num_envs, args.env_workers, device,
                                             seed_start=ckpt["config"]["experiment"]["seed"] + 300000)
    result["wall_seconds"] = time.perf_counter() - t0
    print(json.dumps(result, indent=2, default=str))
    out = Path(args.checkpoint).parent.parent / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    fname = f"eval_{Path(args.checkpoint).stem}_ep{args.episodes}.json"
    (out / fname).write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved to {out / fname}")

if __name__ == "__main__": main()
