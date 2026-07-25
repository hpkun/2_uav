"""Train 3v3 red MAPPO vs fixed-rule blue."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
import yaml
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-config", default="configs/homogeneous_3v3.yaml")
    p.add_argument("--train-config", default="configs/mappo_3v3_fixed_blue.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--total-env-steps", type=int)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--env-workers", type=int, default=None)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--output-dir")
    p.add_argument("--resume")
    return p.parse_args()

def load_config(args):
    with open(args.train_config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config["training"].get("training_mode") != "fixed_rule_blue_3v3":
        raise ValueError("train_mappo_3v3.py requires fixed_rule_blue_3v3")
    config["experiment"]["output_dir"] = config["experiment"].get("output_dir", "outputs/mappo_3v3_fixed_blue")
    config["training"].setdefault("num_env_workers", 4)
    if args.smoke:
        config["training"].update(total_env_steps=16384, num_envs=8, rollout_steps=128, ppo_epochs=2, minibatch_size=512)
        config["experiment"]["output_dir"] = "outputs/mappo_3v3_fixed_blue_smoke"
    for val, sec, key in (
        (args.total_env_steps, "training", "total_env_steps"),
        (args.num_envs, "training", "num_envs"),
        (args.env_workers, "training", "num_env_workers"),
        (args.seed, "experiment", "seed"),
        (args.device, "experiment", "device"),
        (args.output_dir, "experiment", "output_dir"),
    ):
        if val is not None:
            config[sec][key] = val
    return config

def main():
    args = parse_args()
    config = load_config(args)
    t_cfg = config["training"]
    seed = config["experiment"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    total = int(t_cfg["total_env_steps"])
    num_envs = int(t_cfg["num_envs"])
    num_workers = int(t_cfg.get("num_env_workers", 4))
    print(f"num_envs={num_envs} num_env_workers={num_workers} envs_per_worker={num_envs // num_workers}", flush=True)

    trainer = FixedBlue3v3MAPPOTrainer(args.env_config, config)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    output = Path(config["experiment"]["output_dir"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoints / "initial.pt")

    print(f"device={trainer.device}", flush=True)
    start = time.perf_counter()
    rows = []

    try:
        while trainer.env_steps < total:
            remaining = total - trainer.env_steps
            completed = trainer.collect_rollout(remaining)
            metrics = trainer.update()

            if args.smoke:
                finite_vals = [v for v in metrics.values() if isinstance(v, (int, float))]
                if not np.isfinite(finite_vals).all():
                    raise FloatingPointError("non-finite smoke metrics")

            row = {
                "update": trainer.update_count,
                "env_steps": trainer.env_steps,
                **metrics,
            }
            rows.append(row)

            trainer.save_checkpoint(checkpoints / "latest.pt")
            elapsed = time.perf_counter() - start
            env_sps = trainer.env_steps / elapsed if elapsed > 0 else 0
            tmd = trainer._timing
            print(f"update={trainer.update_count} steps={trainer.env_steps} env_sps={env_sps:.1f} "
                  f"env_step={tmd['env_step']:.1f}s policy={tmd['policy_inference']:.1f}s "
                  f"ppo={tmd['ppo_update']:.1f}s loss={metrics['policy_loss']:.4f} "
                  f"val_loss={metrics['value_loss']:.4f}", flush=True)

            # Checkpoint milestones
            eval_interval = int(t_cfg.get("evaluation_interval_env_steps", 100000))
            milestone = (trainer.env_steps // eval_interval) * eval_interval
            if trainer.env_steps > 0 and trainer.env_steps % eval_interval < num_envs * trainer.rollout_steps:
                ckpt_path = checkpoints / f"step_{milestone:06d}.pt"
                if not ckpt_path.exists():
                    trainer.save_checkpoint(ckpt_path)
    finally:
        trainer.close()

    trainer.save_checkpoint(checkpoints / "final.pt")

    # Smoke restore test
    restored_ok = None
    if args.smoke:
        restored = FixedBlue3v3MAPPOTrainer(args.env_config, config)
        restored.load_checkpoint(checkpoints / "final.pt")
        restored.collect_rollout()
        restored_ok = restored.env_steps == trainer.env_steps + restored.rollout_steps * restored.num_envs
        restored.close()
        if not restored_ok:
            raise AssertionError("3v3 smoke continuation failed")

    elapsed_total = time.perf_counter() - start
    tmd = trainer._timing
    summary = {
        "checkpoint_family": "homogeneous_3v3_fixed_blue",
        "checkpoint_version": 1,
        "device": str(trainer.device),
        "actual_environment_steps": trainer.env_steps,
        "updates": trainer.update_count,
        "num_envs": num_envs,
        "num_env_workers": num_workers,
        "environments_per_worker": num_envs // num_workers,
        "environment_step_seconds": tmd["env_step"],
        "policy_inference_seconds": tmd["policy_inference"],
        "ppo_update_seconds": tmd["ppo_update"],
        "total_training_seconds": elapsed_total,
        "environment_steps_per_second": trainer.env_steps / elapsed_total if elapsed_total > 0 else 0,
        "smoke_restore_and_continue_ok": restored_ok,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)

if __name__ == "__main__":
    main()
