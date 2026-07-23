"""训练双 Actor 竞争式 MAPPO。"""
import argparse, csv, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml
from uav_combat.mappo.trainer import MAPPOTrainer, evaluate_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(); parser.add_argument("--env-config", default="configs/homogeneous_1v1.yaml"); parser.add_argument("--train-config", default="configs/mappo_1v1.yaml"); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--total-env-steps", type=int); parser.add_argument("--num-envs", type=int); parser.add_argument("--seed", type=int); parser.add_argument("--device"); parser.add_argument("--resume"); return parser.parse_args()


def load_training_config(args: argparse.Namespace) -> dict:
    with Path(args.train_config).open(encoding="utf-8") as stream: config = yaml.safe_load(stream)
    if args.smoke:
        config["training"].update(total_env_steps=4096, num_envs=2, rollout_steps=64, curriculum_tail_chase_env_steps=2048, ppo_epochs=2, minibatch_size=128); config["evaluation"]["episodes"] = 6
    for value, section, key in ((args.total_env_steps,"training","total_env_steps"),(args.num_envs,"training","num_envs"),(args.seed,"experiment","seed"),(args.device,"experiment","device")):
        if value is not None: config[section][key] = value
    return config


def run_evaluations(trainer: MAPPOTrainer, episodes: int, seed: int) -> dict[str, dict]:
    results = {}
    for team, actor in (("red", trainer.red_actor), ("blue", trainer.blue_actor)):
        for opponent in ("zero", "pursuit"):
            results[f"{team}_{opponent}"] = evaluate_actor(actor, trainer.env_config, episodes, trainer.device, opponent, "both", "all", seed)
    return results


def score(evaluations: dict[str, dict]) -> tuple[float, float]:
    overall = [evaluations[key]["overall"] for key in ("red_zero","blue_zero","red_pursuit","blue_pursuit")]
    return (0.5 * (overall[0]["win_rate"] + overall[1]["win_rate"]) + 0.5 * (overall[2]["win_rate"] + overall[3]["win_rate"]), float(np.mean([x["mean_return"] for x in overall])))


def save_curves(rows: list[dict], path: Path, curriculum: int) -> None:
    steps = [r["env_steps"] for r in rows]; fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes[0,0].plot(steps,[r["red_mean_episode_return"] for r in rows],label="red"); axes[0,0].plot(steps,[r["blue_mean_episode_return"] for r in rows],label="blue"); axes[0,0].set_title("Episode return")
    for key in ("red_win_rate","blue_win_rate","draw_rate"): axes[0,1].plot(steps,[r[key] for r in rows],label=key)
    axes[0,1].set_title("Outcomes")
    axes[1,0].plot(steps,[r["red_policy_loss"] for r in rows],label="red"); axes[1,0].plot(steps,[r["blue_policy_loss"] for r in rows],label="blue"); axes[1,0].set_title("Policy loss")
    axes[1,1].plot(steps,[r["value_loss"] for r in rows]); axes[1,1].set_title("Value loss")
    axes[2,0].plot(steps,[r["red_entropy"] for r in rows],label="red"); axes[2,0].plot(steps,[r["blue_entropy"] for r in rows],label="blue"); axes[2,0].set_title("Entropy")
    for key in ("eval_red_zero_win_rate","eval_blue_zero_win_rate","eval_red_pursuit_win_rate","eval_blue_pursuit_win_rate"): axes[2,1].plot(steps,[r[key] for r in rows],label=key)
    axes[2,1].set_title("Evaluation win rates")
    for axis in axes.flat:
        axis.axvline(curriculum, linestyle="--", color="k"); axis.grid(True); axis.set_xlabel("Environment steps")
        handles, labels = axis.get_legend_handles_labels()
        if handles: axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def main() -> None:
    args=parse_args(); config=load_training_config(args); seed=int(config["experiment"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    trainer=MAPPOTrainer(args.env_config,config)
    if args.resume: trainer.load_checkpoint(args.resume)
    if trainer.device.type=="cuda": torch.cuda.reset_peak_memory_stats(trainer.device)
    gpu=torch.cuda.get_device_name(trainer.device) if trainer.device.type=="cuda" else None
    print(f"device={trainer.device} gpu={gpu} torch={torch.__version__} torch_cuda={torch.version.cuda}")
    output=Path(config["experiment"]["output_dir"]); checkpoints=output/"checkpoints"; checkpoints.mkdir(parents=True,exist_ok=True)
    episodes=int(config["evaluation"]["episodes"]); evaluations=run_evaluations(trainer,episodes,seed+100000); best_score=score(evaluations)
    trainer.save_checkpoint(checkpoints/"initial.pt"); trainer.save_checkpoint(checkpoints/"best.pt"); rows=[]
    total=int(config["training"]["total_env_steps"])
    while trainer.env_steps<total:
        completed=trainer.collect_rollout(); metrics=trainer.update()
        if trainer.update_count%int(config["training"]["eval_interval_updates"])==0 or trainer.env_steps>=total:
            evaluations=run_evaluations(trainer,episodes,seed+100000); candidate=score(evaluations)
            if candidate>best_score: best_score=candidate; trainer.save_checkpoint(checkpoints/"best.pt")
        count=max(len(completed),1); reasons=[e["reason"] for e in completed]; outcomes=[e["outcome"] for e in completed]
        def ev(key,metric): return evaluations[key]["overall"][metric]
        row={"update":trainer.update_count,"env_steps":trainer.env_steps,"phase":trainer.phase(),
             **{k:metrics[k] for k in ("red_policy_loss","blue_policy_loss","value_loss","red_entropy","blue_entropy","red_approx_kl","blue_approx_kl","red_clip_fraction","blue_clip_fraction","red_explained_variance","blue_explained_variance")},
             "red_mean_episode_return":float(np.mean([e["returns"][0] for e in completed])) if completed else 0.0,"blue_mean_episode_return":float(np.mean([e["returns"][1] for e in completed])) if completed else 0.0,"mean_episode_length":float(np.mean([e["length"] for e in completed])) if completed else 0.0,
             "red_win_rate":outcomes.count("red")/count,"blue_win_rate":outcomes.count("blue")/count,"draw_rate":outcomes.count("draw")/count,
             "collision_rate":reasons.count("collision")/count,"boundary_rate":sum(r in {"altitude_boundary","xy_boundary","boundary"} for r in reasons)/count,"max_steps_rate":reasons.count("max_steps")/count,"red_kill_rate":reasons.count("red_kill")/count,"blue_kill_rate":reasons.count("blue_kill")/count}
        for team in ("red","blue"):
            for opponent in ("zero","pursuit"):
                row[f"eval_{team}_{opponent}_win_rate"]=ev(f"{team}_{opponent}","win_rate"); row[f"eval_{team}_{opponent}_mean_return"]=ev(f"{team}_{opponent}","mean_return")
        rows.append(row); trainer.save_checkpoint(checkpoints/"latest.pt")
        if trainer.update_count%int(config["training"]["checkpoint_interval_updates"])==0: trainer.save_checkpoint(checkpoints/f"update_{trainer.update_count}.pt")
        print(f"update={trainer.update_count} steps={trainer.env_steps} phase={row['phase']} red_return={row['red_mean_episode_return']:.3f} blue_return={row['blue_mean_episode_return']:.3f} red_win={row['red_win_rate']:.2f} blue_win={row['blue_win_rate']:.2f} draw={row['draw_rate']:.2f} red_policy={row['red_policy_loss']:.4f} blue_policy={row['blue_policy_loss']:.4f} value={row['value_loss']:.4f} eval={row['eval_red_zero_win_rate']:.2f}/{row['eval_blue_zero_win_rate']:.2f}/{row['eval_red_pursuit_win_rate']:.2f}/{row['eval_blue_pursuit_win_rate']:.2f}")
    trainer.save_checkpoint(checkpoints/"final.pt")
    with (output/"training_metrics.csv").open("w",newline="",encoding="utf-8") as stream: writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_curves(rows,output/"training_curves.png",int(config["training"]["curriculum_tail_chase_env_steps"]))
    peak=torch.cuda.max_memory_allocated(trainer.device) if trainer.device.type=="cuda" else 0; print(f"peak_gpu_memory_bytes={peak}")


if __name__=="__main__": main()
