"""Generate PPT-ready weekly-report figures from existing 3v3 MAPPO results.

This script is intentionally presentation-focused: it reads existing metrics,
evaluation JSONs, environment audit JSONs, and checkpoints.  It does not train,
modify the environment, change rewards, or change algorithms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "mappo_3v3_reward_v3_stable_5m"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "outputs" / "3v3_environment_contract_audit_v4.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "weekly_report_figures"
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs" / "homogeneous_3v3_reward_v3.yaml"

RED = "#c43c39"
BLUE = "#2f6fb6"
GRAY = "#6f7378"
GREEN = "#3b8f5a"
ORANGE = "#d98b2b"
PURPLE = "#7a5195"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _series(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_to_float(r.get(key)) for r in rows], dtype=float)


def _smooth(y: np.ndarray, window: int = 21) -> np.ndarray:
    if len(y) == 0:
        return y
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return y
    filled = y.copy()
    idx = np.arange(len(y))
    filled[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    w = min(window, max(3, len(y) // 12))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    pad = w // 2
    padded = np.pad(filled, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _step_from_eval_path(path: Path) -> int | None:
    if path.name == "evaluation_initial.json":
        return 0
    m = re.match(r"evaluation_step_(\d+)\.json$", path.name)
    if m:
        return int(m.group(1))
    return None


def _load_eval_progress(evals_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(evals_dir.glob("evaluation_*.json")):
        step = _step_from_eval_path(path)
        if step is None:
            continue
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        data = dict(data)
        data["env_steps"] = step
        data["source"] = str(path)
        records.append(data)
    return sorted(records, key=lambda r: r["env_steps"])


def _load_named_eval(run_dir: Path, name: str) -> dict[str, Any] | None:
    p = run_dir / "evaluations" / f"evaluation_{name}.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    data["source"] = str(p)
    return data


def plot_training_curve(run_dir: Path, out_dir: Path) -> Path:
    rows = _read_csv(run_dir / "training_metrics.csv")
    x = _series(rows, "env_steps") / 1e6
    reward = _series(rows, "mean_rollout_tactical_reward")
    ret = _series(rows, "mean_episode_return")
    entropy = _series(rows, "entropy")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x, _smooth(reward), color=RED, label="Rollout tactical reward")
    if np.isfinite(ret).sum() > 5:
        ax.plot(x, _smooth(ret), color=GRAY, label="Episode return")
    ax.set_xlabel("Environment steps (million)")
    ax.set_ylabel("Reward")
    ax.set_title("MAPPO training signal in simplified homogeneous 3v3")
    ax.legend(loc="best", frameon=False)

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(x, _smooth(entropy), color=BLUE, alpha=0.75, label="Policy entropy")
    ax2.set_ylabel("Entropy")
    ax2.legend(loc="upper right", frameon=False)
    return _save(fig, out_dir / "fig1_training_curve.png")


def plot_survival_boundary_curve(run_dir: Path, out_dir: Path) -> Path:
    records = _load_eval_progress(run_dir / "evaluations")
    if not records:
        rows = _read_csv(run_dir / "training_metrics.csv")
        records = [{"env_steps": _to_float(r.get("env_steps")), **r} for r in rows]

    x = np.asarray([_to_float(r["env_steps"]) for r in records]) / 1e6
    red_surv = np.asarray([_to_float(r.get("mean_red_survivors")) for r in records])
    blue_surv = np.asarray([_to_float(r.get("mean_blue_survivors")) for r in records])
    red_bdy = np.asarray([_to_float(r.get("mean_red_boundary_deaths")) for r in records])
    blue_bdy = np.asarray([_to_float(r.get("mean_blue_boundary_deaths")) for r in records])
    draw = np.asarray([_to_float(r.get("draw_rate")) for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True)
    axes[0].plot(x, red_surv, color=RED, marker="o", markersize=3, label="Red survivors")
    axes[0].plot(x, blue_surv, color=BLUE, marker="o", markersize=3, label="Blue survivors")
    axes[0].set_title("Survival changes over checkpoints")
    axes[0].set_xlabel("Environment steps (million)")
    axes[0].set_ylabel("Aircraft / episode")
    axes[0].set_ylim(-0.05, 3.15)
    axes[0].legend(frameon=False)

    axes[1].plot(x, red_bdy, color=ORANGE, marker="o", markersize=3, label="Red boundary deaths")
    axes[1].plot(x, blue_bdy, color=PURPLE, marker="o", markersize=3, label="Blue boundary deaths")
    axes[1].plot(x, draw, color=GRAY, linestyle="--", marker="s", markersize=3, label="Draw rate")
    axes[1].set_title("Safety and non-decisive outcomes")
    axes[1].set_xlabel("Environment steps (million)")
    axes[1].set_ylabel("Rate or count / episode")
    axes[1].set_ylim(-0.05, max(3.1, np.nanmax([red_bdy, blue_bdy]) + 0.2))
    axes[1].legend(frameon=False)
    return _save(fig, out_dir / "fig2_survival_boundary_curve.png")


def plot_environment_timescale(audit_json: Path, out_dir: Path) -> Path:
    with audit_json.open(encoding="utf-8") as f:
        audit = json.load(f)
    configs = audit["config_audits"]
    names = [n for n in ("homogeneous_3v3", "homogeneous_3v3_reward_v3", "homogeneous_3v3_learnable_v4") if n in configs]
    labels = {
        "homogeneous_3v3": "Base",
        "homogeneous_3v3_reward_v3": "Reward v3",
        "homogeneous_3v3_learnable_v4": "Learnable v4",
    }
    merge = [configs[n]["actual_time_to_attack_distance_seconds"]["mean"] for n in names]
    def turn_time(config_name: str, key: str) -> float:
        maneuver = configs[config_name]["maneuver_control_audit"]
        if "speeds" in maneuver:
            return float(maneuver["speeds"]["150_mps"]["positive_yaw"][f"{key}_seconds"])
        return float(maneuver["150_mps"]["max_positive_yaw"][key])

    turn90 = [turn_time(n, "time_to_90_deg") for n in names]
    turn180 = [turn_time(n, "time_to_180_deg") for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - width, merge, width, color=GRAY, label="Mean time to 1000 m")
    ax.bar(x, turn90, width, color=BLUE, label="90° turn time @150 m/s")
    ax.bar(x + width, turn180, width, color=RED, label="180° turn time @150 m/s")
    ax.set_xticks(x, [labels[n] for n in names])
    ax.set_ylabel("Seconds")
    ax.set_title("Scenario timescale calibrated against real maneuver time")
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8, padding=2)
    return _save(fig, out_dir / "fig5_environment_timescale.png")


def plot_checkpoint_comparison(run_dir: Path, out_dir: Path) -> Path:
    entries: list[tuple[str, dict[str, Any]]] = []
    for label, name in [("Initial", "initial"), ("Best", "best"), ("Latest", "final")]:
        data = _load_named_eval(run_dir, name)
        if data is not None:
            entries.append((label, data))
    if not any(label == "Latest" for label, _ in entries):
        progress = _load_eval_progress(run_dir / "evaluations")
        if progress:
            entries.append(("Latest", progress[-1]))
    if len(entries) < 2:
        raise RuntimeError(f"Not enough evaluation files in {run_dir / 'evaluations'}")

    metrics = [
        ("mean_red_attack_kills", "Red attack kills", RED),
        ("mean_blue_attack_kills", "Blue attack kills", BLUE),
        ("mean_red_survivors", "Red survivors", "#e06b68"),
        ("mean_red_boundary_deaths", "Red boundary deaths", ORANGE),
        ("draw_rate", "Draw rate", GRAY),
    ]
    x = np.arange(len(entries))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, (key, label, color) in enumerate(metrics):
        vals = [_to_float(d.get(key)) for _, d in entries]
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=color)
    ax.set_xticks(x, [label for label, _ in entries])
    ax.set_ylabel("Rate or mean count / episode")
    ax.set_title("Checkpoint-level behavior summary")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out_dir / "fig6_checkpoint_comparison.png")


def _load_actor(checkpoint: Path, device: str):
    import torch
    from uav_combat.environment_3v3 import OBS_DIM
    from uav_combat.mappo.networks import GaussianActor

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    n_cfg = ckpt["config"]["network"]
    actor = GaussianActor(
        OBS_DIM, 3, n_cfg["hidden_dim"], n_cfg["log_std_init"],
        log_std_min=n_cfg.get("log_std_min", -5.0),
        log_std_max=n_cfg.get("log_std_max", 2.0),
    ).to(dev)
    actor.load_state_dict(ckpt["shared_red_actor"])
    actor.eval()
    return actor, dev


def export_trajectory(checkpoint: Path, env_config: Path, out_json: Path, seed: int, device: str, max_steps: int | None) -> Path:
    import torch
    from uav_combat.environment_3v3 import Homogeneous3v3AirCombatEnv, RED_IDS
    from uav_combat.rule_policy_3v3 import NearestTargetPursuitPolicy3v3

    actor, dev = _load_actor(checkpoint, device)
    env = Homogeneous3v3AirCombatEnv(env_config)
    act_cfg = env.config["action"]
    blue_policy = NearestTargetPursuitPolicy3v3(
        act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"]
    )
    obs, info = env.reset(seed=seed)
    max_n = max_steps or int(env.config["simulation"]["max_steps"])
    dt = float(env.config["simulation"]["dt"])

    frames: list[dict[str, Any]] = []

    def capture(step_info: dict[str, Any]) -> None:
        aircraft = {}
        for ac in env.aircraft:
            s = ac.state
            aircraft[ac.aircraft_id] = {
                "team": ac.team,
                "x": float(s.x),
                "y": float(s.y),
                "altitude": float(s.altitude),
                "v": float(s.v),
                "theta": float(s.theta),
                "psi": float(s.psi),
                "alive": bool(s.alive),
            }
        frames.append({
            "step": int(env.step_count),
            "time_s": float(env.step_count * dt),
            "aircraft": aircraft,
            "info": {
                "outcome": step_info.get("outcome"),
                "termination_reason": step_info.get("termination_reason"),
                "red_alive_count": step_info.get("red_alive_count"),
                "blue_alive_count": step_info.get("blue_alive_count"),
                "attack_kills": step_info.get("attack_kills"),
                "death_causes": {k: int(v) for k, v in step_info.get("death_causes", {}).items()},
            },
        })

    capture(info)
    terminated = truncated = False
    while not (terminated or truncated) and env.step_count < max_n:
        actions: dict[str, np.ndarray] = {}
        red_obs = []
        red_ids = []
        for aid in RED_IDS:
            ac = env._aircraft_by_id(aid)
            if ac.state.alive:
                red_ids.append(aid)
                red_obs.append(obs[aid])
        if red_obs:
            with torch.no_grad():
                arr = torch.as_tensor(np.asarray(red_obs, dtype=np.float32), device=dev)
                red_actions = actor.deterministic_action(arr).cpu().numpy()
            for aid, action in zip(red_ids, red_actions):
                actions[aid] = np.asarray(action, dtype=np.float32)
        reds = [a for a in env.aircraft if a.team == "red"]
        blues = [a for a in env.aircraft if a.team == "blue"]
        blue_actions, _ = blue_policy.select_actions(blues, reds)
        actions.update({aid: action for aid, action in blue_actions.items() if env._aircraft_by_id(aid).state.alive})
        obs, rewards, terminated, truncated, info = env.step(actions)
        capture(info)

    data = {
        "checkpoint": str(checkpoint),
        "env_config": str(env_config),
        "seed": seed,
        "dt": dt,
        "frames": frames,
        "episode_summary": frames[-1]["info"],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_json


def plot_trajectory(json_path: Path, png_path: Path, title: str, projection: str = "xy") -> Path:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    ids = list(data["frames"][0]["aircraft"].keys())
    red_ids = [i for i in ids if i.startswith("red")]
    blue_ids = [i for i in ids if i.startswith("blue")]
    colors = {aid: RED for aid in red_ids} | {aid: BLUE for aid in blue_ids}
    linestyles = {0: "-", 1: "--", 2: ":"}

    if projection == "3d":
        fig = plt.figure(figsize=(7.6, 6.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.4, 6.2))

    for aid in red_ids + blue_ids:
        pts = np.asarray([
            [fr["aircraft"][aid]["x"], fr["aircraft"][aid]["y"], fr["aircraft"][aid]["altitude"], fr["aircraft"][aid]["alive"]]
            for fr in data["frames"]
        ], dtype=float)
        style = linestyles[int(aid.split("_")[1]) % 3]
        label = aid
        if projection == "3d":
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[aid], linestyle=style, label=label)
            ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=colors[aid], marker="o", s=34)
            ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=colors[aid], marker="x", s=42)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=colors[aid], linestyle=style, label=label)
            ax.scatter(pts[0, 0], pts[0, 1], color=colors[aid], marker="o", s=38)
            ax.scatter(pts[-1, 0], pts[-1, 1], color=colors[aid], marker="x", s=48)

    ax.set_title(title)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    if projection == "3d":
        ax.set_zlabel("Altitude (m)")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.legend(ncol=2, frameon=False, loc="best")
    summary = data.get("episode_summary", {})
    subtitle = f"seed={data.get('seed')}, steps={data['frames'][-1]['step']}, outcome={summary.get('outcome')}, reason={summary.get('termination_reason')}"
    if projection == "3d":
        ax.text2D(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")
    else:
        ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=8, color=GRAY, va="bottom")
    return _save(fig, png_path)


def generate_trajectories(run_dir: Path, env_config: Path, out_dir: Path, seed: int, device: str, max_steps: int | None) -> list[Path]:
    checkpoints = [
        ("initial", run_dir / "checkpoints" / "initial.pt", "Initial checkpoint"),
        ("best", run_dir / "checkpoints" / "best.pt", "Best checkpoint"),
        ("latest", run_dir / "checkpoints" / "latest.pt", "Latest checkpoint"),
    ]
    outputs: list[Path] = []
    for name, ckpt, label in checkpoints:
        if not ckpt.exists():
            continue
        traj_json = out_dir / f"trajectory_{name}.json"
        export_trajectory(ckpt, env_config, traj_json, seed=seed, device=device, max_steps=max_steps)
        outputs.append(plot_trajectory(traj_json, out_dir / f"fig3_trajectory_{name}.png", f"3v3 trajectory: {label}", "xy"))
    if (out_dir / "trajectory_best.json").exists():
        outputs.append(plot_trajectory(out_dir / "trajectory_best.json", out_dir / "fig4_trajectory_best_3d.png", "3v3 trajectory: Best checkpoint (3D)", "3d"))
    return outputs


def write_manifest(out_dir: Path, paths: list[Path], run_dir: Path, audit_json: Path) -> Path:
    manifest = {
        "recommended_core_figure": "fig3_trajectory_best.png",
        "source_run_dir": str(run_dir),
        "source_audit_json": str(audit_json),
        "figures": [str(p) for p in paths],
        "notes": [
            "Figures are generated from existing outputs/checkpoints only.",
            "Trajectory JSON files are light one-episode rollouts for visualization; no training was run.",
            "If final.pt is absent, latest.pt is used as the latest/final-stage trajectory checkpoint.",
        ],
    }
    path = out_dir / "weekly_report_figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--env-config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory-seed", type=int, default=20260726)
    parser.add_argument("--trajectory-device", default="cpu", help="cpu is enough for one-episode visualization rollouts")
    parser.add_argument("--trajectory-max-steps", type=int, default=None)
    parser.add_argument("--skip-trajectories", action="store_true")
    args = parser.parse_args()

    _setup_style()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_training_curve(args.run_dir, out_dir),
        plot_survival_boundary_curve(args.run_dir, out_dir),
        plot_environment_timescale(args.audit_json, out_dir),
        plot_checkpoint_comparison(args.run_dir, out_dir),
    ]
    if not args.skip_trajectories:
        paths.extend(generate_trajectories(
            args.run_dir, args.env_config, out_dir,
            seed=args.trajectory_seed, device=args.trajectory_device,
            max_steps=args.trajectory_max_steps,
        ))
    manifest = write_manifest(out_dir, paths, args.run_dir, args.audit_json)
    print("Generated weekly-report figures:")
    for p in paths:
        print(f"  {p}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
