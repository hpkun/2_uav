"""Evaluate exact-2M TACM ablation checkpoints under one fixed protocol."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.tacm_ablation_protocol import (
    ENV_CONFIG, MANIFEST_DIR, METHODS, PAPER_SEEDS, TOTAL_STEPS, evaluation_command,
)
from env.mavuav import load_environment_config


def validate_checkpoint_identity(payload: dict, method: str, manifest_seed: int) -> dict:
    """Prove that an exact-2M checkpoint is the method/seed declared by its manifest."""
    if method not in METHODS:
        raise RuntimeError(f"unknown paper method: {method!r}")
    config = payload.get("trainer_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint is missing trainer_config identity metadata")
    checkpoint_environment = payload.get("environment_config")
    if not isinstance(checkpoint_environment, dict):
        raise RuntimeError("checkpoint is missing resolved environment_config")
    if load_environment_config(checkpoint_environment) != load_environment_config(ENV_CONFIG):
        raise RuntimeError("checkpoint resolved environment_config differs from formal v3.10 contract")
    checks = {
        "sampled_steps": (int(payload.get("sampled_steps", -1)), TOTAL_STEPS),
        "training_seed": (int(config.get("seed", -1)), int(manifest_seed)),
        "environment_version": (payload.get("environment_version"), "heterogeneous_mavuav_4v4_v3_10"),
        "environment_profile": (payload.get("environment_profile"), "main"),
        "trainer_environment_profile": (config.get("environment_profile"), "main"),
        "actor_variant": (payload.get("actor_variant"), "vanilla"),
        "critic_variant": (payload.get("critic_variant"), "mlp"),
        "method_variant": (payload.get("method_variant"), METHODS[method]["method_variant"]),
        "trainer_method_variant": (config.get("method_variant"), METHODS[method]["method_variant"]),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(
                f"checkpoint identity mismatch for {field}: expected {expected!r}, got {actual!r}"
            )
    actor_keys = tuple(str(key) for key in payload.get("actors", {}))
    has_router = any(".router." in key for key in actor_keys)
    has_experts = any(".experts." in key for key in actor_keys)
    has_role_critics = "role_critic_mav" in payload and "role_critic_uav" in payload
    if method == "happo":
        if has_role_critics or has_router or has_experts:
            raise RuntimeError("HAPPO checkpoint unexpectedly contains role or tactical-mode modules")
    elif method == "no_mode":
        if not has_role_critics or has_router or has_experts:
            raise RuntimeError("TACM w/o Mode checkpoint does not match the RGAA-only architecture")
        if config.get("role_module_enabled") is not True or payload.get("role_advantage_coef") != 0.5:
            raise RuntimeError("TACM w/o Mode checkpoint does not prove the frozen role mechanism")
        if payload.get("role_aux_reward_mode") != "process_plus_own_loss":
            raise RuntimeError("TACM w/o Mode checkpoint has incompatible role auxiliary semantics")
    else:
        expected_coef = 0.0 if method == "no_temporal" else 0.01
        if not has_role_critics or not has_router or not has_experts:
            raise RuntimeError(f"{METHODS[method]['paper_variant']} checkpoint lacks TACM actor/role modules")
        if config.get("role_module_enabled") is not True or payload.get("role_advantage_coef") != 0.5:
            raise RuntimeError("TACM checkpoint does not prove the frozen role mechanism")
        if payload.get("algorithm") != "tacm_rgaa_happo":
            raise RuntimeError("TACM checkpoint algorithm metadata mismatch")
        top_coef = payload.get("tacm_temporal_coef")
        config_coef = config.get("tacm_temporal_coef")
        if top_coef != expected_coef or config_coef != expected_coef:
            raise RuntimeError(
                f"checkpoint temporal coefficient mismatch for {method}: "
                f"expected {expected_coef}, got metadata={top_coef!r}, config={config_coef!r}"
            )
    return {
        "method": method, "training_seed": int(manifest_seed),
        "sampled_steps": TOTAL_STEPS, "method_variant": payload["method_variant"],
        "environment_version": payload["environment_version"],
        "tacm_temporal_coef": payload.get("tacm_temporal_coef"),
    }


def load_training_manifests(method: str) -> list[dict]:
    methods = tuple(METHODS) if method == "all" else (method,)
    manifests = []
    for name in methods:
        path = MANIFEST_DIR / f"{name}_latest.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed training manifest: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "complete" or tuple(data.get("seeds", ())) != PAPER_SEEDS:
            raise RuntimeError(f"training manifest is not a complete 17/23/31 contract: {path}")
        manifests.append(data)
    return manifests


def build_plan(method: str, manifests: list[dict], python: str = sys.executable) -> list[dict]:
    plan = []
    for manifest in manifests:
        for run in manifest["runs"]:
            checkpoint = Path(run["exact_2m_checkpoint"])
            plan.append({
                "method": manifest["method"], "paper_variant": manifest["paper_variant"],
                "training_seed": int(run["seed"]), "checkpoint": str(checkpoint),
                "command": evaluation_command(checkpoint, python=python),
                "evaluation_csv": str(checkpoint.parent / "evaluation_2000000_stochastic.csv"),
                "evaluation_summary": str(checkpoint.parent / "evaluation_2000000_stochastic_summary.json"),
            })
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        methods = tuple(METHODS) if args.method == "all" else (args.method,)
        plan = []
        for method in methods:
            for seed in PAPER_SEEDS:
                checkpoint = Path(f"<exact-2m-{method}-seed{seed}>") / "checkpoint_2000000.pt"
                plan.append({
                    "method": method, "paper_variant": METHODS[method]["paper_variant"],
                    "training_seed": seed, "checkpoint": str(checkpoint),
                    "command": evaluation_command(checkpoint, python=sys.executable),
                })
        print(json.dumps({"method": args.method, "runs": [
            {**row, "command": shlex.join(row["command"])} for row in plan
        ]}, indent=2, ensure_ascii=False))
        return
    manifests = load_training_manifests(args.method)
    plan = build_plan(args.method, manifests)
    if not torch.cuda.is_available():
        raise RuntimeError("formal TACM ablation evaluation requires CUDA; CPU fallback is forbidden")
    for row in plan:
        checkpoint = Path(row["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        row["checkpoint_identity"] = validate_checkpoint_identity(
            payload, str(row["method"]), int(row["training_seed"]),
        )
        for output in (row["evaluation_csv"], row["evaluation_summary"]):
            if Path(output).exists():
                raise FileExistsError(f"refusing to overwrite formal evaluation: {output}")
        subprocess.run(row["command"], cwd=PROJECT_ROOT, check=True)
        row["status"] = "complete"
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MANIFEST_DIR / f"evaluation_{args.method}_{timestamp}.json"
    path.write_text(json.dumps({"status": "complete", "runs": plan}, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
