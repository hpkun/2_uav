"""Validate fair role-guided curriculum configs and method entrypoints."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V39_CONFIGS = {
    "rgaa": ROOT / "configs/happo_rgaa_v39_curriculum.yaml",
    "rgaa_wide": ROOT / "configs/happo_rgaa_wide_v39_curriculum.yaml",
    "dbm_rgaa": ROOT / "configs/happo_dbm_rgaa_v39_curriculum.yaml",
    "tacm_rgaa": ROOT / "configs/happo_tacm_rgaa_v39.yaml",
}
V310_CONFIGS = {
    "rgaa": ROOT / "configs/happo_rgaa_v310_curriculum.yaml",
    "rgaa_wide": ROOT / "configs/happo_rgaa_wide_v310_curriculum.yaml",
    "dbm_rgaa": ROOT / "configs/happo_dbm_rgaa_v310_curriculum.yaml",
    "tacm_rgaa": ROOT / "configs/happo_tacm_rgaa_v310.yaml",
}
CONFIG_GROUPS = {"v3_9": V39_CONFIGS, "v3_10": V310_CONFIGS}
CONFIGS = V39_CONFIGS  # Backward-compatible import for existing audit tests.
ENTRYPOINTS = {
    "rgaa": ROOT / "algorithm/train_happo_rgaa.py",
    "rgaa_wide": ROOT / "algorithm/train_rgaa_wide.py",
    "dbm_rgaa": ROOT / "algorithm/train_dbm_rgaa.py",
    "tacm_rgaa": ROOT / "algorithm/train_tacm_rgaa.py",
}
ENVIRONMENT_CONFIGS = {
    "v3_9": ROOT / "configs/env_v39.yaml",
    "v3_10": ROOT / "configs/env_v310.yaml",
}
COMMON_FIELDS = (
    "environment_profile", "device", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm",
    "actor_log_std_init", "role_advantage_coef", "role_aux_reward_mode",
    "randomization_curriculum_enabled", "curriculum_start_profile",
    "curriculum_end_profile", "curriculum_steps", "final_evaluation_role",
)
EXPECTED_METHODS = {name: name for name in ENTRYPOINTS}


def load_training(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("training"), dict):
        raise RuntimeError(f"invalid training config: {path}")
    return payload["training"]


def entrypoint_method(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "main":
            for keyword in node.keywords:
                if keyword.arg == "method_variant" and isinstance(keyword.value, ast.Constant):
                    values.append(keyword.value.value)
    if len(values) != 1 or not isinstance(values[0], str):
        raise RuntimeError(f"cannot resolve exactly one method_variant from entrypoint: {path}")
    return values[0]


def validate_entrypoints() -> dict[str, Path]:
    for method, path in ENTRYPOINTS.items():
        actual = entrypoint_method(path)
        if actual != method:
            raise RuntimeError(
                f"entrypoint method mismatch: expected {method!r}, got {actual!r} from {path}"
            )
    return dict(ENTRYPOINTS)


def validate_configs(environment_version: str = "v3_9") -> dict[str, dict[str, Any]]:
    if environment_version not in CONFIG_GROUPS:
        raise ValueError(f"unsupported environment version selector: {environment_version}")
    config_paths = CONFIG_GROUPS[environment_version]
    loaded = {name: load_training(path) for name, path in config_paths.items()}
    reference = loaded["tacm_rgaa"]
    for name, config in loaded.items():
        for field in COMMON_FIELDS:
            if config.get(field) != reference.get(field):
                raise RuntimeError(
                    f"curriculum fairness mismatch: {name}.{field}={config.get(field)!r}, "
                    f"TACM={reference.get(field)!r}"
                )
        if config.get("method_variant") != EXPECTED_METHODS[name]:
            raise RuntimeError(f"wrong method_variant in {config_paths[name]}")
        if config.get("environment_profile") != "main" or config.get("device") != "cuda":
            raise RuntimeError(f"invalid main/CUDA contract in {config_paths[name]}")
    if loaded["rgaa"].get("hidden_dim") != 128:
        raise RuntimeError("RGAA hidden_dim contract mismatch")
    if loaded["rgaa_wide"].get("uav_actor_hidden_dim") != 131:
        raise RuntimeError("RGAA-Wide UAV width contract mismatch")
    dbm = loaded["dbm_rgaa"]
    expected_dbm = {
        "dbm_role_count": 2, "dbm_residual_scale": 0.25, "dbm_init_scale": 0.01,
        "dbm_initialization_semantics": "zero_router_antisymmetric_experts_v1",
    }
    for field, value in expected_dbm.items():
        if dbm.get(field) != value:
            raise RuntimeError(f"DBM contract mismatch: {field}")
    validate_entrypoints()
    from env.mavuav import load_environment_config
    resolved_environment = load_environment_config(ENVIRONMENT_CONFIGS[environment_version])
    expected_environment = f"heterogeneous_mavuav_4v4_{environment_version}"
    if resolved_environment["environment_version"] != expected_environment:
        raise RuntimeError("environment config/version mismatch")
    return loaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-version", choices=tuple(CONFIG_GROUPS), default="v3_9")
    args = parser.parse_args()
    validated = validate_configs(args.environment_version)
    print(
        f"Validated fair {args.environment_version} curriculum configs and entrypoints: "
        + ", ".join(validated), flush=True,
    )
