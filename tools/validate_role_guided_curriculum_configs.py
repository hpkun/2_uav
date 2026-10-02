"""Validate fair v3.9 role-guided curriculum experiment configurations."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = {
    "rgaa": ROOT / "configs/happo_rgaa_v39_curriculum.yaml",
    "rgaa_wide": ROOT / "configs/happo_rgaa_wide_v39_curriculum.yaml",
    "dbm_rgaa": ROOT / "configs/happo_dbm_rgaa_v39_curriculum.yaml",
    "tacm_rgaa": ROOT / "configs/happo_tacm_rgaa_v39.yaml",
}
COMMON_FIELDS = (
    "environment_profile", "device", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm",
    "actor_log_std_init", "role_advantage_coef", "role_aux_reward_mode",
    "randomization_curriculum_enabled", "curriculum_start_profile",
    "curriculum_end_profile", "curriculum_steps", "final_evaluation_role",
)
EXPECTED_METHODS = {name: name for name in CONFIGS}


def load_training(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("training"), dict):
        raise RuntimeError(f"invalid training config: {path}")
    return payload["training"]


def validate_configs() -> dict[str, dict[str, Any]]:
    loaded = {name: load_training(path) for name, path in CONFIGS.items()}
    reference = loaded["tacm_rgaa"]
    for name, config in loaded.items():
        for field in COMMON_FIELDS:
            if config.get(field) != reference.get(field):
                raise RuntimeError(
                    f"curriculum fairness mismatch: {name}.{field}={config.get(field)!r}, "
                    f"TACM={reference.get(field)!r}"
                )
        if config.get("method_variant") != EXPECTED_METHODS[name]:
            raise RuntimeError(f"wrong method_variant in {CONFIGS[name]}")
        if config.get("environment_profile") != "main" or config.get("device") != "cuda":
            raise RuntimeError(f"invalid main/CUDA contract in {CONFIGS[name]}")
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
    return loaded


if __name__ == "__main__":
    validated = validate_configs()
    print("Validated fair curriculum configs: " + ", ".join(validated), flush=True)
