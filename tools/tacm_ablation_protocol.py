"""Frozen protocol constants and resolvers for the final TACM paper ablation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER_SEEDS = (17, 23, 31)
TOTAL_STEPS = 2_000_000
NUM_ENVS = 16
CHECKPOINT_INTERVAL = 250_000
LOG_INTERVAL = 100_000
EVAL_INTERVAL = 0
EVALUATION_EPISODES = 200
EVALUATION_ENV_SEED_START = 12_000
EVALUATION_ACTION_SEED_START = 13_000
ENV_CONFIG = ROOT / "configs" / "env_v310.yaml"
MANIFEST_DIR = ROOT / "outputs" / "ablation_manifests"

METHODS: dict[str, dict[str, Any]] = {
    "happo": {
        "paper_variant": "HAPPO",
        "entrypoint": ROOT / "algorithm" / "train_happo.py",
        "config": ROOT / "configs" / "happo_v310_ablation.yaml",
        "method_variant": "baseline",
        "semantics": {"role": False, "mode": False, "context": False, "temporal": False},
    },
    "no_mode": {
        "paper_variant": "TACM w/o Mode",
        "entrypoint": ROOT / "algorithm" / "train_happo_rgaa.py",
        "config": ROOT / "configs" / "happo_rgaa_v310_curriculum.yaml",
        "method_variant": "rgaa",
        "semantics": {"role": True, "mode": False, "context": False, "temporal": False},
    },
    "no_temporal": {
        "paper_variant": "TACM w/o Temporal",
        "entrypoint": ROOT / "algorithm" / "train_tacm_rgaa.py",
        "config": ROOT / "configs" / "happo_tacm_rgaa_v310_no_temporal.yaml",
        "method_variant": "tacm_rgaa",
        "semantics": {"role": True, "mode": True, "context": True, "temporal": False},
    },
    "full": {
        "paper_variant": "Full TACM",
        "entrypoint": ROOT / "algorithm" / "train_tacm_rgaa.py",
        "config": ROOT / "configs" / "happo_tacm_rgaa_v310.yaml",
        "method_variant": "tacm_rgaa",
        "semantics": {"role": True, "mode": True, "context": True, "temporal": True},
    },
}

COMMON_TRAINING_FIELDS = (
    "environment_profile", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm",
    "hidden_dim", "actor_log_std_init", "critic_variant",
    "randomization_curriculum_enabled", "curriculum_start_profile",
    "curriculum_end_profile", "curriculum_steps",
)


def load_method_training(method: str) -> dict[str, Any]:
    spec = METHODS[method]
    with Path(spec["config"]).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    training = deepcopy(document["training"])
    training.update({
        "environment_profile": "main", "num_envs": NUM_ENVS,
        "actor_variant": "vanilla", "critic_variant": "mlp",
        "method_variant": spec["method_variant"],
    })
    return training


def training_command(method: str, seed: int, output_name: str, python: str = "python") -> list[str]:
    spec = METHODS[method]
    return [
        python, "-u", str(Path(spec["entrypoint"]).relative_to(ROOT)),
        "--steps", str(TOTAL_STEPS), "--profile", "main", "--seed", str(seed),
        "--device", "cuda", "--num-envs", str(NUM_ENVS),
        "--config", str(Path(spec["config"]).relative_to(ROOT)),
        "--env-config", str(ENV_CONFIG.relative_to(ROOT)),
        "--output-name", output_name,
        "--checkpoint-interval", str(CHECKPOINT_INTERVAL),
        "--eval-interval", str(EVAL_INTERVAL), "--log-interval", str(LOG_INTERVAL),
        "--final-eval-episodes", "1",
    ]


def evaluation_command(checkpoint: Path, python: str = "python") -> list[str]:
    return [
        python, "-u", "algorithm/evaluate_happo.py", str(checkpoint),
        "--profile", "main", "--episodes", str(EVALUATION_EPISODES),
        "--device", "cuda", "--env-config", str(ENV_CONFIG.relative_to(ROOT)),
        "--action-mode", "stochastic", "--action-seed", str(EVALUATION_ACTION_SEED_START),
        "--env-seed-start", str(EVALUATION_ENV_SEED_START),
    ]
