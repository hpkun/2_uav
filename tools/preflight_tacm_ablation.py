"""Validate the frozen four-method TACM paper ablation before long training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import load_environment_config
from tools.tacm_ablation_protocol import (
    CHECKPOINT_INTERVAL, COMMON_TRAINING_FIELDS, ENV_CONFIG, EVAL_INTERVAL,
    LOG_INTERVAL, METHODS, NUM_ENVS, PAPER_SEEDS, TOTAL_STEPS, load_method_training,
)


def _semantic_flags(trainer: HAPPOTrainer) -> dict[str, bool]:
    return {
        "role": bool(trainer.role_guided_enabled),
        "mode": bool(trainer.dbm_enabled),
        "context": bool(trainer.tacm_enabled),
        "temporal": bool(trainer.tacm_enabled and float(trainer.config["tacm_temporal_coef"]) > 0.0),
    }


def validate_protocol(
    training_configs: Mapping[str, Mapping[str, Any]] | None = None,
    *, inspect_structures: bool = True,
    environment_config: str | Path | Mapping[str, Any] = ENV_CONFIG,
) -> dict[str, Any]:
    configs = {method: dict(value) for method, value in (
        training_configs or {method: load_method_training(method) for method in METHODS}
    ).items()}
    differences: list[dict[str, Any]] = []
    baseline = configs["happo"]
    for method, config in configs.items():
        for field in COMMON_TRAINING_FIELDS:
            if config.get(field) != baseline.get(field):
                differences.append({"method": method, "field": field,
                                    "expected": baseline.get(field), "actual": config.get(field)})
        if config.get("method_variant") != METHODS[method]["method_variant"]:
            differences.append({"method": method, "field": "method_variant",
                                "expected": METHODS[method]["method_variant"],
                                "actual": config.get("method_variant")})
    env = load_environment_config(environment_config)
    if env["environment_version"] != "heterogeneous_mavuav_4v4_v3_10":
        differences.append({"method": "all", "field": "environment_version",
                            "expected": "heterogeneous_mavuav_4v4_v3_10",
                            "actual": env["environment_version"]})

    no_temporal = configs["no_temporal"]
    full = configs["full"]
    tacm_differences: list[dict[str, Any]] = []
    for field in sorted((set(no_temporal) | set(full)) - {"tacm_temporal_coef"}):
        if no_temporal.get(field) != full.get(field):
            item = {"field": field, "no_temporal": no_temporal.get(field),
                    "full": full.get(field)}
            tacm_differences.append(item)
            differences.append({"method": "no_temporal_vs_full", **item})
    coefficient_contract = {
        "no_temporal": no_temporal.get("tacm_temporal_coef"),
        "full": full.get("tacm_temporal_coef"),
    }
    if coefficient_contract != {"no_temporal": 0.0, "full": 0.01}:
        item = {"field": "tacm_temporal_coef",
                "expected": {"no_temporal": 0.0, "full": 0.01},
                "actual": coefficient_contract}
        tacm_differences.append(item)
        differences.append({"method": "no_temporal_vs_full", **item})

    structures: dict[str, Any] = {}
    trainers: dict[str, HAPPOTrainer] = {}
    if inspect_structures:
        try:
            for method, declared in configs.items():
                probe = dict(declared)
                probe.update({"device": "cpu", "num_envs": 1, "seed": PAPER_SEEDS[0]})
                trainer = HAPPOTrainer(env, probe)
                trainers[method] = trainer
                actor_keys = tuple(trainer.actors.state_dict().keys())
                structures[method] = {
                    "semantics": _semantic_flags(trainer),
                    "actor_parameter_count": trainer.actor_parameter_counts["total"],
                    "actor_state_keys": actor_keys,
                    "role_critics_present": bool(trainer.role_guided_enabled),
                    "router_present": any("router" in key for key in actor_keys),
                    "experts_present": any("experts" in key for key in actor_keys),
                    "critic_architecture": trainer.critic_architecture,
                }
                if structures[method]["semantics"] != METHODS[method]["semantics"]:
                    differences.append({"method": method, "field": "structure_semantics",
                                        "expected": METHODS[method]["semantics"],
                                        "actual": structures[method]["semantics"]})
            left, right = structures["no_temporal"], structures["full"]
            for field in ("actor_parameter_count", "actor_state_keys", "critic_architecture"):
                if left[field] != right[field]:
                    differences.append({"method": "no_temporal", "field": field,
                                        "expected": right[field], "actual": left[field]})
        finally:
            for trainer in trainers.values():
                trainer.close()

    runtime = {
        "training_seeds": list(PAPER_SEEDS), "total_steps": TOTAL_STEPS,
        "num_envs": NUM_ENVS, "profile": "main", "device": "cuda",
        "checkpoint_interval": CHECKPOINT_INTERVAL, "log_interval": LOG_INTERVAL,
        "eval_interval": EVAL_INTERVAL,
    }
    return {
        "status": "PASS" if not differences else "FAIL",
        "environment_version": env["environment_version"],
        "common_training_fields": {field: baseline.get(field) for field in COMMON_TRAINING_FIELDS},
        "no_temporal_vs_full_contract": {
            "status": "PASS" if not tacm_differences else "FAIL",
            "allowed_difference": "tacm_temporal_coef",
            "required_values": {"no_temporal": 0.0, "full": 0.01},
            "differences": tacm_differences,
        },
        "runtime_contract": runtime, "structures": structures, "differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_protocol()
    text = json.dumps(report, indent=2, ensure_ascii=False, default=list)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
