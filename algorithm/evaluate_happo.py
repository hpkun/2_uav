"""Evaluate a HAPPO checkpoint without resuming training."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
from typing import Any

import numpy as np
import torch

from algorithm.happo.evaluation import evaluate_actors, evaluate_recurrent_actors, summarize_records
from algorithm.happo.eram import (EntityIndependentActors, EntityAttentionRecurrentCritic,
                                  actor_kwargs, critic_kwargs, validate_eram_metadata)
from algorithm.happo.networks import IndependentActors
from algorithm.happo.dbm_rgaa import (
    DBM_RGAA_METHOD, RGAA_WIDE_METHOD, build_method_actors, dbm_metadata, wide_metadata,
)
from algorithm.happo.tacm_rgaa import TACM_RGAA_METHOD, tacm_metadata
from algorithm.happo.relational_critic import RelationalCentralizedCritic
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, ROLE_REWARD_MODES, load_environment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument(
        "--action-mode", choices=("deterministic", "stochastic"), default="deterministic",
    )
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--env-seed-start", type=int, default=1000)
    return parser.parse_args()


def _resolved_device(requested: str) -> str:
    return "cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested


def validate_checkpoint_contract(payload: dict[str, Any], env_config: dict[str, Any]) -> None:
    """Validate checkpoint metadata against the actually resolved evaluation environment."""
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    expected = (env_config["environment_version"], OBS_DIM, GLOBAL_STATE_DIM)
    if actual != expected:
        raise RuntimeError("incompatible HAPPO checkpoint environment contract")
    version = env_config["environment_version"]
    if version.endswith(("v3_10", "v3_11", "v3_12", "v3_13")):
        expected_capability = {
            "mav_direct_attack_capability": False,
            "mav_direct_attack_shaping": "none",
            "mav_receives_shared_team_kill_reward": True,
        }
        for field, value in expected_capability.items():
            if payload.get(field) != value:
                raise RuntimeError(f"incompatible HAPPO checkpoint combat capability: {field}")
    env_shaping = env_config.get("shaping", {})
    env_mode = ("heterogeneous_role_v1" if version.endswith("v3_7") else
                "heterogeneous_role_coupled_v1" if version.endswith("v3_8") else
                "heterogeneous_role_coupled_gate_v1" if version.endswith(("v3_9", "v3_10", "v3_11", "v3_12", "v3_13")) else
                str(env_shaping.get("mode", "absolute")))
    checkpoint_mode = str(payload.get("reward_mode", payload.get("reward_shaping_mode", "absolute")))
    if checkpoint_mode != env_mode:
        raise RuntimeError("incompatible HAPPO checkpoint reward mode")
    if env_mode == "potential":
        checkpoint_gamma = float(payload.get("shaping_gamma", float("nan")))
        if not np.isfinite(checkpoint_gamma) or not np.isclose(
            checkpoint_gamma, float(env_shaping["gamma"]), rtol=0.0, atol=1e-12,
        ):
            raise RuntimeError("incompatible HAPPO checkpoint shaping gamma")
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    method_variant = payload.get(
        "method_variant", trainer_config.get("method_variant", "baseline"),
    )
    if version == "heterogeneous_mavuav_4v4_v3_13":
        variants = (payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla")),
                    payload.get("critic_variant", trainer_config.get("critic_variant", "mlp")), method_variant)
        if variants != ("vanilla", "mlp", "baseline") or payload.get("weapon_engagement_mode") != "single_target_lock":
            raise RuntimeError("incompatible v3.13 vanilla single-target weapon contract")
    if method_variant in (DBM_RGAA_METHOD, RGAA_WIDE_METHOD, TACM_RGAA_METHOD):
        if "environment_config" not in payload:
            raise RuntimeError(
                f"{method_variant} checkpoint is missing resolved environment_config"
            )
        checkpoint_environment = load_environment_config(payload["environment_config"])
        if checkpoint_environment != env_config:
            raise RuntimeError(
                "evaluation resolved environment config differs from checkpoint"
            )


def main(expected_critic_variant: str = "mlp") -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = _resolved_device(args.device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if args.env_config:
        env_config: dict[str, Any] = load_environment_config(args.env_config.expanduser().resolve())
    elif "environment_config" in payload:
        env_config = load_environment_config(payload["environment_config"])
    else:
        env_config = load_environment_config(None)
    validate_checkpoint_contract(payload, env_config)
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    actor_variant = payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla"))
    is_eram = actor_variant == "entity_recurrent"
    if actor_variant != "vanilla" and not is_eram:
        raise RuntimeError("incompatible actor architecture: vanilla evaluator requires a vanilla checkpoint")
    method_variant = payload.get("method_variant", trainer_config.get("method_variant", "baseline"))
    if method_variant not in (
        "baseline", "agp", "cf_happo", "rdc_happo", "rgaa", "cr_rgaa", "lp_cr_rgaa",
        "ls_rgaa", "lsa_rgaa", DBM_RGAA_METHOD, RGAA_WIDE_METHOD, TACM_RGAA_METHOD,
    ):
        raise RuntimeError(f"unsupported HAPPO method_variant: {method_variant!r}")
    critic_variant = payload.get("critic_variant", trainer_config.get("critic_variant", "mlp"))
    if is_eram and expected_critic_variant == "mlp":
        expected_critic_variant = "entity_attention_recurrent"
    if critic_variant != expected_critic_variant:
        raise RuntimeError(
            f"incompatible critic variant: evaluator requires {expected_critic_variant!r}, "
            f"checkpoint contains {critic_variant!r}"
        )
    critic_architecture = payload.get("critic_architecture")
    if is_eram:
        if (method_variant != "baseline" or env_config["environment_version"] != "heterogeneous_mavuav_4v4_v3_11"
                or env_config.get("role_reward", {}).get("mode") != "heterogeneous_role_coupled_gate_v1"):
            raise RuntimeError("ERAM evaluator requires baseline v3.11 coupled-gate reward")
        if load_environment_config(payload["environment_config"]) != env_config:
            raise RuntimeError("ERAM evaluation environment differs from checkpoint")
        validate_eram_metadata(payload, trainer_config)
    if critic_variant == "relational":
        if method_variant != "baseline":
            raise RuntimeError("RC-HAPPO evaluator requires method_variant='baseline'")
        if critic_architecture != RelationalCentralizedCritic.architecture():
            raise RuntimeError("incompatible relational critic architecture metadata")
    if is_eram:
        actors = EntityIndependentActors(**actor_kwargs(trainer_config)).to(device)
        if payload.get("actor_architecture") != actors.actors[0].architecture():
            raise RuntimeError("incompatible ERAM actor architecture")
        # Critic is validated, never used for action generation.
        with torch.random.fork_rng(devices=[]):
            expected_architecture = EntityAttentionRecurrentCritic(**critic_kwargs(trainer_config)).architecture()
        if critic_architecture != expected_architecture:
            raise RuntimeError("incompatible ERAM critic architecture")
    elif method_variant in (DBM_RGAA_METHOD, RGAA_WIDE_METHOD, TACM_RGAA_METHOD):
        contract_config = dict(trainer_config)
        contract_config["method_variant"] = (
            DBM_RGAA_METHOD if method_variant == TACM_RGAA_METHOD else method_variant
        )
        expected_method_metadata = (
            dbm_metadata(contract_config)
            if method_variant in (DBM_RGAA_METHOD, TACM_RGAA_METHOD)
            else wide_metadata(contract_config)
        )
        for field, expected_value in expected_method_metadata.items():
            if payload.get(field) != expected_value:
                raise RuntimeError(
                    f"incompatible {method_variant} checkpoint contract: {field}"
                )
        if method_variant == TACM_RGAA_METHOD:
            for field, expected_value in tacm_metadata(trainer_config).items():
                if payload.get(field) != expected_value:
                    raise RuntimeError(f"incompatible TACM-RGAA checkpoint contract: {field}")
        actors = build_method_actors(
            method_variant=(DBM_RGAA_METHOD if method_variant == TACM_RGAA_METHOD else method_variant),
            training_seed=int(trainer_config["seed"]),
            hidden_dim=int(trainer_config["hidden_dim"]),
            log_std_init=float(trainer_config.get("actor_log_std_init", -0.5)),
            role_module_enabled=bool(trainer_config.get("role_module_enabled", True)),
            dbm_role_count=int(trainer_config.get("dbm_role_count", 2)),
            dbm_residual_scale=float(trainer_config.get("dbm_residual_scale", 0.25)),
            dbm_expert_init_scale=float(trainer_config.get("dbm_init_scale", 0.01)),
            uav_actor_hidden_dim=int(trainer_config.get("uav_actor_hidden_dim", 131)),
        ).to(device)
    else:
        actors = IndependentActors(
            hidden_dim=int(trainer_config["hidden_dim"]),
            log_std_init=float(trainer_config.get("actor_log_std_init", -0.5)),
        ).to(device)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    version = env_config["environment_version"]
    reward_mode = ("heterogeneous_role_v1" if version.endswith("v3_7") else
                   "heterogeneous_role_coupled_v1" if version.endswith("v3_8") else
                   "heterogeneous_role_coupled_gate_v1" if version.endswith(("v3_9", "v3_10", "v3_11", "v3_12", "v3_13")) else
                   str(env_config.get("shaping", {}).get("mode", "absolute")))
    training_profile = str(payload["environment_profile"])
    rows = []
    algorithm = (
        "eram_happo" if is_eram else
        "rc_happo" if critic_variant == "relational" else
        "lp_cr_rgaa_happo" if method_variant == "lp_cr_rgaa" else
        "ls_rgaa_happo" if method_variant == "ls_rgaa" else
        "lsa_rgaa_happo" if method_variant == "lsa_rgaa" else
        "cr_rgaa_happo" if method_variant == "cr_rgaa" else
        "rgaa_happo" if method_variant == "rgaa" else
        "tacm_rgaa_happo" if method_variant == TACM_RGAA_METHOD else
        "dbm_rgaa_happo" if method_variant == DBM_RGAA_METHOD else
        "rgaa_wide_happo" if method_variant == RGAA_WIDE_METHOD else
        method_variant if method_variant in ("cf_happo", "rdc_happo") else
        "happo_agp" if method_variant == "agp" else "happo"
    )
    deterministic = args.action_mode == "deterministic"
    effective_action_seed = None if deterministic else int(args.action_seed)
    evaluator = evaluate_recurrent_actors if is_eram else evaluate_actors
    records = evaluator(
        actors, env_config, args.episodes, args.profile, seed=args.env_seed_start, device=device,
        deterministic=deterministic,
        action_seed=effective_action_seed,
    )
    rows.append({
            "checkpoint": checkpoint.name, "sampled_steps": int(payload.get("sampled_steps", 0)),
            "algorithm": algorithm,
            **({"actor_variant": actor_variant, "base_algorithm": "happo"} if is_eram else {}),
            "method_variant": method_variant,
            "critic_variant": critic_variant,
            "blue_target_strategy": env_config["blue_policy"]["target_strategy"], "training_profile": training_profile,
            "evaluation_profile": args.profile,
            "evaluation_environment_seed_start": int(args.env_seed_start),
            "episodes": args.episodes, "evaluation_episodes": args.episodes,
            "training_seed": int(trainer_config.get("seed", 0)),
            "action_mode": args.action_mode,
            "configured_action_seed": int(args.action_seed),
            "effective_action_seed": effective_action_seed,
            "action_seed": effective_action_seed,
            "environment_version": env_config["environment_version"],
            **({"weapon_engagement_mode": env_config["combat"]["weapon_engagement_mode"]}
               if version.endswith("v3_13") else {}),
            "reward_mode": reward_mode,
            "reward_shaping_mode": reward_mode if reward_mode not in ROLE_REWARD_MODES else None,
            "shaping_gamma": float(env_config.get("shaping", {}).get("gamma", 0.0)) if reward_mode not in ROLE_REWARD_MODES else None,
            "training_gamma": float(trainer_config.get("gamma", 0.99)),
            **summarize_records(records),
        })
    label = checkpoint.stem.removeprefix("checkpoint_")
    suffix = "" if deterministic else "_stochastic"
    csv_path = checkpoint.parent / f"evaluation_{label}{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    summary_path = checkpoint.parent / f"evaluation_{label}{suffix}_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump({
            "algorithm": algorithm, "checkpoint": str(checkpoint), "training_profile": training_profile,
            **({"actor_variant": actor_variant, "base_algorithm": "happo",
                "actor_architecture": payload["actor_architecture"]} if is_eram else {}),
            "evaluation_profile": args.profile, "method_variant": method_variant,
            "evaluation_environment_seed_start": int(args.env_seed_start),
            "evaluation_episodes": args.episodes,
            "training_seed": int(trainer_config.get("seed", 0)),
            "action_mode": args.action_mode,
            "configured_action_seed": int(args.action_seed),
            "effective_action_seed": effective_action_seed,
            "action_seed": effective_action_seed,
            "environment_version": env_config["environment_version"],
            **({"weapon_engagement_mode": env_config["combat"]["weapon_engagement_mode"],
                "blue_target_strategy": env_config["blue_policy"]["target_strategy"]}
               if version.endswith("v3_13") else {}),
            "reward_mode": reward_mode,
            "reward_shaping_mode": reward_mode if reward_mode not in ROLE_REWARD_MODES else None,
            "shaping_gamma": float(env_config.get("shaping", {}).get("gamma", 0.0)) if reward_mode not in ROLE_REWARD_MODES else None,
            "training_gamma": float(trainer_config.get("gamma", 0.99)),
            "critic_variant": critic_variant, "critic_architecture": critic_architecture,
            "device": device, "results": rows,
        }, stream, indent=2, ensure_ascii=False)
    print({"evaluation_csv": str(csv_path), "summary_json": str(summary_path), "results": rows}, flush=True)


if __name__ == "__main__":
    main()
