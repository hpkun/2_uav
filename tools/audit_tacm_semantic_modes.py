"""Read-only semantic/mechanism audit for frozen TACM-RGAA-v1 checkpoints.

The audit intentionally performs ordinary policy execution once per decision
step.  All teacher, intervention, geometry, and event diagnostics are
deterministic side computations and never feed back into the environment or
the action sampler.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.happo.dbm_rgaa import DBMGaussianActor
from algorithm.happo.tacm_rgaa import (
    TACM_MODE_LABELS,
    TACM_RGAA_METHOD,
    context_distillation_loss,
    tactical_teacher,
    temporal_router_loss,
)
from env.geometry import compute_pairwise_geometry
from env.mavuav import (
    BLUE_IDS,
    RED_IDS,
    UNARMED_MAV_ENVIRONMENT_VERSION,
    HeterogeneousMAVUAVAirCombatEnv,
    load_environment_config,
)
from env.reward_role_v39 import attack_gate_indicator
from tools.audit_continuation_horizon import load_tacm_checkpoint


UAV_IDS = RED_IDS[1:]
EXPECTED_VERSION = UNARMED_MAV_ENVIRONMENT_VERSION
OUTPUT_FILES = {
    "summary": "semantic_audit_summary.json",
    "steps": "semantic_step_records.csv",
    "episodes": "semantic_episode_summary.csv",
    "interventions": "semantic_mode_intervention.csv",
    "events": "semantic_event_analysis.csv",
    "last_blue": "semantic_last_blue_analysis.csv",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_finite_or_none(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] = ()) -> None:
    fields = list(fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_semantic_audit_contract(
    loaded: Mapping[str, Any], environment_config: str | Path | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require the exact frozen v3.10 TACM contract and an exact env config."""
    payload = loaded["payload"]
    checkpoint_config = loaded["environment_config"]
    if checkpoint_config["environment_version"] != EXPECTED_VERSION:
        raise RuntimeError("TACM semantic-mode audit requires the exact v3.10 environment")
    if payload.get("algorithm") != "tacm_rgaa_happo":
        raise RuntimeError("TACM semantic-mode audit requires algorithm='tacm_rgaa_happo'")
    trainer = payload.get("trainer_config", payload.get("config", {}))
    if payload.get("method_variant", trainer.get("method_variant")) != TACM_RGAA_METHOD:
        raise RuntimeError("TACM semantic-mode audit requires method_variant='tacm_rgaa'")
    if int(payload.get("sampled_steps", 0)) != 2_000_000:
        raise RuntimeError("TACM semantic-mode audit requires an exact 2,000,000-step checkpoint")
    if bool(checkpoint_config["combat"].get("mav_can_attack", True)):
        raise RuntimeError("v3.10 semantic audit requires an unarmed MAV")
    if environment_config is not None:
        supplied = load_environment_config(environment_config)
        if supplied != checkpoint_config:
            raise RuntimeError("audit environment config must exactly match checkpoint environment_config")
    return checkpoint_config


def static_contract_audit() -> dict[str, Any]:
    """Machine-readable audit of the frozen implementation (not runtime claims)."""
    return {
        "platform_role_auxiliary_advantage": {
            "status": "confirmed",
            "actor_advantage": "Norm(A_team_i) + role_advantage_coef * Norm(A_aux_i)",
            "role_advantage_coef": 0.5,
            "auxiliary_reward": "agent process reward + same-agent own-loss event",
            "excluded_from_auxiliary": ["shared kill", "shared terminal", "shared safety", "other-agent loss"],
            "critic_sharing": {"MAV": "independent", "UAV1-UAV3": "one shared network"},
            "evidence": [
                "algorithm/happo/rgaa.py:34-64,109-172",
                "algorithm/happo/trainer.py:438-452,1837-1911",
            ],
        },
        "uav_mode_actor": {
            "status": "confirmed",
            "mean": "base_mean + 0.25 * sum_k(router_probability_k * tanh(expert_k(hidden)))",
            "router_output_dim": 2,
            "expert_count": 2,
            "private_per_uav": True,
            "uav_actor_parameter_sharing": False,
            "mav_actor": "ordinary independent GaussianActor",
            "log_std": "one learned 3D parameter per actor, clamp[-5,2] only when constructing Normal",
            "initialization": "zero router logits; expert2=-expert1; uniform mixture cancels exactly",
            "evidence": ["algorithm/happo/dbm_rgaa.py:40-188", "algorithm/common/networks.py:10-35"],
        },
        "tactical_teacher": {
            "status": "confirmed_no_invisible_blue_geometry_used",
            "state_source": "pre-action centralized global state",
            "visibility": "alive Blue retained only when at least one alive Red is within its configured sensor range",
            "engagement_readiness": "mean(angle_quality, distance_quality, normalized attack streak, exact gate)",
            "mav_threat": "maximum readiness among visible Blue->MAV pairs",
            "interception_suitability": "mean(angle_quality, distance_quality) for UAV->most-threatening-Blue",
            "assignment": "softmax(suitability/tau_group) over alive UAVs",
            "responsibility": "mav_threat * assignment",
            "engagement_modulation": "engagement_readiness * (1-responsibility)",
            "teacher_probability": "softmax([adjusted engagement, responsibility]/tau_teacher)",
            "confidence": "1 - binary_entropy(q)/log(2)",
            "evidence": ["algorithm/happo/tacm_rgaa.py:27-111", "env/mavuav.py:711-722"],
        },
        "context_distillation": {
            "status": "confirmed",
            "loss": "sum(confidence * KL(q_teacher || p_router))/max(sum(confidence),1)",
            "teacher_detached": True,
            "encoder_hidden_detached": True,
            "direct_gradient_targets": ["router"],
            "no_direct_gradient": ["encoder", "base_head", "expert1", "expert2", "log_std"],
            "evidence": ["algorithm/happo/tacm_rgaa.py:114-124"],
        },
        "temporal_consistency": {
            "status": "confirmed",
            "loss": "confidence-weighted squared L2 between adjacent router distributions (not KL)",
            "valid_pair": [
                "active_t", "active_t+1", "not terminated_t", "not truncated_t",
                "no kill/death event_t", "same engagement target", "same MAV threat target",
                "same teacher dominant mode",
            ],
            "weight": "min(confidence_t, confidence_t+1)",
            "previous_router_stop_gradient": True,
            "direct_gradient_targets": ["router at t+1"],
            "order": "PPO+context -> temporal router-only -> recompute log probability -> HAPPO factor",
            "evidence": ["algorithm/happo/tacm_rgaa.py:127-141", "algorithm/happo/trainer.py:1956-2017"],
        },
        "decentralized_execution": {
            "status": "confirmed",
            "execution_inputs": "each actor receives only its own 100D local observation",
            "training_only": ["tactical teacher", "global state", "team critic", "role critics", "temporal targets"],
            "evidence": ["algorithm/happo/evaluation.py:42-78", "algorithm/evaluate_happo.py:100-167"],
        },
        "v3_10_combat": {
            "status": "confirmed",
            "mav_can_attack": False,
            "attack_capable_red": list(UAV_IDS),
            "evidence": ["env/mavuav.py:511-543", "configs/env_v310.yaml"],
        },
    }


def forced_mode_intervention(actor: DBMGaussianActor, observation: torch.Tensor) -> dict[str, np.ndarray | float]:
    """Compute same-observation mode interventions without sampling or mutation."""
    with torch.no_grad():
        details = actor.network.details(observation)
    base = details["base_mean"][0]
    probs = details["router_probabilities"][0]
    experts = details["expert_outputs"][0]
    rho = float(actor.network.residual_scale)
    natural = base + rho * (probs[:, None] * experts).sum(0)
    engagement = base + rho * experts[0]
    support = base + rho * experts[1]
    uniform = base + 0.5 * rho * (experts[0] + experts[1])
    output: dict[str, np.ndarray | float] = {
        "base_mean": base.cpu().numpy(),
        "router_probabilities": probs.cpu().numpy(),
        "expert_engagement": experts[0].cpu().numpy(),
        "expert_support": experts[1].cpu().numpy(),
        "natural_residual": (natural - base).cpu().numpy(),
        "natural_mean": natural.cpu().numpy(),
        "engagement_mean": engagement.cpu().numpy(),
        "support_mean": support.cpu().numpy(),
        "uniform_mean": uniform.cpu().numpy(),
        "natural_action": torch.tanh(natural).cpu().numpy(),
        "engagement_action": torch.tanh(engagement).cpu().numpy(),
        "support_action": torch.tanh(support).cpu().numpy(),
        "uniform_action": torch.tanh(uniform).cpu().numpy(),
    }
    output["expert_output_distance"] = float(torch.linalg.vector_norm(experts[0] - experts[1]).item())
    output["mean_mode_separation"] = float(torch.linalg.vector_norm(engagement - support).item())
    output["action_mode_separation"] = float(torch.linalg.vector_norm(torch.tanh(engagement) - torch.tanh(support)).item())
    output["natural_to_engagement_action"] = float(torch.linalg.vector_norm(torch.tanh(natural) - torch.tanh(engagement)).item())
    output["natural_to_support_action"] = float(torch.linalg.vector_norm(torch.tanh(natural) - torch.tanh(support)).item())
    output["residual_norm"] = float(torch.linalg.vector_norm(natural - base).item())
    output["base_norm"] = float(torch.linalg.vector_norm(base).item())
    output["residual_base_ratio"] = float(output["residual_norm"] / max(float(output["base_norm"]), 1e-12))
    return output


def diagnostic_rng_invariant(actor: DBMGaussianActor, observation: torch.Tensor) -> dict[str, np.ndarray | float]:
    """Assert that intervention diagnostics do not alter CPU/CUDA RNG state."""
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if observation.device.type == "cuda" else None
    result = forced_mode_intervention(actor, observation)
    if not torch.equal(cpu, torch.get_rng_state()):
        raise AssertionError("TACM intervention diagnostics changed the CPU Torch RNG")
    if cuda is not None:
        after = torch.cuda.get_rng_state_all()
        if len(cuda) != len(after) or any(not torch.equal(a, b) for a, b in zip(cuda, after)):
            raise AssertionError("TACM intervention diagnostics changed the CUDA RNG")
    return result


def _geometry(env: HeterogeneousMAVUAVAirCombatEnv, attacker: str, target: str | None) -> dict[str, Any]:
    if target is None or target not in env.entities:
        return {"distance": None, "ata_deg": None, "aa_deg": None, "gate": False, "streak": None}
    geometry = compute_pairwise_geometry(env.entities[attacker].state, env.entities[target].state)
    combat = env.config["combat"]
    gate = bool(attack_gate_indicator(
        geometry.distance, geometry.ata, geometry.aa,
        float(combat["distance"][0]), float(combat["distance"][1]),
        np.deg2rad(float(combat["ata_deg"])), np.deg2rad(float(combat["aa_deg"])),
    ))
    return {
        "distance": float(geometry.distance), "ata_deg": float(np.rad2deg(geometry.ata)),
        "aa_deg": float(np.rad2deg(geometry.aa)), "gate": gate,
        "streak": int(env._attack_streak.get((attacker, target), 0)),
    }


def _vec_fields(prefix: str, value: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{axis}": float(value[index]) for index, axis in enumerate(("ux", "uy", "uz"))}


def _intervention_row(base: Mapping[str, Any], intervention: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(base)
    for key in (
        "base_mean", "router_probabilities", "expert_engagement", "expert_support",
        "natural_residual", "natural_mean", "engagement_mean", "support_mean", "uniform_mean",
        "natural_action", "engagement_action", "support_action", "uniform_action",
    ):
        row[key] = _json(np.asarray(intervention[key]).tolist())
    for key in (
        "expert_output_distance", "mean_mode_separation", "action_mode_separation",
        "natural_to_engagement_action", "natural_to_support_action", "residual_norm",
        "base_norm", "residual_base_ratio",
    ):
        row[key] = float(intervention[key])
    eng_action = np.asarray(intervention["engagement_action"])
    sup_action = np.asarray(intervention["support_action"])
    row.update(_vec_fields("engagement_minus_support_action", eng_action - sup_action))
    return row


def _set_episode_action_seed(seed: int, device: str) -> None:
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def run_episode(
    loaded: Mapping[str, Any], *, episode: int, environment_seed: int,
    action_seed: int, action_mode: str, profile: str, device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run one ordinary episode and collect side-channel diagnostics."""
    config = loaded["environment_config"]
    env = HeterogeneousMAVUAVAirCombatEnv(config, profile=profile)
    observations, _ = env.reset(seed=environment_seed)
    deterministic = action_mode == "deterministic"
    if not deterministic:
        _set_episode_action_seed(action_seed, device)
    step_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    previous: dict[str, dict[str, Any]] = {}
    done = False
    final_info: dict[str, Any] = {}
    while not done:
        decision_step = int(env.step_count + 1)
        pre_kills = len(env._red_attack_kills)
        alive_red = [aid for aid in RED_IDS if env.entities[aid].state.alive]
        alive_blue = [aid for aid in BLUE_IDS if env.entities[aid].state.alive]
        teacher = tactical_teacher(
            env.global_state(), config,
            tau_group=float(loaded["payload"]["trainer_config"]["tacm_tau_group"]),
            tau_teacher=float(loaded["payload"]["trainer_config"]["tacm_tau_teacher"]),
        )
        actions: list[np.ndarray] = []
        current: dict[str, dict[str, Any]] = {}
        for agent_index, aid in enumerate(RED_IDS):
            actor = loaded["actors"].actors[agent_index]
            observation = torch.as_tensor(observations[aid], device=device).unsqueeze(0)
            if aid in UAV_IDS and env.entities[aid].state.alive:
                if not isinstance(actor, DBMGaussianActor):
                    raise TypeError("TACM UAV actor is not DBMGaussianActor")
                intervention = diagnostic_rng_invariant(actor, observation)
                ui = UAV_IDS.index(aid)
                engagement_target_index = int(teacher.engagement_target[0, ui])
                threat_target_index = int(teacher.threat_target[0])
                engagement_target = BLUE_IDS[engagement_target_index] if engagement_target_index >= 0 else None
                threat_target = BLUE_IDS[threat_target_index] if threat_target_index >= 0 else None
                eng_geometry = _geometry(env, aid, engagement_target)
                intercept_geometry = _geometry(env, aid, threat_target)
                threat_geometry = _geometry(env, threat_target, "MAV") if threat_target else _geometry(env, "MAV", None)
                p_teacher = teacher.probabilities[0, ui]
                p_router = np.asarray(intervention["router_probabilities"])
                entropy = float(-(p_router * np.log(np.maximum(p_router, 1e-12))).sum())
                base = {
                    "checkpoint": str(loaded["checkpoint"]), "training_seed": loaded["training_seed"],
                    "sampled_steps": loaded["sampled_steps"], "episode": episode,
                    "environment_seed": environment_seed, "action_seed": None if deterministic else action_seed,
                    "action_mode": action_mode, "decision_step": decision_step, "uav_id": aid,
                    "red_kills_pre": pre_kills, "alive_red_count": len(alive_red),
                    "alive_blue_count": len(alive_blue), "alive_red": ";".join(alive_red),
                    "alive_blue": ";".join(alive_blue),
                    "teacher_engagement_probability": float(p_teacher[0]),
                    "teacher_support_probability": float(p_teacher[1]),
                    "teacher_confidence": float(teacher.confidence[0, ui]),
                    "router_engagement_probability": float(p_router[0]),
                    "router_support_probability": float(p_router[1]),
                    "teacher_dominant_mode": int(np.argmax(p_teacher)),
                    "router_dominant_mode": int(np.argmax(p_router)),
                    "engagement_readiness": float(teacher.engagement_scores[0, ui]),
                    "support_responsibility": float(teacher.cover_responsibility[0, ui]),
                    "cover_assignment": float(teacher.cover_assignment[0, ui]),
                    "mav_threat": float(teacher.mav_threat[0]),
                    "engagement_target": engagement_target,
                    "threat_target": threat_target,
                    "router_entropy": entropy,
                    "engagement_distance": eng_geometry["distance"],
                    "engagement_ata_deg": eng_geometry["ata_deg"],
                    "engagement_aa_deg": eng_geometry["aa_deg"],
                    "engagement_exact_gate": eng_geometry["gate"],
                    "engagement_attack_streak": eng_geometry["streak"],
                    "intercept_distance": intercept_geometry["distance"],
                    "intercept_ata_deg": intercept_geometry["ata_deg"],
                    "intercept_aa_deg": intercept_geometry["aa_deg"],
                    "intercept_exact_gate": intercept_geometry["gate"],
                    "threat_to_mav_distance": threat_geometry["distance"],
                    "threat_to_mav_ata_deg": threat_geometry["ata_deg"],
                    "threat_to_mav_aa_deg": threat_geometry["aa_deg"],
                    "threat_to_mav_exact_gate": threat_geometry["gate"],
                    "threat_to_mav_attack_streak": threat_geometry["streak"],
                    "target_switch": previous.get(aid, {}).get("engagement_target") not in (None, engagement_target),
                    "threat_switch": previous.get(aid, {}).get("threat_target") not in (None, threat_target),
                    "teacher_mode_switch": previous.get(aid, {}).get("teacher_dominant_mode") not in (None, int(np.argmax(p_teacher))),
                    "transition_kill": False, "transition_death": False,
                    "transition_event": False, "temporal_pair_valid": False,
                    "red_kill_targets": "", "red_kill_attackers": "", "red_death_targets": "",
                }
                for key in (
                    "base_mean", "expert_engagement", "expert_support", "natural_residual",
                    "natural_mean", "engagement_mean", "support_mean", "uniform_mean",
                    "natural_action", "engagement_action", "support_action", "uniform_action",
                ):
                    base[key] = _json(np.asarray(intervention[key]).tolist())
                for key in (
                    "expert_output_distance", "mean_mode_separation", "action_mode_separation",
                    "natural_to_engagement_action", "natural_to_support_action", "residual_norm",
                    "base_norm", "residual_base_ratio",
                ):
                    base[key] = float(intervention[key])
                step_rows.append(base)
                intervention_rows.append(_intervention_row(base, intervention))
                current[aid] = base
            with torch.no_grad():
                action, _ = actor.sample(observation, deterministic=deterministic)
            actions.append(action.squeeze(0).cpu().numpy().astype(np.float32, copy=True))
        # Current state closes the previous temporal pair using the exact training mask semantics.
        for aid, prior in previous.items():
            now = current.get(aid)
            prior["temporal_pair_valid"] = bool(
                now is not None
                and not prior.get("terminated", False) and not prior.get("truncated", False)
                and not prior["transition_event"]
                and prior["engagement_target"] == now["engagement_target"]
                and prior["threat_target"] == now["threat_target"]
                and prior["teacher_dominant_mode"] == now["teacher_dominant_mode"]
            )
        observations, _, terminated, truncated, info = env.step(np.asarray(actions, np.float32))
        done = bool(terminated or truncated)
        final_info = info
        red_kill_events = [event for event in info["attack_events"] if event["target"] in BLUE_IDS]
        red_deaths = [aid for aid in info["death_causes"] if aid in RED_IDS]
        for row in current.values():
            row.update({
                "transition_kill": bool(red_kill_events), "transition_death": bool(red_deaths),
                "transition_event": bool(red_kill_events or red_deaths),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "red_kill_targets": ";".join(event["target"] for event in red_kill_events),
                "red_kill_attackers": ";".join(event["attacker"] for event in red_kill_events),
                "red_death_targets": ";".join(red_deaths),
                "natural_sampled_action": _json(actions[RED_IDS.index(row["uav_id"])].tolist()),
            })
        previous = current
    summary = final_info["episode_summary"]
    for row in step_rows:
        row["episode_outcome"] = summary["outcome"]
        row["episode_red_attack_kills"] = int(summary["red_attack_kills"])
    episode_row = {
        "checkpoint": str(loaded["checkpoint"]), "training_seed": loaded["training_seed"],
        "sampled_steps": loaded["sampled_steps"], "episode": episode,
        "environment_seed": environment_seed, "action_seed": None if deterministic else action_seed,
        "action_mode": action_mode, "outcome": summary["outcome"],
        "episode_return": float(summary["episode_return"]), "episode_length": int(summary["episode_length"]),
        "red_attack_kills": int(summary["red_attack_kills"]), "blue_attack_kills": int(summary["blue_attack_kills"]),
        "mav_survived": bool(summary["mav_survived"]), "uav_survivors": int(summary["red_uav_survivors"]),
        "semantic_step_rows": len(step_rows),
    }
    return step_rows, intervention_rows, episode_row


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and np.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else None


def _corr(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float | None:
    pairs = [(float(row[left]), float(row[right])) for row in rows if row.get(left) is not None and row.get(right) is not None]
    if len(pairs) < 2 or np.std([p[0] for p in pairs]) == 0 or np.std([p[1] for p in pairs]) == 0:
        return None
    return float(np.corrcoef(np.asarray(pairs).T)[0, 1])


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("mean", "std", "median", "q05", "q25", "q75", "q95")}
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)), "q05": float(np.quantile(array, .05)),
        "q25": float(np.quantile(array, .25)), "q75": float(np.quantile(array, .75)),
        "q95": float(np.quantile(array, .95)),
    }


def alignment_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    q = np.asarray([[row["teacher_engagement_probability"], row["teacher_support_probability"]] for row in rows])
    p = np.asarray([[row["router_engagement_probability"], row["router_support_probability"]] for row in rows])
    confidence = np.asarray([row["teacher_confidence"] for row in rows])
    kl = (q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p, 1e-12)))).sum(1)
    agree = q.argmax(1) == p.argmax(1)
    teacher_entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(1)
    router_entropy = -(p * np.log(np.maximum(p, 1e-12))).sum(1)
    # Disjoint rank-quantile buckets avoid overlap when confidence has ties.
    order = np.argsort(confidence, kind="stable")
    edge = max(1, len(order) // 4)
    bucket_indices = {
        "low": order[:edge], "middle": order[edge:len(order) - edge],
        "high": order[len(order) - edge:],
    }
    bucket_stats = {}
    for label, indices in bucket_indices.items():
        mask = np.zeros(len(confidence), dtype=bool); mask[indices] = True
        bucket_stats[label] = {
            "count": int(mask.sum()), "confidence_range": [float(confidence[mask].min()), float(confidence[mask].max())] if mask.any() else None,
            "agreement": float(agree[mask].mean()) if mask.any() else None,
            "mean_kl": float(kl[mask].mean()) if mask.any() else None,
        }
    denom = max(float(confidence.sum()), 1e-12)
    readiness = np.asarray([row["engagement_readiness"] for row in rows])
    support = np.asarray([row["support_responsibility"] for row in rows])
    def quartile_comparison(source: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
        lo, hi = np.quantile(source, [.25, .75])
        return {"bottom_q25_boundary": float(lo), "top_q25_boundary": float(hi),
                "bottom_q25_mean": float(outcome[source <= lo].mean()),
                "top_q25_mean": float(outcome[source >= hi].mean())}
    return {
        "rows": len(rows), "confidence_weighted_kl": float((confidence * kl).sum() / denom),
        "mean_l1": float(np.abs(q - p).sum(1).mean()), "hard_mode_agreement": float(agree.mean()),
        "confidence_weighted_hard_agreement": float((confidence * agree).sum() / denom),
        "mean_router_entropy": float(router_entropy.mean()), "mean_teacher_entropy": float(teacher_entropy.mean()),
        "confidence_quantile_buckets": {"definition": "disjoint stable rank bottom 25% / middle 50% / top 25%", **bucket_stats},
        "engagement_probability_correlation": _corr(rows, "teacher_engagement_probability", "router_engagement_probability"),
        "support_probability_correlation": _corr(rows, "teacher_support_probability", "router_support_probability"),
        "readiness_router_engagement_quartiles": quartile_comparison(readiness, p[:, 0]),
        "support_responsibility_router_support_quartiles": quartile_comparison(support, p[:, 1]),
    }


def expert_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for aid in UAV_IDS:
        current = [row for row in rows if row["uav_id"] == aid]
        result[aid] = {
            "states": len(current),
            "expert_output_distance": _distribution(row["expert_output_distance"] for row in current),
            "forced_action_separation": _distribution(row["action_mode_separation"] for row in current),
            "residual_norm": _distribution(row["residual_norm"] for row in current),
            "residual_base_ratio": _distribution(row["residual_base_ratio"] for row in current),
            "engagement_mode_occupancy": _mean(current, "router_engagement_probability"),
            "hard_engagement_occupancy": float(np.mean([row["router_engagement_probability"] > row["router_support_probability"] for row in current])) if current else None,
            "router_entropy": _distribution(row["router_entropy"] for row in current),
            "router_engagement_probability": _distribution(row["router_engagement_probability"] for row in current),
        }
    return result


def behavior_associations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive future behavior by router mode; episodes are never crossed."""
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["episode"]), str(row["uav_id"]))].append(row)
    output = []
    for subset_label in ("all", "teacher_high_confidence"):
        confidences = np.asarray([float(row["teacher_confidence"]) for row in rows])
        high = float(np.quantile(confidences, .75)) if len(confidences) else 1.0
        for mode in (0, 1):
            for horizon in (1, 3, 5, 10):
                records = []
                for series in groups.values():
                    for index, row in enumerate(series):
                        if int(row["router_dominant_mode"]) != mode:
                            continue
                        if subset_label != "all" and float(row["teacher_confidence"]) < high:
                            continue
                        future = series[index + 1:index + 1 + horizon]
                        if not future:
                            continue
                        same_target = [item for item in future if item["engagement_target"] == row["engagement_target"]]
                        same_threat = [item for item in future if item["threat_target"] == row["threat_target"]]
                        records.append({
                            "gate_entry": any(bool(item["engagement_exact_gate"]) for item in same_target),
                            "streak_increase": any((item["engagement_attack_streak"] or 0) > (row["engagement_attack_streak"] or 0) for item in same_target),
                            **{f"streak_ge_{level}": any((item["engagement_attack_streak"] or 0) >= level for item in same_target) for level in (1, 2, 3)},
                            "red_kill": any(bool(item["transition_kill"]) for item in future),
                            "distance_improvement": (float(row["engagement_distance"]) - float(same_target[-1]["engagement_distance"])) if same_target and row["engagement_distance"] is not None and same_target[-1]["engagement_distance"] is not None else None,
                            "ata_improvement": (float(row["engagement_ata_deg"]) - float(same_target[-1]["engagement_ata_deg"])) if same_target and row["engagement_ata_deg"] is not None and same_target[-1]["engagement_ata_deg"] is not None else None,
                            "aa_improvement": (float(row["engagement_aa_deg"]) - float(same_target[-1]["engagement_aa_deg"])) if same_target and row["engagement_aa_deg"] is not None and same_target[-1]["engagement_aa_deg"] is not None else None,
                            "intercept_distance_improvement": (float(row["intercept_distance"]) - float(same_threat[-1]["intercept_distance"])) if same_threat and row["intercept_distance"] is not None and same_threat[-1]["intercept_distance"] is not None else None,
                            "intercept_ata_improvement": (float(row["intercept_ata_deg"]) - float(same_threat[-1]["intercept_ata_deg"])) if same_threat and row["intercept_ata_deg"] is not None and same_threat[-1]["intercept_ata_deg"] is not None else None,
                            "intercept_aa_improvement": (float(row["intercept_aa_deg"]) - float(same_threat[-1]["intercept_aa_deg"])) if same_threat and row["intercept_aa_deg"] is not None and same_threat[-1]["intercept_aa_deg"] is not None else None,
                            "intercept_gate": any(bool(item["intercept_exact_gate"]) for item in same_threat),
                            "mav_threat_reduction": float(row["mav_threat"]) - float(future[-1]["mav_threat"]),
                            "threat_geometry_worsened": bool(same_threat and row["threat_to_mav_ata_deg"] is not None and same_threat[-1]["threat_to_mav_ata_deg"] is not None and float(same_threat[-1]["threat_to_mav_ata_deg"]) > float(row["threat_to_mav_ata_deg"])),
                            "threat_blue_killed": any(row["threat_target"] in str(item["red_kill_targets"]).split(";") for item in future),
                        })
                output.append({
                    "subset": subset_label, "confidence_top_q25_boundary": high if subset_label != "all" else None,
                    "router_mode": TACM_MODE_LABELS[mode], "horizon_steps": horizon, "states": len(records),
                    **{field: _mean(records, field) for field in (
                        "gate_entry", "streak_increase", "streak_ge_1", "streak_ge_2", "streak_ge_3", "red_kill",
                        "distance_improvement", "ata_improvement", "aa_improvement", "intercept_distance_improvement",
                        "intercept_ata_improvement", "intercept_aa_improvement", "intercept_gate",
                        "mav_threat_reduction", "threat_geometry_worsened", "threat_blue_killed",
                    )},
                })
    return output


def event_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["episode"]), str(row["uav_id"]))].append(row)
    stable_pairs = []
    event_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_windows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    run_lengths: list[int] = []
    response_latency: list[int] = []
    for series in groups.values():
        run = 1
        for index in range(len(series) - 1):
            left, right = series[index], series[index + 1]
            switched = int(left["router_dominant_mode"]) != int(right["router_dominant_mode"])
            movement = abs(float(left["router_engagement_probability"]) - float(right["router_engagement_probability"])) + abs(float(left["router_support_probability"]) - float(right["router_support_probability"]))
            pair = {"hard_switch": switched, "router_l1_movement": movement}
            if bool(left["temporal_pair_valid"]):
                stable_pairs.append(pair)
            labels = []
            if left["transition_kill"]: labels.append("kill")
            if left["transition_death"]: labels.append("death")
            if right["target_switch"]: labels.append("engagement_target_switch")
            if right["threat_switch"]: labels.append("mav_threat_target_switch")
            if right["teacher_mode_switch"]: labels.append("teacher_dominant_mode_switch")
            for label in labels:
                event_pairs[label].append(pair)
            if switched:
                run_lengths.append(run); run = 1
            else:
                run += 1
            if right["teacher_mode_switch"]:
                new_mode = int(right["teacher_dominant_mode"])
                latency = next((delay for delay in (0, 1, 2, 3) if index + 1 + delay < len(series) and int(series[index + 1 + delay]["router_dominant_mode"]) == new_mode), None)
                if latency is not None: response_latency.append(latency)
        for index, row in enumerate(series):
            transition_labels = []
            state_labels = []
            if row["transition_kill"]: transition_labels.append("kill")
            if row["transition_death"]: transition_labels.append("death")
            if row["target_switch"]: state_labels.append("engagement_target_switch")
            if row["threat_switch"]: state_labels.append("mav_threat_target_switch")
            if row["teacher_mode_switch"]: state_labels.append("teacher_dominant_mode_switch")
            for window in (1, 2, 3):
                if index + window < len(series):
                    right = series[index + window]
                    movement = abs(float(row["router_engagement_probability"]) - float(right["router_engagement_probability"])) + abs(float(row["router_support_probability"]) - float(right["router_support_probability"]))
                    for label in transition_labels:
                        event_windows[(label, window)].append({
                            "hard_switch": int(row["router_dominant_mode"]) != int(right["router_dominant_mode"]),
                            "router_l1_movement": movement,
                            "router_matches_current_teacher": int(right["router_dominant_mode"]) == int(right["teacher_dominant_mode"]),
                        })
                if index - window >= 0:
                    left = series[index - window]
                    movement = abs(float(left["router_engagement_probability"]) - float(row["router_engagement_probability"])) + abs(float(left["router_support_probability"]) - float(row["router_support_probability"]))
                    for label in state_labels:
                        event_windows[(label, window)].append({
                            "hard_switch": int(left["router_dominant_mode"]) != int(row["router_dominant_mode"]),
                            "router_l1_movement": movement,
                            "router_matches_current_teacher": int(row["router_dominant_mode"]) == int(row["teacher_dominant_mode"]),
                        })
        run_lengths.append(run)
    output = [{
        "context": "stable_temporal_pairs", "pairs": len(stable_pairs),
        "hard_switch_rate": _mean(stable_pairs, "hard_switch"),
        "router_l1_movement": _mean(stable_pairs, "router_l1_movement"),
        "router_probability_variance": float(np.var([row["router_engagement_probability"] for row in rows])) if rows else None,
        "router_entropy": _mean(rows, "router_entropy"),
        "same_mode_run_length_mean": float(np.mean(run_lengths)) if run_lengths else None,
        "same_mode_run_length_median": float(np.median(run_lengths)) if run_lengths else None,
    }]
    for label, pairs in event_pairs.items():
        output.append({"context": label, "pairs": len(pairs), "hard_switch_rate": _mean(pairs, "hard_switch"),
                       "router_l1_movement": _mean(pairs, "router_l1_movement")})
    for (label, window), pairs in sorted(event_windows.items()):
        output.append({
            "context": f"{label}_window", "event": label, "window_steps": window,
            "alignment": "transition event: state t to t+h; state switch: state t-h to t",
            "pairs": len(pairs), "hard_switch_rate": _mean(pairs, "hard_switch"),
            "router_l1_movement": _mean(pairs, "router_l1_movement"),
            "router_matches_current_teacher_rate": _mean(pairs, "router_matches_current_teacher"),
        })
    teacher_switches = event_pairs.get("teacher_dominant_mode_switch", [])
    output.append({
        "context": "teacher_mode_response", "pairs": len(teacher_switches),
        "response_observed_within_3_rate": len(response_latency) / len(teacher_switches) if teacher_switches else None,
        "response_latency_mean": float(np.mean(response_latency)) if response_latency else None,
        "response_latency_median": float(np.median(response_latency)) if response_latency else None,
        **{f"follow_within_{horizon}_rate": float(np.mean(np.asarray(response_latency) <= horizon)) if response_latency else None for horizon in (1, 2, 3)},
    })
    return output


def last_blue_analysis(rows: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    episode_map = {int(row["episode"]): row for row in episodes}
    output = []
    for episode, summary in episode_map.items():
        if summary["outcome"] != "draw" or int(summary["red_attack_kills"]) != 3:
            continue
        phase = [row for row in rows if int(row["episode"]) == episode and int(row["red_kills_pre"]) >= 3]
        for aid in UAV_IDS:
            current = [row for row in phase if row["uav_id"] == aid]
            if not current: continue
            output.append({
                "episode": episode, "environment_seed": summary["environment_seed"], "uav_id": aid,
                "steps": len(current), "teacher_engagement_probability": _mean(current, "teacher_engagement_probability"),
                "router_engagement_probability": _mean(current, "router_engagement_probability"),
                "router_support_probability": _mean(current, "router_support_probability"),
                "mav_threat": _mean(current, "mav_threat"),
                "engagement_target_counts": _json(Counter(str(row["engagement_target"]) for row in current)),
                "engagement_mode_occupancy": float(np.mean([row["router_dominant_mode"] == 0 for row in current])),
                "hard_mode_switch_rate": float(np.mean([current[i]["router_dominant_mode"] != current[i - 1]["router_dominant_mode"] for i in range(1, len(current))])) if len(current) > 1 else 0.0,
                "expert_separation": _mean(current, "expert_output_distance"),
                "attack_gate_entry_rate": float(np.mean([bool(row["engagement_exact_gate"]) for row in current])),
                "max_attack_streak": max((int(row["engagement_attack_streak"] or 0) for row in current), default=0),
                "last_blue_distance": _mean(current, "engagement_distance"),
                "last_blue_ata_deg": _mean(current, "engagement_ata_deg"),
                "last_blue_aa_deg": _mean(current, "engagement_aa_deg"),
            })
    return output


def episode_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int = 7319, samples: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if not rows: return {}
    fields = ("episode_return", "red_attack_kills", "blue_attack_kills", "episode_length", "mav_survived", "uav_survivors")
    result = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows])
        boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
        result[field] = {"mean": float(values.mean()), "bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    outcomes = [row["outcome"] for row in rows]
    for outcome in ("red", "blue", "draw"):
        values = np.asarray([item == outcome for item in outcomes], dtype=float)
        boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
        result[f"{outcome}_rate"] = {"mean": float(values.mean()), "bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    return result


def evidence_categories(
    alignment: Mapping[str, Any], experts: Mapping[str, Any], events: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]], last_blue: Sequence[Mapping[str, Any]], episodes: int,
) -> dict[str, Any]:
    """Conservative interpretive labels; raw statistics remain authoritative."""
    enough = episodes >= 20
    stable = next((row for row in events if row["context"] == "stable_temporal_pairs"), {})
    engage = [row for row in associations if row["subset"] == "teacher_high_confidence" and row["router_mode"] == "engagement"]
    support = [row for row in associations if row["subset"] == "teacher_high_confidence" and row["router_mode"] == "cover_support"]
    return {
        "router_semantic_alignment": {"evidence_strength": "moderate" if enough and alignment else "inconclusive", "key_statistics": alignment, "interpretation": "Association with the centralized teacher; not causal proof."},
        "expert_behavioral_separation": {"evidence_strength": "moderate" if enough and experts else "inconclusive", "key_statistics": experts, "interpretation": "Same-state interventions test available control separation, not realized causal benefit."},
        "event_aware_persistence": {"evidence_strength": "moderate" if enough and stable else "inconclusive", "key_statistics": stable, "interpretation": "Stable/event-conditioned router movement is descriptive."},
        "engagement_behavior_association": {"evidence_strength": "weak" if enough and engage else "inconclusive", "key_statistics": engage, "interpretation": "Future behavior conditional on router state; insufficient to establish causality."},
        "support_behavior_association": {"evidence_strength": "weak" if enough and support else "inconclusive", "key_statistics": support, "interpretation": "Future threat geometry conditional on router state; insufficient to establish protection causality."},
        "last_blue_cleanup_behavior": {"evidence_strength": "weak" if enough and last_blue else "inconclusive", "key_statistics": {"three_kill_draw_uav_rows": len(last_blue)}, "interpretation": "Descriptive last-Blue behavior only."},
        "decentralized_execution_contract": {"evidence_strength": "strong", "key_statistics": {"teacher_called_for_action": False, "critic_called_for_action": False, "actor_input": "local observation"}, "interpretation": "Confirmed by code path and action-invariance tests."},
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    loaded = load_tacm_checkpoint(args.checkpoint, args.device)
    config = validate_semantic_audit_contract(loaded, args.env_config)
    step_rows: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        rows, mode_rows, summary = run_episode(
            loaded, episode=episode, environment_seed=args.env_seed_start + episode,
            action_seed=args.action_seed + episode, action_mode=args.action_mode,
            profile=args.profile, device=args.device,
        )
        step_rows.extend(rows); interventions.extend(mode_rows); episode_rows.append(summary)
    associations = behavior_associations(step_rows)
    events = event_analysis(step_rows)
    last_blue = last_blue_analysis(step_rows, episode_rows)
    alignment = alignment_summary(step_rows)
    experts = expert_summary(interventions)
    write_csv(output / OUTPUT_FILES["steps"], step_rows)
    write_csv(output / OUTPUT_FILES["episodes"], episode_rows)
    write_csv(output / OUTPUT_FILES["interventions"], interventions)
    write_csv(output / OUTPUT_FILES["events"], events + associations)
    write_csv(output / OUTPUT_FILES["last_blue"], last_blue, fields=(
        "episode", "environment_seed", "uav_id", "steps",
        "teacher_engagement_probability", "router_engagement_probability",
        "router_support_probability", "mav_threat", "engagement_target_counts",
        "engagement_mode_occupancy", "hard_mode_switch_rate", "expert_separation",
        "attack_gate_entry_rate", "max_attack_streak", "last_blue_distance",
        "last_blue_ata_deg", "last_blue_aa_deg",
    ))
    summary = {
        "audit": "TACM-RGAA-v1 semantic-mode audit", "read_only": True,
        "checkpoint": str(loaded["checkpoint"]), "training_seed": loaded["training_seed"],
        "sampled_steps": loaded["sampled_steps"], "environment_version": config["environment_version"],
        "profile": args.profile, "episodes": args.episodes,
        "environment_seed_range": [args.env_seed_start, args.env_seed_start + args.episodes - 1],
        "action_mode": args.action_mode,
        "action_seed_range": None if args.action_mode == "deterministic" else [args.action_seed, args.action_seed + args.episodes - 1],
        "device": args.device, "output_files": OUTPUT_FILES,
        "static_contract_audit": static_contract_audit(),
        "episode_statistics": episode_bootstrap(episode_rows),
        "router_semantic_alignment": alignment, "expert_behavioral_separation": experts,
        "event_aware_persistence": events, "behavior_associations": associations,
        "last_blue_cleanup": {"three_kill_draw_uav_rows": len(last_blue)},
        "evidence_categories": evidence_categories(alignment, experts, events, associations, last_blue, args.episodes),
        "limitations": [
            "Step records are temporally correlated and are not treated as independent inferential samples.",
            "Behavior associations and same-state interventions do not establish causal tactical semantics.",
            "Bootstrap intervals resample episodes, not steps.",
        ],
    }
    with (output / OUTPUT_FILES["summary"]).open("w", encoding="utf-8") as stream:
        json.dump(_finite_or_none(summary), stream, ensure_ascii=False, indent=2, allow_nan=False)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--profile", choices=("learnability", "main"), default="main")
    parser.add_argument("--env-seed-start", type=int, default=1000)
    parser.add_argument("--action-seed", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--action-mode", choices=("deterministic", "stochastic"), default="stochastic")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps({
        "checkpoint": summary["checkpoint"], "training_seed": summary["training_seed"],
        "episodes": summary["episodes"], "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
