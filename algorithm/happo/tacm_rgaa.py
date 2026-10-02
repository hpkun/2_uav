"""Tactical-context teacher and router-only losses for TACM-RGAA-v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from env.geometry import compute_pairwise_geometry
from env.models import AircraftState
from env.mavuav import BLUE_IDS, CROSS_TEAM_ATTACK_PAIRS, ENTITY_IDS, RED_IDS
from env.reward_role_v37 import uav_distance_reward
from env.reward_role_v39 import attack_gate_indicator, uav_angle_quality
from .rgaa import RoleAdvantageRolloutBuffer


TACM_RGAA_METHOD = "tacm_rgaa"
TACM_RGAA_VERSION = 1
TACM_MODE_LABELS = ("engagement", "cover_support")
TACM_TEACHER_SEMANTICS = "team_visible_team_aware_tactical_context_v1"
TACM_TEMPORAL_SEMANTICS = "event_aware_router_consistency_v1"


@dataclass(frozen=True)
class TacticalTeacherBatch:
    probabilities: np.ndarray
    confidence: np.ndarray
    engagement_scores: np.ndarray
    cover_responsibility: np.ndarray
    cover_assignment: np.ndarray
    engagement_target: np.ndarray
    threat_target: np.ndarray
    mav_threat: np.ndarray


def _decode_state(block: np.ndarray, config: Mapping[str, Any]) -> AircraftState:
    bx, by = config["battlefield"]["x"], config["battlefield"]["y"]
    bh = config["battlefield"]["altitude"]
    def inverse(value: float, bounds: Any) -> float:
        return float(bounds[0] + 0.5 * (float(value) + 1.0) * (bounds[1] - bounds[0]))
    return AircraftState(
        inverse(block[0], bx), inverse(block[1], by), inverse(block[2], bh),
        float(block[3]) * 400.0, float(block[4]) * np.pi,
        float(block[5]) * np.pi, bool(block[6] > 0.5),
    )


def _quality(attacker: AircraftState, target: AircraftState, streak: float, combat: Mapping[str, Any]) -> tuple[float, float, float]:
    geometry = compute_pairwise_geometry(attacker, target)
    minimum, maximum = (float(v) for v in combat["distance"])
    angle = uav_angle_quality(geometry.ata, geometry.aa)
    distance = uav_distance_reward(geometry.distance, minimum, maximum)
    gate = attack_gate_indicator(
        geometry.distance, geometry.ata, geometry.aa, minimum, maximum,
        np.deg2rad(float(combat["ata_deg"])), np.deg2rad(float(combat["aa_deg"])),
    )
    return float(angle), float(distance), float((angle + distance + float(streak) + gate) / 4.0)


def tactical_teacher(
    global_states: np.ndarray, config: Mapping[str, Any], *, tau_group: float = 0.25,
    tau_teacher: float = 0.25,
) -> TacticalTeacherBatch:
    """Compute deterministic pre-action teacher targets from team-visible state."""
    states = np.asarray(global_states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None]
    n = len(states); u = 3
    probs = np.full((n, u, 2), 0.5, np.float32)
    confidence = np.zeros((n, u), np.float32)
    engagement = np.zeros((n, u), np.float32)
    responsibility = np.zeros((n, u), np.float32)
    assignment = np.zeros((n, u), np.float32)
    engagement_target = np.full((n, u), -1, np.int64)
    threat_target = np.full(n, -1, np.int64)
    mav_threat = np.zeros(n, np.float32)
    pair_index = {pair: index for index, pair in enumerate(CROSS_TEAM_ATTACK_PAIRS)}
    combat = config["combat"]
    for env_index, row in enumerate(states):
        entities = {aid: _decode_state(row[i * 10:(i + 1) * 10], config) for i, aid in enumerate(ENTITY_IDS)}
        visible: list[str] = []
        for blue in BLUE_IDS:
            if not entities[blue].alive:
                continue
            for red in RED_IDS:
                if not entities[red].alive:
                    continue
                sensor = float(config["sensing"]["MAV_range" if red == "MAV" else "UAV_range"])
                if compute_pairwise_geometry(entities[red], entities[blue]).distance <= sensor:
                    visible.append(blue); break
        if not visible:
            continue
        threat_values = []
        for blue in visible:
            streak = row[80 + pair_index[(blue, "MAV")]]
            threat_values.append(_quality(entities[blue], entities["MAV"], streak, combat)[2])
        best_threat = int(np.argmax(threat_values))
        threat_blue = visible[best_threat]
        threat_target[env_index] = BLUE_IDS.index(threat_blue)
        mav_threat[env_index] = threat_values[best_threat]
        suitability = np.full(u, -np.inf, np.float64)
        for ui, aid in enumerate(RED_IDS[1:]):
            if not entities[aid].alive:
                continue
            values = []
            for blue in visible:
                streak = row[80 + pair_index[(aid, blue)]]
                values.append(_quality(entities[aid], entities[blue], streak, combat)[2])
            best = int(np.argmax(values))
            engagement[env_index, ui] = values[best]
            engagement_target[env_index, ui] = BLUE_IDS.index(visible[best])
            a, d, _ = _quality(entities[aid], entities[threat_blue], 0.0, combat)
            suitability[ui] = 0.5 * (a + d)
        alive = np.isfinite(suitability)
        if alive.any():
            logits = suitability[alive] / float(tau_group)
            weights = np.exp(logits - logits.max()); weights /= weights.sum()
            assignment[env_index, alive] = weights
            responsibility[env_index] = mav_threat[env_index] * assignment[env_index]
        adjusted = engagement[env_index] * (1.0 - responsibility[env_index])
        teacher_logits = np.stack((adjusted, responsibility[env_index]), axis=-1) / float(tau_teacher)
        teacher_logits -= teacher_logits.max(axis=-1, keepdims=True)
        q = np.exp(teacher_logits); q /= q.sum(axis=-1, keepdims=True)
        probs[env_index] = q
        entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=-1)
        confidence[env_index] = 1.0 - entropy / np.log(2.0)
        confidence[env_index, ~alive] = 0.0
    return TacticalTeacherBatch(probs, confidence, engagement, responsibility, assignment,
                                engagement_target, threat_target, mav_threat)


class TACMRoleAdvantageRolloutBuffer(RoleAdvantageRolloutBuffer):
    def __init__(self, horizon: int, num_envs: int) -> None:
        super().__init__(horizon, num_envs)
        s = (horizon, num_envs)
        self.teacher_probabilities = np.zeros(s + (3, 2), np.float32)
        self.teacher_confidence = np.zeros(s + (3,), np.float32)
        self.engagement_scores = np.zeros(s + (3,), np.float32)
        self.cover_responsibility = np.zeros(s + (3,), np.float32)
        self.cover_assignment = np.zeros(s + (3,), np.float32)
        self.engagement_target = np.full(s + (3,), -1, np.int64)
        self.threat_target = np.full(s, -1, np.int64)
        self.transition_event = np.zeros(s, bool)
        self.mav_threat = np.zeros(s, np.float32)

    def insert_tacm(self, teacher: TacticalTeacherBatch, transition_event: np.ndarray, **kwargs: Any) -> None:
        i = self.position
        super().insert(**kwargs)
        self.teacher_probabilities[i] = teacher.probabilities
        self.teacher_confidence[i] = teacher.confidence
        self.engagement_scores[i] = teacher.engagement_scores
        self.cover_responsibility[i] = teacher.cover_responsibility
        self.cover_assignment[i] = teacher.cover_assignment
        self.engagement_target[i] = teacher.engagement_target
        self.threat_target[i] = teacher.threat_target
        self.transition_event[i] = np.asarray(transition_event, dtype=bool)
        self.mav_threat[i] = teacher.mav_threat


def router_probabilities_detached(actor: Any, observations: torch.Tensor) -> torch.Tensor:
    hidden = actor.network.encoder(observations).detach()
    return torch.softmax(actor.network.router(hidden), dim=-1)


def context_distillation_loss(actor: Any, observations: torch.Tensor, teacher: torch.Tensor,
                              confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = router_probabilities_detached(actor, observations)
    kl = (teacher.detach() * (teacher.detach().clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1)
    denom = confidence.sum().clamp_min(1.0)
    return (confidence * kl).sum() / denom, kl


def temporal_router_loss(actor: Any, observations: torch.Tensor, active: torch.Tensor,
                         terminated: torch.Tensor, truncated: torch.Tensor,
                         event: torch.Tensor, targets: torch.Tensor,
                         threat: torch.Tensor, teacher: torch.Tensor,
                         confidence: torch.Tensor) -> tuple[torch.Tensor, int]:
    p = router_probabilities_detached(actor, observations.reshape(-1, observations.shape[-1])).reshape(*observations.shape[:2], 2)
    valid = active[:-1] & active[1:] & ~terminated[:-1] & ~truncated[:-1] & ~event[:-1]
    valid &= targets[:-1].eq(targets[1:]) & threat[:-1].eq(threat[1:])
    valid &= teacher[:-1].argmax(-1).eq(teacher[1:].argmax(-1))
    weight = torch.minimum(confidence[:-1], confidence[1:]) * valid.float()
    raw = ((p[1:] - p[:-1].detach()) ** 2).sum(-1)
    count = int(valid.sum().item())
    return (weight * raw).sum() / weight.sum().clamp_min(1.0), count


def tacm_context_coefficient(sampled_steps: int, start: float, end: float, anneal_steps: int) -> float:
    alpha = min(max(float(sampled_steps) / max(int(anneal_steps), 1), 0.0), 1.0)
    return float(end + (start - end) * (1.0 - alpha))


def tacm_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("method_variant") != TACM_RGAA_METHOD:
        return {}
    return {
        "tacm_rgaa_version": TACM_RGAA_VERSION,
        "tacm_mode_labels": list(TACM_MODE_LABELS),
        "tacm_teacher_semantics": TACM_TEACHER_SEMANTICS,
        "tacm_temporal_semantics": TACM_TEMPORAL_SEMANTICS,
        "tacm_tau_group": float(config["tacm_tau_group"]),
        "tacm_tau_teacher": float(config["tacm_tau_teacher"]),
        "tacm_context_coef_start": float(config["tacm_context_coef_start"]),
        "tacm_context_coef_end": float(config["tacm_context_coef_end"]),
        "tacm_context_anneal_steps": int(config["tacm_context_anneal_steps"]),
        "tacm_temporal_coef": float(config["tacm_temporal_coef"]),
    }
