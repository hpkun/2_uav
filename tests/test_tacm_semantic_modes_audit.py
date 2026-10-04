from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.dbm_rgaa import DBMGaussianActor, build_method_actors
from algorithm.happo.tacm_rgaa import context_distillation_loss, tactical_teacher, temporal_router_loss
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.models import AircraftState
from tools.audit_tacm_semantic_modes import (
    diagnostic_rng_invariant,
    forced_mode_intervention,
    static_contract_audit,
    validate_semantic_audit_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def short_v310(max_steps: int = 4):
    config = deepcopy(load_environment_config(ROOT / "configs" / "env_v310.yaml"))
    config["simulation"]["max_decision_steps"] = max_steps
    return config


def tacm_config(**updates):
    with (ROOT / "configs" / "happo_tacm_rgaa_v310.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["training"]
    config.update({"device": "cpu", "num_envs": 1, "rollout_steps": 4,
                   "ppo_epochs": 1, "minibatch_size": 4, "seed": 7,
                   "randomization_curriculum_enabled": False})
    config.update(updates)
    return config


def _set(env, aid, xyz, psi=0.0, alive=True):
    env.entities[aid].state = AircraftState(*xyz, 220.0, 0.0, psi, alive)


def test_teacher_no_visible_blue_is_uniform_and_zero_confidence():
    env = HeterogeneousMAVUAVAirCombatEnv(short_v310(), seed=3, randomize=False)
    env.reset()
    for aid in RED_IDS:
        _set(env, aid, (-19000.0, -19000.0, 5000.0))
    for aid in BLUE_IDS:
        _set(env, aid, (19000.0, 19000.0, 5000.0))
    teacher = tactical_teacher(env.global_state(), env.config)
    assert np.array_equal(teacher.probabilities, np.full((1, 3, 2), .5, np.float32))
    assert np.array_equal(teacher.confidence, np.zeros((1, 3), np.float32))
    assert np.all(teacher.engagement_target == -1)
    assert np.all(teacher.threat_target == -1)


def test_context_loss_gradient_is_strictly_router_only():
    trainer = HAPPOTrainer(short_v310(), tacm_config())
    try:
        actor = trainer.actors.actors[1]
        observation = torch.randn(12, 100)
        teacher = torch.tensor([[.85, .15]]).repeat(12, 1)
        confidence = torch.ones(12)
        loss, _ = context_distillation_loss(actor, observation, teacher, confidence)
        loss.backward()
        assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
                   for parameter in actor.network.router.parameters())
        assert all(parameter.grad is None for parameter in actor.network.encoder.parameters())
        assert all(parameter.grad is None for parameter in actor.network.base_head.parameters())
        assert all(parameter.grad is None for expert in actor.network.experts for parameter in expert.parameters())
        assert actor.log_std.grad is None
    finally:
        trainer.close()


@pytest.mark.parametrize("invalid", (
    "event", "terminated", "truncated", "inactive", "engagement", "threat", "teacher_mode",
))
def test_temporal_pair_masks_every_frozen_boundary(invalid):
    trainer = HAPPOTrainer(short_v310(), tacm_config())
    try:
        actor = trainer.actors.actors[1]
        observations = torch.randn(2, 1, 100)
        active = torch.ones(2, 1, dtype=torch.bool)
        terminated = torch.zeros(2, 1, dtype=torch.bool)
        truncated = torch.zeros(2, 1, dtype=torch.bool)
        event = torch.zeros(2, 1, dtype=torch.bool)
        engagement = torch.zeros(2, 1, dtype=torch.long)
        threat = torch.zeros(2, 1, dtype=torch.long)
        teacher = torch.tensor([[[.8, .2]], [[.8, .2]]])
        confidence = torch.ones(2, 1)
        if invalid == "event": event[0] = True
        elif invalid == "terminated": terminated[0] = True
        elif invalid == "truncated": truncated[0] = True
        elif invalid == "inactive": active[1] = False
        elif invalid == "engagement": engagement[1] = 1
        elif invalid == "threat": threat[1] = 1
        elif invalid == "teacher_mode": teacher[1] = torch.tensor([.2, .8])
        _, count = temporal_router_loss(
            actor, observations, active, terminated, truncated, event,
            engagement, threat, teacher, confidence,
        )
        assert count == 0
    finally:
        trainer.close()


def test_temporal_stable_pair_is_valid_and_router_only_gradient():
    trainer = HAPPOTrainer(short_v310(), tacm_config())
    try:
        actor = trainer.actors.actors[1]
        observations = torch.randn(2, 1, 100)
        active = torch.ones(2, 1, dtype=torch.bool)
        clear = torch.zeros(2, 1, dtype=torch.bool)
        target = torch.zeros(2, 1, dtype=torch.long)
        teacher = torch.tensor([[[.8, .2]], [[.8, .2]]])
        loss, count = temporal_router_loss(
            actor, observations, active, clear, clear, clear,
            target, target, teacher, torch.ones(2, 1),
        )
        assert count == 1
        loss.backward()
        assert any(parameter.grad is not None for parameter in actor.network.router.parameters())
        assert all(parameter.grad is None for parameter in actor.network.encoder.parameters())
        assert all(parameter.grad is None for parameter in actor.network.base_head.parameters())
        assert all(parameter.grad is None for expert in actor.network.experts for parameter in expert.parameters())
    finally:
        trainer.close()


def test_forced_mode_intervention_matches_exact_dbm_formula():
    actors = build_method_actors(method_variant="dbm_rgaa", training_seed=7)
    actor = actors.actors[1]
    assert isinstance(actor, DBMGaussianActor)
    observation = torch.randn(1, 100)
    result = forced_mode_intervention(actor, observation)
    base = np.asarray(result["base_mean"])
    probs = np.asarray(result["router_probabilities"])
    eng = np.asarray(result["expert_engagement"])
    support = np.asarray(result["expert_support"])
    rho = actor.network.residual_scale
    assert np.allclose(result["natural_mean"], base + rho * (probs[0] * eng + probs[1] * support))
    assert np.allclose(result["engagement_mean"], base + rho * eng)
    assert np.allclose(result["support_mean"], base + rho * support)
    assert np.allclose(result["uniform_mean"], base + .5 * rho * (eng + support))
    assert np.allclose(result["engagement_action"], np.tanh(result["engagement_mean"]))


def test_intervention_diagnostics_are_rng_free_and_do_not_change_action():
    actor = build_method_actors(method_variant="dbm_rgaa", training_seed=9).actors[1]
    observation = torch.randn(1, 100)
    torch.manual_seed(2026)
    expected, _ = actor.sample(observation)
    torch.manual_seed(2026)
    diagnostic_rng_invariant(actor, observation)
    actual, _ = actor.sample(observation)
    assert torch.equal(actual, expected)


def test_initial_router_and_antisymmetric_experts_cancel_exactly():
    actor = build_method_actors(method_variant="dbm_rgaa", training_seed=11).actors[2]
    observation = torch.randn(16, 100)
    details = actor.network.details(observation)
    assert torch.equal(details["router_probabilities"], torch.full((16, 2), .5))
    assert torch.allclose(details["expert_outputs"][:, 0], -details["expert_outputs"][:, 1], atol=1e-7)
    assert torch.allclose(details["final_mean"], details["base_mean"], atol=1e-7)


def test_checkpoint_contract_requires_exact_2m_v310_and_exact_env():
    config = load_environment_config(ROOT / "configs" / "env_v310.yaml")
    loaded = {
        "payload": {"algorithm": "tacm_rgaa_happo", "method_variant": "tacm_rgaa",
                    "sampled_steps": 2_000_000, "trainer_config": {"method_variant": "tacm_rgaa"}},
        "environment_config": config,
    }
    assert validate_semantic_audit_contract(loaded, config) == config
    wrong = deepcopy(config); wrong["sensing"]["UAV_range"] += 1.0
    with pytest.raises(RuntimeError, match="exactly match"):
        validate_semantic_audit_contract(loaded, wrong)
    loaded["payload"]["sampled_steps"] = 1_999_999
    with pytest.raises(RuntimeError, match="2,000,000"):
        validate_semantic_audit_contract(loaded, config)


def test_v310_mav_remains_unarmed_during_audit_imports():
    env = HeterogeneousMAVUAVAirCombatEnv(short_v310(), seed=5, randomize=False)
    env.reset()
    _set(env, "MAV", (0.0, 0.0, 5000.0), psi=0.0)
    _set(env, "Blue1", (2000.0, 0.0, 5000.0), psi=0.0)
    env._attack_streak[("MAV", "Blue1")] = 2
    events, deaths = env._resolve_attacks()
    assert env.config["combat"]["mav_can_attack"] is False
    assert env._attack_streak[("MAV", "Blue1")] == 0
    assert all(event["attacker"] != "MAV" for event in events)
    assert deaths.get("Blue1") != "red_attack"


def test_static_contract_records_ctde_and_squared_l2_not_kl():
    contract = static_contract_audit()
    assert contract["decentralized_execution"]["status"] == "confirmed"
    assert "squared L2" in contract["temporal_consistency"]["loss"]
    assert contract["v3_10_combat"]["mav_can_attack"] is False

