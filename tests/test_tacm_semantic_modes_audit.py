from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.dbm_rgaa import DBMGaussianActor, build_method_actors
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.tacm_rgaa import context_distillation_loss, tactical_teacher, temporal_router_loss
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import BLUE_IDS, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.models import AircraftState
from tools.audit_tacm_semantic_modes import (
    _continuous_future,
    alignment_summary,
    behavior_associations,
    diagnostic_rng_invariant,
    event_analysis,
    event_centered_records,
    evidence_sections,
    forced_mode_intervention,
    last_blue_analysis,
    run_episode,
    static_contract_audit,
    target_change,
    teacher_response_analysis,
    temporal_pair_diagnostics,
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


def _audit_row(step: int, *, episode: int = 0, aid: str = "UAV1", target="Blue1",
               threat="Blue2", router=(.8, .2), teacher=(.8, .2), confidence=.8,
               **updates):
    row = {
        "episode": episode, "uav_id": aid, "decision_step": step, "uav_active": True,
        "engagement_target": target, "threat_target": threat,
        "router_engagement_probability": router[0], "router_support_probability": router[1],
        "router_dominant_mode": int(np.argmax(router)),
        "teacher_engagement_probability": teacher[0], "teacher_support_probability": teacher[1],
        "teacher_dominant_mode": int(np.argmax(teacher)), "teacher_confidence": confidence,
        "router_entropy": float(-(np.asarray(router) * np.log(np.asarray(router))).sum()),
        "engagement_readiness": .5, "support_responsibility": .5, "mav_threat": .6,
        "engagement_distance": 2500.0, "engagement_ata_deg": 20.0, "engagement_aa_deg": 40.0,
        "engagement_exact_gate": False, "engagement_attack_streak": 0,
        "intercept_distance": 3000.0, "intercept_ata_deg": 30.0, "intercept_aa_deg": 50.0,
        "intercept_exact_gate": False,
        "threat_to_mav_distance": 2000.0, "threat_to_mav_ata_deg": 20.0,
        "threat_to_mav_aa_deg": 30.0, "threat_to_mav_exact_gate": True,
        "target_switch": False, "engagement_target_change_type": "none",
        "threat_switch": False, "threat_target_change_type": "none",
        "teacher_mode_switch": False, "transition_kill": False, "transition_death": False,
        "transition_event": False, "terminated": False, "truncated": False,
        "temporal_mask_valid": False, "temporal_weight": 0.0, "temporal_effective": False,
        "red_kill_targets": "", "red_kill_attackers": "",
    }
    row.update(updates)
    return row


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


def test_temporal_mask_weight_and_effective_supervision_are_distinct():
    left = _audit_row(1, confidence=0.0)
    right = _audit_row(2, confidence=.8)
    zero = temporal_pair_diagnostics(left, right)
    assert zero["temporal_mask_valid"] is True
    assert zero["temporal_weight"] == 0.0
    assert zero["temporal_effective"] is False
    left["teacher_confidence"] = .3
    weighted = temporal_pair_diagnostics(left, right)
    assert weighted["temporal_mask_valid"] is True
    assert weighted["temporal_weight"] == pytest.approx(.3)
    assert weighted["temporal_effective"] is True


def test_alignment_summary_reports_disjoint_confidence_buckets_and_exact_agreement():
    rows = [
        _audit_row(1, teacher=(.9, .1), router=(.9, .1), confidence=.1),
        _audit_row(2, teacher=(.8, .2), router=(.8, .2), confidence=.3),
        _audit_row(3, teacher=(.2, .8), router=(.2, .8), confidence=.7),
        _audit_row(4, teacher=(.1, .9), router=(.9, .1), confidence=.9),
    ]
    summary = alignment_summary(rows)
    buckets = summary["confidence_quantile_buckets"]
    assert sum(buckets[label]["count"] for label in ("low", "middle", "high")) == 4
    assert summary["hard_mode_agreement"] == .75
    assert summary["confidence_weighted_kl"] > 0.0


@pytest.mark.parametrize(("previous", "current", "expected"), (
    (None, "Blue1", "acquisition"), ("Blue1", None, "loss"),
    ("Blue1", "Blue2", "replacement"), ("Blue1", "Blue1", "none"),
))
def test_adjacent_target_change_types(previous, current, expected):
    changed, change_type = target_change(True, previous, current)
    assert change_type == expected
    assert changed is (expected != "none")
    assert target_change(False, None, "Blue1") == (False, "none")


def test_continuous_target_segment_stops_at_first_switch_and_never_rejoins():
    rows = [
        _audit_row(1, target="Blue1"), _audit_row(2, target="Blue1"),
        _audit_row(3, target="Blue2"), _audit_row(4, target="Blue1"),
    ]
    segment = _continuous_future(rows, 0, 10, "engagement_target")
    assert [row["decision_step"] for row in segment] == [2]
    rows[0]["engagement_target"] = None
    assert _continuous_future(rows, 0, 10, "engagement_target") == []


def _association(rows, mode="engagement", horizon=1):
    return next(row for row in behavior_associations(rows)
                if row["subset"] == "all" and row["router_mode"] == mode
                and row["horizon_steps"] == horizon)


def test_kill_association_distinguishes_team_target_and_multi_attacker_membership():
    current = _audit_row(1, target="Blue1")
    unrelated = _audit_row(
        2, target="Blue1", transition_kill=True,
        red_kill_targets="Blue2", red_kill_attackers="UAV3",
    )
    summary = _association([current, unrelated])
    assert summary["any_team_kill"] == 1.0
    assert summary["engagement_target_killed"] == 0.0
    assert summary["engagement_target_killed_by_this_uav"] == 0.0
    multi = _audit_row(
        2, target="Blue1", transition_kill=True,
        red_kill_targets="Blue1;Blue1", red_kill_attackers="UAV2;UAV1",
    )
    summary = _association([current, multi])
    assert summary["engagement_target_killed"] == 1.0
    assert summary["engagement_target_killed_by_this_uav"] == 1.0


def test_stable_context_statistics_exclude_unstable_large_router_change():
    rows = [
        _audit_row(1, router=(.80, .20), temporal_mask_valid=True,
                   temporal_weight=.7, temporal_effective=True),
        _audit_row(2, router=(.81, .19), temporal_mask_valid=False,
                   temporal_weight=0.0, temporal_effective=False),
        _audit_row(3, router=(.05, .95), temporal_mask_valid=False,
                   temporal_weight=0.0, temporal_effective=False),
    ]
    summary = event_analysis(rows, [])
    stable = next(row for row in summary if row.get("context") == "effective_supervised_stable_context")
    global_row = next(row for row in summary if row.get("context") == "all_active_uav_states")
    assert stable["pairs"] == 1
    assert stable["router_l1_movement"] == pytest.approx(.02)
    assert stable["router_engagement_probability_variance"] < global_row["router_engagement_probability_variance"]
    assert stable["same_mode_run_length_mean"] == 2.0


def test_event_centered_window_is_symmetric_and_never_crosses_agent_or_episode():
    rows = [_audit_row(step, target="Blue1") for step in range(1, 8)]
    rows[3].update(target="Blue2", target_switch=True,
                   engagement_target_change_type="replacement")
    rows.extend(_audit_row(step, episode=1, target="Blue3") for step in range(1, 8))
    rows.extend(_audit_row(step, episode=0, aid="UAV2", target="Blue4") for step in range(1, 8))
    centered = [row for row in event_centered_records(rows)
                if row["event_type"] == "engagement_target_replacement"]
    assert {row["relative_step"] for row in centered} == set(range(-3, 4))
    assert {row["episode"] for row in centered} == {0}
    assert {row["uav_id"] for row in centered} == {"UAV1"}


def test_teacher_response_rejects_transient_switch_and_tracks_sustained_latency():
    transient = [
        _audit_row(1, episode=0, teacher=(.8, .2), router=(.8, .2)),
        _audit_row(2, episode=0, teacher=(.2, .8), router=(.8, .2), teacher_mode_switch=True),
        _audit_row(3, episode=0, teacher=(.8, .2), router=(.2, .8), teacher_mode_switch=True),
    ]
    sustained = [
        _audit_row(1, episode=1, teacher=(.8, .2), router=(.8, .2)),
        _audit_row(2, episode=1, teacher=(.2, .8), router=(.8, .2), teacher_mode_switch=True),
        _audit_row(3, episode=1, teacher=(.2, .8), router=(.8, .2)),
        _audit_row(4, episode=1, teacher=(.2, .8), router=(.2, .8)),
        _audit_row(5, episode=1, teacher=(.2, .8), router=(.2, .8)),
    ]
    response = {row["horizon_steps"]: row for row in teacher_response_analysis(transient + sustained)}
    assert response[1]["eligible_sustained_switches"] == 1
    assert response[1]["follow_within_horizon_rate"] == 0.0
    assert response[2]["eligible_sustained_switches"] == 1
    assert response[2]["follow_within_horizon_rate"] == 1.0
    assert response[3]["eligible_sustained_switches"] == 1
    assert response[3]["follow_within_horizon_rate"] == 1.0
    assert response[2]["response_latency_mean"] == 2.0


def test_support_geometry_is_componentwise_and_stops_at_threat_switch():
    current = _audit_row(1, router=(.2, .8), threat="Blue2", mav_threat=.8,
                         threat_to_mav_distance=2000.0, threat_to_mav_ata_deg=10.0,
                         threat_to_mav_aa_deg=20.0, threat_to_mav_exact_gate=True)
    future = _audit_row(2, router=(.2, .8), threat="Blue2", mav_threat=.3,
                        threat_to_mav_distance=2600.0, threat_to_mav_ata_deg=30.0,
                        threat_to_mav_aa_deg=50.0, threat_to_mav_exact_gate=False)
    switched = _audit_row(3, router=(.2, .8), threat="Blue3", mav_threat=.1,
                          threat_to_mav_distance=9000.0)
    summary = _association([current, future, switched], mode="cover_support", horizon=3)
    assert summary["continuous_threat_steps"] == pytest.approx(.5)  # starts at steps 1 and 2
    # The first state's valid continuous segment reports the transparent component changes.
    one = _association([current, future], mode="cover_support", horizon=1)
    assert one["threat_to_mav_distance_change"] == 600.0
    assert one["threat_to_mav_ata_change"] == 20.0
    assert one["threat_to_mav_aa_change"] == 30.0
    assert one["threat_to_mav_gate_lost"] == 1.0
    assert one["mav_threat_score_reduction"] == pytest.approx(.5)
    assert "threat_geometry_worsened" not in one


def test_last_blue_summary_counts_reacquisition_and_effective_temporal_pairs():
    rows = [
        _audit_row(40, target=None, threat="Blue4", red_kills_pre=3,
                   router=(.2, .8), temporal_mask_valid=True, temporal_weight=0.0,
                   temporal_effective=False),
        _audit_row(41, target="Blue4", threat="Blue4", red_kills_pre=3,
                   router=(.8, .2), target_switch=True,
                   engagement_target_change_type="acquisition",
                   teacher_mode_switch=True, temporal_mask_valid=True,
                   temporal_weight=.5, temporal_effective=True,
                   engagement_exact_gate=True, engagement_attack_streak=2),
        _audit_row(42, target=None, threat=None, red_kills_pre=3,
                   router=(.8, .2), target_switch=True,
                   engagement_target_change_type="loss", threat_switch=True,
                   threat_target_change_type="loss", temporal_mask_valid=False,
                   temporal_weight=0.0, temporal_effective=False),
    ]
    episodes = [{"episode": 0, "outcome": "draw", "red_attack_kills": 3,
                 "environment_seed": 1000}]
    result = last_blue_analysis(rows, episodes)
    assert len(result) == 1
    summary = result[0]
    assert summary["target_acquisition_count"] == 1
    assert summary["target_loss_count"] == 1
    assert summary["threat_target_loss_count"] == 1
    assert summary["teacher_mode_switch_count"] == 1
    assert summary["temporal_effective_rate"] == pytest.approx(1 / 3)
    assert summary["engagement_mode_occupancy"] == pytest.approx(2 / 3)


def test_evidence_sections_never_assign_automatic_strength_labels():
    sections = evidence_sections({"rows": 1}, {"UAV1": {"states": 1}}, [], [], [], 100)
    text = str(sections)
    assert "evidence_strength" not in text
    assert not any(label in text for label in ("strong", "moderate", "weak"))
    assert sections["router_semantic_alignment"]["data_status"] == "available"


def test_one_episode_semantic_audit_matches_formal_natural_policy_evaluator(tmp_path):
    trainer = HAPPOTrainer(short_v310(max_steps=4), tacm_config())
    try:
        loaded = {
            "checkpoint": tmp_path / "synthetic.pt", "payload": {"trainer_config": trainer.config},
            "environment_config": trainer.environment_config, "actors": trainer.actors,
            "training_seed": 7, "sampled_steps": 2_000_000,
        }
        _, _, audited = run_episode(
            loaded, episode=0, environment_seed=1000, action_seed=2000,
            action_mode="stochastic", profile="main", device="cpu",
        )
        formal = evaluate_actors(
            trainer.actors, trainer.environment_config, episodes=1, profile="main",
            seed=1000, device="cpu", deterministic=False, action_seed=2000,
        )[0]
        assert audited["outcome"] == formal["outcome"]
        assert audited["episode_length"] == formal["episode_length"]
        assert audited["red_attack_kills"] == formal["red_attack_kills"]
        assert audited["blue_attack_kills"] == formal["blue_attack_kills"]
        assert audited["mav_survived"] == formal["mav_survived"]
        assert audited["uav_survivors"] == formal["red_uav_survivors"]
    finally:
        trainer.close()
