from __future__ import annotations

import csv
import json
import math
from copy import deepcopy

import numpy as np
import pytest
import torch

import tools.audit_combat_failures as failure_audit
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.networks import IndependentActors
from env.mavuav import (
    BLUE_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.models import AircraftState
from env.reward import situation_reward
from tools.audit_combat_failures import (
    EPISODE_FIELDS, _classify_draw, _phase_records, _record_red_kill_steps,
    _reward_credit_records, _survivor_tail_visibility, actor_distribution_metadata, audit_actors,
    blue_action_diagnostics, blue_geometry_diagnostics,
    load_vanilla_baseline_checkpoint, summarize_audit, write_audit,
)


def short_config(steps: int = 3):
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = steps
    return config


def synthetic_row(outcome: str, kills: int) -> dict:
    return {
        "outcome": outcome, "red_attack_kills": kills, "blue_attack_kills": 1,
        "episode_length": 75, "episode_return": float(kills), "mav_survived": 1,
        "red_uav_survivors": 2, "first_kill_step": 10 if kills >= 1 else None,
        "second_kill_step": 20 if kills >= 2 else None, "third_kill_step": 30 if kills >= 3 else None,
        "fourth_kill_step": 40 if kills >= 4 else None, "post_third_steps_after_kill": 45 if kills >= 3 else None,
        "post_third_visible_fraction": 1.0 if kills == 3 else None,
        "post_third_longest_invisible_streak": 0 if kills == 3 else None,
        "post_third_min_distance": 2000.0 if kills == 3 else None,
        "post_third_final_distance": 2500.0 if kills == 3 else None,
        "post_third_mean_best_closing_rate": 10.0 if kills == 3 else None,
        "post_third_max_streak": 2 if kills == 3 else None,
        "steps_after_last_kill": 75 if kills == 0 else 75 - 10 * kills,
        "tail_min_survivor_visible_fraction": 1.0,
        "tail_max_survivor_invisible_streak": 0,
        "blue_target_id": '{"UAV1": 1}', "blue_target_is_MAV": 0.0,
        "blue_boundary_recovery_active": 0.0, "blue_horizontal_recovery_active": 0.0,
        "reward_credit_samples": 4,
        "reward_best_equals_closest_count": 3, "reward_best_closing_comparable_samples": 3,
        "reward_best_equals_best_closing_count": 2,
        "post_last_reward_credit_samples": 4, "post_last_reward_best_equals_closest_count": 3,
        "post_last_reward_best_closing_comparable_samples": 3,
        "post_last_reward_best_equals_best_closing_count": 2,
        "failure_classification": "OTHER" if outcome == "draw" else "",
    }


def test_deterministic_audit_matches_formal_evaluator_for_fixed_seeds():
    torch.manual_seed(17)
    actors = IndependentActors(hidden_dim=8)
    config = short_config(3)
    formal = evaluate_actors(actors, config, 3, "main", seed=1000, device="cpu")
    rows, _ = audit_actors(actors, config, 3, "main", seed=1000, device="cpu", policy_mode="deterministic")
    for expected, actual in zip(formal, rows):
        assert actual["outcome"] == expected["outcome"]
        assert actual["episode_length"] == expected["episode_length"]
        assert actual["red_attack_kills"] == expected["red_attack_kills"]
        assert actual["blue_attack_kills"] == expected["blue_attack_kills"]
        assert np.isclose(actual["episode_return"], expected["episode_return"])


def test_red_kill_steps_count_unique_successful_blue_deaths():
    steps = []
    _record_red_kill_steps(steps, {"death_causes": {"Blue2": "red_attack", "Blue1": "red_attack", "UAV1": "blue_attack"}}, 12)
    _record_red_kill_steps(steps, {"death_causes": {"Blue3": "red_attack"}}, 19)
    assert steps == [12, 12, 19]


def test_kill_histograms_and_completion_probabilities_are_exact():
    rows = [synthetic_row("draw", 0), synthetic_row("draw", 2), synthetic_row("draw", 3), synthetic_row("red", 4), synthetic_row("blue", 1)]
    summary = summarize_audit(rows, "deterministic", 1000)
    assert summary["red_kill_histogram"] == {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1}
    assert summary["draw_kill_histogram"] == {"0": 1, "1": 0, "2": 1, "3": 1, "4": 0}
    assert summary["red_win_kill_histogram"]["4"] == 1
    completion = summary["task_completion_structure"]
    assert completion == {
        "P_K_ge_1": 0.8, "P_K_ge_2": 0.6, "P_K_ge_3": 0.4, "P_K_eq_4": 0.2,
        "P_K_ge_2_given_K_ge_1": 0.75, "P_K_ge_3_given_K_ge_2": 2 / 3,
        "P_K_eq_4_given_K_ge_3": 0.5, "P_Blue_win": 0.2,
    }
    assert summary["draw_kill_distribution_fraction"] == {"0": 1 / 3, "1": 0.0, "2": 1 / 3, "3": 1 / 3}


def test_full_geometry_requires_one_identical_pair_and_reads_real_streak():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=1)
    for red_id in ("UAV2", "UAV3"):
        env.entities[red_id].state.alive = False
    env.entities["Blue1"].state = AircraftState(2000, 0, 5000, 225, 0, np.pi / 2, True)
    env.entities["MAV"].state = AircraftState(0, 0, 5000, 325, 0, 0, True)
    env.entities["UAV1"].state = AircraftState(2000, -2000, 5000, 225, 0, 0, True)
    env._attack_streak[("MAV", "Blue1")] = 2
    diagnostic = blue_geometry_diagnostics(env, "Blue1", {})
    assert diagnostic["any_pair_in_distance_window"]
    assert diagnostic["any_pair_ATA_ok"] and diagnostic["any_pair_AA_ok"]
    assert not diagnostic["any_pair_full_geometry_ok"]
    assert diagnostic["max_red_attack_streak"] == 2


def test_reward_best_closest_and_closing_identities_are_pairwise_correct():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=2)
    for red_id in ("UAV2", "UAV3"):
        env.entities[red_id].state.alive = False
    env.entities["Blue1"].state = AircraftState(2000, 0, 5000, 225, 0, np.pi, True)
    env.entities["MAV"].state = AircraftState(0, 0, 5000, 325, 0, 0, True)
    env.entities["UAV1"].state = AircraftState(-1000, 0, 5000, 225, 0, 0, True)
    previous = {("MAV", "Blue1"): 2100.0, ("UAV1", "Blue1"): 4000.0}
    diagnostic = blue_geometry_diagnostics(env, "Blue1", previous)
    expected_reward_best = max(
        ("MAV", "UAV1"),
        key=lambda red_id: situation_reward(env.entities[red_id].state, env.entities["Blue1"].state),
    )
    assert diagnostic["closest_red_id"] == "MAV"
    assert diagnostic["best_closing_red_id"] == "UAV1"
    assert diagnostic["reward_best_red_id"] == expected_reward_best
    assert diagnostic["reward_best_equals_closest"] == (expected_reward_best == "MAV")
    assert diagnostic["reward_best_equals_best_closing"] == (expected_reward_best == "UAV1")


def test_invisible_blue_keeps_ground_truth_geometry_but_is_not_reward_credit():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=2)
    env.team_visible = lambda blue_id: False
    diagnostic = blue_geometry_diagnostics(env, "Blue1", {})
    assert not diagnostic["team_visible"]
    assert diagnostic["pairs"] and diagnostic["min_distance"] > 0.0
    assert diagnostic["reward_best_red_id"] in RED_IDS
    assert _reward_credit_records([diagnostic]) == []


def test_reward_credit_records_filter_visibility_for_closest_and_closing_samples():
    records = [
        {
            "decision_step": 1, "blue_id": "Blue1", "team_visible": True,
            "reward_best_red_id": "MAV", "closest_red_id": "MAV",
            "best_closing_red_id": "UAV1", "reward_best_equals_closest": True,
            "reward_best_equals_best_closing": False,
        },
        {
            "decision_step": 1, "blue_id": "Blue2", "team_visible": False,
            "reward_best_red_id": "UAV2", "closest_red_id": "MAV",
            "best_closing_red_id": "UAV2", "reward_best_equals_closest": False,
            "reward_best_equals_best_closing": True,
        },
    ]
    visible = _reward_credit_records(records)
    comparable = [record for record in visible if record["best_closing_red_id"] is not None]
    assert len(visible) == 1
    assert sum(record["reward_best_equals_closest"] for record in visible) == 1
    assert len(comparable) == 1
    assert sum(record["reward_best_equals_best_closing"] for record in comparable) == 0
    assert np.mean([record["reward_best_equals_closest"] for record in visible]) == 1.0


def test_post_last_reward_credit_filters_invisible_records():
    records = [
        {"decision_step": 4, "team_visible": True},
        {"decision_step": 6, "team_visible": True},
        {"decision_step": 6, "team_visible": False},
        {"decision_step": 7, "team_visible": False},
    ]
    tail = _phase_records(records, 5)
    assert len(tail) == 3
    assert _reward_credit_records(tail) == [records[1]]


def test_episode_reward_credit_fields_match_formal_visible_blue_set(monkeypatch):
    class Blue1VisibleEnv(HeterogeneousMAVUAVAirCombatEnv):
        def team_visible(self, blue_id):
            return blue_id == "Blue1" and self.entities[blue_id].state.alive

    monkeypatch.setattr(failure_audit, "HeterogeneousMAVUAVAirCombatEnv", Blue1VisibleEnv)
    actors = IndependentActors(hidden_dim=8)
    rows, _ = audit_actors(
        actors, short_config(1), 1, "main", seed=41,
        device="cpu", policy_mode="deterministic",
    )
    row = rows[0]
    assert row["red_attack_kills"] == 0
    assert row["reward_credit_samples"] == 1
    assert row["post_last_reward_credit_samples"] == 1

    env = Blue1VisibleEnv(short_config(1), profile="main")
    env.reset(seed=41)
    expected = max(
        situation_reward(env.entities[red_id].state, env.entities["Blue1"].state)
        for red_id in RED_IDS
    )
    assert env._team_situation_reward() == pytest.approx(expected)


@pytest.mark.parametrize(
    "altitude,theta,expected_recovery",
    [(5000.0, 0.0, False), (1500.0, -0.8, True), (19_500.0, 0.6, True)],
)
def test_blue_action_audit_matches_formal_direct_controller_state(altitude, theta, expected_recovery):
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=3)
    blue = env.entities["Blue1"]
    blue.state.h, blue.state.theta = altitude, theta
    diagnostic = blue_action_diagnostics(env, "Blue1")
    actual = env.blue_policy.action(blue, {red_id: env.entities[red_id] for red_id in RED_IDS})
    assert diagnostic["blue_target_id"] == env.blue_policy.select_target(
        blue, {red_id: env.entities[red_id] for red_id in RED_IDS},
    ).aircraft_id
    assert diagnostic["blue_boundary_recovery_active"] is expected_recovery
    assert np.isfinite(actual).all()


def test_output_csv_json_fields_are_complete_and_finite(tmp_path):
    actors = IndependentActors(hidden_dim=8)
    rows, summary = audit_actors(actors, short_config(2), 1, "main", seed=7, device="cpu", policy_mode="stochastic")
    csv_path, json_path = write_audit(tmp_path / "audit", rows, summary)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert tuple(saved[0]) == EPISODE_FIELDS
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["episodes"] == 1 and loaded["policy_mode"] == "stochastic"

    def assert_finite(value):
        if isinstance(value, dict):
            for nested in value.values(): assert_finite(nested)
        elif isinstance(value, list):
            for nested in value: assert_finite(nested)
        elif isinstance(value, float):
            assert math.isfinite(value)

    assert_finite(loaded)


def test_checkpoint_loader_rejects_nonbaseline_contract(tmp_path):
    actors = IndependentActors(hidden_dim=8)
    checkpoint = tmp_path / "bad.pt"
    torch.save({
        "environment_version": ENVIRONMENT_VERSION,
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla", "critic_variant": "mlp", "method_variant": "agp",
        "trainer_config": {"hidden_dim": 8}, "actors": actors.state_dict(),
    }, checkpoint)
    with pytest.raises(RuntimeError, match="method_variant='baseline'"):
        load_vanilla_baseline_checkpoint(checkpoint, "cpu")


@pytest.mark.parametrize("version", ["heterogeneous_mavuav_4v4_v3_2", "heterogeneous_mavuav_4v4_v3_3"])
def test_failure_audit_rejects_pre_v34_checkpoint(tmp_path, version):
    actors = IndependentActors(hidden_dim=8)
    checkpoint = tmp_path / "v32.pt"
    torch.save({
        "environment_version": version,
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla", "critic_variant": "mlp", "method_variant": "baseline",
        "trainer_config": {"hidden_dim": 8}, "actors": actors.state_dict(),
    }, checkpoint)
    with pytest.raises(RuntimeError, match="incompatible checkpoint environment contract"):
        load_vanilla_baseline_checkpoint(checkpoint, "cpu")


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"third_kill_step": 65, "final_max_attack_streak": 2}, "LATE_PROGRESS / POSSIBLE_HORIZON"),
        ({"tail_max_survivor_invisible_streak": 12}, "TARGET_LOSS"),
        ({"tail_visible_fraction": 1.0, "tail_far_fraction": 0.8, "tail_fraction_positive_closing": 0.2}, "CANNOT_CLOSE"),
        ({"tail_distance_window_fraction": 0.5, "tail_full_geometry_fraction": 0.0}, "BAD_GEOMETRY"),
        ({"tail_max_streak": 1}, "STREAK_INTERRUPTED"),
    ],
)
def test_failure_classification_priority_is_mechanical(updates, expected):
    row = {
        "first_kill_step": None, "second_kill_step": None, "third_kill_step": None, "fourth_kill_step": None,
        "final_max_attack_streak": 0, "tail_full_geometry_fraction": 0.0,
        "tail_longest_all_invisible_streak": 0, "tail_max_survivor_invisible_streak": 0,
        "tail_visible_fraction": 0.0,
        "tail_far_fraction": 0.0, "tail_fraction_positive_closing": 0.0,
        "tail_distance_window_fraction": 0.0, "tail_max_streak": 0,
    }
    row.update(updates)
    assert _classify_draw(row) == expected


def test_per_survivor_visibility_detects_partial_target_loss_and_stable_tie_break():
    records = []
    for step in range(1, 13):
        records.extend([
            {"decision_step": step, "blue_id": "Blue1", "team_visible": True},
            {"decision_step": step, "blue_id": "Blue2", "team_visible": False},
        ])
    per_survivor, worst = _survivor_tail_visibility(records, ["Blue1", "Blue2"], 0)
    assert per_survivor[0]["visible_fraction"] == 1.0
    assert per_survivor[1]["longest_invisible_streak"] == 12
    assert worst["blue_id"] == "Blue2"
    row = {
        "first_kill_step": None, "second_kill_step": None, "third_kill_step": None,
        "fourth_kill_step": None, "final_max_attack_streak": 0,
        "tail_full_geometry_fraction": 0.0, "tail_longest_all_invisible_streak": 0,
        "tail_max_survivor_invisible_streak": 12, "tail_visible_fraction": 0.5,
        "tail_far_fraction": 0.0, "tail_fraction_positive_closing": 0.0,
        "tail_distance_window_fraction": 0.0, "tail_max_streak": 0,
    }
    assert _classify_draw(row) == "TARGET_LOSS"


def test_phase_starts_strictly_after_kill_and_three_kill_visibility_definitions_match():
    records = [
        {"decision_step": step, "blue_id": "Blue4", "team_visible": step != 22}
        for step in range(19, 24)
    ]
    post_kill = _phase_records(records, 20, "Blue4")
    assert [record["decision_step"] for record in post_kill] == [21, 22, 23]
    per_survivor, _ = _survivor_tail_visibility(records, ["Blue4"], 20)
    assert per_survivor[0]["visible_fraction"] == pytest.approx(2 / 3)
    assert per_survivor[0]["longest_invisible_streak"] == 1


def test_blue_controller_audit_reports_target_and_emergency_flags():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=11)
    blue = env.entities["Blue1"]
    blue.state.h = 1500.0
    blue.state.theta = -0.8
    diagnostic = blue_action_diagnostics(env, "Blue1")
    assert set(diagnostic) == {
        "blue_target_id", "blue_target_is_MAV",
        "blue_boundary_recovery_active", "blue_horizontal_recovery_active",
    }
    assert diagnostic["blue_target_id"] in RED_IDS
    assert diagnostic["blue_boundary_recovery_active"]


def test_post_last_reward_credit_is_sample_weighted_and_grouped_by_kill_count():
    short = synthetic_row("draw", 0)
    long = synthetic_row("draw", 2)
    short.update({
        "post_last_reward_credit_samples": 1, "post_last_reward_best_equals_closest_count": 1,
        "post_last_reward_best_closing_comparable_samples": 1,
        "post_last_reward_best_equals_best_closing_count": 1,
    })
    long.update({
        "post_last_reward_credit_samples": 9, "post_last_reward_best_equals_closest_count": 0,
        "post_last_reward_best_closing_comparable_samples": 3,
        "post_last_reward_best_equals_best_closing_count": 0,
    })
    summary = summarize_audit([short, long], "deterministic", 1000)
    assert summary["reward_credit"]["post_last_reward_best_equals_closest_fraction"] == 0.1
    assert summary["reward_credit"]["post_last_reward_best_equals_best_closing_fraction"] == 0.25
    assert summary["reward_credit_by_kill_count"]["0"]["post_last_reward_best_equals_closest_fraction"] == 1.0
    assert summary["reward_credit_by_kill_count"]["2"]["post_last_reward_best_equals_closest_fraction"] == 0.0


def test_reward_credit_group_with_no_visible_post_last_samples_is_null():
    row = synthetic_row("draw", 3)
    row.update({
        "post_last_reward_credit_samples": 0,
        "post_last_reward_best_equals_closest_count": 0,
        "post_last_reward_best_closing_comparable_samples": 0,
        "post_last_reward_best_equals_best_closing_count": 0,
    })
    summary = summarize_audit([row], "deterministic", 1000)
    group = summary["reward_credit_by_kill_count"]["3"]
    assert group["post_last_reward_best_equals_closest_fraction"] is None
    assert group["post_last_reward_best_equals_best_closing_fraction"] is None


def test_stochastic_audit_is_reproducible_for_same_seed_and_reports_actor_std():
    torch.manual_seed(29)
    actors = IndependentActors(hidden_dim=8)
    config = short_config(2)
    rows_a, summary_a = audit_actors(actors, config, 2, "main", seed=77, device="cpu", policy_mode="stochastic")
    rows_b, summary_b = audit_actors(actors, config, 2, "main", seed=77, device="cpu", policy_mode="stochastic")
    keys = ("outcome", "episode_length", "red_attack_kills", "blue_attack_kills", "episode_return")
    assert [[row[key] for key in keys] for row in rows_a] == [[row[key] for key in keys] for row in rows_b]
    assert summary_a["actor_policy_distribution"] == summary_b["actor_policy_distribution"]
    metadata = actor_distribution_metadata(actors)
    assert [entry["agent_id"] for entry in metadata["actors"]] == list(RED_IDS)
    assert all(len(entry["log_std"]) == len(entry["std"]) == 3 for entry in metadata["actors"])


def test_first_pair_appearance_is_omitted_from_closing_denominator():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=31)
    first = blue_geometry_diagnostics(env, "Blue1", {})
    assert first["best_closing_red_id"] is None
    assert first["best_closing_rate"] is None
