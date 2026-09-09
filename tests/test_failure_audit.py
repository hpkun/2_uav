from __future__ import annotations

import csv
import json
import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.networks import IndependentActors
from env.mavuav import BLUE_IDS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.models import AircraftState
from env.reward import situation_reward
from tools.audit_combat_failures import (
    EPISODE_FIELDS, _classify_draw, _record_red_kill_steps, audit_actors,
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
        "recovery_candidate_count": 27, "recovery_changed_choice_count": 0,
        "max_recovery_guard": 3000.0, "reward_credit_samples": 4,
        "reward_best_equals_closest_count": 3, "reward_best_closing_comparable_samples": 3,
        "reward_best_equals_best_closing_count": 2, "failure_classification": "OTHER" if outcome == "draw" else "",
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
    assert summary["task_completion_structure"] == {
        "P_K_ge_1": 0.8, "P_K_ge_2": 0.6, "P_K_ge_3": 0.4, "P_K_eq_4": 0.2, "P_Blue_win": 0.2,
    }


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


@pytest.mark.parametrize("altitude,theta", [(5000.0, 0.0), (5000.0, -0.8), (18000.0, 0.6)])
def test_recovery_audit_safe_greedy_matches_formal_blue_policy(altitude, theta):
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=3)
    blue = env.entities["Blue1"]
    blue.state.h, blue.state.theta = altitude, theta
    diagnostic = blue_action_diagnostics(env, "Blue1")
    actual = env.blue_policy.action(blue, {red_id: env.entities[red_id] for red_id in RED_IDS})
    np.testing.assert_array_equal(diagnostic["actual_safe_greedy_action"], actual)
    assert diagnostic["one_step_rejected_count"] + diagnostic["altitude_rejected_count"] <= 27


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
        "environment_version": "heterogeneous_mavuav_4v4_v3_2",
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "actor_variant": "vanilla", "critic_variant": "mlp", "method_variant": "agp",
        "trainer_config": {"hidden_dim": 8}, "actors": actors.state_dict(),
    }, checkpoint)
    with pytest.raises(RuntimeError, match="method_variant='baseline'"):
        load_vanilla_baseline_checkpoint(checkpoint, "cpu")


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"third_kill_step": 65, "final_max_attack_streak": 2}, "LATE_PROGRESS / POSSIBLE_HORIZON"),
        ({"tail_longest_all_invisible_streak": 12}, "TARGET_LOSS"),
        ({"tail_visible_fraction": 1.0, "tail_far_fraction": 0.8, "tail_fraction_positive_closing": 0.2}, "CANNOT_CLOSE"),
        ({"tail_distance_window_fraction": 0.5, "tail_full_geometry_fraction": 0.0}, "BAD_GEOMETRY"),
        ({"tail_max_streak": 1}, "STREAK_INTERRUPTED"),
    ],
)
def test_failure_classification_priority_is_mechanical(updates, expected):
    row = {
        "first_kill_step": None, "second_kill_step": None, "third_kill_step": None, "fourth_kill_step": None,
        "final_max_attack_streak": 0, "tail_full_geometry_fraction": 0.0,
        "tail_longest_all_invisible_streak": 0, "tail_visible_fraction": 0.0,
        "tail_far_fraction": 0.0, "tail_fraction_positive_closing": 0.0,
        "tail_distance_window_fraction": 0.0, "tail_max_streak": 0,
    }
    row.update(updates)
    assert _classify_draw(row) == expected
