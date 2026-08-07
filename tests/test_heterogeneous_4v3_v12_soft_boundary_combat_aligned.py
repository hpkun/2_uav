from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_combat.config import load_config
from uav_combat.environment_4v3_v12 import (
    DEATH_LOCK_V12,
    REWARD_COMPONENT_KEYS_V12,
    FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv,
    aggregate_team_reward_v12,
    reward_group_totals_v12,
)
from uav_combat.happo.evaluation_4v3 import evaluate_rule_vs_rule_4v3
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer, compute_best_score_4v3, summarize_4v3_episodes
from uav_combat.mappo.vector_env_4v3_v12 import make_combat_vector_env_4v3_v12
from uav_combat.scenario_4v3_v12 import validate_heterogeneous_4v3_v12_config


ENV_CONFIG = Path("configs/heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")
TRAIN_CONFIG = Path("configs/happo_heterogeneous_4v3_main_v12_soft_boundary_combat_aligned.yaml")


def make_env(seed: int = 42):
    env = FunctionalHeterogeneous4v3V12SoftBoundaryCombatAlignedEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def zero_actions():
    return {f"red_{index}": np.zeros(3, dtype=np.float32) for index in range(4)}


def test_v12_identity_and_reward_contract():
    env = make_env()
    assert env.variant == "functional_heterogeneous_4v3_v12_soft_boundary_combat_aligned"
    assert env.reward_contract_version == "v12_soft_boundary_combat_aligned"
    assert env.config["combat"]["reward_mode"] == "functional_heterogeneous_4v3_team_v12"


def test_v12_config_is_frozen_and_validates():
    cfg = load_config(ENV_CONFIG)
    validate_heterogeneous_4v3_v12_config(cfg)
    boundary = cfg["boundary"]
    assert boundary["mode"] == "soft_containment"
    assert boundary["horizontal_soft_margin"] == 800.0
    assert boundary["altitude_soft_margin"] == 250.0
    assert boundary["max_recovery_blend"] == 0.85
    assert boundary["hard_horizontal_buffer"] == 50.0
    assert boundary["hard_altitude_buffer"] == 20.0


def test_v12_shared_combat_profile_and_roles():
    env = make_env()
    combats = [env._by_id(aid) for aid in ("red_1", "red_2", "red_3", "blue_0", "blue_1", "blue_2")]
    assert len({id(ac.spec) for ac in combats}) == 1
    assert all(ac.role == "combat" and ac.can_attack for ac in combats)
    assert env._by_id("red_0").role == "support"
    assert env._by_id("red_0").can_attack is False


def test_v12_observation_and_global_state_dimensions():
    env = make_env()
    obs, state, masks = env._observations()
    assert obs.shape == (7, 118)
    assert state.shape == (70,)
    assert masks.shape == (7,)
    assert np.isfinite(obs).all() and np.isfinite(state).all()


def test_v12_mirrored_reset_preserves_paired_combat_dynamics():
    env = make_env(7)
    for red_id, blue_id in zip(("red_1", "red_2", "red_3"), ("blue_0", "blue_1", "blue_2")):
        red = env._by_id(red_id).state
        blue = env._by_id(blue_id).state
        assert red.v == pytest.approx(blue.v)
        assert red.altitude == pytest.approx(blue.altitude)
        assert np.cos(red.psi - blue.psi) < -0.98


def test_v12_reward_component_keys_are_complete():
    env = make_env()
    result = env.step(zero_actions())
    reward, info = result[3], result[6]
    assert set(info["reward_components"]) == set(REWARD_COMPONENT_KEYS_V12)
    assert np.isfinite(reward)
    assert np.isfinite(list(info["reward_components"].values())).all()


def test_v12_reward_groups_are_exactly_six_mutually_exclusive_groups():
    groups = reward_group_totals_v12({key: 0.0 for key in REWARD_COMPONENT_KEYS_V12})
    assert set(groups) == {"mission", "combat_evt", "support_evt", "half_lock_evt", "boundary_evt", "dense"}


def test_v12_team_reward_reconstructs_from_groups_once():
    components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
    components.update({
        "mission_outcome_reward": 1.0,
        "blue_kill_event_reward": 8.0,
        "support_unique_detection_reward": 0.02,
        "support_cue_to_half_lock_reward": 0.5,
        "boundary_event_penalty": -0.1,
        "total_dense_reward": 0.05,
    })
    assert aggregate_team_reward_v12(components) == pytest.approx(9.47)
    assert aggregate_team_reward_v12(components) != pytest.approx(9.52)


def test_v12_dense_is_clipped_once_and_boundary_is_outside_dense():
    env = make_env()
    components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V12}
    components.update({
        "combat_geometry_progress_reward": 0.04,
        "combat_lock_progress_reward": 0.04,
        "support_formation_progress_reward": 0.04,
        "total_dense_reward": 0.05,
        "boundary_event_penalty": -0.1,
    })
    assert aggregate_team_reward_v12(components) == pytest.approx(-0.05)
    assert reward_group_totals_v12(components)["dense"] == pytest.approx(0.05)
    assert reward_group_totals_v12(components)["boundary_evt"] == pytest.approx(-0.1)


def test_soft_recovery_is_inactive_in_center():
    env = make_env()
    aircraft = env._by_id("red_1")
    corrected, blend = env._boundary_recovery_action(aircraft, np.array([0.2, -0.3, 0.4], dtype=np.float32))
    assert blend == 0.0
    assert corrected.tolist() == pytest.approx([0.2, -0.3, 0.4])


def test_soft_recovery_activates_inside_horizontal_margin():
    env = make_env()
    aircraft = env._by_id("red_1")
    aircraft.state.x = 19_500.0
    corrected, blend = env._boundary_recovery_action(aircraft, np.zeros(3, dtype=np.float32))
    assert 0.0 < blend < 0.85
    assert corrected[0] < 0.0


def test_soft_recovery_activates_inside_altitude_margin():
    env = make_env()
    aircraft = env._by_id("red_1")
    aircraft.state.z = -5_850.0
    corrected, blend = env._boundary_recovery_action(aircraft, np.zeros(3, dtype=np.float32))
    assert 0.0 < blend < 0.85
    assert corrected[1] < 0.0


def test_soft_recovery_preserves_speed_action_exactly():
    env = make_env()
    aircraft = env._by_id("red_1")
    aircraft.state.x = 19_900.0
    action = np.array([0.1, 0.2, -0.73], dtype=np.float32)
    corrected, _ = env._boundary_recovery_action(aircraft, action)
    assert corrected[2] == pytest.approx(action[2])


@pytest.mark.parametrize("aircraft_id", ["red_0", "red_1", "blue_0"])
def test_all_roles_use_one_boundary_recovery_helper(aircraft_id):
    env = make_env()
    aircraft = env._by_id(aircraft_id)
    aircraft.state.x = 19_900.0
    _, blend = env._boundary_recovery_action(aircraft, np.zeros(3, dtype=np.float32))
    assert blend > 0.0


def test_recovery_heading_points_to_battlefield_center():
    env = make_env()
    state = env._by_id("red_1").state
    state.x, state.y = 19_900.0, 100.0
    state.psi = 0.0
    assert env._recovery_heading(state) == pytest.approx(np.arctan2(-100.0, -19_900.0))


def test_recovery_pitch_targets_altitude_center():
    env = make_env()
    state = env._by_id("red_1").state
    state.z = -5_900.0
    assert env._recovery_pitch(state) < 0.0
    state.z = -1000.0
    assert env._recovery_pitch(state) > 0.0


@pytest.mark.parametrize("field,value", [("x", 20_100.0), ("y", -20_100.0)])
def test_hard_horizontal_projection_keeps_aircraft_alive(field, value):
    env = make_env()
    aircraft = env._by_id("red_1")
    setattr(aircraft.state, field, value)
    assert env._project_hard_boundary(aircraft)
    assert aircraft.state.alive
    assert abs(aircraft.state.x) <= 19_950.0
    assert abs(aircraft.state.y) <= 19_950.0


@pytest.mark.parametrize("altitude", [400.0, 6100.0])
def test_hard_altitude_projection_keeps_aircraft_alive(altitude):
    env = make_env()
    aircraft = env._by_id("blue_0")
    aircraft.state.z = -altitude
    assert env._project_hard_boundary(aircraft)
    assert aircraft.state.alive
    assert 520.0 <= aircraft.state.altitude <= 5980.0


def test_hard_projection_points_heading_and_pitch_inward():
    env = make_env()
    aircraft = env._by_id("red_1")
    aircraft.state.x = 20_100.0
    aircraft.state.psi = 0.0
    env._project_hard_boundary(aircraft)
    assert np.cos(aircraft.state.psi) < 0.0
    assert abs(aircraft.state.theta) <= np.pi / 2


def test_hard_contact_is_recorded_by_team_without_boundary_death():
    env = make_env()
    aircraft = env._by_id("red_0")
    aircraft.state.x = 20_100.0
    env._project_hard_boundary(aircraft)
    assert env._episode_metrics["support_boundary_hard_contacts"] == 1.0
    assert env._death_causes["red_0"] == 0


def test_step_boundary_contact_does_not_create_death_or_kill():
    env = make_env()
    env._by_id("red_1").state.x = 19_990.0
    env._by_id("red_1").state.psi = 0.0
    result = env.step(zero_actions())
    assert not result[4]
    assert result[6]["episode_summary"] is None
    assert env._by_id("red_1").state.alive
    assert env._attack_kills == {"red": 0, "blue": 0}


def test_red_boundary_penalty_is_capped_at_minus_point_four():
    env = make_env()
    for key in ("red_boundary_hard_contacts_step", "support_boundary_hard_contacts_step"):
        env._episode_metrics[key] = 10.0
    _, components = env._compute_reward(
        dict(env.targets), {}, dict(env.lock_progress), {}, set(), {}, env._direct_visible_ids()
    )
    assert components["boundary_event_penalty"] == pytest.approx(-0.4)


def test_blue_boundary_contact_is_not_in_red_reward():
    env = make_env()
    env._episode_metrics["blue_boundary_hard_contacts_step"] = 3.0
    _, components = env._compute_reward(
        dict(env.targets), {}, dict(env.lock_progress), {}, set(), {}, env._direct_visible_ids()
    )
    assert components["boundary_event_penalty"] == pytest.approx(0.0)


def test_support_automatic_reward_stays_small_without_combat_progress():
    env = make_env()
    env._episode_metrics["support_unique_detection_events_step"] = 3.0
    env._episode_metrics["support_cue_to_direct_events_step"] = 3.0
    _, components = env._compute_reward(
        dict(env.targets), {}, dict(env.lock_progress), {}, set(), {}, env._direct_visible_ids()
    )
    automatic = components["support_unique_detection_reward"] + components["support_cue_to_direct_reward"]
    assert automatic == pytest.approx(0.12)
    assert automatic < 0.30
    assert components["support_assisted_kill_reward"] == 0.0


def test_lock_kill_is_the_only_combat_death_path():
    env = make_env()
    for aircraft in env.aircraft:
        aircraft.state.alive = False
    env._by_id("red_1").state.alive = True
    env._by_id("blue_0").state.alive = True
    env._by_id("red_1").state.x = 0.0
    env._by_id("red_1").state.y = 0.0
    env._by_id("red_1").state.psi = 0.0
    env._by_id("blue_0").state.x = 700.0
    env._by_id("blue_0").state.y = 0.0
    env._by_id("blue_0").state.psi = 0.0
    env.targets["red_1"] = "blue_0"
    env.lock_progress["red_1"] = 0.99
    direct = {ac.aircraft_id: set() for ac in env.aircraft}
    direct["red_1"] = {"blue_0"}
    _, killers = env._update_locks(direct)
    assert killers == {"blue_0": "red_1"}
    assert env._by_id("blue_0").state.alive


def test_strict_full_elimination_requires_three_red_lock_kills():
    env = make_env()
    for bid in ("blue_0", "blue_1", "blue_2"):
        env._by_id(bid).state.alive = False
        env._death_causes[bid] = DEATH_LOCK_V12
    env._attack_kills["red"] = 3
    assert env._terminal_result() == (True, "red", "red_full_elimination")
    summary = env._episode_summary("red", "red_full_elimination")
    assert summary["strict_full_elimination"] is True
    assert summary["strict_full_elimination_rate"] == 1.0
    assert summary["red_lock_kills"] == 3


def test_blue_alive_zero_with_wrong_red_lock_kills_raises_consistency_error():
    env = make_env()
    for bid in ("blue_0", "blue_1", "blue_2"):
        env._by_id(bid).state.alive = False
        env._death_causes[bid] = DEATH_LOCK_V12
    env._attack_kills["red"] = 2
    with pytest.raises(RuntimeError, match="full-elimination consistency"):
        env._terminal_result()


def test_full_elimination_requires_red_combat_survivor():
    env = make_env()
    for bid in ("blue_0", "blue_1", "blue_2"):
        env._by_id(bid).state.alive = False
        env._death_causes[bid] = DEATH_LOCK_V12
    for cid in ("red_1", "red_2", "red_3"):
        env._by_id(cid).state.alive = False
        env._death_causes[cid] = DEATH_LOCK_V12
    env._attack_kills.update({"red": 3, "blue": 3})
    assert env._terminal_result()[2] == "mutual_elimination_draw"


def test_non_lock_death_is_reported_when_blue_is_still_alive():
    env = make_env()
    env._by_id("blue_0").state.alive = False
    env._death_causes["blue_0"] = 1
    summary = env._episode_summary(None, None)
    assert summary["non_lock_blue_death_count"] == 1
    assert summary["full_elimination_consistency_pass"] is True


def test_v12_state_dict_roundtrip_restores_boundary_and_lock_state():
    env = make_env()
    env.lock_progress["red_1"] = 0.37
    env._episode_metrics["red_boundary_hard_contacts"] = 2.0
    state = env.state_dict()
    restored = make_env(13)
    restored.load_state_dict(state)
    assert restored.lock_progress["red_1"] == pytest.approx(0.37)
    assert restored._episode_metrics["red_boundary_hard_contacts"] == pytest.approx(2.0)


def test_v12_state_rejects_other_variant():
    env = make_env()
    state = env.state_dict()
    state["environment_variant"] = "functional_heterogeneous_4v3_v11_target_lock_support_cue"
    with pytest.raises(ValueError, match="variant mismatch"):
        env.load_state_dict(state)


def test_v12_vector_local_shapes_and_step():
    vec = make_combat_vector_env_4v3_v12(ENV_CONFIG, 2, 0, seed=42)
    try:
        obs, state, masks = vec.reset()
        assert obs.shape == (2, 7, 118)
        assert state.shape == (2, 70)
        result = vec.step(np.zeros((2, 4, 3), dtype=np.float32))
        assert result.team_rewards.shape == (2,)
        assert result.red_reward_components.shape == (2, len(REWARD_COMPONENT_KEYS_V12))
    finally:
        vec.close()


def test_v12_summary_exposes_strict_and_boundary_metrics():
    env = make_env()
    record = env._episode_summary("draw", "timeout_draw")
    summary = summarize_4v3_episodes([record])
    for key in (
        "task_win_rate", "strict_full_elimination_rate", "full_elimination_consistency_pass",
        "non_lock_blue_death_count", "non_lock_red_combat_death_count",
        "red_boundary_soft_recovery_step_rate", "blue_boundary_soft_recovery_step_rate",
        "support_boundary_soft_recovery_step_rate", "red_boundary_hard_contacts",
        "blue_boundary_hard_contacts", "support_boundary_hard_contacts",
    ):
        assert key in summary


def test_v12_full_elimination_rate_alias_equals_strict_rate():
    env = make_env()
    for bid in ("blue_0", "blue_1", "blue_2"):
        env._by_id(bid).state.alive = False
        env._death_causes[bid] = DEATH_LOCK_V12
    env._attack_kills["red"] = 3
    summary = summarize_4v3_episodes([env._episode_summary("red", "red_full_elimination")])
    assert summary["full_elimination_rate"] == summary["strict_full_elimination_rate"] == 1.0


def test_v12_best_score_is_strict_lexicographic_order():
    strict = {
        "strict_full_elimination_rate": 1.0, "at_least_two_kill_rate": 1.0,
        "task_win_rate": 1.0, "any_kill_rate": 1.0, "mean_red_kills": 3.0,
        "mean_red_combat_survivors": 2.0, "support_assisted_kill_rate": 0.0,
        "mean_episode_length": 500.0,
    }
    high_reward_like = dict(strict, strict_full_elimination_rate=0.0, task_win_rate=1.0, mean_red_kills=3.0)
    assert compute_best_score_4v3(strict)[0] > compute_best_score_4v3(high_reward_like)[0]


def test_v12_best_score_prefers_two_kills_over_timeout_win():
    two_kills = {
        "strict_full_elimination_rate": 0.0, "at_least_two_kill_rate": 0.5,
        "task_win_rate": 0.0, "any_kill_rate": 1.0, "mean_red_kills": 2.0,
        "mean_red_combat_survivors": 1.0, "support_assisted_kill_rate": 0.0,
        "mean_episode_length": 500.0,
    }
    timeout = dict(two_kills, at_least_two_kill_rate=0.0, task_win_rate=1.0)
    assert compute_best_score_4v3(two_kills)[0] > compute_best_score_4v3(timeout)[0]


def test_v12_trainer_uses_separate_schedule_and_signature():
    cfg = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    cfg["training"].update({"num_envs": 1, "num_env_workers": 0, "total_env_steps": 8, "rollout_steps": 2})
    trainer = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    try:
        assert trainer.is_v12 is True
        assert trainer.schedule_env_steps == 3_000_000
        assert trainer.training_signature()["schedule_env_steps"] == 3_000_000
    finally:
        trainer.close()


def test_v11_without_schedule_defaults_to_total_budget():
    cfg = yaml.safe_load(Path("configs/happo_heterogeneous_4v3_main_v11_target_lock_support_cue.yaml").read_text(encoding="utf-8"))
    cfg["training"].update({"num_envs": 1, "num_env_workers": 0, "total_env_steps": 8, "rollout_steps": 2})
    trainer = HAPPO4v3Trainer("configs/heterogeneous_4v3_main_v11_target_lock_support_cue.yaml", cfg)
    try:
        assert trainer.is_v11 is True
        assert trainer.schedule_env_steps == 8
    finally:
        trainer.close()


def test_v12_rule_baseline_routes_to_v12_variant():
    summary = evaluate_rule_vs_rule_4v3(ENV_CONFIG, episodes=1, seed=20000, workers=1, red_policy="random")
    assert summary["episodes"] == 1.0
    assert summary["episode_records"][0]["environment_variant"] == "functional_heterogeneous_4v3_v12_soft_boundary_combat_aligned"
