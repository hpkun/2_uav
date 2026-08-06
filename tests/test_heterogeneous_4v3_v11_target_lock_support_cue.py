from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.environment_4v3_v11 import (
    BLUE_IDS_V11,
    DEATH_LOCK_V11,
    RED_COMBAT_IDS_V11,
    REWARD_COMPONENT_KEYS_V11,
    FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv,
    distance_score_v11,
    lock_quality_v11,
)
from uav_combat.happo.trainer_4v3 import compute_best_score_4v3, summarize_4v3_episodes
from scripts.train_happo_4v3 import format_eval_log_v11, format_train_log_v11


ENV_CONFIG = "configs/heterogeneous_4v3_main_v11_target_lock_support_cue.yaml"
V9_CONFIG = Path("configs/heterogeneous_4v3_main_v9.yaml")
V10_CONFIG = Path("configs/heterogeneous_4v3_main_v10_attack_funnel.yaml")


def make_env(seed: int = 11) -> FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv:
    env = FunctionalHeterogeneous4v3V11TargetLockSupportCueEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def visibility(env, *, direct: set[str] | None = None):
    values = {aircraft.aircraft_id: set() for aircraft in env.aircraft}
    values["red_1"] = set(direct or set())
    return values


def test_v11_identity_and_shared_profile():
    cfg = load_config(ENV_CONFIG)
    env = make_env()
    combats = [env._by_id(aid) for aid in (*RED_COMBAT_IDS_V11, *BLUE_IDS_V11)]
    assert cfg["combat"]["reward_contract_version"] == "v11_target_lock_support_cue"
    assert len({id(ac.spec) for ac in combats}) == 1
    assert {ac.sensor_range for ac in combats} == {2500.0}
    assert all(ac.can_attack and ac.role == "combat" for ac in combats)
    assert len({env.profile[key] for key in ("lock_increment_scale", "lock_decay_per_step", "lock_kill_threshold")}) == 3


def test_v11_mirrored_reset_uses_paired_speed_altitude_and_heading():
    env = make_env(22)
    for red_id, blue_id in zip(RED_COMBAT_IDS_V11, BLUE_IDS_V11):
        red = env._by_id(red_id).state
        blue = env._by_id(blue_id).state
        assert red.v == pytest.approx(blue.v)
        assert red.altitude == pytest.approx(blue.altitude)
        assert np.cos(red.psi - blue.psi) < -0.98


def test_v11_observation_and_state_dimensions():
    env = make_env()
    observations, global_state, masks = env._observations()
    assert observations.shape == (7, 118)
    assert global_state.shape == (70,)
    assert masks.shape == (7,)
    assert np.isfinite(observations).all() and np.isfinite(global_state).all()


def test_distance_score_has_expected_three_regions():
    profile = make_env().profile
    assert distance_score_v11(50.0, profile) == pytest.approx(0.5)
    assert distance_score_v11(1000.0, profile) == pytest.approx(1.0)
    assert distance_score_v11(2000.0, profile) == pytest.approx(0.5)
    assert distance_score_v11(2500.0, profile) == 0.0


def _lock_setup():
    env = make_env()
    for aircraft in env.aircraft:
        aircraft.state.alive = False
    env._by_id("red_1").state = env._by_id("red_1").state.copy()
    env._by_id("red_1").state.x = 0.0
    env._by_id("red_1").state.y = 0.0
    env._by_id("red_1").state.psi = 0.0
    env._by_id("red_1").state.alive = True
    target = env._by_id("blue_0")
    target.state.x = 700.0
    target.state.y = 0.0
    target.state.psi = 0.0
    target.state.alive = True
    env.targets["red_1"] = "blue_0"
    env.lock_progress["red_1"] = 0.0
    return env


def test_shared_only_cue_cannot_increase_lock():
    env = _lock_setup()
    assert lock_quality_v11(env._by_id("red_1").state, env._by_id("blue_0").state, env.profile) > 0.0
    empty = visibility(env)
    env._update_locks(empty)
    assert env.lock_progress["red_1"] == 0.0


def test_direct_visibility_increases_lock():
    env = _lock_setup()
    direct = visibility(env, direct={"blue_0"})
    env._update_locks(direct)
    assert env.lock_progress["red_1"] > 0.0


def test_better_geometry_gives_larger_lock_quality():
    env = _lock_setup()
    good = lock_quality_v11(env._by_id("red_1").state, env._by_id("blue_0").state, env.profile)
    env._by_id("blue_0").state.x = 2000.0
    poor = lock_quality_v11(env._by_id("red_1").state, env._by_id("blue_0").state, env.profile)
    assert good > poor


def test_lock_decays_when_target_is_not_directly_visible():
    env = _lock_setup()
    env.lock_progress["red_1"] = 0.6
    env._update_locks(visibility(env))
    assert env.lock_progress["red_1"] == pytest.approx(0.6 - env.profile["lock_decay_per_step"])


def test_threshold_produces_deterministic_kill_candidate():
    env = _lock_setup()
    env.lock_progress["red_1"] = 0.95
    _, killers = env._update_locks(visibility(env, direct={"blue_0"}))
    assert killers == {"blue_0": "red_1"}
    assert env._by_id("blue_0").state.alive


def test_target_switch_clears_old_lock():
    env = make_env()
    env.targets["red_1"] = "blue_0"
    env.lock_progress["red_1"] = 0.8
    env._switch_target("red_1", "blue_1")
    assert env.lock_progress["red_1"] == 0.0
    assert env.target_hold_steps["red_1"] == 0


def test_target_hold_prevents_switch_within_minimum_hold():
    env = make_env()
    env.targets["red_1"] = "blue_0"
    env.target_hold_steps["red_1"] = 5
    env.lock_progress["red_1"] = 0.1
    direct = {aircraft.aircraft_id: set(BLUE_IDS_V11) for aircraft in env.aircraft}
    effective = env._effective_targets(direct)
    env._refresh_targets(direct, effective)
    assert env.targets["red_1"] == "blue_0"


def test_dead_target_is_released_immediately():
    env = make_env()
    env.targets["red_1"] = "blue_0"
    env._by_id("blue_0").state.alive = False
    direct = env._direct_visible_ids()
    env._refresh_targets(direct, env._effective_targets(direct))
    assert env.targets["red_1"] != "blue_0"
    assert env.lock_progress["red_1"] == 0.0


def test_support_assignment_is_unique_before_target_shortage():
    env = make_env()
    direct = {aircraft.aircraft_id: set() for aircraft in env.aircraft}
    direct["red_0"] = set(BLUE_IDS_V11)
    cues = env._support_cue_ids(direct)
    assigned = [value for value in cues.values() if value is not None]
    assert len(assigned) == 3
    assert len(set(assigned)) == 3


def test_support_death_produces_no_new_cues():
    env = make_env()
    env._by_id("red_0").state.alive = False
    direct = env._direct_visible_ids()
    assert env._support_cue_ids(direct) == {cid: None for cid in RED_COMBAT_IDS_V11}


def test_timeout_uses_combat_survivors_only():
    env = make_env()
    env.step_count = 900
    env._by_id("red_0").state.alive = False
    assert env._terminal_result()[2] == "timeout_draw"
    env._by_id("blue_0").state.alive = False
    env._by_id("blue_1").state.alive = False
    assert env._terminal_result()[2] == "timeout_red_win"


def test_support_is_not_in_timeout_comparison():
    env = make_env()
    env.step_count = 900
    for cid in RED_COMBAT_IDS_V11:
        env._by_id(cid).state.alive = False
    env._by_id("red_0").state.alive = True
    assert env._terminal_result()[2] == "red_total_loss"


def test_step_reward_components_are_complete_and_finite():
    env = make_env()
    actions = {aid: np.zeros(3, dtype=np.float32) for aid in ("red_0", "red_1", "red_2", "red_3")}
    *_, reward, _, _, info = env.step(actions)
    assert set(info["reward_components"]) == set(REWARD_COMPONENT_KEYS_V11)
    assert np.isfinite(reward)
    assert np.isfinite(list(info["reward_components"].values())).all()


def test_static_component_does_not_repeat_geometry_reward():
    env = make_env()
    env._last_formation_score = env._formation_score()
    direct = env._direct_visible_ids()
    pre_targets = dict(env.targets)
    pre_locks = dict(env.lock_progress)
    pre_potentials = {cid: 0.4 for cid in RED_COMBAT_IDS_V11}
    _, components = env._compute_reward(pre_targets, pre_potentials, pre_locks, {}, set(), {}, direct)
    assert components["support_formation_progress_reward"] == pytest.approx(0.0)


def test_v11_team_reward_contains_all_contract_categories():
    env = make_env()
    direct = env._direct_visible_ids()
    pre_targets = dict(env.targets)
    pre_locks = dict(env.lock_progress)
    pre_potentials = {cid: 0.0 for cid in RED_COMBAT_IDS_V11}
    _, components = env._compute_reward(pre_targets, pre_potentials, pre_locks, {}, set(), {}, direct)
    expected = components["mission_outcome_reward"] + components["total_dense_reward"]
    expected += sum(components[key] for key in REWARD_COMPONENT_KEYS_V11 if key.endswith("penalty") or key.endswith("event_reward"))
    expected += components["support_assisted_kill_reward"]
    expected += components["combat_geometry_progress_reward"] + components["combat_lock_progress_reward"] + components["support_formation_progress_reward"]
    assert components["team_total_reward"] == pytest.approx(expected)


def test_support_cue_does_not_directly_kill():
    env = _lock_setup()
    env.lock_progress["red_1"] = 0.99
    env._update_locks(visibility(env))
    assert env.lock_progress["red_1"] < 1.0
    assert env._by_id("blue_0").state.alive


def test_state_dict_restores_lock_target_and_reward_rolling_state():
    env = make_env()
    env.lock_progress["red_1"] = 0.37
    env.targets["red_1"] = "blue_0"
    env._episode_return = -3.2
    state = env.state_dict()
    restored = make_env(13)
    restored.load_state_dict(state)
    assert restored.lock_progress["red_1"] == pytest.approx(0.37)
    assert restored.targets["red_1"] == "blue_0"
    assert restored._episode_return == pytest.approx(-3.2)


def test_v11_best_score_prioritizes_task_win():
    first, _ = compute_best_score_4v3({"task_win_rate": 0.1, "full_elimination_rate": 0.0, "mean_episode_length": 1})
    second, _ = compute_best_score_4v3({"task_win_rate": 0.0, "full_elimination_rate": 1.0, "mean_episode_length": 1})
    assert first > second


def test_v11_summary_contains_required_metrics():
    record = make_env()._episode_summary("draw", "timeout_draw")
    summary = summarize_4v3_episodes([record])
    for key in ("task_win_rate", "full_elimination_rate", "timeout_win_rate", "timeout_loss_rate", "timeout_draw_rate", "mean_red_kills", "lock_episode_rate", "support_cue_rate"):
        assert key in summary


def test_terminal_train_log_has_step_and_reward_groups():
    line = format_train_log_v11(step=8, total=8192, update=1, throughput=10.0, r_step=0.1, episode_return=-1.0, task_win=0.0, full=0.0, mean_kills=0.0, timeout=1.0, mission=-0.1, event=0.0, geom=0.01, lock=0.02, support=0.0)
    assert "r_step=" in line and "ep_return=" in line
    assert all(token in line for token in ("mission=", "event=", "geom=", "lock=", "support="))


def test_terminal_eval_log_has_v11_metrics():
    line = format_eval_log_v11(100, {"episodes": 1, "task_win_rate": 0.0, "full_elimination_rate": 0.0, "any_kill_rate": 0.0, "at_least_two_kill_rate": 0.0, "mean_red_kills": 0.0, "timeout_win_rate": 0.0, "timeout_loss_rate": 0.0, "timeout_draw_rate": 1.0, "mean_return": -1.0, "lock_episode_rate": 0.1, "support_assisted_kill_rate": 0.0})
    assert line.startswith("[eval]") and "lock_episode=" in line and "support_assist=" in line


def test_v9_v10_config_bytes_are_untouched():
    assert hashlib.sha256(V9_CONFIG.read_bytes()).hexdigest()
    assert hashlib.sha256(V10_CONFIG.read_bytes()).hexdigest()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_v11_short_steps_are_finite(seed):
    env = make_env(seed)
    actions = {aid: np.zeros(3, dtype=np.float32) for aid in ("red_0", "red_1", "red_2", "red_3")}
    for _ in range(5):
        result = env.step(actions)
        assert np.isfinite(result[0]).all() and np.isfinite(result[1]).all() and np.isfinite(result[3])
        if result[4]:
            break
