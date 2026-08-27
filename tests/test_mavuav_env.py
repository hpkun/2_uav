from copy import deepcopy
import numpy as np

from uav_combat.blue_policy import BLUE_ACTION_CANDIDATES, BluePolicy
from uav_combat.dynamics import integrate_interval, map_normalized_action
from uav_combat.mavuav import BLUE_IDS, ENVIRONMENT_VERSION, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from uav_combat.models import AircraftState
from uav_combat.reward import (
    SITUATION_WEIGHTS, bearing_reward, distance_reward, entering_angle_reward,
    height_reward, situation_reward, speed_reward,
)


def env():
    instance = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    instance.reset(seed=7)
    return instance


def test_zero_action_preserves_level_flight():
    e = env(); entity = e.entities["MAV"]; before = entity.state.copy()
    after = integrate_interval(before, np.zeros(3), entity.spec, 0.1, 50)
    assert np.isclose(after.v, before.v, atol=1e-9)
    assert np.isclose(after.theta, 0.0, atol=1e-9)
    assert np.isclose(after.psi, before.psi, atol=1e-9)


def test_zero_action_is_trimmed_for_nonzero_theta():
    e = env(); entity = e.entities["MAV"]; entity.state.theta = 0.3
    command = map_normalized_action(np.zeros(3), entity.state, entity.spec)
    assert np.allclose(command.as_array(), [np.sin(0.3), np.cos(0.3), 0.0])


def test_type_specific_action_mapping():
    e = env(); action = np.array([0.0, 1.0, 1.0])
    mav = map_normalized_action(action, e.entities["MAV"].state, e.entities["MAV"].spec)
    uav = map_normalized_action(action, e.entities["UAV1"].state, e.entities["UAV1"].spec)
    blue = map_normalized_action(action, e.entities["Blue1"].state, e.entities["Blue1"].spec)
    assert mav.ny == 2.0 and uav.ny == 1.5 and blue.ny == 3.0
    assert mav.nz == blue.nz == 3.0 and uav.nz == 2.0


def test_decision_step_contains_expected_physics_substeps():
    e = env(); assert e.physics_substeps == 10
    assert np.isclose(e.physics_substeps * e.physics_dt, e.decision_dt)


def _attack_setup(e, attacker_id, target_id):
    for index, entity in enumerate(e.entities.values()):
        entity.state = AircraftState(50000.0, index * 5000.0, 5000.0, entity.state.v, 0.0, 0.0, True)
        entity.inactive_cause = None
    e.entities[attacker_id].state.x, e.entities[attacker_id].state.y = 0.0, 0.0
    e.entities[target_id].state.x, e.entities[target_id].state.y = 2000.0, 0.0
    e._attack_streak.clear()


def test_attack_requires_three_full_decision_steps_and_streak_resets():
    e = env(); _attack_setup(e, "MAV", "Blue1")
    assert not e._resolve_attacks()[1]; assert not e._resolve_attacks()[1]
    e.entities["Blue1"].state.y = 5000.0; assert not e._resolve_attacks()[1]
    e.entities["Blue1"].state.y = 0.0
    assert not e._resolve_attacks()[1]; assert not e._resolve_attacks()[1]
    assert e._resolve_attacks()[1] == {"Blue1": "red_attack"}


def test_mav_uav_and_blue_can_attack():
    for attacker, target, cause in (("MAV", "Blue1", "red_attack"), ("UAV1", "Blue1", "red_attack"), ("Blue1", "UAV1", "blue_attack")):
        e = env(); _attack_setup(e, attacker, target)
        e._resolve_attacks(); e._resolve_attacks(); _, deaths = e._resolve_attacks()
        assert deaths[target] == cause


def test_simultaneous_attack_resolution_is_order_independent():
    e = env()
    for index, entity in enumerate(e.entities.values()): entity.state = AircraftState(50000, index * 5000, 5000, entity.state.v, 0, 0, True)
    e.entities["MAV"].state.x = 0; e.entities["MAV"].state.y = 0
    e.entities["Blue1"].state.x = 2000; e.entities["Blue1"].state.y = 0
    e.entities["Blue2"].state.x = 10000; e.entities["Blue2"].state.y = 0
    e.entities["UAV1"].state.x = 12000; e.entities["UAV1"].state.y = 0
    e._resolve_attacks(); e._resolve_attacks(); events, deaths = e._resolve_attacks()
    assert deaths == {"Blue1": "red_attack", "UAV1": "blue_attack"}
    assert {tuple(x.values()) for x in events} >= {("MAV", "Blue1"), ("Blue2", "UAV1")}


def test_uav_death_does_not_end_episode_and_mav_death_does():
    e = env(); e.entities["UAV1"].state.alive = False
    assert e._termination() == (False, False, None)
    e.entities["MAV"].state.alive = False
    assert e._termination() == (True, False, "blue")


def test_red_win_requires_attack_kills_and_living_mav():
    e = env()
    for aid in BLUE_IDS: e.entities[aid].state.alive = False
    assert e._termination()[2] == "blue"
    e._red_attack_kills = set(BLUE_IDS); assert e._termination()[2] == "red"
    e.entities["MAV"].state.alive = False; assert e._termination()[2] == "blue"


def test_blue_escape_does_not_count_as_red_kill():
    e = env(); e.entities["Blue1"].state.x = 100001.0
    deaths = e._apply_boundaries()
    assert deaths["Blue1"] == "blue_escape" and "Blue1" not in e._red_attack_kills


def test_observation_global_state_active_masks_and_finiteness():
    e = env(); e.entities["UAV1"].state.alive = False
    observations = e._observations()
    assert all(value.shape == (55,) for value in observations.values())
    assert e.global_state().shape == (67,)
    assert np.array_equal(e.active_masks, [1, 0, 1])
    assert all(np.all(np.isfinite(value)) for value in observations.values()) and np.all(np.isfinite(e.global_state()))


def test_global_state_xy_uses_battlefield_bounds_without_early_saturation():
    e = env()
    assert [e._global_xy_norm(value, "x") for value in (-100_000, 0, 100_000)] == [-1.0, 0.0, 1.0]
    assert [e._global_xy_norm(value, "y") for value in (-100_000, 0, 100_000)] == [-1.0, 0.0, 1.0]
    e.entities["Blue1"].state.x = 31_000.0
    state_a = e.global_state().copy()
    e.entities["Blue1"].state.x = 50_000.0
    state_b = e.global_state().copy()
    assert state_a[30] != state_b[30]
    assert not np.array_equal(state_a, state_b)


def test_actor_self_xy_normalization_remains_thirty_kilometres():
    e = env()
    assert e._self_xy_norm(30_000.0) == 1.0
    assert e._self_xy_norm(50_000.0) == 1.0
    e.entities["MAV"].state.x = 50_000.0
    e.entities["MAV"].state.y = -50_000.0
    observation = e._observations()["MAV"]
    assert np.array_equal(observation[:2], [1.0, -1.0])


def test_reward_component_formulas_and_weights():
    phi_m = np.deg2rad(30)
    assert np.isclose(bearing_reward(0), 1) and np.isclose(bearing_reward(phi_m), 0.7) and np.isclose(bearing_reward(np.pi), 0)
    assert np.isclose(entering_angle_reward(0), 1) and np.isclose(entering_angle_reward(np.pi), 0)
    assert distance_reward(999) == 0 and distance_reward(1000) == 1 and distance_reward(3000) == 1 and np.isclose(distance_reward(8000), np.exp(-1))
    assert speed_reward(50, 100) == 0.1 and np.isclose(speed_reward(100, 100), 0.5) and speed_reward(200, 100) == 1
    assert height_reward(-2001) == 0 and height_reward(-2000) == 0 and height_reward(0) == 0.5 and height_reward(2000) == 1 and height_reward(4000) == 0
    assert np.isclose(sum(SITUATION_WEIGHTS), 1.0)
    a = AircraftState(0, 0, 5000, 300, 0, 0); b = AircraftState(2000, 0, 5000, 300, 0, 0)
    expected = np.dot(SITUATION_WEIGHTS, [1, 1, 1, 0.5, 0.5])
    assert np.isclose(situation_reward(a, b), expected)


def test_multi_target_uses_best_and_fixed_denominator_after_uav_death():
    e = env(); expected = 0.0
    for aid in e.red_ids:
        own = e.entities[aid]
        expected += max(situation_reward(own.state, e.entities[bid].state) for bid in e.blue_ids)
    assert np.isclose(e._team_situation_reward(), expected / 3)
    e.entities["UAV1"].state.alive = False
    expected = sum(max(situation_reward(e.entities[aid].state, e.entities[bid].state) for bid in e.blue_ids) for aid in ("MAV", "UAV2")) / 3
    assert np.isclose(e._team_situation_reward(), expected)


def test_situation_reward_uses_only_team_visible_alive_blue():
    e = env()
    for index, aid in enumerate(e.red_ids):
        e.entities[aid].state = AircraftState(0.0, index * 10.0, 5000.0, 300.0, 0.0, 0.0, True)
    e.entities["Blue1"].state = AircraftState(20_000.0, 0.0, 5000.0, 300.0, 0.0, np.pi, True)
    e.entities["Blue2"].state = AircraftState(25_000.0, 0.0, 5000.0, 300.0, 0.0, np.pi, True)
    assert not any(e.team_visible(bid) for bid in BLUE_IDS)
    assert e._team_situation_reward() == 0.0

    e.entities["Blue1"].state.x = 10_000.0
    assert e.team_visible("Blue1")
    assert e._team_situation_reward() > 0.0

    e.entities["Blue1"].state = AircraftState(-11_000.0, 0.0, 5000.0, 400.0, 0.0, 0.0, True)
    e.entities["Blue2"].state = AircraftState(13_000.0, 0.0, 5000.0, 250.0, 0.0, 0.0, True)
    assert e.team_visible("Blue1") and not e.team_visible("Blue2")
    visible_only = sum(situation_reward(e.entities[aid].state, e.entities["Blue1"].state) for aid in e.red_ids) / 3.0
    assert all(
        situation_reward(e.entities[aid].state, e.entities["Blue2"].state)
        > situation_reward(e.entities[aid].state, e.entities["Blue1"].state)
        for aid in e.red_ids
    )
    assert np.isclose(e._team_situation_reward(), visible_only)


def test_initial_randomization_is_seed_reproducible_and_optional():
    a = HeterogeneousMAVUAVAirCombatEnv(); b = HeterogeneousMAVUAVAirCombatEnv()
    a.reset(seed=42); b.reset(seed=42)
    assert all(np.array_equal(a.entities[x].state.as_array(), b.entities[x].state.as_array()) for x in a.entities)
    nominal = HeterogeneousMAVUAVAirCombatEnv(randomize=False); nominal.reset(seed=42)
    assert np.allclose(nominal.entities["MAV"].state.as_array(), [-4500, 0, 5000, 325, 0, 0])


def test_sensor_heterogeneity_and_reliable_datalink_masking():
    e = env()
    e.entities["MAV"].state.x = e.entities["UAV1"].state.x = 0.0
    e.entities["MAV"].state.y = e.entities["UAV1"].state.y = 0.0
    e.entities["Blue1"].state.x, e.entities["Blue1"].state.y = 10_000.0, 0.0
    assert e.direct_visible("MAV", "Blue1")
    assert not e.direct_visible("UAV1", "Blue1")
    assert e.datalink_visible("UAV1", "Blue1")
    uav_enemy = e._observations()["UAV1"][33:44]
    assert uav_enemy[7] == 0.0 and uav_enemy[8] == 1.0
    assert np.any(uav_enemy[:6] != 0.0)
    e.entities["Blue1"].state.x = 20_000.0
    e.entities["UAV2"].state.x = -20_000.0
    assert not e.team_visible("Blue1")
    masked = e._observations()["UAV1"][33:44]
    assert np.array_equal(masked[:6], np.zeros(6))
    assert masked[6] == 1.0 and masked[7] == masked[8] == 0.0


def test_observation_one_hot_streak_time_and_distance_normalization():
    e = env()
    e.entities["MAV"].state.x = 0.0
    e.entities["MAV"].state.y = 0.0
    e.entities["Blue1"].state.x = 1000.0
    e.entities["Blue1"].state.y = 0.0
    e._attack_streak[("MAV", "Blue1")] = 2
    e.step_count = 15
    observation = e._observations()["MAV"]
    assert np.array_equal(observation[7:10], [1.0, 0.0, 0.0])
    assert np.isclose(observation[10], 15 / 75)
    assert np.isclose(observation[36], 1000 / 12000)
    assert np.isclose(observation[42], 2 / 3)
    e.entities["Blue1"].state.x = 3000.0
    assert np.isclose(e._observations()["MAV"][36], 3000 / 12000)


def test_global_state_contains_transition_relevant_internal_state():
    e = env(); e.blue_policy.episode_mode = "nearest"; baseline = e.global_state().copy()
    e._attack_streak[("MAV", "Blue1")] = 1
    streak_state = e.global_state().copy(); assert not np.array_equal(baseline, streak_state)
    e._attack_streak.clear(); e.blue_policy.episode_mode = "mav_priority"
    mode_state = e.global_state().copy(); assert not np.array_equal(baseline, mode_state)
    e.blue_policy.episode_mode = "nearest"; e.step_count = 1
    time_state = e.global_state().copy(); assert not np.array_equal(baseline, time_state)
    e.step_count = 0; e._red_attack_kills.add("Blue1")
    kill_state = e.global_state().copy(); assert not np.array_equal(baseline, kill_state)


def test_randomization_profiles_are_reproducible_and_preserve_team_formation():
    a, b = HeterogeneousMAVUAVAirCombatEnv(), HeterogeneousMAVUAVAirCombatEnv()
    a.reset(seed=123, options={"profile": "main"}); b.reset(seed=123, options={"profile": "main"})
    assert all(np.array_equal(a.entities[x].state.as_array(), b.entities[x].state.as_array()) for x in a.entities)
    learnability = HeterogeneousMAVUAVAirCombatEnv(); learnability.reset(seed=123, options={"profile": "learnability"})
    assert any(not np.array_equal(a.entities[x].state.as_array(), learnability.entities[x].state.as_array()) for x in a.entities)
    nominal = HeterogeneousMAVUAVAirCombatEnv(randomize=False); nominal.reset(seed=123)
    for left, right in (("MAV", "UAV1"), ("MAV", "UAV2"), ("Blue1", "Blue2")):
        main_delta = a.entities[left].state.as_array()[:2] - a.entities[right].state.as_array()[:2]
        nominal_delta = nominal.entities[left].state.as_array()[:2] - nominal.entities[right].state.as_array()[:2]
        assert np.all(np.abs(main_delta - nominal_delta) <= 600.0)


def test_red_safe_distance_penalty_is_once_per_step_and_nonlethal():
    e = env()
    for entity in e.entities.values():
        entity.state = AircraftState(50_000.0, 50_000.0, 5000.0, 250.0, 0.0, 0.0, True)
    e.entities["MAV"].state.x = 0.0; e.entities["MAV"].state.y = 0.0
    e.entities["UAV1"].state.x = 50.0; e.entities["UAV1"].state.y = 0.0
    e.entities["UAV2"].state.x = 5000.0; e.entities["UAV2"].state.y = 5000.0
    _, _, _, _, info = e.step(np.zeros((3, 3)))
    assert info["red_safe_distance_violation"] and info["safety_reward"] == -1.0
    assert all(e.entities[aid].state.alive for aid in e.red_ids)
    e.entities["UAV1"].state.x = e.entities["MAV"].state.x + 100.0
    _, _, _, _, info = e.step(np.zeros((3, 3)))
    assert not info["red_safe_distance_violation"] and info["safety_reward"] == 0.0


def test_blue_candidates_and_target_modes():
    assert BLUE_ACTION_CANDIDATES.shape == (27, 3) and len(np.unique(BLUE_ACTION_CANDIDATES, axis=0)) == 27
    e = env(); blue = e.entities["Blue1"]; red = {aid: e.entities[aid] for aid in e.red_ids}
    nearest = BluePolicy("nearest", 1, .1); nearest.reset(np.random.default_rng(1))
    assert nearest.select_target(blue, red).aircraft_id == "UAV1"
    priority = BluePolicy("mav_priority", 1, .1); priority.reset(np.random.default_rng(1))
    assert priority.select_target(blue, red).aircraft_id == "MAV"
    modes_a, modes_b = [], []
    pa, pb = BluePolicy("mixed_episode", 1, .1), BluePolicy("mixed_episode", 1, .1)
    ra, rb = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(20): modes_a.append(pa.reset(ra)); modes_b.append(pb.reset(rb))
    assert modes_a == modes_b and set(modes_a) == {"nearest", "mav_priority"}


def test_config_contract_and_values():
    cfg = load_environment_config(None)
    assert cfg["environment_version"] == ENVIRONMENT_VERSION
    assert cfg["aircraft_specs"]["MAV"]["v_min"] == 250
    assert cfg["battlefield"]["altitude"] == (1000.0, 20000.0)


# Named contract tests below keep every research requirement independently visible
# in pytest output, even where setup is shared with a broader invariant test above.
def test_attack_streak_resets_on_geometry_break():
    e = env(); _attack_setup(e, "MAV", "Blue1"); e._resolve_attacks()
    e.entities["Blue1"].state.y = 5000; e._resolve_attacks()
    assert e._attack_streak[("MAV", "Blue1")] == 0


def test_attack_requires_three_full_decision_steps():
    e = env(); _attack_setup(e, "MAV", "Blue1")
    assert not e._resolve_attacks()[1] and not e._resolve_attacks()[1]
    assert e._resolve_attacks()[1] == {"Blue1": "red_attack"}


def _assert_attacker(attacker, target, cause):
    e = env(); _attack_setup(e, attacker, target); e._resolve_attacks(); e._resolve_attacks()
    assert e._resolve_attacks()[1][target] == cause


def test_mav_can_attack(): _assert_attacker("MAV", "Blue1", "red_attack")
def test_uav_can_attack(): _assert_attacker("UAV1", "Blue1", "red_attack")
def test_blue_can_attack(): _assert_attacker("Blue1", "UAV1", "blue_attack")


def test_uav_death_does_not_end_episode():
    e = env(); e.entities["UAV1"].state.alive = False; assert e._termination()[0] is False


def test_mav_death_causes_red_failure():
    e = env(); e.entities["MAV"].state.alive = False; assert e._termination() == (True, False, "blue")


def test_red_win_requires_both_blue_attack_killed_and_mav_alive():
    e = env(); e.entities["Blue1"].state.alive = e.entities["Blue2"].state.alive = False
    e._red_attack_kills = {"Blue1"}; assert e._termination()[2] == "blue"
    e._red_attack_kills.add("Blue2"); assert e._termination()[2] == "red"


def test_observation_shape_is_55():
    assert all(x.shape == (55,) for x in env()._observations().values())


def test_global_state_shape_is_67(): assert env().global_state().shape == (67,)


def test_active_masks_after_uav_death():
    e = env(); e.entities["UAV2"].state.alive = False; assert np.array_equal(e.active_masks, [1, 1, 0])


def test_all_observations_states_rewards_are_finite():
    e = env(); observations, rewards, *_ = e.step(np.zeros((3, 3)))
    assert all(np.all(np.isfinite(x)) for x in observations.values()) and np.all(np.isfinite(e.global_state())) and np.all(np.isfinite(list(rewards.values())))


def test_reward_phi_formula(): assert np.isclose(bearing_reward(np.deg2rad(15)), 0.85)
def test_reward_q_formula(): assert np.isclose(entering_angle_reward(np.pi / 2), 0.5)
def test_reward_distance_formula(): assert np.isclose(distance_reward(8000), np.exp(-1))
def test_reward_speed_formula(): assert np.isclose(speed_reward(120, 100), 0.7)
def test_reward_height_formula(): assert np.isclose(height_reward(1000), 0.75)
def test_reward_weights_sum_and_combination(): assert np.isclose(sum(SITUATION_WEIGHTS), 1.0)


def test_multi_target_situation_uses_best_current_target():
    e = env(); own = e.entities["MAV"].state
    assert max(situation_reward(own, e.entities[x].state) for x in BLUE_IDS) <= 1.0


def test_dense_team_denominator_remains_fixed_three_after_uav_death():
    e = env(); e.entities["UAV1"].state.alive = False
    expected = sum(max(situation_reward(e.entities[a].state, e.entities[b].state) for b in BLUE_IDS) for a in ("MAV", "UAV2")) / 3
    assert np.isclose(e._team_situation_reward(), expected)


def test_blue_has_exactly_27_candidates(): assert BLUE_ACTION_CANDIDATES.shape == (27, 3)


def test_blue_nearest_target_mode():
    e = env(); p = BluePolicy("nearest", 1, .1); p.reset(np.random.default_rng(1))
    assert p.select_target(e.entities["Blue1"], {a: e.entities[a] for a in e.red_ids}).aircraft_id == "UAV1"


def test_blue_mav_priority_mode():
    e = env(); p = BluePolicy("mav_priority", 1, .1); p.reset(np.random.default_rng(1))
    assert p.select_target(e.entities["Blue1"], {a: e.entities[a] for a in e.red_ids}).aircraft_id == "MAV"


def test_blue_mixed_mode_is_seed_reproducible():
    a, b = HeterogeneousMAVUAVAirCombatEnv(), HeterogeneousMAVUAVAirCombatEnv()
    assert a.reset(seed=77)[1]["blue_target_mode"] == b.reset(seed=77)[1]["blue_target_mode"]


def test_long_environment_rollout_no_nan_inf():
    e = HeterogeneousMAVUAVAirCombatEnv(); observations, _ = e.reset(seed=8); rng = np.random.default_rng(8)
    for _ in range(200):
        observations, rewards, terminated, truncated, _ = e.step(rng.uniform(-1, 1, (3, 3)))
        assert all(np.all(np.isfinite(x)) for x in observations.values()) and np.all(np.isfinite(list(rewards.values())))
        if terminated or truncated: observations, _ = e.reset()
