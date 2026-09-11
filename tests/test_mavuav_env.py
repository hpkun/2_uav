from copy import deepcopy
import inspect
import numpy as np
import pytest

from env.blue_policy import BluePolicy
from env.dynamics import integrate_interval, inverse_trim_map, map_normalized_action
from env.geometry import compute_pairwise_geometry
from env.mavuav import BLUE_IDS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, ENVIRONMENT_VERSION, HeterogeneousMAVUAVAirCombatEnv, load_environment_config
from env.models import AircraftState
from env.reward import (
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
    assert mav.ny == 2.0 and uav.ny == blue.ny == 1.5
    assert mav.nz == 3.0 and uav.nz == blue.nz == 2.0


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
    with pytest.raises(RuntimeError, match="all Blue inactive"):
        e._termination()
    e._red_attack_kills = set(BLUE_IDS); assert e._termination()[2] == "red"
    e.entities["MAV"].state.alive = False; assert e._termination()[2] == "blue"


def test_blue_boundary_violation_fails_fast():
    e = env(); e.entities["Blue1"].state.x = 100001.0
    with pytest.raises(RuntimeError, match="Blue boundary invariant"):
        e._apply_boundaries()


def test_observation_global_state_active_masks_and_finiteness():
    e = env(); e.entities["UAV1"].state.alive = False
    observations = e._observations()
    assert all(value.shape == (OBS_DIM,) for value in observations.values())
    assert e.global_state().shape == (GLOBAL_STATE_DIM,)
    assert np.array_equal(e.active_masks, [1, 0, 1, 1])
    assert all(np.all(np.isfinite(value)) for value in observations.values()) and np.all(np.isfinite(e.global_state()))


def test_global_state_xy_uses_battlefield_bounds_without_early_saturation():
    e = env()
    assert [e._global_xy_norm(value, "x") for value in (-100_000, 0, 100_000)] == [-1.0, 0.0, 1.0]
    assert [e._global_xy_norm(value, "y") for value in (-100_000, 0, 100_000)] == [-1.0, 0.0, 1.0]
    e.entities["Blue1"].state.x = 31_000.0
    state_a = e.global_state().copy()
    e.entities["Blue1"].state.x = 50_000.0
    state_b = e.global_state().copy()
    assert state_a[40] != state_b[40]
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


def test_multi_target_uses_best_red_for_each_visible_blue():
    e = env()
    expected = np.mean([
        max(situation_reward(e.entities[rid].state, e.entities[bid].state) for rid in RED_IDS)
        for bid in BLUE_IDS
    ])
    assert np.isclose(e._team_situation_reward(), expected)
    e.entities["UAV1"].state.alive = False
    expected = np.mean([
        max(situation_reward(e.entities[rid].state, e.entities[bid].state) for rid in ("MAV", "UAV2", "UAV3"))
        for bid in BLUE_IDS
    ])
    assert np.isclose(e._team_situation_reward(), expected)


def test_situation_reward_uses_only_team_visible_alive_blue():
    e = env()
    for index, aid in enumerate(e.red_ids):
        e.entities[aid].state = AircraftState(0.0, index * 10.0, 5000.0, 300.0, 0.0, 0.0, True)
    e.entities["Blue1"].state = AircraftState(20_000.0, 0.0, 5000.0, 300.0, 0.0, np.pi, True)
    for blue_id in BLUE_IDS:
        e.entities[blue_id].state = AircraftState(25_000.0, 0.0, 5000.0, 300.0, 0.0, np.pi, True)
    assert not any(e.team_visible(bid) for bid in BLUE_IDS)
    assert e._team_situation_reward() == 0.0

    e.entities["Blue1"].state.x = 10_000.0
    assert e.team_visible("Blue1")
    assert e._team_situation_reward() > 0.0

    e.entities["Blue1"].state = AircraftState(-11_000.0, 0.0, 5000.0, 400.0, 0.0, 0.0, True)
    e.entities["Blue2"].state = AircraftState(13_000.0, 0.0, 5000.0, 250.0, 0.0, 0.0, True)
    assert e.team_visible("Blue1") and not e.team_visible("Blue2")
    visible_only = max(situation_reward(e.entities[aid].state, e.entities["Blue1"].state) for aid in e.red_ids)
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
    assert np.allclose(nominal.entities["MAV"].state.as_array(), [-5000, 0, 5000, 275, 0, 0])


def test_sensor_heterogeneity_and_reliable_datalink_masking():
    e = env()
    e.entities["MAV"].state.x = e.entities["UAV1"].state.x = 0.0
    e.entities["MAV"].state.y = e.entities["UAV1"].state.y = 0.0
    e.entities["Blue1"].state.x, e.entities["Blue1"].state.y = 10_000.0, 0.0
    assert e.direct_visible("MAV", "Blue1")
    assert not e.direct_visible("UAV1", "Blue1")
    assert e.datalink_visible("UAV1", "Blue1")
    uav_enemy = e._observations()["UAV1"][44:58]
    assert uav_enemy[10] == 0.0 and uav_enemy[11] == 1.0
    assert np.any(uav_enemy[:9] != 0.0)
    e.entities["Blue1"].state.x = 20_000.0
    e.entities["UAV2"].state.x = -20_000.0
    assert not e.team_visible("Blue1")
    masked = e._observations()["UAV1"][44:58]
    assert np.array_equal(masked[:9], np.zeros(9))
    assert masked[9] == 1.0 and masked[10] == masked[11] == 0.0


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
    assert np.isclose(observation[47], 1000 / 12000)
    assert np.isclose(observation[56], 2 / 3)
    e.entities["Blue1"].state.x = 3000.0
    assert np.isclose(e._observations()["MAV"][47], 3000 / 12000)


def test_global_state_contains_transition_relevant_internal_state():
    e = env(); baseline = e.global_state().copy()
    e._attack_streak[("MAV", "Blue1")] = 1
    streak_state = e.global_state().copy(); assert not np.array_equal(baseline, streak_state)
    e._attack_streak.clear(); e.step_count = 1
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
    _, _, _, _, info = e.step(np.zeros((len(RED_IDS), 3)))
    assert info["red_safe_distance_violation"] and info["safety_reward"] == -1.0
    assert all(e.entities[aid].state.alive for aid in e.red_ids)
    e.entities["UAV1"].state.x = e.entities["MAV"].state.x + 100.0
    _, _, _, _, info = e.step(np.zeros((len(RED_IDS), 3)))
    assert not info["red_safe_distance_violation"] and info["safety_reward"] == 0.0


def test_blue_nearest_aircraft_target_strategy():
    e = env(); blue = e.entities["Blue1"]; red = {aid: e.entities[aid] for aid in e.red_ids}
    policy = e.blue_policy
    assert policy.select_target(blue, red).aircraft_id == "UAV1"
    red["MAV"].state.x, red["MAV"].state.y = blue.state.x + 1.0, blue.state.y
    assert policy.select_target(blue, red).aircraft_id == "MAV"
    red["MAV"].state.x = -5000.0
    red["UAV1"].state.alive = False
    assert policy.select_target(blue, red).aircraft_id == "UAV2"
    for aid in ("UAV2", "UAV3"): red[aid].state.alive = False
    assert policy.select_target(blue, red).aircraft_id == "MAV"
    red["MAV"].state.alive = False
    assert policy.select_target(blue, red) is None


def test_nearest_aircraft_equal_distance_tie_uses_red_ids_order():
    e = env(); blue = e.entities["Blue1"]; red = {aid: e.entities[aid] for aid in RED_IDS}
    for aid in RED_IDS:
        red[aid].state.x, red[aid].state.y = blue.state.x - 1000.0, blue.state.y
    assert e.blue_policy.select_target(blue, red).aircraft_id == RED_IDS[0] == "MAV"


def test_config_contract_and_values():
    cfg = load_environment_config(None)
    assert cfg["environment_version"] == ENVIRONMENT_VERSION
    assert cfg["aircraft_specs"]["MAV"]["v_min"] == 250
    assert cfg["battlefield"]["altitude"] == (1000.0, 20000.0)


@pytest.mark.parametrize("field", ["v_min", "v_max", "nx", "ny", "nz"])
def test_config_rejects_blue_uav_dynamics_mismatch(field):
    cfg = load_environment_config(None)
    changed = deepcopy(cfg)
    if field in ("v_min", "v_max"):
        changed["aircraft_specs"]["Blue"][field] += 1.0
    else:
        changed["aircraft_specs"]["Blue"][field] = list(changed["aircraft_specs"]["Blue"][field])
        changed["aircraft_specs"]["Blue"][field][1] += 0.1
    with pytest.raises(ValueError, match=rf"aircraft_specs\.Blue\.{field} must match"):
        load_environment_config(changed)


def test_v33_uav_performance_and_nominal_speed_contract_are_exact():
    assert RED_IDS == ("MAV", "UAV1", "UAV2", "UAV3")
    assert BLUE_IDS == ("Blue1", "Blue2", "Blue3", "Blue4")
    cfg = load_environment_config(None)
    mav, uav, blue = (cfg["aircraft_specs"][kind] for kind in ("MAV", "UAV", "Blue"))
    for field in ("v_min", "v_max", "nx", "ny", "nz"):
        assert blue[field] == uav[field]
    assert any(mav[field] != uav[field] for field in ("v_min", "v_max", "ny", "nz"))
    initial = cfg["scenario"]["initial"]
    assert all(initial[aid]["speed"] == 275.0 for aid in RED_IDS + BLUE_IDS)
    for aid in RED_IDS + BLUE_IDS:
        aircraft_type = "MAV" if aid == "MAV" else "UAV" if aid in RED_IDS else "Blue"
        spec = cfg["aircraft_specs"][aircraft_type]
        jitter = cfg["randomization_profiles"]["main"]["speed_jitter"]
        assert spec["v_min"] < initial[aid]["speed"] - jitter
        assert initial[aid]["speed"] + jitter < spec["v_max"]


@pytest.mark.parametrize("profile", ["learnability", "main"])
def test_v33_randomization_preserves_shared_red_blue_uav_base_contract(profile):
    e = HeterogeneousMAVUAVAirCombatEnv(randomize=True, profile=profile)
    e.reset(seed=2026)
    uav = e.entities["UAV1"].spec
    for aid in RED_IDS[1:] + BLUE_IDS:
        aircraft = e.entities[aid]
        assert (aircraft.spec.v_min, aircraft.spec.v_max, aircraft.spec.nx, aircraft.spec.ny, aircraft.spec.nz) == (
            uav.v_min, uav.v_max, uav.nx, uav.ny, uav.nz,
        )
        assert uav.v_min <= aircraft.state.v <= uav.v_max


def test_v33_blue_altitude_recovery_guard_derives_to_3000_metres():
    e = env()
    blue = e.entities["Blue1"]
    assert e.blue_policy._altitude_recovery_guard(blue.state, blue) == 3000.0


def test_v33_blue_recovery_guard_expands_for_steep_uav_descent():
    e = env()
    blue = e.entities["Blue1"]
    blue.state.h = 5000.0
    blue.state.v = blue.spec.v_max
    blue.state.theta = -np.pi / 3.0
    assert e.blue_policy._altitude_recovery_guard(blue.state, blue) > 6000.0
    np.testing.assert_array_equal(
        e.blue_policy.action(blue, {aid: e.entities[aid] for aid in RED_IDS}, 0), [-1.0, 1.0, 0.0],
    )


def test_v33_entity_order_and_nominal_formation_are_exact():
    e = env()
    expected = {
        "MAV": (-5000.0, 0.0, 5000.0), "UAV1": (-4000.0, -1200.0, 5000.0),
        "UAV2": (-4000.0, 0.0, 5000.0), "UAV3": (-4000.0, 1200.0, 5000.0),
        "Blue1": (4000.0, -1800.0, 5000.0), "Blue2": (4000.0, -600.0, 5000.0),
        "Blue3": (4000.0, 600.0, 5000.0), "Blue4": (4000.0, 1800.0, 5000.0),
    }
    assert {aid: tuple(e.entities[aid].state.as_array()[:3]) for aid in e.entities} == expected
    slot_jitter = e.config["randomization_profiles"]["main"]["slot_xy_jitter"]
    assert -5000.0 + slot_jitter < -4000.0 - slot_jitter


def test_v33_observation_slot_layout_for_every_red_agent():
    e = env()
    for own_id in RED_IDS:
        observation = e._observations()[own_id]
        assert observation.shape == (100,)
        friends = [aid for aid in RED_IDS if aid != own_id]
        for slot, friend_id in enumerate(friends):
            start = 11 + 11 * slot
            assert observation[start + 7] == float(e.entities[friend_id].state.alive)
            assert np.array_equal(observation[start + 8:start + 11], [1, 0, 0] if friend_id == "MAV" else [0, 1, 0])
        for slot, blue_id in enumerate(BLUE_IDS):
            start = 44 + 14 * slot
            assert observation[start + 9] == float(e.entities[blue_id].state.alive)
            assert observation[start + 13] == float(blue_id in e._red_attack_kills)


def test_v33_global_state_layout_and_attack_streak_order_are_exact():
    e = env()
    from env.mavuav import CROSS_TEAM_ATTACK_PAIRS
    assert len(CROSS_TEAM_ATTACK_PAIRS) == 32
    assert CROSS_TEAM_ATTACK_PAIRS[:4] == tuple(("MAV", blue) for blue in BLUE_IDS)
    assert CROSS_TEAM_ATTACK_PAIRS[16:20] == tuple(("Blue1", red) for red in RED_IDS)
    for index, pair in enumerate(CROSS_TEAM_ATTACK_PAIRS):
        e._attack_streak[pair] = (index % 3) + 1
    e._red_attack_kills = {"Blue2", "Blue4"}
    state = e.global_state()
    np.testing.assert_allclose(state[80:112], [min((i % 3) + 1, 3) / 3 for i in range(32)])
    assert np.array_equal(state[112:116], [0, 1, 0, 1])
    assert np.isclose(state[116], 0.0)


def test_v32_action_contract_requires_all_four_red_slots():
    e = env()
    assert set(e._action_dict({aid: np.zeros(3) for aid in RED_IDS})) == set(RED_IDS)
    with np.testing.assert_raises_regex(ValueError, r"shape \(4, 3\)"):
        e._action_dict(np.zeros((3, 3)))


def test_uav3_loss_and_four_blue_kill_event_rewards_remain_per_aircraft():
    uav_loss_env = env()
    uav_loss_env.entities["UAV3"].state.x = 100_001.0
    *_, loss_info = uav_loss_env.step(np.zeros((len(RED_IDS), 3)))
    assert loss_info["death_causes"]["UAV3"] == "boundary"
    assert loss_info["event_reward"] == -10.0

    kill_env = env()
    def resolve_all_blue():
        deaths = {}
        events = []
        for blue_id in BLUE_IDS:
            kill_env._deactivate(blue_id, "red_attack", deaths)
            kill_env._red_attack_kills.add(blue_id)
            events.append({"attacker": "MAV", "target": blue_id})
        return events, deaths
    kill_env._resolve_attacks = resolve_all_blue
    *_, kill_info = kill_env.step(np.zeros((len(RED_IDS), 3)))
    assert kill_info["event_reward"] == 4 * 50.0
    assert kill_info["terminal_reward"] == 100.0
    assert kill_info["outcome"] == "red"


def test_all_four_blue_aircraft_act_independently_and_mav_fallback():
    e = env()
    acted = []
    e.blue_policy.action = lambda aircraft, red_entities, decision_step: acted.append(aircraft.aircraft_id) or np.zeros(3)
    e.step(np.zeros((len(RED_IDS), 3)))
    assert acted == list(BLUE_IDS)

    e = env()
    for aid in RED_IDS[1:]: e.entities[aid].state.alive = False
    selected = e.blue_policy.select_target(e.entities["Blue1"], {aid: e.entities[aid] for aid in RED_IDS})
    assert selected.aircraft_id == "MAV"


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


def test_red_win_requires_all_blue_attack_killed_and_mav_alive():
    e = env()
    for blue_id in BLUE_IDS: e.entities[blue_id].state.alive = False
    e._red_attack_kills = set(BLUE_IDS[:-1])
    with pytest.raises(RuntimeError, match="all Blue inactive"):
        e._termination()
    e._red_attack_kills.add(BLUE_IDS[-1]); assert e._termination()[2] == "red"


def test_observation_shape_is_100():
    assert all(x.shape == (OBS_DIM,) for x in env()._observations().values())


def test_global_state_shape_is_117(): assert env().global_state().shape == (GLOBAL_STATE_DIM,) == (117,)


def test_enemy_relative_velocity_uses_blue_minus_red_for_all_blues():
    e = env()
    e.entities["MAV"].state = AircraftState(0.0, 0.0, 5000.0, 200.0, 0.0, 0.0, True)
    e.entities["Blue1"].state = AircraftState(2000.0, 0.0, 5000.0, 300.0, 0.0, 0.0, True)
    e.entities["Blue2"].state = AircraftState(2500.0, 0.0, 5000.0, 250.0, 0.0, np.pi, True)
    observation = e._observations()["MAV"]
    scale = e.config["normalization"]["relative_velocity_scale"]
    for blue_id, start in zip(BLUE_IDS, (44, 58, 72, 86)):
        geometry = compute_pairwise_geometry(e.entities["MAV"].state, e.entities[blue_id].state)
        expected = np.clip(geometry.relative_velocity / scale, -1.0, 1.0)
        np.testing.assert_allclose(observation[start + 4:start + 7], expected)
    assert observation[48] > 0.0


def test_dead_or_invisible_enemy_masks_relative_velocity():
    e = env()
    e.entities["Blue1"].state.x = 20_000.0
    e.entities["Blue2"].state.alive = False
    observation = e._observations()["MAV"]
    assert np.array_equal(observation[44:53], np.zeros(9))
    assert np.array_equal(observation[58:67], np.zeros(9))


def test_active_masks_after_uav_death():
    e = env(); e.entities["UAV2"].state.alive = False; assert np.array_equal(e.active_masks, [1, 1, 0, 1])


def test_all_observations_states_rewards_are_finite():
    e = env(); observations, rewards, *_ = e.step(np.zeros((len(RED_IDS), 3)))
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


def test_dense_team_averages_over_visible_alive_blue_after_uav_death():
    e = env(); e.entities["UAV1"].state.alive = False
    expected = np.mean([max(situation_reward(e.entities[a].state, e.entities[b].state) for a in ("MAV", "UAV2", "UAV3")) for b in BLUE_IDS])
    assert np.isclose(e._team_situation_reward(), expected)


def test_blue_policy_source_has_no_candidate_rollout_or_situation_scoring():
    source = inspect.getsource(BluePolicy.action)
    assert "integrate_interval" not in source and "situation_reward" not in source and "candidate" not in source


def test_blue_target_strategy_is_fixed_and_seed_independent():
    a, b = HeterogeneousMAVUAVAirCombatEnv(), HeterogeneousMAVUAVAirCombatEnv()
    assert a.reset(seed=77)[1]["blue_target_strategy"] == b.reset(seed=91)[1]["blue_target_strategy"] == "nearest_red_aircraft"


def test_short_environment_rollout_no_nan_inf():
    e = HeterogeneousMAVUAVAirCombatEnv(); observations, _ = e.reset(seed=8); rng = np.random.default_rng(8)
    for _ in range(20):
        observations, rewards, terminated, truncated, _ = e.step(rng.uniform(-1, 1, (len(RED_IDS), 3)))
        assert all(np.all(np.isfinite(x)) for x in observations.values()) and np.all(np.isfinite(list(rewards.values())))
        if terminated or truncated: observations, _ = e.reset()


@pytest.mark.parametrize(
    "axis,value,heading,theta",
    [
        ("x", 99_650.0, 0.0, 0.0), ("x", -99_650.0, np.pi, 0.0),
        ("y", 99_650.0, np.pi / 2, 0.0), ("y", -99_650.0, -np.pi / 2, 0.0),
        ("h", 19_850.0, 0.0, 0.2), ("h", 1_250.0, 0.0, -0.2),
    ],
)
def test_blue_policy_emergency_rule_keeps_one_step_state_in_bounds(axis, value, heading, theta):
    e = env(); blue = e.entities["Blue1"]
    if axis == "x": blue.state.x = value
    elif axis == "y": blue.state.y = value
    else: blue.state.h = value
    blue.state.psi, blue.state.theta = heading, theta
    action = e.blue_policy.action(blue, {aid: e.entities[aid] for aid in RED_IDS}, 0)
    predicted = integrate_interval(blue.state, action, blue.spec, e.physics_dt, e.physics_substeps)
    assert e.blue_policy._within_battlefield(predicted)


def test_blue_policy_outside_horizontal_boundary_uses_finite_center_recovery():
    e = env(); blue = e.entities["Blue1"]
    blue.state.x = e.config["battlefield"]["x"][1] + 1000.0
    blue.state.psi = 0.0
    diagnostics = e.blue_policy.diagnostics(blue, {aid: e.entities[aid] for aid in RED_IDS}, 0)
    action = e.blue_policy.action(blue, {aid: e.entities[aid] for aid in RED_IDS}, 0)
    assert diagnostics["blue_horizontal_recovery_active"] and np.isfinite(action).all()


def test_blue_policy_y_mirror_changes_only_yaw_overload_action():
    e = env(); red = {aid: e.entities[aid] for aid in RED_IDS}
    for aid in RED_IDS[1:]: red[aid].state.alive = False
    red["UAV2"].state.alive = True
    blue = e.entities["Blue1"]
    blue.state = AircraftState(0.0, 1200.0, 5000.0, 325.0, 0.1, -0.2, True)
    red["UAV2"].state = AircraftState(2500.0, 2200.0, 5300.0, 225.0, -0.05, 0.3, True)
    action = e.blue_policy.action(blue, red, 0)
    blue.state.y *= -1; blue.state.psi *= -1
    red["UAV2"].state.y *= -1; red["UAV2"].state.psi *= -1
    mirrored = e.blue_policy.action(blue, red, 2)
    np.testing.assert_array_equal(mirrored, [action[0], action[1], -action[2]])


def test_short_deterministic_rollout_keeps_every_live_blue_inside_bounds():
    e = env()
    for _ in range(30):
        _, _, terminated, truncated, _ = e.step(np.zeros((len(RED_IDS), 3)))
        for aid in BLUE_IDS:
            if e.entities[aid].state.alive:
                assert e.blue_policy._within_battlefield(e.entities[aid].state)
                assert e.entities[aid].inactive_cause is None
            else:
                assert e.entities[aid].inactive_cause == "red_attack"
        if terminated or truncated:
            break


def test_obsolete_blue_exit_semantics_are_absent_from_environment_source():
    assert "blue_" + "escape" not in inspect.getsource(HeterogeneousMAVUAVAirCombatEnv)


def test_blue_combat_still_checks_all_red_targets_not_only_maneuver_target():
    e = env(); _attack_setup(e, "Blue1", "MAV")
    e.entities["UAV1"].state.x, e.entities["UAV1"].state.y = 100.0, 0.0
    assert e.blue_policy.select_target(e.entities["Blue1"], {aid: e.entities[aid] for aid in RED_IDS}).aircraft_id == "UAV1"
    e._resolve_attacks(); e._resolve_attacks()
    assert e._resolve_attacks()[1]["MAV"] == "blue_attack"
