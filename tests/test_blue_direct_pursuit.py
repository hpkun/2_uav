from __future__ import annotations

import inspect

import numpy as np

from env.blue_policy import BluePolicy
from env.dynamics import integrate_interval, inverse_trim_map, map_normalized_action, wrap_angle
from env.geometry import compute_pairwise_geometry
from env.mavuav import RED_IDS, HeterogeneousMAVUAVAirCombatEnv
from env.models import AircraftState


def make_env() -> HeterogeneousMAVUAVAirCombatEnv:
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=11)
    return env


def red_entities(env):
    return {aircraft_id: env.entities[aircraft_id] for aircraft_id in RED_IDS}


def test_inverse_trim_map_round_trip_for_random_actions():
    env = make_env(); aircraft = env.entities["Blue1"]
    rng = np.random.default_rng(2026)
    for _ in range(1000):
        aircraft.state.theta = float(rng.uniform(-1.0, 1.0))
        action = rng.uniform(-1.0, 1.0, 3)
        mapped = map_normalized_action(action, aircraft.state, aircraft.spec)
        np.testing.assert_allclose(inverse_trim_map(mapped, aircraft.state, aircraft.spec), action, atol=1e-12)


def test_direct_pursuit_actions_are_finite_bounded_and_have_no_lookahead_calls():
    env = make_env(); policy = env.blue_policy
    source = inspect.getsource(BluePolicy)
    assert "integrate_interval" not in source and "situation_reward" not in source
    for blue_id in env.blue_ids:
        action = policy.action(env.entities[blue_id], red_entities(env))
        assert action.shape == (3,) and np.isfinite(action).all()
        assert np.all(np.abs(action) <= 1.0)


def _heading_error_after(target_y: float) -> tuple[float, float, np.ndarray]:
    env = make_env(); blue = env.entities["Blue1"]
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    target = env.entities["MAV"]
    target.state = AircraftState(3000.0, target_y, 5000.0, 275.0, 0.0, 0.0, True)
    for aid in RED_IDS[1:]: env.entities[aid].state.alive = False
    desired = float(np.arctan2(target_y, 3000.0))
    before = abs(wrap_angle(desired - blue.state.psi))
    action = env.blue_policy.action(blue, red_entities(env))
    after_state = integrate_interval(blue.state, action, blue.spec, env.physics_dt, env.physics_substeps)
    return before, abs(wrap_angle(desired - after_state.psi)), action


def test_target_on_right_reduces_negative_heading_error():
    before, after, action = _heading_error_after(-1500.0)
    assert action[2] < 0.0 and after < before


def test_target_on_left_reduces_positive_heading_error():
    before, after, action = _heading_error_after(1500.0)
    assert action[2] > 0.0 and after < before


def _pitch_error_after(target_h: float) -> tuple[float, float, np.ndarray]:
    env = make_env(); blue = env.entities["Blue1"]
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    target = env.entities["MAV"]
    target.state = AircraftState(3000.0, 0.0, target_h, 275.0, 0.0, 0.0, True)
    for aid in RED_IDS[1:]: env.entities[aid].state.alive = False
    desired = float(np.arctan2(target_h - 5000.0, 3000.0))
    action = env.blue_policy.action(blue, red_entities(env))
    after_state = integrate_interval(blue.state, action, blue.spec, env.physics_dt, env.physics_substeps)
    return abs(desired), abs(desired - after_state.theta), action


def test_target_above_reduces_pitch_error():
    before, after, action = _pitch_error_after(6500.0)
    assert action[1] > 0.0 and after < before


def test_target_below_reduces_pitch_error():
    before, after, action = _pitch_error_after(3500.0)
    assert action[1] < 0.0 and after < before


def test_zero_angular_error_has_zero_trim_action_and_holds_speed():
    env = make_env(); blue = env.entities["Blue1"]
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    env.entities["MAV"].state = AircraftState(3000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    for aid in RED_IDS[1:]: env.entities[aid].state.alive = False
    action = env.blue_policy.action(blue, red_entities(env))
    after = integrate_interval(blue.state, action, blue.spec, env.physics_dt, env.physics_substeps)
    np.testing.assert_allclose(action, np.zeros(3), atol=1e-12)
    assert np.isclose(after.v, blue.state.v, atol=1e-9)


def test_altitude_and_horizontal_emergency_rules_are_explicit():
    env = make_env(); blue = env.entities["Blue1"]
    blue.state.h, blue.state.theta = 1500.0, -0.3
    np.testing.assert_array_equal(env.blue_policy.action(blue, red_entities(env)), [-1.0, 1.0, 0.0])
    blue.state.h, blue.state.theta = 19500.0, 0.3
    np.testing.assert_array_equal(env.blue_policy.action(blue, red_entities(env)), [-1.0, -1.0, 0.0])
    blue.state = AircraftState(99_000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    diagnostics = env.blue_policy.diagnostics(blue, red_entities(env))
    assert diagnostics["blue_horizontal_recovery_active"]
    assert env.blue_policy.action(blue, red_entities(env))[2] < 0.0


def test_random_legal_state_stress_has_no_boundary_controller_invariant():
    env = make_env(); rng = np.random.default_rng(91)
    blue = env.entities["Blue1"]
    for _ in range(5000):
        blue.state = AircraftState(
            float(rng.uniform(-100_000.0, 100_000.0)),
            float(rng.uniform(-100_000.0, 100_000.0)),
            float(rng.uniform(1000.0, 20_000.0)),
            float(rng.uniform(blue.spec.v_min, blue.spec.v_max)),
            float(rng.uniform(-np.pi / 3, np.pi / 3)),
            float(rng.uniform(-np.pi, np.pi)), True,
        )
        action = env.blue_policy.action(blue, red_entities(env))
        assert np.isfinite(action).all() and np.all(np.abs(action) <= 1.0)


def test_direct_pursuit_can_enter_formal_attack_geometry():
    env = make_env(); blue = env.entities["Blue1"]; target = env.entities["MAV"]
    blue.state = AircraftState(0.0, 0.0, 5000.0, 300.0, 0.0, 0.0, True)
    target.state = AircraftState(4500.0, 0.0, 5000.0, 250.0, 0.0, 0.0, True)
    for aid in RED_IDS[1:]: env.entities[aid].state.alive = False
    reached = False
    for _ in range(45):
        action = env.blue_policy.action(blue, red_entities(env))
        blue.state = integrate_interval(blue.state, action, blue.spec, env.physics_dt, env.physics_substeps)
        target.state = integrate_interval(target.state, np.zeros(3), target.spec, env.physics_dt, env.physics_substeps)
        geometry = compute_pairwise_geometry(blue.state, target.state)
        reached |= 1000.0 <= geometry.distance <= 3000.0 and geometry.ata < np.deg2rad(30) and geometry.aa < np.deg2rad(90)
    assert reached


def test_short_environment_smoke_is_finite():
    env = HeterogeneousMAVUAVAirCombatEnv(profile="main")
    observations, _ = env.reset(seed=93)
    rng = np.random.default_rng(93)
    for _ in range(75):
        observations, rewards, terminated, truncated, _ = env.step(rng.uniform(-1.0, 1.0, (len(RED_IDS), 3)))
        assert all(np.isfinite(value).all() for value in observations.values())
        assert np.isfinite(list(rewards.values())).all()
        if terminated or truncated:
            break
