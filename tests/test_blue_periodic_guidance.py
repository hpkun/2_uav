from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithm.happo import HAPPOTrainer
from env import MAVUAVVectorEnv
from env.geometry import compute_pairwise_geometry
from env.mavuav import (
    BLUE_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.models import AircraftState


def env() -> HeterogeneousMAVUAVAirCombatEnv:
    result = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    result.reset(seed=17)
    return result


def red_map(environment):
    return {red_id: environment.entities[red_id] for red_id in RED_IDS}


def isolate_target(environment, target_id):
    for red_id in RED_IDS:
        environment.entities[red_id].state.alive = red_id == target_id


def cached(environment, blue_id="Blue1"):
    return environment.blue_policy.state_dict()["guidance_state"][blue_id]


def test_v35_contract_and_frozen_periodic_configuration():
    config = load_environment_config(None)
    assert ENVIRONMENT_VERSION == config["environment_version"] == "heterogeneous_mavuav_4v4_v3_5"
    assert OBS_DIM == 100 and GLOBAL_STATE_DIM == 117
    assert config["blue_policy"] == {
        "target_strategy": "nearest_red_aircraft",
        "guidance_mode": "periodic_heading",
        "target_refresh_steps": 2,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda policy: policy.pop("target_refresh_steps"),
        lambda policy: policy.update(guidance_mode="direct_pursuit"),
        lambda policy: policy.update(target_strategy="mav_priority"),
        lambda policy: policy.update(target_refresh_steps=1),
        lambda policy: policy.update(target_refresh_steps=0),
        lambda policy: policy.update(target_refresh_steps=2.0),
        lambda policy: policy.update(target_refresh_steps=True),
    ],
)
def test_blue_policy_configuration_is_strict_and_frozen(mutation):
    config = deepcopy(load_environment_config(None))
    mutation(config["blue_policy"])
    with pytest.raises(ValueError, match="blue_policy"):
        load_environment_config(config)


def test_refresh_hold_then_refreshes_cached_angles_not_position():
    e = env(); blue = e.entities["Blue1"]; target = e.entities["MAV"]
    isolate_target(e, "MAV")
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    target.state = AircraftState(3000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.blue_policy.action(blue, red_map(e), 0)
    step0 = cached(e).copy()
    target.state.x, target.state.y, target.state.h = 0.0, 3000.0, 6500.0
    e.blue_policy.action(blue, red_map(e), 1)
    assert cached(e) == step0
    e.blue_policy.action(blue, red_map(e), 2)
    step2 = cached(e)
    assert step2["last_refresh_step"] == 2
    assert not np.isclose(step2["desired_heading"], step0["desired_heading"])
    assert not np.isclose(step2["desired_pitch"], step0["desired_pitch"])


def test_target_switches_only_at_refresh_while_cached_target_is_alive():
    e = env(); blue = e.entities["Blue1"]
    for red_id in ("UAV2", "UAV3"):
        e.entities[red_id].state.alive = False
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.entities["UAV1"].state = AircraftState(1000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.entities["MAV"].state = AircraftState(5000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.blue_policy.action(blue, red_map(e), 0)
    assert cached(e)["target_id"] == "UAV1"
    e.entities["MAV"].state.x = 100.0
    e.blue_policy.action(blue, red_map(e), 1)
    assert cached(e)["target_id"] == "UAV1"
    e.blue_policy.action(blue, red_map(e), 2)
    assert cached(e)["target_id"] == "MAV"


def test_target_death_forces_immediate_unscheduled_refresh():
    e = env(); blue = e.entities["Blue1"]
    for red_id in ("UAV2", "UAV3"):
        e.entities[red_id].state.alive = False
    e.entities["UAV1"].state = AircraftState(3000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.entities["MAV"].state = AircraftState(1000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.blue_policy.action(blue, red_map(e), 0)
    assert cached(e)["target_id"] == "UAV1"
    e.entities["UAV1"].state.alive = False
    e.blue_policy.action(blue, red_map(e), 1)
    assert cached(e)["target_id"] == "MAV" and cached(e)["last_refresh_step"] == 1


def test_reset_clears_guidance_and_diagnostics_are_read_only():
    e = env(); blue = e.entities["Blue1"]
    e.blue_policy.action(blue, red_map(e), 0)
    before = deepcopy(e.blue_policy.state_dict())
    diagnostics = e.blue_policy.diagnostics(blue, red_map(e), 1)
    assert e.blue_policy.state_dict() == before
    assert diagnostics["blue_target_id"] in RED_IDS
    assert diagnostics["blue_guidance_age"] == 1
    assert not diagnostics["blue_guidance_refresh_due"]
    e.reset(seed=18)
    assert all(value["target_id"] is None for value in e.blue_policy.state_dict()["guidance_state"].values())


def test_recovery_preserves_target_and_forces_refresh_on_first_normal_step():
    e = env(); blue = e.entities["Blue1"]
    blue.state.h, blue.state.theta = 1500.0, -0.3
    np.testing.assert_array_equal(e.blue_policy.action(blue, red_map(e), 0), [-1.0, 1.0, 0.0])
    recovery = cached(e)
    assert recovery["target_id"] is not None and recovery["force_refresh"]
    old_heading = recovery["desired_heading"]
    target = e.entities[recovery["target_id"]]
    target.state.y += 4000.0
    blue.state.h, blue.state.theta = 5000.0, 0.0
    e.blue_policy.action(blue, red_map(e), 1)
    normal = cached(e)
    assert normal["last_refresh_step"] == 1 and not normal["force_refresh"]
    assert not np.isclose(normal["desired_heading"], old_heading)


def test_horizontal_recovery_preserves_target_and_forces_first_normal_refresh():
    e = env(); blue = e.entities["Blue1"]
    blue.state = AircraftState(99_000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    action = e.blue_policy.action(blue, red_map(e), 0)
    recovery = cached(e)
    assert np.isfinite(action).all() and recovery["target_id"] is not None and recovery["force_refresh"]
    old_heading = recovery["desired_heading"]
    target = e.entities[recovery["target_id"]]
    target.state.y += 4000.0
    blue.state.x, blue.state.psi = 0.0, 0.0
    e.blue_policy.action(blue, red_map(e), 1)
    normal = cached(e)
    assert normal["last_refresh_step"] == 1 and not normal["force_refresh"]
    assert not np.isclose(normal["desired_heading"], old_heading)


def test_no_alive_red_clears_guidance_and_returns_zero_action():
    e = env(); blue = e.entities["Blue1"]
    e.blue_policy.action(blue, red_map(e), 0)
    for red in red_map(e).values():
        red.state.alive = False
    np.testing.assert_array_equal(e.blue_policy.action(blue, red_map(e), 1), np.zeros(3))
    assert cached(e)["target_id"] is None


def test_blue_policy_state_dict_round_trip_is_exact():
    source = env(); source.blue_policy.action(source.entities["Blue1"], red_map(source), 0)
    payload = source.blue_policy.state_dict()
    target = env(); target.blue_policy.load_state_dict(payload)
    assert target.blue_policy.state_dict() == payload
    payload["guidance_state"]["Blue1"]["desired_heading"] += 1.0
    assert target.blue_policy.state_dict() != payload


def test_vector_state_restore_crosses_refresh_and_hold_steps_exactly():
    actions = np.random.default_rng(51).uniform(-1.0, 1.0, (5, 1, len(RED_IDS), 3))
    with MAVUAVVectorEnv(1, seed=51, profile="main", parallel=False) as source, \
         MAVUAVVectorEnv(1, seed=999, profile="main", parallel=False) as restored:
        source.reset(); source.step(actions[0])  # saved at environment step 1 (odd)
        state = source.get_env_states(); counts = source.reset_counts.copy()
        restored.reset(); restored.set_env_states(state, counts, source.base_seed)
        for index in range(1, 5):  # decision steps 1/2/3/4: hold/refresh/hold/refresh
            expected = source.step(actions[index]); actual = restored.step(actions[index])
            for expected_value, actual_value in zip(expected[:6], actual[:6]):
                np.testing.assert_array_equal(actual_value, expected_value)
            assert restored.get_env_states() == source.get_env_states()


def test_trainer_checkpoint_restores_odd_step_blue_guidance_for_exact_continuation(tmp_path):
    config = {
        "num_envs": 1, "rollout_steps": 1, "ppo_epochs": 1, "minibatch_size": 1,
        "hidden_dim": 16, "seed": 61, "environment_profile": "main",
    }
    source = HAPPOTrainer(config=config); source.train_update()
    checkpoint = tmp_path / "odd_step.pt"; source.save_checkpoint(checkpoint)
    restored = HAPPOTrainer(config=config); assert restored.load_checkpoint(checkpoint) == 1
    assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    actions = np.random.default_rng(61).uniform(-1.0, 1.0, (1, len(RED_IDS), 3))
    for _ in range(4):
        expected = source.vector_env.step(actions); actual = restored.vector_env.step(actions)
        for expected_value, actual_value in zip(expected[:6], actual[:6]):
            np.testing.assert_array_equal(actual_value, expected_value)
        assert restored.vector_env.get_env_states() == source.vector_env.get_env_states()
    source.close(); restored.close()


def test_realtime_pure_pursuit_nose_on_geometry_makes_red_target_aspect_pi():
    red = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    blue = AircraftState(2000.0, 0.0, 5000.0, 275.0, 0.0, np.pi, True)
    geometry = compute_pairwise_geometry(red, blue)
    assert np.isclose(geometry.ata, 0.0, atol=1e-12)
    assert np.isclose(geometry.aa, np.pi, atol=1e-12)
    assert geometry.aa > np.deg2rad(90.0)


def test_held_heading_creates_a_non_degenerate_red_tail_aspect_opportunity():
    e = env(); blue = e.entities["Blue1"]; red = e.entities["MAV"]
    isolate_target(e, "MAV")
    blue.state = AircraftState(0.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    red.state = AircraftState(0.0, 3000.0, 5000.0, 275.0, 0.0, -np.pi / 2, True)
    e.blue_policy.action(blue, red_map(e), 0)  # cache northward heading
    red.state = AircraftState(-2000.0, 0.0, 5000.0, 275.0, 0.0, 0.0, True)
    e.blue_policy.action(blue, red_map(e), 1)  # hold northward guidance
    geometry = compute_pairwise_geometry(red.state, blue.state)
    assert geometry.ata < np.deg2rad(30.0)
    assert geometry.aa < np.deg2rad(90.0)
