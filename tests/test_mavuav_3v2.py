import numpy as np

from uav_combat import HeterogeneousMAVUAVAirCombatEnv, MAVUAVVectorEnv


def test_reset_step_shapes_and_agents():
    env = HeterogeneousMAVUAVAirCombatEnv()
    obs, info = env.reset()
    assert list(obs) == ["MAV", "UAV1", "UAV2"]
    assert info["agents"] == ["MAV", "UAV1", "UAV2"]
    assert {value.shape for value in obs.values()} == {(40,)}
    next_obs, rewards, terminated, truncated, _ = env.step(np.zeros((3, 3)))
    assert {value.shape for value in next_obs.values()} == {(40,)}
    assert set(rewards) == set(env.red_ids)
    assert len(set(rewards.values())) == 1
    assert not terminated and not truncated


def test_action_maps_to_type_specific_overloads():
    env = HeterogeneousMAVUAVAirCombatEnv(dt=0.2)
    env.reset()
    initial = {aid: env.entities[aid].state.copy() for aid in env.red_ids}
    env.step({aid: np.array([1.0, 1.0, 1.0]) for aid in env.red_ids})
    assert env.entities["MAV"].state.v > initial["MAV"].v
    assert env.entities["MAV"].state.theta > env.entities["UAV1"].state.theta


def test_action_must_be_three_dimensional():
    env = HeterogeneousMAVUAVAirCombatEnv()
    env.reset()
    with np.testing.assert_raises(ValueError):
        env.step(np.zeros((3, 2)))


def test_attack_holds_three_steps_and_red_win_requires_mav_alive():
    env = HeterogeneousMAVUAVAirCombatEnv(dt=0.0)
    env.reset()
    for blue_id, y in (("Blue1", 0), ("Blue2", 500)):
        blue = env.entities[blue_id].state
        blue.x, blue.y, blue.h, blue.psi = 500, y, 3000, 0.0
    zeros = {aid: np.zeros(3) for aid in env.red_ids}
    env.step(zeros); env.step(zeros)
    assert all(env.entities[x].state.alive for x in env.blue_ids)
    _, _, terminated, _, info = env.step(zeros)
    assert terminated and info["outcome"] == "red"


def test_mav_death_terminates_but_uav_death_does_not():
    env = HeterogeneousMAVUAVAirCombatEnv(dt=0.0)
    env.reset(); env.entities["UAV1"].state.alive = False
    assert env.step(np.zeros((3,3)))[2] is False
    env.entities["MAV"].state.alive = False
    assert env.step(np.zeros((3,3)))[2] is True


def test_vector_training_contract():
    env = MAVUAVVectorEnv(2)
    obs, infos = env.reset(seed=7)
    assert obs.shape == (2, 3, 40) and len(infos) == 2
    obs, rewards, terminated, truncated, infos = env.step(np.zeros((2, 3, 3)))
    assert obs.shape == (2, 3, 40)
    assert rewards.shape == (2, 3)
    assert terminated.shape == truncated.shape == (2,)
