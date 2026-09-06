from copy import deepcopy
import os
import numpy as np
import pytest

from env import MAVUAVVectorEnv
from env.mavuav import GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, load_environment_config


def short_config(max_steps=1):
    config = deepcopy(load_environment_config(None)); config["simulation"]["max_decision_steps"] = max_steps
    return config


def test_vector_env_shapes_and_auto_reset():
    with MAVUAVVectorEnv(2, short_config(), seed=3, randomize=False) as env:
        observations, states, masks, infos = env.reset()
        assert observations.shape == (2, len(RED_IDS), OBS_DIM) and states.shape == (2, GLOBAL_STATE_DIM) and masks.shape == (2, len(RED_IDS)) and len(infos) == 2
        observations, states, rewards, terminated, truncated, masks, infos = env.step(np.zeros((2, len(RED_IDS), 3)))
        assert rewards.shape == (2, len(RED_IDS)) and terminated.shape == truncated.shape == (2,)
        assert truncated.all() and all(info["auto_reset"] and "episode_summary" in info for info in infos)
        env.step(np.zeros((2, len(RED_IDS), 3)))


def test_vector_env_shapes():
    with MAVUAVVectorEnv(2, seed=3) as env:
        observations, states, masks, infos = env.reset()
        assert observations.shape == (2, len(RED_IDS), OBS_DIM) and states.shape == (2, GLOBAL_STATE_DIM) and masks.shape == (2, len(RED_IDS)) and len(infos) == 2


def test_vector_env_auto_reset():
    with MAVUAVVectorEnv(1, short_config(), seed=3) as env:
        env.reset()
        *_, truncated, masks, infos = env.step(np.zeros((1, len(RED_IDS), 3)))
        assert truncated[0] and infos[0]["auto_reset"] and np.array_equal(masks[0], np.ones(len(RED_IDS)))


def test_vector_env_can_run_1000_steps_without_manual_reset():
    with MAVUAVVectorEnv(1, short_config(5), seed=4) as env:
        observations, states, masks, _ = env.reset(); rng = np.random.default_rng(4)
        for _ in range(1000):
            observations, states, rewards, _, _, masks, _ = env.step(rng.uniform(-1, 1, (1, len(RED_IDS), 3)))
            assert np.all(np.isfinite(observations)) and np.all(np.isfinite(states)) and np.all(np.isfinite(rewards))


def test_vector_env_uses_distinct_worker_processes_by_default():
    with MAVUAVVectorEnv(3, short_config(), seed=8) as env:
        pids = env.worker_pids
        assert env.parallel
        assert len(set(pids)) == 3
        assert os.getpid() not in pids


def test_parallel_results_match_serial_reference_through_auto_reset():
    config = short_config(3)
    rng = np.random.default_rng(19)
    with MAVUAVVectorEnv(2, config, seed=19, randomize=True, parallel=True) as parallel_env, \
         MAVUAVVectorEnv(2, config, seed=19, randomize=True, parallel=False) as serial_env:
        parallel_reset = parallel_env.reset()
        serial_reset = serial_env.reset()
        for parallel_value, serial_value in zip(parallel_reset[:3], serial_reset[:3]):
            np.testing.assert_array_equal(parallel_value, serial_value)
        for _ in range(7):
            actions = rng.uniform(-1.0, 1.0, (2, len(RED_IDS), 3))
            parallel_step = parallel_env.step(actions)
            serial_step = serial_env.step(actions)
            for parallel_value, serial_value in zip(parallel_step[:6], serial_step[:6]):
                np.testing.assert_array_equal(parallel_value, serial_value)
            assert [info["auto_reset"] for info in parallel_step[6]] == [info["auto_reset"] for info in serial_step[6]]
            np.testing.assert_array_equal(parallel_env.reset_counts, serial_env.reset_counts)


def test_parallel_and_serial_match_with_same_profile_through_auto_reset():
    config = short_config(2)
    rng = np.random.default_rng(31)
    with MAVUAVVectorEnv(2, config, seed=31, profile="learnability", parallel=True) as parallel_env, \
         MAVUAVVectorEnv(2, config, seed=31, profile="learnability", parallel=False) as serial_env:
        parallel_reset, serial_reset = parallel_env.reset(), serial_env.reset()
        for parallel_value, serial_value in zip(parallel_reset[:3], serial_reset[:3]):
            np.testing.assert_array_equal(parallel_value, serial_value)
        for _ in range(6):
            actions = rng.uniform(-1.0, 1.0, (2, len(RED_IDS), 3))
            parallel_step, serial_step = parallel_env.step(actions), serial_env.step(actions)
            for parallel_value, serial_value in zip(parallel_step[:6], serial_step[:6]):
                np.testing.assert_array_equal(parallel_value, serial_value)


def test_worker_error_is_propagated_without_desynchronizing_other_workers():
    with MAVUAVVectorEnv(2, short_config(), seed=23) as env:
        original_pids = env.worker_pids
        with pytest.raises(RuntimeError, match="unknown vector-environment command"):
            env._send_all("invalid-test-command", [None, None])
        assert env.worker_pids == original_pids


def test_vector_environment_state_round_trip_restores_next_transition():
    with MAVUAVVectorEnv(2, short_config(5), seed=41, profile="main") as env:
        env.reset()
        actions = np.random.default_rng(41).uniform(-1.0, 1.0, (2, len(RED_IDS), 3))
        env.step(actions)
        states = env.get_env_states(); counts = env.reset_counts.copy(); base_seed = env.base_seed
        expected = env.step(actions)
        env.set_env_states(states, counts, base_seed)
        restored = env.step(actions)
        for expected_value, restored_value in zip(expected[:6], restored[:6]):
            np.testing.assert_array_equal(expected_value, restored_value)
