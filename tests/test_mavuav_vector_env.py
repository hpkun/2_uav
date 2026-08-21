from copy import deepcopy
import numpy as np

from uav_combat import MAVUAVVectorEnv
from uav_combat.mavuav import load_environment_config


def short_config(max_steps=1):
    config = deepcopy(load_environment_config(None)); config["simulation"]["max_decision_steps"] = max_steps
    return config


def test_vector_env_shapes_and_auto_reset():
    env = MAVUAVVectorEnv(2, short_config(), seed=3, randomize=False)
    observations, states, masks, infos = env.reset()
    assert observations.shape == (2, 3, 40) and states.shape == (2, 40) and masks.shape == (2, 3) and len(infos) == 2
    observations, states, rewards, terminated, truncated, masks, infos = env.step(np.zeros((2, 3, 3)))
    assert rewards.shape == (2, 3) and terminated.shape == truncated.shape == (2,)
    assert truncated.all() and all(info["auto_reset"] and "episode_summary" in info for info in infos)
    env.step(np.zeros((2, 3, 3)))


def test_vector_env_shapes():
    observations, states, masks, infos = MAVUAVVectorEnv(2, seed=3).reset()
    assert observations.shape == (2, 3, 40) and states.shape == (2, 40) and masks.shape == (2, 3) and len(infos) == 2


def test_vector_env_auto_reset():
    env = MAVUAVVectorEnv(1, short_config(), seed=3); env.reset()
    *_, truncated, masks, infos = env.step(np.zeros((1, 3, 3)))
    assert truncated[0] and infos[0]["auto_reset"] and np.array_equal(masks[0], [1, 1, 1])


def test_vector_env_can_run_1000_steps_without_manual_reset():
    env = MAVUAVVectorEnv(1, short_config(5), seed=4)
    observations, states, masks, _ = env.reset(); rng = np.random.default_rng(4)
    for _ in range(1000):
        observations, states, rewards, _, _, masks, _ = env.step(rng.uniform(-1, 1, (1, 3, 3)))
        assert np.all(np.isfinite(observations)) and np.all(np.isfinite(states)) and np.all(np.isfinite(rewards))
