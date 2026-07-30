from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_combat.config import load_config
from uav_combat.environment_3v3 import OBS_DIM, Homogeneous3v3AirCombatEnv
from uav_combat.mappo.vector_env_3v3 import (
    LocalCombatVectorEnv3v3,
    SubprocessCombatVectorEnv3v3,
)
from uav_combat.rule_policy_3v3 import FunctionalHeterogeneousTeamPolicy3v3, make_team_rule_policy_3v3
from uav_combat.scenario_3v3 import ALL_IDS, BLUE_IDS, RED_IDS


ROOT = Path(__file__).parents[1]
HETERO = ROOT / "configs" / "heterogeneous_3v3_functional_v1.yaml"
V6 = ROOT / "configs" / "homogeneous_3v3_learnable_v6_task_aligned.yaml"


def _env(seed: int = 0) -> Homogeneous3v3AirCombatEnv:
    env = Homogeneous3v3AirCombatEnv(HETERO)
    env.reset(seed)
    return env


def _set_state(env, aid: str, x: float, y: float, altitude: float = 3000.0,
               v: float = 150.0, theta: float = 0.0, psi: float = 0.0,
               alive: bool = True) -> None:
    a = env._aircraft_by_id(aid)
    a.state.x = x
    a.state.y = y
    a.state.z = -altitude
    a.state.v = v
    a.state.theta = theta
    a.state.psi = psi
    a.state.alive = alive


def _spread_defaults(env) -> None:
    coords = {
        "red_0": (-10000.0, -4000.0, 0.0),
        "red_1": (-10000.0, 0.0, 0.0),
        "red_2": (-10000.0, 4000.0, 0.0),
        "blue_0": (10000.0, -4000.0, np.pi),
        "blue_1": (10000.0, 0.0, np.pi),
        "blue_2": (10000.0, 4000.0, np.pi),
    }
    for aid, (x, y, psi) in coords.items():
        _set_state(env, aid, x, y, psi=psi, alive=True)


def _zero_actions():
    return {aid: np.zeros(3, dtype=np.float32) for aid in ALL_IDS}


def test_new_config_only_changes_v6_by_heterogeneous_reward_and_rules():
    v6 = load_config(V6)
    h = load_config(HETERO)
    for key in ("simulation", "action", "aircraft", "battlefield", "scenario"):
        assert h[key] == v6[key]
    assert h["combat"]["reward_mode"] == "functional_heterogeneous_team_v1"
    assert h["combat"]["timeout_outcome_mode"] == v6["combat"]["timeout_outcome_mode"]
    assert h["blue_rule_policy"] == {"mode": "functional_heterogeneous_team_v1"}
    assert h["red_rule_policy"] == {"mode": "functional_heterogeneous_team_v1"}


def test_roles_capabilities_and_specs_are_mirrored():
    env = _env()
    roles = {a.aircraft_id: a.role for a in env.aircraft}
    assert roles == {
        "red_0": "support", "red_1": "combat", "red_2": "combat",
        "blue_0": "support", "blue_1": "combat", "blue_2": "combat",
    }
    for aid in ("red_0", "blue_0"):
        a = env._aircraft_by_id(aid)
        assert a.sensor_range == 6000.0
        assert a.can_attack is False
    for aid in ("red_1", "red_2", "blue_1", "blue_2"):
        a = env._aircraft_by_id(aid)
        assert a.sensor_range == 3000.0
        assert a.can_attack is True
    assert env._aircraft_by_id("red_0").spec == env._aircraft_by_id("red_1").spec
    env.close() if hasattr(env, "close") else None


def test_direct_visibility_boundaries_for_combat_and_support():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_1", 0.0, 0.0)
    for distance, expected in ((2999.9, True), (3000.0, True), (3000.1, False)):
        _set_state(env, "blue_1", distance, 0.0)
        assert env._direct_visible(env._aircraft_by_id("red_1"), env._aircraft_by_id("blue_1")) is expected
    _set_state(env, "red_0", 0.0, 0.0)
    _set_state(env, "blue_1", 6000.0, 0.0)
    assert env._direct_visible(env._aircraft_by_id("red_0"), env._aircraft_by_id("blue_1")) is True


def test_support_sharing_stops_when_support_dies_and_combat_to_combat_does_not_share():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 5000.0)
    _set_state(env, "red_2", 0.0, -5000.0)
    _set_state(env, "blue_1", 5000.0, 0.0)
    assert "blue_1" in env._effective_visible_enemy_ids(env._aircraft_by_id("red_1"))
    env._aircraft_by_id("red_0").state.alive = False
    assert "blue_1" not in env._effective_visible_enemy_ids(env._aircraft_by_id("red_1"))
    _set_state(env, "red_2", 2500.0, 0.0)
    assert "blue_1" not in env._effective_visible_enemy_ids(env._aircraft_by_id("red_1"))


def test_heterogeneous_observation_fixed_enemy_slots_and_status_values():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_1", 0.0, 0.0)
    _set_state(env, "blue_0", 2500.0, 0.0, alive=True)
    _set_state(env, "blue_1", 3500.0, 0.0, alive=True)
    _set_state(env, "blue_2", 4500.0, 0.0, alive=False)
    env._aircraft_by_id("red_0").state.alive = False
    obs = env._agent_observation(env._aircraft_by_id("red_1"))
    assert obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs))
    enemy0, enemy1, enemy2 = obs[32:44], obs[44:56], obs[56:68]
    assert enemy0[-1] == 1.0
    assert enemy1[-1] == 0.0
    assert np.all(enemy1[:-1] == 0.0)
    assert enemy2[-1] == -1.0
    assert np.all(enemy2[:-1] == 0.0)


def test_support_shared_visibility_changes_combat_enemy_slot_but_not_support_slot():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 5000.0)
    _set_state(env, "blue_1", 5000.0, 0.0)
    combat_obs = env._agent_observation(env._aircraft_by_id("red_1"))
    support_obs = env._agent_observation(env._aircraft_by_id("red_0"))
    assert combat_obs[44 + 11] == 1.0
    assert support_obs[44 + 11] == 1.0
    env._aircraft_by_id("red_0").state.alive = False
    combat_obs = env._agent_observation(env._aircraft_by_id("red_1"))
    assert combat_obs[44 + 11] == 0.0


def test_support_cannot_attack_even_in_attack_envelope():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0, psi=0.0)
    _set_state(env, "blue_1", 800.0, 0.0, psi=0.0)
    _, _, _, _, info = env.step(_zero_actions())
    assert info["attacks"]["red_0"] is None
    assert info["attack_kills"]["red"] == 0


def test_hidden_target_in_attack_envelope_cannot_be_attacked():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_1", 0.0, 0.0, psi=0.0)
    _set_state(env, "blue_1", 800.0, 0.0, psi=0.0)
    env._aircraft_by_id("red_1").sensor_range = 100.0
    env._aircraft_by_id("red_0").state.alive = False
    _, _, _, _, info = env.step(_zero_actions())
    assert info["attacks"]["red_1"] is None
    assert info["attack_kills"]["red"] == 0


def test_direct_and_shared_combat_attack_and_shared_kill_counting():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_1", 0.0, 0.0, psi=0.0)
    _set_state(env, "blue_1", 800.0, 0.0, psi=0.0)
    _, _, _, _, info = env.step(_zero_actions())
    assert info["attack_kills"]["red"] == 1
    assert info["heterogeneous_metrics"]["red_kills_with_shared_observation"] == 0

    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0, psi=0.0)
    _set_state(env, "red_1", 0.0, 100.0, psi=0.0)
    _set_state(env, "blue_1", 800.0, 0.0, psi=0.0)
    env._aircraft_by_id("red_1").sensor_range = 500.0
    _, _, _, _, info = env.step(_zero_actions())
    assert info["attack_kills"]["red"] == 1
    assert info["heterogeneous_metrics"]["red_kills_with_shared_observation"] == 1


def test_reward_targets_only_use_effective_visible_enemies_and_coverage_is_nonredundant():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 5000.0)
    _set_state(env, "blue_1", 5000.0, 0.0)
    _, rewards, _, _, info = env.step(_zero_actions())
    assert info["reward_targets"]["red_1"] == "blue_1"
    assert 0.0 <= info["heterogeneous_metrics"]["red_support_coverage_ratio"] <= 1.0
    assert info["heterogeneous_metrics"]["red_useful_shared_target_count"] >= 1
    assert rewards["red_0"] == rewards["red_1"] == rewards["red_2"]
    assert np.isfinite(rewards["red_0"])


def test_functional_rule_policy_uses_visible_pairs_and_support_rear_hold():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "blue_0", 0.0, 0.0, psi=0.0)
    _set_state(env, "blue_1", 1200.0, 0.0, psi=0.0)
    _set_state(env, "blue_2", 1200.0, 200.0, psi=0.0)
    _set_state(env, "red_1", 4000.0, 0.0)
    policy = make_team_rule_policy_3v3(load_config(HETERO), team="blue")
    assert isinstance(policy, FunctionalHeterogeneousTeamPolicy3v3)
    actions, targets = policy.select_actions(
        [env._aircraft_by_id(aid) for aid in BLUE_IDS],
        [env._aircraft_by_id(aid) for aid in RED_IDS],
        visible_enemy_ids_by_own={"blue_0": set(), "blue_1": {"red_1"}, "blue_2": set()},
    )
    assert targets["blue_1"] == "red_1"
    assert targets["blue_2"] is None
    assert np.array_equal(actions["blue_2"], np.zeros(3, dtype=np.float32))
    assert actions["blue_0"].shape == (3,)
    assert np.all(np.isfinite(actions["blue_0"]))
    assert np.all(actions["blue_0"] >= -1.0) and np.all(actions["blue_0"] <= 1.0)


def test_vector_env_policy_modes_and_episode_fields_local():
    vec = LocalCombatVectorEnv3v3(HETERO, 1)
    try:
        modes = vec.policy_modes()
        assert modes["blue_policy"] == ["functional_heterogeneous_team_v1"]
        assert modes["red_policy"] == ["functional_heterogeneous_team_v1"]
        obs, gs, am = vec.reset([{"seed": 123}])
        assert obs.shape == (1, 6, OBS_DIM)
        result = vec.step_rules(np.array([[1, 1]], dtype=np.int8))
        assert np.all(np.isfinite(result.observations))
        assert np.all(np.isfinite(result.global_states))
        assert result.episode_red_kills_with_shared_observation.shape == (1,)
        assert result.episode_red_mean_support_coverage_ratio.shape == (1,)
    finally:
        vec.close()


def test_vector_env_worker_runs_functional_policy():
    vec = SubprocessCombatVectorEnv3v3(HETERO, 2, 2)
    try:
        modes = vec.policy_modes()
        assert modes["blue_policy"] == ["functional_heterogeneous_team_v1"] * 2
        obs, gs, am = vec.reset([{"seed": 10}, {"seed": 11}])
        result = vec.step_rules(np.ones((2, 2), dtype=np.int8))
        assert np.all(np.isfinite(result.observations))
        assert np.all(np.isfinite(result.team_rewards))
    finally:
        vec.close()


def test_local_and_worker_same_seed_rule_rollout_match_for_100_steps():
    local = LocalCombatVectorEnv3v3(HETERO, 2)
    worker = SubprocessCombatVectorEnv3v3(HETERO, 2, 2)
    try:
        specs = [{"seed": 501}, {"seed": 502}]
        lo = local.reset(specs)
        wo = worker.reset(specs)
        for left, right in zip(lo, wo):
            np.testing.assert_allclose(left, right, atol=1e-6)
        modes = np.ones((2, 2), dtype=np.int8)
        for _ in range(100):
            lr = local.step_rules(modes)
            wr = worker.step_rules(modes)
            for field in (
                "observations", "global_states", "team_rewards", "terminated", "truncated",
                "alive_masks", "attack_targets", "step_death_causes", "team_alive_counts",
                "step_red_attack_kills", "step_blue_attack_kills",
                "episode_red_kills_with_shared_observation",
                "episode_blue_kills_with_shared_observation",
                "episode_red_mean_support_coverage_ratio",
                "episode_blue_mean_support_coverage_ratio",
            ):
                np.testing.assert_allclose(getattr(lr, field), getattr(wr, field), atol=1e-6)
            done = np.where(lr.episode_valid)[0]
            if len(done):
                reset_specs = [{"seed": 9000 + int(i)} for i in done]
                lo = local.reset_at(done, reset_specs)
                wo = worker.reset_at(done, reset_specs)
                for left, right in zip(lo, wo):
                    np.testing.assert_allclose(left, right, atol=1e-6)
    finally:
        local.close()
        worker.close()
