from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

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


def _write_config(tmp_path: Path, cfg: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _heading_opposite(a: float, b: float, atol: float = 1e-9) -> bool:
    diff = abs(a - b)
    diff = min(diff, 2.0 * np.pi - diff)
    return bool(np.isclose(diff, np.pi, atol=atol))


def _assert_logical_mirror_pairs(env: Homogeneous3v3AirCombatEnv) -> None:
    spec_fields = (
        "v_min", "v_max", "theta_min", "theta_max", "nx_min", "nx_max",
        "nz_min", "nz_max", "phi_max", "yaw_rate_max", "pitch_rate_max",
        "acceleration_max", "k_yaw", "k_pitch", "k_speed",
    )
    for rid, bid in zip(RED_IDS, BLUE_IDS):
        red = env._aircraft_by_id(rid)
        blue = env._aircraft_by_id(bid)
        np.testing.assert_allclose(
            [red.state.x, red.state.y, red.state.z],
            [-blue.state.x, -blue.state.y, blue.state.z],
            atol=1e-9,
        )
        assert red.state.altitude == pytest.approx(blue.state.altitude)
        assert red.state.v == pytest.approx(blue.state.v)
        assert red.state.theta == pytest.approx(blue.state.theta)
        assert _heading_opposite(red.state.psi, blue.state.psi)
        assert red.state.alive == blue.state.alive
        assert red.role == blue.role
        assert red.sensor_range == blue.sensor_range
        assert red.can_attack == blue.can_attack
        for field in spec_fields:
            assert getattr(red.spec, field) == getattr(blue.spec, field)
    assert env._aircraft_by_id("red_0").role == env._aircraft_by_id("blue_0").role == "support"
    assert env._aircraft_by_id("red_1").role == env._aircraft_by_id("blue_1").role == "combat"
    assert env._aircraft_by_id("red_2").role == env._aircraft_by_id("blue_2").role == "combat"


def test_new_config_only_changes_v6_by_heterogeneous_reward_and_rules():
    v6 = load_config(V6)
    h = load_config(HETERO)
    for key in ("simulation", "action", "aircraft", "battlefield", "scenario"):
        assert h[key] == v6[key]
    assert h["combat"]["reward_mode"] == "functional_heterogeneous_team_v1"
    assert h["combat"]["timeout_outcome_mode"] == v6["combat"]["timeout_outcome_mode"]
    assert h["blue_rule_policy"] == {"mode": "functional_heterogeneous_team_v1"}
    assert h["red_rule_policy"] == {"mode": "functional_heterogeneous_team_v1"}
    assert h["heterogeneous"]["information_sharing"] == {"support_to_combat": True}
    assert "observation" not in h["heterogeneous"]


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


def test_heterogeneous_logical_id_mirror_pairs_for_32_default_jitter_seeds():
    env = Homogeneous3v3AirCombatEnv(HETERO)
    for seed in range(32):
        env.reset(seed)
        _assert_logical_mirror_pairs(env)


def test_heterogeneous_logical_id_mirror_pairs_without_jitter(tmp_path):
    cfg = load_config(HETERO)
    cfg["scenario"]["separation_min"] = 4200.0
    cfg["scenario"]["separation_max"] = 4200.0
    cfg["scenario"]["speed_jitter"] = 0.0
    cfg["scenario"]["altitude_jitter"] = 0.0
    cfg["scenario"]["heading_jitter"] = 0.0
    env = Homogeneous3v3AirCombatEnv(_write_config(tmp_path, cfg))
    for seed in range(4):
        env.reset(seed)
        _assert_logical_mirror_pairs(env)


def test_v6_historical_reverse_physical_pairing_is_preserved():
    env = Homogeneous3v3AirCombatEnv(V6)
    env.reset(17)
    for rid, bid in (("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")):
        red = env._aircraft_by_id(rid)
        blue = env._aircraft_by_id(bid)
        np.testing.assert_allclose(
            [red.state.x, red.state.y, red.state.z],
            [-blue.state.x, -blue.state.y, blue.state.z],
            atol=1e-9,
        )
        assert red.state.v == pytest.approx(blue.state.v)
        assert _heading_opposite(red.state.psi, blue.state.psi)
    red0 = env._aircraft_by_id("red_0")
    blue0 = env._aircraft_by_id("blue_0")
    assert not np.allclose([red0.state.x, red0.state.y], [-blue0.state.x, -blue0.state.y], atol=1e-9)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda cfg: cfg["heterogeneous"]["roles"].pop("blue_2"), "exactly cover"),
        (lambda cfg: cfg["heterogeneous"]["roles"].update({"ghost_0": "combat"}), "exactly cover"),
        (lambda cfg: cfg["heterogeneous"]["roles"].update({"red_1": "support"}), "fixed roles"),
        (lambda cfg: cfg["heterogeneous"]["roles"].update({"red_1": "jammer"}), "fixed roles"),
        (lambda cfg: cfg["heterogeneous"]["sensor_range"].update({"combat": 0.0}), "finite and positive"),
        (lambda cfg: cfg["heterogeneous"]["sensor_range"].update({"combat": -1.0}), "finite and positive"),
        (lambda cfg: cfg["heterogeneous"]["sensor_range"].update({"combat": float("nan")}), "finite and positive"),
        (lambda cfg: cfg["heterogeneous"]["sensor_range"].update({"combat": float("inf")}), "finite and positive"),
        (lambda cfg: cfg["heterogeneous"]["can_attack"].update({"combat": "true"}), "must be bool"),
        (lambda cfg: cfg["heterogeneous"]["support_rule"].update({"mode": "loiter"}), "unsupported support rule mode"),
        (lambda cfg: cfg["heterogeneous"]["support_rule"].update({"follow_distance": -1.0}), "follow_distance"),
    ],
)
def test_invalid_heterogeneous_configs_raise_clear_errors(tmp_path, mutator, match):
    cfg = load_config(HETERO)
    mutator(cfg)
    with pytest.raises((ValueError, KeyError), match=match):
        Homogeneous3v3AirCombatEnv(_write_config(tmp_path, cfg))


def test_default_config_shared_observation_kills_are_structurally_expected_zero():
    cfg = load_config(HETERO)
    assert cfg["heterogeneous"]["sensor_range"]["combat"] >= cfg["combat"]["attack_distance_max"]
    vec = LocalCombatVectorEnv3v3(HETERO, 2)
    try:
        vec.reset([{"seed": 800}, {"seed": 801}])
        total_red_shared = 0
        total_blue_shared = 0
        modes = np.ones((2, 2), dtype=np.int8)
        episodes = 0
        while episodes < 4:
            result = vec.step_rules(modes)
            done = np.where(result.episode_valid)[0]
            total_red_shared += int(np.sum(result.episode_red_kills_with_shared_observation[done]))
            total_blue_shared += int(np.sum(result.episode_blue_kills_with_shared_observation[done]))
            episodes += len(done)
            if len(done):
                vec.reset_at(done, [{"seed": 900 + int(i)} for i in done])
        assert total_red_shared == 0
        assert total_blue_shared == 0
    finally:
        vec.close()


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


def test_support_sharing_stops_when_support_dies_and_combats_never_share():
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


def test_support_to_combat_false_disables_effective_visibility_rule_and_attack(tmp_path):
    cfg = load_config(HETERO)
    cfg["heterogeneous"]["information_sharing"]["support_to_combat"] = False
    env = Homogeneous3v3AirCombatEnv(_write_config(tmp_path, cfg))
    env.reset(0)
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0, psi=0.0)
    _set_state(env, "red_1", 0.0, 100.0, psi=0.0)
    _set_state(env, "blue_1", 800.0, 0.0, psi=0.0)
    env._aircraft_by_id("red_1").sensor_range = 500.0
    assert "blue_1" in env._direct_visible_enemy_ids(env._aircraft_by_id("red_0"))
    assert "blue_1" not in env._effective_visible_enemy_ids(env._aircraft_by_id("red_1"))
    assert env._support_coverage("red")[1] >= 1

    policy = make_team_rule_policy_3v3(env.config, team="red")
    actions, targets = policy.select_actions(
        [env._aircraft_by_id(aid) for aid in RED_IDS],
        [env._aircraft_by_id(aid) for aid in BLUE_IDS],
        visible_enemy_ids_by_own=env.visible_enemy_ids_by_own("red"),
    )
    assert targets["red_1"] is None
    np.testing.assert_array_equal(actions["red_1"], np.zeros(3, dtype=np.float32))

    _, _, _, _, info = env.step(_zero_actions())
    assert info["attacks"]["red_1"] is None
    assert info["attack_kills"]["red"] == 0


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


def test_support_shared_visibility_changes_combat_enemy_visibility_but_not_support_slot():
    env = _env()
    _spread_defaults(env)
    _set_state(env, "red_0", 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 5000.0)
    _set_state(env, "blue_1", 5000.0, 0.0)
    combat_obs = env._agent_observation(env._aircraft_by_id("red_1"))
    support_obs = env._agent_observation(env._aircraft_by_id("red_0"))
    assert 1.0 in combat_obs[[43, 55, 67]]
    assert 1.0 in support_obs[[43, 55, 67]]
    env._aircraft_by_id("red_0").state.alive = False
    combat_obs = env._agent_observation(env._aircraft_by_id("red_1"))
    assert 1.0 not in combat_obs[[43, 55, 67]]


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


def test_audit_heterogeneous_symmetry_script_smoke_generates_json(tmp_path):
    output = tmp_path / "audit.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "audit_heterogeneous_3v3_symmetry.py"),
        "--env-config", str(HETERO),
        "--episodes", "2",
        "--seed", "42000",
        "--num-envs", "1",
        "--env-workers", "1",
        "--output-json", str(output),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    assert completed.returncode == 0
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["episodes"] == 2
    assert data["mirror_preflight_passed"] is True
    assert data["termination"]["max_steps_count"] + data["termination"]["red_elimination_count"] + data["termination"]["blue_elimination_count"] + data["termination"]["mutual_elimination_count"] == 2
    assert set(data["complete_attack_elimination"]) == {
        "red_complete_elimination_success_count",
        "blue_complete_elimination_success_count",
    }
    assert data["integrity"] == {"finite_failures": 0, "ledger_failures": 0, "worker_failures": 0}
