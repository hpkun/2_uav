"""Tests for 3v3 episode-level death ledger and attack kill accounting."""
from pathlib import Path
import numpy as np
import pytest
from uav_combat.environment_3v3 import (
    Homogeneous3v3AirCombatEnv, RED_IDS, BLUE_IDS,
    DEATH_NONE, DEATH_ATTACK, DEATH_BOUNDARY_XY, DEATH_BOUNDARY_ALTITUDE,
    DEATH_COLLISION_FRIENDLY, DEATH_COLLISION_CROSS,
    _make_episode_summary,
)
from uav_combat.mappo.vector_env_3v3 import (
    VectorStepResult3v3, LocalCombatVectorEnv3v3,
    make_combat_vector_env_3v3, SubprocessCombatVectorEnv3v3,
)

CONFIG = Path(__file__).parents[1] / "configs" / "homogeneous_3v3.yaml"


def _set(env, aid, x, y, z=-3000.0, v=150.0, psi=0.0):
    a = env._aircraft_by_id(aid); a.state.x, a.state.y, a.state.z = x, y, z
    a.state.v, a.state.psi = v, psi; a.state.theta = 0.0

def _all_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft}

def _run_to_end(env):
    while True:
        acts = _all_actions(env)
        obs, rewards, term, trunc, info = env.step(acts)
        if term or trunc:
            es = info.get("episode_summary"); assert es is not None
            return es


class TestScenario:
    def test_headings_differ_by_pi(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        for seed in range(10):
            ac = sc.reset(seed)
            for i in range(3):
                r = next(a for a in ac if a.aircraft_id == f"red_{i}")
                b = next(a for a in ac if a.aircraft_id == f"blue_{i}")
                diff = abs(r.state.psi - b.state.psi)
                diff = min(diff, 2 * np.pi - diff)
                assert np.isclose(diff, np.pi, atol=1e-9), f"seed={seed} slot={i}: |psi_r-psi_b|={diff:.6f} != pi"

    def test_paired_speed_and_altitude_match(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        for seed in range(10):
            ac = sc.reset(seed)
            for i in range(3):
                r = next(a for a in ac if a.aircraft_id == f"red_{i}")
                b = next(a for a in ac if a.aircraft_id == f"blue_{i}")
                assert r.state.v == b.state.v
                assert r.state.altitude == b.state.altitude

    def test_team_size_not_3_raises(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        cfg["scenario"]["team_size"] = 5
        with pytest.raises(ValueError, match="team_size"):
            Homogeneous3v3Scenario(cfg)

    def test_no_initial_attack_collision_boundary(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        obs, info = env.reset(42)
        assert all(v is None for v in info.get("attacks", {}).values())
        for a in env.aircraft:
            bf = env.config["battlefield"]
            assert abs(a.state.x) <= bf["x_limit"]
            assert abs(a.state.y) <= bf["y_limit"]
            assert bf["altitude_min"] <= a.state.altitude <= bf["altitude_max"]
        for i in range(len(env.aircraft)):
            for j in range(i+1, len(env.aircraft)):
                d = np.linalg.norm(env.aircraft[i].state.as_array()[:3] - env.aircraft[j].state.as_array()[:3])
                assert d > bf["collision_distance"]

    def test_seed_reproducible(self):
        env1, env2 = Homogeneous3v3AirCombatEnv(CONFIG), Homogeneous3v3AirCombatEnv(CONFIG)
        o1, _ = env1.reset(123); o2, _ = env2.reset(123)
        for aid in RED_IDS + BLUE_IDS:
            assert np.allclose(o1[aid], o2[aid])

    def test_global_rotation_preserves_distances(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(200)
        dists = []
        for i, a1 in enumerate(env.aircraft):
            for a2 in env.aircraft[i+1:]:
                dists.append(np.linalg.norm(a1.state.as_array()[:3] - a2.state.as_array()[:3]))
        assert all(d > 0 and np.isfinite(d) for d in dists)


class TestDeathLedger:
    def test_initial_all_alive(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(42)
        es = _make_episode_summary({aid: DEATH_NONE for aid in RED_IDS + BLUE_IDS},
                                    {"red": 0, "blue": 0}, 3, 3, None, None, 0)
        assert es["red_survivors"] == 3
        assert es["red_death_causes"]["attack_deaths"] == 0

    def test_single_attack_kill(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(42)
        _set(env, "red_0", 0, 0, psi=0); _set(env, "blue_0", 500, 0, psi=0)
        _set(env, "blue_1", 10000, 10000); _set(env, "blue_2", -10000, -10000)
        _set(env, "red_1", 10000, -10000); _set(env, "red_2", -10000, 10000)
        es = _run_to_end(env)
        assert es["blue_death_causes"]["attack_deaths"] >= 1

    def test_focus_fire_kill_count_one(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(44)
        _set(env, "red_0", 0, -40, psi=0); _set(env, "red_1", 0, 40, psi=0)
        _set(env, "blue_0", 500, 0, psi=0)
        _set(env, "red_2", -10000, -10000); _set(env, "blue_1", 10000, 10000); _set(env, "blue_2", -10000, 10000)
        es = _run_to_end(env)
        assert es["red_attack_kills"] == 1
        assert es["blue_death_causes"]["attack_deaths"] == 1

    def test_three_kills_complete_elimination(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(45)
        _set(env, "red_0", 0, -40, psi=0); _set(env, "red_1", 0, 0, psi=0); _set(env, "red_2", 0, 40, psi=0)
        _set(env, "blue_0", 500, -40, psi=0); _set(env, "blue_1", 500, 0, psi=0); _set(env, "blue_2", 500, 40, psi=0)
        es = _run_to_end(env)
        assert es["red_attack_kills"] == 3
        assert es["red_complete_elimination_success"]

    def test_boundary_death_not_success(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(46)
        for bid in BLUE_IDS: _set(env, bid, env.config["battlefield"]["x_limit"] - 10, 0, v=250, psi=0)
        _set(env, "red_0", 0, 0); _set(env, "red_1", 0, 40); _set(env, "red_2", 0, -40)
        def boundary_acts(e):
            return {a.aircraft_id: np.array([0.0, 0.0, 1.0], np.float32) if a.team == "blue" and a.state.alive
                    else np.zeros(3, np.float32) for a in e.aircraft}
        while True:
            acts = boundary_acts(env)
            obs, rewards, term, trunc, info = env.step(acts)
            if term or trunc:
                es = info["episode_summary"]
                if es["environment_outcome"] == "red":
                    assert not es["red_complete_elimination_success"]
                break

    def test_sync_mutual_kill(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(47)
        _set(env, "red_0", 0, -50, psi=0); _set(env, "blue_0", 500, -50, psi=0)
        _set(env, "blue_1", 0, 50, psi=0); _set(env, "red_1", 500, 50, psi=0)
        _set(env, "red_2", -10000, -10000); _set(env, "blue_2", 10000, 10000)
        es = _run_to_end(env)
        assert es["red_attack_kills"] >= 1 and es["blue_attack_kills"] >= 1

    def test_death_ledger_always_balances(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        for seed in range(20):
            env.reset(seed + 100)
            for _ in range(600):
                alive = [a for a in env.aircraft if a.state.alive]
                if not alive: break
                acts = {a.aircraft_id: np.zeros(3, np.float32) for a in alive}
                obs, rewards, term, trunc, info = env.step(acts)
                if term or trunc:
                    es = info["episode_summary"]
                    for team in ("red", "blue"):
                        dc = es[f"{team}_death_causes"]
                        t = es[f"{team}_survivors"] + dc["attack_deaths"] + dc["boundary_deaths"] + dc["collision_deaths"]
                        assert t == 3, f"seed={seed} {team}: {t} != 3; {es}"
                    assert es["red_attack_kills"] == es["blue_death_causes"]["attack_deaths"]
                    assert es["blue_attack_kills"] == es["red_death_causes"]["attack_deaths"]
                    break

    def test_friendly_cross_collision_separate(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(48)
        _set(env, "red_0", 0, 0); _set(env, "red_1", 10, 0)  # friendly collision
        _set(env, "red_2", 0, 50)
        _set(env, "blue_0", 10000, 0); _set(env, "blue_1", -10000, 0); _set(env, "blue_2", 0, 10000)
        es = _run_to_end(env)
        assert es["red_death_causes"]["friendly_collision_deaths"] >= 1
        assert es["red_death_causes"]["cross_team_collision_deaths"] == 0


class TestGeometry:
    def test_geometry_not_all_zero(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG); env.reset(42)
        acts = _all_actions(env)
        env.step(acts)
        ng = env.step(acts)[4].get("nearest_enemy_geometry", {})
        for aid in RED_IDS + BLUE_IDS:
            g = ng.get(aid, {})
            if g.get("target_id") is not None:
                assert g["distance"] > 0

    def test_geometry_finite(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        for seed in range(5):
            env.reset(seed + 50)
            for _ in range(50):
                acts = _all_actions(env)
                obs, rewards, term, trunc, info = env.step(acts)
                ng = info.get("nearest_enemy_geometry", {})
                for aid, g in ng.items():
                    assert np.isfinite(g["distance"]) and np.isfinite(g["ata"]) and np.isfinite(g["aa"])
                if term or trunc: break


class TestRulePolicy:
    def test_per_env_independent(self):
        vec = LocalCombatVectorEnv3v3(CONFIG, 4)
        specs = [{"seed": i} for i in range(4)]
        vec.reset(specs)
        assert len(vec.blue_policies) == 4
        assert len(vec.red_policies) == 4
        # Each policy should be independent
        assert vec.blue_policies[0] is not vec.blue_policies[1]
        assert vec.red_policies[0] is not vec.red_policies[1]
        vec.close()

    def test_reset_counters_clears_all(self):
        from uav_combat.rule_policy_3v3 import NearestTargetPursuitPolicy3v3
        pol = NearestTargetPursuitPolicy3v3(np.pi, np.pi/3, 50.0)
        pol.target_switch_count["b0"] = 5
        pol.target_selection_count["b0"] = 10
        pol.focus_fire_count = 3
        pol.reset_counters()
        assert pol.target_switch_count == {}
        assert pol.target_selection_count == {}
        assert pol.focus_fire_count == 0


class TestVectorEnv:
    def test_namedtuple_fields(self):
        r = make_combat_vector_env_3v3(CONFIG, 4, 2)
        specs = [{"seed": i} for i in range(4)]
        r.reset(specs)
        res = r.step(np.zeros((4, 3, 3), dtype=np.float32))
        assert isinstance(res, VectorStepResult3v3)
        assert res.observations.shape == (4, 6, 68)
        assert res.global_states.shape == (4, 48)
        assert res.episode_valid.shape == (4,)
        r.close()

    def test_episode_fields_nonzero_on_done(self):
        vec = LocalCombatVectorEnv3v3(CONFIG, 2)
        vec.reset([{"seed": 42}, {"seed": 43}])
        # Run until at least one episode is done
        for _ in range(600):
            r = vec.step(np.zeros((2, 3, 3), dtype=np.float32))
            if r.episode_valid.any():
                vi = int(np.where(r.episode_valid)[0][0])
                assert r.episode_length[vi] > 0
                break
        vec.close()


class TestMAPPO:
    def test_alive_only_advantage_normalization(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 3, "device": "cpu", "output_dir": "tmp"}, "network": {"hidden_dim": 32, "log_std_init": -0.5},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 64, "num_envs": 2, "num_env_workers": 2,
                               "rollout_steps": 16, "ppo_epochs": 1, "minibatch_size": 32, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2, "deterministic": True}}
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer(CONFIG, config)
            t.collect_rollout()
            m = t.update()
            assert np.isfinite(m["policy_loss"])
            assert "alive_actor_sample_fraction" in m
            t.close()
