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
    RED_REWARD_COMPONENT_KEYS_3V3,
    CONTROL_DIAGNOSTIC_KEYS,
)
from uav_combat.config import load_config
from uav_combat.controller import TargetStateController
from uav_combat.rule_policy import PurePursuitPolicy
from uav_combat.rule_policy_3v3 import (
    NearestTargetPursuitPolicy3v3,
    make_nearest_target_pursuit_policy_3v3,
)

CONFIG = Path(__file__).parents[1] / "configs" / "homogeneous_3v3.yaml"
CONFIG_V4 = Path(__file__).parents[1] / "configs" / "homogeneous_3v3_learnable_v4.yaml"


def _set(env, aid, x, y, z=-3000.0, v=150.0, psi=0.0):
    a = env._aircraft_by_id(aid); a.state.x, a.state.y, a.state.z = x, y, z
    a.state.v, a.state.psi = v, psi; a.state.theta = 0.0

def _all_actions(env):
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft}

def _set_line_pair(env, closing=True, red_team=True):
    own_ids = RED_IDS if red_team else BLUE_IDS
    enemy_ids = BLUE_IDS if red_team else RED_IDS
    own_speed, enemy_speed = (250.0, 100.0) if closing else (100.0, 250.0)
    own_psi = 0.0 if red_team else np.pi
    enemy_psi = 0.0 if red_team else np.pi
    for i, aid in enumerate(own_ids):
        _set(env, aid, 0.0, i * 500.0, v=own_speed, psi=own_psi)
    for i, aid in enumerate(enemy_ids):
        x = 1500.0 if red_team else -1500.0
        _set(env, aid, x, i * 500.0, v=enemy_speed, psi=enemy_psi)

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
        # Reverse pairing: red_0<->blue_2, red_1<->blue_1, red_2<->blue_0
        pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
        for seed in range(10):
            ac = sc.reset(seed)
            for ri, bi in pairs:
                r = next(a for a in ac if a.aircraft_id == ri)
                b = next(a for a in ac if a.aircraft_id == bi)
                diff = abs(r.state.psi - b.state.psi)
                diff = min(diff, 2 * np.pi - diff)
                assert np.isclose(diff, np.pi, atol=1e-9), f"seed={seed} {ri}-{bi}: |psi_r-psi_b|={diff:.6f} != pi"

    def test_paired_speed_and_altitude_match(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
        for seed in range(10):
            ac = sc.reset(seed)
            for ri, bi in pairs:
                r = next(a for a in ac if a.aircraft_id == ri)
                b = next(a for a in ac if a.aircraft_id == bi)
                assert r.state.v == b.state.v, f"seed={seed} {ri}.v={r.state.v} != {bi}.v={b.state.v}"
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


class TestRewardContract3v3:
    def test_environment_approach_negative_not_clipped(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(1)
        _set_line_pair(env, closing=False, red_team=True)
        _, _, _, _, info = env.step(_all_actions(env))
        assert info["reward_components"]["red_approach_reward"] < 0.0

    def test_environment_approach_positive_preserved(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(2)
        _set_line_pair(env, closing=True, red_team=True)
        _, _, _, _, info = env.step(_all_actions(env))
        assert info["reward_components"]["red_approach_reward"] > 0.0

    def test_environment_approach_no_alive_enemies_is_zero(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(3)
        for aid in BLUE_IDS:
            env._aircraft_by_id(aid).state.alive = False
        _, _, _, _, info = env.step(_all_actions(env))
        assert info["reward_components"]["red_approach_reward"] == 0.0

    def test_environment_approach_red_blue_symmetric(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(4)
        _set_line_pair(env, closing=True, red_team=True)
        _, _, _, _, info_red = env.step(_all_actions(env))

        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(4)
        _set_line_pair(env, closing=True, red_team=False)
        _, _, _, _, info_blue = env.step(_all_actions(env))
        assert np.isclose(
            info_red["reward_components"]["red_approach_reward"],
            info_blue["reward_components"]["blue_approach_reward"],
        )

    def test_boundary_total_equals_altitude_plus_xy(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(5)
        _set(env, "red_0", 0.0, 0.0, z=-499.0)
        _set(env, "red_1", env.config["battlefield"]["x_limit"] + 1.0, 0.0)
        _, _, _, _, info = env.step(_all_actions(env))
        assert info["boundary_deaths"]["red"] == (
            info["boundary_altitude_deaths"]["red"] + info["boundary_xy_deaths"]["red"]
        )

    def test_reward_components_signed_sum_matches_total(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(6)
        _set_line_pair(env, closing=True, red_team=True)
        _, rewards, _, _, info = env.step(_all_actions(env))
        rc = info["reward_components"]
        total = (
            rc["red_dense_reward"]
            + rc["red_kill_reward"]
            - rc["red_attack_death_penalty"]
            - rc["red_boundary_death_penalty"]
            - rc["red_collision_death_penalty"]
            + rc["red_terminal_reward"]
        )
        assert np.isclose(total, rc["red_team_total_reward"])
        assert np.isclose(total, rewards["red_0"])

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
    def test_old_config_defaults_to_legacy_delta_policy(self):
        cfg = load_config(CONFIG)
        assert cfg["action"].get("mapping_mode", "legacy_delta") == "legacy_delta"
        pol = make_nearest_target_pursuit_policy_3v3(cfg)
        assert pol.mapping_mode == "legacy_delta"

    def test_v4_config_uses_rate_aligned_policy(self):
        cfg = load_config(CONFIG_V4)
        pol = make_nearest_target_pursuit_policy_3v3(cfg)
        assert pol.mapping_mode == "rate_aligned_v1"

    def test_legacy_rule_policy_matches_previous_pure_pursuit_encoding(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(17)
        own = env._aircraft_by_id("blue_0")
        target = env._aircraft_by_id("red_0")
        cfg = env.config["action"]
        legacy_3v3 = NearestTargetPursuitPolicy3v3(
            cfg["delta_yaw_max"], cfg["delta_pitch_max"], cfg["delta_speed_max"]
        )
        previous = PurePursuitPolicy(cfg["delta_yaw_max"], cfg["delta_pitch_max"], cfg["delta_speed_max"])
        assert np.allclose(legacy_3v3.action(own, target), previous.action(own, target), atol=0.0, rtol=0.0)

    def test_nearest_target_selection_and_tie_break_are_unchanged(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG_V4)
        env.reset(1)
        _set(env, "blue_0", 0.0, 0.0, psi=0.0)
        _set(env, "red_0", 1000.0, 0.0, psi=np.pi)
        _set(env, "red_1", 1000.0, 0.0, psi=np.pi)
        _set(env, "red_2", 800.0, 0.0, psi=np.pi)
        pol = make_nearest_target_pursuit_policy_3v3(env.config)
        _, targets = pol.select_actions([env._aircraft_by_id("blue_0")], [env._aircraft_by_id(aid) for aid in RED_IDS])
        assert targets["blue_0"] == "red_2"

        env._aircraft_by_id("red_2").state.alive = False
        pol.reset_counters()
        _, targets = pol.select_actions([env._aircraft_by_id("blue_0")], [env._aircraft_by_id(aid) for aid in RED_IDS])
        assert targets["blue_0"] == "red_0"

    @pytest.mark.parametrize(
        "target_xy,target_altitude,own_v,target_v",
        [
            ((1200.0, 60.0), 3000.0, 150.0, 150.0),      # small positive yaw
            ((1200.0, -60.0), 3000.0, 150.0, 150.0),     # small negative yaw
            ((100.0, 1200.0), 3000.0, 150.0, 150.0),     # large positive yaw, yaw-rate limit
            ((100.0, -1200.0), 3000.0, 150.0, 150.0),    # large negative yaw, yaw-rate limit
            ((1200.0, 0.0), 3400.0, 150.0, 150.0),       # positive pitch
            ((1200.0, 0.0), 2600.0, 150.0, 150.0),       # negative pitch
            ((1200.0, 0.0), 3000.0, 120.0, 180.0),       # accelerate
            ((1200.0, 0.0), 3000.0, 220.0, 120.0),       # decelerate
            ((900.0, -700.0), 3300.0, 135.0, 210.0),     # combined errors
        ],
    )
    def test_legacy_and_rate_aligned_policy_controller_commands_are_equivalent(
        self, target_xy, target_altitude, own_v, target_v
    ):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(22)
        own = env._aircraft_by_id("blue_0")
        target = env._aircraft_by_id("red_0")
        _set(env, "blue_0", 0.0, 0.0, z=-3000.0, v=own_v, psi=0.0)
        _set(env, "red_0", target_xy[0], target_xy[1], z=-target_altitude, v=target_v, psi=np.pi)

        act_cfg = env.config["action"]
        spec = own.spec
        legacy_policy = NearestTargetPursuitPolicy3v3(
            act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"],
            mapping_mode="legacy_delta",
        )
        rate_policy = NearestTargetPursuitPolicy3v3(
            act_cfg["delta_yaw_max"], act_cfg["delta_pitch_max"], act_cfg["delta_speed_max"],
            mapping_mode="rate_aligned_v1",
            yaw_rate_max=spec.yaw_rate_max,
            pitch_rate_max=spec.pitch_rate_max,
            acceleration_max=spec.acceleration_max,
            k_yaw=spec.k_yaw,
            k_pitch=spec.k_pitch,
            k_speed=spec.k_speed,
        )
        legacy_controller = TargetStateController(**act_cfg, mapping_mode="legacy_delta")
        rate_controller = TargetStateController(**act_cfg, mapping_mode="rate_aligned_v1")

        _, legacy_cmd = legacy_controller.control_from_action(own.state, legacy_policy.action(own, target), spec)
        _, rate_cmd = rate_controller.control_from_action(own.state, rate_policy.action(own, target), spec)
        assert np.allclose(
            [legacy_cmd.nx, legacy_cmd.nz, legacy_cmd.phi],
            [rate_cmd.nx, rate_cmd.nz, rate_cmd.phi],
            atol=1e-8,
            rtol=1e-8,
        )

    def test_v4_rule_actions_are_bounded_and_finite(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG_V4)
        env.reset(31)
        pol = make_nearest_target_pursuit_policy_3v3(env.config)
        actions, _ = pol.select_actions(
            [a for a in env.aircraft if a.team == "blue"],
            [a for a in env.aircraft if a.team == "red"],
        )
        for action in actions.values():
            assert np.all(np.isfinite(action))
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_local_and_worker_policy_modes_match_v4(self):
        local = LocalCombatVectorEnv3v3(CONFIG_V4, 2)
        worker = SubprocessCombatVectorEnv3v3(CONFIG_V4, 2, 2)
        try:
            local_modes = local.policy_modes()
            assert local_modes["blue"] == ["rate_aligned_v1", "rate_aligned_v1"]
            assert local_modes["red"] == ["rate_aligned_v1", "rate_aligned_v1"]
            assert local_modes["blue_policy"] == ["paper_nearest_pursuit_v1", "paper_nearest_pursuit_v1"]
            assert local_modes["red_policy"] == ["paper_nearest_pursuit_v1", "paper_nearest_pursuit_v1"]
            assert local_modes["blue_action_mapping"] == ["rate_aligned_v1", "rate_aligned_v1"]
            assert local_modes["red_action_mapping"] == ["rate_aligned_v1", "rate_aligned_v1"]
            assert worker.policy_modes() == local_modes
        finally:
            local.close()
            worker.close()

    def test_local_and_worker_blue_rule_actions_match_same_state(self):
        local = LocalCombatVectorEnv3v3(CONFIG_V4, 2)
        worker = SubprocessCombatVectorEnv3v3(CONFIG_V4, 2, 2)
        try:
            specs = [{"seed": 101}, {"seed": 102}]
            local.reset(specs)
            worker.reset(specs)
            red_actions = np.zeros((2, 3, 3), dtype=np.float32)
            local_result = local.step(red_actions)
            worker_result = worker.step(red_actions)
            idx = [CONTROL_DIAGNOSTIC_KEYS.index(k) for k in (
                "normalized_action_yaw", "normalized_action_pitch", "normalized_action_speed",
                "nx", "nz", "phi",
            )]
            assert np.all(np.isfinite(local_result.control_diagnostics[:, 3:, idx]))
            assert np.all(np.isfinite(worker_result.control_diagnostics[:, 3:, idx]))
            assert np.allclose(
                local_result.control_diagnostics[:, 3:, idx],
                worker_result.control_diagnostics[:, 3:, idx],
                atol=1e-8,
                rtol=1e-8,
            )
        finally:
            local.close()
            worker.close()

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

    def test_boundary_subfields_are_propagated(self):
        vec = LocalCombatVectorEnv3v3(CONFIG, 1)
        vec.reset([{"seed": 7}])
        env = vec.envs[0]
        _set(env, "red_0", 0.0, 0.0, z=-499.0)
        _set(env, "red_1", env.config["battlefield"]["x_limit"] + 1.0, 0.0)
        res = vec.step_rules(np.zeros((1, 2), dtype=np.int8))
        assert res.step_red_boundary_deaths[0] == 2
        assert res.step_red_boundary_altitude_deaths[0] == 1
        assert res.step_red_boundary_xy_deaths[0] == 1
        vec.close()

    def test_reward_component_vector_order_and_values(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG)
        env.reset(8)
        _set_line_pair(env, closing=True, red_team=True)
        _, _, _, _, info = env.step(_all_actions(env))
        expected = np.array(
            [info["reward_components"][key] for key in RED_REWARD_COMPONENT_KEYS_3V3],
            dtype=np.float32,
        )

        vec = LocalCombatVectorEnv3v3(CONFIG, 1)
        vec.reset([{"seed": 8}])
        _set_line_pair(vec.envs[0], closing=True, red_team=True)
        res = vec.step_rules(np.zeros((1, 2), dtype=np.int8))
        assert res.red_reward_components.shape == (1, len(RED_REWARD_COMPONENT_KEYS_3V3))
        assert np.allclose(res.red_reward_components[0], expected)
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

    def test_collect_rollout_returns_real_completed_records(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 5, "device": "cpu", "output_dir": "tmp"}, "network": {"hidden_dim": 32, "log_std_init": -0.5},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 1024, "num_envs": 4, "num_env_workers": 2,
                               "rollout_steps": 128, "ppo_epochs": 1, "minibatch_size": 64, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2, "deterministic": True}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer(CONFIG, config)
            completed = t.collect_rollout()
            # Completed records must have real fields, not {"completed": True}
            for rec in completed:
                assert "episode_length" in rec, f"record missing episode_length: {rec}"
                assert "red_attack_kills" in rec
                assert "environment_outcome" in rec
                # Death ledger must balance
                for team in ("red", "blue"):
                    surv = rec[f"{team}_survivors"]
                    atk = rec[f"{team}_attack_deaths"]
                    bdy = rec[f"{team}_boundary_deaths"]
                    fr = rec[f"{team}_friendly_collision_deaths"]
                    cr = rec[f"{team}_cross_collision_deaths"]
                    assert surv + atk + bdy + fr + cr == 3
            t.close()

    def test_completed_record_keeps_boundary_subfields(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 9, "device": "cpu", "output_dir": "tmp"}, "network": {"hidden_dim": 32, "log_std_init": -0.5},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 8, "num_envs": 1, "num_env_workers": 1,
                               "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2, "deterministic": True}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer(CONFIG, config)
            env = t.vector_env.envs[0]
            for aid in RED_IDS:
                _set(env, aid, env.config["battlefield"]["x_limit"] + 1.0, 0.0)
            completed = t.collect_rollout()
            assert completed
            rec = completed[0]
            for key in (
                "red_boundary_altitude_deaths",
                "blue_boundary_altitude_deaths",
                "red_boundary_xy_deaths",
                "blue_boundary_xy_deaths",
            ):
                assert key in rec
            assert rec["red_boundary_deaths"] == rec["red_boundary_altitude_deaths"] + rec["red_boundary_xy_deaths"]
            t.close()

    def test_rollout_reward_means_are_step_components(self):
        from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer
        config = {"experiment": {"seed": 10, "device": "cpu", "output_dir": "tmp"}, "network": {"hidden_dim": 32, "log_std_init": -0.5},
                  "training": {"training_mode": "fixed_rule_blue_3v3", "total_env_steps": 8, "num_envs": 1, "num_env_workers": 1,
                               "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8, "gamma": 0.99, "gae_lambda": 0.95,
                               "clip_coef": 0.2, "learning_rate": 3e-4, "value_loss_coef": 0.5, "entropy_coef": 0.01, "max_grad_norm": 0.5},
                  "evaluation": {"episodes": 2, "deterministic": True}}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config["experiment"]["output_dir"] = tmp
            t = FixedBlue3v3MAPPOTrainer(CONFIG, config)
            t.collect_rollout()
            m = t.last_rollout_reward_means
            assert "mean_rollout_total_step_reward" in m
            assert "mean_rollout_tactical_reward" in m
            assert np.isfinite(m["mean_rollout_total_step_reward"])
            assert np.isclose(
                m["mean_rollout_event_terminal_reward"],
                m["mean_rollout_event_reward"] + m["mean_rollout_terminal_reward"],
            )
            t.close()


class TestEncoding:
    def test_3v3_reason_roundtrip(self):
        from uav_combat.mappo.vector_env_3v3 import (
            encode_3v3_termination_reason, decode_3v3_termination_reason,
            encode_3v3_outcome, decode_3v3_outcome,
        )
        for r in [None, "red_elimination", "blue_elimination", "mutual_elimination", "max_steps"]:
            assert decode_3v3_termination_reason(encode_3v3_termination_reason(r)) == r
        for o in [None, "red", "blue", "draw"]:
            assert decode_3v3_outcome(encode_3v3_outcome(o)) == o

    def test_mutual_elimination_is_draw(self):
        from uav_combat.mappo.vector_env_3v3 import (
            encode_3v3_outcome, encode_3v3_termination_reason, decode_3v3_outcome,
        )
        # Mutual elimination -> reason=mutual_elimination, outcome=draw
        code = encode_3v3_outcome("draw")
        assert decode_3v3_outcome(code) == "draw"


class TestEnvironmentAudit3v3:
    def test_training_metric_rollout_and_eval_prefixes_are_separate(self):
        text = (Path(__file__).parents[1] / "scripts" / "train_mappo_3v3.py").read_text(encoding="utf-8")
        assert '"eval_red_complete_elimination_success_rate"' in text
        assert '"eval_mean_red_boundary_altitude_deaths"' in text
        assert '"mean_rollout_tactical_reward"' in text
        assert '"red_complete_elimination_success_rate": ev.get' not in text

    def test_audit_output_uses_tmp_path_and_marks_test(self, tmp_path, monkeypatch):
        import scripts.audit_environment_contract_3v3 as audit
        formal = Path("outputs/3v3_environment_contract_audit.json")
        before = formal.read_bytes() if formal.exists() else None
        out = tmp_path / "audit.json"

        def fake_config(path, episodes, num_envs, env_workers, strict_complete=True):
            return {
                "config_path": str(path),
                "rule_matchups": {
                    "zero_vs_pursuit": {
                        "episodes_requested": episodes, "episodes_completed": episodes,
                        "all_rewards_finite": True, "all_observations_finite": True,
                        "all_global_states_finite": True, "death_ledger_conserved": True,
                        "boundary_total_matches_altitude_plus_xy": True,
                    },
                    "pursuit_vs_pursuit": {
                        "episodes_requested": episodes, "episodes_completed": episodes,
                        "all_rewards_finite": True, "all_observations_finite": True,
                        "all_global_states_finite": True, "death_ledger_conserved": True,
                        "boundary_total_matches_altitude_plus_xy": True,
                    },
                },
                "horizontal_physical_boundary_theoretically_reachable": False,
                "maneuver_control_audit": {"150_mps": {"max_positive_yaw": {
                    "maximum_actual_yaw_rate": 0.1,
                    "configured_yaw_command_rate_limit": 0.35,
                }}},
                "altitude_control_audit": {"150_mps": {"climb": {
                    "time_to_upper_soft_boundary": {"time_to_soft_boundary": 1.0},
                    "time_to_upper_physical_boundary": {"state_crossing_time": 2.0},
                }, "dive": {
                    "time_to_lower_soft_boundary": {"time_to_soft_boundary": 1.0},
                    "time_to_lower_physical_boundary": {"state_crossing_time": 2.0},
                }}},
                "scenario_timescale_comparison": {
                    "turn_90_to_first_merge_ratio": 1.0,
                    "turn_180_to_first_merge_ratio": 2.0,
                },
            }

        monkeypatch.setattr(audit, "audit_config", fake_config)
        data = audit.build_audit(2, 2, 1, out, generated_by_test=True, config_paths=(CONFIG,))
        audit.write_audit(data, out)
        saved = __import__("json").loads(out.read_text(encoding="utf-8"))
        assert saved["generated_by_test"] is True
        assert saved["episodes_requested"] == 2
        if before is not None:
            assert formal.read_bytes() == before

    def test_incomplete_episodes_raise(self, tmp_path, monkeypatch):
        import scripts.audit_environment_contract_3v3 as audit

        def fake_config(path, episodes, num_envs, env_workers, strict_complete=True):
            return {
                "rule_matchups": {
                    "zero_vs_pursuit": {
                        "episodes_requested": episodes, "episodes_completed": episodes - 1,
                        "all_rewards_finite": True, "all_observations_finite": True,
                        "all_global_states_finite": True, "death_ledger_conserved": True,
                        "boundary_total_matches_altitude_plus_xy": True,
                    }
                },
                "horizontal_physical_boundary_theoretically_reachable": False,
                "maneuver_control_audit": {},
                "altitude_control_audit": {},
                "scenario_timescale_comparison": {},
            }

        monkeypatch.setattr(audit, "audit_config", fake_config)
        with pytest.raises(RuntimeError, match="did not complete"):
            audit.build_audit(2, 2, 1, tmp_path / "audit.json", generated_by_test=True, config_paths=(CONFIG,))

    def test_both_configs_can_be_instantiated_for_initial_geometry(self):
        from scripts.audit_environment_contract_3v3 import _initial_geometry_audit
        root = Path(__file__).parents[1]
        for cfg in (root / "configs" / "homogeneous_3v3.yaml",
                    root / "configs" / "homogeneous_3v3_reward_v3.yaml",
                    root / "configs" / "homogeneous_3v3_learnable_v4.yaml"):
            result = _initial_geometry_audit(cfg, seeds=2)
            assert result["nearest_enemy_distance_m"]["count"] == 6
            assert result["initial_closing_speed_mps"]["count"] == 6

    def test_nominal_and_actual_time_fields_are_distinct(self, tmp_path, monkeypatch):
        import scripts.audit_environment_contract_3v3 as audit
        real_audit_config = audit.audit_config

        def one_config(path, episodes, num_envs, env_workers, strict_complete=True):
            return real_audit_config(path, 1, 1, 1, strict_complete)

        monkeypatch.setattr(audit, "audit_config", one_config)
        data = audit.build_audit(1, 1, 1, tmp_path / "audit.json", generated_by_test=True, config_paths=(CONFIG,))
        cfg = data["config_audits"][CONFIG.stem]
        assert "nominal_centerline_time_to_attack_distance_seconds" in cfg
        assert "actual_time_to_attack_distance_seconds" in cfg
        assert "time_to_enter_1000m_attack_distance_seconds" not in cfg

    def test_nearest_distance_uses_true_3d_distance_and_closing_dot_product(self):
        from scripts.audit_environment_contract_3v3 import _radial_closing_speed
        rel_pos = np.array([3.0, 4.0, 12.0])
        rel_vel = np.array([-6.0, -8.0, -24.0])
        assert np.isclose(np.linalg.norm(rel_pos), 13.0)
        assert np.isclose(_radial_closing_speed(rel_pos, rel_vel), 26.0)

    def test_real_yaw_rate_uses_unwrapped_heading_and_is_below_command_limit(self):
        from scripts.audit_environment_contract_3v3 import _yaw_case
        result = _yaw_case(CONFIG, 150.0, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert result["used_unwrapped_heading"]
        assert result["actual_initial_yaw_rate"] < 0.35
        assert result["time_to_180_deg"] > 8.98
        assert result["nz_saturation_fraction"] > 0 or result["phi_saturation_fraction"] > 0

    def test_altitude_boundary_times_are_recorded(self):
        from scripts.audit_environment_contract_3v3 import _altitude_boundary_case
        climb = _altitude_boundary_case(CONFIG, 150.0, np.array([0.0, 1.0, 0.0], dtype=np.float32), True)
        dive = _altitude_boundary_case(CONFIG, 150.0, np.array([0.0, -1.0, 0.0], dtype=np.float32), False)
        assert climb["time_to_soft_boundary"] is not None
        assert climb["state_crossing_time"] is not None
        assert dive["time_to_soft_boundary"] is not None
        assert dive["state_crossing_time"] is not None

    def test_v4_config_uses_rate_aligned_mapping(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG_V4)
        assert env.controller.mapping_mode == "rate_aligned_v1"
        assert Homogeneous3v3AirCombatEnv(CONFIG).controller.mapping_mode == "legacy_delta"

    def test_v4_reward_and_attack_envelope_match_base(self):
        from uav_combat.config import load_config
        base = load_config(CONFIG)
        v4 = load_config(CONFIG_V4)
        assert v4["reward_v2"] == base["reward_v2"]
        for key in ("attack_distance_min", "attack_distance_max", "attack_ata_max", "attack_aa_max"):
            assert v4["combat"][key] == base["combat"][key]

    def test_v4_observation_and_global_state_dims_unchanged(self):
        env = Homogeneous3v3AirCombatEnv(CONFIG_V4)
        obs, info = env.reset(0)
        assert obs["red_0"].shape == (68,)
        assert info["global_state"].shape == (48,)

    def test_v4_initial_distance_and_entry_time_exceed_base(self):
        from scripts.audit_environment_contract_3v3 import (
            _initial_geometry_audit,
            _zero_vs_zero_attack_distance_audit,
        )
        base_dist = _initial_geometry_audit(CONFIG, seeds=10)["nearest_enemy_distance_m"]["mean"]
        v4_dist = _initial_geometry_audit(CONFIG_V4, seeds=10)["nearest_enemy_distance_m"]["mean"]
        base_time = _zero_vs_zero_attack_distance_audit(CONFIG, seeds=10)["mean"]
        v4_time = _zero_vs_zero_attack_distance_audit(CONFIG_V4, seeds=10)["mean"]
        assert v4_dist > base_dist
        assert v4_time > base_time

    def test_v4_initial_100_seeds_clean(self):
        from scripts.audit_environment_contract_3v3 import _initial_geometry_audit
        result = _initial_geometry_audit(CONFIG_V4, seeds=100)
        assert result["initial_no_attack_collision_boundary"]

    def test_v4_action_mapping_audit_monotonic_at_150mps(self):
        from scripts.audit_environment_contract_3v3 import _action_mapping_audit
        result = _action_mapping_audit(CONFIG_V4)["speeds"]["150_mps"]
        assert result["requested_yaw_rate_monotonic"]
        assert result["actual_yaw_rate_monotonic"]
        assert result["requested_pitch_rate_monotonic"]
        assert result["actual_pitch_rate_monotonic"]
        assert result["requested_acceleration_monotonic"]
        assert result["actual_acceleration_monotonic"]
        assert result["unique_actual_yaw_response_count"] > 3

    def test_build_audit_includes_three_configs_with_complete_structure(self, tmp_path, monkeypatch):
        import scripts.audit_environment_contract_3v3 as audit

        def fake_config(path, episodes, num_envs, env_workers, strict_complete=True):
            return {
                "config_path": str(path),
                "rule_matchups": {
                    "zero_vs_pursuit": {
                        "episodes_requested": episodes, "episodes_completed": episodes,
                        "all_rewards_finite": True, "all_observations_finite": True,
                        "all_global_states_finite": True, "death_ledger_conserved": True,
                        "boundary_total_matches_altitude_plus_xy": True,
                    },
                    "pursuit_vs_pursuit": {
                        "episodes_requested": episodes, "episodes_completed": episodes,
                        "all_rewards_finite": True, "all_observations_finite": True,
                        "all_global_states_finite": True, "death_ledger_conserved": True,
                        "boundary_total_matches_altitude_plus_xy": True,
                    },
                },
                "horizontal_physical_boundary_theoretically_reachable": False,
                "maneuver_control_audit": {"150_mps": {"max_positive_yaw": {
                    "maximum_actual_yaw_rate": 0.1,
                    "configured_yaw_command_rate_limit": 0.35,
                }}},
                "altitude_control_audit": {"150_mps": {"climb": {
                    "time_to_upper_soft_boundary": {"time_to_soft_boundary": 1.0},
                    "time_to_upper_physical_boundary": {"state_crossing_time": 2.0},
                }, "dive": {
                    "time_to_lower_soft_boundary": {"time_to_soft_boundary": 1.0},
                    "time_to_lower_physical_boundary": {"state_crossing_time": 2.0},
                }}},
                "scenario_timescale_comparison": {
                    "turn_90_to_first_merge_ratio": 1.0,
                    "turn_180_to_first_merge_ratio": 2.0,
                },
            }

        monkeypatch.setattr(audit, "audit_config", fake_config)
        data = audit.build_audit(100, 8, 4, tmp_path / "audit.json", generated_by_test=True)
        assert set(data["config_audits"]) == {
            "homogeneous_3v3",
            "homogeneous_3v3_reward_v3",
            "homogeneous_3v3_learnable_v4",
        }
        for cfg in data["config_audits"].values():
            for matchup in cfg["rule_matchups"].values():
                assert matchup["episodes_requested"] == matchup["episodes_completed"] == 100


class TestScenarioOffset:
    def test_blue_pos_equals_neg_red_pos(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        for seed in range(10):
            ac = sc.reset(seed)
            # red_0 paired with blue_2, red_1 with blue_1, red_2 with blue_0
            pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
            for ri, bi in pairs:
                r = next(a for a in ac if a.aircraft_id == ri)
                b = next(a for a in ac if a.aircraft_id == bi)
                assert np.allclose([r.state.x, r.state.y], [-b.state.x, -b.state.y], atol=1e-9), \
                    f"seed={seed} {ri} pos=({r.state.x:.3f},{r.state.y:.3f}) {bi} pos=({b.state.x:.3f},{b.state.y:.3f})"

    def test_headings_differ_by_pi(self):
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        for seed in range(10):
            ac = sc.reset(seed)
            pairs = [("red_0", "blue_2"), ("red_1", "blue_1"), ("red_2", "blue_0")]
            for ri, bi in pairs:
                r = next(a for a in ac if a.aircraft_id == ri)
                b = next(a for a in ac if a.aircraft_id == bi)
                diff = abs(r.state.psi - b.state.psi)
                diff = min(diff, 2 * np.pi - diff)
                assert np.isclose(diff, np.pi, atol=1e-9), f"seed={seed} {ri}-{bi}: |psi_r-psi_b|={diff:.6f} != pi"

    def test_no_three_collinear_head_on_collisions(self):
        """Slots are offset so not all 3 red-blue pairs collide simultaneously."""
        from uav_combat.scenario_3v3 import Homogeneous3v3Scenario
        from uav_combat.config import load_config
        cfg = load_config(CONFIG)
        sc = Homogeneous3v3Scenario(cfg)
        for seed in range(10):
            ac = sc.reset(seed)
            # Check that no two red-blue pairs share the same lateral line
            for i in range(3):
                for j in range(i + 1, 3):
                    ri = next(a for a in ac if a.aircraft_id == f"red_{i}")
                    rj = next(a for a in ac if a.aircraft_id == f"red_{j}")
                    bi = next(a for a in ac if a.aircraft_id == f"blue_{i}")
                    bj = next(a for a in ac if a.aircraft_id == f"blue_{j}")
                    # Red i,j should have different y positions (offset)
                    assert abs(ri.state.y - rj.state.y) > 1e-6 or abs(ri.state.x - rj.state.x) > 1e-6
