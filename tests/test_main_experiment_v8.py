from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.config import load_config
from uav_combat.environment_3v3 import (
    DEATH_ATTACK,
    DEATH_BOUNDARY_ALTITUDE,
    DEATH_BOUNDARY_XY,
    DEATH_COLLISION_CROSS,
    DEATH_COLLISION_FRIENDLY,
    DEATH_NONE,
    OBS_DIM,
    Homogeneous3v3AirCombatEnv,
)
from uav_combat.happo.evaluation_3v3 import _summarize as summarize_happo
from uav_combat.happo.trainer_3v3 import HAPPO3v3Trainer
from uav_combat.mappo.evaluation_3v3 import _summarize as summarize_mappo
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer, compute_best_score, compute_best_score_fields
from uav_combat.mappo.vector_env_3v3 import LocalCombatVectorEnv3v3, SubprocessCombatVectorEnv3v3
from uav_combat.main_experiment_v8 import (
    best_score_fields_for_config,
    build_main_v8_contract_metadata,
    compute_best_score_for_config,
    filter_public_metrics_for_config,
    validate_main_v8_contract,
)
from uav_combat.models import AircraftState
from uav_combat.rewards import compute_paper_reward_geometry, paper_equation25_local_reward
from uav_combat.rule_policy_3v3 import make_team_rule_policy_3v3
from uav_combat.scenario_3v3 import Homogeneous3v3Scenario

ROOT = Path(__file__).parents[1]
HOMO_V8 = ROOT / "configs" / "homogeneous_3v3_main_v8.yaml"
HETERO_V8 = ROOT / "configs" / "heterogeneous_3v3_main_v8.yaml"
MAPPO_V8 = ROOT / "configs" / "mappo_3v3_main_v8.yaml"
HAPPO_V8 = ROOT / "configs" / "happo_3v3_main_v8.yaml"
HAPPO_HETERO_V8 = ROOT / "configs" / "happo_heterogeneous_3v3_main_v8.yaml"
V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _state(
    x: float,
    y: float,
    psi: float,
    altitude: float = 3000.0,
    theta: float = 0.0,
    alive: bool = True,
) -> AircraftState:
    return AircraftState(x=x, y=y, z=-altitude, v=150.0, theta=theta, psi=psi, alive=alive)


def _set(
    env: Homogeneous3v3AirCombatEnv,
    aid: str,
    x: float,
    y: float,
    psi: float = 0.0,
    altitude: float = 3000.0,
    theta: float = 0.0,
    alive: bool = True,
) -> None:
    env._aircraft_by_id(aid).state = _state(x, y, psi, altitude, theta, alive)


def _zero_actions(env: Homogeneous3v3AirCombatEnv) -> dict[str, np.ndarray]:
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _rewrite_config(tmp_path: Path, base: Path, mutate) -> Path:
    cfg = load_config(base)
    mutate(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _paper_cfg() -> dict:
    return load_config(HOMO_V8)["reward_v8"]


def _tiny_train_config(path: Path, tmp_path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = deepcopy(yaml.safe_load(f))
    cfg["experiment"]["device"] = "cpu"
    cfg["experiment"]["output_dir"] = str(tmp_path / path.stem)
    cfg["training"]["total_env_steps"] = 8
    cfg["training"]["num_envs"] = 1
    cfg["training"]["num_env_workers"] = 1
    cfg["training"]["rollout_steps"] = 1
    cfg["training"]["minibatch_size"] = 1
    cfg["training"]["ppo_epochs"] = 1
    return cfg


def _summary_record(**overrides):
    base = {
        "red_attack_kills": 0,
        "blue_attack_kills": 0,
        "red_attack_deaths": 0,
        "blue_attack_deaths": 0,
        "red_complete_elimination_success": False,
        "blue_complete_elimination_success": False,
        "environment_outcome": "draw",
        "red_survivors": 3,
        "blue_survivors": 3,
        "red_boundary_deaths": 0,
        "blue_boundary_deaths": 0,
        "red_boundary_altitude_deaths": 0,
        "blue_boundary_altitude_deaths": 0,
        "red_boundary_xy_deaths": 0,
        "blue_boundary_xy_deaths": 0,
        "red_friendly_collision_deaths": 0,
        "blue_friendly_collision_deaths": 0,
        "red_cross_collision_deaths": 0,
        "blue_cross_collision_deaths": 0,
        "red_kills_with_shared_observation": 0,
        "blue_kills_with_shared_observation": 0,
        "red_mean_support_coverage_ratio": 0.0,
        "blue_mean_support_coverage_ratio": 0.0,
        "red_support_survived": True,
        "blue_support_survived": True,
        "termination_reason": "max_steps",
        "episode_length": 600,
        "red_any_attack_kill": False,
        "blue_any_attack_kill": False,
    }
    base.update(overrides)
    return base


def test_v8_configs_are_paper_segmented_and_old_configs_unchanged():
    homo = load_config(HOMO_V8)
    hetero = load_config(HETERO_V8)
    assert homo["combat"]["reward_mode"] == "task_aligned_paper_segmented_team_v8"
    assert hetero["combat"]["reward_mode"] == "task_aligned_heterogeneous_paper_segmented_team_v8"
    assert homo["battlefield"]["collision_distance"] == 0.0
    assert hetero["battlefield"]["collision_distance"] == 0.0
    assert homo["reward_v8"]["paper_segment_distance"] == 4000.0
    assert hetero["reward_heterogeneous_v8"]["paper_segment_distance"] == 4000.0
    assert hetero["blue_rule_policy"]["mode"] == "functional_heterogeneous_nearest_pursuit_v8"
    assert hetero["red_rule_policy"]["mode"] == "functional_heterogeneous_nearest_pursuit_v8"
    for cfg in (homo, hetero):
        assert not ({"preferred_distance", "distance_scale", "terminal_reward"} & set(cfg["combat"]))
    assert load_config(V4)["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert load_config(V4)["battlefield"]["collision_distance"] > 0.0
    assert load_config(V7)["combat"]["reward_mode"] == "paper_segmented_team_v4"
    assert load_config(V7)["battlefield"]["collision_distance"] > 0.0
    validate_main_v8_contract(homo)
    validate_main_v8_contract(hetero)


def test_v8_contract_rejects_positive_collision_distance(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HOMO_V8,
        lambda cfg: cfg["battlefield"].update({"collision_distance": 30.0}),
    )
    with pytest.raises(ValueError, match="collision_distance == 0.0"):
        Homogeneous3v3AirCombatEnv(path)


def test_old_v8_continuous_aliases_are_rejected(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HETERO_V8,
        lambda cfg: cfg["combat"].update({"reward_mode": "task_aligned_heterogeneous_team_v8"}),
    )
    with pytest.raises(ValueError, match="task_aligned_heterogeneous_paper_segmented_team_v8"):
        Homogeneous3v3Scenario(load_config(path))


def test_paper_horizontal_geometry_angles_and_height_angle_are_finite():
    own = _state(0.0, 0.0, 0.0)
    target = _state(1000.0, 0.0, 0.0, altitude=4000.0)
    geom = compute_paper_reward_geometry(own, target)
    assert geom["horizontal_ata"] == pytest.approx(0.0)
    assert geom["horizontal_aa"] == pytest.approx(0.0)
    assert geom["height_angle"] == pytest.approx(np.arctan2(1000.0, 1000.0))

    target_same_xy = _state(0.0, 0.0, 0.0, altitude=3500.0)
    zero = compute_paper_reward_geometry(own, target_same_xy)
    assert np.isfinite(list(zero.values())).all()
    assert zero["horizontal_ata"] == pytest.approx(np.pi)
    assert zero["horizontal_aa"] == pytest.approx(np.pi)
    assert zero["height_angle"] == pytest.approx(np.pi / 2.0)


def test_altitude_changes_height_angle_but_not_horizontal_ata():
    flat = compute_paper_reward_geometry(_state(0, 0, 0), _state(1000, 0, np.pi / 2, altitude=3000.0))
    high = compute_paper_reward_geometry(_state(0, 0, 0), _state(1000, 0, np.pi / 2, altitude=5000.0))
    assert flat["horizontal_ata"] == pytest.approx(high["horizontal_ata"])
    assert high["height_angle"] > flat["height_angle"]


@pytest.mark.parametrize("distance, expected", [(1500.0, 0.0), (3999.0, 0.0), (4000.0, 0.001)])
def test_r3_uses_4000m_distance_gate(distance: float, expected: float):
    reward = paper_equation25_local_reward(_state(0, 0, 0), _state(distance, 0, 0), _paper_cfg())
    assert reward["guide"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "target, expected",
    [
        (_state(4000.0, 0.0, 0.0, altitude=3000.0), 0.001),
        (_state(4000.0, 0.0, 0.0, altitude=7000.0), 0.0),
        (_state(4000.0 * np.cos(np.deg2rad(35)), 4000.0 * np.sin(np.deg2rad(35)), 0.0), 0.0),
    ],
)
def test_r3_requires_horizontal_ata_and_height_angle_gates(target: AircraftState, expected: float):
    reward = paper_equation25_local_reward(_state(0, 0, 0), target, _paper_cfg())
    assert reward["guide"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "angle_deg, altitude_delta, expected",
    [
        (0.0, 0.0, 0.10),
        (10.0, 0.0, 0.02),
        (25.0, 0.0, 0.01),
        (2.0, 2000.0 * np.tan(np.deg2rad(25.0)), 0.01),
        (35.0, 0.0, 0.0),
    ],
)
def test_r41_uses_4000m_and_tiers_by_both_horizontal_ata_and_height_angle(
    angle_deg: float,
    altitude_delta: float,
    expected: float,
):
    horizontal = 2000.0
    angle = np.deg2rad(angle_deg)
    target = _state(
        horizontal * np.cos(angle),
        horizontal * np.sin(angle),
        0.0,
        altitude=3000.0 + altitude_delta,
    )
    reward = paper_equation25_local_reward(_state(0, 0, 0), target, _paper_cfg())
    assert reward["attack_advantage"] == pytest.approx(expected)


def test_r41_is_not_limited_to_attack_envelope_and_has_no_100m_lower_bound():
    cfg = _paper_cfg()
    assert paper_equation25_local_reward(_state(0, 0, 0), _state(3500, 0, 0), cfg)["attack_advantage"] == pytest.approx(0.10)
    assert paper_equation25_local_reward(_state(0, 0, 0), _state(50, 0, 0), cfg)["attack_advantage"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    "enemy_state, expected",
    [
        (_state(-600.0, 0.0, 0.0), -0.150),
        (_state(-600.0, 0.0, np.deg2rad(10)), -0.025),
        (_state(-600.0, 0.0, np.deg2rad(25)), -0.015),
        (_state(600.0, 0.0, 0.0), 0.0),
    ],
)
def test_r42_uses_real_reverse_geometry(enemy_state: AircraftState, expected: float):
    reward = paper_equation25_local_reward(_state(0, 0, 0), enemy_state, _paper_cfg())
    assert reward["threat"] == pytest.approx(expected)


def test_v8_team_aggregation_and_event_rewards_ignore_collision_penalty():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    parts = {
        "red": {"approach_reward": 0.0, "attack_advantage_reward": 0.10 / 3.0, "threat_penalty": 0.0, "dense_reward": 0.10 / 3.0},
        "blue": {"approach_reward": 0.0, "attack_advantage_reward": 0.0, "threat_penalty": 0.0, "dense_reward": 0.0},
    }
    rewards, rc, _ = env._compute_v8_segmented_rewards(
        {"red": 1, "blue": 0},
        {"red_0": DEATH_ATTACK, "red_1": DEATH_BOUNDARY_XY, "red_2": DEATH_COLLISION_CROSS},
        parts,
        {},
    )
    assert rc["red_kill_reward"] == pytest.approx(10.0 / 3.0)
    assert rc["red_attack_death_penalty"] == pytest.approx(-10.0 / 3.0)
    assert rc["red_boundary_death_penalty"] == pytest.approx(-10.0 / 3.0)
    assert rc["red_collision_death_penalty"] == 0.0
    assert rc["red_terminal_reward"] == 0.0
    assert rewards["red_0"] == pytest.approx((0.10 + 10.0 - 10.0 - 10.0) / 3.0)


def test_v8_collision_distance_zero_disables_collision_detection_and_death():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0.0, 0.0, psi=0.0)
    _set(env, "red_1", 0.0, 0.0, psi=0.0)
    _set(env, "red_2", 0.0, 1000.0, psi=0.0)
    _set(env, "blue_0", 9000.0, 0.0, psi=np.pi)
    _set(env, "blue_1", 9000.0, 1000.0, psi=np.pi)
    _set(env, "blue_2", 9000.0, -1000.0, psi=np.pi)

    _, _, terminated, truncated, info = env.step(_zero_actions(env))

    assert info["collision_pairs"] == []
    assert info["collision_deaths"] == {"red": 0, "blue": 0}
    assert DEATH_COLLISION_FRIENDLY not in info["death_causes"].values()
    assert DEATH_COLLISION_CROSS not in info["death_causes"].values()
    assert env._aircraft_by_id("red_0").state.alive
    assert env._aircraft_by_id("red_1").state.alive
    assert not terminated
    assert not truncated


def test_v8_collision_disabled_death_ledger_contains_only_attack_and_boundary_causes():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0.0, 0.0, psi=0.0)
    _set(env, "red_1", 0.0, 0.0, psi=0.0)
    _set(env, "red_2", 25000.0, 0.0, psi=0.0)  # one boundary death is allowed
    _set(env, "blue_0", 9000.0, 0.0, psi=np.pi)
    _set(env, "blue_1", 9000.0, 1000.0, psi=np.pi)
    _set(env, "blue_2", 9000.0, -1000.0, psi=np.pi)

    _, _, _, _, info = env.step(_zero_actions(env))

    allowed = {DEATH_NONE, DEATH_ATTACK, DEATH_BOUNDARY_ALTITUDE, DEATH_BOUNDARY_XY}
    assert set(env._episode_death_causes.values()) <= allowed
    assert info["collision_pairs"] == []
    assert info["reward_components"]["red_collision_death_penalty"] == 0.0
    assert info["reward_components"]["blue_collision_death_penalty"] == 0.0


def test_v8_attack_kill_steps_record_simultaneous_unique_target_deaths(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HOMO_V8,
        lambda cfg: cfg["simulation"].update({"max_steps": 1}),
    )
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(0)
    _set(env, "red_0", 0.0, 0.0, psi=0.0)
    _set(env, "red_1", 0.0, 1000.0, psi=0.0)
    _set(env, "red_2", 0.0, -1000.0, psi=0.0)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0)
    _set(env, "blue_1", 500.0, 1000.0, psi=0.0)
    _set(env, "blue_2", 9000.0, -1000.0, psi=0.0)

    _, _, _, truncated, info = env.step(_zero_actions(env))

    assert truncated
    summary = info["episode_summary"]
    assert summary["red_attack_kill_steps"] == [1, 1]
    assert summary["red_first_attack_kill_step"] == 1
    assert summary["red_second_attack_kill_step"] == 1
    assert summary["red_third_attack_kill_step"] is None
    assert summary["red_attack_window_steps"] == 1
    assert summary["red_r41_active_steps"] == 1


def test_v8_multiple_attackers_same_target_count_one_kill_step_and_reset_clears(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HOMO_V8,
        lambda cfg: cfg["simulation"].update({"max_steps": 1}),
    )
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(0)
    _set(env, "red_0", 0.0, 0.0, psi=0.0)
    _set(env, "red_1", 0.0, 100.0, psi=0.0)
    _set(env, "red_2", 0.0, -1000.0, psi=0.0)
    _set(env, "blue_0", 500.0, 0.0, psi=0.0)
    _set(env, "blue_1", 9000.0, 1000.0, psi=0.0)
    _set(env, "blue_2", 9000.0, -1000.0, psi=0.0)

    _, _, _, _, info = env.step(_zero_actions(env))

    summary = info["episode_summary"]
    assert summary["red_attack_kills"] == 1
    assert summary["red_attack_kill_steps"] == [1]
    env.reset(1)
    assert env._episode_attack_kill_steps == {"red": [], "blue": []}
    assert env._episode_tactical_window_steps["red"]["attack_window"] == 0


def test_vector_env_missing_kill_timing_uses_minus_one(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HOMO_V8,
        lambda cfg: cfg["simulation"].update({"max_steps": 1}),
    )
    vec = LocalCombatVectorEnv3v3(path, 1)
    try:
        vec.reset([{"seed": 0}])
        result = vec.step_rules(np.asarray([[0, 0]], dtype=np.int8))
        assert result.episode_valid[0]
        assert int(result.episode_red_first_attack_kill_step[0]) == -1
        assert int(result.episode_blue_first_attack_kill_step[0]) == -1
        assert result.episode_red_r3_active_steps.dtype == np.int32
    finally:
        vec.close()


def test_positive_collision_distance_still_enables_historical_collision_behavior(tmp_path):
    env = Homogeneous3v3AirCombatEnv(V4)
    env.reset(0)
    _set(env, "red_0", 0.0, 0.0, psi=0.0)
    _set(env, "red_1", 0.0, 0.0, psi=0.0)
    _set(env, "red_2", 0.0, 1000.0, psi=0.0)
    _set(env, "blue_0", 9000.0, 0.0, psi=np.pi)
    _set(env, "blue_1", 9000.0, 1000.0, psi=np.pi)
    _set(env, "blue_2", 9000.0, -1000.0, psi=np.pi)

    _, _, _, _, info = env.step(_zero_actions(env))

    assert ("red_0", "red_1") in info["collision_pairs"]
    assert info["death_causes"]["red_0"] == DEATH_COLLISION_FRIENDLY
    assert info["death_causes"]["red_1"] == DEATH_COLLISION_FRIENDLY


def test_v8_has_68d_observation_and_no_persistent_target_state():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    obs, info = env.reset(0)
    assert OBS_DIM == 68
    assert obs["red_0"].shape == (68,)
    assert "v8_metrics" not in info
    for name in ("_engagement_targets", "_previous_engagement_distances", "_episode_target_switch_count"):
        assert not hasattr(env, name)


def test_heterogeneous_support_has_no_r3_r41_r42_but_can_receive_event_loss():
    env = Homogeneous3v3AirCombatEnv(HETERO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)      # support
    _set(env, "red_1", 0, 1000, psi=0.0)   # combat
    _set(env, "red_2", 0, -1000, alive=False)
    _set(env, "blue_0", 600, 0, psi=0.0)
    _set(env, "blue_1", 600, 1000, psi=0.0)
    _set(env, "blue_2", 5000, 0, alive=False)
    effective = {a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft}
    parts, targets = env._capture_v8_segmented_pre_attack(effective)
    assert targets["red_0"] == "blue_0"
    assert parts["red"]["attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    rewards, rc, _ = env._compute_v8_segmented_rewards({"red": 0, "blue": 0}, {"red_0": DEATH_ATTACK}, parts, targets)
    assert rc["red_attack_death_penalty"] == pytest.approx(-10.0 / 3.0)
    assert rewards["red_0"] == pytest.approx(rc["red_team_total_reward"])


def test_heterogeneous_v8_independent_nearest_control_target_matches_reward_target():
    env = Homogeneous3v3AirCombatEnv(HETERO_V8)
    env.reset(0)
    _set(env, "red_0", -2000, 0, psi=0.0)
    _set(env, "red_1", 0, 0, psi=0.0)
    _set(env, "red_2", 0, 100, psi=0.0)
    _set(env, "blue_0", 5000, 0, psi=np.pi)
    _set(env, "blue_1", 800, 0, psi=np.pi)
    _set(env, "blue_2", 1200, 0, psi=np.pi)
    policy = make_team_rule_policy_3v3(env.config, team="red")
    visible = {a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft}
    red_aircraft = [a for a in env.aircraft if a.team == "red"]
    blue_aircraft = [a for a in env.aircraft if a.team == "blue"]
    _, control_targets = policy.select_actions(red_aircraft, blue_aircraft, visible_enemy_ids_by_own=visible)
    _, reward_targets = env._capture_v8_segmented_pre_attack(visible)
    assert policy.policy_name == "functional_heterogeneous_nearest_pursuit_v8"
    assert control_targets["red_1"] == control_targets["red_2"] == "blue_1"
    assert reward_targets["red_1"] == control_targets["red_1"]
    assert reward_targets["red_2"] == control_targets["red_2"]


def test_old_functional_heterogeneous_v1_one_to_one_is_unchanged(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HETERO_V8,
        lambda cfg: (
            cfg["combat"].update({"reward_mode": "functional_heterogeneous_team_v1"}),
            cfg["blue_rule_policy"].update({"mode": "functional_heterogeneous_team_v1"}),
            cfg["red_rule_policy"].update({"mode": "functional_heterogeneous_team_v1"}),
        ),
    )
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(0)
    _set(env, "red_1", 0, 0, psi=0.0)
    _set(env, "red_2", 0, 100, psi=0.0)
    _set(env, "blue_1", 800, 0, psi=np.pi)
    _set(env, "blue_2", 1200, 0, psi=np.pi)
    policy = make_team_rule_policy_3v3(env.config, team="red")
    visible = {a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft}
    red_aircraft = [a for a in env.aircraft if a.team == "red"]
    blue_aircraft = [a for a in env.aircraft if a.team == "blue"]
    _, targets = policy.select_actions(red_aircraft, blue_aircraft, visible_enemy_ids_by_own=visible)
    assert policy.policy_name == "functional_heterogeneous_team_v1"
    assert len({targets["red_1"], targets["red_2"]}) == 2


def test_local_and_worker_v8_policy_modes_and_step_consistency():
    local = LocalCombatVectorEnv3v3(HETERO_V8, 2)
    worker = SubprocessCombatVectorEnv3v3(HETERO_V8, 2, 2)
    try:
        specs = [{"seed": 10}, {"seed": 11}]
        lo = local.reset(specs)
        wo = worker.reset(specs)
        assert np.allclose(lo[0], wo[0])
        assert set(local.policy_modes()["blue_policy"]) == {"functional_heterogeneous_nearest_pursuit_v8"}
        assert local.policy_modes()["blue_policy"] == worker.policy_modes()["blue_policy"]
        actions = np.zeros((2, 3, 3), dtype=np.float32)
        lr = local.step(actions)
        wr = worker.step(actions)
        assert np.all(np.isfinite(lr.team_rewards))
        assert np.allclose(lr.team_rewards, wr.team_rewards)
    finally:
        local.close()
        worker.close()


def test_evaluation_collision_metrics_feed_best_score():
    records = [
        _summary_record(red_friendly_collision_deaths=1, blue_cross_collision_deaths=2),
        _summary_record(red_cross_collision_deaths=3, blue_friendly_collision_deaths=1),
    ]
    mappo = summarize_mappo(records, elapsed=1.0)
    happo = summarize_happo(records, elapsed=1.0)
    assert mappo["mean_red_collision_deaths"] == pytest.approx(2.0)
    assert mappo["mean_blue_collision_deaths"] == pytest.approx(1.5)
    assert happo["mean_red_collision_deaths"] == pytest.approx(2.0)
    clean = {**mappo, "mean_red_collision_deaths": 0.0}
    assert compute_best_score(clean) > compute_best_score(mappo)
    assert "neg_mean_red_collision_deaths" in compute_best_score_fields(mappo)


def test_v8_best_score_fields_exclude_collision_term():
    record = _summary_record(red_friendly_collision_deaths=3)
    summary = summarize_mappo([record], elapsed=1.0)
    fields = compute_best_score_fields(summary, include_collision=False)
    assert "neg_mean_red_collision_deaths" not in fields
    assert tuple(fields) == (
        "red_complete_elimination_success_rate",
        "red_any_attack_kill_rate",
        "mean_red_attack_kills",
        "mean_red_survivors",
        "neg_mean_red_boundary_deaths",
        "neg_max_steps_rate",
        "neg_mean_episode_length",
    )
    cfg = load_config(HOMO_V8)
    assert best_score_fields_for_config(cfg) == tuple(fields)
    assert compute_best_score_for_config(summary, cfg) == tuple(fields.values())


def test_v8_public_metrics_filter_collision_and_environment_contract():
    cfg = load_config(HOMO_V8)
    metrics = {
        "mean_red_collision_deaths": 0.0,
        "nested": {"blue_collision_deaths": 0, "mean_red_attack_kills": 1.0},
    }
    filtered = filter_public_metrics_for_config(metrics, cfg)
    assert "mean_red_collision_deaths" not in filtered
    assert "blue_collision_deaths" not in filtered["nested"]
    assert filtered["nested"]["mean_red_attack_kills"] == 1.0
    contract = build_main_v8_contract_metadata(cfg)
    assert contract["collision_enabled"] is False
    assert contract["allowed_death_causes"] == ["ATTACK", "BOUNDARY_ALTITUDE", "BOUNDARY_XY"]
    assert contract["observation_dim"] == 68


def test_mappo_happo_and_heterogeneous_happo_trainers_construct(tmp_path):
    trainers = [
        FixedBlue3v3MAPPOTrainer(HOMO_V8, _tiny_train_config(MAPPO_V8, tmp_path)),
        HAPPO3v3Trainer(HOMO_V8, _tiny_train_config(HAPPO_V8, tmp_path)),
        HAPPO3v3Trainer(HETERO_V8, _tiny_train_config(HAPPO_HETERO_V8, tmp_path)),
    ]
    try:
        assert all(t.training_signature()["env_config_sha256"] for t in trainers)
    finally:
        for trainer in trainers:
            trainer.close()


def test_fixed_seed_v8_rollout_reward_state_and_loss_are_finite(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(HOMO_V8, _tiny_train_config(MAPPO_V8, tmp_path))
    try:
        completed = trainer.collect_rollout(remaining=1)
        metrics = trainer.update()
        assert isinstance(completed, list)
        assert np.all(np.isfinite(trainer.current_observations))
        assert np.all(np.isfinite(trainer.current_global_states))
        numeric = [v for v in metrics.values() if isinstance(v, (int, float, np.integer, np.floating))]
        assert np.all(np.isfinite(numeric))
        assert trainer.last_rollout_reward_means["mean_rollout_terminal_reward"] == 0.0
    finally:
        trainer.close()
