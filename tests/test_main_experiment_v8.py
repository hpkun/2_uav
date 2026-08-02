from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.config import load_config
from uav_combat.environment_3v3 import ALL_IDS, Homogeneous3v3AirCombatEnv
from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.happo.trainer_3v3 import HAPPO3v3Trainer
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer, compute_best_score
from uav_combat.mappo.vector_env_3v3 import LocalCombatVectorEnv3v3, SubprocessCombatVectorEnv3v3
from uav_combat.models import AircraftState
from uav_combat.rewards import continuous_distance_progress, coupled_attack_advantage, soft_boundary_risk


ROOT = Path(__file__).parents[1]
HOMO_V8 = ROOT / "configs" / "homogeneous_3v3_main_v8.yaml"
HETERO_V8 = ROOT / "configs" / "heterogeneous_3v3_main_v8.yaml"
MAPPO_V8 = ROOT / "configs" / "mappo_3v3_main_v8.yaml"
HAPPO_V8 = ROOT / "configs" / "happo_3v3_main_v8.yaml"
HAPPO_HETERO_V8 = ROOT / "configs" / "happo_heterogeneous_3v3_main_v8.yaml"
V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _state(x: float, y: float, altitude: float = 3000.0, psi: float = 0.0) -> AircraftState:
    return AircraftState(x=x, y=y, z=-altitude, v=150.0, theta=0.0, psi=psi, alive=True)


def _set(env: Homogeneous3v3AirCombatEnv, aid: str, x: float, y: float, altitude: float = 3000.0,
         psi: float = 0.0, alive: bool = True) -> None:
    ac = env._aircraft_by_id(aid)
    ac.state.x = float(x)
    ac.state.y = float(y)
    ac.state.z = -float(altitude)
    ac.state.v = 150.0
    ac.state.theta = 0.0
    ac.state.psi = float(psi)
    ac.state.alive = bool(alive)


def _zero_actions(env: Homogeneous3v3AirCombatEnv) -> dict[str, np.ndarray]:
    return {a.aircraft_id: np.zeros(3, dtype=np.float32) for a in env.aircraft if a.state.alive}


def _rewrite_config(tmp_path: Path, base: Path, mutate) -> Path:
    cfg = load_config(base)
    mutate(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_v8_config_is_independent_and_v7_unchanged():
    v8 = load_config(HOMO_V8)
    v7 = load_config(V7)
    assert v8["combat"]["reward_mode"] == "task_aligned_continuous_team_v8"
    assert v8["blue_rule_policy"]["mode"] == "greedy_team_pursuit_v1"
    assert v8["action"]["mapping_mode"] == "rate_aligned_v1"
    assert v7["combat"]["reward_mode"] == "paper_segmented_team_v4"


def test_continuous_progress_sign_and_inside_attack_range():
    assert continuous_distance_progress(800.0, 770.0, 30.0) > 0.0
    assert continuous_distance_progress(770.0, 800.0, 30.0) < 0.0
    assert continuous_distance_progress(500.0, 470.0, 30.0) > 0.0
    assert continuous_distance_progress(None, 470.0, 30.0) == 0.0


def test_attack_geometry_preferred_distance_and_angle_monotonicity():
    own = _state(0.0, 0.0, psi=0.0)
    preferred = _state(600.0, 0.0, psi=0.0)
    far = _state(1800.0, 0.0, psi=0.0)
    near = _state(80.0, 0.0, psi=0.0)
    off_ata = _state(600.0, 600.0, psi=0.0)
    bad_aspect = _state(600.0, 0.0, psi=np.pi)
    cfg = load_config(HOMO_V8)["reward_v8"]
    score_pref = coupled_attack_advantage(own, preferred, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert score_pref > coupled_attack_advantage(own, far, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert score_pref > coupled_attack_advantage(own, near, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert score_pref > coupled_attack_advantage(own, off_ata, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert score_pref > coupled_attack_advantage(own, bad_aspect, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])


def test_reverse_threat_direction_is_not_same_geometry():
    red = _state(0.0, 0.0, psi=0.0)
    blue = _state(-600.0, 0.0, psi=0.0)
    cfg = load_config(HOMO_V8)["reward_v8"]
    attack = coupled_attack_advantage(red, blue, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    threat = coupled_attack_advantage(blue, red, cfg["preferred_distance"], cfg["distance_sigma"], cfg["ata_sigma"], cfg["aa_sigma"])
    assert attack != threat
    assert threat > attack


def test_soft_boundary_risk_separates_safe_altitude_and_xy():
    cfg = load_config(HOMO_V8)
    bf, rv = cfg["battlefield"], cfg["reward_v8"]
    safe = soft_boundary_risk(_state(0.0, 0.0, altitude=3000.0), bf["x_limit"], bf["y_limit"], bf["altitude_min"], bf["altitude_max"], rv["horizontal_soft_ratio"], rv["altitude_soft_margin"])
    xy_edge = soft_boundary_risk(_state(19500.0, 0.0, altitude=3000.0), bf["x_limit"], bf["y_limit"], bf["altitude_min"], bf["altitude_max"], rv["horizontal_soft_ratio"], rv["altitude_soft_margin"])
    alt_edge = soft_boundary_risk(_state(0.0, 0.0, altitude=550.0), bf["x_limit"], bf["y_limit"], bf["altitude_min"], bf["altitude_max"], rv["horizontal_soft_ratio"], rv["altitude_soft_margin"])
    assert safe["total_risk"] == 0.0
    assert xy_edge["xy_risk"] > safe["xy_risk"]
    assert alt_edge["altitude_risk"] > safe["altitude_risk"]


def test_engagement_target_reset_keep_reselect_and_progress_first_switch_zero():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0)
    _set(env, "blue_0", 800, 0, psi=np.pi, alive=True)
    _set(env, "blue_1", 1200, 0, psi=np.pi, alive=True)
    _set(env, "blue_2", 2000, 0, psi=np.pi, alive=False)
    for aid in ("red_1", "red_2"):
        _set(env, aid, -5000, 0, alive=False)
    env._reset_v8_tracking()
    env._initialize_v8_engagement_targets()
    assert env._select_v8_engagement_target(env._aircraft_by_id("red_0")).aircraft_id == "blue_0"
    _set(env, "blue_0", 900, 0, psi=np.pi, alive=True)
    assert env._select_v8_engagement_target(env._aircraft_by_id("red_0")).aircraft_id == "blue_0"
    env._aircraft_by_id("blue_0").state.alive = False
    assert env._select_v8_engagement_target(env._aircraft_by_id("red_0")).aircraft_id == "blue_1"
    assert env._previous_engagement_distances["red_0"] is None
    parts, targets = env._capture_v8_pre_attack_dense(
        {aid: {e for e in ALL_IDS if e.startswith("blue")} for aid in ALL_IDS},
        {"red": (0.0, 0), "blue": (0.0, 0)},
    )
    assert targets["red_0"] == "blue_1"
    assert parts["red"]["approach_reward"] == pytest.approx(0.0)


def test_v8_reward_target_and_attack_target_consistency():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0, alive=True)
    _set(env, "blue_0", 80, 0, psi=np.pi, alive=True)       # engagement target, too near to attack
    _set(env, "blue_1", 600, 0, psi=np.pi, alive=True)      # attackable but not engagement target
    _set(env, "blue_2", 2000, 0, psi=np.pi, alive=False)
    for aid in ("red_1", "red_2"):
        _set(env, aid, -5000, 0, alive=False)
    env._reset_v8_tracking()
    env._initialize_v8_engagement_targets()
    _, _, _, _, info = env.step(_zero_actions(env))
    assert info["reward_targets"]["red_0"] == "blue_0"
    assert info["attacks"]["red_0"] is None


def test_support_cannot_attack_and_information_reward_requires_useful_shared_target(tmp_path):
    path = _rewrite_config(
        tmp_path,
        HETERO_V8,
        lambda cfg: (
            cfg["heterogeneous"]["sensor_range"].update({"combat": 500.0, "support": 6000.0})
        ),
    )
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0, alive=True)
    _set(env, "red_1", 0, 1000, psi=0.0, alive=True)
    _set(env, "red_2", 0, -1000, psi=0.0, alive=False)
    _set(env, "blue_0", 3500, 0, psi=np.pi, alive=True)
    _set(env, "blue_1", 3500, 1000, psi=np.pi, alive=True)
    _set(env, "blue_2", 9000, 0, psi=np.pi, alive=False)
    env._reset_v8_tracking()
    env._initialize_v8_engagement_targets()
    _, rewards, _, _, info = env.step(_zero_actions(env))
    assert info["attacks"]["red_0"] is None
    assert info["reward_components"]["red_support_information_reward"] > 0.0
    assert np.isfinite(rewards["red_0"])


def test_v8_team_reward_equals_component_sum_without_double_counting():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(1)
    _, rewards, _, _, info = env.step(_zero_actions(env))
    rc = info["reward_components"]
    expected = rc["red_dense_reward"] + rc["red_event_reward"] + rc["red_terminal_reward"]
    assert rc["red_team_total_reward"] == pytest.approx(expected)
    assert rewards["red_0"] == pytest.approx(rc["red_team_total_reward"])


def test_v8_timeout_terminal_truth_table(tmp_path):
    path = _rewrite_config(tmp_path, HOMO_V8, lambda cfg: cfg["simulation"].update({"max_steps": 1}))
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(2)
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "max_steps"
    assert info["reward_components"]["red_terminal_reward"] < 0.0
    assert info["reward_components"]["blue_terminal_reward"] > 0.0


def test_attack_window_metrics_are_geometry_based():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0, alive=True)
    _set(env, "blue_0", 600, 0, psi=0.0, alive=True)
    for aid in ("red_1", "red_2", "blue_1", "blue_2"):
        _set(env, aid, 5000, 5000, alive=False)
    env._reset_v8_tracking()
    env._initialize_v8_engagement_targets()
    _, _, terminated, truncated, info = env.step(_zero_actions(env))
    assert info["v8_metrics"]["red_attack_window_agent_steps"] >= 1
    if terminated or truncated:
        assert info["episode_summary"]["red_any_attack_window"] is True


def test_best_score_orders_v8_primary_metrics():
    better = {
        "red_complete_elimination_success_rate": 0.0,
        "red_any_attack_kill_rate": 1.0,
        "mean_red_attack_kills": 1.0,
    }
    worse = {
        "red_complete_elimination_success_rate": 0.0,
        "red_any_attack_kill_rate": 0.0,
        "mean_red_attack_kills": 2.0,
    }
    assert compute_best_score(better) > compute_best_score(worse)


def _tiny_train_config(path: Path, tmp_path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = deepcopy(yaml.safe_load(f))
    cfg["experiment"]["device"] = "cpu"
    cfg["experiment"]["output_dir"] = str(tmp_path / "out")
    cfg["training"]["total_env_steps"] = 8
    cfg["training"]["num_envs"] = 1
    cfg["training"]["num_env_workers"] = 1
    cfg["training"]["rollout_steps"] = 1
    cfg["training"]["minibatch_size"] = 1
    cfg["training"]["ppo_epochs"] = 1
    return cfg


def test_mappo_and_happo_v8_configs_construct_trainers(tmp_path):
    mappo = FixedBlue3v3MAPPOTrainer(HOMO_V8, _tiny_train_config(MAPPO_V8, tmp_path))
    happo = HAPPO3v3Trainer(HOMO_V8, _tiny_train_config(HAPPO_V8, tmp_path))
    hetero_happo = HAPPO3v3Trainer(HETERO_V8, _tiny_train_config(HAPPO_HETERO_V8, tmp_path))
    try:
        assert mappo.training_signature()["env_config_sha256"]
        assert happo.training_signature()["env_config_sha256"]
        assert hetero_happo.training_signature()["env_config_sha256"]
    finally:
        mappo.close()
        happo.close()
        hetero_happo.close()


def test_local_and_worker_v8_policy_and_step_consistency():
    local = LocalCombatVectorEnv3v3(HOMO_V8, 2)
    worker = SubprocessCombatVectorEnv3v3(HOMO_V8, 2, 2)
    try:
        specs = [{"seed": 10}, {"seed": 11}]
        lo = local.reset(specs)
        wo = worker.reset(specs)
        assert np.allclose(lo[0], wo[0])
        assert local.policy_modes()["blue_policy"] == worker.policy_modes()["blue_policy"]
        actions = np.zeros((2, 3, 3), dtype=np.float32)
        lr = local.step(actions)
        wr = worker.step(actions)
        assert np.all(np.isfinite(lr.team_rewards))
        assert np.all(np.isfinite(wr.team_rewards))
        assert np.allclose(lr.team_rewards, wr.team_rewards)
        assert np.all(lr.episode_red_attack_window_agent_steps == wr.episode_red_attack_window_agent_steps)
    finally:
        local.close()
        worker.close()


def test_fixed_seed_v8_rollout_and_loss_are_finite(tmp_path):
    trainer = FixedBlue3v3MAPPOTrainer(HOMO_V8, _tiny_train_config(MAPPO_V8, tmp_path))
    try:
        completed = trainer.collect_rollout(remaining=1)
        metrics = trainer.update()
        assert isinstance(completed, list)
        numeric = [v for v in metrics.values() if isinstance(v, (int, float, np.integer, np.floating))]
        assert np.all(np.isfinite(numeric))
        assert trainer.last_rollout_reward_means
    finally:
        trainer.close()
