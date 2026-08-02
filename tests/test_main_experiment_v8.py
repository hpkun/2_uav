
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.config import load_config
from uav_combat.environment_3v3 import OBS_DIM, Homogeneous3v3AirCombatEnv
from uav_combat.happo.trainer_3v3 import HAPPO3v3Trainer
from uav_combat.mappo.trainer_3v3 import FixedBlue3v3MAPPOTrainer, compute_best_score, compute_best_score_fields
from uav_combat.mappo.vector_env_3v3 import LocalCombatVectorEnv3v3, SubprocessCombatVectorEnv3v3
from uav_combat.models import AircraftState
from uav_combat.rewards import paper_segmented_local_reward

ROOT = Path(__file__).parents[1]
HOMO_V8 = ROOT / "configs" / "homogeneous_3v3_main_v8.yaml"
HETERO_V8 = ROOT / "configs" / "heterogeneous_3v3_main_v8.yaml"
MAPPO_V8 = ROOT / "configs" / "mappo_3v3_main_v8.yaml"
HAPPO_V8 = ROOT / "configs" / "happo_3v3_main_v8.yaml"
HAPPO_HETERO_V8 = ROOT / "configs" / "happo_heterogeneous_3v3_main_v8.yaml"
V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
V7 = ROOT / "configs" / "homogeneous_3v3_learnable_v7_paper_segmented.yaml"


def _state(x: float, y: float, psi: float, altitude: float = 3000.0, alive: bool = True) -> AircraftState:
    return AircraftState(x=x, y=y, z=-altitude, v=150.0, theta=0.0, psi=psi, alive=alive)


def _set(env: Homogeneous3v3AirCombatEnv, aid: str, x: float, y: float, psi: float = 0.0,
         altitude: float = 3000.0, alive: bool = True) -> None:
    ac = env._aircraft_by_id(aid)
    ac.state = _state(x, y, psi, altitude, alive)


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
    cfg["experiment"]["output_dir"] = str(tmp_path / "out")
    cfg["training"]["total_env_steps"] = 8
    cfg["training"]["num_envs"] = 1
    cfg["training"]["num_env_workers"] = 1
    cfg["training"]["rollout_steps"] = 1
    cfg["training"]["minibatch_size"] = 1
    cfg["training"]["ppo_epochs"] = 1
    return cfg


def test_v8_configs_are_simplified_and_history_configs_unchanged():
    homo = load_config(HOMO_V8)
    hetero = load_config(HETERO_V8)
    assert homo["combat"]["reward_mode"] == "task_aligned_paper_segmented_team_v8"
    assert hetero["combat"]["reward_mode"] == "task_aligned_heterogeneous_paper_segmented_team_v8"
    assert homo["blue_rule_policy"]["mode"] == "paper_nearest_pursuit_v1"
    removed = {"progress_weight", "support_information_weight", "time_penalty", "dense_reward_min", "complete_elimination_bonus"}
    assert not (removed & set(homo["reward_v8"]))
    assert not (removed & set(hetero["reward_heterogeneous_v8"]))
    assert load_config(V4)["combat"]["reward_mode"] == "paper_coupled_team_v2"
    assert load_config(V7)["combat"]["reward_mode"] == "paper_segmented_team_v4"


def test_v8_has_no_persistent_engagement_target_state_and_observation_is_68():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    obs, info = env.reset(0)
    forbidden = ("_engagement_targets", "_previous_engagement_distances", "_episode_target_switch_count")
    assert all(not hasattr(env, name) for name in forbidden)
    assert obs["red_0"].shape == (OBS_DIM,)
    assert OBS_DIM == 68
    assert "v8_metrics" not in info


def test_nearest_target_recomputed_each_step_and_tie_breaks_by_id():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0)
    _set(env, "blue_1", 1000, 0)
    _set(env, "blue_0", -1000, 0)
    _set(env, "blue_2", 3000, 0)
    assert env._v8_nearest_reward_target(env._aircraft_by_id("red_0")).aircraft_id == "blue_0"
    env._aircraft_by_id("blue_0").state.alive = False
    assert env._v8_nearest_reward_target(env._aircraft_by_id("red_0")).aircraft_id == "blue_1"


def test_nearest_target_attack_causality_does_not_attack_other_attackable_enemy():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)
    _set(env, "blue_0", 80, 0, psi=0.0)       # nearest, below attack_distance_min
    _set(env, "blue_1", 600, 0, psi=0.0)      # attackable geometry, but not nearest
    _set(env, "blue_2", 5000, 0, alive=False)
    _set(env, "red_1", -5000, 0, alive=False)
    _set(env, "red_2", -6000, 0, alive=False)
    _, _, _, _, info = env.step(_zero_actions(env))
    assert info["reward_targets"]["red_0"] == "blue_0"
    assert info["attacks"]["red_0"] is None


def test_paper_segmented_r3_and_r41_tiers_use_existing_helper():
    cfg = _paper_cfg()
    attack_min, attack_max = 100.0, 1000.0
    r3 = paper_segmented_local_reward(_state(0, 0, 0), _state(1500, 0, 0), attack_min, attack_max, cfg)
    assert r3["guide"] == pytest.approx(0.001)
    coarse = paper_segmented_local_reward(
        _state(0, 0, 0), _state(600 * np.cos(np.deg2rad(25)), 600 * np.sin(np.deg2rad(25)), np.deg2rad(25)), attack_min, attack_max, cfg)
    medium = paper_segmented_local_reward(
        _state(0, 0, 0), _state(600 * np.cos(np.deg2rad(10)), 600 * np.sin(np.deg2rad(10)), np.deg2rad(10)), attack_min, attack_max, cfg)
    fine = paper_segmented_local_reward(_state(0, 0, 0), _state(600, 0, 0), attack_min, attack_max, cfg)
    assert coarse["attack_advantage"] == pytest.approx(0.01)
    assert medium["attack_advantage"] == pytest.approx(0.02)
    assert fine["attack_advantage"] == pytest.approx(0.10)


def test_paper_segmented_r42_tiers_use_existing_helper():
    cfg = _paper_cfg()
    attack_min, attack_max = 100.0, 1000.0
    coarse = paper_segmented_local_reward(_state(0, 0, 0), _state(-600, 0, np.deg2rad(25)), attack_min, attack_max, cfg)
    medium = paper_segmented_local_reward(_state(0, 0, 0), _state(-600, 0, np.deg2rad(10)), attack_min, attack_max, cfg)
    fine = paper_segmented_local_reward(_state(0, 0, 0), _state(-600, 0, 0), attack_min, attack_max, cfg)
    assert coarse["threat"] == pytest.approx(-0.015)
    assert medium["threat"] == pytest.approx(-0.025)
    assert fine["threat"] == pytest.approx(-0.150)


def test_v8_team_reward_is_local_paper_sum_divided_by_fixed_three():
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)
    _set(env, "blue_0", 600, 0, psi=0.0)
    for aid in ("red_1", "red_2", "blue_1", "blue_2"):
        env._aircraft_by_id(aid).state.alive = False
    parts, targets = env._capture_v8_segmented_pre_attack({a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft})
    assert targets["red_0"] == "blue_0"
    assert parts["red"]["attack_advantage_reward"] == pytest.approx(0.10 / 3.0)
    rewards, rc, _ = env._compute_v8_segmented_rewards({"red": 0, "blue": 0}, {}, parts, targets)
    assert rc["red_team_total_reward"] == pytest.approx(rc["red_dense_reward"])
    assert rewards["red_0"] == pytest.approx(rc["red_team_total_reward"])


def test_v8_event_rewards_are_paper_scale_and_no_terminal_timeout_bonus(tmp_path):
    env = Homogeneous3v3AirCombatEnv(HOMO_V8)
    env.reset(0)
    parts = {"red": {"approach_reward": 0.0, "attack_advantage_reward": 0.0, "threat_penalty": 0.0, "dense_reward": 0.0},
             "blue": {"approach_reward": 0.0, "attack_advantage_reward": 0.0, "threat_penalty": 0.0, "dense_reward": 0.0}}
    rewards, rc, _ = env._compute_v8_segmented_rewards({"red": 1, "blue": 0}, {"red_0": 5, "red_1": 1}, parts, {})
    assert rc["red_kill_reward"] == pytest.approx(10.0 / 3.0)
    assert rc["red_attack_death_penalty"] == pytest.approx(-10.0 / 3.0)
    assert rc["red_boundary_death_penalty"] == pytest.approx(-10.0 / 3.0)
    assert rc["red_terminal_reward"] == 0.0
    path = _rewrite_config(tmp_path, HOMO_V8, lambda cfg: cfg["simulation"].update({"max_steps": 1}))
    env2 = Homogeneous3v3AirCombatEnv(path)
    env2.reset(1)
    _, _, terminated, truncated, info = env2.step(_zero_actions(env2))
    assert not terminated and truncated
    assert info["reward_components"]["red_terminal_reward"] == 0.0


def test_heterogeneous_support_has_capability_not_reward_shaping():
    env = Homogeneous3v3AirCombatEnv(HETERO_V8)
    env.reset(0)
    _set(env, "red_0", 0, 0, psi=0.0)      # support
    _set(env, "red_1", 0, 1000, psi=0.0)   # combat
    _set(env, "red_2", 0, -1000, alive=False)
    _set(env, "blue_0", 600, 0, psi=0.0)
    _set(env, "blue_1", 600, 1000, psi=0.0)
    _set(env, "blue_2", 5000, 0, alive=False)
    parts, targets = env._capture_v8_segmented_pre_attack({a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft})
    assert targets["red_0"] == "blue_0"
    assert "red_support_information_reward" not in env.step(_zero_actions(env))[4]["reward_components"]
    assert parts["red"]["attack_advantage_reward"] == pytest.approx(0.10 / 3.0)


def test_heterogeneous_combat_selects_nearest_effective_visible_enemy(tmp_path):
    path = _rewrite_config(tmp_path, HETERO_V8, lambda cfg: cfg["heterogeneous"]["sensor_range"].update({"combat": 500.0, "support": 6000.0}))
    env = Homogeneous3v3AirCombatEnv(path)
    env.reset(0)
    _set(env, "red_0", 0, 0)
    _set(env, "red_1", 0, 0)
    _set(env, "blue_0", 3500, 0)
    _set(env, "blue_1", 450, 0, alive=True)
    target = env._v8_nearest_reward_target(env._aircraft_by_id("red_1"), {a.aircraft_id: env._effective_visible_enemy_ids(a) for a in env.aircraft})
    assert target.aircraft_id == "blue_1"


def test_best_score_simplified_lexicographic_order():
    complete = {"red_complete_elimination_success_rate": 1.0, "mean_red_attack_kills": 0.0}
    kills = {"red_complete_elimination_success_rate": 0.0, "red_any_attack_kill_rate": 1.0, "mean_red_attack_kills": 1.0}
    no_kills = {"red_complete_elimination_success_rate": 0.0, "red_any_attack_kill_rate": 0.0, "mean_red_attack_kills": 2.0}
    assert compute_best_score(complete) > compute_best_score(kills) > compute_best_score(no_kills)
    assert list(compute_best_score_fields(kills)) == [
        "red_complete_elimination_success_rate", "red_any_attack_kill_rate", "mean_red_attack_kills",
        "mean_red_survivors", "neg_mean_red_boundary_deaths", "neg_mean_red_collision_deaths",
        "neg_max_steps_rate", "neg_mean_episode_length",
    ]


def test_local_and_worker_v8_policy_and_step_consistency():
    local = LocalCombatVectorEnv3v3(HOMO_V8, 2)
    worker = SubprocessCombatVectorEnv3v3(HOMO_V8, 2, 2)
    try:
        specs = [{"seed": 10}, {"seed": 11}]
        lo = local.reset(specs)
        wo = worker.reset(specs)
        assert np.allclose(lo[0], wo[0])
        assert set(local.policy_modes()["blue_policy"]) == {"paper_nearest_pursuit_v1"}
        assert local.policy_modes()["blue_policy"] == worker.policy_modes()["blue_policy"]
        actions = np.zeros((2, 3, 3), dtype=np.float32)
        lr = local.step(actions)
        wr = worker.step(actions)
        assert np.all(np.isfinite(lr.team_rewards))
        assert np.allclose(lr.team_rewards, wr.team_rewards)
    finally:
        local.close(); worker.close()


def test_mappo_happo_and_heterogeneous_happo_trainers_construct(tmp_path):
    trainers = [
        FixedBlue3v3MAPPOTrainer(HOMO_V8, _tiny_train_config(MAPPO_V8, tmp_path)),
        HAPPO3v3Trainer(HOMO_V8, _tiny_train_config(HAPPO_V8, tmp_path)),
        HAPPO3v3Trainer(HETERO_V8, _tiny_train_config(HAPPO_HETERO_V8, tmp_path)),
    ]
    try:
        assert all(t.training_signature()["env_config_sha256"] for t in trainers)
    finally:
        for t in trainers:
            t.close()


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
