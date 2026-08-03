from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from uav_combat.config import load_config
from uav_combat.environment_4v3 import (
    BLUE_TEAM_SIZE_4V3,
    GS_DIM_4V3,
    OBS_DIM_4V3,
    RED_TEAM_SIZE_4V3,
    FunctionalHeterogeneous4v3AirCombatEnv,
)
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer, best_score_fields_4v3
from uav_combat.mappo.vector_env_4v3 import make_combat_vector_env_4v3
from uav_combat.models import AircraftState
from uav_combat.rule_policy_4v3 import make_rule_policy_4v3
from uav_combat.scenario_4v3 import BLUE_IDS_4V3, RED_COMBAT_IDS_4V3, RED_IDS_4V3


ENV_CONFIG = "configs/heterogeneous_4v3_main_v9.yaml"
TRAIN_CONFIG = "configs/happo_heterogeneous_4v3_main_v9.yaml"


def _env(seed: int = 7) -> FunctionalHeterogeneous4v3AirCombatEnv:
    env = FunctionalHeterogeneous4v3AirCombatEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def _tiny_train_config() -> dict:
    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text(encoding="utf-8"))
    cfg["experiment"]["device"] = "cpu"
    cfg["experiment"]["seed"] = 123
    cfg["training"]["num_envs"] = 2
    cfg["training"]["num_env_workers"] = 0
    cfg["training"]["rollout_steps"] = 4
    cfg["training"]["total_env_steps"] = 8
    cfg["training"]["ppo_epochs"] = 1
    cfg["training"]["minibatch_size"] = 4
    return cfg


def _set_state(env: FunctionalHeterogeneous4v3AirCombatEnv, aid: str, x: float, y: float, psi: float, alive: bool = True) -> None:
    ac = env._by_id(aid)
    ac.state = AircraftState(x, y, -3000.0, 150.0, 0.0, psi, alive)


def test_v9_config_contract_and_reward_mode() -> None:
    cfg = load_config(ENV_CONFIG)
    assert cfg["scenario"]["red_team_size"] == 4
    assert cfg["scenario"]["blue_team_size"] == 3
    assert cfg["combat"]["reward_mode"] == "functional_heterogeneous_4v3_team_v9"
    assert cfg["battlefield"]["collision_distance"] == 0.0


def test_roles_and_team_sizes_are_fixed() -> None:
    env = _env()
    assert [env._by_id(aid).role for aid in RED_IDS_4V3] == ["support", "combat", "combat", "combat"]
    assert [env._by_id(aid).role for aid in BLUE_IDS_4V3] == ["combat", "combat", "combat"]
    assert len(env._team("red")) == RED_TEAM_SIZE_4V3
    assert len(env._team("blue")) == BLUE_TEAM_SIZE_4V3


def test_all_aircraft_share_same_physical_spec() -> None:
    env = _env()
    specs = [env._by_id(aid).spec for aid in RED_IDS_4V3 + BLUE_IDS_4V3]
    assert all(spec == specs[0] for spec in specs)


def test_functional_heterogeneity_sensor_and_attack_permissions() -> None:
    env = _env()
    assert env._by_id("red_0").sensor_range == 6000.0
    assert all(env._by_id(aid).sensor_range == 1800.0 for aid in RED_COMBAT_IDS_4V3)
    assert all(env._by_id(aid).sensor_range == 3000.0 for aid in BLUE_IDS_4V3)
    assert not env._by_id("red_0").can_attack
    assert all(env._by_id(aid).can_attack for aid in RED_COMBAT_IDS_4V3 + BLUE_IDS_4V3)


def test_reset_observation_global_state_and_masks_shapes_are_finite() -> None:
    obs, gs, mask = _env().reset(8)
    assert obs.shape == (7, OBS_DIM_4V3)
    assert gs.shape == (GS_DIM_4V3,)
    assert mask.shape == (7,)
    assert np.isfinite(obs).all()
    assert np.isfinite(gs).all()
    assert np.isfinite(mask).all()


def test_initial_geometry_support_sees_before_red_combat() -> None:
    env = _env(9)
    direct = env._direct_visible_ids()
    assert len(direct["red_0"] & set(BLUE_IDS_4V3)) > 0
    assert all(len(direct[cid] & set(BLUE_IDS_4V3)) == 0 for cid in RED_COMBAT_IDS_4V3)


def test_support_sharing_only_to_red_combat() -> None:
    env = _env(10)
    direct = env._direct_visible_ids()
    effective = env._effective_visible_ids(direct)
    shared = direct["red_0"] & set(BLUE_IDS_4V3)
    assert shared
    assert all(shared <= effective[cid] for cid in RED_COMBAT_IDS_4V3)
    assert all(effective[bid] == direct[bid] for bid in BLUE_IDS_4V3)


def test_support_death_stops_information_sharing() -> None:
    env = _env(11)
    env._by_id("red_0").state.alive = False
    direct = env._direct_visible_ids()
    effective = env._effective_visible_ids(direct)
    assert all(effective[cid] == direct[cid] for cid in RED_COMBAT_IDS_4V3)


def test_source_encoding_values_cover_dead_hidden_shared_direct() -> None:
    env = _env(12)
    direct = {"red_1": {"blue_1"}, "red_0": {"blue_0"}, "blue_0": set()}
    for aid in RED_IDS_4V3 + BLUE_IDS_4V3:
        direct.setdefault(aid, set())
    effective = {aid: set(v) for aid, v in direct.items()}
    effective["red_1"].add("blue_0")
    env._by_id("blue_2").state.alive = False
    assert env._info_source(env._by_id("red_1"), env._by_id("blue_2"), direct, effective) == -1
    assert env._info_source(env._by_id("red_1"), env._by_id("blue_0"), direct, effective) == 1
    assert env._info_source(env._by_id("red_1"), env._by_id("blue_1"), direct, effective) == 2
    assert env._info_source(env._by_id("red_1"), env._by_id("blue_2"), direct, effective) == -1


def test_red_support_has_zero_attack_kill_permission() -> None:
    env = _env(13)
    _set_state(env, "red_0", 0.0, 0.0, 0.0)
    _set_state(env, "blue_0", 500.0, 0.0, np.pi)
    for cid in RED_COMBAT_IDS_4V3:
        env._by_id(cid).state.alive = False
    obs, gs, mask, reward, done, _, info = env.step({"red_0": np.array([1.0, 0.0, 0.0], np.float32)})
    assert env._by_id("blue_0").state.alive
    assert np.isfinite(reward)


def test_red_attack_requires_direct_not_shared_only() -> None:
    env = _env(14)
    _set_state(env, "red_0", 0.0, 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 0.0, 0.0)
    _set_state(env, "blue_0", 700.0, 0.0, np.pi)
    env._by_id("red_1").sensor_range = 50.0
    env._by_id("red_0").sensor_range = 6000.0
    env.step({"red_1": np.zeros(3, np.float32)})
    assert env._by_id("blue_0").state.alive


def test_blue_can_attack_red_support_by_direct_sensing() -> None:
    env = _env(15)
    _set_state(env, "red_0", 0.0, 0.0, np.pi)
    _set_state(env, "blue_0", 700.0, 0.0, np.pi)
    for cid in RED_COMBAT_IDS_4V3:
        env._by_id(cid).state.alive = False
    env.step({})
    assert not env._by_id("red_0").state.alive


def test_support_death_is_not_terminal_while_red_combat_alive() -> None:
    env = _env(16)
    env._by_id("red_0").state.alive = False
    done, _, _ = env._termination()
    assert not done


def test_all_red_combat_dead_is_loss_even_if_support_alive() -> None:
    env = _env(17)
    for cid in RED_COMBAT_IDS_4V3:
        env._by_id(cid).state.alive = False
    assert env._termination() == (True, "blue", "red_all_combat_eliminated")


def test_all_blue_dead_with_red_combat_alive_is_red_win() -> None:
    env = _env(18)
    for bid in BLUE_IDS_4V3:
        env._by_id(bid).state.alive = False
    assert env._termination() == (True, "red", "red_complete_elimination_success")


def test_mutual_combat_elimination_not_red_win() -> None:
    env = _env(19)
    for bid in BLUE_IDS_4V3:
        env._by_id(bid).state.alive = False
    for cid in RED_COMBAT_IDS_4V3:
        env._by_id(cid).state.alive = False
    assert env._termination() == (True, "draw", "mutual_combat_elimination")


def test_reward_components_are_finite_and_dense_is_clipped() -> None:
    env = _env(20)
    *_, reward, _, _, info = env.step({})
    components = info["reward_components"]
    assert np.isfinite(reward)
    assert all(np.isfinite(v) for v in components.values())
    assert -0.03 <= components["total_dense_reward"] <= 0.03


def test_support_metric_records_shared_to_direct_without_erasing_kill_credit() -> None:
    env = _env(21)
    env._last_support_only_shared_step["red_1"]["blue_0"] = 3
    direct = {aid: set() for aid in RED_IDS_4V3 + BLUE_IDS_4V3}
    direct["red_1"].add("blue_0")
    env.step_count = 10
    env._update_support_metrics(direct, {aid: set(v) for aid, v in direct.items()})
    assert env._share_to_direct_delays == [7]
    assert env._last_support_only_shared_step["red_1"]["blue_0"] == 3


def test_rule_policy_is_role_aware_and_actions_bounded() -> None:
    env = _env(22)
    policy = make_rule_policy_4v3(load_config(ENV_CONFIG), "red")
    actions, targets = env.red_rule_actions()
    assert policy.policy_name == "functional_heterogeneous_4v3_nearest_pursuit_v9"
    assert targets["red_0"] is None
    for aid, action in actions.items():
        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_local_vector_env_shapes_and_policy_modes() -> None:
    vec = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=2, num_env_workers=0, seed=30)
    try:
        obs, gs, masks = vec.reset()
        assert obs.shape == (2, 7, OBS_DIM_4V3)
        assert gs.shape == (2, GS_DIM_4V3)
        assert masks.shape == (2, 7)
        modes = vec.policy_modes()
        assert modes["blue"] == ["functional_heterogeneous_4v3_nearest_pursuit_v9"] * 2
    finally:
        vec.close()


def test_multiprocessing_vector_env_shapes_and_policy_modes() -> None:
    vec = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=2, num_env_workers=1, seed=31)
    try:
        obs, gs, masks = vec.reset()
        assert obs.shape == (2, 7, OBS_DIM_4V3)
        assert gs.shape == (2, GS_DIM_4V3)
        assert masks.shape == (2, 7)
        assert vec.policy_modes()["red"] == ["functional_heterogeneous_4v3_nearest_pursuit_v9"] * 2
    finally:
        vec.close()


def test_vector_step_returns_team_reward_for_four_red_agents_contract() -> None:
    vec = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=2, num_env_workers=0, seed=32)
    try:
        result = vec.step(np.zeros((2, 4, 3), dtype=np.float32))
        assert result.team_rewards.shape == (2,)
        assert result.red_reward_components.shape[0] == 2
        assert np.isfinite(result.team_rewards).all()
    finally:
        vec.close()


def test_happo_config_uses_four_actor_slots() -> None:
    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text(encoding="utf-8"))
    assert cfg["training"]["training_mode"] == "fixed_rule_blue_heterogeneous_4v3_happo"
    assert cfg["training"]["team_size"] == 4
    assert cfg["training"]["observation_dims"] == [OBS_DIM_4V3] * 4
    assert cfg["training"]["action_dims"] == [3, 3, 3, 3]


def test_happo_trainer_initializes_four_actors_and_can_collect_update() -> None:
    trainer = HAPPO4v3Trainer(ENV_CONFIG, _tiny_train_config())
    try:
        episodes = trainer.collect_rollout()
        metrics = trainer.update()
        assert trainer.actors.team_size == 4
        assert trainer.env_steps == 8
        assert all(np.isfinite(v) for v in metrics.values())
    finally:
        trainer.close()


def test_happo_checkpoint_roundtrip_preserves_four_actors(tmp_path: Path) -> None:
    cfg = _tiny_train_config()
    trainer = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    ckpt = tmp_path / "happo_4v3.pt"
    try:
        trainer.collect_rollout()
        trainer.update()
        before = [p.detach().clone() for p in trainer.actors.parameters()]
        trainer.save_checkpoint(ckpt)
    finally:
        trainer.close()
    restored = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    try:
        restored.load_checkpoint(ckpt)
        after = [p.detach().clone() for p in restored.actors.parameters()]
        assert len(after) == len(before)
        assert all(torch.allclose(a, b) for a, b in zip(before, after))
        assert restored.update_count == 1
    finally:
        restored.close()


def test_best_score_fields_match_v9_order() -> None:
    assert best_score_fields_4v3() == [
        "red_complete_elimination_success_rate",
        "red_at_least_two_attack_kill_rate",
        "red_any_attack_kill_rate",
        "mean_red_attack_kills",
        "support_assisted_kill_rate",
        "mean_red_combat_survivors",
        "negative_timeout_rate",
        "negative_mean_episode_length",
    ]
