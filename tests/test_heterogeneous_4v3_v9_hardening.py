from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from uav_combat.environment_4v3 import DEATH_ATTACK, DEATH_BOUNDARY_XY, FunctionalHeterogeneous4v3AirCombatEnv
from uav_combat.happo.evaluation_4v3 import (
    build_evaluation_seed_manifest,
    evaluate_happo_fixed_blue_4v3,
    evaluation_seeds_from_manifest,
    validate_evaluation_seed_manifest,
)
from uav_combat.happo.trainer_4v3 import HAPPO4v3Trainer, summarize_4v3_episodes
from uav_combat.mappo.vector_env_4v3 import make_combat_vector_env_4v3
from uav_combat.models import AircraftState
from uav_combat.rule_policy_4v3 import make_rule_policy_4v3
from uav_combat.scenario_4v3 import BLUE_IDS_4V3, RED_COMBAT_IDS_4V3


ENV_CONFIG = "configs/heterogeneous_4v3_main_v9.yaml"
TRAIN_CONFIG = "configs/happo_heterogeneous_4v3_main_v9.yaml"


def _env(seed: int = 100) -> FunctionalHeterogeneous4v3AirCombatEnv:
    env = FunctionalHeterogeneous4v3AirCombatEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def _tiny_config() -> dict:
    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text(encoding="utf-8"))
    cfg["experiment"].update({"device": "cpu", "seed": 314})
    cfg["training"].update({
        "num_envs": 2,
        "num_env_workers": 0,
        "rollout_steps": 4,
        "total_env_steps": 8,
        "ppo_epochs": 1,
        "minibatch_size": 4,
    })
    return cfg


def test_seed_manifest_is_fixed_non_overlapping_and_hash_valid() -> None:
    manifest = build_evaluation_seed_manifest(
        42, selection_episodes=3, test_episodes=4, selection_seed_offset=50000, test_seed_offset=150000
    )
    validate_evaluation_seed_manifest(manifest)
    assert evaluation_seeds_from_manifest(manifest, "selection") == [50042, 50043, 50044]
    assert evaluation_seeds_from_manifest(manifest, "test", 2) == [150042, 150043]
    assert set(manifest["selection"]["seeds"]).isdisjoint(manifest["test"]["seeds"])


def test_summary_assisted_rate_uses_global_kill_ratio() -> None:
    records = [
        {"episode_length": 10, "red_attack_kills": 3, "support_assisted_kills": 1, "reward_components": {}},
        {"episode_length": 10, "red_attack_kills": 1, "support_assisted_kills": 1, "reward_components": {}},
    ]
    summary = summarize_4v3_episodes(records)
    assert summary["support_assisted_kill_rate"] == pytest.approx(2.0 / 4.0)
    assert summary["support_assisted_episode_rate"] == 1.0


def test_blue_noncombat_elimination_gets_minus_ten_without_success_bonus() -> None:
    env = _env(101)
    for bid in BLUE_IDS_4V3:
        env._by_id(bid).state.alive = False
        env._death_causes[bid] = DEATH_BOUNDARY_XY
    empty_visibility = {ac.aircraft_id: set() for ac in env.aircraft}
    reward, components = env._compute_reward(empty_visibility, empty_visibility, {}, 0)
    assert env._termination() == (True, "draw", "blue_noncombat_elimination")
    assert components["mission_reward"] == -10.0
    assert components["mission_reward"] != 30.0
    assert reward < 0.0


def test_support_active_denominator_and_pair_coverage_are_bounded() -> None:
    env = _env(102)
    direct = {ac.aircraft_id: set() for ac in env.aircraft}
    effective = {ac.aircraft_id: set() for ac in env.aircraft}
    direct["red_0"].update(BLUE_IDS_4V3)
    effective["red_1"].add("blue_0")
    env._record_support_share_metrics(direct, effective)
    assert env._episode_metrics["support_active_steps"] == 1
    assert env._support_only_pair_count(direct, effective) == 1
    assert 0.0 <= env._support_only_pair_ratio(direct, effective) <= 1.0


def test_invalid_formation_direction_has_no_rear_reward_or_unstable_rule_action() -> None:
    env = _env(103)
    for cid in RED_COMBAT_IDS_4V3:
        ac = env._by_id(cid)
        ac.state = AircraftState(ac.state.x, ac.state.y, ac.state.z, 0.0, ac.state.theta, ac.state.psi, True)
    reference = env.formation_reference()
    assert reference["direction_valid"] is False
    assert env._support_position_score()[2] == 0.0
    actions, _ = env.red_rule_actions()
    assert np.allclose(actions["red_0"], 0.0)


def test_grouped_vector_env_exposes_four_workers_and_two_envs_each() -> None:
    vec = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=8, num_env_workers=4, seed=104)
    try:
        assert vec.num_workers == 4
        assert vec.envs_per_worker == 2
        assert len(vec._processes) == 4
        obs, _, _ = vec.reset()
        assert obs.shape[0] == 8
    finally:
        vec.close()


def test_local_and_subprocess_reset_at_match() -> None:
    local = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=1, num_env_workers=0, seed=105)
    worker = make_combat_vector_env_4v3(ENV_CONFIG, num_envs=1, num_env_workers=1, seed=105)
    try:
        left = local.reset_at(0, 7777)
        right = worker.reset_at(0, 7777)
        assert all(np.allclose(a, b) for a, b in zip(left, right))
    finally:
        local.close()
        worker.close()


def test_environment_state_roundtrip_preserves_next_transition() -> None:
    first = _env(106)
    first.step({})
    saved = first.state_dict()
    second = _env(999)
    second.load_state_dict(saved)
    left = first.step({})
    right = second.step({})
    assert np.allclose(left[0], right[0])
    assert np.allclose(left[1], right[1])
    assert left[3] == pytest.approx(right[3])
    assert left[4:6] == right[4:6]


def test_partial_rollout_stops_on_exact_vector_boundary() -> None:
    trainer = HAPPO4v3Trainer(ENV_CONFIG, _tiny_config())
    try:
        trainer.collect_rollout(max_env_steps=6)
        assert trainer.effective_rollout_steps == 3
        assert trainer.env_steps == 6
        metrics = trainer.update()
        assert metrics["effective_rollout_steps"] == 3.0
    finally:
        trainer.close()


def test_fixed_seed_evaluation_records_each_seed_once() -> None:
    cfg = _tiny_config()
    trainer = HAPPO4v3Trainer(ENV_CONFIG, cfg)
    try:
        summary = evaluate_happo_fixed_blue_4v3(
            trainer.actors, ENV_CONFIG, seeds=[60001, 60002], num_envs=2, num_env_workers=0, device="cpu"
        )
    finally:
        trainer.close()
    records = summary["episode_records"]
    assert [record["episode_seed"] for record in records] == [60001, 60002]
    assert len({record["episode_seed"] for record in records}) == 2
