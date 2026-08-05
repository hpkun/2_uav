from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_happo_4v3_post_training import PAIR_FIELDS
from uav_combat.config import load_config
from uav_combat.environment_4v3 import (
    BLUE_IDS_4V3,
    RED_COMBAT_IDS_4V3,
    RED_IDS_4V3,
    FunctionalHeterogeneous4v3AirCombatEnv,
    _angle_score,
    _attack_readiness,
    _f_distance,
)
from uav_combat.geometry import compute_pairwise_geometry
from uav_combat.models import AircraftState


ENV_CONFIG = "configs/heterogeneous_4v3_main_v10_attack_funnel.yaml"
V9_ENV_CONFIG = Path("configs/heterogeneous_4v3_main_v9.yaml")
V9_TRAIN_CONFIG = Path("configs/happo_heterogeneous_4v3_main_v9.yaml")


def _env(seed: int = 700) -> FunctionalHeterogeneous4v3AirCombatEnv:
    env = FunctionalHeterogeneous4v3AirCombatEnv(ENV_CONFIG)
    env.reset(seed)
    return env


def _set_state(env: FunctionalHeterogeneous4v3AirCombatEnv, aid: str, x: float, y: float, psi: float, alive: bool = True) -> None:
    env._by_id(aid).state = AircraftState(x, y, -3000.0, 150.0, 0.0, psi, alive)


def _empty_visibility(env: FunctionalHeterogeneous4v3AirCombatEnv) -> dict[str, set[str]]:
    return {ac.aircraft_id: set() for ac in env.aircraft}


def _attack_env(*, attackers: tuple[str, ...] = ("red_1",), age: int | None = None) -> FunctionalHeterogeneous4v3AirCombatEnv:
    env = _env()
    for ac in env.aircraft:
        ac.state.alive = False
    for index, aid in enumerate(attackers):
        _set_state(env, aid, 0.0, float(index * 20), 0.0)
    _set_state(env, "blue_0", 700.0, 0.0, 0.0)
    env.step_count = 100
    if age is not None:
        share_step = env.step_count + 1 - age
        for aid in attackers:
            env._last_support_only_shared_step[aid]["blue_0"] = share_step
    return env


def test_v10_config_is_explicit_and_v9_yaml_hashes_are_unchanged() -> None:
    v10 = load_config(ENV_CONFIG)
    assert v10["combat"]["reward_contract_version"] == "v10_attack_funnel"
    assert hashlib.sha256(V9_ENV_CONFIG.read_bytes()).hexdigest().upper() == (
        "A32F261B0A14201F221A0615EBBD23711C6A40AE0F10B3E9F1A690910026B4E5"
    )
    assert hashlib.sha256(V9_TRAIN_CONFIG.read_bytes()).hexdigest().upper() == (
        "1E67A5421BDE43956B6E7A182C9CF2029716EA1093791289D40F9BA4254C124C"
    )


def test_v10_angle_score_improves_monotonically_and_stays_continuous() -> None:
    assert _angle_score(np.deg2rad(20), np.deg2rad(60)) > 0.0
    assert _angle_score(np.deg2rad(10), np.deg2rad(60)) > _angle_score(np.deg2rad(20), np.deg2rad(60))
    assert _angle_score(np.deg2rad(20), np.deg2rad(50)) > _angle_score(np.deg2rad(20), np.deg2rad(60))
    assert _angle_score(0.0, np.deg2rad(90)) > _angle_score(0.0, np.deg2rad(90) + 0.1)


def test_v10_readiness_is_additive_and_distance_improves_outside_strict_gate() -> None:
    env = _env()
    attacker = AircraftState(0.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    target = AircraftState(1500.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    readiness = _attack_readiness(attacker, target, 100.0, 1000.0, 2000.0, env._readiness_mode)
    assert readiness == pytest.approx(0.75)
    assert _f_distance(2000.0, 100.0, 1000.0, 2000.0) < _f_distance(1000.0, 100.0, 1000.0, 2000.0)


def test_v10_strict_attack_gate_remains_unchanged() -> None:
    env = _env()
    attacker = AircraftState(0.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    target = AircraftState(1001.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    assert not env.attack_model.can_attack(attacker, target)
    target = AircraftState(700.0, 0.0, -3000.0, 150.0, 0.0, 0.0)
    assert env.attack_model.can_attack(attacker, target)


def test_v10_shared_only_target_cannot_be_attacked() -> None:
    env = _env()
    for ac in env.aircraft:
        ac.state.alive = False
    _set_state(env, "red_0", 0.0, 0.0, 0.0)
    _set_state(env, "red_1", 0.0, 0.0, 0.0)
    _set_state(env, "blue_0", 700.0, 0.0, 0.0)
    env._by_id("red_1").sensor_range = 50.0
    env.step({"red_1": np.zeros(3, dtype=np.float32)})
    assert env._by_id("blue_0").state.alive


def test_dense_clip_records_raw_positive_and_negative_saturation(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env()
    visibility = _empty_visibility(env)
    env.reward_contract["support_dense"]["position_scale"] = 1.0
    monkeypatch.setattr(env, "_support_position_score", lambda: (1.0, 0.0, 0.0))
    env._compute_reward(visibility, visibility, {}, 0)
    assert env._episode_metrics["dense_clip_positive_saturation_steps"] == 1
    assert env._last_raw_dense_reward > 0.03

    monkeypatch.setattr("uav_combat.environment_4v3._boundary_risk", lambda *args, **kwargs: 1.0)
    env.reward_contract["support_dense"]["boundary_scale"] = 1.0
    env._compute_reward(visibility, visibility, {}, 0)
    assert env._episode_metrics["dense_clip_negative_saturation_steps"] == 1
    assert env._episode_metrics["raw_dense_reward_count"] == 2
    assert env._episode_metrics["raw_dense_reward_min"] < -0.03


def test_assisted_window_51_steps_old_fails_and_50_steps_old_succeeds() -> None:
    too_old = _attack_env(age=51)
    too_old.step({"red_1": np.zeros(3, dtype=np.float32)})
    assert too_old._episode_metrics["support_assisted_kills"] == 0

    valid = _attack_env(age=50)
    valid.step({"red_1": np.zeros(3, dtype=np.float32)})
    assert valid._episode_metrics["support_assisted_kills"] == 1
    assert valid._share_to_kill_delays == [50]


def test_assist_is_at_most_once_per_target_and_uses_most_recent_valid_share() -> None:
    env = _attack_env(attackers=("red_1", "red_2"))
    env._last_support_only_shared_step["red_1"]["blue_0"] = 61
    env._last_support_only_shared_step["red_2"]["blue_0"] = 56
    env.step({"red_1": np.zeros(3, dtype=np.float32), "red_2": np.zeros(3, dtype=np.float32)})
    assert env._episode_metrics["support_assisted_kills"] == 1
    assert env._share_to_kill_delays == [40]


def test_dense_summary_and_state_restore_include_compatibility_defaults() -> None:
    env = _env()
    env._last_raw_dense_reward = 0.012
    env._episode_metrics["raw_dense_reward_count"] = 1
    env._episode_metrics["raw_dense_reward_sum"] = 0.012
    env._episode_metrics["raw_dense_reward_min"] = 0.012
    env._episode_metrics["raw_dense_reward_max"] = 0.012
    summary = env._episode_summary(None, "timeout")
    assert summary["dense_clip_saturation_rate"] == 0.0
    assert summary["raw_dense_reward_mean"] == pytest.approx(0.012)
    state = env.state_dict()
    assert state["last_raw_dense_reward"] == pytest.approx(0.012)
    state.pop("last_raw_dense_reward")
    restored = _env(701)
    restored.load_state_dict(state)
    assert restored._last_raw_dense_reward == 0.0


def test_pair_level_audit_contract_names_are_explicit() -> None:
    required = {
        "checkpoint", "episode_seed", "step", "combat_id", "target_id",
        "target_is_current_effective_target", "visibility_source", "distance", "ATA", "AA",
        "distance_score", "angle_score", "geometry_readiness", "distance_gate", "ATA_gate",
        "AA_gate", "attack_window", "combat_action_yaw", "combat_action_pitch",
        "combat_action_speed", "combat_alive", "target_alive",
    }
    assert required <= set(PAIR_FIELDS)
