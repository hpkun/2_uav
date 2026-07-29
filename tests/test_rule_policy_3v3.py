from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from uav_combat.config import aircraft_spec, load_config
from uav_combat.mappo.vector_env_3v3 import LocalCombatVectorEnv3v3, SubprocessCombatVectorEnv3v3
from uav_combat.models import Aircraft, AircraftState
from uav_combat.rule_policy_3v3 import (
    GreedyTeamPursuitPolicy3v3,
    NearestTargetPursuitPolicy3v3,
    make_team_rule_policy_3v3,
)


ROOT = Path(__file__).parents[1]
CONFIG_V4 = ROOT / "configs" / "homogeneous_3v3_learnable_v4.yaml"
CONFIG_V5 = ROOT / "configs" / "homogeneous_3v3_learnable_v5_greedy_blue.yaml"


def _aircraft(aid: str, spec, x: float, y: float, z: float = -3000.0,
              psi: float = 0.0, theta: float = 0.0, alive: bool = True) -> Aircraft:
    return Aircraft(aid, aid.split("_")[0], spec, AircraftState(x, y, z, 150.0, theta, psi, alive))


def _config_v4():
    return load_config(CONFIG_V4)


def _spec():
    return aircraft_spec(_config_v4())


def _greedy_policy():
    return make_team_rule_policy_3v3(load_config(CONFIG_V5), team="blue")


def test_default_config_uses_paper_nearest_policy():
    cfg = _config_v4()
    cfg.pop("blue_rule_policy", None)
    policy = make_team_rule_policy_3v3(cfg, team="blue")
    assert isinstance(policy, NearestTargetPursuitPolicy3v3)
    assert policy.policy_name == "paper_nearest_pursuit_v1"


def test_v4_config_still_uses_paper_nearest_policy():
    policy = make_team_rule_policy_3v3(load_config(CONFIG_V4), team="blue")
    assert policy.policy_name == "paper_nearest_pursuit_v1"
    assert isinstance(policy, NearestTargetPursuitPolicy3v3)


def test_v5_config_uses_greedy_blue_and_paper_red():
    cfg = load_config(CONFIG_V5)
    blue = make_team_rule_policy_3v3(cfg, team="blue")
    red = make_team_rule_policy_3v3(cfg, team="red")
    assert isinstance(blue, GreedyTeamPursuitPolicy3v3)
    assert blue.policy_name == "greedy_team_pursuit_v1"
    assert red.policy_name == "paper_nearest_pursuit_v1"


def test_unknown_rule_policy_mode_raises_value_error():
    cfg = deepcopy(_config_v4())
    cfg["blue_rule_policy"] = {"mode": "not_a_policy"}
    with pytest.raises(ValueError, match="unknown blue_rule_policy mode"):
        make_team_rule_policy_3v3(cfg, team="blue")


def test_three_blue_three_red_greedy_assigns_unique_targets():
    spec = _spec()
    blues = [
        _aircraft("blue_0", spec, 0.0, -1000.0),
        _aircraft("blue_1", spec, 0.0, 0.0),
        _aircraft("blue_2", spec, 0.0, 1000.0),
    ]
    reds = [
        _aircraft("red_0", spec, 1000.0, -1000.0),
        _aircraft("red_1", spec, 1000.0, 0.0),
        _aircraft("red_2", spec, 1000.0, 1000.0),
    ]
    _, targets = _greedy_policy().select_actions(blues, reds)
    assert set(targets.values()) == {"red_0", "red_1", "red_2"}
    assert len(set(targets.values())) == 3


def test_three_blue_one_red_all_alive_blue_may_focus_same_target():
    spec = _spec()
    blues = [_aircraft(f"blue_{i}", spec, 0.0, float(i) * 100.0) for i in range(3)]
    reds = [_aircraft("red_0", spec, 1000.0, 0.0)]
    _, targets = _greedy_policy().select_actions(blues, reds)
    assert targets == {"blue_0": "red_0", "blue_1": "red_0", "blue_2": "red_0"}


def test_two_blue_three_red_assigns_two_targets_only():
    spec = _spec()
    blues = [_aircraft("blue_0", spec, 0.0, -500.0), _aircraft("blue_1", spec, 0.0, 500.0)]
    reds = [
        _aircraft("red_0", spec, 1000.0, -500.0),
        _aircraft("red_1", spec, 1000.0, 500.0),
        _aircraft("red_2", spec, 1000.0, 1500.0),
    ]
    _, targets = _greedy_policy().select_actions(blues, reds)
    assert set(targets) == {"blue_0", "blue_1"}
    assert len(set(targets.values())) == 2


def test_attackable_pair_is_prioritized_over_nearer_non_attackable_pair():
    spec = _spec()
    blue = _aircraft("blue_0", spec, 0.0, 0.0, psi=0.0)
    attackable = _aircraft("red_0", spec, 900.0, 0.0, psi=0.0)
    nearer_not_attackable = _aircraft("red_1", spec, 700.0, 450.0, psi=0.0)
    _, targets = _greedy_policy().select_actions([blue], [nearer_not_attackable, attackable])
    assert targets["blue_0"] == "red_0"


def test_switch_penalty_keeps_target_for_tiny_improvement_and_reassigns_when_dead():
    spec = _spec()
    policy = _greedy_policy()
    blue = _aircraft("blue_0", spec, 0.0, 0.0)
    red_0 = _aircraft("red_0", spec, 1000.0, 0.0)
    red_1 = _aircraft("red_1", spec, 1100.0, 0.0)
    _, first_targets = policy.select_actions([blue], [red_0, red_1])
    assert first_targets["blue_0"] == "red_0"

    red_1.state.x = 900.0
    _, second_targets = policy.select_actions([blue], [red_0, red_1])
    assert second_targets["blue_0"] == "red_0"

    red_0.state.alive = False
    _, third_targets = policy.select_actions([blue], [red_0, red_1])
    assert third_targets["blue_0"] == "red_1"


def test_tie_break_is_deterministic_by_aircraft_id():
    spec = _spec()
    policy = _greedy_policy()
    blue = _aircraft("blue_0", spec, 0.0, 0.0)
    reds = [_aircraft("red_1", spec, 1000.0, -100.0), _aircraft("red_0", spec, 1000.0, 100.0)]
    observed = [policy.select_actions([blue], reds)[1]["blue_0"] for _ in range(5)]
    assert observed == ["red_0"] * 5


def test_greedy_actions_are_finite_bounded_and_dead_blue_is_zero():
    spec = _spec()
    blues = [_aircraft("blue_0", spec, 0.0, 0.0), _aircraft("blue_1", spec, 0.0, 100.0, alive=False)]
    reds = [_aircraft("red_0", spec, 1000.0, 0.0)]
    actions, targets = _greedy_policy().select_actions(blues, reds)
    assert actions["blue_0"].shape == (3,)
    assert np.all(np.isfinite(actions["blue_0"]))
    assert np.all(actions["blue_0"] >= -1.0)
    assert np.all(actions["blue_0"] <= 1.0)
    assert np.array_equal(actions["blue_1"], np.zeros(3, dtype=np.float32))
    assert targets["blue_1"] is None


def test_greedy_actions_zero_when_no_alive_enemy():
    spec = _spec()
    blues = [_aircraft("blue_0", spec, 0.0, 0.0)]
    reds = [_aircraft("red_0", spec, 1000.0, 0.0, alive=False)]
    actions, targets = _greedy_policy().select_actions(blues, reds)
    assert np.array_equal(actions["blue_0"], np.zeros(3, dtype=np.float32))
    assert targets["blue_0"] is None


def test_local_vector_env_uses_v5_greedy_blue_policy():
    vec = LocalCombatVectorEnv3v3(CONFIG_V5, 1)
    try:
        modes = vec.policy_modes()
        assert modes["blue"] == ["rate_aligned_v1"]
        assert modes["blue_policy"] == ["greedy_team_pursuit_v1"]
        assert modes["red_policy"] == ["paper_nearest_pursuit_v1"]
        assert modes["blue_action_mapping"] == ["rate_aligned_v1"]
    finally:
        vec.close()


def test_multiprocessing_vector_env_uses_v5_greedy_blue_policy():
    vec = SubprocessCombatVectorEnv3v3(CONFIG_V5, 2, 2)
    try:
        modes = vec.policy_modes()
        assert modes["blue"] == ["rate_aligned_v1", "rate_aligned_v1"]
        assert modes["blue_policy"] == ["greedy_team_pursuit_v1", "greedy_team_pursuit_v1"]
        assert modes["red_policy"] == ["paper_nearest_pursuit_v1", "paper_nearest_pursuit_v1"]
        assert modes["red_action_mapping"] == ["rate_aligned_v1", "rate_aligned_v1"]
    finally:
        vec.close()
