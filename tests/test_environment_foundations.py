from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from env.mavuav import (
    BLUE_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from tools.audit_environment_foundations import audit_foundations, write_summary


def test_v34_foundation_contract_is_exact_and_frozen_fields_are_unchanged():
    config = load_environment_config(None)
    assert ENVIRONMENT_VERSION == config["environment_version"] == "heterogeneous_mavuav_4v4_v3_4"
    assert (OBS_DIM, GLOBAL_STATE_DIM) == (100, 117)
    assert config["sensing"] == {"MAV_range": 12000.0, "UAV_range": 8000.0}
    assert config["simulation"] == {"decision_dt": 1.0, "physics_dt": 0.1, "max_decision_steps": 75}
    assert config["combat"] == {"distance": (1000.0, 3000.0), "ata_deg": 30.0, "aa_deg": 90.0, "hold_steps": 3}
    assert config["reward"] == {
        "blue_kill": 50.0, "uav_loss": -10.0, "mav_loss": -100.0,
        "terminal_red_win": 100.0, "terminal_blue_win": -100.0, "terminal_draw": 0.0,
    }


def test_v33_initial_speed_and_rear_formation_avoid_clipping_by_construction():
    config = load_environment_config(None)
    initial = config["scenario"]["initial"]
    assert all(initial[aircraft_id]["speed"] == 275.0 for aircraft_id in RED_IDS + BLUE_IDS)
    assert initial["MAV"]["position"] == [-5000.0, 0.0, 5000.0]
    assert all(initial[aircraft_id]["position"][0] == -4000.0 for aircraft_id in RED_IDS[1:])
    jitter = config["randomization_profiles"]["main"]["speed_jitter"]
    for aircraft_id in RED_IDS + BLUE_IDS:
        kind = "MAV" if aircraft_id == "MAV" else "UAV" if aircraft_id in RED_IDS else "Blue"
        spec = config["aircraft_specs"][kind]
        assert spec["v_min"] < 275.0 - jitter < 275.0 + jitter < spec["v_max"]
    slot_jitter = config["randomization_profiles"]["main"]["slot_xy_jitter"]
    assert initial["MAV"]["position"][0] + slot_jitter < initial["UAV1"]["position"][0] - slot_jitter


def test_config_rejects_speed_randomization_that_can_touch_a_speed_bound():
    config = deepcopy(load_environment_config(None))
    config["randomization_profiles"]["main"]["speed_jitter"] = 25.0
    with pytest.raises(ValueError, match="strictly inside its speed limits"):
        load_environment_config(config)


def test_blue_true_state_nearest_target_can_transition_between_uav_and_mav():
    env = HeterogeneousMAVUAVAirCombatEnv(randomize=False)
    env.reset(seed=7)
    blue = env.entities["Blue1"]
    red = {aircraft_id: env.entities[aircraft_id] for aircraft_id in RED_IDS}
    assert env.blue_policy.select_target(blue, red).aircraft_id == "UAV1"
    red["MAV"].state.x, red["MAV"].state.y = blue.state.x - 10.0, blue.state.y
    assert env.blue_policy.select_target(blue, red).aircraft_id == "MAV"
    for aircraft_id in RED_IDS[1:]:
        red[aircraft_id].state.alive = False
    assert env.blue_policy.select_target(blue, red).aircraft_id == "MAV"
    red["MAV"].state.alive = False
    assert env.blue_policy.select_target(blue, red) is None


def test_foundation_audit_10000_resets_is_reproducible_and_has_expected_invariants():
    first = audit_foundations("main", 10_000, 1000)
    second = audit_foundations("main", 10_000, 1000)
    assert first == second
    assert first["samples"] == 10_000
    assert first["formation"]["P_MAV_behind_all_UAVs"] == 1.0
    assert first["formation"]["minimum_MAV_rear_offset_m"] >= 400.0
    assert first["speed_clipping"]["count"] == 0
    assert first["initial_combat_legality"]["samples_with_any_pair_at_hold_condition"] == 0


def test_foundation_json_is_strictly_finite_and_contains_no_raw_samples(tmp_path):
    summary = audit_foundations("learnability", 20, 2000)
    path = write_summary(tmp_path / "foundation", summary)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "foundation_summary.json"
    assert loaded["samples"] == 20 and "raw_states" not in loaded

    def check(value):
        if isinstance(value, dict):
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)
        elif isinstance(value, float):
            assert math.isfinite(value)

    check(loaded)
