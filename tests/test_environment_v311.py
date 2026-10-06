"""Isolated v3.11 contracts; no changes to historical environment fixtures."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.evaluate_happo import validate_checkpoint_contract
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import (
    BLUE_IDS, RED_IDS, OBS_DIM, GLOBAL_STATE_DIM, SUPPORT_MAV_ENVIRONMENT_VERSION,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.models import AircraftState

ROOT = Path(__file__).resolve().parents[1]


def config(version="v311"):
    return load_environment_config(ROOT / "configs" / f"env_{version}.yaml")


def nominal_env():
    env = HeterogeneousMAVUAVAirCombatEnv(config(), randomize=False)
    env.reset(seed=1)
    return env


def test_config_only_allowed_differences():
    old, new = config("v310"), config()
    assert new["environment_version"] == SUPPORT_MAV_ENVIRONMENT_VERSION
    assert new["sensing"] == {"MAV_range": 12000.0, "UAV_range": 5000.0}
    assert new["combat"]["mav_can_attack"] is False
    restored = deepcopy(new)
    for key in ("environment_version", "sensing", "scenario"):
        restored[key] = old[key]
    assert restored == old
    assert new["scenario"]["default_profile"] == old["scenario"]["default_profile"]
    for aid in (*RED_IDS, *BLUE_IDS):
        for key in ("speed", "heading_deg"):
            assert new["scenario"]["initial"][aid][key] == old["scenario"]["initial"][aid][key]


def test_nominal_geometry_and_information_support():
    env = nominal_env()
    positions = {"MAV": [-6000, 0, 6700],
                 "UAV1": [-4000, -1200, 6000], "UAV2": [-4000, 0, 6000],
                 "UAV3": [-4000, 1200, 6000], "Blue1": [4000, -1800, 6000],
                 "Blue2": [4000, -600, 6000], "Blue3": [4000, 600, 6000],
                 "Blue4": [4000, 1800, 6000]}
    observations = env._observations()
    assert env.global_state().shape == (GLOBAL_STATE_DIM,)
    for aid, position in positions.items():
        state = env.entities[aid].state
        np.testing.assert_array_equal([state.x, state.y, state.h], position)
        assert state.v == 275.0
        assert np.isclose(np.cos(state.psi), 1.0 if aid in RED_IDS else -1.0)
        assert np.isclose(np.sin(state.psi), 0.0)
    for index, bid in enumerate(BLUE_IDS):
        assert env.direct_visible("MAV", bid) and env.team_visible(bid)
        for aid in RED_IDS[1:]:
            assert not env.direct_visible(aid, bid)
            assert env.datalink_visible(aid, bid)
            block = observations[aid][44 + 14 * index:58 + 14 * index]
            assert observations[aid].shape == (OBS_DIM,)
            assert block[10] == 0 and block[11] == 1
            assert np.any(block[:9] != 0)
    env.entities["UAV1"].state.x = 0.0
    assert env.direct_visible("UAV1", "Blue1")
    assert not env.datalink_visible("UAV1", "Blue1")


@pytest.mark.parametrize("attacker", ["MAV", "UAV1", "UAV2", "UAV3"])
def test_attack_capability(attacker):
    env = nominal_env()
    for index, aid in enumerate((*RED_IDS, *BLUE_IDS)):
        env.entities[aid].state = AircraftState(-50000 + 7000 * index, 30000, 6000, 200, 0, 0, True)
    env.entities[attacker].state = AircraftState(0, 0, 6000, 200, 0, 0, True)
    env.entities["Blue1"].state = AircraftState(2000, 0, 6000, 200, 0, 0, True)
    for step in range(3):
        events, deaths = env._resolve_attacks()
        if step < 2 or attacker == "MAV":
            assert "Blue1" not in deaths
        else:
            assert {"attacker": attacker, "target": "Blue1"} in events
            assert deaths["Blue1"] == "red_attack"
    if attacker == "MAV":
        assert env._attack_streak[("MAV", "Blue1")] == 0


@pytest.mark.parametrize("version", ["v310", "v311"])
def test_termination_unchanged(version):
    env = HeterogeneousMAVUAVAirCombatEnv(config(version), randomize=False)
    env.reset(seed=1)
    for aid in RED_IDS[1:]:
        env.entities[aid].state.alive = False
    assert env._termination() == (False, False, None)
    env.step_count = 75
    assert env._termination() == (False, True, "draw")
    env.entities["MAV"].state.alive = False
    assert env._termination() == (True, False, "blue")
    env.reset(seed=1)
    for bid in BLUE_IDS:
        env.entities[bid].state.alive = False
    env._red_attack_kills = set(BLUE_IDS)
    assert env._termination() == (True, False, "red")


def test_reject_armed_mav():
    cfg = config()
    cfg["combat"]["mav_can_attack"] = True
    with pytest.raises(ValueError, match="mav_can_attack must be false"):
        load_environment_config(cfg)


def test_vanilla_cpu_rollout_update_checkpoint_contract(tmp_path):
    training = {"device": "cpu", "num_envs": 1, "rollout_steps": 2,
                "hidden_dim": 8, "ppo_epochs": 1, "minibatch_size": 2,
                "seed": 1, "actor_variant": "vanilla", "method_variant": "baseline"}
    trainer = HAPPOTrainer(config(), training)
    checkpoint = tmp_path / "tiny.pt"
    try:
        trainer.collect_rollout()
        metrics = trainer.update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)))
        assert trainer.reward_mode == "heterogeneous_role_coupled_gate_v1"
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["sampled_steps"] == 2
    assert payload["environment_version"] == SUPPORT_MAV_ENVIRONMENT_VERSION
    assert payload["mav_direct_attack_capability"] is False
    assert payload["mav_direct_attack_shaping"] == "none"
    assert payload["mav_receives_shared_team_kill_reward"] is True
    validate_checkpoint_contract(payload, config())
    invalid = deepcopy(payload)
    del invalid["mav_direct_attack_capability"]
    with pytest.raises(RuntimeError, match="combat capability"):
        validate_checkpoint_contract(invalid, config())
    resumed = HAPPOTrainer(config(), training)
    try:
        assert resumed.load_checkpoint(checkpoint) == 2
    finally:
        resumed.close()


def test_frozen_role_guided_contract_not_extended():
    with pytest.raises(ValueError, match="requires"):
        HAPPOTrainer(config(), {"device": "cpu", "method_variant": "rgaa"})
