from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from algorithm.evaluate_happo import main as evaluate_happo_main, validate_checkpoint_contract
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.tacm_rgaa import tactical_teacher
from algorithm.happo.trainer import HAPPOTrainer
from env.mavuav import (
    BLUE_IDS, CROSS_TEAM_ATTACK_PAIRS, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv, load_environment_config,
)
from env.models import AircraftState
from tools.audit_continuation_horizon import load_tacm_checkpoint, validate_audit_contract


ROOT = Path(__file__).resolve().parents[1]


def config(version: str):
    return deepcopy(load_environment_config(ROOT / "configs" / f"env_{version}.yaml"))


def state(x, y=0.0, *, heading=0.0, alive=True):
    return AircraftState(float(x), float(y), 5000.0, 200.0, 0.0, float(heading), alive)


def isolate_pair(env, attacker: str, target: str):
    for index, aid in enumerate((*RED_IDS, *BLUE_IDS)):
        env.entities[aid].state = state(-50000 + index * 7000, 30000)
    env.entities[attacker].state = state(0.0)
    env.entities[target].state = state(2000.0)
    env._attack_streak = {}


def resolve_three(env):
    events = []; deaths = {}
    for _ in range(3):
        events, deaths = env._resolve_attacks()
    return events, deaths


def test_v310_mav_cannot_accumulate_streak_or_attack():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    env.reset(seed=1); isolate_pair(env, "MAV", "Blue1")
    for _ in range(5):
        events, deaths = env._resolve_attacks()
        assert env._attack_streak[("MAV", "Blue1")] == 0
        assert not any(row["attacker"] == "MAV" for row in events)
        assert "Blue1" not in deaths and env.entities["Blue1"].state.alive


def test_v310_uav_attack_and_blue_to_mav_attack_remain_enabled():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    env.reset(seed=2); isolate_pair(env, "UAV1", "Blue1")
    events, deaths = resolve_three(env)
    assert {"attacker": "UAV1", "target": "Blue1"} in events
    assert deaths["Blue1"] == "red_attack"

    env.reset(seed=2); isolate_pair(env, "Blue1", "MAV")
    events, deaths = resolve_three(env)
    assert {"attacker": "Blue1", "target": "MAV"} in events
    assert deaths["MAV"] == "blue_attack"
    assert env._termination() == (True, False, "blue")


def test_v310_all_blue_killed_by_uav_is_red_win():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    env.reset(seed=3)
    for index, aid in enumerate((*RED_IDS, *BLUE_IDS)):
        env.entities[aid].state = state(-50000 + index * 7000, 30000)
    env.entities["UAV1"].state = state(0.0)
    for index, bid in enumerate(BLUE_IDS):
        env.entities[bid].state = state(2000.0, -100.0 + index * 60.0)
        env._attack_streak[("UAV1", bid)] = 2
    events, deaths = env._resolve_attacks()
    assert all(deaths[bid] == "red_attack" for bid in BLUE_IDS)
    assert all(("UAV1", bid) in {(row["attacker"], row["target"]) for row in events} for bid in BLUE_IDS)
    assert env._termination() == (True, False, "red")


def test_v310_dimensions_and_mav_streak_slots_are_preserved_zero():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    observations, _ = env.reset(seed=4)
    assert all(value.shape == (OBS_DIM,) for value in observations.values())
    assert env.global_state().shape == (GLOBAL_STATE_DIM,)
    isolate_pair(env, "MAV", "Blue1")
    env._attack_streak[("MAV", "Blue1")] = 9
    env._resolve_attacks()
    global_state = env.global_state()
    pair_index = {pair: index for index, pair in enumerate(CROSS_TEAM_ATTACK_PAIRS)}
    assert all(global_state[80 + pair_index[("MAV", bid)]] == 0.0 for bid in BLUE_IDS)


def test_v39_legacy_mav_attack_remains_exactly_enabled():
    cfg = config("v39")
    assert "mav_can_attack" not in cfg["combat"]
    env = HeterogeneousMAVUAVAirCombatEnv(cfg, randomize=False)
    env.reset(seed=5); isolate_pair(env, "MAV", "Blue1")
    events, deaths = resolve_three(env)
    assert {"attacker": "MAV", "target": "Blue1"} in events
    assert deaths["Blue1"] == "red_attack"


def test_v310_mav_reward_is_role_process_plus_shared_team_event_only():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    env.reset(seed=6); isolate_pair(env, "UAV1", "Blue1")
    env._attack_streak[("UAV1", "Blue1")] = 2
    env._apply_boundaries = lambda: {}
    rewards_obs, rewards, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert info["death_causes"].get("Blue1") == "red_attack"
    assert info["event_reward"] == env.config["reward"]["blue_kill"] == 100.0
    assert not (terminated or truncated)
    assert np.isclose(rewards["MAV"], info["mav_process_reward"] + info["event_reward"])
    assert np.isclose(rewards["UAV1"], info["uav1_process_reward"] + info["event_reward"])
    assert "mav_R_gate" not in info and "mav_gate_indicator" not in info
    assert rewards_obs["MAV"].shape == (OBS_DIM,)


def tiny_tacm_config(device="cpu"):
    with (ROOT / "configs/happo_tacm_rgaa_v310.yaml").open(encoding="utf-8") as stream:
        training = yaml.safe_load(stream)["training"]
    training.update({
        "device": device, "num_envs": 1, "rollout_steps": 2,
        "ppo_epochs": 1, "minibatch_size": 2, "seed": 31,
        "randomization_curriculum_enabled": False,
    })
    return training


def test_v310_tacm_checkpoint_evaluator_and_continuation_contract(tmp_path):
    env_config = config("v310")
    trainer = HAPPOTrainer(env_config, tiny_tacm_config())
    checkpoint = tmp_path / "checkpoint_final.pt"
    try:
        trainer.collect_rollout(); metrics = trainer.update()
        assert all(np.isfinite(float(value)) for value in metrics.values() if isinstance(value, (int, float)))
        trainer.save_checkpoint(checkpoint)
        records = evaluate_actors(
            trainer.actors, env_config, 1, "learnability", seed=123,
            device="cpu", deterministic=True,
        )
        assert len(records) == 1
    finally:
        trainer.close()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["mav_direct_attack_capability"] is False
    assert payload["mav_direct_attack_shaping"] == "none"
    assert payload["mav_receives_shared_team_kill_reward"] is True
    validate_checkpoint_contract(payload, env_config)
    assert validate_audit_contract(payload, env_config) == 75
    loaded = load_tacm_checkpoint(checkpoint, "cpu")
    assert loaded["environment_config"]["environment_version"].endswith("v3_10")
    resumed = HAPPOTrainer(env_config, tiny_tacm_config())
    try:
        assert resumed.load_checkpoint(checkpoint) == 2
    finally:
        resumed.close()


def test_tacm_teacher_ignores_mav_to_blue_attack_streak_slots():
    env = HeterogeneousMAVUAVAirCombatEnv(config("v310"), randomize=False)
    env.reset(seed=8)
    state_a = env.global_state()
    state_b = state_a.copy()
    pair_index = {pair: index for index, pair in enumerate(CROSS_TEAM_ATTACK_PAIRS)}
    for bid in BLUE_IDS:
        state_b[80 + pair_index[("MAV", bid)]] = 1.0
    first = tactical_teacher(state_a, env.config)
    second = tactical_teacher(state_b, env.config)
    for field in (
        "probabilities", "confidence", "engagement_scores", "cover_responsibility",
        "cover_assignment", "engagement_target", "threat_target", "mav_threat",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


@torch.no_grad()
def test_v310_cuda_tiny_tacm_rollout_update_save_load_and_standalone_evaluator(
    tmp_path, monkeypatch,
):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA smoke requires a CUDA-capable test host")
    env_config = config("v310")
    trainer_config = tiny_tacm_config("cuda")
    checkpoint = tmp_path / "checkpoint_final.pt"
    trainer = HAPPOTrainer(env_config, trainer_config)
    try:
        trainer.collect_rollout()
        # update needs gradients; temporarily leave the no-grad test context.
        with torch.enable_grad():
            metrics = trainer.update()
        assert all(
            np.isfinite(float(value)) for value in metrics.values()
            if isinstance(value, (int, float))
        )
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()

    resumed = HAPPOTrainer(env_config, trainer_config)
    try:
        assert resumed.load_checkpoint(checkpoint) == 2
    finally:
        resumed.close()

    monkeypatch.setattr(sys, "argv", [
        "evaluate_happo.py", str(checkpoint), "--profile", "learnability",
        "--episodes", "1", "--device", "cuda", "--env-config",
        str(ROOT / "configs/env_v310.yaml"),
    ])
    evaluate_happo_main()
    assert (tmp_path / "evaluation_final.csv").is_file()
    assert (tmp_path / "evaluation_final_summary.json").is_file()
