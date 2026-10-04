from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
import torch

from algorithm.happo.trainer import HAPPOTrainer, _apply_tacm_temporal_update
from env.mavuav import load_environment_config
from tools.analyze_tacm_ablation_final import analyze
from tools.preflight_tacm_ablation import validate_protocol
from tools.run_tacm_ablation_training import build_plan
from tools.tacm_ablation_protocol import (
    CHECKPOINT_INTERVAL, EVAL_INTERVAL, EVALUATION_ACTION_SEED_START,
    EVALUATION_ENV_SEED_START, EVALUATION_EPISODES, LOG_INTERVAL, METHODS,
    NUM_ENVS, PAPER_SEEDS, TOTAL_STEPS, evaluation_command, load_method_training,
)


ROOT = Path(__file__).resolve().parents[1]


def short_v310():
    config = deepcopy(load_environment_config(ROOT / "configs" / "env_v310.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def probe_config(method: str) -> dict:
    config = load_method_training(method)
    config.update({"device": "cpu", "num_envs": 1, "rollout_steps": 2,
                   "ppo_epochs": 1, "minibatch_size": 2, "seed": 17,
                   "randomization_curriculum_enabled": False})
    return config


@pytest.mark.parametrize(("coefficient", "expected_step"), ((0.0, False), (0.01, True)))
def test_temporal_optimizer_update_is_strictly_controlled_by_positive_coefficient(
    coefficient, expected_step,
):
    trainer = HAPPOTrainer(short_v310(), probe_config("full"))
    try:
        actor = trainer.actors.actors[1]
        optimizer = trainer.actor_optimizers[1]
        # Establish Adam momentum/state as a preceding PPO/context update would.
        optimizer.zero_grad(); actor.network.router.weight.sum().backward(); optimizer.step()
        before = {name: value.detach().clone() for name, value in actor.network.router.named_parameters()}
        changed = _apply_tacm_temporal_update(
            actor, optimizer, actor.network.router.weight.sum(), 3,
            coefficient, trainer.config["max_grad_norm"],
        )
        assert changed is expected_step
        equal = all(torch.equal(before[name], value.detach())
                    for name, value in actor.network.router.named_parameters())
        assert equal is (not expected_step)
    finally:
        trainer.close()


def test_no_temporal_and_full_have_identical_architecture_and_initialization():
    no_temporal = HAPPOTrainer(short_v310(), probe_config("no_temporal"))
    full = HAPPOTrainer(short_v310(), probe_config("full"))
    try:
        assert no_temporal.actor_parameter_counts == full.actor_parameter_counts
        assert no_temporal.actors.state_dict().keys() == full.actors.state_dict().keys()
        for key, value in no_temporal.actors.state_dict().items():
            assert torch.equal(value, full.actors.state_dict()[key]), key
        assert no_temporal.mav_role_critic.architecture() == full.mav_role_critic.architecture()
        assert no_temporal.uav_role_critic.architecture() == full.uav_role_critic.architecture()
        assert no_temporal.critic_architecture == full.critic_architecture
        assert no_temporal.config["tacm_temporal_coef"] == 0.0
        assert full.config["tacm_temporal_coef"] == 0.01
    finally:
        no_temporal.close(); full.close()


def test_no_mode_reuses_rgaa_and_happo_has_no_role_or_mode_modules():
    no_mode = HAPPOTrainer(short_v310(), probe_config("no_mode"))
    happo = HAPPOTrainer(short_v310(), probe_config("happo"))
    try:
        assert no_mode.rgaa_enabled and no_mode.role_guided_enabled
        assert not no_mode.dbm_enabled and not no_mode.tacm_enabled
        assert hasattr(no_mode, "mav_role_critic") and hasattr(no_mode, "uav_role_critic")
        assert not any("router" in key or "experts" in key for key in no_mode.actors.state_dict())
        assert not happo.role_guided_enabled and not happo.dbm_enabled and not happo.tacm_enabled
        assert happo.mav_role_critic is None and happo.uav_role_critic is None
        assert not any("router" in key or "experts" in key for key in happo.actors.state_dict())
    finally:
        no_mode.close(); happo.close()


def test_preflight_passes_exact_contract_and_fails_each_common_drift():
    configs = {method: load_method_training(method) for method in METHODS}
    assert validate_protocol(configs, inspect_structures=False)["status"] == "PASS"
    mutations = {
        "actor_learning_rate": 9e-4, "curriculum_steps": 123,
        "hidden_dim": 64, "rollout_steps": 32,
    }
    for field, value in mutations.items():
        changed = deepcopy(configs); changed["full"][field] = value
        report = validate_protocol(changed, inspect_structures=False)
        assert report["status"] == "FAIL"
        assert any(item["field"] == field for item in report["differences"])
    wrong_env = load_environment_config(ROOT / "configs" / "env_v39.yaml")
    report = validate_protocol(configs, inspect_structures=False, environment_config=wrong_env)
    assert report["status"] == "FAIL"
    assert any(item["field"] == "environment_version" for item in report["differences"])


def test_training_dry_plan_has_only_fixed_seeds_and_budget():
    for method in METHODS:
        plan = build_plan(method, "TEST", python="python")
        assert [row["seed"] for row in plan] == list(PAPER_SEEDS)
        assert len({row["seed"] for row in plan}) == 3
        for row in plan:
            command = row["command"]
            assert command[command.index("--steps") + 1] == str(TOTAL_STEPS)
            assert command[command.index("--num-envs") + 1] == str(NUM_ENVS)
            assert command[command.index("--checkpoint-interval") + 1] == str(CHECKPOINT_INTERVAL)
            assert command[command.index("--log-interval") + 1] == str(LOG_INTERVAL)
            assert command[command.index("--eval-interval") + 1] == str(EVAL_INTERVAL)
            assert "--resume" not in command


def test_formal_evaluation_command_is_fixed_stochastic_main_exact_protocol(tmp_path):
    checkpoint = tmp_path / "checkpoint_2000000.pt"
    command = evaluation_command(checkpoint, python="python")
    assert command[command.index("--episodes") + 1] == str(EVALUATION_EPISODES)
    assert command[command.index("--env-seed-start") + 1] == str(EVALUATION_ENV_SEED_START)
    assert command[command.index("--action-seed") + 1] == str(EVALUATION_ACTION_SEED_START)
    assert command[command.index("--action-mode") + 1] == "stochastic"
    assert command[command.index("--profile") + 1] == "main"
    assert str(checkpoint) in command


def test_final_analyzer_uses_fixed_formal_results_and_nearest_late_milestones(tmp_path):
    manifest_dir = tmp_path / "manifests"; manifest_dir.mkdir()
    for method_index, method in enumerate(METHODS):
        runs = []
        for seed_index, seed in enumerate(PAPER_SEEDS):
            run_dir = tmp_path / f"{method}_{seed}"; run_dir.mkdir()
            value = float(method_index + seed_index) / 10.0
            result = {
                "sampled_steps": TOTAL_STEPS, "evaluation_episodes": EVALUATION_EPISODES,
                "evaluation_environment_seed_start": EVALUATION_ENV_SEED_START,
                "effective_action_seed": EVALUATION_ACTION_SEED_START,
                "action_mode": "stochastic", "evaluation_profile": "main",
                "environment_version": "heterogeneous_mavuav_4v4_v3_10",
                "red_win_rate": value, "blue_win_rate": 0.0, "draw_rate": 1.0 - value,
                "mean_episode_return": 100.0 + value, "mean_red_attack_kills": value * 4,
                "mean_blue_attack_kills": 0.0, "MAV_survival_rate": 1.0,
                "mean_UAV_survivors": 3.0, "mean_episode_length": 50.0,
            }
            (run_dir / "evaluation_2000000_stochastic_summary.json").write_text(
                json.dumps({"results": [result]}), encoding="utf-8",
            )
            with (run_dir / "training.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("sampled_steps", "red_win_rate", "mean_episode_return", "mean_red_attack_kills"))
                writer.writeheader()
                for step in (1_598_000, 1_701_000, 1_799_000, 1_902_000, 2_000_000):
                    writer.writerow({"sampled_steps": step, "red_win_rate": value,
                                     "mean_episode_return": 100 + value,
                                     "mean_red_attack_kills": 4 * value})
            runs.append({"seed": seed, "output_folder": str(run_dir)})
        (manifest_dir / f"{method}_latest.json").write_text(json.dumps({
            "status": "complete", "seeds": list(PAPER_SEEDS), "runs": runs,
        }), encoding="utf-8")
    output = tmp_path / "analysis"
    analyze(manifest_dir, output)
    assert {path.name for path in output.iterdir()} == {
        "final_ablation_per_seed.csv", "final_ablation_summary.csv",
        "final_training_stability.csv", "final_ablation_summary.json",
    }
    with (output / "final_training_stability.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert rows[0]["milestone_1600000_actual_steps"] == "1598000"
