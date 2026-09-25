from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.happo.cr_rgaa import CR_RGAA_METHOD, conflict_aware_fusion
from algorithm.happo.lp_cr_rgaa import (
    LP_CR_GATE_VERSION, LP_CR_RGAA_METHOD, loss_preserving_directional_fusion,
)
from algorithm.happo.rgaa import RGAA_METHOD, ROLE_AUX_REWARD_MODE, RoleAdvantageRolloutBuffer
from algorithm.happo.trainer import HAPPOTrainer
from algorithm.train_happo import _algorithm_name
from env.mavuav import RED_IDS, load_environment_config
from tools.audit_role_guided_run import validate_checkpoint_contract


ROOT = Path(__file__).resolve().parents[1]


def short_v39():
    config = deepcopy(load_environment_config("configs/env_v39.yaml"))
    config["simulation"]["max_decision_steps"] = 2
    return config


def trainer_config(**updates):
    config = {
        "method_variant": LP_CR_RGAA_METHOD,
        "actor_variant": "vanilla", "critic_variant": "mlp",
        "environment_profile": "learnability", "device": "cpu", "num_envs": 2,
        "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
        "hidden_dim": 16, "seed": 31, "role_advantage_coef": 0.5,
        "role_aux_reward_mode": ROLE_AUX_REWARD_MODE,
        "cr_rgaa_relational_dim": 16, "cr_rgaa_attention_heads": 4,
        "cr_rgaa_relational_value_coef": 1.0, "cr_rgaa_gate_beta": 2.0,
        "cr_rgaa_lambda_floor_ratio": 0.2,
    }
    config.update(updates)
    return config


def test_directional_gate_preserves_penalty_and_only_suppresses_optimistic_conflict():
    team = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0])
    role = torch.tensor([1.0, -1.0, 1.0, -1.0, 2.0])
    combined, lambdas, consistency = loss_preserving_directional_fusion(
        team, role, role_advantage_coef=0.5, beta=2.0, lambda_floor_ratio=0.2,
    )
    assert torch.equal(consistency, team * role)
    assert torch.allclose(combined, team + lambdas * role)
    assert lambdas[0] == 0.5  # agreement
    assert lambdas[1] == 0.5  # team-positive own-loss penalty is preserved
    assert 0.1 < lambdas[2] < 0.5  # only team-negative/role-positive is suppressed
    assert torch.isclose(
        lambdas[2], lambdas.new_tensor(0.1 + 0.4 * np.exp(-2.0)), atol=1e-7,
    )
    assert lambdas[3] == 0.5
    assert lambdas[4] == 0.5  # neutral


def test_frozen_cr_gate_still_suppresses_both_conflict_directions():
    team = torch.tensor([1.0, -1.0])
    role = torch.tensor([-1.0, 1.0])
    _, cr_lambda, _ = conflict_aware_fusion(
        team, role, role_advantage_coef=0.5, beta=2.0, lambda_floor_ratio=0.2,
    )
    _, lp_lambda, _ = loss_preserving_directional_fusion(
        team, role, role_advantage_coef=0.5, beta=2.0, lambda_floor_ratio=0.2,
    )
    assert cr_lambda[0] < 0.5 and cr_lambda[1] < 0.5
    assert lp_lambda[0] == 0.5 and lp_lambda[1] < 0.5


def test_directional_suppression_is_bounded_and_converges_to_floor():
    team = torch.tensor([-0.01, -1.0, -100.0])
    role = torch.tensor([1.0, 1.0, 100.0])
    _, lambdas, _ = loss_preserving_directional_fusion(
        team, role, role_advantage_coef=0.5, beta=2.0, lambda_floor_ratio=0.2,
    )
    assert torch.all((lambdas >= 0.1) & (lambdas <= 0.5))
    assert lambdas[0] > lambdas[1] > lambdas[2]
    assert torch.isclose(lambdas[2], torch.tensor(0.1), atol=1e-6)


def _force_role_advantages(trainer: HAPPOTrainer) -> None:
    assert isinstance(trainer.buffer, RoleAdvantageRolloutBuffer)
    values = np.asarray([-2.0, 2.0, -1.0, 1.0], dtype=np.float32).reshape(2, 2)
    for agent in range(4):
        trainer.buffer.role_advantages[:, :, agent] = values


def test_lp_fusion_is_the_actual_ppo_advantage_and_diagnostics_are_finite():
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    try:
        trainer.collect_rollout()
        _force_role_advantages(trainer)
        metrics = trainer.update()
        expected = (
            trainer.last_lp_cr_rgaa_team_normalized_advantages
            + trainer.last_lp_cr_rgaa_adaptive_lambdas
            * trainer.last_lp_cr_rgaa_role_normalized_advantages
        )
        assert torch.equal(trainer.last_lp_cr_rgaa_combined_advantages, expected)
        assert torch.equal(trainer.last_lp_cr_rgaa_ppo_advantages, expected)
        required = {
            "cr_role_critic_total_loss", "cr_attention_entropy", "cr_lambda_mean",
            "lp_team_positive_role_negative_rate",
            "lp_team_negative_role_positive_rate",
            "lp_lambda_team_positive_role_negative_mean",
            "lp_lambda_team_negative_role_positive_mean",
            *(f"lp_team_positive_role_negative_rate_{aid}" for aid in RED_IDS),
            *(f"lp_team_negative_role_positive_rate_{aid}" for aid in RED_IDS),
        }
        assert required <= metrics.keys()
        assert all(np.isfinite(metrics[key]) for key in required)
        # Whenever the preserved quadrant is present its conditional mean is exactly lambda0.
        for aid in RED_IDS:
            if metrics[f"lp_team_positive_role_negative_rate_{aid}"] > 0.0:
                assert metrics[f"lp_lambda_team_positive_role_negative_mean_{aid}"] == 0.5
    finally:
        trainer.close()


def test_lp_reuses_cr_initialization_and_relational_critic_without_main_rng_pollution():
    baseline = HAPPOTrainer(short_v39(), trainer_config(method_variant="baseline"))
    baseline_rng = torch.get_rng_state().clone()
    cr = HAPPOTrainer(short_v39(), trainer_config(method_variant=CR_RGAA_METHOD))
    lp = HAPPOTrainer(short_v39(), trainer_config())
    try:
        assert torch.equal(torch.get_rng_state(), baseline_rng)
        assert all(torch.equal(a, b) for a, b in zip(baseline.actors.parameters(), lp.actors.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(baseline.critic.parameters(), lp.critic.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(cr.relational_role_critic.parameters(), lp.relational_role_critic.parameters()))
        assert _algorithm_name("vanilla", LP_CR_RGAA_METHOD, "mlp") == "lp_cr_rgaa_happo"
    finally:
        baseline.close(); cr.close(); lp.close()


def test_lp_checkpoint_metadata_exact_resume_and_cross_method_rejection(tmp_path):
    source = HAPPOTrainer(short_v39(), trainer_config())
    source.train_update()
    checkpoint = tmp_path / "lp.pt"
    source.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "lp_cr_rgaa_happo"
    assert payload["method_variant"] == LP_CR_RGAA_METHOD
    assert payload["credit_gate"]["type"] == LP_CR_GATE_VERSION
    restored = HAPPOTrainer(short_v39(), trainer_config())
    cr = HAPPOTrainer(short_v39(), trainer_config(method_variant=CR_RGAA_METHOD))
    rgaa = HAPPOTrainer(short_v39(), trainer_config(method_variant=RGAA_METHOD))
    try:
        assert restored.load_checkpoint(checkpoint) == source.env_steps
        assert restored.cr_rgaa_rng.bit_generator.state == source.cr_rgaa_rng.bit_generator.state
        assert all(torch.equal(a, b) for a, b in zip(
            source.relational_role_critic.parameters(), restored.relational_role_critic.parameters(),
        ))
        for other in (cr, rgaa):
            with pytest.raises(RuntimeError, match="resume method mismatch"):
                other.load_checkpoint(checkpoint)
        contract = validate_checkpoint_contract(payload, short_v39())
        assert contract["method_variant"] == LP_CR_RGAA_METHOD
    finally:
        source.close(); restored.close(); cr.close(); rgaa.close()


def test_lp_config_differs_from_cr_only_by_method():
    with open("configs/happo_cr_rgaa_v39.yaml", encoding="utf-8") as stream:
        cr = yaml.safe_load(stream)["training"]
    with open("configs/happo_lp_cr_rgaa_v39.yaml", encoding="utf-8") as stream:
        lp = yaml.safe_load(stream)["training"]
    lp["method_variant"] = CR_RGAA_METHOD
    assert lp == cr


def test_standalone_evaluator_loads_lp_actor_only(tmp_path):
    trainer = HAPPOTrainer(short_v39(), trainer_config())
    checkpoint = tmp_path / "lp.pt"
    try:
        trainer.save_checkpoint(checkpoint)
    finally:
        trainer.close()
    result = subprocess.run(
        [sys.executable, "algorithm/evaluate_happo.py", str(checkpoint),
         "--profile", "learnability", "--episodes", "1", "--device", "cpu",
         "--action-mode", "stochastic", "--action-seed", "2000"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "evaluation_lp_stochastic_summary.json").read_text())
    assert summary["algorithm"] == "lp_cr_rgaa_happo"
    assert summary["method_variant"] == LP_CR_RGAA_METHOD


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_lp_tiny_cuda_update_is_finite():
    trainer = HAPPOTrainer(
        short_v39(), trainer_config(device="cuda", num_envs=1, rollout_steps=1, minibatch_size=4),
    )
    try:
        _, metrics = trainer.train_update()
        assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
        assert next(trainer.relational_role_critic.parameters()).is_cuda
    finally:
        trainer.close()
