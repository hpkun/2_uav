from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from algorithm.happo import HAPPOTrainer, RelationalCentralizedCritic
from algorithm.happo.evaluation import evaluate_actors
from algorithm.happo.networks import IndependentActors
from env.mavuav import GLOBAL_STATE_DIM, RED_IDS, load_environment_config


EXPECTED_ARCHITECTURE = {
    "entity_count": 8,
    "entity_input_dim": 10,
    "entity_embed_dim": 64,
    "attention_heads": 4,
    "context_input_dim": 39,
    "context_embed_dim": 64,
    "value_hidden_dim": 128,
}


def _state(batch: int = 2) -> torch.Tensor:
    state = torch.randn(batch, GLOBAL_STATE_DIM)
    entities = state[:, :80].reshape(batch, 8, 10)
    entities[..., 6] = 1.0
    return state


def _short_env():
    config = deepcopy(load_environment_config(None))
    config["simulation"]["max_decision_steps"] = 2
    return config


def _config(**updates):
    config = {
        "num_envs": 2, "rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 4,
        "hidden_dim": 16, "seed": 17, "device": "cpu", "critic_variant": "relational",
        "actor_variant": "vanilla", "method_variant": "baseline",
    }
    config.update(updates)
    return config


def test_relational_critic_shapes_are_finite_and_parameter_count_is_locked():
    critic = RelationalCentralizedCritic()
    assert critic(_state()).shape == (2,)
    assert critic(_state(1)[0]).shape == ()
    assert torch.isfinite(critic(_state())).all()
    assert critic.architecture() == EXPECTED_ARCHITECTURE
    assert sum(parameter.numel() for parameter in critic.parameters()) == 94017


def test_entity_and_context_parsing_have_exact_boundaries():
    critic = RelationalCentralizedCritic()
    captured = {}
    def capture_entities(_module, args):
        captured["entities"] = args[0].detach().clone()

    def capture_context(_module, args):
        captured["context"] = args[0].detach().clone()

    critic.entity_encoder[0].register_forward_pre_hook(capture_entities)
    critic.context_encoder[0].register_forward_pre_hook(capture_context)
    state = torch.arange(GLOBAL_STATE_DIM, dtype=torch.float32).unsqueeze(0)
    state[:, 6:80:10] = 1.0
    critic(state)
    assert torch.equal(captured["entities"], state[:, :80].reshape(1, 8, 10))
    assert torch.equal(captured["context"], state[:, 80:119])


def test_entity_encoder_is_shared_and_dead_masks_are_finite_and_zero():
    critic = RelationalCentralizedCritic()
    assert isinstance(critic.entity_encoder[0], nn.Linear)
    assert len([module for module in critic.modules() if isinstance(module, nn.Linear) and module.in_features == 10]) == 1
    state = _state(3)
    state[0, 16] = 0.0
    state[1, 6:80:10] = 0.0
    tokens = critic.encode_entities(state)
    assert torch.count_nonzero(tokens[0, 1]) == 0
    assert torch.count_nonzero(tokens[1]) == 0
    assert torch.isfinite(tokens).all() and torch.isfinite(critic(state)).all()


def test_both_entity_and_context_branches_affect_value():
    torch.manual_seed(5)
    critic = RelationalCentralizedCritic()
    base = _state(1)
    context_changed = base.clone(); context_changed[0, 80] += 1.0
    entity_changed = base.clone(); entity_changed[0, 0] += 1.0
    assert not torch.equal(critic(base), critic(context_changed))
    assert not torch.equal(critic(base), critic(entity_changed))


def test_rc_happo_trainer_smoke_update_checkpoint_resume_and_strict_contract(tmp_path):
    trainer = HAPPOTrainer(_short_env(), _config())
    assert isinstance(trainer.actors, IndependentActors)
    assert len(trainer.actors.actors) == len(RED_IDS) == 4
    assert isinstance(trainer.critic, RelationalCentralizedCritic)
    completed = trainer.collect_rollout()
    assert completed
    metrics = trainer.update()
    assert sorted(metrics["agent_update_order"]) == list(range(4))
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    checkpoint = tmp_path / "rc.pt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["critic_variant"] == "relational"
    assert payload["critic_architecture"] == EXPECTED_ARCHITECTURE
    assert payload["critic_parameter_count"] == 94017

    restored = HAPPOTrainer(_short_env(), _config())
    assert restored.load_checkpoint(checkpoint) == trainer.env_steps
    restored.collect_rollout(); restored.update()
    records = evaluate_actors(
        restored.actors, restored.environment_config, episodes=1, blue_target_mode="nearest",
        profile="main", seed=1000, device="cpu",
    )
    assert len(records) == 1 and np.isfinite(records[0]["episode_return"])

    vanilla = HAPPOTrainer(_short_env(), _config(critic_variant="mlp"))
    with pytest.raises(RuntimeError, match="critic variant"):
        vanilla.load_checkpoint(checkpoint)
    vanilla_checkpoint = tmp_path / "vanilla.pt"
    vanilla.save_checkpoint(vanilla_checkpoint)
    with pytest.raises(RuntimeError, match="critic variant"):
        restored.load_checkpoint(vanilla_checkpoint)

    payload["critic_architecture"] = {**EXPECTED_ARCHITECTURE, "attention_heads": 8}
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save(payload, bad_checkpoint)
    with pytest.raises(RuntimeError, match="critic architecture"):
        restored.load_checkpoint(bad_checkpoint)
    trainer.close(); restored.close(); vanilla.close()


def test_recurrent_relational_combination_is_rejected():
    with pytest.raises(ValueError, match="only supported with vanilla actors"):
        HAPPOTrainer(config=_config(actor_variant="recurrent"))
