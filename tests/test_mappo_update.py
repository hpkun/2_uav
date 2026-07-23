from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_combat.mappo.trainer import MAPPOTrainer

CONFIG=Path(__file__).parents[1]/"configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {"experiment":{"seed":3,"device":"cpu","output_dir":str(tmp_path)},"network":{"hidden_dim":32,"log_std_init":-0.5},"training":{"training_mode":"paper_staged","total_env_steps":256,"num_envs":2,"rollout_steps":32,"straight_tail_chase_env_steps":64,"pursuit_tail_chase_env_steps":128,"fixed_training_env_steps":256,"fixed_opponent_gate_win_rate":0.7,"ppo_epochs":1,"minibatch_size":64,"gamma":0.99,"gae_lambda":0.95,"clip_coef":0.2,"learning_rate":3e-4,"value_loss_coef":0.5,"entropy_coef":0.01,"max_grad_norm":0.5,"eval_interval_updates":10,"checkpoint_interval_updates":10},"evaluation":{"episodes":2,"deterministic":True}}


def changed(before,module): return any(not torch.equal(value,module.state_dict()[key]) for key,value in before.items())


def test_fixed_training_updates_red_only_and_checkpoint_roundtrip(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); red_before=deepcopy(trainer.red_actor.state_dict()); blue_before=deepcopy(trainer.blue_actor.state_dict()); blue_optimizer=deepcopy(trainer.blue_actor_optimizer.state_dict()); trainer.collect_rollout(); metrics=trainer.update()
    assert changed(red_before,trainer.red_actor) and not changed(blue_before,trainer.blue_actor); assert blue_optimizer==trainer.blue_actor_optimizer.state_dict(); assert np.isfinite(metrics["red_policy_loss"]) and np.isnan(metrics["blue_policy_loss"])
    observation=torch.zeros(1,14); path=tmp_path/"v3.pt"; trainer.save_checkpoint(path); restored=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); restored.load_checkpoint(path)
    assert torch.equal(trainer.red_actor.deterministic_action(observation),restored.red_actor.deterministic_action(observation))


def test_three_phase_boundaries_and_fixed_opponent_modes(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); assert trainer.phase()=="straight_tail_chase" and trainer.fixed_opponent_mode()=="zero" and trainer._reset_args()==("tail_chase","red")
    trainer.env_steps=64; assert trainer.phase()=="pursuit_tail_chase" and trainer.fixed_opponent_mode()=="pursuit" and trainer._reset_args()==("tail_chase","red")
    trainer.env_steps=128; assert trainer.phase()=="pursuit_all_scenarios" and trainer.fixed_opponent_mode()=="pursuit" and trainer._reset_args()==(None,None)
    trainer.env_steps=256; assert trainer.phase()=="fixed_gate" and trainer.fixed_opponent_mode()=="pursuit"


def test_stage_a_forces_red_rear_and_blue_zero_actions(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); trainer.collect_rollout(64)
    assert np.all(trainer.buffer.actions[:,:,1]==0)
    for env in trainer.envs:
        red,blue=env.aircraft; assert np.dot(red.state.velocity_vector()[:2],blue.state.as_array()[:2]-red.state.as_array()[:2])>0


def test_exact_partial_rollout_and_inactive_critic_head(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); trainer.collect_rollout(32); assert trainer.env_steps==32 and trainer.buffer.rollout_steps==16
    final=trainer.critic.network[-1]; before=(final.weight[1].detach().clone(),final.bias[1].detach().clone()); trainer.update(); assert torch.equal(before[0],final.weight[1]) and torch.equal(before[1],final.bias[1])


def test_old_checkpoint_rejected(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); old=tmp_path/"old.pt"; torch.save({"checkpoint_version":2},old)
    with pytest.raises(RuntimeError,match="v2"): trainer.load_checkpoint(old)
