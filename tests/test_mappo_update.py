from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_combat.mappo.trainer import MAPPOTrainer

CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {"experiment":{"seed":3,"device":"cpu","output_dir":str(tmp_path)},"network":{"hidden_dim":32,"log_std_init":-0.5},"training":{"training_mode":"paper_staged","total_env_steps":128,"num_envs":2,"rollout_steps":64,"fixed_tail_chase_env_steps":64,"fixed_training_env_steps":128,"fixed_opponent_gate_win_rate":0.7,"alternating_block_env_steps":64,"competitive_training_env_steps":128,"ppo_epochs":1,"minibatch_size":128,"gamma":0.99,"gae_lambda":0.95,"clip_coef":0.2,"learning_rate":3e-4,"value_loss_coef":0.5,"entropy_coef":0.01,"max_grad_norm":0.5,"eval_interval_updates":10,"checkpoint_interval_updates":10},"evaluation":{"episodes":2,"deterministic":True}}


def changed(before, module): return any(not torch.equal(v,module.state_dict()[k]) for k,v in before.items())


def test_two_actors_are_independent_and_smoke_roundtrip(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); assert trainer.red_actor is not trainer.blue_actor
    assert all(a is not b for a,b in zip(trainer.red_actor.parameters(),trainer.blue_actor.parameters()))
    red_before,blue_before,critic_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict()),deepcopy(trainer.critic.state_dict()); blue_optimizer_before=deepcopy(trainer.blue_actor_optimizer.state_dict())
    trainer.collect_rollout(); metrics=trainer.update()
    assert all(np.isfinite(metrics[key]) for key in ("red_policy_loss", "red_entropy", "value_loss"))
    assert np.isnan(metrics["blue_policy_loss"]) and np.isnan(metrics["blue_entropy"])
    assert changed(red_before,trainer.red_actor) and not changed(blue_before,trainer.blue_actor) and changed(critic_before,trainer.critic)
    assert blue_optimizer_before==trainer.blue_actor_optimizer.state_dict()
    observation=torch.zeros(1,14); expected_red=trainer.red_actor.deterministic_action(observation).detach(); expected_blue=trainer.blue_actor.deterministic_action(observation).detach()
    path=tmp_path/"v2.pt"; trainer.save_checkpoint(path); restored=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); restored.load_checkpoint(path)
    assert torch.equal(expected_red,restored.red_actor.deterministic_action(observation).detach()); assert torch.equal(expected_blue,restored.blue_actor.deterministic_action(observation).detach())


def test_independent_advantages_update_only_matching_actor(tmp_path):
    config=tiny_config(tmp_path); config["training"]["entropy_coef"]=0.0; trainer=MAPPOTrainer(CONFIG,config); trainer.collect_rollout()
    trainer.buffer.advantages[:,:,0]=np.linspace(-1,1,128).reshape(64,2); trainer.buffer.advantages[:,:,1]=0
    red_before,blue_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict())
    trainer._update_actor(0); trainer._update_actor(1)
    assert changed(red_before,trainer.red_actor) and not changed(blue_before,trainer.blue_actor)
    trainer=MAPPOTrainer(CONFIG,config); trainer.collect_rollout(); trainer.buffer.advantages[:,:,0]=0; trainer.buffer.advantages[:,:,1]=np.linspace(-1,1,128).reshape(64,2)
    red_before,blue_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict()); trainer._update_actor(0); trainer._update_actor(1)
    assert not changed(red_before,trainer.red_actor) and changed(blue_before,trainer.blue_actor)


def test_curriculum_phase_and_old_checkpoint_rejected(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); assert trainer.phase()=="fixed_tail_chase" and trainer._reset_args()==("tail_chase","red")
    trainer.env_steps=64; assert trainer.phase()=="fixed_all_scenarios" and trainer._reset_args()==(None,None)
    old=tmp_path/"old.pt"; torch.save({"checkpoint_version":2,"red_actor":{},"blue_actor":{}},old)
    with pytest.raises(RuntimeError,match="v2"): trainer.load_checkpoint(old)


def test_exact_partial_rollout_and_gate_copy(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); completed=trainer.collect_rollout(64)
    assert trainer.env_steps==64 and trainer.buffer.rollout_steps==32
    trainer.update(); trainer.env_steps=128
    assert trainer.phase()=="fixed_gate" and trainer.active_side()=="red"
    red=deepcopy(trainer.red_actor.state_dict()); trainer.copy_red_to_blue_for_competition()
    assert trainer.phase()=="alternating_competitive" and trainer.active_side()=="red"
    assert all(torch.equal(value,trainer.blue_actor.state_dict()[key]) for key,value in red.items())


def test_alternating_freeze_preserves_inactive_actor_and_critic_head_history(tmp_path):
    config=tiny_config(tmp_path); trainer=MAPPOTrainer(CONFIG,config); trainer.fixed_gate_result=True
    trainer.env_steps=192  # competitive block 1: blue is active
    trainer.collect_rollout(); trainer.update()
    trainer.env_steps=256  # competitive block 2: red is active
    trainer.collect_rollout(); blue_before=deepcopy(trainer.blue_actor.state_dict()); final=trainer.critic.network[-1]
    blue_head_before=(final.weight[1].detach().clone(),final.bias[1].detach().clone())
    weight_state=trainer.critic_optimizer.state[final.weight]["exp_avg"][1].clone(); bias_state=trainer.critic_optimizer.state[final.bias]["exp_avg"][1].clone()
    trainer.update()
    assert not changed(blue_before,trainer.blue_actor)
    assert torch.equal(blue_head_before[0],final.weight[1]) and torch.equal(blue_head_before[1],final.bias[1])
    assert torch.equal(weight_state,trainer.critic_optimizer.state[final.weight]["exp_avg"][1])
    assert torch.equal(bias_state,trainer.critic_optimizer.state[final.bias]["exp_avg"][1])
