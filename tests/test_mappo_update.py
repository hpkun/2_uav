from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_combat.mappo.trainer import MAPPOTrainer

CONFIG = Path(__file__).parents[1] / "configs/homogeneous_1v1.yaml"


def tiny_config(tmp_path):
    return {"experiment":{"seed":3,"device":"cpu","output_dir":str(tmp_path)},"network":{"hidden_dim":32,"log_std_init":-0.5},"training":{"total_env_steps":128,"num_envs":2,"rollout_steps":64,"curriculum_tail_chase_env_steps":64,"ppo_epochs":1,"minibatch_size":128,"gamma":0.99,"gae_lambda":0.95,"clip_coef":0.2,"learning_rate":3e-4,"value_loss_coef":0.5,"entropy_coef":0.01,"max_grad_norm":0.5,"eval_interval_updates":10,"checkpoint_interval_updates":10},"evaluation":{"episodes":2,"deterministic":True}}


def changed(before, module): return any(not torch.equal(v,module.state_dict()[k]) for k,v in before.items())


def test_two_actors_are_independent_and_smoke_roundtrip(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); assert trainer.red_actor is not trainer.blue_actor
    assert all(a is not b for a,b in zip(trainer.red_actor.parameters(),trainer.blue_actor.parameters()))
    red_before,blue_before,critic_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict()),deepcopy(trainer.critic.state_dict())
    trainer.collect_rollout(); metrics=trainer.update(); assert all(np.isfinite(v) for v in metrics.values())
    assert changed(red_before,trainer.red_actor) and changed(blue_before,trainer.blue_actor) and changed(critic_before,trainer.critic)
    observation=torch.zeros(1,14); expected_red=trainer.red_actor.deterministic_action(observation).detach(); expected_blue=trainer.blue_actor.deterministic_action(observation).detach()
    path=tmp_path/"v2.pt"; trainer.save_checkpoint(path); restored=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); restored.load_checkpoint(path)
    assert torch.equal(expected_red,restored.red_actor.deterministic_action(observation).detach()); assert torch.equal(expected_blue,restored.blue_actor.deterministic_action(observation).detach())


def test_independent_advantages_update_only_matching_actor(tmp_path):
    config=tiny_config(tmp_path); config["training"]["entropy_coef"]=0.0; trainer=MAPPOTrainer(CONFIG,config); trainer.collect_rollout()
    trainer.buffer.advantages[:,:,0]=np.linspace(-1,1,128).reshape(64,2); trainer.buffer.advantages[:,:,1]=0
    red_before,blue_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict())
    trainer._update_actor(trainer.red_actor,trainer.red_actor_optimizer,0,"red"); trainer._update_actor(trainer.blue_actor,trainer.blue_actor_optimizer,1,"blue")
    assert changed(red_before,trainer.red_actor) and not changed(blue_before,trainer.blue_actor)
    trainer=MAPPOTrainer(CONFIG,config); trainer.collect_rollout(); trainer.buffer.advantages[:,:,0]=0; trainer.buffer.advantages[:,:,1]=np.linspace(-1,1,128).reshape(64,2)
    red_before,blue_before=deepcopy(trainer.red_actor.state_dict()),deepcopy(trainer.blue_actor.state_dict()); trainer._update_actor(trainer.red_actor,trainer.red_actor_optimizer,0,"red"); trainer._update_actor(trainer.blue_actor,trainer.blue_actor_optimizer,1,"blue")
    assert not changed(red_before,trainer.red_actor) and changed(blue_before,trainer.blue_actor)


def test_curriculum_phase_and_old_checkpoint_rejected(tmp_path):
    trainer=MAPPOTrainer(CONFIG,tiny_config(tmp_path)); assert trainer.phase()=="tail_chase_curriculum" and trainer._reset_scenario()=="tail_chase"
    trainer.env_steps=64; assert trainer.phase()=="all_scenarios" and trainer._reset_scenario() is None
    old=tmp_path/"old.pt"; torch.save({"actor":{},"critic":{}},old)
    with pytest.raises(RuntimeError,match="旧共享Actor"): trainer.load_checkpoint(old)
