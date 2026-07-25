from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from uav_combat.mappo.trainer import MAPPOTrainer,POLICIES,SCENARIOS

ENV_CONFIG=Path(__file__).parents[1]/"configs/homogeneous_1v1.yaml"

def tiny_config(tmp_path):
    return {"experiment":{"seed":3,"device":"cpu","output_dir":str(tmp_path)},"network":{"hidden_dim":32,"log_std_init":-.5},"training":{"training_mode":"alternating_self_play","total_env_steps":256,"num_envs":2,"rollout_steps":16,"alternating_block_env_steps":64,"ppo_epochs":1,"minibatch_size":32,"gamma":.99,"gae_lambda":.95,"clip_coef":.2,"learning_rate":3e-4,"value_loss_coef":.5,"entropy_coef":.01,"max_grad_norm":.5,"eval_interval_updates":10,"checkpoint_interval_updates":10,"opponent_history_latest_probability":.7},"evaluation":{"episodes":2,"deterministic":True}}

def same(left,right):return all(torch.equal(v,right[k]) for k,v in left.items())
def changed(before,module):return any(not torch.equal(v,module.state_dict()[k]) for k,v in before.items())

def test_policy_actors_start_equal_but_independent(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path))
    assert same(t.policy_a_actor.state_dict(),t.policy_b_actor.state_dict())
    assert all(a is not b and a.data_ptr()!=b.data_ptr() for a,b in zip(t.policy_a_actor.parameters(),t.policy_b_actor.parameters()))
    assert t.policy_a_critic is not t.policy_b_critic
    assert len({id(getattr(t,f"policy_{p}_{kind}_optimizer")) for p in POLICIES for kind in ("actor","critic")})==4

@pytest.mark.parametrize("policy,steps",(("a",0),("b",64)))
def test_block_updates_only_active_policy(tmp_path,policy,steps):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));t.env_steps=steps;t.configure_block_opponent(steps//64,policy,force=True);t.reset_environments()
    before={p:{k:deepcopy(getattr(t,f"policy_{p}_{k}").state_dict()) for k in ("actor","critic")} for p in POLICIES}
    frozen=next(p for p in POLICIES if p!=policy);ao=deepcopy(getattr(t,f"policy_{frozen}_actor_optimizer").state_dict());co=deepcopy(getattr(t,f"policy_{frozen}_critic_optimizer").state_dict())
    t.collect_rollout();metrics=t.update(policy)
    assert changed(before[policy]["actor"],t._actor(policy)) and changed(before[policy]["critic"],t._critic(policy))
    assert same(before[frozen]["actor"],t._actor(frozen).state_dict()) and same(before[frozen]["critic"],t._critic(frozen).state_dict())
    assert ao==getattr(t,f"policy_{frozen}_actor_optimizer").state_dict() and co==getattr(t,f"policy_{frozen}_critic_optimizer").state_dict()
    assert np.isfinite(metrics[f"policy_{policy}_policy_loss"]) and np.isnan(metrics[f"policy_{frozen}_policy_loss"])

def test_active_policy_alternates_at_block_boundaries(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path))
    for steps,policy,index in ((0,"a",0),(63,"a",0),(64,"b",1),(128,"a",2)):
        t.env_steps=steps;assert t.active_policy()==policy and t.block_index()==index

def test_rollout_is_policy_centric_and_contains_both_colors(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));t.configure_block_opponent(0,"a",force=True);t.reset_environments();t.collect_rollout()
    assert t.buffer.observations.shape==(16,2,14) and t.buffer.actions.shape==(16,2,3) and t.buffer.values.shape==(16,2)
    assert set(np.unique(t.buffer.active_teams))=={0,1}

def test_scenario_and_role_scheduler_is_balanced(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));t.configure_block_opponent(0,"a",force=True);t.reset_environments()
    for _ in range(48):t._next_reset(t.envs[0])
    assert abs(t.active_team_counts["red"]-t.active_team_counts["blue"])<=1
    assert max(t.tail_combo_counts.values())-min(t.tail_combo_counts.values())<=1

def test_v6_checkpoint_roundtrip_and_v5_rejection(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));t.configure_block_opponent(0,"a",force=True);t.reset_environments();t.collect_rollout();t.update("a")
    t.quick_best_score=(.1,.2,.3,-.4,-.5,-.6);path=tmp_path/"v6.pt";t.save_checkpoint(path);raw=torch.load(path,weights_only=False)
    required={"policy_a_actor","policy_b_actor","policy_a_critic","policy_b_critic","policy_a_actor_optimizer","policy_b_actor_optimizer","policy_a_actor_history","policy_b_actor_history","training_signature","current_active_teams","quick_best_score","history_selection_counts"}
    assert raw["checkpoint_version"]==6 and required<=raw.keys()
    restored=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));restored.load_checkpoint(path)
    assert restored.env_steps==t.env_steps and restored.quick_best_score==t.quick_best_score
    assert same(t.policy_a_actor.state_dict(),restored.policy_a_actor.state_dict())
    old=tmp_path/"v5.pt";torch.save({"checkpoint_version":5},old)
    with pytest.raises(RuntimeError,match="v5 and earlier"):restored.load_checkpoint(old)

def test_signature_allows_only_excluded_runtime_fields(tmp_path):
    config=tiny_config(tmp_path);t=MAPPOTrainer(ENV_CONFIG,config);path=tmp_path/"v6.pt";t.save_checkpoint(path)
    allowed=deepcopy(config);allowed["training"]["total_env_steps"]=999;allowed["experiment"].update(device="cpu",output_dir="elsewhere")
    MAPPOTrainer(ENV_CONFIG,allowed).load_checkpoint(path)
    bad=deepcopy(config);bad["training"]["gamma"]=.5
    with pytest.raises(RuntimeError,match="signature mismatch"):MAPPOTrainer(ENV_CONFIG,bad).load_checkpoint(path)

def test_history_selection_counters_separate_forced_latest_old(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));t._select_opponent_generation("b")
    assert t.history_selection_counts=={"forced_single_generation":1,"sampled_latest":0,"sampled_old":0}
    t.policy_b_actor_history.extend([deepcopy(t.policy_b_actor_history[0]),deepcopy(t.policy_b_actor_history[0])])
    class R:
        def __init__(self,base):self.base=base;self.values=iter((.1,.9))
        def random(self):return next(self.values)
        def integers(self,*a,**k):return self.base.integers(*a,**k)
        @property
        def bit_generator(self):return self.base.bit_generator
    t.opponent_rng=R(np.random.default_rng(4));t._select_opponent_generation("b");t._select_opponent_generation("b")
    assert t.history_selection_counts["sampled_latest"]==1 and t.history_selection_counts["sampled_old"]==1

def test_behavior_actor_frozen_and_generation_fixed_within_block(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path));info=t.configure_block_opponent(0,"a",force=True);behavior=deepcopy(t.policy_b_behavior_actor.state_dict())
    t.reset_environments();t.collect_rollout();assert t.configure_block_opponent(0,"a")["opponent_generation"]==info["opponent_generation"]
    assert same(behavior,t.policy_b_behavior_actor.state_dict()) and all(not p.requires_grad for p in t.policy_b_behavior_actor.parameters())

def test_four_blocks_append_active_histories(tmp_path):
    t=MAPPOTrainer(ENV_CONFIG,tiny_config(tmp_path))
    while t.env_steps<256:
        block=t.block_index();active=t.active_policy();end=(block+1)*64;t.configure_block_opponent(block,active,force=True);t.reset_environments()
        while t.env_steps<end:t.collect_rollout(end-t.env_steps);t.update(active)
        t.finish_block(active,block)
    assert [r["active_policy"] for r in t.block_history]==["a","b","a","b"]
    assert len(t.policy_a_actor_history)==len(t.policy_b_actor_history)==3

def test_train_cli_output_dir_override(tmp_path):
    from argparse import Namespace
    from scripts.train_mappo import load_config
    args=Namespace(train_config=str(Path(__file__).parents[1]/"configs/mappo_1v1.yaml"),smoke=False,total_env_steps=2_000_000,num_envs=None,seed=None,device="cuda",output_dir=str(tmp_path/"v6"),resume=None,env_config=str(ENV_CONFIG))
    config=load_config(args)
    assert config["training"]["total_env_steps"]==2_000_000
    assert config["experiment"]["device"]=="cuda"
    assert config["experiment"]["output_dir"]==str(tmp_path/"v6")
