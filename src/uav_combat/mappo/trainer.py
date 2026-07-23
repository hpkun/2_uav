"""Three-stage fixed-opponent PPO training using the existing MAPPO data layout."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch
from torch import nn
from ..environment import HomogeneousAirCombatEnv
from ..rule_policy import PurePursuitPolicy
from .buffer import MAPPOBuffer
from .networks import CentralizedCritic,GaussianActor

AGENT_IDS=("red_0","blue_0")


def resolve_device(requested:str)->torch.device:
    if requested=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device(requested)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
    return device


def new_funnel()->dict[str,Any]:
    return {"minimum_distance":np.inf,"ever_within_4000m":False,"ever_within_attack_distance":False,"ever_satisfy_ata":False,"ever_satisfy_aa":False,"ever_distance_and_ata":False,"ever_distance_and_aa":False,"ever_ata_and_aa":False,"ever_full_attack_envelope":False,"ever_satisfy_attack_envelope":False,"attack_envelope_steps":0,"minimum_distance_violation":np.inf,"minimum_ata_violation":np.inf,"minimum_aa_violation":np.inf,"minimum_combined_violation":np.inf,"kill":False,"boundary":False,"max_steps":False,"collision":False}


def update_funnel(funnel:dict[str,Any],geometry:Any,combat:dict[str,Any],attacked:bool)->None:
    distance_gate=combat["attack_distance_min"]<=geometry.distance<=combat["attack_distance_max"]
    ata_gate=geometry.ata<=combat["attack_ata_max"]; aa_gate=geometry.aa<=combat["attack_aa_max"]
    distance_and_ata=distance_gate and ata_gate; distance_and_aa=distance_gate and aa_gate; ata_and_aa=ata_gate and aa_gate; full=distance_gate and ata_gate and aa_gate
    distance_violation=max(combat["attack_distance_min"]-geometry.distance,geometry.distance-combat["attack_distance_max"],0.0)
    ata_violation=max(geometry.ata-combat["attack_ata_max"],0.0); aa_violation=max(geometry.aa-combat["attack_aa_max"],0.0)
    combined=distance_violation/combat["attack_distance_max"]+ata_violation/np.pi+aa_violation/np.pi
    funnel["minimum_distance"]=min(funnel["minimum_distance"],geometry.distance); funnel["ever_within_4000m"]|=geometry.distance<=4000.0; funnel["ever_within_attack_distance"]|=distance_gate; funnel["ever_satisfy_ata"]|=ata_gate; funnel["ever_satisfy_aa"]|=aa_gate
    funnel["ever_distance_and_ata"]|=distance_and_ata; funnel["ever_distance_and_aa"]|=distance_and_aa; funnel["ever_ata_and_aa"]|=ata_and_aa; funnel["ever_full_attack_envelope"]|=full; funnel["ever_satisfy_attack_envelope"]|=full; funnel["attack_envelope_steps"]+=int(full)
    funnel["minimum_distance_violation"]=min(funnel["minimum_distance_violation"],distance_violation); funnel["minimum_ata_violation"]=min(funnel["minimum_ata_violation"],ata_violation); funnel["minimum_aa_violation"]=min(funnel["minimum_aa_violation"],aa_violation); funnel["minimum_combined_violation"]=min(funnel["minimum_combined_violation"],combined)
    if attacked and not full: raise AssertionError("an attack cannot occur outside the full attack envelope")


def finish_funnel(funnel:dict[str,Any],reason:str|None,result:str)->None:
    funnel["kill"]=reason=="red_kill"; funnel["boundary"]=reason in {"boundary","xy_boundary","altitude_boundary"}; funnel["max_steps"]=reason=="max_steps"; funnel["collision"]=reason=="collision"


def summarize_records(records:list[dict[str,Any]])->dict[str,float|int]:
    count=len(records)
    if not count: raise ValueError("cannot summarize zero episodes")
    rate=lambda key:sum(bool(row["funnel"][key]) for row in records)/count
    envelopes=sum(row["funnel"]["ever_full_attack_envelope"] for row in records); kills=sum(row["funnel"]["kill"] for row in records)
    result={"episodes":count,"wins":sum(row["result"]=="win" for row in records),"losses":sum(row["result"]=="loss" for row in records),"draws":sum(row["result"]=="draw" for row in records)}
    result.update({"win_rate":result["wins"]/count,"loss_rate":result["losses"]/count,"draw_rate":result["draws"]/count,"mean_return":float(np.mean([row["return"] for row in records])),"mean_episode_length":float(np.mean([row["length"] for row in records])),"boundary_rate":rate("boundary"),"max_steps_rate":rate("max_steps"),"collision_rate":rate("collision"),"within_4000_rate":rate("ever_within_4000m"),"attack_distance_entry_rate":rate("ever_within_attack_distance"),"ata_gate_rate":rate("ever_satisfy_ata"),"aa_gate_rate":rate("ever_satisfy_aa"),"distance_and_ata_entry_rate":rate("ever_distance_and_ata"),"distance_and_aa_entry_rate":rate("ever_distance_and_aa"),"ata_and_aa_entry_rate":rate("ever_ata_and_aa"),"full_attack_envelope_entry_rate":rate("ever_full_attack_envelope"),"attack_envelope_entry_rate":rate("ever_full_attack_envelope"),"attack_to_kill_conversion_rate":kills/envelopes if envelopes else 0.0,"kill_rate":kills/count,"mean_minimum_distance_violation":float(np.mean([row["funnel"]["minimum_distance_violation"] for row in records])),"mean_minimum_ata_violation":float(np.mean([row["funnel"]["minimum_ata_violation"] for row in records])),"mean_minimum_aa_violation":float(np.mean([row["funnel"]["minimum_aa_violation"] for row in records])),"mean_minimum_combined_violation":float(np.mean([row["funnel"]["minimum_combined_violation"] for row in records]))})
    if all("reason" in row for row in records): result["termination_reason_counts"]={reason:sum(row["reason"]==reason for row in records) for reason in ("red_kill","blue_kill","mutual_kill","collision","altitude_boundary","xy_boundary","boundary","max_steps")}
    result.update({"joint_distance_ata_rate":result["distance_and_ata_entry_rate"],"joint_distance_aa_rate":result["distance_and_aa_entry_rate"],"joint_ata_aa_rate":result["ata_and_aa_entry_rate"]})
    return result


class MAPPOTrainer:
    def __init__(self,env_config:str|Path,config:dict[str,Any])->None:
        self.env_config,self.config=str(env_config),config; t,n,e=config["training"],config["network"],config["experiment"]
        self.device=resolve_device(e["device"]); self.num_envs=int(t["num_envs"]); self.rollout_steps=int(t["rollout_steps"]); self.training_mode=t.get("training_mode","paper_staged")
        if t["minibatch_size"]>self.num_envs*self.rollout_steps: raise ValueError("minibatch_size exceeds active transitions")
        self.red_actor=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(self.device); self.blue_actor=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(self.device); self.critic=CentralizedCritic(14,n["hidden_dim"]).to(self.device)
        self.red_actor_optimizer=torch.optim.Adam(self.red_actor.parameters(),lr=t["learning_rate"]); self.blue_actor_optimizer=torch.optim.Adam(self.blue_actor.parameters(),lr=t["learning_rate"]); self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t["learning_rate"])
        self.envs=[HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]; self.buffer=MAPPOBuffer(self.rollout_steps,self.num_envs); self.rng=np.random.default_rng(e["seed"]); self.env_steps=0; self.update_count=0; self.fixed_gate_result=None
        self.current_observations=[env.reset(e["seed"]+i,"tail_chase",rear_team="red")[0] for i,env in enumerate(self.envs)]; self.episode_returns=np.zeros((self.num_envs,2)); self.episode_lengths=np.zeros(self.num_envs,dtype=int); self.funnels=[new_funnel() for _ in self.envs]; self.reward_sums=[{} for _ in self.envs]; self.last_control_diagnostics=[]

    def phase(self)->str:
        t=self.config["training"]
        if self.env_steps<t["straight_tail_chase_env_steps"]: return "straight_tail_chase"
        if self.env_steps<t["pursuit_tail_chase_env_steps"]: return "pursuit_tail_chase"
        if self.env_steps<t["fixed_training_env_steps"]: return "pursuit_all_scenarios"
        return "fixed_gate"

    def fixed_opponent_mode(self)->str:
        return "zero" if self.phase()=="straight_tail_chase" else "pursuit"

    def active_side(self)->str: return "red"

    def _reset_args(self)->tuple[str|None,str|None]:
        return ("tail_chase","red") if self.phase() in {"straight_tail_chase","pursuit_tail_chase"} else (None,None)

    def reset_environments(self)->None:
        self.current_observations=[]; scenario,rear=self._reset_args()
        for env in self.envs: self.current_observations.append(env.reset(int(self.rng.integers(2**31-1)),scenario,rear)[0])
        self.episode_returns.fill(0); self.episode_lengths.fill(0); self.funnels=[new_funnel() for _ in self.envs]; self.reward_sums=[{} for _ in self.envs]

    def collect_rollout(self,remaining_env_steps:int|None=None)->list[dict[str,Any]]:
        steps=self.rollout_steps if remaining_env_steps is None else min(self.rollout_steps,remaining_env_steps//self.num_envs)
        if steps<=0: raise ValueError("remaining_env_steps must contain at least one full vector step")
        if self.buffer.rollout_steps!=steps: self.buffer=MAPPOBuffer(steps,self.num_envs)
        self.buffer.clear(); completed=[]; self.last_control_diagnostics=[]; t=self.config["training"]; opponent=self.fixed_opponent_mode()
        for _ in range(steps):
            obs=np.asarray([[row["red_0"],row["blue_0"]] for row in self.current_observations],np.float32); states=np.asarray([env.global_state() for env in self.envs],np.float32)
            with torch.no_grad(): red_actions,red_logs=self.red_actor.sample_action(torch.as_tensor(obs[:,0],device=self.device)); values=self.critic(torch.as_tensor(states,device=self.device))
            blue_actions=np.zeros((self.num_envs,3),np.float32); actions=np.stack((red_actions.cpu().numpy(),blue_actions),1); logs=np.stack((red_logs.cpu().numpy(),np.zeros(self.num_envs,np.float32)),1); rewards=np.zeros((self.num_envs,2),np.float32); dones=np.zeros(self.num_envs,bool); next_observations=[]
            for index,env in enumerate(self.envs):
                if opponent=="pursuit":
                    red,blue=env.aircraft; cfg=env.config["action"]; actions[index,1]=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]).action(blue,red)
                observation,reward,terminated,truncated,info=env.step({"red_0":actions[index,0],"blue_0":actions[index,1]}); rewards[index]=[reward["red_0"],reward["blue_0"]]; self.episode_returns[index]+=rewards[index]; self.episode_lengths[index]+=1
                update_funnel(self.funnels[index],info["geometries"]["red_0"],env.config["combat"],info["attacks"]["red_0"])
                for key,value in info["reward_terms"]["red_0"].items(): self.reward_sums[index][key]=self.reward_sums[index].get(key,0.0)+value
                self.last_control_diagnostics.append(info["control_diagnostics"]["red_0"]); done=terminated or truncated; dones[index]=done
                if done:
                    result="win" if info["outcome"]=="red" else ("draw" if info["outcome"]=="draw" else "loss"); finish_funnel(self.funnels[index],info["termination_reason"],result)
                    completed.append({"returns":self.episode_returns[index].copy(),"length":int(self.episode_lengths[index]),"outcome":info["outcome"],"result":result,"reason":info["termination_reason"],"scenario_name":info["scenario_name"],"funnel":dict(self.funnels[index]),"reward_components":dict(self.reward_sums[index])})
                    self.episode_returns[index]=0; self.episode_lengths[index]=0; self.funnels[index]=new_funnel(); self.reward_sums[index]={}; scenario,rear=self._reset_args(); observation,_=env.reset(int(self.rng.integers(2**31-1)),scenario,rear)
                next_observations.append(observation)
            self.buffer.add(obs,states,actions,logs,rewards,values.cpu().numpy(),dones); self.current_observations=next_observations; self.env_steps+=self.num_envs
        with torch.no_grad(): last=self.critic(torch.as_tensor(np.asarray([env.global_state() for env in self.envs],np.float32),device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last,t["gamma"],t["gae_lambda"]); return completed

    def _update_actor(self)->dict[str,float]:
        actor,opt,label=self.red_actor,self.red_actor_optimizer,"red"; t=self.config["training"]; obs=torch.as_tensor(self.buffer.observations[:,:,0].reshape(-1,14),device=self.device); actions=torch.as_tensor(self.buffer.actions[:,:,0].reshape(-1,3),device=self.device); old=torch.as_tensor(self.buffer.log_probs[:,:,0].reshape(-1),device=self.device); advantage=torch.as_tensor(self.buffer.advantages[:,:,0].reshape(-1),device=self.device); advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8); data=[]
        for _ in range(t["ppo_epochs"]):
            order=self.rng.permutation(len(obs))
            for start in range(0,len(order),t["minibatch_size"]):
                idx=torch.as_tensor(order[start:start+t["minibatch_size"]],device=self.device); new,entropy=actor.evaluate_actions(obs[idx],actions[idx]); log_ratio=new-old[idx]; ratio=log_ratio.exp(); clipped=ratio.clamp(1-t["clip_coef"],1+t["clip_coef"]); policy_loss=-torch.minimum(ratio*advantage[idx],clipped*advantage[idx]).mean(); loss=policy_loss-t["entropy_coef"]*entropy.mean(); opt.zero_grad(); loss.backward(); grad=nn.utils.clip_grad_norm_(actor.parameters(),t["max_grad_norm"]); self._finite(label,loss,grad); opt.step(); data.append((policy_loss.item(),entropy.mean().item(),(((ratio-1)-log_ratio).mean()).item(),((ratio-1).abs()>t["clip_coef"]).float().mean().item(),float(grad)))
        values=np.asarray(data); return {"policy_loss":float(values[:,0].mean()),"entropy":float(values[:,1].mean()),"approx_kl":float(values[:,2].mean()),"clip_fraction":float(values[:,3].mean()),"grad_norm":float(values[:,4].mean()),"advantage_mean":float(advantage.mean()),"advantage_std":float(advantage.std(unbiased=False)),"advantage_nonzero_rate":float((advantage.abs()>1e-8).float().mean())}

    def update(self,active_override:str|None=None)->dict[str,float]:
        actor_metrics=self._update_actor(); metrics={f"red_{key}":value for key,value in actor_metrics.items()}; metrics.update({f"blue_{key}":np.nan for key in actor_metrics})
        t=self.config["training"]; states=torch.as_tensor(self.buffer.global_states.reshape(-1,14),device=self.device); returns=torch.as_tensor(self.buffer.returns.reshape(-1,2),device=self.device); losses=[]; final=self.critic.network[-1]; saved_weight=final.weight.data[1].clone(); saved_bias=final.bias.data[1].clone(); saved_optimizer=[]
        for parameter in (final.weight,final.bias):
            state=self.critic_optimizer.state.get(parameter,{}); saved_optimizer.append({key:value[1].clone() for key,value in state.items() if torch.is_tensor(value) and value.shape==parameter.shape})
        for _ in range(t["ppo_epochs"]):
            order=self.rng.permutation(len(states))
            for start in range(0,len(order),t["minibatch_size"]):
                idx=torch.as_tensor(order[start:start+t["minibatch_size"]],device=self.device); loss=((self.critic(states[idx])[:,0]-returns[idx,0])**2).mean(); self.critic_optimizer.zero_grad(); (t["value_loss_coef"]*loss).backward(); grad=nn.utils.clip_grad_norm_(self.critic.parameters(),t["max_grad_norm"]); self._finite("critic",loss,grad); self.critic_optimizer.step(); losses.append(loss.item())
        final.weight.data[1]=saved_weight; final.bias.data[1]=saved_bias
        for parameter,saved in zip((final.weight,final.bias),saved_optimizer):
            for key,value in saved.items(): self.critic_optimizer.state[parameter][key][1]=value
        metrics["value_loss"]=float(np.mean(losses)); self.update_count+=1; return metrics

    @staticmethod
    def _finite(label:str,*values:torch.Tensor)->None:
        if not all(torch.isfinite(value).all() for value in values): raise FloatingPointError(f"non-finite {label}")

    def save_checkpoint(self,path:str|Path)->None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save({"checkpoint_version":3,"red_actor":self.red_actor.state_dict(),"blue_actor":self.blue_actor.state_dict(),"critic":self.critic.state_dict(),"red_actor_optimizer":self.red_actor_optimizer.state_dict(),"blue_actor_optimizer":self.blue_actor_optimizer.state_dict(),"critic_optimizer":self.critic_optimizer.state_dict(),"environment_steps":self.env_steps,"env_steps":self.env_steps,"update":self.update_count,"training_mode":self.training_mode,"training_stage":self.phase(),"active_side":"red","alternating_block_index":0,"fixed_gate_result":self.fixed_gate_result,"config":self.config,"seed":self.config["experiment"]["seed"],"python_random_state":random.getstate(),"numpy_rng_state":self.rng.bit_generator.state,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

    def load_checkpoint(self,path:str|Path,load_optimizers:bool=True)->None:
        checkpoint=torch.load(path,map_location=self.device,weights_only=False)
        if checkpoint.get("checkpoint_version",0)<3: raise RuntimeError("v2 and earlier checkpoints are incompatible with the paper_staged v3 protocol")
        self.red_actor.load_state_dict(checkpoint["red_actor"]); self.blue_actor.load_state_dict(checkpoint["blue_actor"]); self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers: self.red_actor_optimizer.load_state_dict(checkpoint["red_actor_optimizer"]); self.blue_actor_optimizer.load_state_dict(checkpoint["blue_actor_optimizer"]); self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.env_steps=int(checkpoint["environment_steps"]); self.update_count=int(checkpoint["update"]); self.fixed_gate_result=checkpoint["fixed_gate_result"]; random.setstate(checkpoint["python_random_state"]); self.rng.bit_generator.state=checkpoint["numpy_rng_state"]; torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"] is not None: torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        self.reset_environments()


def evaluate_actor(actor:GaussianActor,env_config:str|Path,episodes:int,device:torch.device,opponent:str="zero",side:str="red",scenario:str="all",seed:int=10000)->dict[str,Any]:
    templates=("tail_chase","offset_head_on","crossing"); records=[]; actor.eval()
    for episode in range(episodes):
        learned=side; name=templates[episode%3] if scenario=="all" else scenario; env=HomogeneousAirCombatEnv(env_config); observations,_=env.reset(seed+episode,name,"red" if name=="tail_chase" else None); cfg=env.config["action"]; pursuit=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]); total=0.0; funnel=new_funnel()
        while True:
            with torch.no_grad(): learned_action=actor.deterministic_action(torch.as_tensor(observations[f"{learned}_0"],dtype=torch.float32,device=device)[None]).squeeze().cpu().numpy()
            red,blue=env.aircraft; opponent_action=np.zeros(3) if opponent=="zero" else pursuit.action(blue,red); actions={"red_0":learned_action,"blue_0":opponent_action}
            observations,rewards,terminated,truncated,info=env.step(actions); total+=rewards[f"{learned}_0"]; update_funnel(funnel,info["geometries"][f"{learned}_0"],env.config["combat"],info["attacks"][f"{learned}_0"])
            if terminated or truncated: break
        result="win" if info["outcome"]==learned else ("draw" if info["outcome"]=="draw" else "loss"); finish_funnel(funnel,info["termination_reason"],result); records.append({"scenario":name,"result":result,"reason":info["termination_reason"],"return":total,"length":info["step_count"],"funnel":funnel})
    actor.train(); return {"overall":summarize_records(records),"by_scenario":{name:summarize_records([row for row in records if row["scenario"]==name]) for name in templates if any(row["scenario"]==name for row in records)}}
