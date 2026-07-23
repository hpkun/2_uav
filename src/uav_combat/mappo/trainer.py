"""论文参考固定对手协议与项目适配竞争训练。"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch
from torch import nn
from ..environment import HomogeneousAirCombatEnv
from ..rule_policy import PurePursuitPolicy
from .buffer import MAPPOBuffer
from .networks import CentralizedCritic, GaussianActor

AGENT_IDS = ("red_0", "blue_0")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available")
    return device


def _new_funnel() -> dict[str, Any]:
    return {"minimum_distance": np.inf, "ever_within_4000m": False, "ever_within_attack_distance": False, "ever_satisfy_ata": False, "ever_satisfy_aa": False, "ever_satisfy_attack_envelope": False, "attack_envelope_steps": 0, "kill": False, "boundary": False, "max_steps": False}


class MAPPOTrainer:
    """固定对手优先，并可在门槛后交替冻结的训练器。"""

    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config, self.config = str(env_config), config; t,n,e=config["training"],config["network"],config["experiment"]
        self.device=resolve_device(e["device"]); self.num_envs=int(t["num_envs"]); self.rollout_steps=int(t["rollout_steps"]); self.training_mode=t.get("training_mode","paper_staged")
        if t["minibatch_size"]>self.num_envs*self.rollout_steps: raise ValueError("minibatch_size exceeds active transitions")
        self.red_actor=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(self.device); self.blue_actor=GaussianActor(14,3,n["hidden_dim"],n["log_std_init"]).to(self.device); self.critic=CentralizedCritic(14,n["hidden_dim"]).to(self.device)
        self.red_actor_optimizer=torch.optim.Adam(self.red_actor.parameters(),lr=t["learning_rate"]); self.blue_actor_optimizer=torch.optim.Adam(self.blue_actor.parameters(),lr=t["learning_rate"]); self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t["learning_rate"])
        self.envs=[HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]; self.buffer=MAPPOBuffer(self.rollout_steps,self.num_envs); self.rng=np.random.default_rng(e["seed"]); self.env_steps=0; self.update_count=0; self.fixed_gate_result=None
        self.current_observations=[env.reset(e["seed"]+i,"tail_chase",rear_team="red")[0] for i,env in enumerate(self.envs)]; self.episode_returns=np.zeros((self.num_envs,2)); self.episode_lengths=np.zeros(self.num_envs,dtype=int); self.funnels=[_new_funnel() for _ in self.envs]; self.reward_sums=[{} for _ in self.envs]; self.last_control_diagnostics=[]

    def phase(self) -> str:
        t=self.config["training"]
        if self.training_mode=="simultaneous_dual_actor": return "simultaneous_dual_actor"
        if self.env_steps<t["fixed_tail_chase_env_steps"]: return "fixed_tail_chase"
        if self.env_steps<t["fixed_training_env_steps"]: return "fixed_all_scenarios"
        return "alternating_competitive" if self.fixed_gate_result else "fixed_gate"

    def active_side(self) -> str:
        if self.phase() in {"fixed_tail_chase","fixed_all_scenarios","fixed_gate"}: return "red"
        if self.phase()=="simultaneous_dual_actor": return "both"
        block=(self.env_steps-self.config["training"]["fixed_training_env_steps"])//self.config["training"]["alternating_block_env_steps"]
        return "red" if block%2==0 else "blue"

    def _reset_args(self) -> tuple[str|None,str|None]:
        return ("tail_chase","red") if self.phase()=="fixed_tail_chase" else (None,None)

    def copy_red_to_blue_for_competition(self) -> None:
        """门槛通过后复制 red 策略，并重建 blue optimizer 清除旧动量。"""
        self.blue_actor.load_state_dict(deepcopy(self.red_actor.state_dict())); self.blue_actor_optimizer=torch.optim.Adam(self.blue_actor.parameters(),lr=self.config["training"]["learning_rate"]); self.fixed_gate_result=True

    def reset_environments(self) -> None:
        """Start a new episode batch when crossing a protocol-stage boundary."""
        self.current_observations=[]
        for env in self.envs:
            observation,_=env.reset(int(self.rng.integers(2**31-1)))
            self.current_observations.append(observation)
        self.episode_returns.fill(0); self.episode_lengths.fill(0)
        self.funnels=[_new_funnel() for _ in self.envs]; self.reward_sums=[{} for _ in self.envs]

    def collect_rollout(self, remaining_env_steps: int | None = None) -> list[dict[str,Any]]:
        steps=self.rollout_steps if remaining_env_steps is None else min(self.rollout_steps,remaining_env_steps//self.num_envs)
        if steps<=0: raise ValueError("remaining_env_steps must contain at least one full vector step")
        if self.buffer.rollout_steps!=steps: self.buffer=MAPPOBuffer(steps,self.num_envs)
        self.buffer.clear(); completed=[]; self.last_control_diagnostics=[]; t=self.config["training"]
        for _ in range(steps):
            obs=np.asarray([[x["red_0"],x["blue_0"]] for x in self.current_observations],np.float32); states=np.asarray([env.global_state() for env in self.envs],np.float32)
            with torch.no_grad():
                ra,rl=self.red_actor.sample_action(torch.as_tensor(obs[:,0],device=self.device)); values=self.critic(torch.as_tensor(states,device=self.device))
                if self.phase() in {"fixed_tail_chase","fixed_all_scenarios","fixed_gate"}:
                    ba=torch.zeros((self.num_envs,3),device=self.device); bl=torch.zeros(self.num_envs,device=self.device)
                else: ba,bl=self.blue_actor.sample_action(torch.as_tensor(obs[:,1],device=self.device))
            actions=np.stack((ra.cpu().numpy(),ba.cpu().numpy()),1); logs=np.stack((rl.cpu().numpy(),bl.cpu().numpy()),1); rewards=np.zeros((self.num_envs,2),np.float32); dones=np.zeros(self.num_envs,bool); next_obs=[]
            for i,env in enumerate(self.envs):
                if self.phase() in {"fixed_tail_chase","fixed_all_scenarios","fixed_gate"}:
                    red,blue=env.aircraft; cfg=env.config["action"]; pursuit=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]); actions[i,1]=pursuit.action(blue,red)
                observation,reward,terminated,truncated,info=env.step({"red_0":actions[i,0],"blue_0":actions[i,1]}); rewards[i]=[reward["red_0"],reward["blue_0"]]; self.episode_returns[i]+=rewards[i]; self.episode_lengths[i]+=1
                geometry=info["geometries"]["red_0"]; funnel=self.funnels[i]; combat=env.config["combat"]
                funnel["minimum_distance"]=min(funnel["minimum_distance"],geometry.distance); funnel["ever_within_4000m"]|=geometry.distance<=4000; funnel["ever_within_attack_distance"]|=combat["attack_distance_min"]<=geometry.distance<=combat["attack_distance_max"]; funnel["ever_satisfy_ata"]|=geometry.ata<=combat["attack_ata_max"]; funnel["ever_satisfy_aa"]|=geometry.aa<=combat["attack_aa_max"]; funnel["ever_satisfy_attack_envelope"]|=info["attacks"]["red_0"]; funnel["attack_envelope_steps"]+=int(info["attacks"]["red_0"])
                for key,value in info["reward_terms"]["red_0"].items(): self.reward_sums[i][key]=self.reward_sums[i].get(key,0.0)+value
                self.last_control_diagnostics.extend(info["control_diagnostics"].values()); done=terminated or truncated; dones[i]=done
                if done:
                    reason=info["termination_reason"]; funnel["kill"]=reason=="red_kill"; funnel["boundary"]=reason in {"boundary","xy_boundary","altitude_boundary"}; funnel["max_steps"]=reason=="max_steps"
                    completed.append({"returns":self.episode_returns[i].copy(),"length":int(self.episode_lengths[i]),"outcome":info["outcome"],"reason":reason,"scenario_name":info["scenario_name"],"funnel":dict(funnel),"reward_components":dict(self.reward_sums[i])})
                    self.episode_returns[i]=0; self.episode_lengths[i]=0; self.funnels[i]=_new_funnel(); self.reward_sums[i]={}; scenario,rear=self._reset_args(); observation,_=env.reset(int(self.rng.integers(2**31-1)),scenario,rear)
                next_obs.append(observation)
            self.buffer.add(obs,states,actions,logs,rewards,values.cpu().numpy(),dones); self.current_observations=next_obs; self.env_steps+=self.num_envs
        with torch.no_grad(): last=self.critic(torch.as_tensor(np.asarray([e.global_state() for e in self.envs],np.float32),device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last,t["gamma"],t["gae_lambda"]); return completed

    def _update_actor(self,side:int) -> dict[str,float]:
        actor,opt,label=(self.red_actor,self.red_actor_optimizer,"red") if side==0 else (self.blue_actor,self.blue_actor_optimizer,"blue"); t=self.config["training"]; obs=torch.as_tensor(self.buffer.observations[:,:,side].reshape(-1,14),device=self.device); actions=torch.as_tensor(self.buffer.actions[:,:,side].reshape(-1,3),device=self.device); old=torch.as_tensor(self.buffer.log_probs[:,:,side].reshape(-1),device=self.device); adv=torch.as_tensor(self.buffer.advantages[:,:,side].reshape(-1),device=self.device); adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8); data=[]
        for _ in range(t["ppo_epochs"]):
            order=self.rng.permutation(len(obs))
            for start in range(0,len(order),t["minibatch_size"]):
                idx=torch.as_tensor(order[start:start+t["minibatch_size"]],device=self.device); new,entropy=actor.evaluate_actions(obs[idx],actions[idx]); lr=new-old[idx]; ratio=lr.exp(); clipped=ratio.clamp(1-t["clip_coef"],1+t["clip_coef"]); pl=-torch.minimum(ratio*adv[idx],clipped*adv[idx]).mean(); loss=pl-t["entropy_coef"]*entropy.mean(); opt.zero_grad(); loss.backward(); grad=nn.utils.clip_grad_norm_(actor.parameters(),t["max_grad_norm"]); self._finite(label,loss,grad); opt.step(); data.append((pl.item(),entropy.mean().item(),(((ratio-1)-lr).mean()).item(),((ratio-1).abs()>t["clip_coef"]).float().mean().item(),float(grad)))
        a=np.asarray(data); return {"policy_loss":float(a[:,0].mean()),"entropy":float(a[:,1].mean()),"approx_kl":float(a[:,2].mean()),"clip_fraction":float(a[:,3].mean()),"grad_norm":float(a[:,4].mean()),"advantage_mean":float(adv.mean()),"advantage_std":float(adv.std(unbiased=False)),"advantage_nonzero_rate":float((adv.abs()>1e-8).float().mean())}

    def update(self,active_override: str | None = None) -> dict[str,float]:
        active=active_override or self.active_side(); metrics={}; nan={k:np.nan for k in ("policy_loss","entropy","approx_kl","clip_fraction","grad_norm","advantage_mean","advantage_std","advantage_nonzero_rate")}
        for side,name in enumerate(("red","blue")): metrics.update({f"{name}_{k}":v for k,v in (self._update_actor(side) if active in {name,"both"} else nan).items()})
        t=self.config["training"]; states=torch.as_tensor(self.buffer.global_states.reshape(-1,14),device=self.device); returns=torch.as_tensor(self.buffer.returns.reshape(-1,2),device=self.device); active_indices=(0,1) if active=="both" else ((0,) if active=="red" else (1,)); losses=[]
        final=self.critic.network[-1]; inactive=[i for i in (0,1) if i not in active_indices]; saved_w=final.weight.data[inactive].clone(); saved_b=final.bias.data[inactive].clone(); saved_optimizer_rows=[]
        for parameter in (final.weight,final.bias):
            state=self.critic_optimizer.state.get(parameter,{})
            saved_optimizer_rows.append({key:value[inactive].clone() for key,value in state.items() if torch.is_tensor(value) and value.shape==parameter.shape})
        for _ in range(t["ppo_epochs"]):
            order=self.rng.permutation(len(states))
            for start in range(0,len(order),t["minibatch_size"]):
                idx=torch.as_tensor(order[start:start+t["minibatch_size"]],device=self.device); predicted=self.critic(states[idx])[:,active_indices]; target=returns[idx][:,active_indices]; loss=((predicted-target)**2).mean(); self.critic_optimizer.zero_grad(); (t["value_loss_coef"]*loss).backward(); grad=nn.utils.clip_grad_norm_(self.critic.parameters(),t["max_grad_norm"]); self._finite("critic",loss,grad); self.critic_optimizer.step(); losses.append(loss.item())
        if inactive:
            final.weight.data[inactive]=saved_w; final.bias.data[inactive]=saved_b
            for parameter,saved in zip((final.weight,final.bias),saved_optimizer_rows):
                for key,value in saved.items(): self.critic_optimizer.state[parameter][key][inactive]=value
        metrics["value_loss"]=float(np.mean(losses)); self.update_count+=1; return metrics

    @staticmethod
    def _finite(label:str,*values:torch.Tensor)->None:
        if not all(torch.isfinite(v).all() for v in values): raise FloatingPointError(f"non-finite {label}")

    def save_checkpoint(self,path:str|Path)->None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save({"checkpoint_version":3,"red_actor":self.red_actor.state_dict(),"blue_actor":self.blue_actor.state_dict(),"critic":self.critic.state_dict(),"red_actor_optimizer":self.red_actor_optimizer.state_dict(),"blue_actor_optimizer":self.blue_actor_optimizer.state_dict(),"critic_optimizer":self.critic_optimizer.state_dict(),"environment_steps":self.env_steps,"env_steps":self.env_steps,"update":self.update_count,"training_mode":self.training_mode,"training_stage":self.phase(),"active_side":self.active_side(),"alternating_block_index":max(0,(self.env_steps-self.config["training"]["fixed_training_env_steps"])//self.config["training"]["alternating_block_env_steps"]),"fixed_gate_result":self.fixed_gate_result,"config":self.config,"seed":self.config["experiment"]["seed"],"python_random_state":random.getstate(),"numpy_rng_state":self.rng.bit_generator.state,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state":cuda_state},path)

    def load_checkpoint(self,path:str|Path,load_optimizers:bool=True)->None:
        c=torch.load(path,map_location=self.device,weights_only=False)
        if c.get("checkpoint_version",0)<3: raise RuntimeError("v2及更早检查点不兼容paper_staged v3训练协议")
        self.red_actor.load_state_dict(c["red_actor"]); self.blue_actor.load_state_dict(c["blue_actor"]); self.critic.load_state_dict(c["critic"])
        if load_optimizers: self.red_actor_optimizer.load_state_dict(c["red_actor_optimizer"]); self.blue_actor_optimizer.load_state_dict(c["blue_actor_optimizer"]); self.critic_optimizer.load_state_dict(c["critic_optimizer"])
        self.env_steps=int(c["environment_steps"]); self.update_count=int(c["update"]); self.fixed_gate_result=c["fixed_gate_result"]; random.setstate(c["python_random_state"]); self.rng.bit_generator.state=c["numpy_rng_state"]; torch.set_rng_state(c["torch_cpu_rng_state"])
        if torch.cuda.is_available() and c["torch_cuda_rng_state"] is not None: torch.cuda.set_rng_state_all(c["torch_cuda_rng_state"])


def evaluate_actor(actor:GaussianActor,env_config:str|Path,episodes:int,device:torch.device,opponent:str="zero",side:str="red",scenario:str="all",seed:int=10000)->dict[str,Any]:
    templates=("tail_chase","offset_head_on","crossing"); records=[]; actor.eval()
    for episode in range(episodes):
        learned=side if side!="both" else ("red" if episode%2==0 else "blue"); name=templates[episode%3] if scenario=="all" else scenario; env=HomogeneousAirCombatEnv(env_config); observations,_=env.reset(seed+episode,name,"red" if name=="tail_chase" and learned=="red" else None); cfg=env.config["action"]; pursuit=PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]); total=0.0; funnel=_new_funnel()
        for _ in range(env.config["simulation"]["max_steps"]):
            actions={}
            for aid in AGENT_IDS:
                team=aid.split("_")[0]
                if team==learned:
                    with torch.no_grad(): actions[aid]=actor.deterministic_action(torch.as_tensor(observations[aid],dtype=torch.float32,device=device)[None]).squeeze().cpu().numpy()
                elif opponent=="zero": actions[aid]=np.zeros(3)
                else: own=next(a for a in env.aircraft if a.aircraft_id==aid); target=next(a for a in env.aircraft if a.team!=own.team); actions[aid]=pursuit.action(own,target)
            observations,rewards,terminated,truncated,info=env.step(actions); total+=rewards[f"{learned}_0"]; g=info["geometries"][f"{learned}_0"]; combat=env.config["combat"]; funnel["minimum_distance"]=min(funnel["minimum_distance"],g.distance); funnel["ever_within_4000m"]|=g.distance<=4000; funnel["ever_within_attack_distance"]|=combat["attack_distance_min"]<=g.distance<=combat["attack_distance_max"]; funnel["ever_satisfy_ata"]|=g.ata<=combat["attack_ata_max"]; funnel["ever_satisfy_aa"]|=g.aa<=combat["attack_aa_max"]; funnel["ever_satisfy_attack_envelope"]|=info["attacks"][f"{learned}_0"]; funnel["attack_envelope_steps"]+=int(info["attacks"][f"{learned}_0"])
            if terminated or truncated: break
        result="win" if info["outcome"]==learned else ("draw" if info["outcome"]=="draw" else "loss"); funnel["kill"]=result=="win"; funnel["boundary"]=info["termination_reason"] in {"boundary","xy_boundary","altitude_boundary"}; funnel["max_steps"]=info["termination_reason"]=="max_steps"; records.append({"scenario":name,"result":result,"return":total,"length":info["step_count"],"funnel":funnel})
    actor.train()
    def summary(items):
        c=len(items); envelope=sum(x["funnel"]["ever_satisfy_attack_envelope"] for x in items); kills=sum(x["funnel"]["kill"] for x in items)
        return {"episodes":c,"wins":sum(x["result"]=="win" for x in items),"losses":sum(x["result"]=="loss" for x in items),"draws":sum(x["result"]=="draw" for x in items),"win_rate":sum(x["result"]=="win" for x in items)/c,"loss_rate":sum(x["result"]=="loss" for x in items)/c,"draw_rate":sum(x["result"]=="draw" for x in items)/c,"mean_return":float(np.mean([x["return"] for x in items])),"mean_episode_length":float(np.mean([x["length"] for x in items])),"boundary_rate":sum(x["funnel"]["boundary"] for x in items)/c,"max_steps_rate":sum(x["funnel"]["max_steps"] for x in items)/c,"within_4000_rate":sum(x["funnel"]["ever_within_4000m"] for x in items)/c,"attack_distance_entry_rate":sum(x["funnel"]["ever_within_attack_distance"] for x in items)/c,"ata_gate_rate":sum(x["funnel"]["ever_satisfy_ata"] for x in items)/c,"aa_gate_rate":sum(x["funnel"]["ever_satisfy_aa"] for x in items)/c,"attack_envelope_entry_rate":envelope/c,"attack_to_kill_conversion_rate":kills/envelope if envelope else 0.0,"kill_rate":kills/c}
    return {"overall":summary(records),"by_scenario":{n:summary([x for x in records if x["scenario"]==n]) for n in templates if any(x["scenario"]==n for x in records)}}
