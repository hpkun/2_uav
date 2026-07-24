"""Alternating-freeze competitive MAPPO training and evaluation utilities."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import random

import numpy as np
import torch
from torch import nn

from ..environment import HomogeneousAirCombatEnv
from ..rule_policy import PurePursuitPolicy
from .buffer import MAPPOBuffer
from .networks import CentralizedCritic, GaussianActor

AGENT_IDS = ("red_0", "blue_0")
TEAMS = ("red", "blue")
SCENARIOS = ("tail_chase", "offset_head_on", "crossing")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def new_funnel() -> dict[str, Any]:
    return {
        "minimum_distance": np.inf, "ever_within_4000m": False,
        "ever_within_attack_distance": False, "ever_satisfy_ata": False,
        "ever_satisfy_aa": False, "ever_distance_and_ata": False,
        "ever_distance_and_aa": False, "ever_ata_and_aa": False,
        "ever_full_attack_envelope": False, "ever_satisfy_attack_envelope": False,
        "attack_envelope_steps": 0, "minimum_distance_violation": np.inf,
        "minimum_ata_violation": np.inf, "minimum_aa_violation": np.inf,
        "minimum_combined_violation": np.inf, "kill": False, "boundary": False,
        "max_steps": False, "collision": False,
    }


def update_funnel(funnel: dict[str, Any], geometry: Any, combat: dict[str, Any], attacked: bool) -> None:
    distance_gate = combat["attack_distance_min"] <= geometry.distance <= combat["attack_distance_max"]
    ata_gate = geometry.ata <= combat["attack_ata_max"]
    aa_gate = geometry.aa <= combat["attack_aa_max"]
    dv = max(combat["attack_distance_min"] - geometry.distance, geometry.distance - combat["attack_distance_max"], 0.0)
    av = max(geometry.ata - combat["attack_ata_max"], 0.0)
    aav = max(geometry.aa - combat["attack_aa_max"], 0.0)
    funnel["minimum_distance"] = min(funnel["minimum_distance"], geometry.distance)
    funnel["ever_within_4000m"] |= geometry.distance <= 4000.0
    funnel["ever_within_attack_distance"] |= distance_gate
    funnel["ever_satisfy_ata"] |= ata_gate
    funnel["ever_satisfy_aa"] |= aa_gate
    funnel["ever_distance_and_ata"] |= distance_gate and ata_gate
    funnel["ever_distance_and_aa"] |= distance_gate and aa_gate
    funnel["ever_ata_and_aa"] |= ata_gate and aa_gate
    funnel["ever_full_attack_envelope"] |= distance_gate and ata_gate and aa_gate
    funnel["ever_satisfy_attack_envelope"] = funnel["ever_full_attack_envelope"]
    funnel["attack_envelope_steps"] += int(distance_gate and ata_gate and aa_gate)
    funnel["minimum_distance_violation"] = min(funnel["minimum_distance_violation"], dv)
    funnel["minimum_ata_violation"] = min(funnel["minimum_ata_violation"], av)
    funnel["minimum_aa_violation"] = min(funnel["minimum_aa_violation"], aav)
    funnel["minimum_combined_violation"] = min(funnel["minimum_combined_violation"], dv / combat["attack_distance_max"] + av / np.pi + aav / np.pi)
    if attacked and not (distance_gate and ata_gate and aa_gate):
        raise AssertionError("an attack cannot occur outside the full attack envelope")


def finish_funnel(funnel: dict[str, Any], reason: str | None, team: str) -> None:
    if team in {"win", "loss", "draw"}:  # compatibility for the rule-baseline reporter
        team = "red"
    funnel["kill"] = reason == f"{team}_kill"
    funnel["boundary"] = reason in {"boundary", "xy_boundary", "altitude_boundary"}
    funnel["max_steps"] = reason == "max_steps"
    funnel["collision"] = reason == "collision"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize legacy red-perspective rule-policy records."""
    if not records: raise ValueError("cannot summarize zero episodes")
    n = len(records); funnels = [row["funnel"] for row in records]
    rate = lambda key: float(np.mean([bool(f[key]) for f in funnels]))
    wins = sum(row["result"] == "win" for row in records); losses = sum(row["result"] == "loss" for row in records); draws = n - wins - losses
    envelopes = sum(f["ever_full_attack_envelope"] for f in funnels); kills = sum(f["kill"] for f in funnels)
    return {"episodes": n, "wins": wins, "losses": losses, "draws": draws, "win_rate": wins / n, "loss_rate": losses / n, "draw_rate": draws / n, "mean_return": float(np.mean([row["return"] for row in records])), "mean_episode_length": float(np.mean([row["length"] for row in records])), "boundary_rate": rate("boundary"), "max_steps_rate": rate("max_steps"), "collision_rate": rate("collision"), "within_4000_rate": rate("ever_within_4000m"), "attack_distance_entry_rate": rate("ever_within_attack_distance"), "ata_gate_rate": rate("ever_satisfy_ata"), "aa_gate_rate": rate("ever_satisfy_aa"), "distance_and_ata_entry_rate": rate("ever_distance_and_ata"), "distance_and_aa_entry_rate": rate("ever_distance_and_aa"), "ata_and_aa_entry_rate": rate("ever_ata_and_aa"), "full_attack_envelope_entry_rate": rate("ever_full_attack_envelope"), "attack_envelope_entry_rate": rate("ever_full_attack_envelope"), "attack_to_kill_conversion_rate": kills / envelopes if envelopes else 0.0, "kill_rate": kills / n}


def _funnel_summary(records: list[dict[str, Any]], team: str) -> dict[str, float]:
    funnels = [row["funnels"][team] for row in records]
    n = len(funnels)
    if not n:
        return {}
    rate = lambda key: float(np.mean([bool(f[key]) for f in funnels]))
    envelopes = sum(bool(f["ever_full_attack_envelope"]) for f in funnels)
    kills = sum(bool(f["kill"]) for f in funnels)
    return {
        "within_4000_rate": rate("ever_within_4000m"),
        "attack_distance_entry_rate": rate("ever_within_attack_distance"),
        "ata_gate_rate": rate("ever_satisfy_ata"), "aa_gate_rate": rate("ever_satisfy_aa"),
        "distance_and_ata_entry_rate": rate("ever_distance_and_ata"),
        "distance_and_aa_entry_rate": rate("ever_distance_and_aa"),
        "ata_and_aa_entry_rate": rate("ever_ata_and_aa"),
        "full_attack_envelope_entry_rate": rate("ever_full_attack_envelope"),
        "attack_to_kill_conversion_rate": kills / envelopes if envelopes else 0.0,
        "kill_rate": kills / n,
        "mean_minimum_distance": float(np.mean([f["minimum_distance"] for f in funnels])),
        "mean_minimum_combined_violation": float(np.mean([f["minimum_combined_violation"] for f in funnels])),
    }


def summarize_competitive_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero episodes")
    n = len(records)
    red_wins = sum(row["outcome"] == "red" for row in records)
    blue_wins = sum(row["outcome"] == "blue" for row in records)
    draws = n - red_wins - blue_wins
    boundary = sum(row["reason"] in {"boundary", "xy_boundary", "altitude_boundary"} for row in records)
    collision = sum(row["reason"] == "collision" for row in records)
    return {
        "episodes": n, "red_wins": red_wins, "blue_wins": blue_wins, "draws": draws,
        "red_win_rate": red_wins / n, "blue_win_rate": blue_wins / n, "draw_rate": draws / n,
        "decisive_rate": (red_wins + blue_wins) / n, "boundary_rate": boundary / n,
        "collision_rate": collision / n,
        "mean_episode_length": float(np.mean([row["length"] for row in records])),
        "red_mean_return": float(np.mean([row["returns"][0] for row in records])),
        "blue_mean_return": float(np.mean([row["returns"][1] for row in records])),
        "termination_reason_counts": {reason: sum(row["reason"] == reason for row in records) for reason in ("red_kill", "blue_kill", "mutual_kill", "collision", "altitude_boundary", "xy_boundary", "boundary", "max_steps")},
        "red_funnel": _funnel_summary(records, "red"), "blue_funnel": _funnel_summary(records, "blue"),
    }


class MAPPOTrainer:
    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config, self.config = str(env_config), config
        t, n, e = config["training"], config["network"], config["experiment"]
        if t.get("training_mode") != "alternating_self_play":
            raise ValueError("training_mode must be alternating_self_play")
        random.seed(e["seed"]); torch.manual_seed(e["seed"])
        self.device = resolve_device(e["device"])
        self.num_envs, self.rollout_steps = int(t["num_envs"]), int(t["rollout_steps"])
        if t["minibatch_size"] > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size exceeds active transitions")
        self.red_actor = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        self.blue_actor = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        self.blue_actor.load_state_dict(deepcopy(self.red_actor.state_dict()))
        self.red_critic = CentralizedCritic(14, n["hidden_dim"]).to(self.device)
        self.blue_critic = CentralizedCritic(14, n["hidden_dim"]).to(self.device)
        lr = t["learning_rate"]
        self.red_actor_optimizer = torch.optim.Adam(self.red_actor.parameters(), lr=lr)
        self.blue_actor_optimizer = torch.optim.Adam(self.blue_actor.parameters(), lr=lr)
        self.red_critic_optimizer = torch.optim.Adam(self.red_critic.parameters(), lr=lr)
        self.blue_critic_optimizer = torch.optim.Adam(self.blue_critic.parameters(), lr=lr)
        self.envs = [HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]
        self.buffer = MAPPOBuffer(self.rollout_steps, self.num_envs)
        self.rng = np.random.default_rng(e["seed"])
        self.env_steps = self.update_count = self.scenario_counter = self.tail_rear_counter = 0
        self.scenario_counts = {name: 0 for name in SCENARIOS}
        self.tail_rear_counts = {team: 0 for team in TEAMS}
        self.current_observations: list[dict[str, np.ndarray]] = []
        self.current_scenarios: list[str] = []
        self.current_rear_teams: list[str | None] = []
        self.episode_returns = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.funnels = [{team: new_funnel() for team in TEAMS} for _ in self.envs]
        self.last_control_diagnostics = {team: [] for team in TEAMS}
        self.reset_environments()

    @property
    def block_env_steps(self) -> int:
        return int(self.config["training"]["alternating_block_env_steps"])

    def block_index(self) -> int:
        return self.env_steps // self.block_env_steps

    def active_side(self) -> str:
        return TEAMS[self.block_index() % 2]

    def _next_reset(self, env: HomogeneousAirCombatEnv) -> tuple[dict[str, np.ndarray], str, str | None]:
        scenario = SCENARIOS[self.scenario_counter % len(SCENARIOS)]
        self.scenario_counter += 1; self.scenario_counts[scenario] += 1
        rear = None
        if scenario == "tail_chase":
            rear = TEAMS[self.tail_rear_counter % 2]
            self.tail_rear_counter += 1; self.tail_rear_counts[rear] += 1
        observation, _ = env.reset(int(self.rng.integers(2**31 - 1)), scenario, rear)
        return observation, scenario, rear

    def reset_environments(self) -> None:
        self.current_observations, self.current_scenarios, self.current_rear_teams = [], [], []
        for env in self.envs:
            observation, scenario, rear = self._next_reset(env)
            self.current_observations.append(observation); self.current_scenarios.append(scenario); self.current_rear_teams.append(rear)
        self.episode_returns.fill(0); self.episode_lengths.fill(0)
        self.funnels = [{team: new_funnel() for team in TEAMS} for _ in self.envs]

    def collect_rollout(self, remaining_env_steps: int | None = None) -> list[dict[str, Any]]:
        steps = self.rollout_steps if remaining_env_steps is None else min(self.rollout_steps, remaining_env_steps // self.num_envs)
        if steps <= 0:
            raise ValueError("remaining_env_steps must contain at least one full vector step")
        if self.buffer.rollout_steps != steps:
            self.buffer = MAPPOBuffer(steps, self.num_envs)
        self.buffer.clear(); completed: list[dict[str, Any]] = []
        self.last_control_diagnostics = {team: [] for team in TEAMS}
        for _ in range(steps):
            obs = np.asarray([[row[agent] for agent in AGENT_IDS] for row in self.current_observations], np.float32)
            states = np.asarray([env.global_state() for env in self.envs], np.float32)
            with torch.no_grad():
                red_actions, red_logs = self.red_actor.sample_action(torch.as_tensor(obs[:, 0], device=self.device))
                blue_actions, blue_logs = self.blue_actor.sample_action(torch.as_tensor(obs[:, 1], device=self.device))
                state_tensor = torch.as_tensor(states, device=self.device)
                values = torch.stack((self.red_critic(state_tensor), self.blue_critic(state_tensor)), dim=1)
            actions = torch.stack((red_actions, blue_actions), dim=1).cpu().numpy()
            logs = torch.stack((red_logs, blue_logs), dim=1).cpu().numpy()
            rewards = np.zeros((self.num_envs, 2), np.float32); dones = np.zeros(self.num_envs, bool); next_obs = []
            for index, env in enumerate(self.envs):
                observation, reward, terminated, truncated, info = env.step({agent: actions[index, side] for side, agent in enumerate(AGENT_IDS)})
                rewards[index] = [reward[agent] for agent in AGENT_IDS]
                self.episode_returns[index] += rewards[index]; self.episode_lengths[index] += 1
                for team, agent in zip(TEAMS, AGENT_IDS):
                    update_funnel(self.funnels[index][team], info["geometries"][agent], env.config["combat"], info["attacks"][agent])
                    self.last_control_diagnostics[team].append(info["control_diagnostics"][agent])
                done = terminated or truncated; dones[index] = done
                if done:
                    for team in TEAMS: finish_funnel(self.funnels[index][team], info["termination_reason"], team)
                    completed.append({"returns": self.episode_returns[index].copy(), "length": int(self.episode_lengths[index]), "outcome": info["outcome"], "reason": info["termination_reason"], "scenario": self.current_scenarios[index], "rear_team": self.current_rear_teams[index], "funnels": deepcopy(self.funnels[index])})
                    self.episode_returns[index] = 0; self.episode_lengths[index] = 0
                    self.funnels[index] = {team: new_funnel() for team in TEAMS}
                    observation, scenario, rear = self._next_reset(env)
                    self.current_scenarios[index], self.current_rear_teams[index] = scenario, rear
                next_obs.append(observation)
            self.buffer.add(obs, states, actions, logs, rewards, values.cpu().numpy(), dones)
            self.current_observations = next_obs; self.env_steps += self.num_envs
        with torch.no_grad():
            states = torch.as_tensor(np.asarray([env.global_state() for env in self.envs], np.float32), device=self.device)
            last = torch.stack((self.red_critic(states), self.blue_critic(states)), dim=1).cpu().numpy()
        t = self.config["training"]
        self.buffer.compute_returns_and_advantages(last, t["gamma"], t["gae_lambda"])
        return completed

    @staticmethod
    def _finite(label: str, *values: torch.Tensor) -> None:
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError(f"non-finite {label}")

    def _update_actor(self, side: int) -> dict[str, float]:
        actor = (self.red_actor, self.blue_actor)[side]; optimizer = (self.red_actor_optimizer, self.blue_actor_optimizer)[side]
        t = self.config["training"]
        obs = torch.as_tensor(self.buffer.observations[:, :, side].reshape(-1, 14), device=self.device)
        actions = torch.as_tensor(self.buffer.actions[:, :, side].reshape(-1, 3), device=self.device)
        old = torch.as_tensor(self.buffer.log_probs[:, :, side].reshape(-1), device=self.device)
        advantage = torch.as_tensor(self.buffer.advantages[:, :, side].reshape(-1), device=self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8); data = []
        for _ in range(t["ppo_epochs"]):
            order = self.rng.permutation(len(obs))
            for start in range(0, len(obs), t["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + t["minibatch_size"]], device=self.device)
                new, entropy = actor.evaluate_actions(obs[idx], actions[idx]); log_ratio = new - old[idx]; ratio = log_ratio.exp()
                clipped = ratio.clamp(1 - t["clip_coef"], 1 + t["clip_coef"])
                policy_loss = -torch.minimum(ratio * advantage[idx], clipped * advantage[idx]).mean()
                loss = policy_loss - t["entropy_coef"] * entropy.mean()
                optimizer.zero_grad(); loss.backward(); grad = nn.utils.clip_grad_norm_(actor.parameters(), t["max_grad_norm"])
                self._finite(TEAMS[side], loss, grad); optimizer.step()
                data.append((policy_loss.item(), entropy.mean().item(), (((ratio - 1) - log_ratio).mean()).item(), ((ratio - 1).abs() > t["clip_coef"]).float().mean().item(), float(grad)))
        values = np.asarray(data)
        return {"policy_loss": float(values[:, 0].mean()), "entropy": float(values[:, 1].mean()), "approx_kl": float(values[:, 2].mean()), "clip_fraction": float(values[:, 3].mean()), "grad_norm": float(values[:, 4].mean()), "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std(unbiased=False)), "advantage_nonzero_rate": float((advantage.abs() > 1e-8).float().mean())}

    def _update_critic(self, side: int) -> dict[str, float]:
        critic = (self.red_critic, self.blue_critic)[side]; optimizer = (self.red_critic_optimizer, self.blue_critic_optimizer)[side]
        t = self.config["training"]
        states = torch.as_tensor(self.buffer.global_states.reshape(-1, 14), device=self.device)
        returns = torch.as_tensor(self.buffer.returns[:, :, side].reshape(-1), device=self.device); losses = []; grads = []
        for _ in range(t["ppo_epochs"]):
            order = self.rng.permutation(len(states))
            for start in range(0, len(states), t["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + t["minibatch_size"]], device=self.device)
                loss = ((critic(states[idx]) - returns[idx]) ** 2).mean()
                optimizer.zero_grad(); (t["value_loss_coef"] * loss).backward(); grad = nn.utils.clip_grad_norm_(critic.parameters(), t["max_grad_norm"])
                self._finite(f"{TEAMS[side]} critic", loss, grad); optimizer.step(); losses.append(loss.item()); grads.append(float(grad))
        return {"value_loss": float(np.mean(losses)), "critic_grad_norm": float(np.mean(grads))}

    def update(self, active_override: str | None = None) -> dict[str, float]:
        active = active_override or self.active_side()
        if active not in TEAMS: raise ValueError("active side must be red or blue")
        side = TEAMS.index(active); actor = self._update_actor(side); critic = self._update_critic(side)
        keys = tuple(actor) + tuple(critic); metrics: dict[str, float] = {}
        for team in TEAMS:
            for key in keys: metrics[f"{team}_{key}"] = (actor | critic)[key] if team == active else np.nan
        self.update_count += 1; metrics["active_side"] = active
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_version": 4, "red_actor": self.red_actor.state_dict(), "blue_actor": self.blue_actor.state_dict(),
            "red_critic": self.red_critic.state_dict(), "blue_critic": self.blue_critic.state_dict(),
            "red_actor_optimizer": self.red_actor_optimizer.state_dict(), "blue_actor_optimizer": self.blue_actor_optimizer.state_dict(),
            "red_critic_optimizer": self.red_critic_optimizer.state_dict(), "blue_critic_optimizer": self.blue_critic_optimizer.state_dict(),
            "environment_steps": self.env_steps, "env_steps": self.env_steps, "update": self.update_count,
            "training_mode": "alternating_self_play", "active_side": self.active_side(), "alternating_block_index": self.block_index(),
            "scenario_counter": self.scenario_counter, "tail_rear_counter": self.tail_rear_counter,
            "scenario_counts": self.scenario_counts, "tail_rear_counts": self.tail_rear_counts, "config": self.config,
            "python_random_state": random.getstate(), "numpy_rng_state": self.rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(), "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("checkpoint_version") != 4:
            raise RuntimeError("checkpoint is incompatible: alternating self-play requires v4; v3 fixed-opponent checkpoints are rejected")
        for name in ("red_actor", "blue_actor", "red_critic", "blue_critic"):
            getattr(self, name).load_state_dict(checkpoint[name])
        if load_optimizers:
            for name in ("red_actor_optimizer", "blue_actor_optimizer", "red_critic_optimizer", "blue_critic_optimizer"):
                getattr(self, name).load_state_dict(checkpoint[name])
        self.env_steps = int(checkpoint["environment_steps"]); self.update_count = int(checkpoint["update"])
        self.scenario_counter = int(checkpoint.get("scenario_counter", 0)); self.tail_rear_counter = int(checkpoint.get("tail_rear_counter", 0))
        self.scenario_counts = dict(checkpoint.get("scenario_counts", self.scenario_counts)); self.tail_rear_counts = dict(checkpoint.get("tail_rear_counts", self.tail_rear_counts))
        random.setstate(checkpoint["python_random_state"]); self.rng.bit_generator.state = checkpoint["numpy_rng_state"]; torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"] is not None: torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        self.reset_environments()


def _actor_policy(actor: GaussianActor, device: torch.device) -> Callable[[np.ndarray, HomogeneousAirCombatEnv, str], np.ndarray]:
    def act(observation: np.ndarray, _env: HomogeneousAirCombatEnv, _team: str) -> np.ndarray:
        with torch.no_grad():
            return actor.deterministic_action(torch.as_tensor(observation, dtype=torch.float32, device=device)[None]).squeeze(0).cpu().numpy()
    return act


def _rule_policy(kind: str) -> Callable[[np.ndarray, HomogeneousAirCombatEnv, str], np.ndarray]:
    def act(_observation: np.ndarray, env: HomogeneousAirCombatEnv, team: str) -> np.ndarray:
        if kind == "zero": return np.zeros(3, dtype=np.float32)
        own, target = env.aircraft if team == "red" else reversed(env.aircraft)
        cfg = env.config["action"]
        return PurePursuitPolicy(cfg["delta_yaw_max"], cfg["delta_pitch_max"], cfg["delta_speed_max"]).action(own, target)
    return act


def _evaluate_policies(red_policy: Callable, blue_policy: Callable, env_config: str | Path, episodes: int, scenario: str, seed: int) -> dict[str, Any]:
    records = []; tail_index = 0
    for episode in range(episodes):
        name = SCENARIOS[episode % 3] if scenario == "all" else scenario
        rear = None
        if name == "tail_chase": rear = TEAMS[tail_index % 2]; tail_index += 1
        env = HomogeneousAirCombatEnv(env_config); observations, _ = env.reset(seed + episode, name, rear)
        returns = np.zeros(2); funnels = {team: new_funnel() for team in TEAMS}
        while True:
            actions = {"red_0": red_policy(observations["red_0"], env, "red"), "blue_0": blue_policy(observations["blue_0"], env, "blue")}
            observations, rewards, terminated, truncated, info = env.step(actions)
            returns += [rewards[agent] for agent in AGENT_IDS]
            for team, agent in zip(TEAMS, AGENT_IDS): update_funnel(funnels[team], info["geometries"][agent], env.config["combat"], info["attacks"][agent])
            if terminated or truncated: break
        for team in TEAMS: finish_funnel(funnels[team], info["termination_reason"], team)
        records.append({"scenario": name, "rear_team": rear, "outcome": info["outcome"], "reason": info["termination_reason"], "returns": returns, "length": info["step_count"], "funnels": funnels})
    return {"overall": summarize_competitive_records(records), "by_scenario": {name: summarize_competitive_records([row for row in records if row["scenario"] == name]) for name in SCENARIOS if any(row["scenario"] == name for row in records)}, "tail_rear_counts": {team: sum(row["rear_team"] == team for row in records) for team in TEAMS}}


def evaluate_competitive_match(red_actor: GaussianActor, blue_actor: GaussianActor, env_config: str | Path, episodes: int, device: torch.device, scenario: str = "all", seed: int = 10000) -> dict[str, Any]:
    red_actor.eval(); blue_actor.eval()
    result = _evaluate_policies(_actor_policy(red_actor, device), _actor_policy(blue_actor, device), env_config, episodes, scenario, seed)
    red_actor.train(); blue_actor.train(); return result


def evaluate_matchup(red_actor: GaussianActor, blue_actor: GaussianActor, env_config: str | Path, episodes: int, device: torch.device, matchup: str, scenario: str = "all", seed: int = 10000) -> dict[str, Any]:
    policies = {"red": _actor_policy(red_actor, device), "blue": _actor_policy(blue_actor, device)}
    if matchup == "self_play": pass
    elif matchup == "red_vs_zero": policies["blue"] = _rule_policy("zero")
    elif matchup == "red_vs_pursuit": policies["blue"] = _rule_policy("pursuit")
    elif matchup == "blue_vs_zero": policies["red"] = _rule_policy("zero")
    elif matchup == "blue_vs_pursuit": policies["red"] = _rule_policy("pursuit")
    else: raise ValueError(f"unknown matchup: {matchup}")
    return _evaluate_policies(policies["red"], policies["blue"], env_config, episodes, scenario, seed)
