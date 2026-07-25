"""Policy-centric alternating-freeze competitive PPO utilities (v6)."""
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
POLICIES = ("a", "b")
SCENARIOS = ("tail_chase", "offset_head_on", "crossing")
BOUNDARY_REASONS = {"altitude_boundary", "xy_boundary", "boundary"}
CHECKPOINT_VERSION = 6


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def other(value: str, values: tuple[str, str]) -> str:
    return values[1] if value == values[0] else values[0]


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
    dg = combat["attack_distance_min"] <= geometry.distance <= combat["attack_distance_max"]
    tg, ag = geometry.ata <= combat["attack_ata_max"], geometry.aa <= combat["attack_aa_max"]
    dv = max(combat["attack_distance_min"] - geometry.distance, geometry.distance - combat["attack_distance_max"], 0.0)
    av, aav = max(geometry.ata - combat["attack_ata_max"], 0.0), max(geometry.aa - combat["attack_aa_max"], 0.0)
    funnel["minimum_distance"] = min(funnel["minimum_distance"], geometry.distance)
    funnel["ever_within_4000m"] |= geometry.distance <= 4000.0
    funnel["ever_within_attack_distance"] |= dg
    funnel["ever_satisfy_ata"] |= tg
    funnel["ever_satisfy_aa"] |= ag
    funnel["ever_distance_and_ata"] |= dg and tg
    funnel["ever_distance_and_aa"] |= dg and ag
    funnel["ever_ata_and_aa"] |= tg and ag
    funnel["ever_full_attack_envelope"] |= dg and tg and ag
    funnel["ever_satisfy_attack_envelope"] = funnel["ever_full_attack_envelope"]
    funnel["attack_envelope_steps"] += int(dg and tg and ag)
    funnel["minimum_distance_violation"] = min(funnel["minimum_distance_violation"], dv)
    funnel["minimum_ata_violation"] = min(funnel["minimum_ata_violation"], av)
    funnel["minimum_aa_violation"] = min(funnel["minimum_aa_violation"], aav)
    funnel["minimum_combined_violation"] = min(funnel["minimum_combined_violation"], dv / combat["attack_distance_max"] + av / np.pi + aav / np.pi)
    if attacked and not (dg and tg and ag):
        raise AssertionError("an attack cannot occur outside the full attack envelope")


def finish_funnel(funnel: dict[str, Any], reason: str | None, team: str) -> None:
    funnel["kill"] = reason == f"{team}_kill"
    funnel["boundary"] = reason in BOUNDARY_REASONS
    funnel["max_steps"] = reason == "max_steps"
    funnel["collision"] = reason == "collision"


def _boundary_losers(reason: str | None, outcome: str | None) -> set[str]:
    if reason not in BOUNDARY_REASONS:
        return set()
    if outcome in TEAMS:
        return {other(outcome, TEAMS)}
    return set(TEAMS)


def _funnel_summary(records: list[dict[str, Any]], team: str) -> dict[str, float]:
    funnels = [row["funnels"][team] for row in records]
    if not funnels:
        return {}
    rate = lambda key: float(np.mean([bool(f[key]) for f in funnels]))
    envelopes = sum(bool(f["ever_full_attack_envelope"]) for f in funnels)
    kills = sum(bool(f["kill"]) for f in funnels)
    return {
        "within_4000_rate": rate("ever_within_4000m"), "attack_distance_entry_rate": rate("ever_within_attack_distance"),
        "ata_gate_rate": rate("ever_satisfy_ata"), "aa_gate_rate": rate("ever_satisfy_aa"),
        "distance_and_ata_entry_rate": rate("ever_distance_and_ata"), "distance_and_aa_entry_rate": rate("ever_distance_and_aa"),
        "ata_and_aa_entry_rate": rate("ever_ata_and_aa"), "full_attack_envelope_entry_rate": rate("ever_full_attack_envelope"),
        "attack_to_kill_conversion_rate": kills / envelopes if envelopes else 0.0, "kill_rate": kills / len(funnels),
        "mean_minimum_distance": float(np.mean([f["minimum_distance"] for f in funnels])),
        "mean_minimum_combined_violation": float(np.mean([f["minimum_combined_violation"] for f in funnels])),
    }


def summarize_competitive_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero episodes")
    n = len(records)
    count = lambda predicate: sum(1 for row in records if predicate(row))
    result: dict[str, Any] = {
        "episodes": n,
        "red_outcome_wins": count(lambda r: r["outcome"] == "red"),
        "blue_outcome_wins": count(lambda r: r["outcome"] == "blue"),
        "red_kills": count(lambda r: r["reason"] == "red_kill"),
        "blue_kills": count(lambda r: r["reason"] == "blue_kill"),
    }
    result["draws"] = n - result["red_outcome_wins"] - result["blue_outcome_wins"]
    result["red_outcome_win_rate"] = result["red_outcome_wins"] / n
    result["blue_outcome_win_rate"] = result["blue_outcome_wins"] / n
    result["red_kill_rate"] = result["red_kills"] / n
    result["blue_kill_rate"] = result["blue_kills"] / n
    result["draw_rate"] = result["draws"] / n
    result["non_draw_rate"] = 1.0 - result["draw_rate"]
    result["combat_decisive_rate"] = (result["red_kills"] + result["blue_kills"]) / n
    for team in TEAMS:
        losses = count(lambda r, t=team: t in _boundary_losers(r["reason"], r["outcome"]))
        result[f"{team}_boundary_losses"] = losses
        result[f"{team}_boundary_loss_rate"] = losses / n
    for reason in ("altitude_boundary", "xy_boundary", "collision", "mutual_kill", "max_steps"):
        result[f"{reason}_rate"] = count(lambda r, x=reason: r["reason"] == x) / n
    result["boundary_rate"] = count(lambda r: r["reason"] in BOUNDARY_REASONS) / n
    result["collision_count"] = count(lambda r: r["reason"] == "collision")
    result["mutual_kill_count"] = count(lambda r: r["reason"] == "mutual_kill")
    result["max_steps_count"] = count(lambda r: r["reason"] == "max_steps")
    result["mean_episode_length"] = float(np.mean([r["length"] for r in records]))
    result["red_mean_return"] = float(np.mean([r["returns"][0] for r in records]))
    result["blue_mean_return"] = float(np.mean([r["returns"][1] for r in records]))
    result["termination_reason_counts"] = {reason: count(lambda r, x=reason: r["reason"] == x) for reason in ("red_kill", "blue_kill", "mutual_kill", "collision", "altitude_boundary", "xy_boundary", "boundary", "max_steps")}
    result["red_funnel"], result["blue_funnel"] = _funnel_summary(records, "red"), _funnel_summary(records, "blue")

    for policy in POLICIES:
        controlled = [r for r in records if policy in (r.get("red_policy_id"), r.get("blue_policy_id"))]
        kills = sum(r.get("winner_policy") == policy and r["reason"] in {"red_kill", "blue_kill"} for r in controlled)
        boundary_losses = sum(r.get("loser_policy") == policy and r["reason"] in BOUNDARY_REASONS for r in controlled)
        result[f"policy_{policy}_kills"] = kills
        result[f"policy_{policy}_kill_rate"] = kills / len(controlled) if controlled else 0.0
        result[f"policy_{policy}_boundary_losses"] = boundary_losses
        result[f"policy_{policy}_boundary_loss_rate"] = boundary_losses / len(controlled) if controlled else 0.0
        role_rates = {}
        for team in TEAMS:
            subset = [r for r in controlled if r.get(f"policy_{policy}_team") == team]
            role_rates[team] = sum(r.get("winner_policy") == policy and r["reason"] == f"{team}_kill" for r in subset) / len(subset) if subset else 0.0
            result[f"policy_{policy}_as_{team}_kill_rate"] = role_rates[team]
        result[f"policy_{policy}_role_kill_gap"] = abs(role_rates["red"] - role_rates["blue"])
    result["min_policy_kill_rate"] = min(result["policy_a_kill_rate"], result["policy_b_kill_rate"])
    result["policy_kill_imbalance"] = abs(result["policy_a_kill_rate"] - result["policy_b_kill_rate"])
    result["paired_combat_decisive_rate"] = result["combat_decisive_rate"]
    return result


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero episodes")
    n = len(records)
    wins, losses = sum(r["result"] == "win" for r in records), sum(r["result"] == "loss" for r in records)
    return {"episodes": n, "wins": wins, "losses": losses, "draws": n - wins - losses, "win_rate": wins / n, "loss_rate": losses / n, "draw_rate": (n - wins - losses) / n}


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


class MAPPOTrainer:
    def __init__(self, env_config: str | Path, config: dict[str, Any]) -> None:
        self.env_config, self.config = str(env_config), deepcopy(config)
        t, n, e = self.config["training"], self.config["network"], self.config["experiment"]
        if t.get("training_mode") != "alternating_self_play":
            raise ValueError("training_mode must be alternating_self_play")
        random.seed(e["seed"]); torch.manual_seed(e["seed"])
        self.device = resolve_device(e["device"])
        self.num_envs, self.rollout_steps = int(t["num_envs"]), int(t["rollout_steps"])
        if t["minibatch_size"] > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size exceeds active transitions")
        self.policy_a_actor = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        self.policy_b_actor = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        self.policy_b_actor.load_state_dict(deepcopy(self.policy_a_actor.state_dict()))
        self.policy_a_critic = CentralizedCritic(14, n["hidden_dim"]).to(self.device)
        self.policy_b_critic = CentralizedCritic(14, n["hidden_dim"]).to(self.device)
        lr = t["learning_rate"]
        self.policy_a_actor_optimizer = torch.optim.Adam(self.policy_a_actor.parameters(), lr=lr)
        self.policy_b_actor_optimizer = torch.optim.Adam(self.policy_b_actor.parameters(), lr=lr)
        self.policy_a_critic_optimizer = torch.optim.Adam(self.policy_a_critic.parameters(), lr=lr)
        self.policy_b_critic_optimizer = torch.optim.Adam(self.policy_b_critic.parameters(), lr=lr)
        self.policy_a_behavior_actor, self.policy_b_behavior_actor = self._new_behavior_actor(), self._new_behavior_actor()
        self.policy_a_actor_history, self.policy_b_actor_history = [_cpu_state_dict(self.policy_a_actor)], [_cpu_state_dict(self.policy_b_actor)]
        initial = lambda: [{"generation": 0, "block_index": -1, "env_steps": 0, "active_policy": "initial"}]
        self.policy_a_generation_metadata, self.policy_b_generation_metadata = initial(), initial()
        self.history_selection_counts = {"forced_single_generation": 0, "sampled_latest": 0, "sampled_old": 0}
        self.block_history: list[dict[str, Any]] = []
        self.current_block_index: int | None = None
        self.current_active_policy: str | None = None
        self.current_opponent_policy: str | None = None
        self.current_opponent_generation: int | None = None
        self.current_opponent_is_latest: bool | None = None
        self.current_opponent_history_size = 0
        self.active_generation_before: int | None = None
        self.active_generation_after: int | None = None
        self._finished_blocks: set[int] = set()
        self.envs = [HomogeneousAirCombatEnv(self.env_config) for _ in range(self.num_envs)]
        self.buffer = MAPPOBuffer(self.rollout_steps, self.num_envs)
        self.rng = np.random.default_rng(e["seed"])
        self.opponent_rng = np.random.default_rng(e["seed"] + 1)
        self.env_steps = self.update_count = self.scenario_counter = 0
        self.scenario_counts = {name: 0 for name in SCENARIOS}
        self.active_team_counts = {team: 0 for team in TEAMS}
        self.tail_combo_counts = {f"{team}_{role}": 0 for team in TEAMS for role in ("rear", "front")}
        self.block_active_team_counts = {team: 0 for team in TEAMS}
        self.block_tail_combo_counts = {f"{team}_{role}": 0 for team in TEAMS for role in ("rear", "front")}
        self.block_episode_counter = 0
        self.tail_combo_counter = 0
        self.non_tail_tie_counter = 0
        self.current_observations: list[dict[str, np.ndarray]] = []
        self.current_scenarios: list[str] = []
        self.current_rear_teams: list[str | None] = []
        self.current_active_teams: list[str] = []
        self.current_policy_a_teams: list[str] = []
        self.current_policy_b_teams: list[str] = []
        self.episode_returns = np.zeros((self.num_envs, 2), dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.funnels = [{team: new_funnel() for team in TEAMS} for _ in self.envs]
        self.last_control_diagnostics = {team: [] for team in TEAMS}
        self.quick_best_score: tuple[float, ...] | None = None
        self.candidate_checkpoints: list[str] = []
        self.reset_environments()

    def _new_behavior_actor(self) -> GaussianActor:
        n = self.config["network"]
        actor = GaussianActor(14, 3, n["hidden_dim"], n["log_std_init"]).to(self.device)
        for parameter in actor.parameters(): parameter.requires_grad_(False)
        actor.eval()
        return actor

    def _actor(self, policy: str, behavior: bool = False) -> GaussianActor:
        return getattr(self, f"policy_{policy}_{'behavior_' if behavior else ''}actor")

    def _critic(self, policy: str) -> CentralizedCritic:
        return getattr(self, f"policy_{policy}_critic")

    def _history(self, policy: str) -> list[dict[str, torch.Tensor]]:
        return getattr(self, f"policy_{policy}_actor_history")

    def _metadata(self, policy: str) -> list[dict[str, Any]]:
        return getattr(self, f"policy_{policy}_generation_metadata")

    @property
    def block_env_steps(self) -> int:
        return int(self.config["training"]["alternating_block_env_steps"])

    @property
    def opponent_history_latest_probability(self) -> float:
        return float(self.config["training"].get("opponent_history_latest_probability", 0.7))

    def block_index(self) -> int:
        return self.env_steps // self.block_env_steps

    def active_policy(self) -> str:
        return POLICIES[self.block_index() % 2]

    def _select_opponent_generation(self, policy: str) -> tuple[int, bool]:
        history = self._history(policy)
        if len(history) == 1:
            self.history_selection_counts["forced_single_generation"] += 1
            return 0, True
        if float(self.opponent_rng.random()) < self.opponent_history_latest_probability:
            self.history_selection_counts["sampled_latest"] += 1
            return len(history) - 1, True
        self.history_selection_counts["sampled_old"] += 1
        return int(self.opponent_rng.integers(0, len(history) - 1)), False

    def configure_block_opponent(self, block_index: int | None = None, active_policy: str | None = None, force: bool = False) -> dict[str, Any]:
        block_index = self.block_index() if block_index is None else int(block_index)
        active_policy = self.active_policy() if active_policy is None else active_policy
        if active_policy not in POLICIES: raise ValueError("active_policy must be a or b")
        if self.current_block_index == block_index and not force: return self.current_opponent_info()
        preserve_initial_roles = self.env_steps == 0 and self.current_block_index is None and bool(self.current_observations)
        opponent = other(active_policy, POLICIES)
        generation, latest = self._select_opponent_generation(opponent)
        behavior = self._actor(opponent, behavior=True)
        behavior.load_state_dict(self._history(opponent)[generation]); behavior.eval()
        self.current_block_index, self.current_active_policy = block_index, active_policy
        self.current_opponent_policy, self.current_opponent_generation = opponent, generation
        self.current_opponent_is_latest, self.current_opponent_history_size = latest, len(self._history(opponent))
        self.active_generation_before, self.active_generation_after = len(self._history(active_policy)) - 1, None
        if not preserve_initial_roles:
            self.block_episode_counter = 0
            self.block_active_team_counts = {team: 0 for team in TEAMS}
            self.block_tail_combo_counts = {f"{team}_{role}": 0 for team in TEAMS for role in ("rear", "front")}
        info = self.current_opponent_info()
        self.block_history.append({**info, "block_index": block_index, "active_policy": active_policy, "start_env_steps": self.env_steps})
        return info

    def current_opponent_info(self) -> dict[str, Any]:
        return {"opponent_policy": self.current_opponent_policy, "opponent_generation": self.current_opponent_generation, "opponent_is_latest": self.current_opponent_is_latest, "opponent_history_size": self.current_opponent_history_size, "active_generation_before": self.active_generation_before, "active_generation_after": self.active_generation_after}

    def finish_block(self, active_policy: str | None = None, block_index: int | None = None) -> dict[str, Any]:
        active_policy = active_policy or self.current_active_policy
        block_index = self.current_block_index if block_index is None else block_index
        if active_policy is None or block_index is None: raise RuntimeError("block opponent must be configured")
        if block_index in self._finished_blocks: return self.current_opponent_info()
        history, metadata = self._history(active_policy), self._metadata(active_policy)
        generation = len(history)
        history.append(_cpu_state_dict(self._actor(active_policy)))
        metadata.append({"generation": generation, "block_index": int(block_index), "env_steps": int(self.env_steps), "active_policy": active_policy})
        self.active_generation_after = generation; self._finished_blocks.add(int(block_index))
        if self.block_history and self.block_history[-1]["block_index"] == block_index:
            self.block_history[-1].update({"end_env_steps": self.env_steps, "active_generation_after": generation, "active_team_counts": deepcopy(self.block_active_team_counts), "tail_combo_counts": deepcopy(self.block_tail_combo_counts)})
        return self.current_opponent_info()

    def _next_reset(self, env: HomogeneousAirCombatEnv) -> tuple[dict[str, np.ndarray], str, str | None, str]:
        scenario = SCENARIOS[self.scenario_counter % len(SCENARIOS)]
        self.scenario_counter += 1; self.scenario_counts[scenario] += 1
        rear = None
        if scenario == "tail_chase":
            active_team, rear_role = (("red", "rear"), ("red", "front"), ("blue", "rear"), ("blue", "front"))[self.tail_combo_counter % 4]
            self.tail_combo_counter += 1
            rear = active_team if rear_role == "rear" else other(active_team, TEAMS)
            combo = f"{active_team}_{rear_role}"
            self.tail_combo_counts[combo] += 1; self.block_tail_combo_counts[combo] += 1
        else:
            red_count, blue_count = self.block_active_team_counts["red"], self.block_active_team_counts["blue"]
            if red_count != blue_count:
                active_team = "red" if red_count < blue_count else "blue"
            else:
                next_scenario = SCENARIOS[self.scenario_counter % len(SCENARIOS)]
                if next_scenario == "tail_chase":
                    next_tail_team = (("red", "rear"), ("red", "front"), ("blue", "rear"), ("blue", "front"))[self.tail_combo_counter % 4][0]
                    active_team = other(next_tail_team, TEAMS)
                else:
                    active_team = TEAMS[self.non_tail_tie_counter % 2]; self.non_tail_tie_counter += 1
        self.block_episode_counter += 1; self.active_team_counts[active_team] += 1; self.block_active_team_counts[active_team] += 1
        observation, _ = env.reset(int(self.rng.integers(2**31 - 1)), scenario, rear)
        return observation, scenario, rear, active_team

    def reset_environments(self) -> None:
        self.current_observations, self.current_scenarios, self.current_rear_teams = [], [], []
        self.current_active_teams, self.current_policy_a_teams, self.current_policy_b_teams = [], [], []
        for env in self.envs:
            observation, scenario, rear, active_team = self._next_reset(env)
            a_team = active_team if self.current_active_policy != "b" else other(active_team, TEAMS)
            self.current_observations.append(observation); self.current_scenarios.append(scenario); self.current_rear_teams.append(rear)
            self.current_active_teams.append(active_team); self.current_policy_a_teams.append(a_team); self.current_policy_b_teams.append(other(a_team, TEAMS))
        self.episode_returns.fill(0); self.episode_lengths.fill(0)
        self.funnels = [{team: new_funnel() for team in TEAMS} for _ in self.envs]

    def _episode_record(self, index: int, info: dict[str, Any]) -> dict[str, Any]:
        a_team, b_team = self.current_policy_a_teams[index], self.current_policy_b_teams[index]
        winner_policy = "a" if info["outcome"] == a_team else "b" if info["outcome"] == b_team else None
        loser_policy = other(winner_policy, POLICIES) if winner_policy in POLICIES else None
        return {"returns": self.episode_returns[index].copy(), "length": int(self.episode_lengths[index]), "outcome": info["outcome"], "reason": info["termination_reason"], "termination_reason": info["termination_reason"], "scenario": self.current_scenarios[index], "rear_team": self.current_rear_teams[index], "funnels": deepcopy(self.funnels[index]), "policy_a_team": a_team, "policy_b_team": b_team, "red_policy_id": "a" if a_team == "red" else "b", "blue_policy_id": "a" if a_team == "blue" else "b", "active_policy": self.current_active_policy, "active_policy_team": self.current_active_teams[index], "opponent_policy": self.current_opponent_policy, "opponent_generation": self.current_opponent_generation, "winner_policy": winner_policy, "loser_policy": loser_policy}

    def collect_rollout(self, remaining_env_steps: int | None = None) -> list[dict[str, Any]]:
        self.configure_block_opponent()
        steps = self.rollout_steps if remaining_env_steps is None else min(self.rollout_steps, remaining_env_steps // self.num_envs)
        if steps <= 0: raise ValueError("remaining_env_steps must contain one full vector step")
        if self.buffer.rollout_steps != steps: self.buffer = MAPPOBuffer(steps, self.num_envs)
        self.buffer.clear(); completed: list[dict[str, Any]] = []
        self.last_control_diagnostics = {team: [] for team in TEAMS}
        active = self.current_active_policy
        assert active in POLICIES
        active_actor, opponent_actor, critic = self._actor(active), self._actor(other(active, POLICIES), behavior=True), self._critic(active)
        for _ in range(steps):
            active_obs = np.asarray([row[f"{team}_0"] for row, team in zip(self.current_observations, self.current_active_teams)], np.float32)
            opponent_obs = np.asarray([row[f"{other(team, TEAMS)}_0"] for row, team in zip(self.current_observations, self.current_active_teams)], np.float32)
            states = np.asarray([env.global_state(team) for env, team in zip(self.envs, self.current_active_teams)], np.float32)
            with torch.no_grad():
                active_actions, logs = active_actor.sample_action(torch.as_tensor(active_obs, device=self.device))
                opponent_actions, _ = opponent_actor.sample_action(torch.as_tensor(opponent_obs, device=self.device))
                values = critic(torch.as_tensor(states, device=self.device))
            aa, oa = active_actions.cpu().numpy(), opponent_actions.cpu().numpy()
            rewards = np.zeros(self.num_envs, np.float32); dones = np.zeros(self.num_envs, bool); next_obs = []
            active_team_codes = np.asarray([TEAMS.index(team) for team in self.current_active_teams], np.int8)
            for index, env in enumerate(self.envs):
                active_team = self.current_active_teams[index]; opponent_team = other(active_team, TEAMS)
                actions = {f"{active_team}_0": aa[index], f"{opponent_team}_0": oa[index]}
                observation, reward, terminated, truncated, info = env.step(actions)
                color_rewards = np.asarray([reward[a] for a in AGENT_IDS])
                rewards[index] = reward[f"{active_team}_0"]
                self.episode_returns[index] += color_rewards; self.episode_lengths[index] += 1
                for team, agent in zip(TEAMS, AGENT_IDS):
                    update_funnel(self.funnels[index][team], info["geometries"][agent], env.config["combat"], info["attacks"][agent])
                    self.last_control_diagnostics[team].append(info["control_diagnostics"][agent])
                done = terminated or truncated; dones[index] = done
                if done:
                    for team in TEAMS: finish_funnel(self.funnels[index][team], info["termination_reason"], team)
                    completed.append(self._episode_record(index, info))
                    self.episode_returns[index] = 0; self.episode_lengths[index] = 0
                    self.funnels[index] = {team: new_funnel() for team in TEAMS}
                    observation, scenario, rear, active_team = self._next_reset(env)
                    a_team = active_team if active == "a" else other(active_team, TEAMS)
                    self.current_scenarios[index], self.current_rear_teams[index], self.current_active_teams[index] = scenario, rear, active_team
                    self.current_policy_a_teams[index], self.current_policy_b_teams[index] = a_team, other(a_team, TEAMS)
                next_obs.append(observation)
            self.buffer.add(active_obs, states, aa, logs.cpu().numpy(), rewards, values.cpu().numpy(), dones, active_team_codes)
            self.current_observations = next_obs; self.env_steps += self.num_envs
        with torch.no_grad():
            states = np.asarray([env.global_state(team) for env, team in zip(self.envs, self.current_active_teams)], np.float32)
            last_values = critic(torch.as_tensor(states, device=self.device)).cpu().numpy()
        t = self.config["training"]
        self.buffer.compute_returns_and_advantages(last_values, t["gamma"], t["gae_lambda"])
        return completed

    @staticmethod
    def _finite(label: str, *values: torch.Tensor) -> None:
        if not all(torch.isfinite(v).all() for v in values): raise FloatingPointError(f"non-finite {label}")

    def _update_actor(self, policy: str) -> dict[str, float]:
        actor, optimizer, t = self._actor(policy), getattr(self, f"policy_{policy}_actor_optimizer"), self.config["training"]
        obs = torch.as_tensor(self.buffer.observations.reshape(-1, 14), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, 3), device=self.device)
        old = torch.as_tensor(self.buffer.log_probs.reshape(-1), device=self.device)
        advantage = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8); data = []
        for _ in range(t["ppo_epochs"]):
            order = self.rng.permutation(len(obs))
            for start in range(0, len(obs), t["minibatch_size"]):
                idx = torch.as_tensor(order[start:start + t["minibatch_size"]], device=self.device)
                new, entropy = actor.evaluate_actions(obs[idx], actions[idx]); log_ratio = new - old[idx]; ratio = log_ratio.exp()
                policy_loss = -torch.minimum(ratio * advantage[idx], ratio.clamp(1-t["clip_coef"], 1+t["clip_coef"]) * advantage[idx]).mean()
                loss = policy_loss - t["entropy_coef"] * entropy.mean()
                optimizer.zero_grad(); loss.backward(); grad = nn.utils.clip_grad_norm_(actor.parameters(), t["max_grad_norm"])
                self._finite(policy, loss, grad); optimizer.step()
                data.append((policy_loss.item(), entropy.mean().item(), (((ratio-1)-log_ratio).mean()).item(), ((ratio-1).abs()>t["clip_coef"]).float().mean().item(), float(grad)))
        v = np.asarray(data)
        return {"policy_loss": float(v[:,0].mean()), "entropy": float(v[:,1].mean()), "approx_kl": float(v[:,2].mean()), "clip_fraction": float(v[:,3].mean()), "grad_norm": float(v[:,4].mean()), "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std(unbiased=False)), "advantage_nonzero_rate": float((advantage.abs()>1e-8).float().mean())}

    def _update_critic(self, policy: str) -> dict[str, float]:
        critic, optimizer, t = self._critic(policy), getattr(self, f"policy_{policy}_critic_optimizer"), self.config["training"]
        states = torch.as_tensor(self.buffer.global_states.reshape(-1,14), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device); losses=[]; grads=[]
        for _ in range(t["ppo_epochs"]):
            order = self.rng.permutation(len(states))
            for start in range(0,len(states),t["minibatch_size"]):
                idx=torch.as_tensor(order[start:start+t["minibatch_size"]],device=self.device)
                loss=((critic(states[idx])-returns[idx])**2).mean()
                optimizer.zero_grad(); (t["value_loss_coef"]*loss).backward(); grad=nn.utils.clip_grad_norm_(critic.parameters(),t["max_grad_norm"])
                self._finite(f"{policy} critic",loss,grad); optimizer.step(); losses.append(loss.item()); grads.append(float(grad))
        return {"value_loss":float(np.mean(losses)),"critic_grad_norm":float(np.mean(grads))}

    def update(self, active_override: str | None = None) -> dict[str, Any]:
        active = active_override or self.current_active_policy or self.active_policy()
        if active not in POLICIES: raise ValueError("active policy must be a or b")
        values = self._update_actor(active) | self._update_critic(active)
        metrics: dict[str, Any] = {"active_policy": active}
        for policy in POLICIES:
            for key, value in values.items(): metrics[f"policy_{policy}_{key}"] = value if policy == active else np.nan
        self.update_count += 1; metrics.update(self.current_opponent_info()); return metrics

    def training_signature(self) -> dict[str, Any]:
        t, n, env = self.config["training"], self.config["network"], self.envs[0].config
        signature_config = deepcopy(self.config)
        signature_config["training"].pop("total_env_steps", None)
        signature_config["experiment"].pop("device", None)
        signature_config["experiment"].pop("output_dir", None)
        return {"network": deepcopy(n), "ppo": {k: t[k] for k in ("learning_rate","gamma","gae_lambda","clip_coef","entropy_coef","value_loss_coef","max_grad_norm","ppo_epochs","minibatch_size")}, "num_envs": self.num_envs, "rollout_steps": self.rollout_steps, "alternating_block_env_steps": self.block_env_steps, "opponent_history_latest_probability": self.opponent_history_latest_probability, "reward_mode": env["combat"].get("reward_mode","coupled_difference"), "environment": deepcopy(env), "config": signature_config}

    def save_checkpoint(self, path: str | Path) -> None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        data: dict[str,Any]={"checkpoint_version":CHECKPOINT_VERSION,"config":self.config,"training_signature":self.training_signature(),"environment_steps":self.env_steps,"env_steps":self.env_steps,"update":self.update_count,"training_mode":"alternating_self_play","active_policy":self.active_policy(),"current_active_policy":self.current_active_policy,"current_opponent_policy":self.current_opponent_policy,"current_opponent_generation":self.current_opponent_generation,"current_opponent_is_latest":self.current_opponent_is_latest,"current_opponent_history_size":self.current_opponent_history_size,"active_generation_before":self.active_generation_before,"active_generation_after":self.active_generation_after,"current_block_index":self.current_block_index,"finished_blocks":sorted(self._finished_blocks),"history_selection_counts":self.history_selection_counts,"block_history":self.block_history,"scenario_counter":self.scenario_counter,"scenario_counts":self.scenario_counts,"active_team_counts":self.active_team_counts,"tail_combo_counts":self.tail_combo_counts,"block_episode_counter":self.block_episode_counter,"tail_combo_counter":self.tail_combo_counter,"non_tail_tie_counter":self.non_tail_tie_counter,"block_active_team_counts":self.block_active_team_counts,"block_tail_combo_counts":self.block_tail_combo_counts,"current_active_teams":self.current_active_teams,"current_policy_a_teams":self.current_policy_a_teams,"current_policy_b_teams":self.current_policy_b_teams,"current_scenarios":self.current_scenarios,"current_rear_teams":self.current_rear_teams,"quick_best_score":self.quick_best_score,"candidate_checkpoints":self.candidate_checkpoints,"python_random_state":random.getstate(),"numpy_rng_state":self.rng.bit_generator.state,"opponent_numpy_rng_state":self.opponent_rng.bit_generator.state,"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        for policy in POLICIES:
            for kind in ("actor","critic"): data[f"policy_{policy}_{kind}"]=getattr(self,f"policy_{policy}_{kind}").state_dict()
            for kind in ("actor_optimizer","critic_optimizer"): data[f"policy_{policy}_{kind}"]=getattr(self,f"policy_{policy}_{kind}").state_dict()
            data[f"policy_{policy}_behavior_actor"]=self._actor(policy,True).state_dict()
            data[f"policy_{policy}_actor_history"]=self._history(policy)
            data[f"policy_{policy}_generation_metadata"]=self._metadata(policy)
        torch.save(data,path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool=True) -> None:
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise RuntimeError("v6 checkpoint required; v5 and earlier bind policies to colors and cannot be resumed or migrated implicitly")
        if checkpoint.get("training_signature") != self.training_signature():
            raise RuntimeError("training signature mismatch; only total_env_steps, device, and output_dir may change on resume")
        for policy in POLICIES:
            for kind in ("actor","critic"): getattr(self,f"policy_{policy}_{kind}").load_state_dict(checkpoint[f"policy_{policy}_{kind}"])
            if load_optimizers:
                for kind in ("actor_optimizer","critic_optimizer"): getattr(self,f"policy_{policy}_{kind}").load_state_dict(checkpoint[f"policy_{policy}_{kind}"])
            self._actor(policy,True).load_state_dict(checkpoint[f"policy_{policy}_behavior_actor"])
            setattr(self,f"policy_{policy}_actor_history",[{k:v.detach().cpu().clone() for k,v in state.items()} for state in checkpoint[f"policy_{policy}_actor_history"]])
            setattr(self,f"policy_{policy}_generation_metadata",checkpoint[f"policy_{policy}_generation_metadata"])
        for key in ("current_active_policy","current_opponent_policy","current_opponent_generation","current_opponent_is_latest","current_opponent_history_size","active_generation_before","active_generation_after","current_block_index","scenario_counter","block_episode_counter","quick_best_score"):
            setattr(self,key,checkpoint.get(key))
        self.env_steps=int(checkpoint["environment_steps"]); self.update_count=int(checkpoint["update"])
        self._finished_blocks=set(checkpoint.get("finished_blocks",[])); self.history_selection_counts=dict(checkpoint["history_selection_counts"])
        self.block_history=list(checkpoint.get("block_history",[])); self.scenario_counts=dict(checkpoint["scenario_counts"])
        self.active_team_counts=dict(checkpoint["active_team_counts"]); self.tail_combo_counts=dict(checkpoint["tail_combo_counts"])
        self.tail_combo_counter=int(checkpoint["tail_combo_counter"]); self.non_tail_tie_counter=int(checkpoint["non_tail_tie_counter"])
        self.block_active_team_counts=dict(checkpoint["block_active_team_counts"]); self.block_tail_combo_counts=dict(checkpoint["block_tail_combo_counts"]); self.candidate_checkpoints=list(checkpoint.get("candidate_checkpoints",[]))
        random.setstate(checkpoint["python_random_state"]); self.rng.bit_generator.state=checkpoint["numpy_rng_state"]; self.opponent_rng.bit_generator.state=checkpoint["opponent_numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        if torch.cuda.is_available() and checkpoint["torch_cuda_rng_state"] is not None: torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
        # Resume begins from a new episode batch while retaining deterministic scheduler state.
        self.reset_environments()


def _actor_policy(actor: GaussianActor, device: torch.device) -> Callable:
    def act(observation: np.ndarray, _env: HomogeneousAirCombatEnv, _team: str) -> np.ndarray:
        with torch.no_grad(): return actor.deterministic_action(torch.as_tensor(observation,dtype=torch.float32,device=device)[None]).squeeze(0).cpu().numpy()
    return act


def _rule_policy(kind: str) -> Callable:
    def act(_observation: np.ndarray, env: HomogeneousAirCombatEnv, team: str) -> np.ndarray:
        if kind=="zero": return np.zeros(3,dtype=np.float32)
        own,target=env.aircraft if team=="red" else reversed(env.aircraft); cfg=env.config["action"]
        return PurePursuitPolicy(cfg["delta_yaw_max"],cfg["delta_pitch_max"],cfg["delta_speed_max"]).action(own,target)
    return act


def _run_episode(red_policy: Callable, blue_policy: Callable, red_id: str, blue_id: str, env_config: str|Path, seed: int, scenario: str, rear: str|None) -> dict[str,Any]:
    env=HomogeneousAirCombatEnv(env_config); observations,_=env.reset(seed,scenario,rear); returns=np.zeros(2); funnels={t:new_funnel() for t in TEAMS}
    while True:
        actions={"red_0":red_policy(observations["red_0"],env,"red"),"blue_0":blue_policy(observations["blue_0"],env,"blue")}
        observations,rewards,terminated,truncated,info=env.step(actions); returns += [rewards[a] for a in AGENT_IDS]
        for team,agent in zip(TEAMS,AGENT_IDS): update_funnel(funnels[team],info["geometries"][agent],env.config["combat"],info["attacks"][agent])
        if terminated or truncated: break
    for team in TEAMS: finish_funnel(funnels[team],info["termination_reason"],team)
    winner_policy=red_id if info["outcome"]=="red" else blue_id if info["outcome"]=="blue" else None
    loser_policy=blue_id if winner_policy==red_id else red_id if winner_policy==blue_id else None
    return {"scenario":scenario,"rear_team":rear,"seed":seed,"outcome":info["outcome"],"reason":info["termination_reason"],"termination_reason":info["termination_reason"],"returns":returns,"length":info["step_count"],"funnels":funnels,"red_policy_id":red_id,"blue_policy_id":blue_id,"policy_a_team":"red" if red_id=="a" else "blue" if blue_id=="a" else None,"policy_b_team":"red" if red_id=="b" else "blue" if blue_id=="b" else None,"winner_policy":winner_policy,"loser_policy":loser_policy}


def evaluate_paired_policies(policy_a: Callable, policy_b: Callable, env_config: str|Path, episodes: int, scenario: str="all", seed: int=10000, policy_a_id: str="a", policy_b_id: str="b") -> dict[str,Any]:
    if episodes<=0 or episodes%2: raise ValueError("paired evaluation --episodes must be a positive even number")
    records=[]; tail_index=0
    for pair in range(episodes//2):
        name=SCENARIOS[pair%3] if scenario=="all" else scenario
        rear=None
        if name=="tail_chase": rear=TEAMS[tail_index%2]; tail_index+=1
        episode_seed=seed+pair
        records.append(_run_episode(policy_a,policy_b,policy_a_id,policy_b_id,env_config,episode_seed,name,rear))
        records.append(_run_episode(policy_b,policy_a,policy_b_id,policy_a_id,env_config,episode_seed,name,rear))
    overall=summarize_competitive_records(records)
    by={name:summarize_competitive_records([r for r in records if r["scenario"]==name]) for name in SCENARIOS if any(r["scenario"]==name for r in records)}
    overall["worst_scenario_combat_decisive_rate"]=min(v["combat_decisive_rate"] for v in by.values())
    return {"overall":overall,"by_scenario":by,"tail_rear_counts":{t:sum(r["rear_team"]==t for r in records) for t in TEAMS},"records":records}


def evaluate_competitive_match(policy_a_actor: GaussianActor, policy_b_actor: GaussianActor, env_config: str|Path, episodes: int, device: torch.device, scenario: str="all", seed: int=10000) -> dict[str,Any]:
    policy_a_actor.eval(); policy_b_actor.eval()
    result=evaluate_paired_policies(_actor_policy(policy_a_actor,device),_actor_policy(policy_b_actor,device),env_config,episodes,scenario,seed)
    policy_a_actor.train(); policy_b_actor.train(); return result


def evaluate_matchup(policy_a_actor: GaussianActor, policy_b_actor: GaussianActor, env_config: str|Path, episodes: int, device: torch.device, matchup: str, scenario: str="all", seed: int=10000) -> dict[str,Any]:
    actors={"a":_actor_policy(policy_a_actor,device),"b":_actor_policy(policy_b_actor,device)}
    if matchup=="a_vs_b": left,right,left_id,right_id=actors["a"],actors["b"],"a","b"; learned=None
    elif matchup in {"a_vs_zero","a_vs_pursuit","b_vs_zero","b_vs_pursuit"}:
        learned=matchup[0]; kind=matchup.split("_vs_")[1]; left,right,left_id,right_id=actors[learned],_rule_policy(kind),learned,kind
    else: raise ValueError(f"unknown matchup: {matchup}")
    result=evaluate_paired_policies(left,right,env_config,episodes,scenario,seed,left_id,right_id); result["matchup"]=matchup; result["learned_policy"]=learned
    if learned:
        o=result["overall"]
        result.update({"learned_policy_kills":o[f"policy_{learned}_kills"],"learned_policy_kill_rate":o[f"policy_{learned}_kill_rate"],"learned_policy_boundary_losses":o[f"policy_{learned}_boundary_losses"],"learned_policy_boundary_loss_rate":o[f"policy_{learned}_boundary_loss_rate"],"learned_as_red_kill_rate":o[f"policy_{learned}_as_red_kill_rate"],"learned_as_blue_kill_rate":o[f"policy_{learned}_as_blue_kill_rate"],"learned_role_kill_gap":o[f"policy_{learned}_role_kill_gap"]})
    return result


def competitive_score(evaluation: dict[str,Any]) -> tuple[float,...]:
    o=evaluation["overall"]
    worst=o.get("worst_scenario_combat_decisive_rate")
    if worst is None: worst=min(v["combat_decisive_rate"] for v in evaluation["by_scenario"].values())
    return (float(worst),float(o["min_policy_kill_rate"]),float(o["paired_combat_decisive_rate"]),-float(max(o["policy_a_boundary_loss_rate"],o["policy_b_boundary_loss_rate"])),-float(max(o["policy_a_role_kill_gap"],o["policy_b_role_kill_gap"])),-float(o["collision_rate"]))
