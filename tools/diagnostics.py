"""Read-only diagnostics and orchestration helpers for learnability calibration.

Nothing in this module changes the environment's scientific semantics.  It only
observes trainer rollouts and standalone deterministic evaluation episodes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from env.geometry import compute_pairwise_geometry
from algorithm.common.buffer import RolloutBuffer
from env.mavuav import (
    BLUE_IDS, ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS,
    HeterogeneousMAVUAVAirCombatEnv,
)
from env.reward import situation_reward

AGENT_NAMES = ("MAV", "UAV1", "UAV2")
ACTION_NAMES = ("ux", "uy", "uz")
OBSERVATION_FEATURES = (
    "self_x", "self_y", "self_altitude", "self_speed", "self_theta", "self_psi", "self_alive",
    "self_type_MAV", "self_type_UAV", "self_type_Blue", "time_fraction",
    "friend1_dx", "friend1_dy", "friend1_dh", "friend1_distance", "friend1_dvx", "friend1_dvy", "friend1_dvh", "friend1_alive", "friend1_type_MAV", "friend1_type_UAV", "friend1_type_Blue",
    "friend2_dx", "friend2_dy", "friend2_dh", "friend2_distance", "friend2_dvx", "friend2_dvy", "friend2_dvh", "friend2_alive", "friend2_type_MAV", "friend2_type_UAV", "friend2_type_Blue",
    "blue1_dx", "blue1_dy", "blue1_dh", "blue1_distance", "blue1_dvx", "blue1_dvy", "blue1_dvh", "blue1_ata", "blue1_aa", "blue1_alive", "blue1_direct_visible", "blue1_datalink_visible", "blue1_own_attack_streak", "blue1_killed_by_red",
    "blue2_dx", "blue2_dy", "blue2_dh", "blue2_distance", "blue2_dvx", "blue2_dvy", "blue2_dvh", "blue2_ata", "blue2_aa", "blue2_alive", "blue2_direct_visible", "blue2_datalink_visible", "blue2_own_attack_streak", "blue2_killed_by_red",
)
OBSERVATION_GROUPS: dict[str, tuple[int, ...]] = {
    "self_position": (0, 1), "self_altitude": (2,), "self_speed": (3,),
    "self_theta_psi": (4, 5), "self_alive_type": (6, 7, 8, 9), "time_fraction": (10,),
    "friendly_relative_position": (11, 12, 13, 22, 23, 24),
    "friendly_distance": (14, 25),
    "friendly_relative_velocity": (15, 16, 17, 26, 27, 28),
    "friendly_alive_type": (18, 19, 20, 21, 29, 30, 31, 32),
    "enemy_relative_position": (33, 34, 35, 47, 48, 49),
    "enemy_distance": (36, 50), "enemy_relative_velocity": (37, 38, 39, 51, 52, 53),
    "enemy_ata": (40, 54), "enemy_aa": (41, 55), "enemy_alive": (42, 56),
    "enemy_visibility": (43, 44, 57, 58),
    "enemy_own_attack_streak": (45, 59), "enemy_killed_by_red": (46, 60),
}


def fixed_evaluation_seeds(count: int, start: int = 1000) -> list[int]:
    """Return the same contiguous evaluation seeds at every checkpoint."""
    if count <= 0:
        raise ValueError("evaluation episode count must be positive")
    return list(range(int(start), int(start) + int(count)))


def _statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {key: 0.0 for key in ("mean", "std", "min", "max", "p01", "p99", "mean_abs", "saturation_rate", "fraction_abs_le_0p05")}
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("diagnostic input contains NaN or Inf")
    return {
        "mean": float(values.mean()), "std": float(values.std()),
        "min": float(values.min()), "max": float(values.max()),
        "p01": float(np.quantile(values, 0.01)), "p99": float(np.quantile(values, 0.99)),
        "mean_abs": float(np.abs(values).mean()),
        "saturation_rate": float((np.abs(values) >= 0.95).mean()),
        "fraction_abs_le_0p05": float((np.abs(values) <= 0.05).mean()),
    }


@dataclass
class TrainingDiagnostics:
    """Cumulative training-rollout observations and actually executed actions."""

    max_observation_samples: int = 200_000
    observation_batches: list[np.ndarray] = field(default_factory=list)
    observation_sample_count: int = 0
    observation_feature_statistics: list[dict[str, float]] | None = None
    observation_group_statistics: dict[str, dict[str, float]] | None = None
    action_count: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.int64))
    action_sum: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    action_sum_sq: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    action_abs_sum: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    action_saturation_count: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.int64))

    def _observe_actions(self, actions: np.ndarray, active_masks: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 3, 3)
        masks = np.asarray(active_masks, dtype=np.float32).reshape(-1, 3) > 0.5
        for agent_index in range(3):
            values = np.asarray(actions[masks[:, agent_index], agent_index, :], dtype=np.float64)
            if not len(values):
                continue
            self.action_count[agent_index] += len(values)
            self.action_sum[agent_index] += values.sum(axis=0)
            self.action_sum_sq[agent_index] += np.square(values).sum(axis=0)
            self.action_abs_sum[agent_index] += np.abs(values).sum(axis=0)
            self.action_saturation_count[agent_index] += (np.abs(values) >= 0.95).sum(axis=0)

    def _freeze_observations(self) -> None:
        """Keep exact requested summaries and release the capped raw sample."""
        if self.observation_feature_statistics is not None or not self.observation_batches:
            return
        observations = np.concatenate(self.observation_batches, axis=0)
        self.observation_feature_statistics = [
            _statistics(observations[:, index]) for index in range(len(OBSERVATION_FEATURES))
        ]
        self.observation_group_statistics = {
            name: _statistics(observations[:, indices]) for name, indices in OBSERVATION_GROUPS.items()
        }
        self.observation_batches.clear()

    def observe_rollout(self, buffer: RolloutBuffer) -> None:
        observations = np.asarray(buffer.observations, dtype=np.float32).reshape(-1, OBS_DIM)
        remaining = self.max_observation_samples - self.observation_sample_count
        if remaining > 0:
            kept = observations[:remaining].copy()
            self.observation_batches.append(kept)
            self.observation_sample_count += len(kept)
            if self.observation_sample_count >= self.max_observation_samples:
                self._freeze_observations()
        self._observe_actions(buffer.actions, buffer.active_masks)

    def observation_rows(self, algorithm: str, seed: int, sampled_steps: int) -> list[dict[str, Any]]:
        if self.observation_sample_count == 0:
            return []
        if self.observation_feature_statistics is None:
            observations = np.concatenate(self.observation_batches, axis=0)
            feature_statistics = [_statistics(observations[:, index]) for index in range(len(OBSERVATION_FEATURES))]
            group_statistics = {name: _statistics(observations[:, indices]) for name, indices in OBSERVATION_GROUPS.items()}
        else:
            feature_statistics = self.observation_feature_statistics
            group_statistics = self.observation_group_statistics or {}
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(OBSERVATION_FEATURES):
            rows.append({"algorithm": algorithm, "seed": seed, "sampled_steps": sampled_steps, "row_type": "feature", "feature": name, "indices": str(index), "samples": self.observation_sample_count, **feature_statistics[index]})
        for name, indices in OBSERVATION_GROUPS.items():
            rows.append({"algorithm": algorithm, "seed": seed, "sampled_steps": sampled_steps, "row_type": "group", "feature": name, "indices": ",".join(map(str, indices)), "samples": self.observation_sample_count * len(indices), **group_statistics[name]})
        return rows

    def action_rows(self, algorithm: str, seed: int, sampled_steps: int) -> list[dict[str, Any]]:
        if not np.any(self.action_count):
            return []
        rows = []
        for agent_index, agent_name in enumerate(AGENT_NAMES):
            for action_index, action_name in enumerate(ACTION_NAMES):
                count = int(self.action_count[agent_index, action_index])
                if count:
                    mean = self.action_sum[agent_index, action_index] / count
                    variance = max(self.action_sum_sq[agent_index, action_index] / count - mean * mean, 0.0)
                    mean_abs = self.action_abs_sum[agent_index, action_index] / count
                    saturation_rate = self.action_saturation_count[agent_index, action_index] / count
                else:
                    mean = variance = mean_abs = saturation_rate = 0.0
                rows.append({"algorithm": algorithm, "seed": seed, "sampled_steps": sampled_steps, "agent": agent_name, "action_dimension": action_name, "samples": count, "mean": float(mean), "std": float(np.sqrt(variance)), "mean_abs": float(mean_abs), "saturation_rate": float(saturation_rate)})
        return rows

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "training_diagnostics_v2",
            "max_observation_samples": self.max_observation_samples,
            "observation_batches": self.observation_batches,
            "observation_sample_count": self.observation_sample_count,
            "observation_feature_statistics": self.observation_feature_statistics,
            "observation_group_statistics": self.observation_group_statistics,
            "action_count": self.action_count,
            "action_sum": self.action_sum,
            "action_sum_sq": self.action_sum_sq,
            "action_abs_sum": self.action_abs_sum,
            "action_saturation_count": self.action_saturation_count,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "TrainingDiagnostics":
        result = cls(int(state["max_observation_samples"]))
        result.observation_batches = [np.asarray(batch, dtype=np.float32) for batch in state.get("observation_batches", [])]
        result.observation_sample_count = int(state["observation_sample_count"])
        if state.get("format") == "training_diagnostics_v2":
            feature_stats = state.get("observation_feature_statistics")
            group_stats = state.get("observation_group_statistics")
            result.observation_feature_statistics = list(feature_stats) if feature_stats is not None else None
            result.observation_group_statistics = dict(group_stats) if group_stats is not None else None
            for name in ("action_count", "action_sum", "action_sum_sq", "action_abs_sum", "action_saturation_count"):
                dtype = np.int64 if name in ("action_count", "action_saturation_count") else np.float64
                setattr(result, name, np.asarray(state[name], dtype=dtype).copy())
        else:
            # Version 1 checkpoints stored every raw action and mask.  Aggregate
            # them once on load so resumed training immediately uses compact state.
            for actions, masks in zip(state.get("action_batches", []), state.get("active_mask_batches", [])):
                result._observe_actions(actions, masks)
        if result.observation_sample_count >= result.max_observation_samples:
            result._freeze_observations()
        return result


def target_proxies(env: HeterogeneousMAVUAVAirCombatEnv) -> dict[str, str]:
    """Read-only best-situation target proxy; never feeds policy or reward."""
    alive_blue = [bid for bid in BLUE_IDS if env.entities[bid].state.alive]
    proxies: dict[str, str] = {}
    if not alive_blue:
        return proxies
    for aid in RED_IDS:
        if env.entities[aid].state.alive:
            proxies[aid] = max(alive_blue, key=lambda bid: situation_reward(env.entities[aid].state, env.entities[bid].state))
    return proxies


def _minimum_distances(env: HeterogeneousMAVUAVAirCombatEnv) -> tuple[float, float]:
    cross = [compute_pairwise_geometry(env.entities[r].state, env.entities[b].state).distance for r in RED_IDS for b in BLUE_IDS if env.entities[r].state.alive and env.entities[b].state.alive]
    friendly = [compute_pairwise_geometry(env.entities[RED_IDS[i]].state, env.entities[RED_IDS[j]].state).distance for i in range(3) for j in range(i + 1, 3) if env.entities[RED_IDS[i]].state.alive and env.entities[RED_IDS[j]].state.alive]
    return (min(cross) if cross else np.inf, min(friendly) if friendly else np.inf)


def _policy_actions(policy: Any, algorithm: str, observations: Mapping[str, np.ndarray], device: str) -> np.ndarray:
    with torch.no_grad():
        if algorithm == "mappo":
            batch = torch.as_tensor(np.asarray([observations[aid] for aid in RED_IDS]), device=device)
            actions, _ = policy.sample(batch, deterministic=True)
            return actions.cpu().numpy()
        actions = []
        for index, aid in enumerate(RED_IDS):
            action, _ = policy.actors[index].sample(torch.as_tensor(observations[aid], device=device).unsqueeze(0), deterministic=True)
            actions.append(action.squeeze(0).cpu().numpy())
        return np.asarray(actions)


def evaluate_policy(
    policy: Any,
    algorithm: str,
    env_config: str | Path | Mapping[str, Any] | None,
    blue_mode: str,
    profile: str,
    seeds: Iterable[int],
    sampled_steps: int,
    device: str = "cpu",
    action_mode: str = "learned",
) -> list[dict[str, Any]]:
    """Evaluate fixed seeds and collect episode-level learning diagnostics."""
    records: list[dict[str, Any]] = []
    combat_hold = None
    for episode_seed in seeds:
        env = HeterogeneousMAVUAVAirCombatEnv(env_config, blue_target_mode=blue_mode, profile=profile)
        observations, _ = env.reset(seed=int(episode_seed))
        combat_hold = int(env.config["combat"]["hold_steps"])
        rng = np.random.default_rng(int(episode_seed) + 1_000_000)
        situation_sum = event_sum = terminal_sum = safety_sum = 0.0
        min_cross = min_friendly = np.inf
        attack_window_steps = max_streak = 0
        proxy_counts = {aid: Counter() for aid in RED_IDS}
        proxy_denominators = Counter()
        all_same_steps = two_same_steps = comparable_steps = 0
        done = False
        while not done:
            proxies = target_proxies(env)
            for aid, target in proxies.items():
                proxy_counts[aid][target] += 1
                proxy_denominators[aid] += 1
            if len(proxies) >= 2:
                comparable_steps += 1
                counts = Counter(proxies.values())
                two_same_steps += int(max(counts.values()) >= 2)
                all_same_steps += int(len(proxies) == 3 and max(counts.values()) == 3)
            cross, friendly = _minimum_distances(env)
            min_cross, min_friendly = min(min_cross, cross), min(min_friendly, friendly)
            if action_mode == "zero":
                actions = np.zeros((3, 3), dtype=np.float32)
            elif action_mode == "random":
                actions = rng.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
            else:
                actions = _policy_actions(policy, algorithm, observations, device)
            observations, _, terminated, truncated, info = env.step(actions)
            cross, friendly = _minimum_distances(env)
            min_cross, min_friendly = min(min_cross, cross), min(min_friendly, friendly)
            situation_sum += float(info["team_situation"])
            event_sum += float(info["event_reward"])
            terminal_sum += float(info["terminal_reward"])
            safety_sum += float(info["safety_reward"])
            current_max = max(env._attack_streak.values(), default=0)
            if info["attack_events"]:
                current_max = max(current_max, combat_hold)
            max_streak = max(max_streak, current_max)
            attack_window_steps += int(current_max > 0)
            done = bool(terminated or truncated)
        summary = dict(info["episode_summary"])
        summary.update({
            "algorithm": algorithm, "sampled_steps": int(sampled_steps), "blue_mode": blue_mode,
            "evaluation_seed": int(episode_seed), "action_mode": action_mode, "environment_profile": profile,
            "situation_reward_sum": float(situation_sum), "event_reward_sum": float(event_sum),
            "terminal_reward_sum": float(terminal_sum), "safety_reward_sum": float(safety_sum),
            "minimum_cross_team_distance": float(min_cross), "minimum_friendly_red_distance": float(min_friendly),
            "attack_window_active_steps": int(attack_window_steps), "maximum_attack_streak": int(max_streak),
            "all_three_same_target_steps": int(all_same_steps), "two_or_more_same_target_steps": int(two_same_steps),
            "target_comparable_steps": int(comparable_steps),
        })
        for aid in RED_IDS:
            denominator = max(1, proxy_denominators[aid])
            for bid in BLUE_IDS:
                summary[f"{aid}_proxy_{bid}_frequency"] = proxy_counts[aid][bid] / denominator
        records.append(summary)
    return records


def evaluation_summary(records: list[Mapping[str, Any]], algorithm: str, seed: int, sampled_steps: int, blue_mode: str) -> dict[str, Any]:
    if not records:
        raise ValueError("evaluation records cannot be empty")
    n = len(records)
    return {
        "sampled_steps": sampled_steps, "algorithm": algorithm, "seed": seed, "blue_mode": blue_mode,
        "environment_profile": records[0]["environment_profile"], "episodes": n,
        "red_win_rate": sum(r["outcome"] == "red" for r in records) / n,
        "blue_win_rate": sum(r["outcome"] == "blue" for r in records) / n,
        "draw_rate": sum(r["outcome"] == "draw" for r in records) / n,
        "MAV_survival_rate": float(np.mean([r["mav_survived"] for r in records])),
        "mean_UAV_survivors": float(np.mean([r["red_uav_survivors"] for r in records])),
        "mean_red_attack_kills": float(np.mean([r["red_attack_kills"] for r in records])),
        "mean_blue_attack_kills": float(np.mean([r["blue_attack_kills"] for r in records])),
        "mean_episode_length": float(np.mean([r["episode_length"] for r in records])),
        "mean_episode_return": float(np.mean([r["episode_return"] for r in records])),
    }


def geometry_summary(records: list[Mapping[str, Any]], algorithm: str, seed: int, sampled_steps: int, blue_mode: str) -> dict[str, Any]:
    return {
        "sampled_steps": sampled_steps, "algorithm": algorithm, "seed": seed, "blue_mode": blue_mode, "episodes": len(records),
        "mean_minimum_cross_team_distance": float(np.mean([r["minimum_cross_team_distance"] for r in records])),
        "mean_minimum_friendly_red_distance": float(np.mean([r["minimum_friendly_red_distance"] for r in records])),
        "fraction_cross_team_below_100m": float(np.mean([r["minimum_cross_team_distance"] < 100.0 for r in records])),
        "fraction_friendly_red_below_100m": float(np.mean([r["minimum_friendly_red_distance"] < 100.0 for r in records])),
        "mean_attack_window_active_steps": float(np.mean([r["attack_window_active_steps"] for r in records])),
        "mean_maximum_attack_streak": float(np.mean([r["maximum_attack_streak"] for r in records])),
    }


def target_concentration_summary(records: list[Mapping[str, Any]], algorithm: str, seed: int, sampled_steps: int, blue_mode: str) -> dict[str, Any]:
    comparable = sum(int(r["target_comparable_steps"]) for r in records)
    row: dict[str, Any] = {
        "sampled_steps": sampled_steps, "algorithm": algorithm, "seed": seed, "blue_mode": blue_mode,
        "all_three_same_target_rate": sum(int(r["all_three_same_target_steps"]) for r in records) / max(1, comparable),
        "two_or_more_same_target_rate": sum(int(r["two_or_more_same_target_steps"]) for r in records) / max(1, comparable),
    }
    for aid in RED_IDS:
        for bid in BLUE_IDS:
            row[f"{aid}_proxy_{bid}_frequency"] = float(np.mean([r[f"{aid}_proxy_{bid}_frequency"] for r in records]))
    return row


def reward_diagnostic_rows(records: list[Mapping[str, Any]], algorithm: str, seed: int, sampled_steps: int, blue_mode: str) -> list[dict[str, Any]]:
    rows = []
    kills = np.asarray([r["red_attack_kills"] for r in records], dtype=np.float64)
    returns = np.asarray([r["episode_return"] for r in records], dtype=np.float64)
    correlation = float(np.corrcoef(kills, returns)[0, 1]) if kills.std() > 0 and returns.std() > 0 else 0.0
    for outcome in ("red", "blue", "draw"):
        selected = [r for r in records if r["outcome"] == outcome]
        rows.append({
            "sampled_steps": sampled_steps, "algorithm": algorithm, "seed": seed, "blue_mode": blue_mode,
            "outcome": outcome, "episodes": len(selected),
            "mean_situation_reward_sum": float(np.mean([r["situation_reward_sum"] for r in selected])) if selected else 0.0,
            "mean_event_reward_sum": float(np.mean([r["event_reward_sum"] for r in selected])) if selected else 0.0,
            "mean_safety_reward_sum": float(np.mean([r["safety_reward_sum"] for r in selected])) if selected else 0.0,
            "mean_terminal_reward": float(np.mean([r["terminal_reward_sum"] for r in selected])) if selected else 0.0,
            "mean_total_return": float(np.mean([r["episode_return"] for r in selected])) if selected else 0.0,
            "return_red_attack_kills_correlation": correlation,
        })
    return rows


def draw_return_exceeds_red_win(reward_rows: Iterable[Mapping[str, Any]]) -> bool | None:
    """Compare draw/win returns, or return None when either outcome is absent."""
    by_outcome = {str(row["outcome"]): row for row in reward_rows}
    red, draw = by_outcome.get("red"), by_outcome.get("draw")
    if red is None or draw is None or int(float(red["episodes"])) == 0 or int(float(draw["episodes"])) == 0:
        return None
    return float(draw["mean_total_return"]) > float(red["mean_total_return"])


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rule_baselines(output_root: str | Path, episodes: int, env_config: str | Path | Mapping[str, Any] | None = None, profile: str = "main") -> list[dict[str, Any]]:
    """Evaluate zero/random Red against both Blue modes with fixed seeds."""
    directory = Path(output_root) / "rule_baselines"
    result_path = directory / "evaluations.csv"
    summary_path = directory / "summary.json"
    if result_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("environment_profile") != profile:
            raise RuntimeError(
                f"incompatible cached rule baseline environment profile: {summary.get('environment_profile')!r} "
                f"(expected {profile!r})"
            )
        with result_path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    episode_records: dict[str, Any] = {}
    seeds = fixed_evaluation_seeds(episodes)
    for action_mode in ("zero", "random"):
        for blue_mode in ("nearest", "mav_priority"):
            records = evaluate_policy(None, action_mode, env_config, blue_mode, profile, seeds, 0, action_mode=action_mode)
            row = evaluation_summary(records, action_mode, 1, 0, blue_mode)
            row["baseline"] = action_mode
            row["environment_profile"] = profile
            rows.append(row)
            episode_records[f"{action_mode}_{blue_mode}"] = records
    write_csv(result_path, rows)
    write_json(summary_path, {"environment_profile": profile, "evaluation_seeds": seeds, "evaluations": rows, "episode_records": episode_records})
    return rows


def set_rollout_horizon(trainer: Any, horizon: int) -> None:
    """Resize only the current calibration rollout to hit exact step boundaries."""
    if horizon <= 0:
        raise ValueError("rollout horizon must be positive")
    if trainer.buffer.horizon != int(horizon):
        trainer.buffer = RolloutBuffer(int(horizon), int(trainer.config["num_envs"]))


def save_calibration_checkpoint(path: str | Path, trainer: Any, algorithm: str, sampled_steps: int, diagnostics: TrainingDiagnostics) -> None:
    environment_states = trainer.vector_env.get_env_states()
    payload: dict[str, Any] = {
        "format": "mavuav_learnability_calibration_v2", "algorithm": algorithm,
        "environment_version": ENVIRONMENT_VERSION,
        "environment_profile": trainer.config["environment_profile"],
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
        "sampled_steps": int(sampled_steps), "trainer_config": trainer.config,
        "critic": trainer.critic.state_dict(), "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "trainer_numpy_rng": trainer.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if trainer.device.type == "cuda" and torch.cuda.is_available() else None,
        "diagnostics": diagnostics.state_dict(),
        "rollout_state": {
            "observations": trainer.observations, "global_states": trainer.global_states,
            "active_masks": trainer.active_masks, "environment_states": environment_states,
            "vector_reset_counts": trainer.vector_env.reset_counts,
            "vector_base_seed": trainer.vector_env.base_seed,
        },
    }
    if algorithm == "mappo":
        payload.update({"actor": trainer.actor.state_dict(), "actor_optimizer": trainer.actor_optimizer.state_dict()})
    else:
        payload.update({"actors": trainer.actors.state_dict(), "actor_optimizers": [optimizer.state_dict() for optimizer in trainer.actor_optimizers]})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)


def load_calibration_checkpoint(path: str | Path, trainer: Any, algorithm: str) -> tuple[int, TrainingDiagnostics]:
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    expected = {
        "format": "mavuav_learnability_calibration_v2", "algorithm": algorithm,
        "environment_version": ENVIRONMENT_VERSION,
        "environment_profile": trainer.config["environment_profile"],
        "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM,
    }
    mismatches = [f"{key}={payload.get(key)!r} (expected {value!r})" for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError("incompatible calibration checkpoint: " + "; ".join(mismatches))
    trainer.critic.load_state_dict(payload["critic"]); trainer.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    if algorithm == "mappo":
        trainer.actor.load_state_dict(payload["actor"]); trainer.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    else:
        trainer.actors.load_state_dict(payload["actors"])
        for optimizer, state in zip(trainer.actor_optimizers, payload["actor_optimizers"]): optimizer.load_state_dict(state)
    sampled_steps = int(payload["sampled_steps"]); trainer.env_steps = sampled_steps
    trainer.rng.bit_generator.state = payload["trainer_numpy_rng"]
    torch.set_rng_state(payload["torch_rng"].cpu())
    cuda_rng = payload.get("torch_cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    rollout = payload["rollout_state"]
    if np.asarray(rollout["observations"]).shape[-1] != OBS_DIM or np.asarray(rollout["global_states"]).shape[-1] != GLOBAL_STATE_DIM:
        raise RuntimeError("incompatible calibration checkpoint rollout dimensions")
    trainer.observations = np.asarray(rollout["observations"], dtype=np.float32)
    trainer.global_states = np.asarray(rollout["global_states"], dtype=np.float32)
    trainer.active_masks = np.asarray(rollout["active_masks"], dtype=np.float32)
    trainer.vector_env.set_env_states(
        rollout["environment_states"],
        np.asarray(rollout["vector_reset_counts"], dtype=np.int64),
        rollout.get("vector_base_seed", trainer.vector_env.base_seed),
    )
    return sampled_steps, TrainingDiagnostics.from_state_dict(payload["diagnostics"])


def trainer_parameter_snapshot(trainer: Any, algorithm: str) -> list[torch.Tensor]:
    module = trainer.actor if algorithm == "mappo" else trainer.actors
    return [parameter.detach().cpu().clone() for parameter in module.parameters()]


def train_to_sampled_steps(trainer: Any, target_sampled_steps: int, diagnostics: TrainingDiagnostics) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train to an exact transition count without changing update equations.

    Standard rollouts retain the configured horizon.  Only the last rollout before
    an exact checkpoint boundary is shortened.
    """
    target = int(target_sampled_steps)
    num_envs = int(trainer.config["num_envs"])
    if target < trainer.env_steps or target % num_envs != 0:
        raise ValueError("target sampled steps must be >= current steps and divisible by num_envs")
    configured_horizon = int(trainer.config["rollout_steps"])
    episodes: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    while trainer.env_steps < target:
        remaining_vector_steps = (target - trainer.env_steps) // num_envs
        set_rollout_horizon(trainer, min(configured_horizon, remaining_vector_steps))
        completed = trainer.collect_rollout()
        diagnostics.observe_rollout(trainer.buffer)
        update = trainer.update()
        episodes.extend(completed)
        metrics.append({"sampled_steps": int(trainer.env_steps), **update})
    set_rollout_horizon(trainer, configured_horizon)
    return episodes, metrics
