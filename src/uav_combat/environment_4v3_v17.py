"""v17 role Situation + Event + Mission rewards over frozen v16A mechanics."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .environment_4v3_v11 import distance_score_v11
from .environment_4v3_v15 import GS_DIM_V15, OBS_DIM_V15
from .environment_4v3_v16 import (
    FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv,
)
from .geometry import compute_pairwise_geometry
from .scenario_4v3_v17 import (
    AGENT_REWARD_COMPONENT_KEYS_V17,
    BLUE_IDS_V17,
    RED_COMBAT_IDS_V17,
    RED_IDS_V17,
    REWARD_COMPONENT_KEYS_V17,
    REWARD_CONTRACT_VERSION_V17,
    FunctionalHeterogeneous4v3V17Scenario,
    resolved_reward_contract_v17,
    validate_heterogeneous_4v3_v17_config,
)

OBS_DIM_V17 = OBS_DIM_V15
GS_DIM_V17 = GS_DIM_V15
BLUE_TEAM_SIZE_V17 = len(BLUE_IDS_V17)

COMBAT_ACTIVE_REWARD_KEYS_V17 = (
    "combat_situation_reward",
    "own_kill_reward",
    "death_penalty",
    "mission_reward",
)
SUPPORT_ACTIVE_REWARD_KEYS_V17 = (
    "support_situation_reward",
    "support_team_kill_reward",
    "death_penalty",
    "mission_reward",
)


def combat_angle_score_v17(ata: float, aa: float) -> float:
    return float(np.clip(1.0 - (float(ata) + float(aa)) / math.pi, -1.0, 1.0))


def combat_distance_score_v17(distance: float, profile: dict[str, Any]) -> float:
    return float(2.0 * distance_score_v11(float(distance), profile) - 1.0)


def combat_situation_score_v17(
    angle_score: float,
    distance_score: float,
    *,
    angle_weight: float = 0.6,
    distance_weight: float = 0.4,
) -> float:
    return float(
        np.clip(
            float(angle_weight) * float(angle_score)
            + float(distance_weight) * float(distance_score),
            -1.0,
            1.0,
        )
    )


def situation_reward_v17(score: float, scale: float = 0.01) -> float:
    return float(float(scale) * np.clip(float(score), -1.0, 1.0))


def support_situation_score_v17(
    position_score: float,
    awareness_score: float,
    *,
    position_weight: float = 0.6,
    awareness_weight: float = 0.4,
) -> float:
    return float(
        np.clip(
            float(position_weight) * float(position_score)
            + float(awareness_weight) * float(awareness_score),
            -1.0,
            1.0,
        )
    )


def _empty_agent_components_v17() -> dict[str, dict[str, float]]:
    return {
        agent_id: {key: 0.0 for key in AGENT_REWARD_COMPONENT_KEYS_V17}
        for agent_id in RED_IDS_V17
    }


class FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv(
    FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv
):
    variant = "functional_heterogeneous_4v3_v17_role_situation_event_mission_reward"
    reward_contract_version = REWARD_CONTRACT_VERSION_V17
    default_config_path = (
        "configs/heterogeneous_4v3_main_v17_role_situation_event_mission_reward.yaml"
    )
    scenario_class = FunctionalHeterogeneous4v3V17Scenario
    reward_component_keys = REWARD_COMPONENT_KEYS_V17
    agent_reward_component_keys = AGENT_REWARD_COMPONENT_KEYS_V17

    def _validate_config(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v17_config(config)

    def _resolve_reward_contract(self, config: dict[str, Any]) -> dict[str, Any]:
        return resolved_reward_contract_v17(config)

    def _empty_agent_components(self) -> dict[str, dict[str, float]]:
        return _empty_agent_components_v17()

    def reset(self, seed: int | None = None):
        values = super().reset(seed)
        self._episode_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V17
        }
        self._last_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V17
        }
        return values

    def _support_situation_components(
        self, direct: dict[str, set[str]]
    ) -> tuple[float, float, float, float]:
        if not self._alive("red_0"):
            return 0.0, 0.0, 0.0, 0.0
        position = float(np.clip(2.0 * self._formation_score() - 1.0, -1.0, 1.0))
        alive_blue = [blue_id for blue_id in BLUE_IDS_V17 if self._alive(blue_id)]
        if alive_blue:
            visible = sum(
                blue_id in direct.get("red_0", set()) for blue_id in alive_blue
            )
            awareness = float(2.0 * visible / len(alive_blue) - 1.0)
        else:
            awareness = 0.0
        cfg = self.reward_contract["situation"]
        situation = support_situation_score_v17(
            position,
            awareness,
            position_weight=float(cfg["support_position_weight"]),
            awareness_weight=float(cfg["support_awareness_weight"]),
        )
        reward = situation_reward_v17(situation, float(cfg["scale"]))
        return position, awareness, situation, reward

    def _compute_reward(
        self,
        pre_targets: dict[str, str | None],
        pre_potentials: dict[str, float],
        pre_locks: dict[str, float],
        deaths: dict[str, int],
        half_events: set[tuple[str, str]],
        killers: dict[str, str],
        direct: dict[str, set[str]],
    ) -> tuple[float, dict[str, float]]:
        del pre_targets, pre_potentials, pre_locks, half_events
        values = _empty_agent_components_v17()
        situation_cfg = self.reward_contract["situation"]
        events = self.reward_contract["events"]

        for combat_id in RED_COMBAT_IDS_V17:
            if not self._alive(combat_id):
                continue
            target_id = self.targets.get(combat_id)
            if target_id not in BLUE_IDS_V17 or not self._alive(target_id):
                continue
            geometry = compute_pairwise_geometry(
                self._by_id(combat_id).state, self._by_id(target_id).state
            )
            angle = combat_angle_score_v17(geometry.ata, geometry.aa)
            distance = combat_distance_score_v17(geometry.distance, self.profile)
            score = combat_situation_score_v17(
                angle,
                distance,
                angle_weight=float(situation_cfg["combat_angle_weight"]),
                distance_weight=float(situation_cfg["combat_distance_weight"]),
            )
            row = values[combat_id]
            row["combat_angle_score"] = angle
            row["combat_distance_score"] = distance
            row["combat_situation_score"] = score
            row["combat_situation_reward"] = situation_reward_v17(
                score, float(situation_cfg["scale"])
            )

        position, awareness, support_score, support_reward = (
            self._support_situation_components(direct)
        )
        support = values["red_0"]
        support["support_position_score"] = position
        support["support_awareness_score"] = awareness
        support["support_situation_score"] = support_score
        support["support_situation_reward"] = support_reward

        blue_kills = 0
        for target_id, killer_id in killers.items():
            if target_id in BLUE_IDS_V17 and killer_id in RED_COMBAT_IDS_V17:
                values[killer_id]["own_kill_reward"] += float(
                    events["own_blue_kill"]
                )
                blue_kills += 1
        support["support_team_kill_reward"] = (
            float(events["support_team_kill_total_cap"])
            / float(BLUE_TEAM_SIZE_V17)
            * float(blue_kills)
        )

        for agent_id in deaths:
            if agent_id in RED_IDS_V17:
                values[agent_id]["death_penalty"] += float(
                    events[
                        "support_death" if agent_id == "red_0" else "combat_death"
                    ]
                )

        done, _, reason = self._terminal_result()
        mission = 0.0
        if done:
            mission_cfg = self.reward_contract["mission"]
            mission = float(
                mission_cfg[
                    "strict_full_elimination"
                    if reason == "red_full_elimination"
                    else "all_other_terminal"
                ]
            )
            for agent_id in RED_IDS_V17:
                values[agent_id]["mission_reward"] = mission

        rewards: dict[str, float] = {}
        for agent_id, components in values.items():
            active = (
                SUPPORT_ACTIVE_REWARD_KEYS_V17
                if agent_id == "red_0"
                else COMBAT_ACTIVE_REWARD_KEYS_V17
            )
            components["agent_total_reward"] = float(
                sum(components[key] for key in active)
            )
            rewards[agent_id] = components["agent_total_reward"]

        team_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V17}
        for key in REWARD_COMPONENT_KEYS_V17[:-1]:
            team_components[key] = float(
                np.mean([values[agent_id][key] for agent_id in RED_IDS_V17])
            )
        team_reward = float(np.mean(list(rewards.values())))
        team_components["team_total_reward"] = team_reward
        self._last_raw_dense_reward = float(
            team_components["combat_situation_reward"]
            + team_components["support_situation_reward"]
        )
        self._last_agent_rewards = rewards
        self._last_agent_reward_components = values
        for agent_id in RED_IDS_V17:
            self._episode_agent_returns[agent_id] += rewards[agent_id]
            for key, value in values[agent_id].items():
                self._episode_agent_reward_components[agent_id][key] += value

        numeric = [*team_components.values(), *rewards.values()]
        numeric.extend(value for row in values.values() for value in row.values())
        if not np.isfinite(numeric).all():
            raise FloatingPointError("v17 reward components must be finite")
        return team_reward, team_components

    def _agent_rewards(self, *args: Any, **kwargs: Any):
        raise RuntimeError("v17 agent rewards are computed directly by _compute_reward")

    def _reward_groups(self, components: dict[str, float]) -> dict[str, float]:
        return {
            "situation": float(
                components["combat_situation_reward"]
                + components["support_situation_reward"]
            ),
            "event": float(
                components["own_kill_reward"]
                + components["support_team_kill_reward"]
                + components["death_penalty"]
            ),
            "mission": float(components["mission_reward"]),
        }


FunctionalHeterogeneous4v3AirCombatEnvV17 = (
    FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv
)

__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V17",
    "BLUE_TEAM_SIZE_V17",
    "COMBAT_ACTIVE_REWARD_KEYS_V17",
    "GS_DIM_V17",
    "OBS_DIM_V17",
    "REWARD_COMPONENT_KEYS_V17",
    "SUPPORT_ACTIVE_REWARD_KEYS_V17",
    "FunctionalHeterogeneous4v3AirCombatEnvV17",
    "FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv",
    "combat_angle_score_v17",
    "combat_distance_score_v17",
    "combat_situation_score_v17",
    "situation_reward_v17",
    "support_situation_score_v17",
]
