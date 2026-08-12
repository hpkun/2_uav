"""Compact current-state and event rewards over the frozen v12 4v3 mechanics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .environment_4v3_v11 import distance_score_v11, lock_quality_v11
from .environment_4v3_v12 import DEATH_LOCK_V12
from .environment_4v3_v14 import (
    BLUE_TEAM_SIZE_V14,
    GS_DIM_V14,
    OBS_DIM_V14,
    RED_TEAM_SIZE_V14,
    FunctionalHeterogeneous4v3V14MissionAlignedEnv,
)
from .geometry import compute_pairwise_geometry
from .scenario_4v3_v15 import (
    AGENT_REWARD_COMPONENT_KEYS_V15,
    BLUE_IDS_V15,
    RED_COMBAT_IDS_V15,
    RED_IDS_V15,
    REWARD_COMPONENT_KEYS_V15,
    REWARD_CONTRACT_VERSION_V15,
    FunctionalHeterogeneous4v3V15Scenario,
    resolved_reward_contract_v15,
    validate_heterogeneous_4v3_v15_config,
)

OBS_DIM_V15 = OBS_DIM_V14
GS_DIM_V15 = GS_DIM_V14
RED_TEAM_SIZE_V15 = RED_TEAM_SIZE_V14
BLUE_TEAM_SIZE_V15 = BLUE_TEAM_SIZE_V14


def angle_state_score_v15(
    ata: float, aa: float, profile: dict[str, Any]
) -> float:
    """Diagnostic angle score using the same fade limits as lock mechanics."""
    ata_quality = float(
        np.clip(1.0 - float(ata) / float(profile["lock_ata_fade_max"]), 0.0, 1.0)
    )
    aa_quality = float(
        np.clip(1.0 - float(aa) / float(profile["lock_aa_fade_max"]), 0.0, 1.0)
    )
    return float(2.0 * ata_quality * aa_quality - 1.0)


def distance_state_score_v15(distance: float, profile: dict[str, Any]) -> float:
    return float(2.0 * distance_score_v11(distance, profile) - 1.0)


def combat_state_reward_v15(lock_quality: float, scale: float = 0.02) -> float:
    """Active current-state reward derived only from instantaneous lock quality."""
    quality = float(np.clip(lock_quality, 0.0, 1.0))
    return float(float(scale) * (2.0 * quality - 1.0))


def support_state_reward_v15(position_score: float, awareness_score: float) -> float:
    return float(0.01 * (0.6 * float(position_score) + 0.4 * float(awareness_score)))


def _empty_agent_components_v15() -> dict[str, dict[str, float]]:
    return {
        agent_id: {key: 0.0 for key in AGENT_REWARD_COMPONENT_KEYS_V15}
        for agent_id in RED_IDS_V15
    }


class FunctionalHeterogeneous4v3V15PaperCompactRewardEnv(
    FunctionalHeterogeneous4v3V14MissionAlignedEnv
):
    """v14B-compatible agent streams with only compact v15 active rewards."""

    variant = "functional_heterogeneous_4v3_v15_paper_compact_attack_reward"
    reward_contract_version = REWARD_CONTRACT_VERSION_V15
    default_config_path = (
        "configs/heterogeneous_4v3_main_v15_paper_compact_attack_reward.yaml"
    )
    scenario_class = FunctionalHeterogeneous4v3V15Scenario
    reward_component_keys = REWARD_COMPONENT_KEYS_V15
    agent_reward_component_keys = AGENT_REWARD_COMPONENT_KEYS_V15

    def _validate_config(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v15_config(config)

    def _resolve_reward_contract(self, config: dict[str, Any]) -> dict[str, Any]:
        return resolved_reward_contract_v15(config)

    def _empty_agent_components(self) -> dict[str, dict[str, float]]:
        return _empty_agent_components_v15()

    def reset(self, seed: int | None = None):
        values = super().reset(seed)
        self._episode_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V15
        }
        self._last_reward_components = {
            key: 0.0 for key in REWARD_COMPONENT_KEYS_V15
        }
        return values

    def _support_state_components(
        self, direct: dict[str, set[str]]
    ) -> tuple[float, float, float]:
        if not self._alive("red_0"):
            return 0.0, 0.0, 0.0
        position = float(2.0 * self._formation_score() - 1.0)
        alive_blue = [blue_id for blue_id in BLUE_IDS_V15 if self._alive(blue_id)]
        if alive_blue:
            visible = sum(blue_id in direct.get("red_0", set()) for blue_id in alive_blue)
            awareness = float(2.0 * visible / len(alive_blue) - 1.0)
        else:
            awareness = 0.0
        return position, awareness, support_state_reward_v15(position, awareness)

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
        values = _empty_agent_components_v15()
        events = self.reward_contract["events"]

        blue_kills = sum(
            1
            for target_id, killer_id in killers.items()
            if target_id in BLUE_IDS_V15 and killer_id in RED_COMBAT_IDS_V15
        )
        team_kill = float(events["team_blue_kill"]) * blue_kills
        for agent_id in RED_IDS_V15:
            values[agent_id]["team_kill_reward"] = team_kill

        for target_id, killer_id in killers.items():
            if target_id in BLUE_IDS_V15 and killer_id in RED_COMBAT_IDS_V15:
                values[killer_id]["own_kill_reward"] += float(
                    events["own_blue_kill"]
                )

        for agent_id, cause in deaths.items():
            if cause == DEATH_LOCK_V12 and agent_id in RED_IDS_V15:
                values[agent_id]["death_penalty"] += float(
                    events[
                        "support_death" if agent_id == "red_0" else "combat_death"
                    ]
                )

        for agent_id in RED_IDS_V15:
            contacts = float(
                self._episode_metrics.get(
                    f"{agent_id}_boundary_hard_contacts_step", 0.0
                )
            )
            values[agent_id]["boundary_penalty"] = float(
                events["hard_boundary_contact"]
            ) * contacts

        for combat_id in RED_COMBAT_IDS_V15:
            if not self._alive(combat_id):
                continue
            target_id = self.targets.get(combat_id)
            if target_id not in BLUE_IDS_V15 or not self._alive(target_id):
                angle = -1.0
                distance = -1.0
                lock_quality = 0.0
            else:
                attacker_state = self._by_id(combat_id).state
                target_state = self._by_id(target_id).state
                geometry = compute_pairwise_geometry(
                    attacker_state, target_state
                )
                angle = angle_state_score_v15(
                    geometry.ata, geometry.aa, self.profile
                )
                distance = distance_state_score_v15(
                    geometry.distance, self.profile
                )
                # Reward and accumulation deliberately share the exact same
                # source of truth for instantaneous lock feasibility.
                lock_quality = lock_quality_v11(
                    attacker_state, target_state, self.profile
                )
            values[combat_id]["angle_state_reward"] = angle
            values[combat_id]["distance_state_reward"] = distance
            values[combat_id]["combat_state_reward"] = combat_state_reward_v15(
                lock_quality, self.reward_contract["combat_state"]["scale"]
            )

        position, awareness, support_state = self._support_state_components(direct)
        support = values["red_0"]
        support["support_position_state_reward"] = position
        support["support_awareness_state_reward"] = awareness
        support["support_state_reward"] = support_state

        active_keys = (
            "combat_state_reward",
            "own_kill_reward",
            "team_kill_reward",
            "death_penalty",
            "boundary_penalty",
            "support_state_reward",
        )
        rewards: dict[str, float] = {}
        for agent_id, components in values.items():
            components["agent_total_reward"] = float(
                sum(components[key] for key in active_keys)
            )
            rewards[agent_id] = components["agent_total_reward"]

        team_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS_V15}
        for key in REWARD_COMPONENT_KEYS_V15[:-1]:
            team_components[key] = float(
                np.mean([values[agent_id][key] for agent_id in RED_IDS_V15])
            )
        team_reward = float(np.mean(list(rewards.values())))
        team_components["team_total_reward"] = team_reward
        self._last_raw_dense_reward = float(
            team_components["support_state_reward"]
            + team_components["combat_state_reward"]
        )

        self._last_agent_rewards = rewards
        self._last_agent_reward_components = values
        for agent_id in RED_IDS_V15:
            self._episode_agent_returns[agent_id] += rewards[agent_id]
            for key, value in values[agent_id].items():
                self._episode_agent_reward_components[agent_id][key] += value

        numeric = [*team_components.values(), *rewards.values()]
        numeric.extend(value for row in values.values() for value in row.values())
        if not np.isfinite(numeric).all():
            raise FloatingPointError("v15 reward components must be finite")
        return team_reward, team_components

    def _agent_rewards(self, *args: Any, **kwargs: Any):
        raise RuntimeError("v15 agent rewards are computed directly by _compute_reward")

    def _reward_groups(self, components: dict[str, float]) -> dict[str, float]:
        return {
            "support_state": float(components["support_state_reward"]),
            "combat_state": float(components["combat_state_reward"]),
            "own_kill": float(components["own_kill_reward"]),
            "team_kill": float(components["team_kill_reward"]),
            "death": float(components["death_penalty"]),
            "boundary": float(components["boundary_penalty"]),
        }


FunctionalHeterogeneous4v3AirCombatEnvV15 = (
    FunctionalHeterogeneous4v3V15PaperCompactRewardEnv
)

__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V15",
    "BLUE_TEAM_SIZE_V15",
    "GS_DIM_V15",
    "OBS_DIM_V15",
    "RED_TEAM_SIZE_V15",
    "REWARD_COMPONENT_KEYS_V15",
    "FunctionalHeterogeneous4v3AirCombatEnvV15",
    "FunctionalHeterogeneous4v3V15PaperCompactRewardEnv",
    "angle_state_score_v15",
    "combat_state_reward_v15",
    "distance_state_score_v15",
    "support_state_reward_v15",
]
