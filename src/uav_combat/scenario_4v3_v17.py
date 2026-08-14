"""Role Situation + Event + Mission reward contract over frozen v16A mechanics."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .config import aircraft_spec
from .scenario_4v3_v12 import FunctionalHeterogeneous4v3V12Scenario
from .scenario_4v3_v16 import (
    ALL_IDS_V16,
    BLUE_IDS_V16,
    FIXED_ROLES_V16,
    OBSERVATION_CONTRACT_V16A,
    RED_COMBAT_IDS_V16,
    RED_IDS_V16,
    REWARD_CONTRACT_VERSION_V16,
    REWARD_MODE_V16,
    validate_heterogeneous_4v3_v16_config,
)

RED_IDS_V17 = RED_IDS_V16
RED_COMBAT_IDS_V17 = RED_COMBAT_IDS_V16
BLUE_IDS_V17 = BLUE_IDS_V16
ALL_IDS_V17 = ALL_IDS_V16
FIXED_ROLES_V17 = deepcopy(FIXED_ROLES_V16)

REWARD_CONTRACT_VERSION_V17 = "v17_role_situation_event_mission_reward"
REWARD_MODE_V17 = "functional_heterogeneous_4v3_role_credit_v17"
OBSERVATION_CONTRACT_V17 = OBSERVATION_CONTRACT_V16A

REWARD_COMPONENT_KEYS_V17 = (
    "combat_situation_reward",
    "support_situation_reward",
    "own_kill_reward",
    "support_team_kill_reward",
    "death_penalty",
    "mission_reward",
    "team_total_reward",
)

AGENT_REWARD_COMPONENT_KEYS_V17 = (
    "combat_angle_score",
    "combat_distance_score",
    "combat_situation_score",
    "combat_situation_reward",
    "support_position_score",
    "support_awareness_score",
    "support_situation_score",
    "support_situation_reward",
    "own_kill_reward",
    "support_team_kill_reward",
    "death_penalty",
    "mission_reward",
    "agent_total_reward",
)


def resolved_reward_contract_v17(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v17_config(config: dict[str, Any]) -> None:
    combat = config.get("combat", {})
    if combat.get("reward_mode") != REWARD_MODE_V17:
        raise ValueError(f"v17 requires combat.reward_mode={REWARD_MODE_V17}")
    if combat.get("reward_contract_version") != REWARD_CONTRACT_VERSION_V17:
        raise ValueError(
            f"v17 requires reward_contract_version={REWARD_CONTRACT_VERSION_V17}"
        )
    if combat.get("observation_contract") != OBSERVATION_CONTRACT_V17:
        raise ValueError(
            f"v17 requires observation_contract={OBSERVATION_CONTRACT_V17}"
        )

    rewards = config.get("rewards", {})
    expected = {
        "situation": {
            "scale": 0.01,
            "combat_angle_weight": 0.6,
            "combat_distance_weight": 0.4,
            "support_position_weight": 0.6,
            "support_awareness_weight": 0.4,
        },
        "events": {
            "own_blue_kill": 1.0,
            "combat_death": -1.0,
            "support_death": -1.0,
            "support_team_kill_total_cap": 1.0,
        },
        "mission": {
            "strict_full_elimination": 3.0,
            "all_other_terminal": -3.0,
        },
    }
    if set(rewards) != set(expected):
        raise ValueError(
            f"v17 rewards must contain exactly {tuple(expected)}, got {tuple(rewards)}"
        )
    for section, values in expected.items():
        actual = rewards.get(section)
        if not isinstance(actual, dict) or set(actual) != set(values):
            raise ValueError(
                f"v17 rewards.{section} must contain exactly {tuple(values)}"
            )
        for key, expected_value in values.items():
            value = float(actual[key])
            if not math.isfinite(value) or value != expected_value:
                raise ValueError(
                    f"v17 rewards.{section}.{key} must equal {expected_value}"
                )

    # v16A remains the source of truth for every non-reward invariant. This
    # translated copy is validation-only and never enters the active v17 reward.
    frozen = deepcopy(config)
    frozen["combat"]["reward_mode"] = REWARD_MODE_V16
    frozen["combat"]["reward_contract_version"] = REWARD_CONTRACT_VERSION_V16
    frozen["rewards"] = {
        "mission": {
            "red_full_elimination": 0.0,
            "red_total_loss": 0.0,
            "mutual_elimination_draw": 0.0,
            "timeout_red_win": 0.0,
            "timeout_red_loss": 0.0,
            "timeout_draw": 0.0,
        },
        "events": {
            "own_blue_kill": 8.0,
            "combat_death": -4.0,
            "support_death": -4.0,
            "hard_boundary_contact": -0.1,
            "team_blue_kill": 1.0,
        },
        "combat_state": {"scale": 0.02},
        "support_state": {
            "scale": 0.01,
            "position_weight": 0.6,
            "awareness_weight": 0.4,
        },
    }
    validate_heterogeneous_4v3_v16_config(frozen)


class FunctionalHeterogeneous4v3V17Scenario(
    FunctionalHeterogeneous4v3V12Scenario
):
    """Frozen v16A scenario with an independent v17 reward validator."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v17_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V17",
    "ALL_IDS_V17",
    "BLUE_IDS_V17",
    "FIXED_ROLES_V17",
    "OBSERVATION_CONTRACT_V17",
    "RED_COMBAT_IDS_V17",
    "RED_IDS_V17",
    "REWARD_COMPONENT_KEYS_V17",
    "REWARD_CONTRACT_VERSION_V17",
    "REWARD_MODE_V17",
    "FunctionalHeterogeneous4v3V17Scenario",
    "resolved_reward_contract_v17",
    "validate_heterogeneous_4v3_v17_config",
]
