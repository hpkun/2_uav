"""Compact paper-adapted v15 reward contract over frozen v12 mechanics."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .config import aircraft_spec
from .scenario_4v3_v12 import (
    ALL_IDS_V12,
    BLUE_IDS_V12,
    FIXED_ROLES_V12,
    RED_COMBAT_IDS_V12,
    RED_IDS_V12,
    FunctionalHeterogeneous4v3V12Scenario,
    validate_heterogeneous_4v3_v12_config,
)

RED_IDS_V15 = RED_IDS_V12
RED_COMBAT_IDS_V15 = RED_COMBAT_IDS_V12
BLUE_IDS_V15 = BLUE_IDS_V12
ALL_IDS_V15 = ALL_IDS_V12
FIXED_ROLES_V15 = deepcopy(FIXED_ROLES_V12)

REWARD_CONTRACT_VERSION_V15 = "v15_paper_compact_attack_reward"
REWARD_MODE_V15 = "functional_heterogeneous_4v3_role_credit_v15"

REWARD_COMPONENT_KEYS_V15 = (
    "support_state_reward",
    "combat_state_reward",
    "own_kill_reward",
    "team_kill_reward",
    "death_penalty",
    "boundary_penalty",
    "team_total_reward",
)

AGENT_REWARD_COMPONENT_KEYS_V15 = (
    "angle_state_reward",
    "distance_state_reward",
    "combat_state_reward",
    "own_kill_reward",
    "team_kill_reward",
    "death_penalty",
    "boundary_penalty",
    "support_position_state_reward",
    "support_awareness_state_reward",
    "support_state_reward",
    "agent_total_reward",
)


def resolved_reward_contract_v15(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v15_config(config: dict[str, Any]) -> None:
    combat = config.get("combat", {})
    if combat.get("reward_mode") != REWARD_MODE_V15:
        raise ValueError(f"v15 requires combat.reward_mode={REWARD_MODE_V15}")
    if combat.get("reward_contract_version") != REWARD_CONTRACT_VERSION_V15:
        raise ValueError(
            f"v15 requires reward_contract_version={REWARD_CONTRACT_VERSION_V15}"
        )

    # Reuse the v12 validator as the source of truth for every non-reward
    # invariant. The translated rewards exist only for validation and never
    # enter the active v15 reward calculation.
    frozen = deepcopy(config)
    frozen["combat"]["reward_mode"] = "functional_heterogeneous_4v3_team_v12"
    frozen["combat"]["reward_contract_version"] = (
        "v12_soft_boundary_combat_aligned"
    )
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
            "blue_combat_killed": 8.0,
            "red_combat_killed": -4.0,
            "red_support_killed": -4.0,
            "red_boundary_hard_contact": -0.1,
        },
        "combat_progress": {
            "geometry_scale": 0.0,
            "lock_scale": 0.0,
            "half_lock_event": 0.0,
        },
        "support_events": {
            "unique_detection": 0.0,
            "cue_to_direct": 0.0,
            "cue_to_half_lock": 0.0,
            "assisted_kill": 0.0,
        },
        "support_formation": {"progress_scale": 0.0},
        "dense_clip": {"min": -1.0, "max": 1.0},
    }
    validate_heterogeneous_4v3_v12_config(frozen)

    rewards = config.get("rewards", {})
    expected = {
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
    for section, values in expected.items():
        actual = rewards.get(section)
        if not isinstance(actual, dict):
            raise KeyError(f"missing v15 rewards.{section}")
        for key, expected_value in values.items():
            if key not in actual or not math.isfinite(float(actual[key])):
                raise ValueError(f"missing or non-finite rewards.{section}.{key}")
            if float(actual[key]) != expected_value:
                raise ValueError(
                    f"v15 rewards.{section}.{key} must equal {expected_value}"
                )


class FunctionalHeterogeneous4v3V15Scenario(
    FunctionalHeterogeneous4v3V12Scenario
):
    """Frozen v12 mirrored scenario with a v15 reward validation entry point."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v15_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V15",
    "ALL_IDS_V15",
    "BLUE_IDS_V15",
    "FIXED_ROLES_V15",
    "RED_COMBAT_IDS_V15",
    "RED_IDS_V15",
    "REWARD_COMPONENT_KEYS_V15",
    "REWARD_CONTRACT_VERSION_V15",
    "REWARD_MODE_V15",
    "FunctionalHeterogeneous4v3V15Scenario",
    "resolved_reward_contract_v15",
    "validate_heterogeneous_4v3_v15_config",
]
