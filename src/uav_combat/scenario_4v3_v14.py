"""v14 mission-aligned reward contract over the frozen v12 4v3 scenario."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import aircraft_spec
from .scenario_4v3_v12 import (
    ALL_IDS_V12,
    BLUE_IDS_V12,
    FIXED_ROLES_V12,
    RED_COMBAT_IDS_V12,
    RED_IDS_V12,
    REWARD_COMPONENT_KEYS_V12,
    FunctionalHeterogeneous4v3V12Scenario,
    validate_heterogeneous_4v3_v12_config,
)

RED_IDS_V14 = RED_IDS_V12
RED_COMBAT_IDS_V14 = RED_COMBAT_IDS_V12
BLUE_IDS_V14 = BLUE_IDS_V12
ALL_IDS_V14 = ALL_IDS_V12
FIXED_ROLES_V14 = deepcopy(FIXED_ROLES_V12)
REWARD_COMPONENT_KEYS_V14 = REWARD_COMPONENT_KEYS_V12

REWARD_CONTRACT_VERSION_V14 = "v14_mission_aligned_role_credit"
REWARD_MODE_V14 = "functional_heterogeneous_4v3_role_credit_v14"


def resolved_reward_contract_v14(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v14_config(config: dict[str, Any]) -> None:
    """Validate v14 while reusing every frozen v12 non-reward invariant."""
    combat = config.get("combat", {})
    if combat.get("reward_mode") != REWARD_MODE_V14:
        raise ValueError(f"v14 requires combat.reward_mode={REWARD_MODE_V14}")
    if combat.get("reward_contract_version") != REWARD_CONTRACT_VERSION_V14:
        raise ValueError(
            f"v14 requires reward_contract_version={REWARD_CONTRACT_VERSION_V14}"
        )

    # The v12 validator remains the single source of truth for scenario,
    # aircraft, observation-related, lock, cue, and boundary invariants.
    frozen = deepcopy(config)
    frozen["combat"]["reward_mode"] = "functional_heterogeneous_4v3_team_v12"
    frozen["combat"]["reward_contract_version"] = "v12_soft_boundary_combat_aligned"
    validate_heterogeneous_4v3_v12_config(frozen)

    mission = config["rewards"]["mission"]
    expected = {
        "red_full_elimination": 20.0,
        "red_total_loss": -15.0,
        "mutual_elimination_draw": -2.0,
        "timeout_red_win": -15.0,
        "timeout_red_loss": -15.0,
        "timeout_draw": -15.0,
    }
    for key, value in expected.items():
        if float(mission[key]) != value:
            raise ValueError(f"v14 rewards.mission.{key} must equal {value}")


class FunctionalHeterogeneous4v3V14Scenario(FunctionalHeterogeneous4v3V12Scenario):
    """The v12 mirrored placement with a v14-only validation entry point."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v14_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "ALL_IDS_V14",
    "BLUE_IDS_V14",
    "FIXED_ROLES_V14",
    "RED_COMBAT_IDS_V14",
    "RED_IDS_V14",
    "REWARD_COMPONENT_KEYS_V14",
    "REWARD_CONTRACT_VERSION_V14",
    "REWARD_MODE_V14",
    "FunctionalHeterogeneous4v3V14Scenario",
    "resolved_reward_contract_v14",
    "validate_heterogeneous_4v3_v14_config",
]
