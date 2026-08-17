"""v18 recurrent fire-geometry contract over the frozen v17 task."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import aircraft_spec
from .scenario_4v3_v12 import FunctionalHeterogeneous4v3V12Scenario
from .scenario_4v3_v17 import (
    AGENT_REWARD_COMPONENT_KEYS_V17,
    ALL_IDS_V17,
    BLUE_IDS_V17,
    FIXED_ROLES_V17,
    OBSERVATION_CONTRACT_V17,
    RED_COMBAT_IDS_V17,
    RED_IDS_V17,
    REWARD_COMPONENT_KEYS_V17,
    REWARD_CONTRACT_VERSION_V17,
    REWARD_MODE_V17,
    validate_heterogeneous_4v3_v17_config,
)

REWARD_CONTRACT_VERSION_V18 = "v18_role_situation_event_mission_reward"
REWARD_MODE_V18 = "functional_heterogeneous_4v3_role_credit_v18"
OBSERVATION_CONTRACT_V18 = OBSERVATION_CONTRACT_V17
REWARD_COMPONENT_KEYS_V18 = REWARD_COMPONENT_KEYS_V17
AGENT_REWARD_COMPONENT_KEYS_V18 = AGENT_REWARD_COMPONENT_KEYS_V17
RED_IDS_V18 = RED_IDS_V17
RED_COMBAT_IDS_V18 = RED_COMBAT_IDS_V17
BLUE_IDS_V18 = BLUE_IDS_V17
ALL_IDS_V18 = ALL_IDS_V17
FIXED_ROLES_V18 = deepcopy(FIXED_ROLES_V17)


def resolved_reward_contract_v18(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v18_config(config: dict[str, Any]) -> None:
    combat = config.get("combat", {})
    if combat.get("reward_mode") != REWARD_MODE_V18:
        raise ValueError(f"v18 requires combat.reward_mode={REWARD_MODE_V18}")
    if combat.get("reward_contract_version") != REWARD_CONTRACT_VERSION_V18:
        raise ValueError(
            f"v18 requires reward_contract_version={REWARD_CONTRACT_VERSION_V18}"
        )
    if combat.get("observation_contract") != OBSERVATION_CONTRACT_V18:
        raise ValueError(
            f"v18 requires observation_contract={OBSERVATION_CONTRACT_V18}"
        )
    frozen = deepcopy(config)
    frozen["combat"]["reward_mode"] = REWARD_MODE_V17
    frozen["combat"]["reward_contract_version"] = REWARD_CONTRACT_VERSION_V17
    validate_heterogeneous_4v3_v17_config(frozen)


class FunctionalHeterogeneous4v3V18Scenario(
    FunctionalHeterogeneous4v3V12Scenario
):
    """Frozen v17 placement and aircraft contract with v18 validation."""

    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v18_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V18",
    "ALL_IDS_V18",
    "BLUE_IDS_V18",
    "FIXED_ROLES_V18",
    "OBSERVATION_CONTRACT_V18",
    "RED_COMBAT_IDS_V18",
    "RED_IDS_V18",
    "REWARD_COMPONENT_KEYS_V18",
    "REWARD_CONTRACT_VERSION_V18",
    "REWARD_MODE_V18",
    "FunctionalHeterogeneous4v3V18Scenario",
    "resolved_reward_contract_v18",
    "validate_heterogeneous_4v3_v18_config",
]
