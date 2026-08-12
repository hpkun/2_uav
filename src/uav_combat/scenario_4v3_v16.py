"""v16 positive-lock reward contract over frozen v15/v12 mechanics."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import aircraft_spec
from .scenario_4v3_v12 import FunctionalHeterogeneous4v3V12Scenario
from .scenario_4v3_v15 import (
    AGENT_REWARD_COMPONENT_KEYS_V15,
    ALL_IDS_V15,
    BLUE_IDS_V15,
    FIXED_ROLES_V15,
    RED_COMBAT_IDS_V15,
    RED_IDS_V15,
    REWARD_COMPONENT_KEYS_V15,
    REWARD_CONTRACT_VERSION_V15,
    REWARD_MODE_V15,
    validate_heterogeneous_4v3_v15_config,
)

RED_IDS_V16 = RED_IDS_V15
RED_COMBAT_IDS_V16 = RED_COMBAT_IDS_V15
BLUE_IDS_V16 = BLUE_IDS_V15
ALL_IDS_V16 = ALL_IDS_V15
FIXED_ROLES_V16 = deepcopy(FIXED_ROLES_V15)
REWARD_COMPONENT_KEYS_V16 = REWARD_COMPONENT_KEYS_V15
AGENT_REWARD_COMPONENT_KEYS_V16 = AGENT_REWARD_COMPONENT_KEYS_V15

REWARD_CONTRACT_VERSION_V16 = "v16_positive_lock_quality_reward"
REWARD_MODE_V16 = "functional_heterogeneous_4v3_role_credit_v16"
OBSERVATION_CONTRACT_V16A = "legacy_fixed_order"
OBSERVATION_CONTRACT_V16B = "canonical_same_team"
OBSERVATION_CONTRACTS_V16 = (
    OBSERVATION_CONTRACT_V16A,
    OBSERVATION_CONTRACT_V16B,
)


def resolved_reward_contract_v16(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config["rewards"])


def validate_heterogeneous_4v3_v16_config(config: dict[str, Any]) -> None:
    combat = config.get("combat", {})
    if combat.get("reward_mode") != REWARD_MODE_V16:
        raise ValueError(f"v16 requires combat.reward_mode={REWARD_MODE_V16}")
    if combat.get("reward_contract_version") != REWARD_CONTRACT_VERSION_V16:
        raise ValueError(
            f"v16 requires reward_contract_version={REWARD_CONTRACT_VERSION_V16}"
        )
    observation = combat.get("observation_contract")
    if observation not in OBSERVATION_CONTRACTS_V16:
        raise ValueError(
            f"v16 observation_contract must be one of {OBSERVATION_CONTRACTS_V16}"
        )

    # v15 remains the source of truth for every frozen environment and reward
    # scalar. Only the combat-state mapping is versioned by the v16 env class.
    frozen = deepcopy(config)
    frozen["combat"]["reward_mode"] = REWARD_MODE_V15
    frozen["combat"]["reward_contract_version"] = REWARD_CONTRACT_VERSION_V15
    frozen["combat"].pop("observation_contract", None)
    validate_heterogeneous_4v3_v15_config(frozen)


class FunctionalHeterogeneous4v3V16Scenario(
    FunctionalHeterogeneous4v3V12Scenario
):
    def __init__(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v16_config(config)
        self.config = config
        self.spec = aircraft_spec(config)


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V16",
    "ALL_IDS_V16",
    "BLUE_IDS_V16",
    "FIXED_ROLES_V16",
    "OBSERVATION_CONTRACT_V16A",
    "OBSERVATION_CONTRACT_V16B",
    "RED_COMBAT_IDS_V16",
    "RED_IDS_V16",
    "REWARD_COMPONENT_KEYS_V16",
    "REWARD_CONTRACT_VERSION_V16",
    "REWARD_MODE_V16",
    "FunctionalHeterogeneous4v3V16Scenario",
    "resolved_reward_contract_v16",
    "validate_heterogeneous_4v3_v16_config",
]
