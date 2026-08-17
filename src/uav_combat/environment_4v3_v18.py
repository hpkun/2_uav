"""v18 recurrent fire-geometry environment over frozen v17 rewards."""
from __future__ import annotations

from typing import Any

import numpy as np

from .environment_4v3_v11 import distance_score_v11
from .environment_4v3_v17 import (
    GS_DIM_V17,
    OBS_DIM_V17,
    FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv,
)
from .geometry import compute_pairwise_geometry
from .models import AircraftState
from .scenario_4v3_v18 import (
    AGENT_REWARD_COMPONENT_KEYS_V18,
    REWARD_COMPONENT_KEYS_V18,
    REWARD_CONTRACT_VERSION_V18,
    FunctionalHeterogeneous4v3V18Scenario,
    resolved_reward_contract_v18,
    validate_heterogeneous_4v3_v18_config,
)

OBS_DIM_V18 = OBS_DIM_V17
GS_DIM_V18 = GS_DIM_V17


def fire_quality_v18(
    attacker: AircraftState, target: AircraftState, profile: dict[str, Any]
) -> float:
    """Distance × ATA fire-control quality; AA is intentionally excluded."""
    geometry = compute_pairwise_geometry(attacker, target)
    ata_score = float(
        np.clip(
            1.0 - geometry.ata / float(profile["lock_ata_fade_max"]),
            0.0,
            1.0,
        )
    )
    quality = distance_score_v11(geometry.distance, profile) * ata_score
    return float(np.clip(quality, 0.0, 1.0))


class FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv(
    FunctionalHeterogeneous4v3V17RoleSituationEventMissionRewardEnv
):
    variant = "functional_heterogeneous_4v3_v18_recurrent_fire_geometry"
    reward_contract_version = REWARD_CONTRACT_VERSION_V18
    default_config_path = (
        "configs/heterogeneous_4v3_main_v18_recurrent_fire_geometry.yaml"
    )
    scenario_class = FunctionalHeterogeneous4v3V18Scenario
    reward_component_keys = REWARD_COMPONENT_KEYS_V18
    agent_reward_component_keys = AGENT_REWARD_COMPONENT_KEYS_V18

    def _validate_config(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v18_config(config)

    def _resolve_reward_contract(self, config: dict[str, Any]) -> dict[str, Any]:
        return resolved_reward_contract_v18(config)

    def _attack_quality(self, attacker: AircraftState, target: AircraftState) -> float:
        return fire_quality_v18(attacker, target, self.profile)


FunctionalHeterogeneous4v3AirCombatEnvV18 = (
    FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv
)

__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V18",
    "GS_DIM_V18",
    "OBS_DIM_V18",
    "REWARD_COMPONENT_KEYS_V18",
    "FunctionalHeterogeneous4v3AirCombatEnvV18",
    "FunctionalHeterogeneous4v3V18RecurrentFireGeometryEnv",
    "fire_quality_v18",
]
