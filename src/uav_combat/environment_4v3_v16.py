"""v16A/B reward and observation variants over frozen v15 mechanics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .environment_4v3_v15 import (
    AGENT_REWARD_COMPONENT_KEYS_V15,
    GS_DIM_V15,
    OBS_DIM_V15,
    REWARD_COMPONENT_KEYS_V15,
    FunctionalHeterogeneous4v3V15PaperCompactRewardEnv,
)
from .models import Aircraft
from .scenario_4v3_v16 import (
    OBSERVATION_CONTRACT_V16A,
    OBSERVATION_CONTRACT_V16B,
    REWARD_CONTRACT_VERSION_V16,
    FunctionalHeterogeneous4v3V16Scenario,
    resolved_reward_contract_v16,
    validate_heterogeneous_4v3_v16_config,
)

OBS_DIM_V16 = OBS_DIM_V15
GS_DIM_V16 = GS_DIM_V15
REWARD_COMPONENT_KEYS_V16 = REWARD_COMPONENT_KEYS_V15
AGENT_REWARD_COMPONENT_KEYS_V16 = AGENT_REWARD_COMPONENT_KEYS_V15
TEAMMATE_SEGMENT_V16 = slice(12, 36)


def combat_state_reward_v16(lock_quality: float, scale: float = 0.02) -> float:
    """Reward is monotone with the exact lock increment quality, without threshold."""
    return float(float(scale) * np.clip(float(lock_quality), 0.0, 1.0))


def _combat_teammate_key_v16(block: np.ndarray) -> tuple[float, ...]:
    """Pure physical/state key; aircraft identity is intentionally unavailable."""
    rel = np.asarray(
        [block[0] * 6000.0, block[1] * 6000.0, block[2] * 3000.0],
        dtype=np.float64,
    )
    rel_v = np.asarray(block[3:6], dtype=np.float64) * 300.0
    return (
        float(np.dot(rel, rel)),
        float(rel[0]),
        float(rel[1]),
        float(rel[2]),
        float(rel_v[0]),
        float(rel_v[1]),
        float(rel_v[2]),
        float(block[6]),
    )


def canonicalize_same_team_teammate_blocks_v16(
    blocks: np.ndarray, *, observer_is_support: bool
) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float32)
    if values.shape != (3, 8):
        raise ValueError("v16 teammate blocks must have shape (3, 8)")
    rows = [row.copy() for row in values]
    if observer_is_support:
        ordered = sorted(rows, key=_combat_teammate_key_v16)
    else:
        support = [row for row in rows if row[7] > 0.5]
        combat = [row for row in rows if row[7] <= 0.5]
        if len(support) != 1 or len(combat) != 2:
            raise ValueError("v16 Combat observation requires one Support and two Combat teammates")
        ordered = [support[0], *sorted(combat, key=_combat_teammate_key_v16)]
    return np.stack(ordered).astype(np.float32)


class FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv(
    FunctionalHeterogeneous4v3V15PaperCompactRewardEnv
):
    variant = "functional_heterogeneous_4v3_v16a_positive_lock_quality_reward"
    reward_contract_version = REWARD_CONTRACT_VERSION_V16
    observation_contract = OBSERVATION_CONTRACT_V16A
    default_config_path = (
        "configs/heterogeneous_4v3_main_v16a_positive_lock_quality_reward.yaml"
    )
    scenario_class = FunctionalHeterogeneous4v3V16Scenario

    def _validate_config(self, config: dict[str, Any]) -> None:
        validate_heterogeneous_4v3_v16_config(config)
        if config["combat"]["observation_contract"] != self.observation_contract:
            raise ValueError(
                f"{self.variant} requires observation_contract={self.observation_contract}"
            )

    def _resolve_reward_contract(self, config: dict[str, Any]) -> dict[str, Any]:
        return resolved_reward_contract_v16(config)

    def _combat_state_reward(self, lock_quality: float) -> float:
        return combat_state_reward_v16(
            lock_quality, self.reward_contract["combat_state"]["scale"]
        )


class FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv(
    FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv
):
    variant = (
        "functional_heterogeneous_4v3_v16b_positive_lock_quality_canonical_obs"
    )
    observation_contract = OBSERVATION_CONTRACT_V16B
    default_config_path = (
        "configs/heterogeneous_4v3_main_v16b_positive_lock_quality_canonical_obs.yaml"
    )

    def _obs_for(
        self,
        own: Aircraft,
        direct: dict[str, set[str]],
        effective: dict[str, set[str]],
    ) -> np.ndarray:
        observation = super()._obs_for(own, direct, effective)
        if own.team != "red":
            return observation
        result = observation.copy()
        blocks = result[TEAMMATE_SEGMENT_V16].reshape(3, 8)
        result[TEAMMATE_SEGMENT_V16] = canonicalize_same_team_teammate_blocks_v16(
            blocks, observer_is_support=own.role == "support"
        ).reshape(-1)
        if result.shape != (OBS_DIM_V16,) or not np.isfinite(result).all():
            raise FloatingPointError("v16B canonical observation must be finite and 118D")
        return result


__all__ = [
    "AGENT_REWARD_COMPONENT_KEYS_V16",
    "GS_DIM_V16",
    "OBS_DIM_V16",
    "REWARD_COMPONENT_KEYS_V16",
    "TEAMMATE_SEGMENT_V16",
    "FunctionalHeterogeneous4v3V16APositiveLockQualityRewardEnv",
    "FunctionalHeterogeneous4v3V16BPositiveLockQualityCanonicalObsEnv",
    "canonicalize_same_team_teammate_blocks_v16",
    "combat_state_reward_v16",
]
