"""HAPPO baseline for homogeneous 3v3 fixed-blue experiments."""

from .buffer_3v3 import HAPPORolloutBuffer3v3
from .evaluation_3v3 import evaluate_happo_fixed_blue_3v3
from .evaluation_4v3 import evaluate_happo_fixed_blue_4v3
from .networks import CentralizedValueCritic, HAPPOGaussianActor, IndependentHAPPOActors
from .trainer_3v3 import (
    HAPPO3v3Trainer,
    happo_preceding_factor_update,
    normalize_advantages_for_agent,
    ppo_clipped_policy_loss,
    validate_episode_accounting_3v3,
)
from .trainer_4v3 import HAPPO4v3Trainer
from .evaluation_role_shared_4v3 import evaluate_role_shared_happo_fixed_blue_4v3
from .role_shared_buffer import RoleSharedRolloutBuffer4v3
from .role_shared_networks import RecurrentHAPPOGaussianActor, RoleHiddenState, RoleSharedHAPPOActors
from .trainer_role_shared_4v3 import RoleSharedHAPPO4v3Trainer
from .trainer_v14_4v3 import MissionAlignedRoleSharedHAPPO4v3Trainer

__all__ = [
    "CentralizedValueCritic",
    "HAPPO3v3Trainer",
    "HAPPO4v3Trainer",
    "RoleSharedHAPPO4v3Trainer",
    "MissionAlignedRoleSharedHAPPO4v3Trainer",
    "HAPPOGaussianActor",
    "HAPPORolloutBuffer3v3",
    "IndependentHAPPOActors",
    "RecurrentHAPPOGaussianActor",
    "RoleHiddenState",
    "RoleSharedHAPPOActors",
    "RoleSharedRolloutBuffer4v3",
    "evaluate_happo_fixed_blue_3v3",
    "evaluate_happo_fixed_blue_4v3",
    "evaluate_role_shared_happo_fixed_blue_4v3",
    "happo_preceding_factor_update",
    "normalize_advantages_for_agent",
    "ppo_clipped_policy_loss",
    "validate_episode_accounting_3v3",
]
