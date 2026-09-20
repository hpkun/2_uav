"""HAPPO trainer with isolated feed-forward and recurrent actor paths."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn

from algorithm.common.buffer import RolloutBuffer
from algorithm.common.networks import CentralizedCritic
from env.vector_env import MAVUAVVectorEnv
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, ROLE_REWARD_MODES, load_environment_config
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from algorithm.modules.pcta import PCTAIndependentActors, pursuit_consistency
from algorithm.modules.pcta_v2 import PCTAv2IndependentActors, target_behavior_diagnostics
from .networks import IndependentActors
from .recurrent import RecurrentIndependentActors
from .recurrent_buffer import RecurrentRolloutBuffer
from .tam import TAMAttentionCritic, TAMIndependentActors
from .tam_buffer import TAMRolloutBuffer
from .relational_critic import RelationalCentralizedCritic
from .agp import apply_agp
from .credit_buffer import CreditRolloutBuffer
from .counterfactual_credit import (
    CF_METHOD, CREDIT_METHODS, RDC_METHOD, ActionMarginalCreditCritic,
    combine_rdc_component_residual, component_residual, component_weights,
    credit_component_names, extract_credit_components, normalize_credit_advantage,
)


DEFAULTS = {
    "environment_profile": "main", "seed": 1, "device": "cpu", "num_envs": 16, "rollout_steps": 128,
    "gamma": 0.99, "gae_lambda": 0.95, "ppo_epochs": 4, "minibatch_size": 256,
    "clip_coef": 0.2, "actor_learning_rate": 3e-4, "critic_learning_rate": 1e-3,
    "entropy_coef": 0.01, "value_loss_coef": 0.5, "max_grad_norm": 0.5,
    "hidden_dim": 128, "actor_log_std_init": -0.5,
    "actor_variant": "vanilla", "critic_variant": "mlp", "method_variant": "baseline",
    "agp_lambda": 0.5,
    "hrta_entity_dim": 32, "hrta_role_dim": 16, "hrta_fusion_hidden_dim": 64,
    "recurrent_hidden_dim": 128, "recurrent_sequence_length": 16,
    "pcta_context_dim": 64, "pcta_enemy_dim": 32, "pcta_hidden_dim": 128,
    "pcta_consistency_coef": 0.05,
    "pcta_v2_attention_heads": 4, "pcta_v2_context_dim": 64,
    "pcta_v2_enemy_dim": 32, "pcta_v2_target_dim": 32,
    "tam_actor_gru_hidden_dim": 128, "tam_critic_gru_hidden_dim": 128,
    "tam_recurrent_sequence_length": 16, "tam_attention_heads": 4,
    "tam_token_dim": 128, "tam_actor_hidden_layers": [256, 128],
    "tam_critic_hidden_layers": [256, 128], "tam_state_memory": True,
    "tam_attention": True, "tam_inactive_mask": True,
    "tam_value_loss_type": "huber", "tam_huber_delta": 10.0,
}

LEGACY_PCTA_FAMILY = frozenset(("pcta", "pcta_attention_only", "pcta_uniform"))
PCTA_V2_VARIANT = "pcta_v2"
PCTA_FAMILY = frozenset((*LEGACY_PCTA_FAMILY, PCTA_V2_VARIANT))

RESUME_CONFIG_FIELDS = (
    "environment_profile", "seed", "num_envs", "rollout_steps", "gamma", "gae_lambda",
    "ppo_epochs", "minibatch_size", "clip_coef", "actor_learning_rate",
    "critic_learning_rate", "entropy_coef", "value_loss_coef", "max_grad_norm", "hidden_dim",
    "actor_log_std_init",
)

TAM_RESUME_CONFIG_FIELDS = (
    "tam_actor_gru_hidden_dim", "tam_critic_gru_hidden_dim", "tam_recurrent_sequence_length",
    "tam_attention_heads", "tam_token_dim", "tam_actor_hidden_layers", "tam_critic_hidden_layers",
    "tam_state_memory", "tam_attention", "tam_inactive_mask", "tam_value_loss_type", "tam_huber_delta",
)


def preceding_factor_update(factor: torch.Tensor, old_log_prob: torch.Tensor, new_log_prob: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    return (factor * torch.where(active > 0.5, ratio, torch.ones_like(ratio))).detach()


def _restore_cuda_rng_state(states: list[torch.Tensor] | None) -> None:
    """Restore exact CUDA RNG bytes from CPU tensors required by PyTorch."""
    if states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in states])


class HAPPOTrainer:
    def __init__(self, env_config: str | Path | Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.config = deepcopy(DEFAULTS)
        if config: self.config.update(dict(config.get("training", config)))
        c = self.config
        if c["actor_variant"] in ("pcta_attention_only", "pcta_uniform"):
            c["pcta_consistency_coef"] = 0.0
        if c["environment_profile"] not in ("learnability", "main"):
            raise ValueError("environment_profile must be 'learnability' or 'main'")
        if c["method_variant"] not in ("baseline", "agp", *CREDIT_METHODS):
            raise ValueError("invalid method_variant")
        if c["actor_variant"] != "vanilla" and c["method_variant"] != "baseline":
            raise ValueError("non-baseline methods require actor_variant='vanilla'")
        if c["critic_variant"] not in ("mlp", "relational", "tam_attention"):
            raise ValueError("critic_variant must be 'mlp', 'relational' or 'tam_attention'")
        if c["critic_variant"] == "relational" and (
            c["actor_variant"] != "vanilla" or c["method_variant"] != "baseline"
        ):
            raise ValueError("relational critic is only supported with vanilla actors and baseline HAPPO")
        if (c["actor_variant"] == "tam") != (c["critic_variant"] == "tam_attention"):
            raise ValueError("actor_variant='tam' and critic_variant='tam_attention' must be used together")
        if c["actor_variant"] == "tam" and c["tam_value_loss_type"] != "huber":
            raise ValueError("TAM critic currently requires tam_value_loss_type='huber'")
        self.agp_enabled = c["method_variant"] == "agp"
        self.credit_enabled = c["method_variant"] in CREDIT_METHODS
        if float(c["agp_lambda"]) < 0.0:
            raise ValueError("agp_lambda cannot be negative")
        self.device = torch.device(c["device"])
        torch.manual_seed(int(c["seed"]))
        self.rng = np.random.default_rng(int(c["seed"]))
        self.environment_config = load_environment_config(env_config)
        shaping = self.environment_config.get("shaping", {})
        self.reward_shaping_mode = str(shaping.get("mode", "absolute"))
        version = self.environment_config["environment_version"]
        self.reward_mode = ("heterogeneous_role_v1" if version.endswith("v3_7") else
                            "heterogeneous_role_coupled_v1" if version.endswith("v3_8") else
                            "heterogeneous_role_coupled_gate_v1" if version.endswith("v3_9") else
                            self.reward_shaping_mode)
        if self.credit_enabled and (
            version != "heterogeneous_mavuav_4v4_v3_9"
            or self.reward_mode != "heterogeneous_role_coupled_gate_v1"
            or c["actor_variant"] != "vanilla"
            or c["critic_variant"] != "mlp"
        ):
            raise ValueError(
                "CF/RDC-HAPPO requires v3.9 heterogeneous_role_coupled_gate_v1, "
                "actor_variant='vanilla', and critic_variant='mlp'"
            )
        self.shaping_gamma = float(shaping.get("gamma", 0.0))
        if self.reward_shaping_mode == "potential" and not np.isclose(self.shaping_gamma, float(c["gamma"]), rtol=0.0, atol=1e-12):
            raise ValueError(
                f"potential shaping gamma {self.shaping_gamma} must equal training gamma {float(c['gamma'])}"
            )
        self.vector_env = MAVUAVVectorEnv(
            int(c["num_envs"]), self.environment_config, seed=int(c["seed"]), profile=c["environment_profile"],
        )
        if c["actor_variant"] == "vanilla":
            self.actors = IndependentActors(
                hidden_dim=int(c["hidden_dim"]),
                log_std_init=float(c["actor_log_std_init"]),
            ).to(self.device)
        elif c["actor_variant"] == "hrta":
            self.actors = HRTAIndependentActors(
                entity_dim=int(c["hrta_entity_dim"]),
                role_dim=int(c["hrta_role_dim"]),
                fusion_hidden_dim=int(c["hrta_fusion_hidden_dim"]),
                action_dim=3,
            ).to(self.device)
        elif c["actor_variant"] == "structured_uniform":
            self.actors = StructuredUniformIndependentActors(
                entity_dim=int(c["hrta_entity_dim"]),
                role_dim=int(c["hrta_role_dim"]),
                fusion_hidden_dim=int(c["hrta_fusion_hidden_dim"]),
                action_dim=3,
            ).to(self.device)
        elif c["actor_variant"] == "recurrent":
            self.actors = RecurrentIndependentActors(
                observation_dim=OBS_DIM, action_dim=3, hidden_dim=int(c["hidden_dim"]),
                recurrent_hidden_dim=int(c["recurrent_hidden_dim"]),
            ).to(self.device)
        elif c["actor_variant"] == "tam":
            self.actors = TAMIndependentActors(
                observation_dim=OBS_DIM, action_dim=3,
                recurrent_hidden_dim=int(c["tam_actor_gru_hidden_dim"]),
                hidden_layers=tuple(c["tam_actor_hidden_layers"]),
                log_std_init=float(c["actor_log_std_init"]),
                state_memory=bool(c["tam_state_memory"]),
            ).to(self.device)
        elif c["actor_variant"] in LEGACY_PCTA_FAMILY:
            self.actors = PCTAIndependentActors(
                observation_dim=OBS_DIM, action_dim=3,
                context_dim=int(c["pcta_context_dim"]),
                enemy_dim=int(c["pcta_enemy_dim"]),
                hidden_dim=int(c["pcta_hidden_dim"]),
                attention_mode="uniform" if c["actor_variant"] == "pcta_uniform" else "learned",
                log_std_init=float(c["actor_log_std_init"]),
            ).to(self.device)
        elif c["actor_variant"] == PCTA_V2_VARIANT:
            self.actors = PCTAv2IndependentActors(
                observation_dim=OBS_DIM, action_dim=3,
                context_dim=int(c["pcta_v2_context_dim"]),
                enemy_dim=int(c["pcta_v2_enemy_dim"]),
                target_dim=int(c["pcta_v2_target_dim"]),
                attention_heads=int(c["pcta_v2_attention_heads"]),
                hidden_dim=int(c["pcta_hidden_dim"]),
                log_std_init=float(c["actor_log_std_init"]),
            ).to(self.device)
        else:
            raise ValueError(
                "actor_variant must be 'vanilla', 'hrta', 'structured_uniform', 'recurrent', "
                "'tam', 'pcta', 'pcta_attention_only', 'pcta_uniform' or 'pcta_v2'"
            )
        if c["actor_variant"] in LEGACY_PCTA_FAMILY and float(c["pcta_consistency_coef"]) < 0.0:
            raise ValueError("pcta_consistency_coef cannot be negative")
        if c["critic_variant"] == "tam_attention":
            self.critic = TAMAttentionCritic(
                GLOBAL_STATE_DIM,
                recurrent_hidden_dim=int(c["tam_critic_gru_hidden_dim"]),
                token_dim=int(c["tam_token_dim"]),
                attention_heads=int(c["tam_attention_heads"]),
                hidden_layers=tuple(c["tam_critic_hidden_layers"]),
                state_memory=bool(c["tam_state_memory"]),
                attention=bool(c["tam_attention"]),
                inactive_mask=bool(c["tam_inactive_mask"]),
            ).to(self.device)
        elif c["critic_variant"] == "relational":
            self.critic = RelationalCentralizedCritic(GLOBAL_STATE_DIM).to(self.device)
        else:
            self.critic = CentralizedCritic(GLOBAL_STATE_DIM, int(c["hidden_dim"])).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=float(c["actor_learning_rate"])) for actor in self.actors.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(c["critic_learning_rate"]))
        if self.credit_enabled:
            self.credit_component_names = credit_component_names(c["method_variant"])
            self.credit_component_weights = component_weights(c["method_variant"])
            self.credit_critic = ActionMarginalCreditCritic(
                len(self.credit_component_names), hidden_dim=int(c["hidden_dim"]),
            ).to(self.device)
            self.credit_critic_optimizer = torch.optim.Adam(
                self.credit_critic.parameters(), lr=float(c["critic_learning_rate"]),
            )
        else:
            self.credit_component_names = ()
            self.credit_component_weights = np.empty(0, dtype=np.float32)
            self.credit_critic = None
            self.credit_critic_optimizer = None
        self.buffer = self.make_buffer(int(c["rollout_steps"]))
        self.observations, self.global_states, self.active_masks, _ = self.vector_env.reset()
        if self.is_recurrent:
            self.actor_hidden_states = np.zeros(
                (int(c["num_envs"]), len(RED_IDS), self.actor_recurrent_hidden_dim), dtype=np.float32,
            )
            self.actor_recurrent_masks = np.zeros((int(c["num_envs"]), len(RED_IDS)), dtype=np.float32)
        if self.is_tam:
            self.critic_hidden_states = np.zeros(
                (int(c["num_envs"]), int(c["tam_critic_gru_hidden_dim"])), dtype=np.float32,
            )
            self.critic_recurrent_masks = np.zeros(int(c["num_envs"]), dtype=np.float32)
        self.env_steps = 0
        self.completed_episodes: list[dict[str, Any]] = []
        self.last_rollout_metrics = self._empty_rollout_metrics()

    def _empty_rollout_metrics(self) -> dict[str, Any]:
        return {
            "method_variant": self.config["method_variant"],
            "agp_raw_mean": 0.0,
            "agp_raw_mean_abs": 0.0,
            "agp_shaping_mean": 0.0,
            "agp_shaping_mean_abs": 0.0,
        }

    @property
    def actor_architecture(self) -> dict[str, Any]:
        if self.config["actor_variant"] == PCTA_V2_VARIANT:
            return {
                "observation_dim": OBS_DIM, "raw_observation_dim": OBS_DIM,
                "context_input_dim": 44,
                "context_dim": int(self.config["pcta_v2_context_dim"]),
                "enemy_block_dim": 14,
                "enemy_dim": int(self.config["pcta_v2_enemy_dim"]),
                "enemy_slots": 4,
                "attention_type": "additive",
                "attention_heads": int(self.config["pcta_v2_attention_heads"]),
                "pursuit_progress_index": 12,
                "target_dim": int(self.config["pcta_v2_target_dim"]),
                "policy_fusion_dim": OBS_DIM + int(self.config["pcta_v2_target_dim"]) + 4,
                "head_hidden_dim": int(self.config["pcta_hidden_dim"]),
                "action_dim": 3,
                "full_observation_residual": True,
            }
        if self.config["actor_variant"] in LEGACY_PCTA_FAMILY:
            return {
                "observation_dim": OBS_DIM, "context_input_dim": 44,
                "context_dim": int(self.config["pcta_context_dim"]),
                "enemy_block_dim": 14, "enemy_dim": int(self.config["pcta_enemy_dim"]),
                "enemy_slots": 4, "head_hidden_dim": int(self.config["pcta_hidden_dim"]),
                "action_dim": 3,
                "attention_mode": self.pcta_attention_mode,
            }
        if self.config["actor_variant"] == "recurrent":
            return {
                "observation_dim": OBS_DIM,
                "encoder_dim": int(self.config["hidden_dim"]),
                "recurrent_hidden_dim": int(self.config["recurrent_hidden_dim"]),
                "head_dim": int(self.config["hidden_dim"]),
                "action_dim": 3,
            }
        if self.config["actor_variant"] == "tam":
            return {
                "observation_dim": OBS_DIM, "raw_observation_dim": OBS_DIM,
                "state_memory_first": True,
                "gru_input_dim": OBS_DIM,
                "recurrent_hidden_dim": int(self.config["tam_actor_gru_hidden_dim"]),
                "temporal_projection_dim": OBS_DIM,
                "fusion_dim": 2 * OBS_DIM,
                "hidden_layers": list(self.config["tam_actor_hidden_layers"]),
                "action_dim": 3,
                "state_memory": bool(self.config["tam_state_memory"]),
            }
        if self.config["actor_variant"] in ("hrta", "structured_uniform"):
            return {
                "entity_dim": int(self.config["hrta_entity_dim"]),
                "role_dim": int(self.config["hrta_role_dim"]),
                "fusion_hidden_dim": int(self.config["hrta_fusion_hidden_dim"]),
                "action_dim": 3,
            }
        return {"hidden_dim": int(self.config["hidden_dim"]), "action_dim": 3}

    @property
    def actor_parameter_counts(self) -> dict[str, Any]:
        per_agent = [sum(parameter.numel() for parameter in actor.parameters()) for actor in self.actors.actors]
        return {"per_agent": per_agent, "total": sum(per_agent)}

    @property
    def critic_architecture(self) -> dict[str, Any]:
        if self.config["critic_variant"] == "tam_attention":
            return self.critic.architecture()
        if self.config["critic_variant"] == "relational":
            return RelationalCentralizedCritic.architecture()
        return {"state_dim": GLOBAL_STATE_DIM, "hidden_dim": int(self.config["hidden_dim"])}

    @property
    def critic_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.critic.parameters())

    @property
    def is_recurrent(self) -> bool:
        return self.config["actor_variant"] in ("recurrent", "tam")

    @property
    def is_tam(self) -> bool:
        return self.config["actor_variant"] == "tam"

    @property
    def actor_recurrent_hidden_dim(self) -> int:
        field = "tam_actor_gru_hidden_dim" if self.is_tam else "recurrent_hidden_dim"
        return int(self.config[field])

    @property
    def recurrent_sequence_length(self) -> int:
        field = "tam_recurrent_sequence_length" if self.is_tam else "recurrent_sequence_length"
        return int(self.config[field])

    @property
    def tam_metadata(self) -> dict[str, Any]:
        if not self.is_tam:
            return {}
        return {
            "tam_state_memory": bool(self.config["tam_state_memory"]),
            "tam_inactive_mask": bool(self.config["tam_inactive_mask"]),
            "tam_entropy_regularization": True,
            "tam_attention_critic": bool(self.config["tam_attention"]),
            "tam_actor_gru_hidden_dim": int(self.config["tam_actor_gru_hidden_dim"]),
            "tam_critic_gru_hidden_dim": int(self.config["tam_critic_gru_hidden_dim"]),
            "independent_actor_count": len(RED_IDS), "sequential_happo_update": True,
        }

    @property
    def pcta_attention_mode(self) -> str | None:
        if self.config["actor_variant"] not in PCTA_FAMILY:
            return None
        if self.config["actor_variant"] == PCTA_V2_VARIANT:
            return "additive_multihead"
        return "uniform" if self.config["actor_variant"] == "pcta_uniform" else "learned"

    @property
    def pcta_metadata(self) -> dict[str, Any]:
        if self.config["actor_variant"] == PCTA_V2_VARIANT:
            return {
                "attention_mode": self.pcta_attention_mode,
                "pcta_v2_auxiliary_consistency": False,
                "pcta_v2_attention_heads": int(self.config["pcta_v2_attention_heads"]),
                "pcta_v2_context_dim": int(self.config["pcta_v2_context_dim"]),
                "pcta_v2_enemy_dim": int(self.config["pcta_v2_enemy_dim"]),
                "pcta_v2_target_dim": int(self.config["pcta_v2_target_dim"]),
                "pcta_v2_diagnostics_version": 2,
            }
        if self.config["actor_variant"] in LEGACY_PCTA_FAMILY:
            return {
                "attention_mode": self.pcta_attention_mode,
                "pcta_consistency_coef": float(self.config["pcta_consistency_coef"]),
                "effective_pcta_consistency_coef": float(self.config["pcta_consistency_coef"]),
            }
        return {}

    def make_buffer(self, horizon: int) -> RolloutBuffer:
        if self.is_tam:
            return TAMRolloutBuffer(
                horizon, int(self.config["num_envs"]), self.actor_recurrent_hidden_dim,
                int(self.config["tam_critic_gru_hidden_dim"]),
            )
        if self.is_recurrent:
            return RecurrentRolloutBuffer(
                horizon, int(self.config["num_envs"]), self.actor_recurrent_hidden_dim,
            )
        if self.credit_enabled:
            return CreditRolloutBuffer(
                horizon, int(self.config["num_envs"]), len(self.credit_component_names),
            )
        return RolloutBuffer(horizon, int(self.config["num_envs"]))

    @property
    def credit_metadata(self) -> dict[str, Any]:
        if not self.credit_enabled:
            return {}
        assert self.credit_critic is not None
        return {
            "credit_method": (
                "action_marginal_team" if self.config["method_variant"] == CF_METHOD
                else "role_decomposed_action_marginal"
            ),
            "credit_component_names": list(self.credit_component_names),
            "credit_component_weights": self.credit_component_weights.tolist(),
            "credit_critic_architecture": self.credit_critic.architecture(),
            "counterfactual_action_sampling": False,
            "credit_estimator_version": 2,
        }

    def _validate_actor_architecture(self, data: Mapping[str, Any]) -> None:
        checkpoint_variant = data.get("actor_variant", data.get("trainer_config", data.get("config", {})).get("actor_variant", "vanilla"))
        checkpoint_architecture = data.get("actor_architecture")
        if checkpoint_variant != self.config["actor_variant"]:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_variant!r} "
                f"current={self.config['actor_variant']!r}"
            )
        if checkpoint_variant == "pcta" and checkpoint_architecture is not None:
            checkpoint_architecture = dict(checkpoint_architecture)
            checkpoint_architecture.setdefault("attention_mode", "learned")
        if checkpoint_variant in ("hrta", "structured_uniform", "recurrent", "tam", *PCTA_FAMILY) and checkpoint_architecture != self.actor_architecture:
            raise RuntimeError(
                f"incompatible actor architecture: checkpoint={checkpoint_architecture!r} "
                f"current={self.actor_architecture!r}"
            )

    def _validate_critic_architecture(self, data: Mapping[str, Any]) -> None:
        saved_config = data.get("trainer_config", data.get("config", {}))
        checkpoint_variant = data.get("critic_variant", saved_config.get("critic_variant", "mlp"))
        if checkpoint_variant != self.config["critic_variant"]:
            raise RuntimeError(
                f"incompatible critic variant: checkpoint={checkpoint_variant!r} "
                f"current={self.config['critic_variant']!r}"
            )
        checkpoint_architecture = data.get("critic_architecture")
        if checkpoint_variant == "relational" and checkpoint_architecture != self.critic_architecture:
            raise RuntimeError(
                f"incompatible critic architecture: checkpoint={checkpoint_architecture!r} "
                f"current={self.critic_architecture!r}"
            )
        if checkpoint_architecture is not None and checkpoint_architecture != self.critic_architecture:
            raise RuntimeError(
                f"incompatible critic architecture: checkpoint={checkpoint_architecture!r} "
                f"current={self.critic_architecture!r}"
            )

    def collect_rollout(self) -> list[dict[str, Any]]:
        if self.is_recurrent:
            return self._collect_recurrent_rollout()
        self.buffer.reset(); completed = []
        raw_terms: list[np.ndarray] = []
        shaping_terms: list[np.ndarray] = []
        for _ in range(self.buffer.horizon):
            actions, log_probs = [], []
            with torch.no_grad():
                for agent, actor in enumerate(self.actors.actors):
                    action, log_prob = actor.sample(torch.as_tensor(self.observations[:, agent], device=self.device))
                    actions.append(action.cpu().numpy()); log_probs.append(log_prob.cpu().numpy())
                values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
            action_array = np.stack(actions, axis=1); log_prob_array = np.stack(log_probs, axis=1)
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(action_array)
            done = np.logical_or(terminated, truncated)
            if self.agp_enabled:
                training_rewards, raw, shaping = apply_agp(
                    rewards,
                    self.observations,
                    next_obs,
                    done,
                    float(self.environment_config["normalization"]["distance_scale"]),
                    gamma=float(self.config["gamma"]),
                    agp_lambda=float(self.config["agp_lambda"]),
                )
            else:
                training_rewards = rewards
                raw = np.zeros(self.buffer.num_envs, dtype=np.float64)
                shaping = np.zeros(self.buffer.num_envs, dtype=np.float64)
            raw_terms.append(raw)
            shaping_terms.append(shaping)
            if self.credit_enabled:
                if not isinstance(self.buffer, CreditRolloutBuffer):
                    raise TypeError("counterfactual methods require CreditRolloutBuffer")
                credit_rewards = extract_credit_components(
                    infos, rewards, self.config["method_variant"],
                )
                self.buffer.insert(
                    self.observations, self.global_states, action_array, log_prob_array,
                    training_rewards, values, terminated, truncated, self.active_masks,
                    credit_rewards=credit_rewards,
                )
            else:
                self.buffer.insert(self.observations, self.global_states, action_array, log_prob_array, training_rewards, values, terminated, truncated, self.active_masks)
            completed.extend(info["episode_summary"] for info in infos if "episode_summary" in info)
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad(): last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]))
        if self.credit_enabled:
            if not isinstance(self.buffer, CreditRolloutBuffer) or self.credit_critic is None:
                raise TypeError("counterfactual rollout state is incomplete")
            with torch.no_grad():
                flat_states = torch.as_tensor(
                    self.buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device,
                )
                credit_values = self.credit_critic.values(flat_states).reshape(
                    self.buffer.horizon, self.buffer.num_envs, -1,
                ).cpu().numpy()
                last_credit_values = self.credit_critic.values(
                    torch.as_tensor(self.global_states, device=self.device),
                ).cpu().numpy()
            self.buffer.compute_credit_returns(
                credit_values, last_credit_values,
                float(self.config["gamma"]), float(self.config["gae_lambda"]),
            )
        raw_values = np.concatenate(raw_terms) if raw_terms else np.zeros(1)
        shaping_values = np.concatenate(shaping_terms) if shaping_terms else np.zeros(1)
        self.last_rollout_metrics = {
            "method_variant": self.config["method_variant"],
            "agp_raw_mean": float(np.mean(raw_values)),
            "agp_raw_mean_abs": float(np.mean(np.abs(raw_values))),
            "agp_shaping_mean": float(np.mean(shaping_values)),
            "agp_shaping_mean_abs": float(np.mean(np.abs(shaping_values))),
        }
        self.completed_episodes.extend(completed)
        return completed

    def _collect_recurrent_rollout(self) -> list[dict[str, Any]]:
        """Collect a rollout while preserving hidden state across rollout boundaries."""
        if not isinstance(self.buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent actor requires RecurrentRolloutBuffer")
        self.buffer.reset()
        completed: list[dict[str, Any]] = []
        for _ in range(self.buffer.horizon):
            actions: list[np.ndarray] = []
            log_probs: list[np.ndarray] = []
            next_hidden = np.empty_like(self.actor_hidden_states)
            next_critic_hidden = np.empty_like(self.critic_hidden_states) if self.is_tam else None
            with torch.no_grad():
                for agent, actor in enumerate(self.actors.actors):
                    action, log_prob, hidden = actor.sample_step(
                        torch.as_tensor(self.observations[:, agent], device=self.device),
                        torch.as_tensor(self.actor_hidden_states[:, agent], device=self.device),
                        torch.as_tensor(self.actor_recurrent_masks[:, agent], device=self.device),
                    )
                    actions.append(action.cpu().numpy())
                    log_probs.append(log_prob.cpu().numpy())
                    next_hidden[:, agent] = hidden.cpu().numpy()
                if self.is_tam:
                    values_t, critic_hidden_t = self.critic.forward_step(
                        torch.as_tensor(self.global_states, device=self.device),
                        torch.as_tensor(self.critic_hidden_states, device=self.device),
                        torch.as_tensor(self.critic_recurrent_masks, device=self.device),
                    )
                    values = values_t.cpu().numpy()
                    next_critic_hidden[:] = critic_hidden_t.cpu().numpy()
                else:
                    values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
            action_array = np.stack(actions, axis=1)
            log_prob_array = np.stack(log_probs, axis=1)
            if self.is_tam and bool(self.config["tam_inactive_mask"]):
                action_array *= self.active_masks[:, :, None]
            next_obs, next_states, rewards, terminated, truncated, next_masks, infos = self.vector_env.step(action_array)
            done = np.logical_or(terminated, truncated)
            if self.is_tam and not bool(self.config["tam_inactive_mask"]):
                next_recurrent_masks = np.broadcast_to((~done)[:, None], next_masks.shape).astype(np.float32).copy()
            else:
                next_recurrent_masks = next_masks.astype(np.float32) * (~done)[:, None].astype(np.float32)
            next_hidden *= next_recurrent_masks[:, :, None]
            if self.is_tam:
                assert isinstance(self.buffer, TAMRolloutBuffer) and next_critic_hidden is not None
                next_critic_masks = (~done).astype(np.float32)
                next_critic_hidden *= next_critic_masks[:, None]
                self.buffer.insert(
                    self.observations, self.global_states, action_array, log_prob_array, rewards, values,
                    terminated, truncated, self.active_masks, self.actor_hidden_states,
                    self.actor_recurrent_masks, next_hidden, self.critic_hidden_states,
                    self.critic_recurrent_masks, next_critic_hidden,
                )
            else:
                self.buffer.insert(
                    self.observations, self.global_states, action_array, log_prob_array, rewards, values,
                    terminated, truncated, self.active_masks, self.actor_hidden_states,
                    self.actor_recurrent_masks, next_hidden,
                )
            completed.extend(info["episode_summary"] for info in infos if "episode_summary" in info)
            self.observations, self.global_states, self.active_masks = next_obs, next_states, next_masks
            self.actor_hidden_states = next_hidden
            self.actor_recurrent_masks = next_recurrent_masks
            if self.is_tam:
                self.critic_hidden_states = next_critic_hidden
                self.critic_recurrent_masks = next_critic_masks
            self.env_steps += self.buffer.num_envs
        with torch.no_grad():
            if self.is_tam:
                last_values, _ = self.critic.forward_step(
                    torch.as_tensor(self.global_states, device=self.device),
                    torch.as_tensor(self.critic_hidden_states, device=self.device),
                    torch.as_tensor(self.critic_recurrent_masks, device=self.device),
                )
                last_values = last_values.cpu().numpy()
            else:
                last_values = self.critic(torch.as_tensor(self.global_states, device=self.device)).cpu().numpy()
        self.buffer.compute_returns_and_advantages(
            last_values, float(self.config["gamma"]), float(self.config["gae_lambda"]),
        )
        self.last_rollout_metrics = {
            "method_variant": self.config["method_variant"],
            "agp_raw_mean": 0.0, "agp_raw_mean_abs": 0.0,
            "agp_shaping_mean": 0.0, "agp_shaping_mean_abs": 0.0,
        }
        self.completed_episodes.extend(completed)
        return completed

    def _train_credit_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        active_masks: torch.Tensor,
    ) -> dict[str, float]:
        if self.credit_critic is None or self.credit_critic_optimizer is None:
            raise RuntimeError("credit critic is unavailable")
        c = self.config
        total = len(states)
        mini = int(c["minibatch_size"])
        value_sse = baseline_sse = 0.0
        value_count = baseline_count = 0
        per_agent_sse = np.zeros(len(RED_IDS), dtype=np.float64)
        per_agent_count = np.zeros(len(RED_IDS), dtype=np.int64)
        for _ in range(int(c["ppo_epochs"])):
            sample_order = self.rng.permutation(total)
            for start in range(0, total, mini):
                idx = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                value_error = (self.credit_critic.values(states[idx]) - targets[idx]).square()
                value_loss = value_error.mean()
                batch_baseline_sse = torch.zeros((), device=self.device)
                batch_baseline_count = 0
                for agent in range(len(RED_IDS)):
                    active = active_masks[idx, agent] > 0.5
                    if not active.any():
                        continue
                    prediction = self.credit_critic.baseline_for_agent(
                        states[idx][active], actions[idx][active], agent,
                    )
                    error = (prediction - targets[idx][active]).square()
                    batch_baseline_sse = batch_baseline_sse + error.sum()
                    count = error.numel()
                    batch_baseline_count += count
                    per_agent_sse[agent] += float(error.detach().sum().item())
                    per_agent_count[agent] += count
                baseline_loss = (
                    batch_baseline_sse / batch_baseline_count
                    if batch_baseline_count else torch.zeros((), device=self.device)
                )
                loss = 0.5 * (value_loss + baseline_loss)
                self.credit_critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.credit_critic.parameters(), float(c["max_grad_norm"]))
                self.credit_critic_optimizer.step()
                value_sse += float(value_error.detach().sum().item())
                value_count += value_error.numel()
                baseline_sse += float(batch_baseline_sse.detach().item())
                baseline_count += batch_baseline_count
        metrics = {
            "credit_value_loss": value_sse / max(value_count, 1),
            "credit_baseline_loss": baseline_sse / max(baseline_count, 1),
        }
        metrics["credit_total_loss"] = 0.5 * (
            metrics["credit_value_loss"] + metrics["credit_baseline_loss"]
        )
        for agent in range(len(RED_IDS)):
            metrics[f"credit_baseline_loss_{agent}"] = (
                per_agent_sse[agent] / per_agent_count[agent]
                if per_agent_count[agent] else 0.0
            )
        return metrics

    def _action_marginal_actor_advantages(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        component_returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Freeze pre-update B_i residuals before any optimizer sees this rollout."""
        if self.credit_critic is None:
            raise RuntimeError("credit critic is unavailable")
        advantages: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        with torch.no_grad():
            for agent in range(len(RED_IDS)):
                baseline = self.credit_critic.baseline_for_agent(states, actions, agent)
                per_component = component_residual(component_returns, baseline)
                residuals.append(per_component)
                if self.config["method_variant"] == RDC_METHOD:
                    advantages.append(combine_rdc_component_residual(component_returns, baseline))
                else:
                    advantages.append(per_component.squeeze(-1))
        frozen_advantages = torch.stack(advantages, dim=-1).detach()
        frozen_residuals = torch.stack(residuals, dim=1).detach()
        self.last_credit_actor_advantages = frozen_advantages.clone()
        self.last_credit_component_residuals = frozen_residuals.clone()
        return frozen_advantages, frozen_residuals

    def update(self) -> dict[str, Any]:
        if self.is_recurrent:
            return self._update_recurrent()
        c = self.config
        num_agents = len(RED_IDS)
        observations = torch.as_tensor(self.buffer.observations.reshape(-1, num_agents, OBS_DIM), device=self.device)
        actions = torch.as_tensor(self.buffer.actions.reshape(-1, num_agents, 3), device=self.device)
        old_log_probs = torch.as_tensor(self.buffer.log_probs.reshape(-1, num_agents), device=self.device)
        active_masks = torch.as_tensor(self.buffer.active_masks.reshape(-1, num_agents), device=self.device)
        advantages = torch.as_tensor(self.buffer.advantages.reshape(-1), device=self.device)
        states = torch.as_tensor(self.buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device)
        returns = torch.as_tensor(self.buffer.returns.reshape(-1), device=self.device)
        credit_metrics: dict[str, float] = {}
        actor_advantages: torch.Tensor | None = None
        component_residuals: torch.Tensor | None = None
        credit_targets: torch.Tensor | None = None
        if self.credit_enabled:
            if not isinstance(self.buffer, CreditRolloutBuffer):
                raise TypeError("counterfactual methods require CreditRolloutBuffer")
            credit_targets = torch.as_tensor(
                self.buffer.credit_returns.reshape(-1, self.buffer.component_count),
                device=self.device,
            )
            actor_advantages, component_residuals = self._action_marginal_actor_advantages(
                states, actions, credit_targets,
            )
            for agent in range(num_agents):
                active = active_masks[:, agent] > 0.5
                active_advantage = actor_advantages[active, agent]
                credit_metrics[f"credit_adv_mean_abs_{agent}"] = (
                    float(active_advantage.abs().mean().item()) if active.any() else 0.0
                )
                credit_metrics[f"credit_adv_std_{agent}"] = (
                    float(active_advantage.std(unbiased=False).item()) if active.any() else 0.0
                )
            if self.config["method_variant"] == RDC_METHOD:
                assert component_residuals is not None
                for index, name in enumerate(self.credit_component_names):
                    active_values = [
                        component_residuals[active_masks[:, agent] > 0.5, agent, index]
                        for agent in range(num_agents)
                        if (active_masks[:, agent] > 0.5).any()
                    ]
                    credit_metrics[f"rdc_{name}_credit_mean_abs"] = (
                        float(torch.cat(active_values).abs().mean().item()) if active_values else 0.0
                    )
            for index, name in enumerate(self.credit_component_names):
                credit_metrics[f"credit_component_return_std_{name}"] = float(
                    credit_targets[:, index].std(unbiased=False).item()
                )
        factor = torch.ones_like(advantages)
        order = [int(v) for v in self.rng.permutation(num_agents)]
        actor_losses: list[list[float]] = [[] for _ in RED_IDS]; entropies: list[float] = []
        pcta_loss_sum = 0.0
        pcta_pairs = 0
        pcta_attention_entropy_sum = 0.0
        pcta_switches = 0
        pcta_v2_max_attention_sum = 0.0
        pcta_v2_bias_sum = 0.0
        pcta_v2_valid_target_states = 0
        pcta_v2_multi_target_states = 0
        pcta_v2_head_entropy_sum = 0.0
        pcta_v2_head_entropy_count = 0
        pcta_v2_head_max_sum = 0.0
        pcta_v2_head_max_count = 0
        pcta_v2_head_disagreement_sum = 0.0
        pcta_v2_head_disagreement_count = 0
        if c["actor_variant"] in PCTA_FAMILY:
            self.last_pcta_factor_history = [factor.detach().cpu().numpy().copy()]
        clip = float(c["clip_coef"]); mini = int(c["minibatch_size"]); total = len(advantages)
        if self.credit_enabled:
            self.last_credit_normalized_advantages = torch.zeros(
                (total, num_agents), device=self.device,
            )
        for agent in order:
            optimizer = self.actor_optimizers[agent]
            active = active_masks[:, agent] > 0.5
            agent_advantages = (
                actor_advantages[:, agent] if actor_advantages is not None else advantages
            )
            if self.credit_enabled:
                normalized, degenerate = normalize_credit_advantage(agent_advantages, active)
                credit_metrics[f"credit_degenerate_agent_{agent}"] = float(degenerate)
            else:
                normalized = agent_advantages.clone()
            if not self.credit_enabled and active.any():
                normalized = (
                    agent_advantages - agent_advantages[active].mean()
                ) / agent_advantages[active].std(unbiased=False).clamp_min(1e-8)
            if self.credit_enabled:
                self.last_credit_normalized_advantages[:, agent] = normalized.detach()
            for _ in range(int(c["ppo_epochs"])):
                sample_order = self.rng.permutation(total)
                for start in range(0, total, mini):
                    idx = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                    idx = idx[active[idx]]
                    if not len(idx): continue
                    new_log_prob, entropy = self.actors.actors[agent].evaluate_actions(observations[idx, agent], actions[idx, agent])
                    ratio = torch.exp(new_log_prob - old_log_probs[idx, agent])
                    effective = (factor[idx] * normalized[idx]).detach()
                    policy_loss = -torch.minimum(ratio * effective, ratio.clamp(1.0 - clip, 1.0 + clip) * effective).mean()
                    loss = policy_loss - float(c["entropy_coef"]) * entropy.mean()
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.actors.actors[agent].parameters(), float(c["max_grad_norm"])); optimizer.step()
                    actor_losses[agent].append(float(policy_loss.item())); entropies.append(float(entropy.mean().item()))
            if c["actor_variant"] in LEGACY_PCTA_FAMILY:
                temporal = pursuit_consistency(
                    self.actors.actors[agent],
                    observations.reshape(self.buffer.horizon, self.buffer.num_envs, num_agents, OBS_DIM)[:, :, agent],
                    torch.as_tensor(self.buffer.terminated, device=self.device),
                    torch.as_tensor(self.buffer.truncated, device=self.device),
                    active_masks.reshape(self.buffer.horizon, self.buffer.num_envs, num_agents)[:, :, agent],
                )
                if temporal.valid_pairs:
                    if c["actor_variant"] == "pcta":
                        weighted_consistency = float(c["pcta_consistency_coef"]) * temporal.raw_loss
                        optimizer.zero_grad(); weighted_consistency.backward()
                        nn.utils.clip_grad_norm_(self.actors.actors[agent].parameters(), float(c["max_grad_norm"]))
                        optimizer.step()
                    pcta_loss_sum += float(temporal.raw_loss.detach().item()) * temporal.valid_pairs
                    pcta_pairs += temporal.valid_pairs
                    pcta_attention_entropy_sum += temporal.attention_entropy_sum
                    pcta_switches += temporal.target_switches
            with torch.no_grad():
                new_all, _ = self.actors.actors[agent].evaluate_actions(observations[:, agent], actions[:, agent])
                factor = preceding_factor_update(factor, old_log_probs[:, agent], new_all, active_masks[:, agent])
            if c["actor_variant"] in PCTA_FAMILY:
                self.last_pcta_factor_history.append(factor.detach().cpu().numpy().copy())
            if c["actor_variant"] == PCTA_V2_VARIANT:
                temporal = target_behavior_diagnostics(
                    self.actors.actors[agent],
                    observations.reshape(self.buffer.horizon, self.buffer.num_envs, num_agents, OBS_DIM)[:, :, agent],
                    torch.as_tensor(self.buffer.terminated, device=self.device),
                    torch.as_tensor(self.buffer.truncated, device=self.device),
                    active_masks.reshape(self.buffer.horizon, self.buffer.num_envs, num_agents)[:, :, agent],
                )
                pcta_pairs += temporal.valid_pairs
                pcta_attention_entropy_sum += temporal.attention_entropy_sum
                pcta_switches += temporal.target_switches
                pcta_v2_max_attention_sum += temporal.max_attention_weight_sum
                pcta_v2_bias_sum += temporal.pursuit_bias_mean
                pcta_v2_valid_target_states += temporal.valid_target_states
                pcta_v2_multi_target_states += temporal.multi_target_states
                pcta_v2_head_entropy_sum += temporal.head_normalized_entropy_sum
                pcta_v2_head_entropy_count += temporal.head_normalized_entropy_count
                pcta_v2_head_max_sum += temporal.head_max_attention_sum
                pcta_v2_head_max_count += temporal.head_max_attention_count
                pcta_v2_head_disagreement_sum += temporal.head_disagreement_sum
                pcta_v2_head_disagreement_count += temporal.head_disagreement_count
        critic_losses = []
        for _ in range(int(c["ppo_epochs"])):
            sample_order = self.rng.permutation(total)
            for start in range(0, total, mini):
                idx = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                value_loss = (self.critic(states[idx]) - returns[idx]).square().mean()
                self.critic_optimizer.zero_grad(); (float(c["value_loss_coef"]) * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"])); self.critic_optimizer.step()
                critic_losses.append(float(value_loss.item()))
        if self.credit_enabled:
            assert credit_targets is not None
            credit_metrics.update(self._train_credit_critic(
                states, actions, credit_targets, active_masks,
            ))
        metrics: dict[str, Any] = {f"actor_{i}_loss": float(np.mean(actor_losses[i])) if actor_losses[i] else 0.0 for i in range(num_agents)}
        metrics.update({"actor_loss": float(np.mean([v for rows in actor_losses for v in rows])), "critic_loss": float(np.mean(critic_losses)), "entropy": float(np.mean(entropies)), "agent_update_order": order})
        metrics.update(credit_metrics)
        if c["actor_variant"] in LEGACY_PCTA_FAMILY:
            raw_consistency = pcta_loss_sum / pcta_pairs if pcta_pairs else 0.0
            metrics.update({
                "pcta_consistency_loss": raw_consistency,
                "pcta_consistency_weighted_loss": float(c["pcta_consistency_coef"]) * raw_consistency,
                "pcta_valid_temporal_pairs": pcta_pairs,
                "pcta_attention_entropy": pcta_attention_entropy_sum / pcta_pairs if pcta_pairs else 0.0,
                "pcta_target_switch_rate": pcta_switches / pcta_pairs if pcta_pairs else 0.0,
            })
        elif c["actor_variant"] == PCTA_V2_VARIANT:
            metrics.update({
                "pcta_v2_attention_entropy": pcta_attention_entropy_sum / pcta_pairs if pcta_pairs else 0.0,
                "pcta_v2_target_switch_rate": pcta_switches / pcta_pairs if pcta_pairs else 0.0,
                "pcta_v2_valid_temporal_pairs": pcta_pairs,
                "pcta_v2_pursuit_bias_mean": pcta_v2_bias_sum / num_agents,
                "pcta_v2_max_attention_weight": pcta_v2_max_attention_sum / pcta_pairs if pcta_pairs else 0.0,
                "pcta_v2_ensemble_attention_entropy": pcta_attention_entropy_sum / pcta_pairs if pcta_pairs else 0.0,
                "pcta_v2_ensemble_max_attention_weight": pcta_v2_max_attention_sum / pcta_pairs if pcta_pairs else 0.0,
                "pcta_v2_head_normalized_entropy": pcta_v2_head_entropy_sum / pcta_v2_head_entropy_count if pcta_v2_head_entropy_count else 0.0,
                "pcta_v2_head_max_attention_weight": pcta_v2_head_max_sum / pcta_v2_head_max_count if pcta_v2_head_max_count else 0.0,
                "pcta_v2_head_disagreement": pcta_v2_head_disagreement_sum / pcta_v2_head_disagreement_count if pcta_v2_head_disagreement_count else 0.0,
                "pcta_v2_valid_target_states": pcta_v2_valid_target_states,
                "pcta_v2_multi_target_states": pcta_v2_multi_target_states,
            })
        metrics.update(self.last_rollout_metrics)
        if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, float)): raise FloatingPointError("non-finite HAPPO update")
        return metrics

    def _recurrent_sequence_tensors(
        self, agent: int, specs: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer) or not specs:
            raise ValueError("a non-empty recurrent sequence batch is required")
        observations = np.stack([buffer.observations[start:end, env, agent] for env, start, end in specs])
        actions = np.stack([buffer.actions[start:end, env, agent] for env, start, end in specs])
        initial_hidden = np.stack([buffer.actor_hidden_states[start, env, agent] for env, start, _ in specs])
        masks = np.stack([buffer.recurrent_masks[start:end, env, agent] for env, start, end in specs])
        return tuple(torch.as_tensor(value, device=self.device) for value in (
            observations, actions, initial_hidden, masks,
        ))

    def _recurrent_log_probs_all(self, agent: int) -> torch.Tensor:
        """Re-evaluate every transition in ordered TBPTT chunks without padding or drops."""
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent log-prob evaluation requires RecurrentRolloutBuffer")
        result = torch.empty((buffer.horizon, buffer.num_envs), device=self.device)
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for spec in buffer.chunks(self.recurrent_sequence_length):
            groups.setdefault(spec[2] - spec[1], []).append(spec)
        actor = self.actors.actors[agent]
        with torch.no_grad():
            for specs in groups.values():
                obs, actions, initial_hidden, masks = self._recurrent_sequence_tensors(agent, specs)
                log_probs, _, _ = actor.evaluate_actions_sequence(obs, actions, initial_hidden, masks)
                for index, (env, start, end) in enumerate(specs):
                    result[start:end, env] = log_probs[index]
        return result

    def _update_recurrent(self) -> dict[str, Any]:
        buffer = self.buffer
        if not isinstance(buffer, RecurrentRolloutBuffer):
            raise TypeError("recurrent update requires RecurrentRolloutBuffer")
        c = self.config
        old_log_probs = torch.as_tensor(buffer.log_probs, device=self.device)
        active_masks = torch.as_tensor(buffer.active_masks, device=self.device)
        if self.is_tam and not bool(c["tam_inactive_mask"]):
            active_masks = torch.ones_like(active_masks)
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        factor = torch.ones_like(advantages)
        num_agents = len(RED_IDS)
        order = [int(value) for value in self.rng.permutation(num_agents)]
        actor_losses: list[list[float]] = [[] for _ in RED_IDS]
        entropies: list[float] = []
        clip = float(c["clip_coef"])
        mini = int(c["minibatch_size"])
        sequence_length = self.recurrent_sequence_length
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for spec in buffer.chunks(sequence_length):
            groups.setdefault(spec[2] - spec[1], []).append(spec)
        self.last_recurrent_factor_history = [factor.detach().cpu().numpy().copy()]
        for agent in order:
            active = active_masks[:, :, agent] > 0.5
            normalized = advantages.clone()
            if active.any():
                normalized = (
                    advantages - advantages[active].mean()
                ) / advantages[active].std(unbiased=False).clamp_min(1e-8)
            actor = self.actors.actors[agent]
            optimizer = self.actor_optimizers[agent]
            for _ in range(int(c["ppo_epochs"])):
                for length, all_specs in groups.items():
                    chunks_per_batch = max(1, mini // length)
                    shuffled = self.rng.permutation(len(all_specs))
                    for start_index in range(0, len(all_specs), chunks_per_batch):
                        specs = [all_specs[int(index)] for index in shuffled[start_index:start_index + chunks_per_batch]]
                        obs, action_batch, initial_hidden, masks = self._recurrent_sequence_tensors(agent, specs)
                        new_log_prob, entropy, _ = actor.evaluate_actions_sequence(
                            obs, action_batch, initial_hidden, masks,
                        )
                        old = torch.stack([old_log_probs[start:end, env, agent] for env, start, end in specs])
                        batch_active = torch.stack([active_masks[start:end, env, agent] for env, start, end in specs]) > 0.5
                        batch_advantage = torch.stack([normalized[start:end, env] for env, start, end in specs])
                        batch_factor = torch.stack([factor[start:end, env] for env, start, end in specs])
                        if not batch_active.any():
                            continue
                        ratio = torch.exp(new_log_prob[batch_active] - old[batch_active])
                        effective = (batch_factor[batch_active] * batch_advantage[batch_active]).detach()
                        policy_loss = -torch.minimum(
                            ratio * effective,
                            ratio.clamp(1.0 - clip, 1.0 + clip) * effective,
                        ).mean()
                        loss = policy_loss - float(c["entropy_coef"]) * entropy[batch_active].mean()
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), float(c["max_grad_norm"]))
                        optimizer.step()
                        actor_losses[agent].append(float(policy_loss.item()))
                        entropies.append(float(entropy[batch_active].mean().item()))
            new_all = self._recurrent_log_probs_all(agent)
            factor = preceding_factor_update(
                factor, old_log_probs[:, :, agent], new_all, active_masks[:, :, agent],
            )
            self.last_recurrent_factor_history.append(factor.detach().cpu().numpy().copy())

        critic_losses: list[float] = []
        if self.is_tam:
            if not isinstance(buffer, TAMRolloutBuffer):
                raise TypeError("TAM critic requires TAMRolloutBuffer")
            for _ in range(int(c["ppo_epochs"])):
                for length, all_specs in groups.items():
                    chunks_per_batch = max(1, mini // length)
                    shuffled = self.rng.permutation(len(all_specs))
                    for start_index in range(0, len(all_specs), chunks_per_batch):
                        specs = [all_specs[int(index)] for index in shuffled[start_index:start_index + chunks_per_batch]]
                        state_batch = torch.as_tensor(np.stack([
                            buffer.global_states[start:end, env] for env, start, end in specs
                        ]), device=self.device)
                        initial_hidden = torch.as_tensor(np.stack([
                            buffer.critic_hidden_states[start, env] for env, start, _ in specs
                        ]), device=self.device)
                        masks = torch.as_tensor(np.stack([
                            buffer.critic_recurrent_masks[start:end, env] for env, start, end in specs
                        ]), device=self.device)
                        targets = torch.as_tensor(np.stack([
                            buffer.returns[start:end, env] for env, start, end in specs
                        ]), device=self.device)
                        predicted, _ = self.critic.evaluate_values_sequence(state_batch, initial_hidden, masks)
                        value_loss = nn.functional.huber_loss(
                            predicted, targets, delta=float(c["tam_huber_delta"]), reduction="mean",
                        )
                        self.critic_optimizer.zero_grad()
                        (float(c["value_loss_coef"]) * value_loss).backward()
                        nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"]))
                        self.critic_optimizer.step()
                        critic_losses.append(float(value_loss.item()))
        else:
            states = torch.as_tensor(buffer.global_states.reshape(-1, GLOBAL_STATE_DIM), device=self.device)
            returns = torch.as_tensor(buffer.returns.reshape(-1), device=self.device)
            total = len(returns)
            for _ in range(int(c["ppo_epochs"])):
                sample_order = self.rng.permutation(total)
                for start in range(0, total, mini):
                    indices = torch.as_tensor(sample_order[start:start + mini], device=self.device)
                    value_loss = (self.critic(states[indices]) - returns[indices]).square().mean()
                    self.critic_optimizer.zero_grad()
                    (float(c["value_loss_coef"]) * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), float(c["max_grad_norm"]))
                    self.critic_optimizer.step()
                    critic_losses.append(float(value_loss.item()))
        flat_actor_losses = [value for rows in actor_losses for value in rows]
        metrics: dict[str, Any] = {
            f"actor_{agent}_loss": float(np.mean(actor_losses[agent])) if actor_losses[agent] else 0.0
            for agent in range(num_agents)
        }
        metrics.update({
            "actor_loss": float(np.mean(flat_actor_losses)) if flat_actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "agent_update_order": order,
        })
        metrics.update(self.last_rollout_metrics)
        if not all(np.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError("non-finite recurrent HAPPO update")
        return metrics

    def train_update(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        episodes = self.collect_rollout()
        return episodes, self.update()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"environment_version": self.environment_config["environment_version"], "environment_profile": self.config["environment_profile"], "observation_dim": OBS_DIM, "global_state_dim": GLOBAL_STATE_DIM, "actor_variant": self.config["actor_variant"], "critic_variant": self.config["critic_variant"], "method_variant": self.config["method_variant"], "reward_mode": self.reward_mode, "reward_shaping_mode": self.reward_shaping_mode if self.reward_mode not in ROLE_REWARD_MODES else None, "shaping_gamma": self.shaping_gamma if self.reward_mode not in ROLE_REWARD_MODES else None, "training_gamma": float(self.config["gamma"]), "actor_architecture": self.actor_architecture, "critic_architecture": self.critic_architecture, "critic_parameter_count": self.critic_parameter_count, "actors": self.actors.state_dict(), "critic": self.critic.state_dict(), "config": self.config}
        payload.update(self.pcta_metadata)
        payload.update(self.credit_metadata)
        payload.update(self.tam_metadata)
        if self.is_tam:
            payload["algorithm"] = "happo"
        if self.credit_enabled:
            assert self.credit_critic is not None
            payload["credit_critic"] = self.credit_critic.state_dict()
        torch.save(payload, path)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return all state required for an exact continuation of training."""
        state = {
            "format": "happo_training_checkpoint_v1",
            "sampled_steps": int(self.env_steps),
            "environment_version": self.environment_config["environment_version"],
            "environment_profile": self.config["environment_profile"],
            "observation_dim": OBS_DIM,
            "global_state_dim": GLOBAL_STATE_DIM,
            "actor_variant": self.config["actor_variant"],
            "critic_variant": self.config["critic_variant"],
            "method_variant": self.config["method_variant"],
            "reward_shaping_mode": self.reward_shaping_mode if self.reward_mode not in ROLE_REWARD_MODES else None,
            "reward_mode": self.reward_mode,
            "shaping_gamma": self.shaping_gamma if self.reward_mode not in ROLE_REWARD_MODES else None,
            "training_gamma": float(self.config["gamma"]),
            "agp_lambda": float(self.config["agp_lambda"]),
            "actor_architecture": self.actor_architecture,
            "critic_architecture": self.critic_architecture,
            "critic_parameter_count": self.critic_parameter_count,
            "environment_config": deepcopy(self.environment_config),
            "trainer_config": deepcopy(self.config),
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer_states": [optimizer.state_dict() for optimizer in self.actor_optimizers],
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "trainer_numpy_rng": deepcopy(self.rng.bit_generator.state),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rollout_state": {
                "observations": self.observations.copy(),
                "global_states": self.global_states.copy(),
                "active_masks": self.active_masks.copy(),
                "environment_states": self.vector_env.get_env_states(),
                "vector_reset_counts": self.vector_env.reset_counts.copy(),
                "vector_base_seed": self.vector_env.base_seed,
            },
        }
        state.update(self.pcta_metadata)
        state.update(self.credit_metadata)
        state.update(self.tam_metadata)
        if self.is_tam:
            state["algorithm"] = "happo"
        if self.credit_enabled:
            assert self.credit_critic is not None and self.credit_critic_optimizer is not None
            state["credit_critic"] = self.credit_critic.state_dict()
            state["credit_critic_optimizer_state"] = self.credit_critic_optimizer.state_dict()
        if self.is_recurrent:
            state["rollout_state"]["actor_hidden_states"] = self.actor_hidden_states.copy()
            state["rollout_state"]["actor_recurrent_masks"] = self.actor_recurrent_masks.copy()
        if self.is_tam:
            state["rollout_state"]["critic_hidden_states"] = self.critic_hidden_states.copy()
            state["rollout_state"]["critic_recurrent_masks"] = self.critic_recurrent_masks.copy()
        return state

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def load_checkpoint(self, path: str | Path) -> int:
        data = torch.load(path, map_location=self.device, weights_only=False)
        expected = (self.environment_config["environment_version"], OBS_DIM, GLOBAL_STATE_DIM)
        actual = (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim"))
        if actual != expected:
            raise RuntimeError("incompatible checkpoint contract for HAPPO environment")
        checkpoint_mode = data.get("reward_mode", data.get("reward_shaping_mode", "absolute"))
        checkpoint_gamma = float(data.get("shaping_gamma") or 0.0)
        if checkpoint_mode != self.reward_mode or (
            checkpoint_mode == "potential" and not np.isclose(checkpoint_gamma, self.shaping_gamma, rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError("incompatible checkpoint reward shaping contract")
        self._validate_actor_architecture(data)
        self._validate_critic_architecture(data)
        saved_config = data.get("trainer_config", data.get("config", {}))
        if self.config["actor_variant"] in LEGACY_PCTA_FAMILY:
            checkpoint_coef = data.get("pcta_consistency_coef", saved_config.get("pcta_consistency_coef"))
            if checkpoint_coef is None or float(checkpoint_coef) != float(self.config["pcta_consistency_coef"]):
                raise RuntimeError("resume PCTA consistency coefficient mismatch")
            for field in ("pcta_context_dim", "pcta_enemy_dim", "pcta_hidden_dim"):
                if saved_config.get(field) != self.config.get(field):
                    raise RuntimeError(
                        f"resume config mismatch: {field} checkpoint={saved_config.get(field)!r} "
                        f"current={self.config.get(field)!r}"
                    )
            checkpoint_attention_mode = data.get(
                "attention_mode",
                (data.get("actor_architecture") or {}).get("attention_mode", "learned"),
            )
            if checkpoint_attention_mode != self.pcta_attention_mode:
                raise RuntimeError(
                    f"resume PCTA attention mode mismatch: checkpoint={checkpoint_attention_mode!r} "
                    f"current={self.pcta_attention_mode!r}"
                )
        elif self.config["actor_variant"] == PCTA_V2_VARIANT:
            if data.get("pcta_v2_auxiliary_consistency") is not False:
                raise RuntimeError("PCTA-v2 checkpoint must disable auxiliary consistency")
            for field in (
                "pcta_v2_attention_heads", "pcta_v2_context_dim",
                "pcta_v2_enemy_dim", "pcta_v2_target_dim", "pcta_hidden_dim",
            ):
                if saved_config.get(field) != self.config.get(field):
                    raise RuntimeError(
                        f"resume config mismatch: {field} checkpoint={saved_config.get(field)!r} "
                        f"current={self.config.get(field)!r}"
                    )
        checkpoint_method = data.get("method_variant", saved_config.get("method_variant", "baseline"))
        if checkpoint_method != self.config["method_variant"]:
            raise RuntimeError(
                f"resume method mismatch: checkpoint={checkpoint_method!r} "
                f"current={self.config['method_variant']!r}"
            )
        if self.credit_enabled:
            expected_credit = self.credit_metadata
            for field in (
                "credit_method", "credit_component_names", "credit_component_weights",
                "credit_critic_architecture", "counterfactual_action_sampling",
                "credit_estimator_version",
            ):
                if data.get(field) != expected_credit[field]:
                    raise RuntimeError(
                        f"incompatible counterfactual credit contract: {field}"
                    )
            if "credit_critic" not in data or "credit_critic_optimizer_state" not in data:
                raise RuntimeError("counterfactual checkpoint is missing credit training state")
        if self.agp_enabled:
            checkpoint_lambda = data.get("agp_lambda", saved_config.get("agp_lambda"))
            if checkpoint_lambda is None or float(checkpoint_lambda) != float(self.config["agp_lambda"]):
                raise RuntimeError("resume AGP lambda mismatch")
        for field in RESUME_CONFIG_FIELDS:
            checkpoint_value = saved_config.get(
                field,
                DEFAULTS["actor_log_std_init"] if field == "actor_log_std_init" else None,
            )
            current_value = self.config.get(field)
            if checkpoint_value != current_value:
                raise RuntimeError(
                    f"resume config mismatch: {field} checkpoint={checkpoint_value!r} current={current_value!r}"
                )
        if self.is_tam:
            for field in TAM_RESUME_CONFIG_FIELDS:
                if saved_config.get(field, DEFAULTS[field]) != self.config.get(field):
                    raise RuntimeError(
                        f"resume config mismatch: {field} checkpoint={saved_config.get(field, DEFAULTS[field])!r} "
                        f"current={self.config.get(field)!r}"
                    )
        if self.is_recurrent and not self.is_tam:
            for field in ("recurrent_hidden_dim", "recurrent_sequence_length"):
                if saved_config.get(field) != self.config.get(field):
                    raise RuntimeError(
                        f"resume config mismatch: {field} checkpoint={saved_config.get(field)!r} "
                        f"current={self.config.get(field)!r}"
                    )
        if data.get("environment_config") != self.environment_config:
            raise RuntimeError("resume environment config mismatch: resolved content differs from checkpoint")
        self.actors.load_state_dict(data["actors"])
        self.critic.load_state_dict(data["critic"])
        if self.credit_enabled:
            assert self.credit_critic is not None
            self.credit_critic.load_state_dict(data["credit_critic"])
        if "actor_optimizer_states" not in data or "rollout_state" not in data:
            raise RuntimeError("checkpoint contains weights only and cannot resume training")
        for optimizer, state in zip(self.actor_optimizers, data["actor_optimizer_states"]):
            optimizer.load_state_dict(state)
        self.critic_optimizer.load_state_dict(data["critic_optimizer_state"])
        if self.credit_enabled:
            assert self.credit_critic_optimizer is not None
            self.credit_critic_optimizer.load_state_dict(data["credit_critic_optimizer_state"])
        self.rng.bit_generator.state = deepcopy(data["trainer_numpy_rng"])
        torch.set_rng_state(data["torch_rng"].cpu())
        _restore_cuda_rng_state(data.get("cuda_rng"))
        rollout = data["rollout_state"]
        self.observations = np.asarray(rollout["observations"], dtype=np.float32)
        self.global_states = np.asarray(rollout["global_states"], dtype=np.float32)
        self.active_masks = np.asarray(rollout["active_masks"], dtype=np.float32)
        if self.is_recurrent:
            if "actor_hidden_states" not in rollout or "actor_recurrent_masks" not in rollout:
                raise RuntimeError("recurrent checkpoint is missing hidden-state continuation data")
            self.actor_hidden_states = np.asarray(rollout["actor_hidden_states"], dtype=np.float32)
            self.actor_recurrent_masks = np.asarray(rollout["actor_recurrent_masks"], dtype=np.float32)
            expected_hidden = (
                int(self.config["num_envs"]), len(RED_IDS), self.actor_recurrent_hidden_dim,
            )
            if self.actor_hidden_states.shape != expected_hidden:
                raise RuntimeError("checkpoint recurrent hidden-state shape mismatch")
            if self.actor_recurrent_masks.shape != (int(self.config["num_envs"]), len(RED_IDS)):
                raise RuntimeError("checkpoint recurrent mask shape mismatch")
        if self.is_tam:
            for field in ("critic_hidden_states", "critic_recurrent_masks"):
                if field not in rollout:
                    raise RuntimeError("TAM checkpoint is missing critic hidden-state continuation data")
            self.critic_hidden_states = np.asarray(rollout["critic_hidden_states"], dtype=np.float32)
            self.critic_recurrent_masks = np.asarray(rollout["critic_recurrent_masks"], dtype=np.float32)
            if self.critic_hidden_states.shape != (
                int(self.config["num_envs"]), int(self.config["tam_critic_gru_hidden_dim"]),
            ):
                raise RuntimeError("checkpoint TAM critic hidden-state shape mismatch")
            if self.critic_recurrent_masks.shape != (int(self.config["num_envs"]),):
                raise RuntimeError("checkpoint TAM critic recurrent mask shape mismatch")
        self.vector_env.set_env_states(
            rollout["environment_states"],
            np.asarray(rollout["vector_reset_counts"], dtype=np.int64),
            rollout.get("vector_base_seed"),
        )
        self.env_steps = int(data["sampled_steps"])
        self.last_rollout_metrics = self._empty_rollout_metrics()
        return self.env_steps

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        if (data.get("environment_version"), data.get("observation_dim"), data.get("global_state_dim")) != (self.environment_config["environment_version"], OBS_DIM, GLOBAL_STATE_DIM):
            raise RuntimeError("incompatible HAPPO checkpoint environment contract")
        checkpoint_mode = data.get("reward_mode", data.get("reward_shaping_mode", "absolute"))
        checkpoint_gamma = float(data.get("shaping_gamma") or 0.0)
        if checkpoint_mode != self.reward_mode or (
            checkpoint_mode == "potential" and not np.isclose(checkpoint_gamma, self.shaping_gamma, rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError("incompatible HAPPO checkpoint reward shaping contract")
        self._validate_actor_architecture(data)
        self._validate_critic_architecture(data)
        if data.get("environment_profile") != self.config["environment_profile"]:
            raise RuntimeError(
                f"incompatible HAPPO checkpoint environment profile: {data.get('environment_profile')!r} "
                f"(expected {self.config['environment_profile']!r})"
            )
        self.actors.load_state_dict(data["actors"]); self.critic.load_state_dict(data["critic"])

    def close(self) -> None:
        self.vector_env.close()
