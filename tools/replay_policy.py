"""Checkpoint-compatible deterministic/stochastic policy loading for replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithm.happo.networks import IndependentActors
from algorithm.happo.dbm_rgaa import (
    DBMGaussianActor, DBM_RGAA_METHOD, RGAA_WIDE_METHOD,
    build_method_actors, dbm_metadata, wide_metadata,
)
from algorithm.happo.recurrent import RecurrentIndependentActors
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from algorithm.modules.pcta import PCTAIndependentActors
from env.mavuav import (
    ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS, load_environment_config,
)

ARCHITECTURE_KEYS = {"entity_dim", "role_dim", "fusion_hidden_dim", "action_dim"}
RECURRENT_ARCHITECTURE_FIELDS = (
    "observation_dim", "encoder_dim", "recurrent_hidden_dim", "head_dim", "action_dim",
)
RECURRENT_ARCHITECTURE_KEYS = set(RECURRENT_ARCHITECTURE_FIELDS)
PCTA_ARCHITECTURE_FIELDS = (
    "observation_dim", "context_input_dim", "context_dim", "enemy_block_dim",
    "enemy_dim", "enemy_slots", "head_hidden_dim", "action_dim",
)
PCTA_ARCHITECTURE_KEYS = set(PCTA_ARCHITECTURE_FIELDS)


def infer_method_display_name(actor_variant: str, method_variant: str = "baseline") -> str:
    if actor_variant == "recurrent" and method_variant == "baseline":
        return "R-HAPPO"
    if actor_variant == "hrta":
        return "HAPPO-HRTA"
    if actor_variant == "structured_uniform":
        return "HAPPO-Structured-Uniform"
    if actor_variant == "pcta":
        return "PCTA-HAPPO"
    if actor_variant == "vanilla":
        names = {
            "baseline": "HAPPO", "agp": "HAPPO-AGP",
            "rgaa": "RGAA-HAPPO", RGAA_WIDE_METHOD: "RGAA-Wide",
            DBM_RGAA_METHOD: "DBM-RGAA",
        }
        if method_variant in names:
            return names[method_variant]
    raise RuntimeError(
        f"unsupported actor architecture for replay: actor_variant={actor_variant!r}, "
        f"method_variant={method_variant!r}"
    )


@dataclass
class ReplayPolicyAdapter:
    actors: Any
    payload: dict[str, Any]
    device: torch.device
    actor_variant: str
    method_variant: str
    actor_architecture: dict[str, Any] | None
    hidden_states: list[torch.Tensor] | None = field(default=None, init=False, repr=False)
    recurrent_masks: torch.Tensor | None = field(default=None, init=False, repr=False)
    next_hidden_states: list[torch.Tensor] | None = field(default=None, init=False, repr=False)
    deterministic: bool = field(default=True, init=False, repr=False)
    last_diagnostics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def method_display_name(self) -> str:
        return infer_method_display_name(self.actor_variant, self.method_variant)

    def reset_episode(
        self, *, action_mode: str = "deterministic", action_seed: int | None = None,
    ) -> None:
        """Start an independent replay episode without checkpoint rollout memory."""
        if action_mode not in ("deterministic", "stochastic"):
            raise ValueError(f"unsupported action_mode: {action_mode!r}")
        self.deterministic = action_mode == "deterministic"
        if not self.deterministic and action_seed is not None:
            torch.manual_seed(int(action_seed))
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(action_seed))
        self.last_diagnostics = []
        if self.actor_variant != "recurrent":
            return
        self.hidden_states = [actor.initial_hidden(1, device=self.device) for actor in self.actors.actors]
        self.recurrent_masks = torch.zeros((len(RED_IDS), 1), device=self.device)
        self.next_hidden_states = None

    def actions(
        self, observations: dict[str, np.ndarray], *,
        active_masks: np.ndarray | list[float] | None = None,
        collect_diagnostics: bool = False,
    ) -> np.ndarray:
        result: list[np.ndarray] = []
        self.last_diagnostics = []
        active = np.ones(len(RED_IDS), dtype=np.float32) if active_masks is None else np.asarray(active_masks)
        if active.shape != (len(RED_IDS),):
            raise ValueError(f"active_masks must have shape ({len(RED_IDS)},)")
        if self.actor_variant == "recurrent":
            if self.hidden_states is None or self.recurrent_masks is None:
                raise RuntimeError("reset_episode() is required before recurrent replay actions")
            if self.next_hidden_states is not None:
                raise RuntimeError("after_step() is required between recurrent replay actions")
            self.next_hidden_states = []
        with torch.no_grad():
            for index, aid in enumerate(RED_IDS):
                observation = torch.as_tensor(observations[aid], device=self.device).unsqueeze(0)
                if self.actor_variant == "recurrent":
                    action, _, next_hidden = self.actors.actors[index].sample_step(
                        observation, self.hidden_states[index], self.recurrent_masks[index],
                        deterministic=self.deterministic,
                    )
                    self.next_hidden_states.append(next_hidden.detach())
                else:
                    actor = self.actors.actors[index]
                    if (
                        collect_diagnostics and active[index] > 0.5
                        and isinstance(actor, DBMGaussianActor)
                    ):
                        details = actor.mode_diagnostics(observation)
                        probabilities = details["router_probabilities"].squeeze(0)
                        experts = details["expert_outputs"].squeeze(0)
                        self.last_diagnostics.append({
                            "agent_id": aid,
                            "router_probabilities": probabilities.detach().cpu().numpy(),
                            "hard_mode_proxy": int(torch.argmax(probabilities).item() + 1),
                            "expert_divergence": float(torch.linalg.vector_norm(experts[0] - experts[1]).item()),
                            "residual_magnitude": float(torch.linalg.vector_norm(details["scaled_residual"].squeeze(0)).item()),
                        })
                    action, _ = actor.sample(observation, deterministic=self.deterministic)
                result.append(action.squeeze(0).cpu().numpy())
        return np.asarray(result, dtype=np.float32)

    def after_step(self, active_masks: np.ndarray | list[float], done: bool) -> None:
        """Commit recurrent state after one environment decision boundary."""
        if self.actor_variant != "recurrent":
            return
        if done:
            self.reset_episode()
            return
        if self.next_hidden_states is None or len(self.next_hidden_states) != len(RED_IDS):
            raise RuntimeError("actions() must produce recurrent hidden state before after_step()")
        active = np.asarray(active_masks, dtype=np.float32)
        if active.shape != (len(RED_IDS),):
            raise ValueError(f"Red active_masks must have shape ({len(RED_IDS)},), got {active.shape}")
        if not np.all(np.isin(active, (0.0, 1.0))):
            raise ValueError("Red active_masks must contain only 0 or 1")
        masks = torch.as_tensor(active, device=self.device).reshape(len(RED_IDS), 1)
        self.hidden_states = [state * masks[index] for index, state in enumerate(self.next_hidden_states)]
        self.recurrent_masks = masks
        self.next_hidden_states = None


def resolve_device(requested: str) -> torch.device:
    return torch.device("cpu" if requested.startswith("cuda") and not torch.cuda.is_available() else requested)


def load_replay_actors(checkpoint: str | Path, device: str | torch.device = "cpu") -> ReplayPolicyAdapter:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = resolve_device(str(device))
    payload = torch.load(path, map_location=resolved, weights_only=False)
    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    if not isinstance(trainer_config, dict):
        raise RuntimeError("incompatible HAPPO checkpoint: trainer_config/config must be a mapping")
    variant = str(payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla")))
    method = str(payload.get("method_variant", trainer_config.get("method_variant", "baseline")))
    expected_version = (
        "heterogeneous_mavuav_4v4_v3_9"
        if method in ("rgaa", RGAA_WIDE_METHOD, DBM_RGAA_METHOD)
        else ENVIRONMENT_VERSION
    )
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    expected = (expected_version, OBS_DIM, GLOBAL_STATE_DIM)
    if actual != expected:
        raise RuntimeError(f"incompatible HAPPO checkpoint environment contract: expected={expected!r}, actual={actual!r}")
    if "actors" not in payload:
        raise RuntimeError("incompatible HAPPO checkpoint: missing actors state_dict")
    architecture = payload.get("actor_architecture")

    if variant == "vanilla":
        if "hidden_dim" not in trainer_config:
            raise RuntimeError("incompatible vanilla checkpoint: trainer_config.hidden_dim is required")
        if method not in ("baseline", "agp", "rgaa", RGAA_WIDE_METHOD, DBM_RGAA_METHOD):
            raise RuntimeError(f"unsupported HAPPO method_variant: {method!r}")
        if method in (RGAA_WIDE_METHOD, DBM_RGAA_METHOD):
            if "environment_config" not in payload:
                raise RuntimeError(f"{method} replay requires resolved environment_config")
            if load_environment_config(payload["environment_config"])["environment_version"] != expected_version:
                raise RuntimeError(f"incompatible {method} resolved environment contract")
            expected_metadata = (
                dbm_metadata(trainer_config) if method == DBM_RGAA_METHOD
                else wide_metadata(trainer_config)
            )
            for key, value in expected_metadata.items():
                if payload.get(key) != value:
                    raise RuntimeError(f"incompatible {method} checkpoint contract: {key}")
            actors = build_method_actors(
                method_variant=method, training_seed=int(trainer_config["seed"]),
                hidden_dim=int(trainer_config["hidden_dim"]),
                log_std_init=float(trainer_config.get("actor_log_std_init", -0.5)),
                role_module_enabled=bool(trainer_config.get("role_module_enabled", True)),
                dbm_role_count=int(trainer_config.get("dbm_role_count", 2)),
                dbm_residual_scale=float(trainer_config.get("dbm_residual_scale", 0.25)),
                dbm_expert_init_scale=float(trainer_config.get("dbm_init_scale", 0.01)),
                uav_actor_hidden_dim=int(trainer_config.get("uav_actor_hidden_dim", 131)),
            )
        else:
            actors = IndependentActors(
                hidden_dim=int(trainer_config["hidden_dim"]),
                log_std_init=float(trainer_config.get("actor_log_std_init", -0.5)),
            )
            if method == "rgaa":
                if "environment_config" not in payload:
                    raise RuntimeError("RGAA replay requires resolved environment_config")
                if load_environment_config(payload["environment_config"])["environment_version"] != expected_version:
                    raise RuntimeError("incompatible RGAA resolved environment contract")
                for field in ("role_advantage_coef", "role_critic_architecture", "role_aux_reward_mode"):
                    if field not in payload:
                        raise RuntimeError(f"RGAA replay checkpoint is missing metadata: {field}")
    elif variant in ("hrta", "structured_uniform"):
        if not isinstance(architecture, dict) or set(architecture) != ARCHITECTURE_KEYS:
            raise RuntimeError(
                f"incompatible {variant} actor architecture metadata: "
                f"expected keys={sorted(ARCHITECTURE_KEYS)!r}, actual={architecture!r}"
            )
        kwargs = {key: int(architecture[key]) for key in ARCHITECTURE_KEYS}
        cls = HRTAIndependentActors if variant == "hrta" else StructuredUniformIndependentActors
        actors = cls(**kwargs)
    elif variant == "recurrent":
        if method != "baseline":
            raise RuntimeError("recurrent replay supports only method_variant='baseline'")
        if not isinstance(architecture, dict) or set(architecture) != RECURRENT_ARCHITECTURE_KEYS:
            raise RuntimeError(
                "incompatible recurrent actor architecture metadata: "
                f"expected keys={sorted(RECURRENT_ARCHITECTURE_KEYS)!r}, actual={architecture!r}"
            )
        if any(isinstance(architecture[key], bool) or not isinstance(architecture[key], (int, np.integer))
               for key in RECURRENT_ARCHITECTURE_KEYS):
            raise RuntimeError("incompatible recurrent actor architecture metadata: dimensions must be integers")
        architecture = {key: int(architecture[key]) for key in RECURRENT_ARCHITECTURE_FIELDS}
        if architecture["observation_dim"] != OBS_DIM or architecture["action_dim"] != 3:
            raise RuntimeError(
                "incompatible recurrent actor architecture dimensions: "
                f"observation_dim={architecture['observation_dim']}, action_dim={architecture['action_dim']}"
            )
        if architecture["encoder_dim"] != architecture["head_dim"]:
            raise RuntimeError("unsupported recurrent actor architecture: encoder_dim must equal head_dim")
        if architecture["encoder_dim"] <= 0 or architecture["recurrent_hidden_dim"] <= 0:
            raise RuntimeError("incompatible recurrent actor architecture: hidden dimensions must be positive")
        if "hidden_dim" in trainer_config and int(trainer_config["hidden_dim"]) != architecture["encoder_dim"]:
            raise RuntimeError("incompatible recurrent actor architecture: trainer_config.hidden_dim mismatch")
        if ("recurrent_hidden_dim" in trainer_config and
                int(trainer_config["recurrent_hidden_dim"]) != architecture["recurrent_hidden_dim"]):
            raise RuntimeError(
                "incompatible recurrent actor architecture: trainer_config.recurrent_hidden_dim mismatch"
            )
        actors = RecurrentIndependentActors(
            observation_dim=architecture["observation_dim"], action_dim=architecture["action_dim"],
            hidden_dim=architecture["encoder_dim"],
            recurrent_hidden_dim=architecture["recurrent_hidden_dim"],
        )
    elif variant == "pcta":
        if method != "baseline":
            raise RuntimeError("PCTA replay supports only method_variant='baseline'")
        if not isinstance(architecture, dict) or set(architecture) != PCTA_ARCHITECTURE_KEYS:
            raise RuntimeError(
                "incompatible PCTA actor architecture metadata: "
                f"expected keys={sorted(PCTA_ARCHITECTURE_KEYS)!r}, actual={architecture!r}"
            )
        architecture = {key: int(architecture[key]) for key in PCTA_ARCHITECTURE_FIELDS}
        if (
            architecture["observation_dim"] != OBS_DIM
            or architecture["context_input_dim"] != 44
            or architecture["enemy_block_dim"] != 14
            or architecture["enemy_slots"] != 4
            or architecture["action_dim"] != 3
        ):
            raise RuntimeError("incompatible PCTA actor architecture dimensions")
        actors = PCTAIndependentActors(
            observation_dim=architecture["observation_dim"],
            action_dim=architecture["action_dim"],
            context_dim=architecture["context_dim"],
            enemy_dim=architecture["enemy_dim"],
            hidden_dim=architecture["head_hidden_dim"],
        )
    else:
        raise RuntimeError(
            f"unsupported actor architecture for replay: actor_variant={variant!r}, "
            f"checkpoint metadata={{'actor_variant': {payload.get('actor_variant')!r}, "
            f"'actor_architecture': {architecture!r}}}"
        )
    actors = actors.to(resolved)
    actors.load_state_dict(payload["actors"])
    actors.eval()
    return ReplayPolicyAdapter(actors, payload, resolved, variant, method, architecture)
