"""Checkpoint-compatible deterministic policy loading for combat replay."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithm.happo.networks import IndependentActors
from algorithm.happo.recurrent import RecurrentIndependentActors
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS

ARCHITECTURE_KEYS = {"entity_dim", "role_dim", "fusion_hidden_dim", "action_dim"}
RECURRENT_ARCHITECTURE_FIELDS = (
    "observation_dim", "encoder_dim", "recurrent_hidden_dim", "head_dim", "action_dim",
)
RECURRENT_ARCHITECTURE_KEYS = set(RECURRENT_ARCHITECTURE_FIELDS)


def infer_method_display_name(actor_variant: str, method_variant: str = "baseline") -> str:
    if actor_variant == "recurrent" and method_variant == "baseline":
        return "R-HAPPO"
    if actor_variant == "hrta":
        return "HAPPO-HRTA"
    if actor_variant == "structured_uniform":
        return "HAPPO-Structured-Uniform"
    if actor_variant == "vanilla":
        names = {
            "baseline": "HAPPO", "agp": "HAPPO-AGP",
            "curriculum": "HAPPO-Curriculum",
            "agp_curriculum": "HAPPO-AGP-Curriculum",
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

    @property
    def method_display_name(self) -> str:
        return infer_method_display_name(self.actor_variant, self.method_variant)

    def reset_episode(self) -> None:
        """Start an independent replay episode without checkpoint rollout memory."""
        if self.actor_variant != "recurrent":
            return
        self.hidden_states = [actor.initial_hidden(1, device=self.device) for actor in self.actors.actors]
        self.recurrent_masks = torch.zeros((len(RED_IDS), 1), device=self.device)
        self.next_hidden_states = None

    def actions(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        result: list[np.ndarray] = []
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
                        deterministic=True,
                    )
                    self.next_hidden_states.append(next_hidden.detach())
                else:
                    action, _ = self.actors.actors[index].sample(observation, deterministic=True)
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
    actual = (payload.get("environment_version"), payload.get("observation_dim"), payload.get("global_state_dim"))
    expected = (ENVIRONMENT_VERSION, OBS_DIM, GLOBAL_STATE_DIM)
    if actual != expected:
        raise RuntimeError(f"incompatible HAPPO checkpoint environment contract: expected={expected!r}, actual={actual!r}")
    if "actors" not in payload:
        raise RuntimeError("incompatible HAPPO checkpoint: missing actors state_dict")

    trainer_config = payload.get("trainer_config", payload.get("config", {}))
    if not isinstance(trainer_config, dict):
        raise RuntimeError("incompatible HAPPO checkpoint: trainer_config/config must be a mapping")
    variant = str(payload.get("actor_variant", trainer_config.get("actor_variant", "vanilla")))
    method = str(payload.get("method_variant", trainer_config.get("method_variant", "baseline")))
    architecture = payload.get("actor_architecture")

    if variant == "vanilla":
        if "hidden_dim" not in trainer_config:
            raise RuntimeError("incompatible vanilla checkpoint: trainer_config.hidden_dim is required")
        if method not in ("baseline", "agp", "curriculum", "agp_curriculum"):
            raise RuntimeError(f"unsupported HAPPO method_variant: {method!r}")
        actors = IndependentActors(hidden_dim=int(trainer_config["hidden_dim"]))
        architecture = None
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
