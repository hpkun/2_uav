"""Checkpoint-compatible deterministic policy loading for combat replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithm.happo.networks import IndependentActors
from algorithm.modules.hrta import HRTAIndependentActors
from algorithm.modules.structured_uniform import StructuredUniformIndependentActors
from env.mavuav import ENVIRONMENT_VERSION, GLOBAL_STATE_DIM, OBS_DIM, RED_IDS

ARCHITECTURE_KEYS = {"entity_dim", "role_dim", "fusion_hidden_dim", "action_dim"}


def infer_method_display_name(actor_variant: str, method_variant: str = "baseline") -> str:
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

    @property
    def method_display_name(self) -> str:
        return infer_method_display_name(self.actor_variant, self.method_variant)

    def actions(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        result: list[np.ndarray] = []
        with torch.no_grad():
            for index, aid in enumerate(RED_IDS):
                action, _ = self.actors.actors[index].sample(
                    torch.as_tensor(observations[aid], device=self.device).unsqueeze(0),
                    deterministic=True,
                )
                result.append(action.squeeze(0).cpu().numpy())
        return np.asarray(result, dtype=np.float32)


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

