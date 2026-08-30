"""Optional research modules that are not wired into vanilla training."""

from .hrta import HRTAActor, HRTAIndependentActors
from .structured_uniform import (
    StructuredUniformActor,
    StructuredUniformIndependentActors,
    masked_uniform_pool,
)

__all__ = [
    "HRTAActor", "HRTAIndependentActors", "StructuredUniformActor",
    "StructuredUniformIndependentActors", "masked_uniform_pool",
]
