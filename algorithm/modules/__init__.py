"""Optional research modules that are not wired into vanilla training."""

from .hrta import HRTAActor, HRTAIndependentActors

__all__ = ["HRTAActor", "HRTAIndependentActors"]
