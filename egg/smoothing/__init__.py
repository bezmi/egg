"""Grid smoothing: TMOP targets and boundary-layer clustering."""

from egg.smoothing.respace import enforce_boundary_layer_spacing
from egg.smoothing.targets import build_boundary_layer_target

__all__ = ["build_boundary_layer_target", "enforce_boundary_layer_spacing"]
