# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Grid smoothing: TMOP targets and boundary-layer clustering."""

from egg.smoothing.respace import (
    enforce_boundary_layer_spacing,
    first_layer_heights,
    respace_first_layers,
)
from egg.smoothing.targets import build_boundary_layer_target

__all__ = [
    "build_boundary_layer_target",
    "enforce_boundary_layer_spacing",
    "first_layer_heights",
    "respace_first_layers",
]
