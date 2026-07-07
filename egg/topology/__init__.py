# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Multiblock topology: fluent builder, data model, and editable overlay."""

from .builder import TopologyBuilder
from .explicit import ExplicitTopology, editable

__all__ = ["TopologyBuilder", "ExplicitTopology", "editable"]
