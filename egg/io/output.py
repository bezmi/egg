# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Grid file export front-end.

Currently supported formats:

- SU2 native ASCII (:func:`egg.io.su2.export_su2`)
- gdtk/Eilmer lmr structured grids (:func:`egg.io.lmr.export_lmr`)
"""

from egg.io.lmr import export_lmr
from egg.io.su2 import export_su2

__all__ = ["export_lmr", "export_su2"]
