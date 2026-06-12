"""Grid file export front-end.

Currently supported formats:

- SU2 native ASCII (:func:`egg.io.su2.export_su2`)
"""

from egg.io.su2 import export_su2

__all__ = ["export_su2"]
