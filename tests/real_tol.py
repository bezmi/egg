# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Precision-aware comparison tolerance for the Python tests.

The mirror of ``tests/cpp/real_tol.hpp``. The cross-precision parity tests
compare the C++/device results against a double-precision NumPy reference. When
the extension is built with ``egg::real = float`` (``-DEGG_REAL_IS_FP32=ON``)
those results only agree with the double reference to ~1e-4..1e-6, so the
double-tuned tolerances (1e-9..1e-12) would spuriously fail. :func:`real_tol`
floors a tolerance at the fp32 parity level in the float build and is the
identity in the double build (so the double build keeps its exact thresholds).

The build precision is reported by the extension itself via
``cpp_core.REAL_IS_FLOAT``, so no environment flag or heuristic is needed.
"""

from egg._cpp import cpp_core

# True when the loaded extension was compiled with egg::real = float.
REAL_IS_FLOAT: bool = bool(getattr(cpp_core, "REAL_IS_FLOAT", False))

# Relative parity floor vs the double reference (matches the C++ real_tol floor).
_FP32_FLOOR = 5e-4


def real_tol(tol_f64: float, floor: float = _FP32_FLOOR) -> float:
    """Floor a double-tuned tolerance at the fp32 parity level in the float
    build; return it unchanged in the double build.

    ``floor`` defaults to the 5e-4 round-trip parity level. Ill-conditioned
    nonlinear cases (e.g. a briefly-tangled mesh whose near-degenerate cells make
    the line search take a different branch in float) diverge more and can pass a
    larger, explicitly-justified floor.
    """
    return max(tol_f64, floor) if REAL_IS_FLOAT else tol_f64
