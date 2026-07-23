# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The 2D regrid example: solve once, save the net, regrid at a new resolution.

A cold run saves the net; a run at a different resolution loads it
(re-tabulated) and polishes, not a cold solve. The workspace packs into an
.eggy.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "2D", "regrid_demo")
)


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def _final_min_det(grid):
    from egg.smoothing.untangle import grid_min_det
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
    return grid_min_det(grid.global_nodes, es)


def test_cold_then_warm_regrid(tmp_path):
    import case

    cache = str(tmp_path / "net.npz")
    # Cold solve at the base resolution writes the cache.
    grid1, _f1, warm1 = case.run_case(n=1, cache_path=cache, verbose=False)
    assert warm1 is False
    assert os.path.exists(cache)
    assert _final_min_det(grid1) > 0.0

    # A finer grid regrids from the cache (re-tabulated), not a cold solve.
    grid2, _f2, warm2 = case.run_case(n=2, cache_path=cache, verbose=False)
    assert warm2 is True
    assert _final_min_det(grid2) > 0.0
    # The finer grid really is finer.
    assert grid2.global_node_count > grid1.global_node_count


def test_workspace_packs_into_an_eggy(tmp_path):
    import case

    from egg.io import eggy

    cache = str(tmp_path / "net.npz")
    case.run_case(n=1, cache_path=cache, verbose=False)
    out = str(tmp_path / "demo.eggy")
    eggy.pack(out, os.path.dirname(os.path.abspath(case.__file__)))
    assert eggy.is_eggy(out)
