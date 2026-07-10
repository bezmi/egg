# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""End-to-end: the composed block-boundary term runs through the C++ structured
sweep and measurably orthogonalises an oblique seam without inverting cells."""

from __future__ import annotations

import numpy as np
import pytest

from egg.smoothing.interface_ortho import _side_frames
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def _has_cpp():
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(), reason="egg._cpp.cpp_core not built (requires cmake build)"
)


def _pgram_grid(res=(8, 8), shear=1.0):
    """Two sheared blocks sharing an oblique x≈2 seam (a parallelogram domain)."""
    b = TopologyBuilder(d=2)
    for n, p in [
        ("A", (0, 0)),
        ("D", (shear, 2)),
        ("B", (2, 0)),
        ("C", (2 + shear, 2)),
        ("E", (4, 0)),
        ("F", (4 + shear, 2)),
    ]:
        b.add_corner(n, p, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _seam_triples(grid):
    topo = grid.topology
    bn = list(topo.block_specs.keys())
    out = []
    for conn in topo.interface_connections:
        fa = _side_frames(grid, bn, conn.face_a)
        fb = _side_frames(grid, bn, conn.face_b)
        for P, frA in fa.items():
            frB = fb.get(P)
            if frB is not None:
                out.append((P, frA["Q"], frB["Q"], frA["Rp"], frA["Rm"]))
    return out


def _mean_obliquity(X, triples):
    """Mean |cos(angle(crossing edge, seam tangent))|; 0 = orthogonal crossing."""
    vals = []
    for _P, QA, QB, Rp, Rm in triples:
        c = X[QA] - X[QB]
        t = X[Rp] - X[Rm]
        vals.append(abs(float(np.dot(c, t)) / (np.linalg.norm(c) * np.linalg.norm(t))))
    return float(np.mean(vals))


def _run(interface_ortho):
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    grid = _pgram_grid()
    kw = {} if interface_ortho is None else {"interface_ortho": interface_ortho}
    ctx = build_sweep_context(grid, IdentityTarget(2), **kw)
    bsc = build_block_structured_context(grid)
    triples = _seam_triples(grid)
    sess = CppStructuredSweepSession(ctx, bsc, grid.global_nodes, device="cpu")
    _e, m = sess.run(300, phase="barrier", omega=0.8, report_every=0)
    X = np.asarray(sess.get_X()).reshape(-1, 2)
    return _mean_obliquity(X, triples), float(m[-1])


def test_normal_term_orthogonalises_seam_monotonically():
    base, m_base = _run(None)
    assert m_base > 0.0
    prev = base
    for w in (1.0, 5.0, 15.0):
        ob, mdet = _run({"mode": "normal", "weight": w})
        assert mdet > 0.0, f"cells inverted at weight {w}"
        assert ob < prev + 1e-3, f"obliquity not decreasing at weight {w}"
        prev = ob
    # A strong weight makes the seam markedly more orthogonal than base smoothing.
    assert prev < 0.5 * base


def test_continuous_term_runs_and_differs_from_normal():
    _base, _ = _run(None)
    ob_c, m_c = _run({"mode": "continuous", "weight": 5.0})
    ob_n, m_n = _run({"mode": "normal", "weight": 5.0})
    assert m_c > 0.0 and m_n > 0.0
    # Continuous keeps the natural (oblique) crossing straight rather than forcing
    # it normal, so it does NOT drive obliquity toward zero the way normal does.
    assert ob_c > ob_n
