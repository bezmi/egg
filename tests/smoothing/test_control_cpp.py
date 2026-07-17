# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Device control-net reduced GN: parity with the NumPy reference.

The device path (src/control_net.hpp + src/control_solve.hpp, driven through
``CppStructuredSweepSession(control=...)``) must reproduce the NumPy
reference (:mod:`egg.smoothing.control_ref`) on the same single-block problem:
same evaluated grid, matching energy trace, matching accepted state — with the
dense solve replaced by Jacobi-preconditioned matrix-free PCG, so parity is
within ``real_tol``-floored tolerances rather than bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from egg.smoothing.control_ref import (
    boolean_sum_b,
    boundary_ctrl_mask,
    build_control_ref,
    fit_control_net,
    run_control_ref,
)
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from tests.smoothing.test_control_ref import _make_wavy_grid


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


def _make_control_session(grid, ctrl_shape, device="cpu"):
    """Build the structured session with a control wire fitted to the grid."""
    from egg.geometry.control_net import tensor_map
    from egg.smoothing.control_backend import build_control_wire
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    block = grid.blocks[0]
    node_shape = tuple(block.logical_shape)
    cmap = tensor_map(node_shape, ctrl_shape, degree=3)
    X0 = np.asarray(block.nodes, dtype=float)
    C0 = fit_control_net(cmap, X0)
    b_field = boolean_sum_b(cmap, C0, X0)
    free_ctrl = ~boundary_ctrl_mask(tuple(ctrl_shape))
    wire = build_control_wire(cmap, C0, free_ctrl, b_field, block=0)

    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(
        ctx, bsc, np.asarray(grid.global_nodes), device=device, control=wire
    )
    return sess, cmap, C0, b_field


def test_set_c_eval_matches_prolong():
    # The device eval (M C + b into the structured store) must agree with the
    # NumPy prolong on the same net. The session leaves X untouched until the
    # control state is evaluated (nodal runs stay valid before the control
    # phase), so trigger the eval through set_C.
    grid = _make_wavy_grid(n=9)
    sess, cmap, C0, b_field = _make_control_session(grid, (5, 5))
    from tests.real_tol import real_tol

    sess.set_C(C0)
    X_dev = sess.get_X()
    X_ref = cmap.prolong(C0) + b_field
    dof_map = grid.block_dof_maps[0]
    np.testing.assert_allclose(
        X_dev[dof_map.reshape(-1)],
        X_ref.reshape(-1, 2),
        atol=real_tol(1e-12),
    )
    # get_C round-trips the uploaded lattice.
    np.testing.assert_allclose(sess.get_C().reshape(C0.shape), C0, atol=real_tol(1e-14))


def test_run_control_rejects_untangle():
    grid = _make_wavy_grid(n=9)
    sess, *_ = _make_control_session(grid, (5, 5))
    with pytest.raises(Exception, match="untangle"):
        sess.run_control(1, phase="untangle")


def test_run_control_requires_wire():
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    grid = _make_wavy_grid(n=9)
    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(
        ctx, bsc, np.asarray(grid.global_nodes), device="cpu"
    )
    with pytest.raises(Exception, match="control wire"):
        sess.run_control(1)


def test_parity_with_numpy_reference():
    from tests.real_tol import real_tol

    ctrl_shape = (6, 6)

    # NumPy reference on its own copy of the grid.
    grid_ref = _make_wavy_grid()
    cref = build_control_ref(grid_ref, IdentityTarget(d=2), ctrl_shape)
    rep_ref = run_control_ref(cref, max_outer=30)

    # Device run on an identical grid, same fitted net (tight PCG so the step
    # matches the reference's dense solve).
    grid_dev = _make_wavy_grid()
    sess, cmap, _, _ = _make_control_session(grid_dev, ctrl_shape)
    rep_dev = sess.run_control(30, pcg_max_iter=500, pcg_rtol=1e-12, pcg_forcing=False)

    e_ref = np.asarray(rep_ref["energies"])
    e_dev = np.asarray(rep_dev["energies"])
    md_dev = np.asarray(rep_dev["mindets"])

    assert np.all(np.isfinite(e_dev))
    assert np.all(md_dev > 0.0)
    assert np.all(np.diff(e_dev) <= 1e-12)  # monotone accept rule

    # Energy-trace parity: same iteration count regime and matching energies.
    # (never assert exact energies — reduction order; real_tol floors fp32)
    assert abs(rep_dev["iters"] - rep_ref["iters"]) <= 2
    n = min(e_ref.size, e_dev.size)
    np.testing.assert_allclose(e_dev[:n], e_ref[:n], rtol=real_tol(1e-7))
    # Final energies agree tightly.
    np.testing.assert_allclose(e_dev[-1], e_ref[-1], rtol=real_tol(1e-7))

    # Accepted grids agree node-by-node.
    X_dev = sess.get_X()
    X_ref = np.asarray(grid_ref.global_nodes)
    np.testing.assert_allclose(X_dev, X_ref, atol=real_tol(1e-6))

    # The net downloaded from the device regrids validly at 2x (algebraic
    # re-evaluation — the reference's regrid criterion re-checked through the
    # wire).
    from egg.geometry.control_net import tensor_map

    C_dev = sess.get_C().reshape(cmap.ctrl_shape + (2,))
    fine_shape = tuple((s - 1) * 2 + 1 for s in cmap.node_shape)
    fine = tensor_map(fine_shape, cmap.ctrl_shape, degree=3)
    Xf = fine.prolong(C_dev)
    du = np.diff(Xf, axis=0)[:, :-1]
    dv = np.diff(Xf, axis=1)[:-1, :]
    det = du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]
    assert float(det.min()) > 0.0


def test_control_then_nodal_interop():
    # After a control run the session's nodal smoother still works on the
    # accepted state (mixed-mode sanity: shared X buffer, monotone energy).
    grid = _make_wavy_grid(n=9)
    sess, *_ = _make_control_session(grid, (5, 5))
    rep = sess.run_control(5)
    e_ctrl = rep["energies"][-1]
    energies, mindets = sess.run(10, phase="barrier", omega=0.8)
    assert np.isfinite(energies[-1])
    assert mindets[-1] > 0.0
    assert energies[-1] <= e_ctrl + 1e-9
