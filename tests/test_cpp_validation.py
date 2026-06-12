"""Binding-boundary validation: malformed contexts/X must raise a Python
exception, not crash the interpreter with device-side UB.

These exercise the host-side ``validate_context`` / ``checked_num_nodes`` guards
added to ``src/bindings.cpp`` (#1). Skips if the C++ core is not built.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from egg.smoothing.cpp_backend import flatten_context


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def _valid_inputs():
    """A valid (ctx_arrays, X_flat) pair from the 4x4 demo grid."""
    from tests.test_cpp_parity import _mini_context

    ctx, X0, _topo = _mini_context(perturb=True)
    ctx_arrays = flatten_context(ctx)
    X_flat = np.ascontiguousarray(X0, dtype=np.float64).ravel()
    return ctx_arrays, X_flat


def test_valid_inputs_do_not_raise():
    """Sanity: the unmodified inputs run a sweep cleanly."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    cpp_core.cpp_sweep(ctx_arrays, X_flat, 1, device="cpu")


def test_2d_shaped_X_raises():
    """An (N, 2) array (instead of flat (2N,)) must raise, not silently halve N."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    X_2d = X_flat.reshape(-1, 2)  # the classic silent-UB shape
    with pytest.raises(Exception):
        cpp_core.cpp_sweep(ctx_arrays, X_2d, 1, device="cpu")


def test_odd_length_X_raises():
    """An odd-length flat X (not 2 coords per node) must raise."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    X_odd = np.append(X_flat, 0.0)  # make length odd
    with pytest.raises(Exception):
        cpp_core.cpp_sweep(ctx_arrays, X_odd, 1, device="cpu")


def test_out_of_range_index_raises():
    """An out-of-range node index in a group must raise before any device work."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    bad = copy.deepcopy(ctx_arrays)
    num_nodes = X_flat.shape[0] // 2
    # Poison the first group's first corner index well past the node count.
    bad["groups"][0]["gc"] = bad["groups"][0]["gc"].copy()
    bad["groups"][0]["gc"][0] = np.int32(num_nodes + 999)
    with pytest.raises(Exception):
        cpp_core.cpp_sweep(bad, X_flat, 1, device="cpu")


def test_out_of_range_role_raises():
    """A role outside [-1, kNbrsPerCorner] must raise (it indexes J columns via
    c0 = kDim*role, so an out-of-range role would read past the sample's J block)."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    bad = copy.deepcopy(ctx_arrays)
    bad["groups"][0]["role"] = bad["groups"][0]["role"].copy()
    bad["groups"][0]["role"][0] = np.int32(3)  # > kNbrsPerCorner (2 in 2D)
    with pytest.raises(Exception):
        cpp_core.cpp_sweep(bad, X_flat, 1, device="cpu")


def test_out_of_range_energy_stencil_index_raises():
    """An out-of-range index in the energy stencil must also raise."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    bad = copy.deepcopy(ctx_arrays)
    num_nodes = X_flat.shape[0] // 2
    bad["energy_stencil"]["gn0"] = bad["energy_stencil"]["gn0"].copy()
    bad["energy_stencil"]["gn0"][0] = np.int32(num_nodes + 999)
    with pytest.raises(Exception):
        cpp_core.cpp_sweep(bad, X_flat, 1, device="cpu")


def test_dim3_raises_not_implemented():
    """The runtime-dispatch scaffold: dim=3 raises a clean RuntimeError
    ("3D ... not yet implemented") — proving the d-dispatch is wired without
    needing the Phase 2 3D math."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    with pytest.raises(RuntimeError, match="not yet implemented"):
        cpp_core.cpp_sweep(ctx_arrays, X_flat, 1, device="cpu", dim=3)


def test_dim4_raises_value_error():
    """dim=4 is outside the supported {2, 3} set and raises ValueError."""
    from egg._cpp import cpp_core

    ctx_arrays, X_flat = _valid_inputs()
    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        cpp_core.cpp_sweep(ctx_arrays, X_flat, 1, device="cpu", dim=4)
