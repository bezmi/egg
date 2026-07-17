# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Pipeline integration of the control-point smoother.

``tmop_smoother="control_point"`` sequences nodal untangle -> nodal
pre-smooth -> net fit + reduced Gauss-Newton on the device session -> final
eval, stores the net on the grid, and rejects the node-mode interface terms
and clustering targets with clear errors.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "2D", "circles")
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


def _grid():
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle()
    return topo.initialize_grid()


def test_control_point_pipeline_runs_and_stores_net():
    from egg.pipeline import generate_steps

    grid = _grid()
    phases = []
    infos = {}
    for phase, info in generate_steps(grid, tmop_smoother="control_point"):
        phases.append(phase)
        infos[phase] = info
    assert "control" in phases
    assert phases[-1] == "final"
    assert infos["control"]["min_det"] > 0.0
    assert infos["control"]["iters"] > 0
    assert grid.control_net is not None
    assert len(grid.control_net.seams) > 0

    # Energy competitive with the node pipeline run to convergence.
    grid_j = _grid()
    last = {}
    for _phase, info in generate_steps(grid_j, tmop_sweeps=2000, tmop_chunk=500):
        last = info
    assert infos["final"]["energy"] <= 1.05 * last["energy"] + 1e-12

    # Constrained boundary nodes end exactly on their entities.
    Xg = np.asarray(grid.global_nodes)
    for g, ent in grid.dof_constraints.items():
        p = Xg[int(g)]
        assert np.linalg.norm(ent.project(p) - p) < 1e-9


def test_control_point_pipeline_3d():
    """The control-point pipeline phase is dimension-general: the 6-block
    cubed-sphere shell (edge fans, sliding sphere/plane walls) runs through
    generate_steps and improves on the nodal pre-smooth."""
    import importlib.util

    from egg.pipeline import generate_steps

    ex = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "3D",
        "cubed_sphere",
        "cubed_sphere.py",
    )
    spec = importlib.util.spec_from_file_location("cubed_sphere_ex", ex)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = mod.cubed_sphere(4, 8, r0=0.5).build().initialize_grid()

    infos = {}
    for phase, info in generate_steps(grid, tmop_smoother="control_point"):
        infos[phase] = info
    assert infos["control"]["min_det"] > 0.0
    assert infos["control"]["iters"] > 0
    # The control phase must not lose ground on the pre-smoothed energy.
    assert infos["control"]["energy"] <= infos["tmop"]["energy"] + 1e-9
    assert grid.control_net is not None


def test_control_net_save_load_roundtrip(tmp_path):
    """Persistence: the saved net state reloads onto a freshly built topology
    for the same grid and evaluates to the identical fine grid."""
    from egg.io import load_control_net, save_control_net
    from egg.pipeline import generate_steps

    grid = _grid()
    for _ in generate_steps(grid, tmop_smoother="control_point"):
        pass
    topo = grid.control_net
    X_solved = np.array(grid.global_nodes)

    path = tmp_path / "net.npz"
    save_control_net(topo, path)

    grid2 = _grid()  # fresh grid, same topology
    topo2 = load_control_net(grid2, path)
    topo2.write_to_grid()
    np.testing.assert_allclose(np.asarray(grid2.global_nodes), X_solved, atol=1e-12)

    # A mismatched grid is rejected, not silently misapplied.
    from topologies import build_twin_circle

    grid3 = build_twin_circle()[0].initialize_grid()
    with pytest.raises(ValueError):
        load_control_net(grid3, path)


def test_control_point_rejects_node_mode_interface_terms():
    from egg.pipeline import PipelineConfig

    with pytest.raises(ValueError, match="control legs"):
        PipelineConfig(
            tmop_smoother="control_point",
            interface_ortho={"weight": 0.3},
        ).validate()


def test_control_point_clusters_during_solve():
    """Boundary-layer clustering in control mode runs on the same clustered
    target as node mode (chord-parameter fits keep the clustered grid
    representable by the net), so the solve itself shapes the layer profile —
    first-layer heights land near the spec without any post-hoc resampling
    (exact heights stay with the opt-in respace/pin phases)."""
    from egg.pipeline import generate_steps
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle(bl={"first_height": 0.05, "growth": 1.3})
    grid = topo.initialize_grid()
    infos = {}
    for phase, info in generate_steps(grid, tmop_smoother="control_point"):
        infos[phase] = info
    assert "control" in infos
    assert infos["final"]["min_det"] > 0.0
    assert grid.control_net is not None

    X = np.asarray(grid.global_nodes)
    heights = []
    for bi in range(4):  # the O-ring blocks (circle wall on axis 1, side 1)
        dm = grid.block_dof_maps[bi]
        heights += list(np.linalg.norm(X[dm[:, -1]] - X[dm[:, -2]], axis=1))
    assert 0.03 < min(heights) and max(heights) < 0.08  # near the 0.05 spec
