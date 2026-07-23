# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Interface orthogonality/continuity term: zero on a clean seam, restoring."""

import numpy as np
import pytest

from egg.smoothing import batch
from egg.smoothing.interface_ortho import interface_ortho_samples
from egg.topology.builder import TopologyBuilder


def _lr_grid(res=(4, 4)):
    """Two unit blocks sharing the x=2 edge (axis-0 seam)."""
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (2.0, 0.0)),
        ("C", (2.0, 2.0)),
        ("E", (4.0, 0.0)),
        ("F", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _energy(X, s):
    e, _ = batch.energy_and_mindet(
        X, s.gc, s.gn0, s.gn1, s.s0, s.s1, s.W_inv, weight=s.weight
    )
    return e


def _grad_at(X, s, dof):
    """Analytic interface-term gradient (2,) w.r.t. one DOF via batch scatter."""
    hit = s.part_node == dof  # (P, 3)
    rows, which = np.nonzero(hit)
    if rows.size == 0:
        return np.zeros(2)
    role = s.part_role[rows, which]
    g, _ = batch.dof_grad_hess(
        X,
        s.gc[rows],
        s.gn0[rows],
        s.gn1[rows],
        s.s0[rows],
        s.s1[rows],
        s.W_inv[rows],
        role,
        weight=s.weight[rows],
    )
    return g


@pytest.mark.parametrize("mode", ["normal", "continuous"])
def test_zero_on_clean_seam(mode):
    grid = _lr_grid()
    s = interface_ortho_samples(grid, mode=mode, weight=1.0)
    assert len(s) > 0
    assert _energy(grid.global_nodes, s) < 1e-9


def _pgram(res, shear=1.0):
    """Two sheared blocks sharing an oblique seam (so the orthogonality target
    genuinely differs from the current crossing direction)."""
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


def _relax_fade(grid):
    """Total weight removed by turning the clustering relax on (as a fraction)."""
    a0 = interface_ortho_samples(grid, mode="normal", weight=1.0, cluster_relax=0.0)
    a1 = interface_ortho_samples(grid, mode="normal", weight=1.0, cluster_relax=1.0)
    return 1.0 - float(a1.weight.sum() / a0.weight.sum())


def test_cluster_relax_fades_only_on_slivers():
    # The relax fades the term's weight where cells are thin (clustered-like) and
    # leaves it alone on ~square cells. Cubed aniso keeps a merely sheared cell
    # near full weight; a sliver seam loses most of it.
    square = _relax_fade(_pgram(res=(8, 8)))  # aniso ~ 0.1 -> tiny fade
    thin = _relax_fade(_pgram(res=(8, 80)))  # sliver cells -> most weight faded
    assert square < 0.05
    assert thin > 0.5


def _clustered_strip(res=(6, 12)):
    from egg.geometry import LineSegment

    wall = (
        LineSegment((0.0, 0.0), (4.0, 0.0))
        .named("wall")
        .clustered(first_height=0.05, growth=1.2, n_layers=4, n_fixed=3)
    )
    b = TopologyBuilder(d=2)
    for n, p in [("A", (0, 0)), ("B", (4, 0)), ("C", (0, 2)), ("D", (4, 2))]:
        b.add_corner(n, p, fixed=True)
    b.add_block("main", ("A", "C", "B", "D"), res)
    b.associate("main", 1, 0, wall)
    return b.build().initialize_grid()


@pytest.mark.parametrize("mode", ["normal", "continuous"])
def test_pinned_boundary_emits_crossing_samples(mode):
    """A single-block clustered strip has no block seams, so without a pin the
    term is empty; a Pin's frozen band adds crossing samples at the pinned/free
    boundary that push only the free nodes above it (the boundary is read-only,
    so a rigid shift of just the free grid raises the energy)."""
    from egg.smoothing.respace import respace_first_layers

    grid = _clustered_strip()
    pinned = respace_first_layers(grid, grid.topology)
    base = interface_ortho_samples(grid, mode=mode, weight=1.0, cluster_relax=0.0)
    withp = interface_ortho_samples(
        grid, mode=mode, weight=1.0, cluster_relax=0.0, pinned=pinned
    )
    assert len(base) == 0  # no block interface in a single block
    assert len(withp) > 0  # frozen band boundary got crossing samples
    # the samples constrain the free grid above the band: shifting the free
    # nodes (leaving the pinned band put) skews the crossing edges and raises
    # the energy, so the term really acts across the pinned/free boundary.
    pin_set = set(int(x) for x in np.asarray(pinned).ravel())
    referenced = np.unique(np.asarray(withp.part_node))
    free_above = np.array([n for n in referenced if int(n) not in pin_set])
    Xp = grid.global_nodes.copy()
    Xp[free_above, 1] += 0.1
    assert _energy(Xp, withp) > 1e-6


@pytest.mark.parametrize("mode", ["normal", "continuous"])
def test_perturbation_raises_energy(mode):
    grid = _lr_grid()
    s = interface_ortho_samples(grid, mode=mode, weight=1.0)  # frame frozen here
    X = grid.global_nodes.copy()
    # Tilt only the cross-seam (inner) neighbours in y — seam nodes stay put, so
    # this is a genuine skew of the crossing edge, not a rigid translation.
    movers = np.unique(s.part_node[:, 1])
    Xp = X.copy()
    Xp[movers, 1] += 0.2
    assert _energy(Xp, s) > 1e-4


@pytest.mark.parametrize("mode", ["normal", "continuous"])
def test_gradient_matches_finite_difference(mode):
    grid = _lr_grid()
    s = interface_ortho_samples(grid, mode=mode, weight=1.0)
    rng = np.random.default_rng(0)
    X = grid.global_nodes.copy()
    X += 0.05 * rng.normal(size=X.shape)  # away from the exact zero (flat grad)

    # A moving (role >= 0) participant: block-boundary nodes carry role -1 (the
    # term reads but never pushes them), so their analytic grad is 0 by design.
    moving = s.part_node[s.part_role >= 0]
    dof = int(np.unique(moving)[0])
    g = _grad_at(X, s, dof)
    eps = 1e-6
    fd = np.zeros(2)
    for k in range(2):
        Xp = X.copy()
        Xm = X.copy()
        Xp[dof, k] += eps
        Xm[dof, k] -= eps
        fd[k] = (_energy(Xp, s) - _energy(Xm, s)) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-4)
