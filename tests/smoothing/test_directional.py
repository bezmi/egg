# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Directional soft-energy samples: builder output (parallel segments with
frozen wall normals, fan-vertex line/stem tuples) and the NumPy evaluator's
gradients against finite differences in 2D and 3D."""

import math
from itertools import product

import numpy as np
import pytest

from egg.geometry import Circle, LineSegment, Vector3
from egg.geometry.analytic3d import Plane
from egg.smoothing.directional import (
    KIND_LINE,
    KIND_PARALLEL,
    KIND_STEM,
    DirectionalSamples,
    _chain_consistent_normals,
    build_directional_samples,
    directional_energy_grad,
)
from egg.topology import TopologyBuilder


def _fill_multilinear(topo):
    """Fill global_nodes by multilinear blending of each block's corners —
    deterministic coordinates without the TFI/projection pipeline."""
    g = topo.grid
    d = topo.d
    for bi, spec in enumerate(topo.block_specs.values()):
        shape = spec.logical_shape
        corners = np.array([topo.corners[c].position for c in spec.corner_names])
        axes = np.meshgrid(*[np.linspace(0.0, 1.0, s) for s in shape], indexing="ij")
        Xb = np.zeros(shape + (d,))
        for ci, bits in enumerate(product((0, 1), repeat=d)):
            w = np.ones(shape)
            for ax, bit in enumerate(bits):
                w = w * (axes[ax] if bit else 1.0 - axes[ax])
            Xb += w[..., None] * corners[ci]
        g.global_nodes[g.block_dof_maps[bi]] = Xb
    return g


def _strip2d(res: int = 3, wall_blocks: tuple = ("L0", "L1")):
    wall = LineSegment((0.0, 0.0), (2.0, 0.0)).named("wall")
    b = TopologyBuilder(d=2)
    for j, row in enumerate(("w", "m", "t")):
        for i in range(3):
            b.add_corner(f"{row}{i}", (float(i), float(j)))
    for i, name in enumerate(("L0", "L1")):
        b.add_block(
            name,
            corners=(f"w{i}", f"m{i}", f"w{i + 1}", f"m{i + 1}"),
            resolutions=(res, 2),
        )
    for i, name in enumerate(("U0", "U1")):
        b.add_block(
            name,
            corners=(f"m{i}", f"t{i}", f"m{i + 1}", f"t{i + 1}"),
            resolutions=(res, 2),
        )
    for name in wall_blocks:
        b.associate(name, 1, 0, wall)
    return b, wall


def _fan2d(res: int = 3):
    rim = Circle((0.0, 0.0), 1.5).named("rim")
    C = Vector3(0.0, 0.0)
    R = [
        Vector3(math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5))
        for k in range(5)
    ]
    M = [
        Vector3(
            1.5 * math.cos(2 * math.pi * (k + 0.5) / 5),
            1.5 * math.sin(2 * math.pi * (k + 0.5) / 5),
        )
        for k in range(5)
    ]
    b = TopologyBuilder(d=2)
    for k in range(5):
        b.add_block(
            f"b{k}",
            corners=(C, R[(k + 1) % 5], R[k], M[k]),
            resolutions=(res, res),
        )
        b.associate(f"b{k}", 0, 1, rim)
        b.associate(f"b{k}", 1, 1, rim)
    return b, rim, C, R, M


# ---- frozen-normal de-outliering -------------------------------------------


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_chain_normals_reject_a_curled_endpoint():
    """A chain node whose wall projection lands on a degenerate/extrapolated
    endpoint hands back a normal ~90 deg off the trend (the wedge inflow
    spline's exit-plane tip); it must be replaced by its consistent neighbour
    so it cannot corrupt the last segment's reference."""
    good = _unit([-0.558, 0.83])
    n = np.tile(good, (6, 1))
    n[-1] = _unit([-0.83, -0.558])  # the curled endpoint (~90 deg off)
    out = _chain_consistent_normals(n)
    # every node now agrees with the trend (no ~90 deg outlier survives)
    assert np.all(np.abs(out @ good) > 0.9)
    # the good interior nodes are untouched
    assert np.allclose(out[:-1], good)


def test_chain_normals_replace_a_middle_outlier():
    good = _unit([0.0, 1.0])
    n = np.tile(good, (7, 1))
    n[3] = _unit([1.0, 0.0])  # a lone perpendicular spike mid-chain
    out = _chain_consistent_normals(n)
    assert np.allclose(np.abs(out), good)


def test_chain_normals_align_sign_flips():
    """A sign flip between neighbours would cancel in a segment average
    ``n[k] + n[k+1]`` — orientation is unified."""
    g = _unit([0.3, 0.95])
    n = np.array([g, g, -g, g, g])
    out = _chain_consistent_normals(n)
    # all oriented the same way, so no adjacent pair cancels
    for k in range(len(out) - 1):
        assert np.dot(out[k], out[k + 1]) > 0.0


def test_chain_normals_leave_consistent_and_short_arrays():
    g = _unit([0.1, 1.0])
    n = np.tile(g, (5, 1))
    assert np.allclose(_chain_consistent_normals(n), n)
    two = np.tile(g, (2, 1))  # < 3 nodes: nothing to compare against
    assert np.allclose(_chain_consistent_normals(two), two)


# ---- builder ---------------------------------------------------------------


def test_no_declarations_builds_nothing():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, wall = _strip2d()
    grid = _fill_multilinear(b.build())
    assert build_directional_samples(grid) is None


def test_parallel_segments_freeze_wall_normals_and_lengths():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, wall = _strip2d(res=3)
    b.parallel_to("wall", chain=("t0", "t1", "t2"))
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid, lambda_parallel=2.0, energy_scale=3.0)
    assert s.n == 6 and np.all(s.kind == KIND_PARALLEL)
    # the wall is y=0: every frozen normal is (0, ±1)
    assert np.allclose(np.abs(s.ref), [0.0, 1.0], atol=1e-12)
    # weight = lambda * scale * (segment length / mean segment length) —
    # dimensionless, so a uniform chain carries exactly lambda * scale
    assert np.allclose(s.weight, 2.0 * 3.0, rtol=1e-12)
    # segments chain consecutive dofs
    p = grid.topology.parallel_chains[0]
    assert [tuple(t) for t in s.nodes[:, :2]] == list(zip(p.dofs[:-1], p.dofs[1:]))


def test_projection_fallback_still_freezes_unit_normals():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, wall = _strip2d(res=3, wall_blocks=("L0",))
    b.parallel_to("wall", chain=("t0", "t1", "t2"))
    with pytest.warns(UserWarning):
        grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid)
    assert s.n == 6
    assert np.allclose(np.abs(s.ref), [0.0, 1.0], atol=1e-12)
    assert np.all(np.isfinite(s.weight))


def test_fan_frame_line_and_stem_samples():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, rim, C, R, M = _fan2d(res=3)
    b.fan_frame(C, through=(R[0], R[2]), normal=R[1])
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid, lambda_line=4.0, lambda_stem=5.0)
    assert s.n == 2
    f = grid.topology.fan_frames[0]
    (li,) = np.nonzero(s.kind == KIND_LINE)[0]
    (st,) = np.nonzero(s.kind == KIND_STEM)[0]
    assert tuple(s.nodes[li, :3]) == (
        f.through_rails[0][1],
        f.dof,
        f.through_rails[1][1],
    )
    assert tuple(s.nodes[st, :2]) == (f.normal_rail[1], f.dof)
    assert (s.weight[li], s.weight[st]) == (4.0, 5.0)
    # the stem axis is the frozen unit through direction a -> b
    X = grid.global_nodes
    axis = X[f.through_rails[1][1]] - X[f.through_rails[0][1]]
    axis = axis / np.linalg.norm(axis)
    assert np.allclose(s.ref[st], axis, atol=1e-9)


def test_taper_decays_away_from_the_singular_vertex():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, rim, C, R, M = _fan2d(res=3)
    b.parallel_to(rim, chain=(R[0], C, R[2]), taper=0.4)
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid)
    w = s.weight / np.linalg.norm(
        grid.global_nodes[s.nodes[:, 1]] - grid.global_nodes[s.nodes[:, 0]], axis=1
    )
    # C sits mid-chain (segment index 2|3 boundary); the per-length weight
    # peaks beside it and decays monotonically toward both chain ends
    assert w[2] > w[1] > w[0]
    assert w[3] > w[4] > w[5]


def test_fairness_line_samples_at_interior_chain_nodes():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, wall = _strip2d(res=3)
    b.parallel_to("wall", chain=("t0", "t1", "t2"))
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid, lambda_fair=0.5, energy_scale=4.0)
    chain = grid.topology.parallel_chains[0]
    fair = np.nonzero(s.kind == KIND_LINE)[0]
    # one anti-kink sample per interior chain node (no singular corners here)
    assert fair.size == len(chain.dofs) - 2
    d = list(chain.dofs)
    for j, si in enumerate(fair):
        assert tuple(s.nodes[si, :3]) == (d[j], d[j + 1], d[j + 2])
    assert np.allclose(s.weight[fair], 0.5 * 4.0, rtol=1e-12)
    # off by default
    s0 = build_directional_samples(grid)
    assert not np.any(s0.kind == KIND_LINE)


def test_fairness_skips_singular_fan_corners():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    b, rim, C, R, M = _fan2d(res=3)
    b.parallel_to(rim, chain=(R[0], C, R[2]))
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid, lambda_fair=1.0)
    chain = grid.topology.parallel_chains[0]
    fair = np.nonzero(s.kind == KIND_LINE)[0]
    # every interior node gets a sample except the five-fan centre
    centres = {tuple(s.nodes[si, :3])[1] for si in fair}
    mid = chain.dofs[len(chain.dofs) // 2]
    assert fair.size == len(chain.dofs) - 3
    assert mid not in centres


def test_resolves_in_3d():
    pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")
    wall = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)).named("wall")
    b = TopologyBuilder(d=3)
    for x in (0, 1):
        for y in (0, 1, 2):
            for z in (0, 1):
                b.add_corner(f"n{x}{y}{z}", (float(x), float(y), float(z)))
    for name, y0 in (("lo", 0), ("hi", 1)):
        b.add_block(
            name,
            corners=tuple(
                f"n{x}{y0 + y}{z}" for x in (0, 1) for y in (0, 1) for z in (0, 1)
            ),
            resolutions=(2, 2, 2),
        )
    b.associate("lo", 1, 0, wall)
    b.parallel_to("wall", chain=("n020", "n120"))
    grid = _fill_multilinear(b.build())
    s = build_directional_samples(grid)
    assert s.n == 2 and s.d == 3
    assert np.allclose(np.abs(s.ref), [0.0, 1.0, 0.0], atol=1e-12)
    # uniform segments: the dimensionless length ratio is 1, weight = lambda
    assert np.allclose(s.weight, 1.0, rtol=1e-12)


# ---- evaluator vs finite differences ---------------------------------------


def _hand_samples(d: int, rng) -> tuple[np.ndarray, DirectionalSamples]:
    X = rng.normal(size=(8, d))

    def u(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    nodes = np.full((4, 4), -1, dtype=np.int32)
    nodes[0, :2] = (0, 1)
    nodes[1, :3] = (2, 3, 4)
    nodes[2, :2] = (5, 6)
    nodes[3, :2] = (6, 7)
    kind = np.array([KIND_PARALLEL, KIND_LINE, KIND_STEM, KIND_PARALLEL], np.int32)
    ref = np.zeros((4, d))
    ref[0] = u(rng.normal(size=d))
    ref[2] = u(rng.normal(size=d))
    ref[3] = u(rng.normal(size=d))
    weight = rng.uniform(0.5, 2.0, size=4)
    s = DirectionalSamples(
        nodes=nodes, kind=kind, ref=ref, axis=np.zeros((4, d)), weight=weight
    )
    return X, s


@pytest.mark.parametrize("d", [2, 3])
def test_grad_matches_finite_differences(d):
    rng = np.random.default_rng(7 + d)
    X, s = _hand_samples(d, rng)
    e0, g = directional_energy_grad(X, s)
    assert np.isfinite(e0) and e0 > 0.0
    h = 1e-6
    for k in range(X.shape[0]):
        for c in range(d):
            Xp, Xm = X.copy(), X.copy()
            Xp[k, c] += h
            Xm[k, c] -= h
            ep, _ = directional_energy_grad(Xp, s)
            em, _ = directional_energy_grad(Xm, s)
            fd = (ep - em) / (2.0 * h)
            assert g[k, c] == pytest.approx(fd, rel=5e-6, abs=1e-9)


@pytest.mark.parametrize("d", [2, 3])
def test_degenerate_states_stay_finite(d):
    rng = np.random.default_rng(3)
    X, s = _hand_samples(d, rng)
    X[1] = X[0]  # zero-length parallel segment
    X[3] = X[2]  # zero-length line leg
    X[6] = X[5]  # zero-length stem
    e, g = directional_energy_grad(X, s)
    assert np.isfinite(e)
    assert np.all(np.isfinite(g))


def test_zero_energy_at_the_target_configuration():
    # a segment perpendicular to the frozen normal, opposed through legs,
    # and a stem perpendicular to the axis all sit at the energy minimum
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],  # parallel segment along x, normal (0, 1)
            [-1.0, 2.0],
            [0.0, 2.0],
            [1.0, 2.0],  # line: q1 + q2 = 0
            [4.0, 1.0],
            [4.0, 0.0],  # stem along -y, axis (1, 0)
            [9.0, 9.0],
        ]
    )
    nodes = np.full((3, 4), -1, dtype=np.int32)
    nodes[0, :2] = (0, 1)
    nodes[1, :3] = (2, 3, 4)
    nodes[2, :2] = (5, 6)
    s = DirectionalSamples(
        nodes=nodes,
        kind=np.array([KIND_PARALLEL, KIND_LINE, KIND_STEM], np.int32),
        ref=np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]),
        axis=np.zeros((3, 2)),
        weight=np.ones(3),
    )
    e, g = directional_energy_grad(X, s)
    assert e == pytest.approx(0.0, abs=1e-20)
    assert np.allclose(g, 0.0, atol=1e-12)
