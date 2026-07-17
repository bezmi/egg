# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Multi-block control-DOF topology: exact C0/C1 across seams, soft C2.

The control-point analogue of the node DOF model. Each block carries its own
clamped tensor-product net (:mod:`egg.geometry.control_net`); this module
glues them into one global reduced control vector:

- **exact C0**: boundary-facet control lattice points shared between blocks
  collapse to one global control (union-find over control symbols, seam
  correspondence derived from shared fine-node DOFs — no orientation logic of
  its own).
- **exact C1**: across every conforming seam the first-interior control rows
  are eliminated through one auxiliary tangent row per face-lattice index:
  with ``S`` the shared boundary control, ``Q_A``/``Q_B`` the first-interior
  controls and ``mu_X = p / (Delta_X L_X)`` (last knot interval times the
  block's cell count across the seam, so C1 means spacing-continuous in
  global cell index), the constraint ``mu_A (S - Q_A) = -mu_B (S - Q_B)`` is
  *parameterized away*: ``T := mu_A (S - Q_A)``, ``Q_A = S - T/mu_A``,
  ``Q_B = S + T/mu_B``. Crossings compose axis-by-axis by tensor product
  (corner ``S``, tangents, one twist), and the tangent lattices of collinear
  seams unify through the transverse seam — all of it falls out of one signed
  union-find over per-axis symbol tuples.
- **fans**: at a singular corner (valence != 2**d interior, > 2**(d-1) on the
  outer boundary) the tensor composition is unavailable; the fan-adjacent
  face-lattice index keeps plain shared controls (exact C0) and a quadratic
  C1 penalty row instead. 3D edge-fan detection lands with the 3D fan work.
- **soft C2**: sampled second-derivative mismatch across each seam at
  Greville points, quadratic penalty rows in the reduced system.
- **seam orthogonality**: one condition per tangent row ``T_j`` (penalty
  ``w |t_hat . T_j|^2 / |T_j|^2`` or hard ``T_j = h_j n_hat``), which
  orthogonalizes both sides at once since exact C1 makes the first-interior
  legs collinear through ``S``.

Everything is emitted as one affine map ``C_stacked = R q`` (``q`` the global
reduced control vector, fixed entries frozen); the NumPy reference runner
below drives the same reduced Gauss-Newton loop as the single-block reference
over ``q``. The Boolean-sum offset ``b`` is extended from *outer* faces only
(zero on seam faces): b owns CAD exactness, S/T owns cross-seam continuity.

The reference metric layer is 2D (the accepted exception); the topology
machinery itself is dimension-general and the device path carries D = 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace

import numpy as np

from egg.geometry.control_net import (
    bspline_basis,
    clamped_knots,
    fit_knots,
    greville,
    insertion_matrix,
    refine_fit_knots,
    tensor_map,
)
from egg.smoothing.control_ref import (
    ENERGY_TOL,
    boundary_ctrl_mask,
    dense_M,
    fit_control_net,
    reduced_system,
)
from egg.smoothing.solver import build_sweep_context

__all__ = [
    "Seam",
    "ControlTopology",
    "SeamOrthoSpec",
    "build_control_topology",
    "default_ctrl_shapes",
    "detect_control_walls",
    "resample_block",
    "run_control_topo_ref",
    "seam_angle_deviation",
    "seam_c1_jump",
    "seam_curvature_mismatch",
    "watertight_mismatch",
]


def default_ctrl_shapes(grid, r: int = 4) -> list[tuple[int, ...]]:
    """Per-block control counts ``max(4, ceil(n_cells/r) + 3)`` per axis.

    Conforming topology gives equal cell counts across shared faces, so the
    formula yields matching control counts there automatically.
    """
    out = []
    for blk in grid.blocks:
        out.append(tuple(max(4, -(-int(n - 1) // r) + 3) for n in blk.logical_shape))
    return out


# --------------------------------------------------------------------------- #
# Seam discovery (from shared fine-node DOFs — no orientation logic of its own)
# --------------------------------------------------------------------------- #


def _face_slab(axis: int, side: int, shape: tuple[int, ...]) -> tuple:
    sl: list = [slice(None)] * len(shape)
    sl[axis] = 0 if side == 0 else shape[axis] - 1
    return tuple(sl)


@dataclass
class Seam:
    """One conforming block interface, oriented from side ``a``.

    ``perm``/``flips`` map side-a face-lattice indices to side b:
    a-tangential slot ``m`` lands on b-tangential slot ``perm[m]``, reversed
    when ``flips[m]``. ``elim`` flags the face-lattice indices where the
    exact-C1 S/T elimination is active (fans switch it off).
    """

    a: int
    axis_a: int
    side_a: int
    b: int
    axis_b: int
    side_b: int
    perm: tuple[int, ...]
    flips: tuple[bool, ...]
    mu_a: float
    mu_b: float
    tang_shape_a: tuple[int, ...]
    tang_shape_b: tuple[int, ...]
    elim: np.ndarray = field(default=None, repr=False)
    fan: np.ndarray = field(default=None, repr=False)

    def map_tang(self, j: tuple[int, ...]) -> tuple[int, ...]:
        """Side-a face-lattice index -> side-b index."""
        jp = [0] * len(j)
        for m, jm in enumerate(j):
            p = self.perm[m]
            jp[p] = self.tang_shape_b[p] - 1 - jm if self.flips[m] else jm
        return tuple(jp)

    def map_tang_inv(self, jp: tuple[int, ...]) -> tuple[int, ...]:
        """Side-b face-lattice index -> side-a index."""
        j = [0] * len(jp)
        for m in range(len(jp)):
            p = self.perm[m]
            j[m] = self.tang_shape_b[p] - 1 - jp[p] if self.flips[m] else jp[p]
        return tuple(j)


def _end_mu(
    n_ctrl: int, degree: int, side: int, cells: int, knots: np.ndarray | None = None
) -> float:
    """Derivative factor of the clamped basis at a face, per global cell index.

    ``dX/du`` at a clamped end is ``(degree/Delta_end) (P_face - P_next)``;
    dividing by the block's cell count across the seam expresses it per global
    cell index, which is the CFD-relevant continuity scale. ``knots`` must be
    the axis's actual knot vector when it is not clamped-uniform.
    """
    U = clamped_knots(n_ctrl, degree) if knots is None else np.asarray(knots, float)
    if side == 0:
        delta = float(U[degree + 1])
    else:
        delta = 1.0 - float(U[n_ctrl - 1])
    return (degree / delta) / float(cells)


def _match_faces(grid) -> list[dict]:
    """All conforming seams, matched by whole-face fine-node DOF sets."""
    d = grid.topology.d
    faces = []
    for bi, blk in enumerate(grid.blocks):
        dm = grid.block_dof_maps[bi]
        for axis in range(d):
            for side in (0, 1):
                gids = np.asarray(dm[_face_slab(axis, side, blk.logical_shape)])
                faces.append((bi, axis, side, gids))

    matched: set[int] = set()
    seams = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            bi, ai, si, gi = faces[i]
            bj, aj, sj, gj = faces[j]
            if gi.size != gj.size:
                continue
            if not np.array_equal(np.sort(gi.ravel()), np.sort(gj.ravel())):
                continue
            if i in matched or j in matched:
                raise ValueError(
                    "a block face participates in more than one interface; "
                    "only conforming one-to-one seams are supported"
                )
            matched.add(i)
            matched.add(j)
            seams.append(
                {
                    "a": bi,
                    "axis_a": ai,
                    "side_a": si,
                    "b": bj,
                    "axis_b": aj,
                    "side_b": sj,
                    **_face_orientation(gi, gj),
                }
            )
    return seams


def _face_orientation(gi: np.ndarray, gj: np.ndarray) -> dict:
    """Recover the affine index map (perm + flips) sending ``gi`` onto ``gj``."""
    td = gi.ndim
    pos = {int(g): idx for idx, g in np.ndenumerate(gj)}
    origin = np.array(pos[int(gi[(0,) * td])], dtype=int)
    perm = [0] * td
    flips = [False] * td
    for m in range(td):
        e = tuple(1 if t == m else 0 for t in range(td))
        delta = np.array(pos[int(gi[e])], dtype=int) - origin
        p = int(np.argmax(np.abs(delta)))
        if not (np.abs(delta).sum() == 1 and abs(delta[p]) == 1):
            raise ValueError("interface node map is not axis-aligned")
        perm[m] = p
        flips[m] = bool(delta[p] < 0)
    # Validate the affine map on every face node (catches non-conforming
    # layouts that happen to share the same DOF set).
    for idx, g in np.ndenumerate(gi):
        jp = [0] * td
        for m in range(td):
            jp[perm[m]] = gj.shape[perm[m]] - 1 - idx[m] if flips[m] else idx[m]
        if int(gj[tuple(jp)]) != int(g):
            raise ValueError("interface node map is not a per-axis permutation")
    return {"perm": tuple(perm), "flips": tuple(flips)}


# --------------------------------------------------------------------------- #
# Signed union-find over control symbols
# --------------------------------------------------------------------------- #


class _SignedUF:
    """Union-find with a relative sign to the parent (T lattices across a seam
    unify with opposite orientation)."""

    def __init__(self):
        self.parent: dict = {}
        self.sign: dict = {}

    def add(self, x) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.sign[x] = 1

    def find(self, x):
        path = []
        while self.parent[x] != x:
            path.append(x)
            x = self.parent[x]
        s = 1
        for y in reversed(path):
            s *= self.sign[y]
            self.parent[y] = x
            self.sign[y] = s
        return x, self.sign[path[0]] if path else 1

    def union(self, x, y, sgn: int) -> None:
        rx, sx = self.find(x)
        ry, sy = self.find(y)
        if rx == ry:
            if sx != sgn * sy:
                raise ValueError(
                    "inconsistent seam orientation loop (a control DOF is "
                    "forced to zero); check the block topology"
                )
            return
        self.parent[ry] = rx
        self.sign[ry] = sx * sgn * sy  # sy in {-1, +1}, so 1/sy == sy


# --------------------------------------------------------------------------- #
# Per-block encodings: control -> linear combination of symbols
# --------------------------------------------------------------------------- #
# A symbol is (block, comps) with comps a per-axis tuple of ('P', index) or
# ('T', side): all-P symbols are plain lattice points; a T component stands
# for the seam tangent row of that block face. Tensor products of eliminated
# per-axis encodings generate the crossing twists automatically.


def _block_encodings(
    bi: int,
    ctrl_shape: tuple[int, ...],
    seams_by_face: dict,
) -> dict[int, list[tuple[tuple, float]]]:
    d = len(ctrl_shape)
    enc: dict[int, list[tuple[tuple, float]]] = {}
    for m in product(*(range(n) for n in ctrl_shape)):
        terms = [((), 1.0)]
        for k in range(d):
            i = m[k]
            n = ctrl_shape[k]
            axis_terms = [(("P", i), 1.0)]
            for side, face_i, q_i in ((0, 0, 1), (1, n - 1, n - 2)):
                if i != q_i or (k, side) not in seams_by_face:
                    continue
                seam, is_a = seams_by_face[(k, side)]
                tang = tuple(m[t] for t in range(d) if t != k)
                tang_a = tang if is_a else seam.map_tang_inv(tang)
                if not seam.elim[tang_a]:
                    continue  # fan fallback: plain control, C1 by penalty
                mu = seam.mu_a if is_a else seam.mu_b
                axis_terms = [
                    (("P", face_i), 1.0),
                    (("T", side), -1.0 / mu),
                ]
            new_terms = []
            for comps, coeff in terms:
                for comp, c in axis_terms:
                    new_terms.append((comps + (comp,), coeff * c))
            terms = new_terms
        enc[int(np.ravel_multi_index(m, ctrl_shape))] = [
            ((bi, comps), coeff) for comps, coeff in terms
        ]
    return enc


def _map_symbol(sym: tuple, seam: Seam, ctrl_shapes: list) -> tuple[tuple, int] | None:
    """Image of a side-a symbol across a seam (None if it does not lie on it).

    P components on the seam axis must sit on the seam face; T components map
    with sign -1 (the outward tangent rows of the two sides are opposite),
    transverse T components keep their sign (out-of-block is intrinsic).
    """
    bi, comps = sym
    if bi != seam.a:
        return None
    d = len(comps)
    shape_a = ctrl_shapes[seam.a]
    shape_b = ctrl_shapes[seam.b]
    face_i_a = 0 if seam.side_a == 0 else shape_a[seam.axis_a] - 1
    face_i_b = 0 if seam.side_b == 0 else shape_b[seam.axis_b] - 1
    ca = comps[seam.axis_a]
    if ca == ("P", face_i_a):
        cb, sgn = ("P", face_i_b), 1
    elif ca == ("T", seam.side_a):
        cb, sgn = ("T", seam.side_b), -1
    else:
        return None

    taxes_a = [k for k in range(d) if k != seam.axis_a]
    taxes_b = [k for k in range(d) if k != seam.axis_b]
    out = [None] * d
    out[seam.axis_b] = cb
    for m, k in enumerate(taxes_a):
        kp = taxes_b[seam.perm[m]]
        comp = comps[k]
        if comp[0] == "P":
            i = comp[1]
            out[kp] = ("P", shape_b[kp] - 1 - i if seam.flips[m] else i)
        else:
            side = comp[1]
            out[kp] = ("T", 1 - side if seam.flips[m] else side)
    return (seam.b, tuple(out)), sgn


# --------------------------------------------------------------------------- #
# Topology container + builder
# --------------------------------------------------------------------------- #


@dataclass
class ControlTopology:
    """The global reduced control parameterization ``C_stacked = R q``."""

    grid: object
    d: int
    ctrl_shapes: list[tuple[int, ...]]
    cmaps: list  # per-block ControlNetMap
    seams: list[Seam]
    R: np.ndarray  # (N_ctrl_total, n_roots) scalar reduction
    root_free: np.ndarray  # (n_roots,) bool
    root_syms: list  # canonical symbol per root (diagnostics / T lookup)
    q: np.ndarray  # (n_roots, d) current reduced state (fixed rows frozen)
    block_offsets: np.ndarray  # (nb + 1,) stacked-control offsets
    b_fields: list  # per-block Boolean-sum offsets (node_shape + (d,))
    t_root: dict = field(default_factory=dict)  # (seam_idx, tang) -> root
    s_root: dict = field(default_factory=dict)  # (seam_idx, tang) -> root
    fit_residual: float = 0.0  # max control move of the initial C1 projection
    wall_faces: dict = field(default_factory=dict)  # (block, axis, side) -> entity
    root_entity: dict = field(default_factory=dict)  # sliding root col -> entity
    axis_params: list = field(default_factory=list)  # per block, per axis node params

    @property
    def n_roots(self) -> int:
        return self.R.shape[1]

    def stacked_C(self) -> np.ndarray:
        """Current per-block stacked control coordinates (N_ctrl_total, d)."""
        return self.R @ self.q

    def block_C(self, bi: int) -> np.ndarray:
        lo, hi = self.block_offsets[bi], self.block_offsets[bi + 1]
        return self.stacked_C()[lo:hi].reshape(self.ctrl_shapes[bi] + (self.d,))

    def prolong_global(self) -> np.ndarray:
        """Evaluate every block's net (+ b) into global fine-node order.

        Blocks write in order; shared seam nodes take the last writer's value
        (one deterministic owner per node, matching the fine map assembly).
        """
        grid = self.grid
        Xg = np.array(grid.global_nodes, dtype=float)
        for bi, cmap in enumerate(self.cmaps):
            X = cmap.prolong(self.block_C(bi)) + self.b_fields[bi]
            Xg[grid.block_dof_maps[bi].reshape(-1)] = X.reshape(-1, self.d)
        return Xg

    def write_to_grid(self) -> None:
        grid = self.grid
        grid.global_nodes[:] = self.prolong_global()
        for bi, block in enumerate(grid.blocks):
            block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def _touches_outer(sym: tuple, ctrl_shapes: list, seam_faces: set) -> bool:
    """True if any P component sits on a non-seam (outer) block face."""
    bi, comps = sym
    for k, comp in enumerate(comps):
        if comp[0] != "P":
            continue
        n = ctrl_shapes[bi][k]
        for side, idx in ((0, 0), (1, n - 1)):
            if comp[1] == idx and (bi, k, side) not in seam_faces:
                return True
    return False


def _outer_faces(sym: tuple, ctrl_shapes: list, seam_faces: set) -> list[tuple]:
    """All (block, axis, side) outer faces the symbol's P components sit on."""
    bi, comps = sym
    out = []
    for k, comp in enumerate(comps):
        if comp[0] != "P":
            continue
        n = ctrl_shapes[bi][k]
        for side, idx in ((0, 0), (1, n - 1)):
            if comp[1] == idx and (bi, k, side) not in seam_faces:
                out.append((bi, k, side))
    return out


def _has_t(sym: tuple) -> bool:
    return any(c[0] == "T" for c in sym[1])


def detect_control_walls(grid, seam_faces: set) -> dict[tuple[int, int, int], object]:
    """Outer faces riding one CAD entity: ``(block, axis, side) -> entity``.

    A face is a wall when every non-corner fine node on it is constrained to
    the same single entity (mirrors the node DOF model: one entity slides,
    two or more fixes). Faces with unconstrained interior nodes, or mixed
    entities, are not walls — their controls stay fixed.
    """
    d = grid.topology.d
    walls: dict[tuple[int, int, int], object] = {}
    for bi, blk in enumerate(grid.blocks):
        dm = grid.block_dof_maps[bi]
        shape = blk.logical_shape
        for axis in range(d):
            for side in (0, 1):
                if (bi, axis, side) in seam_faces:
                    continue
                face = dm[_face_slab(axis, side, shape)]
                interior = (
                    face[tuple(slice(1, -1) for _ in range(face.ndim))]
                    if face.ndim
                    else face
                )
                gids = np.asarray(interior).ravel()
                if gids.size == 0:
                    continue
                ents = {id(grid.dof_constraints.get(int(g))) for g in gids}
                if None in {grid.dof_constraints.get(int(g)) for g in gids}:
                    continue
                if len(ents) != 1:
                    continue
                walls[(bi, axis, side)] = grid.dof_constraints[int(gids[0])]
    return walls


def _wall_crease_corners(grid, wall_faces: dict, kink_deg: float = 30.0) -> set:
    """Fine-node gids of wall block corners where the boundary kinks.

    Measured on the initial boundary polyline: at each block corner of a
    wall face, collect the unit directions toward the adjacent wall-face
    nodes; the wall passes smoothly through only if some pair is close to
    anti-parallel. A kink beyond ``kink_deg`` marks a geometric crease
    (capsule shoulders/knees on one composite profile) whose control must
    stay pinned — a sliding C2 net rounds it off otherwise.
    """
    X = np.asarray(grid.global_nodes, dtype=float)
    dirs: dict[int, list[np.ndarray]] = {}
    for (bi, axis, side), _ent in wall_faces.items():
        dm = grid.block_dof_maps[bi]
        shape = grid.blocks[bi].logical_shape
        face = np.asarray(dm[_face_slab(axis, side, shape)])
        fshape = face.shape
        for corner in product(*((0, n - 1) for n in fshape)):
            gid = int(face[corner])
            for m, n in enumerate(fshape):
                if n < 2:
                    continue
                nb = list(corner)
                nb[m] = 1 if corner[m] == 0 else n - 2
                v = X[int(face[tuple(nb)])] - X[gid]
                nv = float(np.linalg.norm(v))
                if nv > 0.0:
                    dirs.setdefault(gid, []).append(v / nv)
    creases = set()
    smooth = float(np.cos(np.radians(kink_deg)))
    for gid, vs in dirs.items():
        if len(vs) < 2:
            continue
        best = max(-float(a @ b) for i, a in enumerate(vs) for b in vs[i + 1 :])
        if best < smooth:
            creases.add(gid)
    return creases


def _fan_nodes(grid, seam_faces: set) -> tuple[set, set]:
    """Fine-node ids of singular fans: ``(corner_gids, edge_gids)``.

    A corner is a fan when its block valence is singular: != 2**d for an
    interior corner, > 2**(d-1) for one on the outer boundary. In 3D a
    codim-2 EDGE is a fan when the blocks around it are singular: != 4 for an
    interior edge, > 2 for one on the outer boundary (e.g. the valence-3
    radial edges at a cubed-sphere shell's cube corners); edge fans are keyed
    by the edge's MIDDLE node (interior of the line, so corner valence does
    not leak in; 1-cell edges have no interior node and fall back to the
    corner rule).
    """
    d = grid.topology.d
    valence: dict[int, int] = {}
    on_outer: set[int] = set()
    for bi, blk in enumerate(grid.blocks):
        dm = grid.block_dof_maps[bi]
        shape = blk.logical_shape
        for corner in product(*((0, n - 1) for n in shape)):
            gid = int(dm[corner])
            valence[gid] = valence.get(gid, 0) + 1
            for k in range(d):
                side = 0 if corner[k] == 0 else 1
                if (bi, k, side) not in seam_faces:
                    on_outer.add(gid)
    fan_gids = {
        gid
        for gid, v in valence.items()
        if (gid in on_outer and v > 2 ** (d - 1)) or (gid not in on_outer and v != 2**d)
    }

    # 3D edge fans: valence of the blocks sharing each codim-2 fine-node
    # edge, keyed by the edge's middle node (interior of the line, so corner
    # valence does not leak in; 1-cell edges have no interior node and fall
    # back to the corner rule).
    fan_edge_gids: set[int] = set()
    if d == 3:
        edge_blocks: dict[int, set[int]] = {}
        edge_outer: set[int] = set()
        for bi, blk in enumerate(grid.blocks):
            dm = grid.block_dof_maps[bi]
            shape = blk.logical_shape
            for free_ax in range(3):
                f1, f2 = (a for a in range(3) if a != free_ax)
                for s1 in (0, 1):
                    for s2 in (0, 1):
                        idx: list = [slice(None)] * 3
                        idx[f1] = 0 if s1 == 0 else shape[f1] - 1
                        idx[f2] = 0 if s2 == 0 else shape[f2] - 1
                        line = dm[tuple(idx)]
                        if line.shape[0] < 3:
                            continue
                        gid = int(line[line.shape[0] // 2])
                        edge_blocks.setdefault(gid, set()).add(bi)
                        if (bi, f1, s1) not in seam_faces or (
                            bi,
                            f2,
                            s2,
                        ) not in seam_faces:
                            edge_outer.add(gid)
        fan_edge_gids = {
            g
            for g, bs in edge_blocks.items()
            if (g in edge_outer and len(bs) > 2)
            or (g not in edge_outer and len(bs) != 4)
        }

    return fan_gids, fan_edge_gids


def _fan_mask(grid, seams: list[Seam], ctrl_shapes: list) -> None:
    """Switch the S/T elimination off at fan-adjacent face-lattice indices.

    The whole face-lattice window within distance 1 of the fan corner is
    switched off, not just the corner index: the first-interior corner
    control (1, 1, ...) is eliminated through the tangential index-1 entries
    of every adjacent seam, and its cross-tangent terms sit at transverse
    index 0 — around an odd fan those would unify with sign product -1
    (tangent continuity around an odd valence is impossible), so every
    elimination whose window touches the corner must drop to the penalty.
    The same argument gives the 3D edge bands (distance 1 of a fan edge).
    """
    seam_faces = {(s.a, s.axis_a, s.side_a) for s in seams} | {
        (s.b, s.axis_b, s.side_b) for s in seams
    }
    fan_gids, fan_edge_gids = _fan_nodes(grid, seam_faces)
    if not fan_gids and not fan_edge_gids:
        return
    for s in seams:
        blk = grid.blocks[s.a]
        dm = grid.block_dof_maps[s.a]
        shape = blk.logical_shape
        taxes = [k for k in range(len(shape)) if k != s.axis_a]
        for lat_corner in product(*((0, n - 1) for n in s.tang_shape_a)):
            corner = [0] * len(shape)
            corner[s.axis_a] = 0 if s.side_a == 0 else shape[s.axis_a] - 1
            for m, jm in enumerate(lat_corner):
                k = taxes[m]
                corner[k] = 0 if jm == 0 else shape[k] - 1
            if int(dm[tuple(corner)]) not in fan_gids:
                continue
            window = product(
                *(
                    range(max(0, c - 1), min(n, c + 2))
                    for c, n in zip(lat_corner, s.tang_shape_a)
                )
            )
            for tang in window:
                s.elim[tang] = False
                s.fan[tang] = True
        # Edge bands: a fan edge of this seam face runs along one tangential
        # axis at an end of the other; drop the elimination on the band of
        # lattice indices within distance 1 of it.
        if fan_edge_gids:
            for m, k in enumerate(taxes):
                for e_lat, e_fine in ((0, 0), (s.tang_shape_a[m] - 1, shape[k] - 1)):
                    idx = [slice(None)] * 3
                    idx[s.axis_a] = 0 if s.side_a == 0 else shape[s.axis_a] - 1
                    idx[k] = e_fine
                    line = dm[tuple(idx)]
                    if line.shape[0] < 3:
                        continue
                    if int(line[line.shape[0] // 2]) not in fan_edge_gids:
                        continue
                    for j in range(
                        max(0, e_lat - 1), min(s.tang_shape_a[m], e_lat + 2)
                    ):
                        sl: list = [slice(None), slice(None)]
                        sl[m] = j
                        s.elim[tuple(sl)] = False
                        s.fan[tuple(sl)] = True


def lstsq_q(R, C: np.ndarray) -> np.ndarray:
    """Least-squares reduced state ``argmin_q ||R q - C||`` (sparse-aware).

    ``R`` has full column rank and a tiny per-row support, so the normal
    equations stay sparse and well-scaled; use :func:`lstsq_q_solver` to
    amortize the factorization over repeated right-hand sides.
    """
    return lstsq_q_solver(R)(C)


def lstsq_q_solver(R):
    """Factorized least-squares solver ``C -> q`` for a fixed reduction map."""
    from scipy import sparse as _sp

    if not _sp.issparse(R):
        R = _sp.csr_matrix(R)
    Rt = R.T.tocsr()
    solve = _sp.linalg.factorized((Rt @ R).tocsc())

    def apply(C: np.ndarray) -> np.ndarray:
        rhs = Rt @ np.asarray(C, dtype=float)
        if rhs.ndim == 1:
            return solve(rhs)
        return np.column_stack([solve(rhs[:, k]) for k in range(rhs.shape[1])])

    return apply


def _chord_axis_params(grid) -> list[list[np.ndarray]]:
    """Per block, per axis: mean normalized cumulative-chord node parameters.

    Fitting with the grid's own parameter distribution instead of uniform
    keeps a clustered initial grid representable by the net (clustering is
    sampling-side by design): with uniform parameters a boundary-layer grid
    misfits systematically, b carries the whole clustering, and the
    sliding-wall b re-extension then perturbs hair-thin cells at their own
    scale.
    """
    out = []
    for blk in grid.blocks:
        X = np.asarray(blk.nodes, dtype=float)
        d = X.ndim - 1
        per_axis = []
        for k in range(d):
            seg = np.linalg.norm(np.diff(X, axis=k), axis=-1)
            cum = np.cumsum(seg, axis=k)
            total = np.take(cum, -1, axis=k)
            cumn = cum / np.expand_dims(np.maximum(total, 1e-300), k)
            mean = cumn.mean(axis=tuple(a for a in range(d) if a != k))
            per_axis.append(np.concatenate([[0.0], mean]))
        out.append(per_axis)
    return out


def _axis_chain_groups(raw_seams: list[dict], nb: int, d: int) -> list[list]:
    """Chains of seam-corresponding ``(block, axis)`` pairs with flip parity.

    A seam face is sampled by both blocks' tangential axes, so those axes
    must agree on fit parameters and knots (reversed under a flip), and the
    correspondence is transitive across seams. Union-find with flip parity;
    returns one group per class as ``(block, axis, flipped_vs_root)`` triples
    (singleton axes included).
    """
    parent: dict = {}
    flip: dict = {}

    def find(x):
        if x not in parent:
            parent[x] = x
            flip[x] = False
            return x, False
        if parent[x] == x:
            return x, False
        r, f = find(parent[x])
        parent[x] = r
        flip[x] = flip[x] ^ f
        return r, flip[x]

    def union(a, b, rel_flip):
        ra, fa = find(a)
        rb, fb = find(b)
        if ra == rb:
            return
        parent[rb] = ra
        flip[rb] = fa ^ fb ^ rel_flip

    for r in raw_seams:
        taxes_a = [k for k in range(d) if k != r["axis_a"]]
        taxes_b = [k for k in range(d) if k != r["axis_b"]]
        for m in range(d - 1):
            union(
                (r["a"], taxes_a[m]),
                (r["b"], taxes_b[r["perm"][m]]),
                bool(r["flips"][m]),
            )

    groups: dict = {}
    for bi in range(nb):
        for k in range(d):
            root, f = find((bi, k))
            groups.setdefault(root, []).append((bi, k, f))
    return list(groups.values())


def _harmonize_axis_params(params, groups: list[list]) -> None:
    """Force seam-corresponding block axes to share ONE parameter array.

    Watertightness (bitwise-shared face nodes on the same spline) requires
    chained axes to use identical parameters (reversed under a flip):
    average within each class, then write the class array back to every
    member in its own orientation.
    """
    for members in groups:
        if len(members) < 2:
            continue
        acc = None
        for bi, k, f in members:
            p = params[bi][k]
            p = 1.0 - p[::-1] if f else p
            acc = p if acc is None else acc + p
        acc = acc / len(members)
        acc[0], acc[-1] = 0.0, 1.0
        for bi, k, f in members:
            params[bi][k] = (1.0 - acc[::-1]) if f else acc.copy()


def _fan_axis_ends(grid, seam_faces: set) -> set[tuple[int, int, int]]:
    """``(block, axis, end)`` triples whose axis end lands on a singular fan.

    For a fan corner every axis of every adjacent block ends there; for a 3D
    fan edge the two transverse axes end on it. These are the axis ends where
    the tensor net needs extra tangential resolution (the fan window is
    C1-penalty-only, and the tight turning around the singularity is the one
    place a coarse net systematically underfits).
    """
    d = grid.topology.d
    fan_gids, fan_edge_gids = _fan_nodes(grid, seam_faces)
    marks: set[tuple[int, int, int]] = set()
    if not fan_gids and not fan_edge_gids:
        return marks
    for bi, blk in enumerate(grid.blocks):
        dm = grid.block_dof_maps[bi]
        shape = blk.logical_shape
        for corner in product(*((0, n - 1) for n in shape)):
            if int(dm[corner]) in fan_gids:
                for k in range(d):
                    marks.add((bi, k, 0 if corner[k] == 0 else 1))
        if fan_edge_gids and d == 3:
            for free_ax in range(3):
                f1, f2 = (a for a in range(3) if a != free_ax)
                for s1 in (0, 1):
                    for s2 in (0, 1):
                        idx: list = [slice(None)] * 3
                        idx[f1] = 0 if s1 == 0 else shape[f1] - 1
                        idx[f2] = 0 if s2 == 0 else shape[f2] - 1
                        line = dm[tuple(idx)]
                        if line.shape[0] < 3:
                            continue
                        if int(line[line.shape[0] // 2]) in fan_edge_gids:
                            marks.add((bi, f1, s1))
                            marks.add((bi, f2, s2))
    return marks


def _corner_mindet(X: np.ndarray) -> float:
    """Min corner-jacobian determinant over all cells of one block.

    For every cell and each of its ``2**d`` corners, the determinant of the
    d outward edge vectors (orientation-signed). Pure geometry — a cheap,
    metric-independent validity probe for a candidate fine grid.
    """
    d = X.ndim - 1
    m = np.inf
    for corner in product((0, 1), repeat=d):
        base_sl = tuple(slice(0, -1) if c == 0 else slice(1, None) for c in corner)
        p = X[base_sl]
        edges = []
        for k in range(d):
            sl = tuple(
                (slice(1, None) if corner[a] == 0 else slice(0, -1))
                if a == k
                else base_sl[a]
                for a in range(d)
            )
            edges.append((X[sl] - p) * (1.0 if corner[k] == 0 else -1.0))
        det = np.linalg.det(np.stack(edges, axis=-1))
        m = min(m, float(det.min()))
    return m


def build_control_topology(
    grid,
    ctrl_shapes: list[tuple[int, ...]] | None = None,
    *,
    degree: int = 3,
    c1: bool = True,
    walls: bool = False,
    fit_spacing="uniform",
    fan_refine: int = 0,
    knots: list[list[np.ndarray]] | None = None,
) -> ControlTopology:
    """Build the reduced control parameterization for a conforming multi-block
    grid: per-block nets, seam discovery, C0 union-find, S/T elimination with
    crossing composition, fan fallback, and the initial C1-projected state.

    ``walls=True`` detects outer faces riding one CAD entity
    (:func:`detect_control_walls`) and classifies their plain-P roots as
    sliding (``topo.root_entity``); the initial state projects them onto the
    entity. The runners own the per-iteration tangential frames.

    ``fan_refine`` inserts up to that many extra knots at every axis end
    landing on a singular fan (chain-wide, so seams stay conforming): the fan
    window is C1-penalty-only and the coarse net systematically underfits the
    tight turning there. ``knots`` overrides the per-block per-axis knot
    vectors outright (persistence reload — a net is defined by its knots);
    ``fan_refine`` is ignored then, and ``ctrl_shapes`` must match.
    """
    d = grid.topology.d
    nb = len(grid.blocks)
    if ctrl_shapes is None:
        ctrl_shapes = default_ctrl_shapes(grid)
    ctrl_shapes = [tuple(int(x) for x in s) for s in ctrl_shapes]
    ctrl_shapes_in = list(ctrl_shapes)
    refined_base_knots: dict[tuple[int, int], np.ndarray] = {}

    raw = _match_faces(grid)

    # Fit parameters (per block, per axis) and the knot vectors derived from
    # them. The fit samples the grid's own (seam-harmonized) chord parameters
    # under fit_spacing="chord", so clustered initial grids stay representable
    # by the net; the knots follow the sample quantiles (fit_knots) so every
    # knot span keeps least-squares support — uniform parameters reproduce the
    # clamped-uniform knots exactly.
    if isinstance(fit_spacing, str) and fit_spacing == "chord":
        axis_params = _chord_axis_params(grid)
        _harmonize_axis_params(axis_params, _axis_chain_groups(raw, nb, d))
    elif isinstance(fit_spacing, str) and fit_spacing == "uniform":
        axis_params = [
            [np.linspace(0.0, 1.0, n) for n in blk.logical_shape] for blk in grid.blocks
        ]
    elif not isinstance(fit_spacing, str):
        # Explicit per-block per-axis parameters (persistence reload).
        axis_params = [
            [np.asarray(p, dtype=float) for p in per_block] for per_block in fit_spacing
        ]
    else:
        raise ValueError(
            f"fit_spacing must be 'chord', 'uniform' or explicit parameter "
            f"arrays, got {fit_spacing!r}"
        )

    if knots is not None:
        axis_knots = [
            [np.asarray(u, dtype=float) for u in per_block] for per_block in knots
        ]
        for bi in range(nb):
            derived = tuple(len(axis_knots[bi][k]) - degree - 1 for k in range(d))
            if derived != ctrl_shapes[bi]:
                raise ValueError(
                    f"explicit knots for block {bi} imply control counts "
                    f"{derived}, ctrl_shapes says {ctrl_shapes[bi]}"
                )
    else:
        axis_knots = [
            [
                fit_knots(ctrl_shapes[bi][k], degree, axis_params[bi][k])
                for k in range(d)
            ]
            for bi in range(nb)
        ]
        if fan_refine > 0:
            seam_faces_raw = {(r["a"], r["axis_a"], r["side_a"]) for r in raw} | {
                (r["b"], r["axis_b"], r["side_b"]) for r in raw
            }
            marks = _fan_axis_ends(grid, seam_faces_raw)
            if marks:
                shapes_mut = [list(s) for s in ctrl_shapes]
                for members in _axis_chain_groups(raw, nb, d):
                    # Refinement requests in the chain's canonical orientation
                    # (an end refined anywhere in the chain refines the whole
                    # chain — seam conformity requires chain-equal knots).
                    ends = [0, 0]
                    for bi, k, f in members:
                        for end in (0, 1):
                            if (bi, k, end) in marks:
                                ends[end ^ f] = int(fan_refine)
                    if ends == [0, 0]:
                        continue
                    bi0, k0, f0 = members[0]
                    n_base = shapes_mut[bi0][k0]
                    p0 = np.asarray(axis_params[bi0][k0], dtype=float)
                    if f0:
                        p0 = 1.0 - p0[::-1]
                    U, n_new = refine_fit_knots(n_base, degree, p0, (ends[0], ends[1]))
                    if n_new == n_base:
                        continue
                    # The initial fit runs at the BASE density (a refined-
                    # density least squares has too few samples per end span
                    # and wiggles); the fitted net is densified by exact knot
                    # insertion at the fit site below.
                    U_base = fit_knots(n_base, degree, p0)
                    for bi, k, f in members:
                        axis_knots[bi][k] = (1.0 - U[::-1]) if f else U.copy()
                        refined_base_knots[(bi, k)] = (
                            (1.0 - U_base[::-1]) if f else U_base.copy()
                        )
                        shapes_mut[bi][k] = n_new
                ctrl_shapes = [tuple(s) for s in shapes_mut]

    seams: list[Seam] = []
    for r in raw:
        sa, sb = ctrl_shapes[r["a"]], ctrl_shapes[r["b"]]
        na, nsa = sa[r["axis_a"]], grid.blocks[r["a"]].logical_shape[r["axis_a"]]
        nbv, nsb = sb[r["axis_b"]], grid.blocks[r["b"]].logical_shape[r["axis_b"]]
        tang_a = tuple(sa[k] for k in range(d) if k != r["axis_a"])
        tang_b = tuple(sb[k] for k in range(d) if k != r["axis_b"])
        for m in range(d - 1):
            if tang_a[m] != tang_b[r["perm"][m]]:
                raise ValueError(
                    "control counts differ across a seam; conforming topology "
                    "requires equal tangential control counts"
                )
        s = Seam(
            a=r["a"],
            axis_a=r["axis_a"],
            side_a=r["side_a"],
            b=r["b"],
            axis_b=r["axis_b"],
            side_b=r["side_b"],
            perm=r["perm"],
            flips=r["flips"],
            mu_a=_end_mu(
                na, degree, r["side_a"], nsa - 1, knots=axis_knots[r["a"]][r["axis_a"]]
            ),
            mu_b=_end_mu(
                nbv, degree, r["side_b"], nsb - 1, knots=axis_knots[r["b"]][r["axis_b"]]
            ),
            tang_shape_a=tang_a,
            tang_shape_b=tang_b,
        )
        s.elim = np.full(tang_a, bool(c1))
        s.fan = np.zeros(tang_a, dtype=bool)
        seams.append(s)
    _fan_mask(grid, seams, ctrl_shapes)

    # Per-block encodings (S/T tensor composition), then the signed union-find.
    seams_by_face: list[dict] = [dict() for _ in range(nb)]
    for si, s in enumerate(seams):
        seams_by_face[s.a][(s.axis_a, s.side_a)] = (s, True)
        seams_by_face[s.b][(s.axis_b, s.side_b)] = (s, False)
    encodings = [
        _block_encodings(bi, ctrl_shapes[bi], seams_by_face[bi]) for bi in range(nb)
    ]

    uf = _SignedUF()
    all_syms: set = set()
    for enc in encodings:
        for terms in enc.values():
            for sym, _ in terms:
                all_syms.add(sym)
                uf.add(sym)
    for s in seams:
        for sym in sorted(all_syms):
            img = _map_symbol(sym, s, ctrl_shapes)
            if img is None:
                continue
            target, sgn = img
            if target not in all_syms:
                raise ValueError(
                    f"seam symbol {sym} has no counterpart {target}; the "
                    "interface layout is inconsistent (unmasked fan?)"
                )
            uf.union(sym, target, sgn)

    # Canonical roots -> columns of R.
    root_col: dict = {}
    root_syms: list = []
    for sym in sorted(all_syms):
        r, _ = uf.find(sym)
        if r not in root_col:
            root_col[r] = len(root_syms)
            root_syms.append(r)
    n_roots = len(root_syms)

    block_offsets = np.zeros(nb + 1, dtype=int)
    for bi in range(nb):
        block_offsets[bi + 1] = block_offsets[bi] + int(np.prod(ctrl_shapes[bi]))
    N_total = int(block_offsets[-1])

    # R is assembled sparse (a handful of entries per row: a plain control
    # references one root, an eliminated first-interior control its seam's S
    # and T, a 3D crossing corner up to 8) — a dense (N_total, n_roots) array
    # is the at-scale ceiling, not the solver.
    from scipy import sparse as _sp

    coo_r: list[int] = []
    coo_c: list[int] = []
    coo_v: list[float] = []
    for bi in range(nb):
        for flat, terms in encodings[bi].items():
            row = block_offsets[bi] + flat
            for sym, coeff in terms:
                r, sgn = uf.find(sym)
                coo_r.append(int(row))
                coo_c.append(root_col[r])
                coo_v.append(float(coeff * sgn))
    R = _sp.csr_matrix((coo_v, (coo_r, coo_c)), shape=(N_total, n_roots), dtype=float)

    # Free/slide/fixed. Without walls: a root is fixed when any member symbol
    # touches an outer face. With walls: a plain-P root whose outer faces all
    # ride ONE entity slides on it (recorded in root_entity; the runners give
    # it tangential columns and reproject per iteration); T-carrying roots and
    # mixed-entity roots stay fixed (positions slide, tangent rows do not —
    # the corner/wall window owns the direction there).
    seam_faces = {(s.a, s.axis_a, s.side_a) for s in seams} | {
        (s.b, s.axis_b, s.side_b) for s in seams
    }
    wall_faces = detect_control_walls(grid, seam_faces) if walls else {}
    root_free = np.ones(n_roots, dtype=bool)
    root_ent_sets: dict[int, set] = {}
    root_hard_fixed = np.zeros(n_roots, dtype=bool)
    for sym in all_syms:
        faces = _outer_faces(sym, ctrl_shapes, seam_faces)
        if not faces:
            continue
        r, _ = uf.find(sym)
        col = root_col[r]
        root_free[col] = False
        if _has_t(sym) or any(f not in wall_faces for f in faces):
            root_hard_fixed[col] = True
        else:
            root_ent_sets.setdefault(col, set()).update(
                id(wall_faces[f]) for f in faces
            )
    # Geometric creases: a block-corner fine node where the wall polyline
    # kinks (e.g. a capsule's shoulder/knee on one composite profile) must
    # stay PINNED — a sliding C2 net rounds the corner off otherwise. The
    # kink is measured on the initial boundary polyline at every wall block
    # corner; the corresponding block-corner roots go fixed.
    crease_gids = _wall_crease_corners(grid, wall_faces) if wall_faces else set()
    if crease_gids:
        for sym in all_syms:
            bi, comps = sym
            if _has_t(sym):
                continue
            if not all(
                c[1] in (0, ctrl_shapes[bi][k] - 1) for k, c in enumerate(comps)
            ):
                continue  # not a block-corner control
            corner = tuple(
                0 if c[1] == 0 else grid.blocks[bi].logical_shape[k] - 1
                for k, c in enumerate(comps)
            )
            if int(grid.block_dof_maps[bi][corner]) in crease_gids:
                r, _ = uf.find(sym)
                root_hard_fixed[root_col[r]] = True
    root_entity: dict[int, object] = {}
    if wall_faces:
        ent_by_id = {id(e): e for e in wall_faces.values()}
        for col, ids in root_ent_sets.items():
            if not root_hard_fixed[col] and len(ids) == 1:
                root_entity[col] = ent_by_id[next(iter(ids))]

    # T/S root lookup per seam face-lattice index (seam ortho hard mode +
    # diagnostics). Near a crossing the S/T lattice entries are themselves
    # eliminated by the transverse seam (they are combinations of the
    # crossing's S/T/twist, not plain roots) — those indices are absent here;
    # row-based penalties through R cover them instead.
    t_root: dict = {}
    s_root: dict = {}
    for si, s in enumerate(seams):
        face_i = 0 if s.side_a == 0 else ctrl_shapes[s.a][s.axis_a] - 1
        taxes = [k for k in range(d) if k != s.axis_a]
        for tang in product(*(range(n) for n in s.tang_shape_a)):
            comps = [None] * d
            comps[s.axis_a] = ("P", face_i)
            for m, k in enumerate(taxes):
                comps[k] = ("P", tang[m])
            sym = (s.a, tuple(comps))
            if sym in uf.parent:
                r, _ = uf.find(sym)
                s_root[(si, tang)] = root_col[r]
            if s.elim[tang]:
                comps[s.axis_a] = ("T", s.side_a)
                sym = (s.a, tuple(comps))
                if sym in uf.parent:
                    r, sgn = uf.find(sym)
                    t_root[(si, tang)] = (root_col[r], sgn)

    # Initial state: per-block LSQ fits, projected onto the C1 manifold.
    # Fan-refined axes fit at the base density and densify by exact knot
    # insertion — shape-preserving, so refinement cannot destabilize the fit.
    cmaps = []
    C0 = np.zeros((N_total, d))
    for bi, blk in enumerate(grid.blocks):
        cmap = tensor_map(
            tuple(blk.logical_shape),
            ctrl_shapes[bi],
            degree=degree,
            spacing=axis_params[bi],
            knots=axis_knots[bi],
        )
        cmaps.append(cmap)
        ref_axes = [k for k in range(d) if (bi, k) in refined_base_knots]
        if ref_axes:
            fit_cmap = tensor_map(
                tuple(blk.logical_shape),
                ctrl_shapes_in[bi],
                degree=degree,
                spacing=axis_params[bi],
                knots=[
                    refined_base_knots.get((bi, k), axis_knots[bi][k]) for k in range(d)
                ],
            )
            Cb = fit_control_net(fit_cmap, np.asarray(blk.nodes, dtype=float))
            for k in ref_axes:
                ins = np.setdiff1d(axis_knots[bi][k], refined_base_knots[(bi, k)])
                A, _ = insertion_matrix(degree, refined_base_knots[(bi, k)], ins)
                Cb = np.moveaxis(np.tensordot(A, Cb, axes=([1], [k])), 0, k)
        else:
            Cb = fit_control_net(cmap, np.asarray(blk.nodes, dtype=float))
        C0[block_offsets[bi] : block_offsets[bi + 1]] = Cb.reshape(-1, d)
    q0 = lstsq_q(R, C0)
    # Sliding roots start ON their entity (mirrors the single-block wall
    # reference; b then absorbs the boundary delta against the projected
    # spline).
    for col, ent in root_entity.items():
        q0[col] = ent.project(q0[col])
    C_proj = R @ q0
    fit_residual = float(np.abs(C_proj - C0).max())

    # Boolean-sum b per block: exact on outer faces, identically zero on seams.
    b_fields = []
    state_mindet = np.inf
    for bi, blk in enumerate(grid.blocks):
        node_shape = tuple(blk.logical_shape)
        X0 = np.asarray(blk.nodes, dtype=float)
        Xs = cmaps[bi].prolong(
            C_proj[block_offsets[bi] : block_offsets[bi + 1]].reshape(
                ctrl_shapes[bi] + (d,)
            )
        )
        delta = np.zeros(node_shape + (d,))
        bmask = boundary_ctrl_mask(node_shape)
        delta[bmask] = (X0 - Xs)[bmask]
        for k in range(d):
            for side in (0, 1):
                if (bi, k, side) in seam_faces:
                    delta[_face_slab(k, side, node_shape)] = 0.0
        b_fields.append(_tfi_apply(node_shape, delta, bmask))
        if ctrl_shapes[bi] != ctrl_shapes_in[bi]:
            state_mindet = min(state_mindet, _corner_mindet(Xs + b_fields[bi]))

    # A refined fit that folds is an over-complete fit (too few samples per
    # refined span for a stable least squares) — back the refinement off one
    # level and rebuild rather than hand the solver an invalid start.
    if ctrl_shapes != ctrl_shapes_in and state_mindet <= 0.0:
        return build_control_topology(
            grid,
            ctrl_shapes_in,
            degree=degree,
            c1=c1,
            walls=walls,
            fit_spacing=fit_spacing,
            fan_refine=fan_refine - 1,
        )

    return ControlTopology(
        grid=grid,
        d=d,
        ctrl_shapes=ctrl_shapes,
        cmaps=cmaps,
        seams=seams,
        R=R,
        root_free=root_free,
        root_syms=root_syms,
        q=q0,
        block_offsets=block_offsets,
        b_fields=b_fields,
        t_root=t_root,
        s_root=s_root,
        fit_residual=fit_residual,
        wall_faces=wall_faces,
        root_entity=root_entity,
        axis_params=axis_params,
    )


def wall_b_field(topo: ControlTopology, bi: int, X0: np.ndarray) -> np.ndarray:
    """Re-extend one block's Boolean-sum offset against the CURRENT spline.

    Wall faces take the projection of the spline boundary onto their entity
    (the boundary rides the CAD exactly while the controls slide); other
    outer faces keep the initial exact boundary ``X0``; seam faces stay zero
    (S/T owns cross-seam continuity).
    """
    d = topo.d
    node_shape = tuple(topo.grid.blocks[bi].logical_shape)
    seam_faces = {(s.a, s.axis_a, s.side_a) for s in topo.seams} | {
        (s.b, s.axis_b, s.side_b) for s in topo.seams
    }
    Xs = topo.cmaps[bi].prolong(topo.block_C(bi))
    exact = np.asarray(X0, dtype=float).copy()
    for k in range(d):
        for side in (0, 1):
            f = (bi, k, side)
            if f not in topo.wall_faces:
                continue
            sl = _face_slab(k, side, node_shape)
            ent = topo.wall_faces[f]
            pts = Xs[sl].reshape(-1, d)
            exact[sl] = _project_many(ent, pts).reshape(exact[sl].shape)
    delta = np.zeros(node_shape + (d,))
    bmask = boundary_ctrl_mask(node_shape)
    delta[bmask] = (exact - Xs)[bmask]
    for k in range(d):
        for side in (0, 1):
            if (bi, k, side) in seam_faces:
                delta[_face_slab(k, side, node_shape)] = 0.0
    return _tfi_apply(node_shape, delta, bmask)


def _project_many(ent, pts: np.ndarray) -> np.ndarray:
    """Vectorized entity projection where available (single-point Python
    Newton loops dominate the frame-rebuild profile otherwise)."""
    if hasattr(ent, "project_many"):
        return np.asarray(ent.project_many(pts))
    return np.array([ent.project(p) for p in pts])


def _tfi_apply(node_shape: tuple[int, ...], delta: np.ndarray, bmask) -> np.ndarray:
    """Boolean-sum interior fill of a boundary delta field, vectorized.

    The Boolean-sum fill equals ``sum_ax lerp_ax(delta) - (D-1)*cornerblend``
    with ``lerp_ax`` the 1-D linear interpolation between an axis's two faces
    and ``cornerblend`` the multilinear corner interpolation (the chain of
    all D lerps) — pure O(N*D) array ops, equal to
    ``egg.init.tfi.tfi_fill_interior`` on the interior (the per-node Python
    fill and the dense ``tfi_operator`` both dominated the build/rebuild
    profiles). Boundary rows pass through.
    """
    dims = len(node_shape)
    xi = [np.linspace(0.0, 1.0, n) for n in node_shape]

    def lerp_axis(F, ax):
        sh = [1] * (dims + 1)
        sh[ax] = node_shape[ax]
        t = xi[ax].reshape(sh)
        f0 = np.take(F, [0], axis=ax)
        f1 = np.take(F, [-1], axis=ax)
        return (1.0 - t) * f0 + t * f1

    p_sum = lerp_axis(delta, 0)
    for ax in range(1, dims):
        p_sum = p_sum + lerp_axis(delta, ax)
    corner = delta
    for ax in range(dims):
        corner = lerp_axis(corner, ax)
    out = p_sum - (dims - 1) * corner
    out[bmask] = delta[bmask]
    return out


def resample_block(
    topo: ControlTopology,
    bi: int,
    node_shape: tuple[int, ...] | None = None,
    spacing: list | None = None,
) -> np.ndarray:
    """Evaluate one block's net at a new node sampling — the algebraic regrid.

    Refine/derefine/cluster without a re-solve: new per-axis node counts
    (``node_shape``) and/or clustered parameter distributions (``spacing``,
    per axis in [0, 1]) re-tabulate the basis on the SAME control net. The
    boundary re-acquires exactness through a fresh Boolean-sum correction:
    wall faces project the resampled spline boundary onto their entity, other
    outer faces interpolate the stored correction at the new parameters
    (exact where the boundary is CAD; O(h²)-faithful elsewhere), and seam
    faces stay pure spline — two blocks resampled with matching seam
    parameters remain watertight.
    """
    d = topo.d
    old_shape = tuple(topo.grid.blocks[bi].logical_shape)
    if node_shape is None:
        node_shape = old_shape
    node_shape = tuple(int(n) for n in node_shape)
    if spacing is None:
        spacing = [None] * d
    params = [
        np.linspace(0.0, 1.0, node_shape[k])
        if spacing[k] is None
        else np.asarray(spacing[k], dtype=float)
        for k in range(d)
    ]
    cmap = tensor_map(
        node_shape,
        topo.ctrl_shapes[bi],
        spacing=params,
        knots=topo.cmaps[bi].knots,
    )
    Xs = cmap.prolong(topo.block_C(bi))

    seam_faces = {(s.a, s.axis_a, s.side_a) for s in topo.seams} | {
        (s.b, s.axis_b, s.side_b) for s in topo.seams
    }
    delta = np.zeros(node_shape + (d,))
    bmask = boundary_ctrl_mask(node_shape)
    # Interpolate the stored correction's boundary values at the new
    # parameters (per outer face; overwritten below where the face is CAD).
    from scipy.interpolate import RegularGridInterpolator

    old_params = (
        [np.asarray(p, dtype=float) for p in topo.axis_params[bi]]
        if topo.axis_params
        else [np.linspace(0.0, 1.0, n) for n in old_shape]
    )
    b_old = topo.b_fields[bi]
    for k in range(d):
        for side in (0, 1):
            f = (bi, k, side)
            if f in seam_faces:
                continue
            sl_new = _face_slab(k, side, node_shape)
            sl_old = _face_slab(k, side, old_shape)
            taxes = [a for a in range(d) if a != k]
            interp = RegularGridInterpolator(
                [old_params[a] for a in taxes], b_old[sl_old], method="linear"
            )
            pts = np.stack(
                np.meshgrid(*(params[a] for a in taxes), indexing="ij"), axis=-1
            )
            delta[sl_new] = interp(pts.reshape(-1, d - 1)).reshape(delta[sl_new].shape)
            if f in topo.wall_faces:
                ent = topo.wall_faces[f]
                face_pts = Xs[sl_new].reshape(-1, d)
                proj = _project_many(ent, face_pts)
                delta[sl_new] = (proj - face_pts).reshape(delta[sl_new].shape)
    delta[~bmask] = 0.0
    for k in range(d):
        for side in (0, 1):
            if (bi, k, side) in seam_faces:
                delta[_face_slab(k, side, node_shape)] = 0.0
    return Xs + _tfi_apply(node_shape, delta, bmask)


def cluster_spacings(topo: ControlTopology, specs_by_entity: dict) -> dict:
    """Per-block per-axis clustered node parameters for boundary-layer specs.

    ``specs_by_entity`` maps entity id -> spec dict with ``first_height`` /
    ``growth``. For every wall face whose entity carries a spec, the face's
    NORMAL axis gets a geometric-law parameter distribution toward the wall.
    The normalized first spacing is spec-global (first_height over the MEAN
    column length across every face on that entity), so blocks sharing a
    clustered axis through a seam sample it identically — per-block spacings
    would break watertightness on seam faces that carry the clustered axis.

    Returns ``{block: [spacing_or_None per axis]}`` for blocks with at least
    one clustered axis (ready for :func:`resample_block`).
    """
    from egg.smoothing.respace import _solve_stretch_ratio

    grid = topo.grid
    d = topo.d
    # Mean column length per entity (over all faces on it).
    col_len: dict[int, list[float]] = {}
    for (bi, axis, _side), ent in topo.wall_faces.items():
        if id(ent) not in specs_by_entity:
            continue
        X = np.asarray(grid.blocks[bi].nodes, dtype=float)
        lengths = np.linalg.norm(np.diff(X, axis=axis), axis=-1).sum(axis=axis)
        col_len.setdefault(id(ent), []).extend(np.ravel(lengths).tolist())

    out: dict[int, list] = {}
    ent_groups: dict[int, list] = {}
    for (bi, axis, side), ent in topo.wall_faces.items():
        spec = specs_by_entity.get(id(ent))
        if spec is None:
            continue
        n_cells = int(grid.blocks[bi].logical_shape[axis]) - 1
        mean_len = float(np.mean(col_len[id(ent)]))
        s0 = float(spec["first_height"]) / mean_len
        # Ratio that makes the geometric sum fill [0, 1] exactly (the
        # requested growth shapes the law; the solve pins the total). The
        # law lives in ARC-LENGTH fractions; map it into the net's parameter
        # space through the inverse of the current param -> arc mapping
        # (identity for a uniform-sampled smooth grid; nontrivial when the
        # fit used chord parameters on a clustered grid).
        r = _solve_stretch_ratio(s0, n_cells, 1.0)
        steps = s0 * np.power(r, np.arange(n_cells))
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        arc = arc / arc[-1]
        if side == 1:
            arc = 1.0 - arc[::-1]
        arc_now = _chord_axis_params(grid)[bi][axis]
        params_now = (
            np.asarray(topo.axis_params[bi][axis], dtype=float)
            if topo.axis_params
            else np.linspace(0.0, 1.0, n_cells + 1)
        )
        params = np.interp(arc, arc_now, params_now)
        params[0], params[-1] = 0.0, 1.0
        spacing = out.setdefault(bi, [None] * d)
        spacing[axis] = params
        ent_slot = ent_groups.setdefault(id(ent), [])
        ent_slot.append((bi, axis, side))
    # Blocks sharing a clustered axis through a seam must sample it
    # identically (watertightness): average the mapped parameters per entity
    # (orientation-normalized), then reassign.
    for eid, faces in ent_groups.items():
        if len(faces) < 2:
            continue
        forms = []
        for bi, axis, side in faces:
            p = out[bi][axis]
            forms.append(1.0 - p[::-1] if side == 1 else p)
        if len({f.shape[0] for f in forms}) != 1:
            continue  # non-conforming counts: leave per-block params
        mean = np.mean(forms, axis=0)
        mean[0], mean[-1] = 0.0, 1.0
        for bi, axis, side in faces:
            out[bi][axis] = 1.0 - mean[::-1] if side == 1 else mean.copy()
    return out


# --------------------------------------------------------------------------- #
# Seam diagnostics (acceptance measurements, all on the analytic spline map)
# --------------------------------------------------------------------------- #


def watertight_mismatch(topo: ControlTopology) -> float:
    """Max distance between the two sides' spline evaluations of shared nodes."""
    grid = topo.grid
    worst = 0.0
    vals: dict[int, np.ndarray] = {}
    for bi, cmap in enumerate(topo.cmaps):
        X = cmap.prolong(topo.block_C(bi)) + topo.b_fields[bi]
        dm = grid.block_dof_maps[bi].reshape(-1)
        Xf = X.reshape(-1, topo.d)
        for row, gid in enumerate(dm):
            g = int(gid)
            if g in vals:
                worst = max(worst, float(np.linalg.norm(vals[g] - Xf[row])))
            else:
                vals[g] = Xf[row]
    return worst


def _face_derivative_rows(cmap, axis: int, side: int, order: int) -> np.ndarray:
    """The (order)-th derivative extraction row along ``axis`` at the face."""
    row = 0 if side == 0 else cmap.node_shape[axis] - 1
    src = [cmap.basis, cmap.basis_d1, cmap.basis_d2][order]
    return src[axis][row]


def _seam_samples(topo: ControlTopology, s: Seam, n: int, degree: int = 3):
    """Matched parameter samples along a seam: per tangential axis, ``n``
    interior points in side-a coordinates plus the side-b images."""
    d = topo.d
    taxes_a = [k for k in range(d) if k != s.axis_a]
    taxes_b = [k for k in range(d) if k != s.axis_b]
    ts = np.linspace(0.05, 0.95, n)
    grids = np.meshgrid(*([ts] * (d - 1)), indexing="ij")
    pts_a = np.stack([g.ravel() for g in grids], axis=-1)  # (S, d-1) in [0,1]
    pts_b = np.empty_like(pts_a)
    for m in range(d - 1):
        p = s.perm[m]
        pts_b[:, p] = 1.0 - pts_a[:, m] if s.flips[m] else pts_a[:, m]
    return taxes_a, taxes_b, pts_a, pts_b


def _spline_face_field(
    topo: ControlTopology, bi: int, axis: int, side: int, pts, orders
) -> np.ndarray:
    """Evaluate d^orders X at face parameter points: ``orders[k]`` is the
    derivative order on tangential slot k; the seam-normal axis uses
    derivative order ``orders[-1]`` at the face itself."""
    d = topo.d
    cs = topo.ctrl_shapes[bi]
    C = topo.block_C(bi)
    taxes = [k for k in range(d) if k != axis]
    n_row = _face_derivative_rows(topo.cmaps[bi], axis, side, orders[-1])
    out = np.zeros((pts.shape[0], d))
    for pi in range(pts.shape[0]):
        weights = [None] * d
        for m, k in enumerate(taxes):
            B = bspline_basis(
                cs[k], 3, [pts[pi, m]], n_deriv=2, knots=topo.cmaps[bi].knots[k]
            )[0]
            weights[k] = B[orders[m]][0]
        weights[axis] = n_row
        W = weights[0]
        for k in range(1, d):
            W = np.multiply.outer(W, weights[k])
        out[pi] = np.tensordot(W, C, axes=d)
    return out


def seam_c1_jump(topo: ControlTopology, seam_idx: int, n: int = 9) -> float:
    """Max mismatch of the per-global-cell first normal derivative across the
    seam, sampled on the analytic spline map (zero at eliminated indices)."""
    s = topo.seams[seam_idx]
    _, _, pts_a, pts_b = _seam_samples(topo, s, n)
    La = float(topo.grid.blocks[s.a].logical_shape[s.axis_a] - 1)
    Lb = float(topo.grid.blocks[s.b].logical_shape[s.axis_b] - 1)
    orders = [0] * (topo.d - 1) + [1]
    da = _spline_face_field(topo, s.a, s.axis_a, s.side_a, pts_a, orders) / La
    db = _spline_face_field(topo, s.b, s.axis_b, s.side_b, pts_b, orders) / Lb
    # Outward directions are opposite: continuity means da_out = -db_out; in
    # raw parameter derivatives the sign depends on the sides.
    sgn_a = 1.0 if s.side_a == 1 else -1.0
    sgn_b = 1.0 if s.side_b == 1 else -1.0
    return float(np.abs(sgn_a * da + sgn_b * db).max())


def seam_curvature_mismatch(topo: ControlTopology, seam_idx: int, n: int = 9) -> float:
    """Max mismatch of the per-global-cell second normal derivative (the soft
    C2 target quantity), sampled on the analytic spline map."""
    s = topo.seams[seam_idx]
    _, _, pts_a, pts_b = _seam_samples(topo, s, n)
    La = float(topo.grid.blocks[s.a].logical_shape[s.axis_a] - 1)
    Lb = float(topo.grid.blocks[s.b].logical_shape[s.axis_b] - 1)
    orders = [0] * (topo.d - 1) + [2]
    da = _spline_face_field(topo, s.a, s.axis_a, s.side_a, pts_a, orders) / La**2
    db = _spline_face_field(topo, s.b, s.axis_b, s.side_b, pts_b, orders) / Lb**2
    return float(np.abs(da - db).max())


def seam_angle_deviation(
    topo: ControlTopology, seam_idx: int, n: int = 17, skip: int = 0
) -> np.ndarray:
    """Angle (degrees) between the seam-crossing grid line and the seam normal,
    per sample: measured analytically as the angle between the cross-seam
    derivative and the seam tangent plane (2D: the tangent direction)."""
    s = topo.seams[seam_idx]
    _, _, pts_a, _ = _seam_samples(topo, s, n)
    orders_d = [0] * (topo.d - 1) + [1]
    dn = _spline_face_field(topo, s.a, s.axis_a, s.side_a, pts_a, orders_d)
    sample_shape = (n,) * (topo.d - 1)
    devs = []
    for pi in range(pts_a.shape[0]):
        if skip:
            midx = np.unravel_index(pi, sample_shape)
            if any(not (skip <= i < n - skip) for i in midx):
                continue
        v = dn[pi]
        v = v / np.linalg.norm(v)
        worst = 0.0
        for m in range(topo.d - 1):
            orders_t = [0] * (topo.d - 1) + [0]
            orders_t[m] = 1
            t = _spline_face_field(
                topo, s.a, s.axis_a, s.side_a, pts_a[pi : pi + 1], orders_t
            )[0]
            t = t / np.linalg.norm(t)
            worst = max(worst, abs(float(t @ v)))
        devs.append(np.degrees(np.arcsin(min(worst, 1.0))))
    return np.asarray(devs)


# --------------------------------------------------------------------------- #
# Penalty rows (soft C2, fan C1) and seam orthogonality
# --------------------------------------------------------------------------- #


@dataclass
class SeamOrthoSpec:
    """Per-seam orthogonality dial: ``"off"``, ``"penalty"``, or ``"hard"``.

    ``weight`` (penalty mode) is a scalar or one value per side-a face-lattice
    index (a taper); zero entries disable individual tangent rows.
    """

    ortho: str = "off"
    weight: float | np.ndarray = 1.0


def _c2_rows(
    topo: ControlTopology, degree: int = 3, *, reduced: bool = True
) -> np.ndarray:
    """Quadratic C2 rows: second normal derivative mismatch at tangential
    Greville points, per global cell index squared, scaled by the seam's mean
    first-cell size (nondimensional residual). ``reduced`` returns rows over
    the reduced DOFs (through R); ``reduced=False`` returns the underlying
    stacked-control rows (the device wire's form — identical residuals since
    C = R q)."""
    d = topo.d
    rows = []
    for s in topo.seams:
        La = float(topo.grid.blocks[s.a].logical_shape[s.axis_a] - 1)
        Lb = float(topo.grid.blocks[s.b].logical_shape[s.axis_b] - 1)
        d2a = _face_derivative_rows(topo.cmaps[s.a], s.axis_a, s.side_a, 2) / La**2
        d2b = _face_derivative_rows(topo.cmaps[s.b], s.axis_b, s.side_b, 2) / Lb**2
        taxes_a = [k for k in range(d) if k != s.axis_a]
        taxes_b = [k for k in range(d) if k != s.axis_b]
        gpts = [
            greville(n, degree, knots=topo.cmaps[s.a].knots[k])
            for n, k in zip(s.tang_shape_a, taxes_a)
        ]
        gidx = list(product(*(range(len(g)) for g in gpts)))
        # Scale: mean first-cell size across the seam (from the initial grid).
        scale = _seam_cell_scale(topo, s)
        for gi in gidx:
            row = np.zeros(topo.R.shape[0])
            t_a = [gpts[m][gi[m]] for m in range(d - 1)]
            t_b = [None] * (d - 1)
            for m in range(d - 1):
                p = s.perm[m]
                t_b[p] = 1.0 - t_a[m] if s.flips[m] else t_a[m]
            _accumulate_face_row(row, topo, s.a, s.axis_a, taxes_a, t_a, d2a, +1.0)
            _accumulate_face_row(row, topo, s.b, s.axis_b, taxes_b, t_b, d2b, -1.0)
            rows.append(row * scale)
    if not rows:
        return np.zeros((0, topo.n_roots if reduced else topo.R.shape[0]))
    out = np.asarray(rows)
    return (topo.R.T @ out.T).T if reduced else out


def _accumulate_face_row(
    row: np.ndarray,
    topo: ControlTopology,
    bi: int,
    axis: int,
    taxes: list[int],
    t: list[float],
    normal_row: np.ndarray,
    sign: float,
) -> None:
    """Add a face-sample extraction row (tensor of tangential value bases and
    the given normal-axis row) into a stacked-control row vector."""
    cs = topo.ctrl_shapes[bi]
    weights = [None] * topo.d
    for m, k in enumerate(taxes):
        weights[k] = bspline_basis(
            cs[k], 3, [t[m]], n_deriv=0, knots=topo.cmaps[bi].knots[k]
        )[0][0][0]
    weights[axis] = normal_row
    W = weights[0]
    for k in range(1, topo.d):
        W = np.multiply.outer(W, weights[k])
    lo = topo.block_offsets[bi]
    row[lo : lo + W.size] += sign * W.reshape(-1)


def _seam_cell_scale(topo: ControlTopology, s: Seam) -> float:
    """Inverse mean first-cell size at the seam (nondimensionalizes C1/C2 rows,
    frozen from the initial grid)."""
    blk = topo.grid.blocks[s.a]
    X = np.asarray(blk.nodes, dtype=float)
    face = _face_slab(s.axis_a, s.side_a, blk.logical_shape)
    inner = list(face)
    inner[s.axis_a] = 1 if s.side_a == 0 else blk.logical_shape[s.axis_a] - 2
    h = np.linalg.norm(X[tuple(inner)] - X[face], axis=-1).mean()
    return 1.0 / max(float(h), 1e-300)


def _fan_c1_rows(topo: ControlTopology, *, reduced: bool = True) -> np.ndarray:
    """C1 penalty rows at fan-adjacent (non-eliminated) face-lattice indices:
    residual ``(mu_A (S - Q_A) + mu_B (S - Q_B)) / scale``. ``reduced=False``
    returns the stacked-control rows (device wire form)."""
    d = topo.d
    rows = []
    for s in topo.seams:
        if not bool(s.fan.any()):
            continue
        scale = _seam_cell_scale(topo, s)
        csa, csb = topo.ctrl_shapes[s.a], topo.ctrl_shapes[s.b]
        fa = 0 if s.side_a == 0 else csa[s.axis_a] - 1
        qa = 1 if s.side_a == 0 else csa[s.axis_a] - 2
        fb = 0 if s.side_b == 0 else csb[s.axis_b] - 1
        qb = 1 if s.side_b == 0 else csb[s.axis_b] - 2
        taxes_a = [k for k in range(d) if k != s.axis_a]
        taxes_b = [k for k in range(d) if k != s.axis_b]
        for tang in product(*(range(n) for n in s.tang_shape_a)):
            if not s.fan[tang]:
                continue
            tb = s.map_tang(tang)
            row = np.zeros(topo.R.shape[0])

            def flat(bi, axis, taxes, normal_i, tg, cs):
                idx = [0] * d
                idx[axis] = normal_i
                for m, k in enumerate(taxes):
                    idx[k] = tg[m]
                return topo.block_offsets[bi] + int(np.ravel_multi_index(idx, cs))

            row[flat(s.a, s.axis_a, taxes_a, fa, tang, csa)] += s.mu_a
            row[flat(s.a, s.axis_a, taxes_a, qa, tang, csa)] -= s.mu_a
            row[flat(s.b, s.axis_b, taxes_b, fb, tb, csb)] += s.mu_b
            row[flat(s.b, s.axis_b, taxes_b, qb, tb, csb)] -= s.mu_b
            rows.append(row / scale)
    if not rows:
        return np.zeros((0, topo.n_roots if reduced else topo.R.shape[0]))
    out = np.asarray(rows)
    return (topo.R.T @ out.T).T if reduced else out


def _smooth_rows(topo: ControlTopology, *, reduced: bool = True):
    """Frozen-frame straightness rows over interior control triplets.

    For every interior control of every lattice line of every block axis:
    the components of the control polygon's second divided difference normal
    to the local (frozen) polygon direction, scaled to the local bend angle.
    Zero for collinear controls at ANY spacing — clustering is untouched —
    so a small weight regularizes exactly the near-flat line-bow modes of
    the shape energy that a coarse net otherwise parks in (grid lines bow at
    ~zero energy cost; pointwise node relaxation drifts out of these modes,
    a converging Gauss-Newton step does not). Lattice lines lying on outer
    (non-seam) faces are skipped: wall control rows follow the CAD
    curvature, and straightening them would fight the geometry.

    Fully vectorized; in 3D the two normals per site are an arbitrary
    orthonormal basis of the plane normal to the frozen direction — the
    penalty energy sums the squared normal components, so it is invariant
    to that basis choice.

    Returns ``(row_off, cols, coefs)`` flat CSR arrays over the stacked
    control components (``reduced=False``, the device wire form — every row
    has ``3 d`` entries) or a list of dense rows over the interleaved
    reduced components (``reduced=True``, the reference runner form).
    """
    d = topo.d
    C = np.asarray(topo.R @ topo.q).reshape(-1, d)
    seam_faces = {(s.a, s.axis_a, s.side_a) for s in topo.seams} | {
        (s.b, s.axis_b, s.side_b) for s in topo.seams
    }
    col_parts: list = []
    coef_parts: list = []
    for bi, cs in enumerate(topo.ctrl_shapes):
        off = int(topo.block_offsets[bi])
        cmap = topo.cmaps[bi]
        n_lat = int(np.prod(cs))
        Cb = C[off : off + n_lat].reshape(tuple(cs) + (d,))
        Fb = (off + np.arange(n_lat)).reshape(cs)
        grev = [
            greville(
                cs[k],
                cmap.degrees[k],
                knots=cmap.knots[k] if cmap.knots is not None else None,
            )
            for k in range(d)
        ]
        outer = [[(bi, k, side) not in seam_faces for side in (0, 1)] for k in range(d)]
        for k in range(d):
            if cs[k] < 3:
                continue
            A = np.moveaxis(Cb, k, 0)  # (nk, ...transverse, d)
            F = np.moveaxis(Fb, k, 0)
            taxes = [a for a in range(d) if a != k]
            keep = np.ones(tuple(cs[a] for a in taxes), dtype=bool)
            for m, a in enumerate(taxes):
                if outer[a][0]:
                    sl = [slice(None)] * (d - 1)
                    sl[m] = 0
                    keep[tuple(sl)] = False
                if outer[a][1]:
                    sl = [slice(None)] * (d - 1)
                    sl[m] = cs[a] - 1
                    keep[tuple(sl)] = False
            g = grev[k]
            dm = np.diff(g)[:-1]  # (nk - 2,)
            dp = np.diff(g)[1:]
            ok_i = (dm > 0.0) & (dp > 0.0)
            chord = A[2:] - A[:-2]  # (nk - 2, ...transverse, d)
            L = np.linalg.norm(chord, axis=-1)
            exp = (-1,) + (1,) * (d - 1)
            mask = ok_i.reshape(exp) & (L > 0.0) & keep[None]
            Ls = np.where(L > 0.0, L, 1.0)
            t_hat = chord / Ls[..., None]
            # Second divided difference x (dm dp / length): for a gentle
            # bend this is ~the bend angle, dimensionless.
            ca = np.broadcast_to((2.0 * dp / (dm + dp)).reshape(exp), L.shape) / Ls
            cb = -2.0 / Ls
            cc = np.broadcast_to((2.0 * dm / (dm + dp)).reshape(exp), L.shape) / Ls
            if d == 2:
                normals = np.stack([-t_hat[..., 1], t_hat[..., 0]], axis=-1)[
                    ..., None, :
                ]
            else:
                am = np.argmin(np.abs(t_hat), axis=-1)
                e = np.eye(d)[am]
                v = e - t_hat * np.sum(e * t_hat, axis=-1, keepdims=True)
                n1 = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
                n2 = np.cross(t_hat, n1)
                normals = np.stack([n1, n2], axis=-2)  # (..., d - 1, d)
            fl = np.stack([F[:-2], F[1:-1], F[2:]], axis=-1)  # (..., 3)
            cj = np.stack([ca, cb, cc], axis=-1)  # (..., 3)
            cols_site = ((fl[..., :, None] * d) + np.arange(d)).reshape(
                L.shape + (3 * d,)
            )
            for m in range(d - 1):
                coef_site = (cj[..., :, None] * normals[..., m, None, :]).reshape(
                    L.shape + (3 * d,)
                )
                col_parts.append(cols_site[mask])
                coef_parts.append(coef_site[mask])
    if col_parts:
        cols = np.concatenate(col_parts)
        coefs = np.concatenate(coef_parts)
    else:
        cols = np.zeros((0, 3 * d), dtype=np.int64)
        coefs = np.zeros((0, 3 * d))
    n_rows = cols.shape[0]
    row_off = np.arange(n_rows + 1, dtype=np.int64) * (3 * d)
    if not reduced:
        return row_off, cols.reshape(-1), coefs.reshape(-1)
    from scipy import sparse as _sp

    Rd = topo.R.toarray() if _sp.issparse(topo.R) else np.asarray(topo.R)
    n = topo.n_roots
    out = []
    for j in range(n_rows):
        rowv = np.zeros(n * d)
        for t in range(3 * d):
            st, c = divmod(int(cols[j, t]), d)
            rowv[c::d] += coefs[j, t] * Rd[st]
        out.append(rowv)
    return out


def _t_row(topo: ControlTopology, s: Seam, tang: tuple[int, ...]) -> np.ndarray:
    """Scalar stacked-control row extracting ``T_j = mu_A (S_j - Q_A,j)``."""
    d = topo.d
    cs = topo.ctrl_shapes[s.a]
    fa = 0 if s.side_a == 0 else cs[s.axis_a] - 1
    qa = 1 if s.side_a == 0 else cs[s.axis_a] - 2
    taxes = [k for k in range(d) if k != s.axis_a]
    row = np.zeros(topo.R.shape[0])
    for normal_i, coef in ((fa, s.mu_a), (qa, -s.mu_a)):
        idx = [0] * d
        idx[s.axis_a] = normal_i
        for m, k in enumerate(taxes):
            idx[k] = tang[m]
        row[topo.block_offsets[s.a] + int(np.ravel_multi_index(idx, cs))] = coef
    return row


def _ortho_frame(
    topo: ControlTopology, specs: list[SeamOrthoSpec], h_mins: dict
) -> tuple[list, list]:
    """Frozen seam-orthogonality frame for one outer iteration.

    Penalty entries are interleaved reduced-space rows ``(rowv, w_eff)`` built
    from the T-extraction row ``mu_A (S - Q_A)`` through ``R`` — valid at
    every face-lattice index, including crossing-adjacent ones where T is a
    combination of the crossing's tangent and twist DOFs. Hard entries
    ``(t_root, sign, n_hat, h)`` require T to be a plain root (interior seam
    indices; near crossings the corner window owns the direction anyway) and
    snap the current T onto ``h n_hat``, mirroring the wall leg snap.
    """
    d = topo.d
    C = topo.stacked_C()
    pen, hard = [], []
    for si, (s, spec) in enumerate(zip(topo.seams, specs)):
        if spec.ortho == "off":
            continue
        if spec.ortho not in ("penalty", "hard"):
            raise ValueError(f"unknown seam ortho mode {spec.ortho!r}")
        taxes_a = [k for k in range(d) if k != s.axis_a]
        gpts = [
            greville(n, 3, knots=topo.cmaps[s.a].knots[k])
            for n, k in zip(s.tang_shape_a, taxes_a)
        ]
        w_arr = np.broadcast_to(np.asarray(spec.weight, dtype=float), s.tang_shape_a)
        for tang in product(*(range(n) for n in s.tang_shape_a)):
            if not s.elim[tang]:
                continue
            if spec.ortho == "penalty" and w_arr[tang] == 0.0:
                continue
            t_pt = [gpts[m][tang[m]] for m in range(d - 1)]
            # Seam tangent directions at the sample (frozen frame).
            tangents = []
            for m in range(d - 1):
                orders = [0] * (d - 1) + [0]
                orders[m] = 1
                t = _spline_face_field(
                    topo,
                    s.a,
                    s.axis_a,
                    s.side_a,
                    np.asarray([t_pt]),
                    orders,
                )[0]
                tangents.append(t / np.linalg.norm(t))
            row_c = _t_row(topo, s, tang)
            T = row_c @ C  # current tangent-row value (d,)
            if spec.ortho == "penalty":
                l2 = float(T @ T)
                if l2 <= 0.0:
                    continue
                row_q = topo.R.T @ row_c  # (n_roots,)
                for t_hat in tangents:
                    pen.append((np.kron(row_q, t_hat), float(w_arr[tang]) / l2))
            else:
                key = (si, tang)
                if key not in topo.t_root:
                    continue  # crossing-adjacent: not a plain root
                col, sgn = topo.t_root[key]
                if not topo.root_free[col]:
                    continue
                n_hat = _seam_normal(tangents, d)
                if float(n_hat @ T) < 0.0:
                    n_hat = -n_hat
                h = max(float(n_hat @ T), h_mins.get(key, 0.0))
                topo.q[col] = sgn * h * n_hat
                hard.append((col, sgn, n_hat, h))
    return pen, hard


def _seam_normal(tangents: list[np.ndarray], d: int) -> np.ndarray:
    if d == 2:
        t = tangents[0]
        return np.array([-t[1], t[0]])
    if d == 3:
        n = np.cross(tangents[0], tangents[1])
        return n / np.linalg.norm(n)
    raise NotImplementedError(f"seam normal for d={d}")


# --------------------------------------------------------------------------- #
# Reference runner: reduced Gauss-Newton over the global reduced controls
# --------------------------------------------------------------------------- #


def _owner_M(topo: ControlTopology) -> np.ndarray:
    """Dense stacked map: global fine nodes x stacked controls, one owner block
    per fine node (the last writer, matching ``prolong_global``)."""
    grid = topo.grid
    Mg = np.zeros((grid.global_node_count, topo.R.shape[0]))
    for bi, cmap in enumerate(topo.cmaps):
        Md = dense_M(cmap)
        rows = grid.block_dof_maps[bi].reshape(-1)
        Mg[rows, :] = 0.0
        lo = topo.block_offsets[bi]
        Mg[rows, lo : lo + Md.shape[1]] = Md
    return Mg


def run_control_topo_ref(
    grid,
    target_fn,
    ctrl_shapes: list[tuple[int, ...]] | None = None,
    *,
    metric: str = "shape_2d",
    max_outer: int = 30,
    grad_tol: float = 1e-8,
    alpha_min: float = 1e-4,
    lm0: float = 0.0,
    c1: bool = True,
    c2_weight: float = 0.0,
    ortho: str | list[SeamOrthoSpec] = "off",
    ortho_weight: float | np.ndarray = 1.0,
    fan_c1_weight: float = 1.0,
    smooth_weight: float = 0.0,
    topo: ControlTopology | None = None,
) -> tuple[ControlTopology, dict]:
    """Damped reduced Gauss-Newton over the multi-block reduced controls.

    Same loop shape as the single-block reference: per-iteration frozen ortho
    frame, one global GN system over the free reduced DOFs (penalties composed
    into the system AND the line-search energy), Levenberg ladder, and the
    codebase accept rule on the fine grid. Returns ``(topology, report)``.
    """
    from egg.smoothing.batch import energy_and_mindet, make_chain_J

    d = grid.topology.d
    if d != 2:
        raise NotImplementedError(
            "the NumPy metric layer (egg.smoothing.batch) is 2D; the device "
            "path carries D=3"
        )
    if topo is None:
        topo = build_control_topology(grid, ctrl_shapes, c1=c1)
    if topo.root_entity:
        raise NotImplementedError(
            "sliding wall roots are not handled by this runner yet; the "
            "tangential-frame handling is pending"
        )
    ctx = build_sweep_context(grid, target_fn)
    st = ctx.energy_stencil
    J = make_chain_J(st["s0"].astype(float), st["s1"].astype(float), st["W_inv"])
    Mg = _owner_M(topo)
    shim = SimpleNamespace(ctx=ctx, Mf=Mg, n_free=Mg.shape[1], metric=metric, _J=J)

    if isinstance(ortho, str):
        specs = [SeamOrthoSpec(ortho=ortho, weight=ortho_weight) for _ in topo.seams]
    else:
        specs = list(ortho)
        if len(specs) != len(topo.seams):
            raise ValueError("one SeamOrthoSpec per seam required")
    any_ortho = any(sp.ortho != "off" for sp in specs)
    hard_any = any(sp.ortho == "hard" for sp in specs)

    # Static penalty rows over the reduced DOFs (homogeneous, frozen linear).
    P_static = []
    W_static = []
    if c2_weight > 0.0:
        rows = _c2_rows(topo)
        P_static.append(rows)
        W_static.append(np.full(rows.shape[0], float(c2_weight)))
    fan_rows = _fan_c1_rows(topo)
    if fan_rows.shape[0] and fan_c1_weight > 0.0:
        P_static.append(fan_rows)
        W_static.append(np.full(fan_rows.shape[0], float(fan_c1_weight)))
    P_static = np.vstack(P_static) if P_static else np.zeros((0, topo.n_roots))
    W_static = np.concatenate(W_static) if W_static else np.zeros(0)

    n = topo.n_roots
    from scipy import sparse as _sp

    Rd = topo.R.toarray() if _sp.issparse(topo.R) else topo.R
    R_comp = np.kron(Rd, np.eye(d))
    free_int = np.repeat(topo.root_free, d)

    # Hard-mode h floor per tangent row, from the initial |T|.
    h_mins = {
        key: 1e-3 * float(np.linalg.norm(topo.q[col]))
        for key, (col, _s) in topo.t_root.items()
    }

    def fine_energy_mindet(Xg):
        return energy_and_mindet(
            Xg,
            st["gc"],
            st["gn0"],
            st["gn1"],
            st["s0"],
            st["s1"],
            st["W_inv"],
            metric=metric,
        )

    def pen_energy(q, pen):
        e = 0.0
        if P_static.shape[0]:
            r = P_static @ q  # (rows, d)
            e += 0.5 * float(np.sum(W_static[:, None] * r * r))
        qflat = q.reshape(-1)
        for rowv, w_eff in pen:
            v = float(rowv @ qflat)
            e += 0.5 * w_eff * v * v
        return e

    # Straightness rows: frame frozen at the initial state (they vanish on
    # straight lines, so a stale frame only weakens the restoring force).
    pen_smooth = (
        [(rowv, float(smooth_weight)) for rowv in _smooth_rows(topo)]
        if smooth_weight > 0.0
        else []
    )

    lam = float(lm0)
    energies, mindets, alphas = [], [], []
    converged = False
    iters = 0

    for _ in range(max_outer):
        pen, hard = _ortho_frame(topo, specs, h_mins) if any_ortho else ([], [])
        pen = pen + pen_smooth
        Xg = topo.prolong_global()
        e_fine, md0 = fine_energy_mindet(Xg)
        e0 = e_fine + pen_energy(topo.q, pen)
        if not energies:
            energies.append(e0)
            mindets.append(md0)

        G_full, A_full = reduced_system(Xg, shim)  # over stacked controls
        Gq = R_comp.T @ G_full.reshape(-1)
        Aq = R_comp.T @ A_full @ R_comp
        # Penalty contributions in reduced space (interleaved layout).
        if P_static.shape[0]:
            r = P_static @ topo.q
            for c in range(d):
                Gc = P_static.T @ (W_static * r[:, c])
                Gq[c::d] += Gc
            Hs = P_static.T @ (W_static[:, None] * P_static)
            for c in range(d):
                Aq[c::d, c::d] += Hs
        qflat = topo.q.reshape(-1)
        for rowv, w_eff in pen:
            v = float(rowv @ qflat)
            Gq += w_eff * v * rowv
            Aq += w_eff * np.outer(rowv, rowv)

        # Hard seam rows: reduce T columns to a scalar h along the frozen n.
        if hard_any and hard:
            hard_cols = {col for col, _s, _n, _h in hard}
            keep = [
                i for i in range(n * d) if free_int[i] and (i // d) not in hard_cols
            ]
            R2 = np.zeros((n * d, len(keep) + len(hard)))
            for cnew, i in enumerate(keep):
                R2[i, cnew] = 1.0
            hcol0 = len(keep)
            hmeta = []
            for hi, (col, sgn, n_hat, h) in enumerate(hard):
                R2[d * col : d * col + d, hcol0 + hi] = sgn * n_hat
                hmeta.append((col, sgn, n_hat))
            Gp = R2.T @ Gq
            Ap = R2.T @ Aq @ R2
        else:
            free_cols = np.flatnonzero(free_int)
            R2 = np.zeros((n * d, free_cols.size))
            R2[free_cols, np.arange(free_cols.size)] = 1.0
            Gp = Gq[free_cols]
            Ap = Aq[np.ix_(free_cols, free_cols)]

        if np.abs(Gp).max() < grad_tol:
            converged = True
            break

        accepted = False
        npq = Gp.shape[0]
        for _try in range(8):  # Levenberg ladder
            Areg = Ap + lam * np.eye(npq) if lam > 0.0 else Ap
            try:
                dp = np.linalg.solve(Areg, -Gp)
            except np.linalg.LinAlgError:
                lam = max(10.0 * lam, 1e-8 * float(np.trace(Ap)) / npq)
                continue
            dq = (R2 @ dp).reshape(n, d)
            corr = Mg @ (topo.R @ dq)
            alpha = 1.0
            while alpha >= alpha_min:
                e_f, md = fine_energy_mindet(Xg + alpha * corr)
                e = e_f + pen_energy(topo.q + alpha * dq, pen)
                if np.isfinite(e) and e <= e0 + ENERGY_TOL and md > 0.0:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                topo.q = topo.q + alpha * dq
                energies.append(e)
                mindets.append(md)
                alphas.append(alpha)
                iters += 1
                lam = lam / 3.0 if alpha == 1.0 else lam
                break
            lam = max(10.0 * lam, 1e-8 * float(np.trace(Ap)) / npq)
        if not accepted:
            break

    topo.write_to_grid()
    Xg = topo.prolong_global()
    e_f, md = fine_energy_mindet(Xg)
    return topo, {
        "energies": energies,
        "mindets": mindets,
        "alphas": alphas,
        "final_fine_energy": e_f,
        "final_mindet": md,
        "iters": iters,
        "converged": converged,
    }
