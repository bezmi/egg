# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Control-net wire for the device reduced-GN smoother.

Packs a :class:`~egg.geometry.control_net.ControlNetMap` (plus the fitted net,
free-control mask, and Boolean-sum offset) into the ``control`` dict the C++
session consumes. The dense per-axis basis matrices are banded by construction
(clamped B-splines: at most ``degree + 1`` consecutive nonzeros per row), so
each axis ships as a span offset + ``k_supp`` weights per node; the kernels
form the tensor weights in registers — no global CSR ``M`` is ever built.
"""

from __future__ import annotations

import numpy as np

from egg.geometry.control_net import ControlNetMap

__all__ = ["build_control_wire", "build_control_topo_wire", "run_control_topo"]


def _band_axis(B: np.ndarray, k_supp: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (offsets, weights) of a banded basis matrix ``(n_node, n_ctrl)``.

    Each row's nonzeros lie in a ``k_supp``-wide window; the window is clamped
    to the lattice so the C++ bounds check ``off + k_supp <= n_ctrl`` holds.
    """
    n_node, n_ctrl = B.shape
    if n_ctrl < k_supp:
        raise ValueError(
            f"axis has {n_ctrl} controls < support width {k_supp}; "
            "the net is too coarse for its degree"
        )
    off = np.empty(n_node, dtype=np.int32)
    wt = np.zeros((n_node, k_supp), dtype=np.float64)
    for i in range(n_node):
        nz = np.flatnonzero(B[i] != 0.0)
        first = int(nz[0]) if nz.size else 0
        first = min(first, n_ctrl - k_supp)
        width = (int(nz[-1]) - first + 1) if nz.size else 0
        if width > k_supp:
            raise ValueError(
                f"basis row {i} has support width {width} > k_supp {k_supp}"
            )
        off[i] = first
        wt[i] = B[i, first : first + k_supp]
    return off, wt.reshape(-1)


def build_control_wire(
    cmap: ControlNetMap,
    C0: np.ndarray,
    free_ctrl: np.ndarray,
    b_field: np.ndarray,
    *,
    block: int = 0,
    walls: list | None = None,
    X0: np.ndarray | None = None,
) -> dict:
    """Pack one block's control net into the C++ session's ``control`` dict.

    Parameters
    ----------
    cmap : ControlNetMap
        The block's tensor-product map (per-axis banded bases).
    C0 : ndarray, shape ``ctrl_shape + (d,)``
        Initial control net (e.g. from
        :func:`egg.smoothing.control_ref.fit_control_net`). With walls, the
        wall boundary controls should already be projected onto their entity
        (as :func:`egg.smoothing.control_wall.build_control_wall_ref` does).
    free_ctrl : ndarray of bool, shape ``ctrl_shape``
        Free-control mask (fixed controls never move). With walls this must
        already include the sliding wall controls.
    b_field : ndarray, shape ``node_shape + (d,)``
        Boolean-sum boundary offset (:func:`egg.smoothing.control_ref.
        boolean_sum_b`); with walls the initial re-extension against the
        projected spline boundary.
    block : int
        Structured block index the net drives.
    walls : list of egg.smoothing.control_wall.WallSpec, optional
        Wall faces (sliding controls + orthogonality dial). Wall entities
        must be fixed-size analytic types (line/arc/Bezier — the host-side
        projection path); segmented entities (B-spline, composite) are not
        wired yet.
    X0 : ndarray, shape ``node_shape + (d,)``
        Initial exact node field; required with walls (its non-wall boundary
        rows anchor the Boolean-sum re-extension).
    """
    d = cmap.d
    k_supp = max(cmap.degrees) + 1
    if C0.shape != cmap.ctrl_shape + (d,):
        raise ValueError(f"C0 shape {C0.shape} != {cmap.ctrl_shape + (d,)}")
    if free_ctrl.shape != cmap.ctrl_shape:
        raise ValueError(f"free_ctrl shape {free_ctrl.shape} != {cmap.ctrl_shape}")
    if b_field.shape != cmap.node_shape + (d,):
        raise ValueError(f"b shape {b_field.shape} != {cmap.node_shape + (d,)}")

    wire: dict = {
        "block": int(block),
        "ctrl_shape": np.asarray(cmap.ctrl_shape, dtype=np.int32),
        "k_supp": int(k_supp),
        "C0": np.ascontiguousarray(C0, dtype=np.float64).reshape(-1),
        "free_idx": np.flatnonzero(free_ctrl.reshape(-1)).astype(np.int32),
        "b": np.ascontiguousarray(b_field, dtype=np.float64).reshape(-1),
    }
    for k in range(d):
        off, wt = _band_axis(np.asarray(cmap.basis[k], dtype=np.float64), k_supp)
        wire[f"off{k}"] = off
        wire[f"wt{k}"] = wt

    if walls:
        from egg.geometry.entity_soa import encode_entity_soa
        from egg.smoothing.control_wall import _wall_ctrl_rows

        if X0 is None:
            raise ValueError("walls require X0 (the initial exact node field)")
        if X0.shape != cmap.node_shape + (d,):
            raise ValueError(f"X0 shape {X0.shape} != {cmap.node_shape + (d,)}")
        wire["X0"] = np.ascontiguousarray(X0, dtype=np.float64).reshape(-1)
        wall_dicts = []
        for spec in walls:
            tag, _, records, seg = encode_entity_soa(spec.entity, d=d)
            if seg is not None:
                raise NotImplementedError(
                    "wall entities with segmented payloads (B-spline/composite) "
                    "are not wired to the single-block device wall mode (its "
                    "frame rebuild runs in C++ on fixed-size analytic records); "
                    "use run_control_topo (the multi-block driver — a single "
                    "block is fine), whose host-side frames handle any entity "
                    "with project/tangent_space"
                )
            p0, p1 = _wall_ctrl_rows(spec.axis, spec.side, cmap.ctrl_shape)
            w = np.broadcast_to(
                np.asarray(spec.weight, dtype=np.float64), p0.shape
            ).copy()
            mode = {"off": 0, "penalty": 1, "hard": 2}[spec.ortho]
            wall_dicts.append(
                {
                    "axis": int(spec.axis),
                    "side": int(spec.side),
                    "mode": mode,
                    "weight": w,
                    "p0": p0.astype(np.int32),
                    "p1": p1.astype(np.int32),
                    "tag": int(tag),
                    "params": np.asarray(records, dtype=np.float64),
                }
            )
        wire["walls"] = wall_dicts
    return wire


# --------------------------------------------------------------------------- #
# Multi-block wire (ControlTopology -> device session) + driver
# --------------------------------------------------------------------------- #


def _reduction_csr(
    R: np.ndarray,
    root_free: np.ndarray,
    d: int,
    hard: dict | None = None,
    slide: dict | None = None,
    record_slide: bool = False,
):
    """Component-space reduction CSR from the scalar reduction matrix.

    Rows are stacked control components (control-major, coordinates
    innermost); columns cover the free roots only (fixed roots contribute no
    motion, matching the reference's free-column restriction). ``hard`` maps
    a root index to a unit vector n̂ (with the union-find sign already
    folded): that root's d columns collapse to ONE scalar column with
    per-component coefficients n̂ — the seam-ortho hard parameterization
    T = h·n̂. ``slide`` maps a root index to a frozen tangent basis
    ``(d-1, d)``: that root gets d-1 tangential columns (sliding wall
    controls; such roots are not in ``root_free`` — the frame is their
    freedom).

    Returns ``(nq, off, col, coef)``.
    """
    n_stack, n_roots = R.shape
    hard = hard or {}
    slide = slide or {}
    col0 = np.full(n_roots, -1, dtype=np.int64)
    nq = 0
    for r in range(n_roots):
        if root_free[r]:
            col0[r] = nq
            nq += 1 if r in hard else d
        elif r in slide:
            col0[r] = nq
            nq += d - 1
    off = np.zeros(n_stack * d + 1, dtype=np.int32)
    cols: list[int] = []
    coefs: list[float] = []
    entries: list[list[tuple[int, float]]] = [[] for _ in range(n_stack)]
    from scipy import sparse as _sp

    if _sp.issparse(R):
        Rc = R.tocsr()
        for i in range(n_stack):
            for e in range(Rc.indptr[i], Rc.indptr[i + 1]):
                j = int(Rc.indices[e])
                if col0[j] >= 0:
                    entries[i].append((j, float(Rc.data[e])))
    else:
        nz_rows, nz_cols = np.nonzero(R)
        for i, j in zip(nz_rows, nz_cols):
            if col0[j] >= 0:
                entries[i].append((int(j), float(R[i, j])))
    slide_entries: dict[int, list[tuple[int, int, int, float]]] = {j: [] for j in slide}
    for i in range(n_stack):
        for c in range(d):
            for j, v in entries[i]:
                if j in hard and root_free[j]:
                    cols.append(int(col0[j]))
                    coefs.append(v * float(hard[j][c]))
                elif j in slide and not root_free[j]:
                    t = slide[j]
                    for kt in range(d - 1):
                        cols.append(int(col0[j]) + kt)
                        coefs.append(v * float(t[kt][c]))
                        if record_slide:
                            # (CSR entry index, component, tangent slot, the
                            # frame-invariant scalar coefficient): the C++
                            # frame loop rewrites coef[entry] = base * t[kt][c].
                            slide_entries[j].append((len(cols) - 1, c, kt, float(v)))
                else:
                    cols.append(int(col0[j]) + c)
                    coefs.append(v)
            off[i * d + c + 1] = len(cols)
    out = (
        nq,
        off,
        np.asarray(cols, dtype=np.int32),
        np.asarray(coefs, dtype=np.float64),
    )
    if record_slide:
        return out + (slide_entries,)
    return out


def _penalty_csr(scalar_rows: np.ndarray, weights: np.ndarray, d: int, extra=None):
    """Component penalty CSR from scalar stacked-control rows (each expanded
    to d component rows) plus optional pre-built component rows ``extra``:
    a list of ``(comp_entries, weight)`` with ``comp_entries`` a list of
    ``(component_index, coefficient)``."""
    off = [0]
    cols: list[int] = []
    coefs: list[float] = []
    w: list[float] = []
    for j in range(scalar_rows.shape[0]):
        nz = np.flatnonzero(scalar_rows[j])
        for c in range(d):
            for i in nz:
                cols.append(int(i) * d + c)
                coefs.append(float(scalar_rows[j, i]))
            off.append(len(cols))
            w.append(float(weights[j]))
    for comp_entries, w_eff in extra or []:
        for idx, v in comp_entries:
            cols.append(int(idx))
            coefs.append(float(v))
        off.append(len(cols))
        w.append(float(w_eff))
    return (
        np.asarray(off, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(coefs, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
    )


def _append_penalty_rows(csr, rows, w_val: float):
    """Append flat CSR component rows ``(row_off, cols, coefs)`` (the
    vectorized :func:`~egg.smoothing.control_topology._smooth_rows` form) to
    a ``_penalty_csr`` result at a uniform weight."""
    p_off, p_col, p_coef, p_w = csr
    row_off, cols, coefs = rows
    n_new = row_off.size - 1
    if n_new == 0:
        return csr
    return (
        np.concatenate([p_off, p_off[-1] + row_off[1:]]).astype(np.int32),
        np.concatenate([p_col, cols]).astype(np.int32),
        np.concatenate([p_coef, coefs]),
        np.concatenate([p_w, np.full(n_new, float(w_val))]),
    )


def build_control_topo_wire(
    topo,
    *,
    c2_weight: float = 0.0,
    fan_c1_weight: float = 1.0,
    smooth_weight: float = 0.0,
) -> dict:
    """Pack a :class:`~egg.smoothing.control_topology.ControlTopology` into
    the multi-block ``control`` dict the C++ session consumes: per-block
    banded nets over one stacked control vector, the reduction CSR
    ``dC = R·dq`` restricted to free roots, and the static penalty rows
    (soft C2 + fan C1). Frame-dependent penalties (seam orthogonality) are
    supplied per outer iteration by :func:`run_control_topo`.
    """
    from egg.smoothing.control_topology import _c2_rows, _fan_c1_rows, _smooth_rows

    d = topo.d
    blocks = []
    for bi, cmap in enumerate(topo.cmaps):
        k_supp = max(cmap.degrees) + 1
        bd = {
            "block": int(bi),
            "ctrl_shape": np.asarray(topo.ctrl_shapes[bi], dtype=np.int32),
            "k_supp": int(k_supp),
            "b": np.ascontiguousarray(topo.b_fields[bi], dtype=np.float64).reshape(-1),
        }
        for k in range(d):
            offk, wtk = _band_axis(np.asarray(cmap.basis[k], dtype=np.float64), k_supp)
            bd[f"off{k}"] = offk
            bd[f"wt{k}"] = wtk
        blocks.append(bd)

    nq, r_off, r_col, r_coef = _reduction_csr(topo.R, topo.root_free, d)
    wire: dict = {
        "blocks": blocks,
        "C0": np.ascontiguousarray(topo.stacked_C(), dtype=np.float64).reshape(-1),
        "nq": int(nq),
        "r_off": r_off,
        "r_col": r_col,
        "r_coef": r_coef,
    }

    scalar_rows = []
    weights = []
    if c2_weight > 0.0:
        rows = _c2_rows(topo, reduced=False)
        if rows.shape[0]:
            scalar_rows.append(rows)
            weights.append(np.full(rows.shape[0], float(c2_weight)))
    if fan_c1_weight > 0.0:
        rows = _fan_c1_rows(topo, reduced=False)
        if rows.shape[0]:
            scalar_rows.append(rows)
            weights.append(np.full(rows.shape[0], float(fan_c1_weight)))
    smooth = _smooth_rows(topo, reduced=False) if smooth_weight > 0.0 else None
    if scalar_rows or (smooth is not None and smooth[0].size > 1):
        P = np.vstack(scalar_rows) if scalar_rows else np.zeros((0, topo.R.shape[0]))
        W = np.concatenate(weights) if weights else np.zeros(0)
        csr = _penalty_csr(P, W, d)
        if smooth is not None:
            csr = _append_penalty_rows(csr, smooth, float(smooth_weight))
        wire["p_off"], wire["p_col"], wire["p_coef"], wire["p_w"] = csr
    return wire


def _encode_frame_entity(ent, d: int, arena: list):
    """Device-layout (tag, record) for the C++ frame loop's projections."""
    if d == 2:
        from egg.geometry.entity_encoding import encode_entity

        tag, params = encode_entity(ent, d=2, arena=arena)
        return int(tag), np.asarray(params, dtype=np.float64)
    from egg.geometry.entity_soa import TAG_BSPLINESURF, encode_entity_soa

    tag, _, records, seg = encode_entity_soa(ent, d=3)
    if seg is None:
        return int(tag), np.asarray(records, dtype=np.float64)
    if tag != TAG_BSPLINESURF:
        raise NotImplementedError(
            "3D segmented wall entities other than B-spline/NURBS surfaces "
            "are not wired to the C++ frame loop"
        )
    # NURBS surface: park the variable-length payloads in the shared arena
    # and rewrite the record's offset fields (the SoA CSR encoder leaves them
    # implicit; the frame loop's decode_entity reads explicit arena offsets).
    # The blob grows trim fields at 9-11 (verts_off, loops_off, n_entries) —
    # zero means untrimmed, matching zero-padded older blobs.
    knots_u, knots_v, ctrl, weights, trim_verts, trim_loops = seg

    def put(arr) -> float:
        off = len(arena)
        arena.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())
        return float(off)

    rec = np.zeros(12, dtype=np.float64)
    rec[: len(records)] = np.asarray(records, dtype=np.float64)
    rec[4] = put(knots_u)
    rec[5] = put(knots_v)
    rec[6] = put(ctrl)
    rec[7] = put(weights) if np.asarray(weights).size else 0.0
    if np.asarray(trim_loops).size >= 2:
        rec[9] = put(trim_verts)
        rec[10] = put(trim_loops)
        rec[11] = float(np.asarray(trim_loops).size)
    return int(tag), rec


def _initial_slide_frames(topo, d: int) -> dict:
    """Project sliding roots onto their entities and return the initial
    tangent frames (the C++ loop refreshes them per iteration). Mutates
    ``topo.q`` (the projection) like the driver's frame does."""
    slide0: dict[int, np.ndarray] = {}
    by_ent: dict[int, list[int]] = {}
    ent_of: dict[int, object] = {}
    for col, ent in topo.root_entity.items():
        by_ent.setdefault(id(ent), []).append(col)
        ent_of[id(ent)] = ent
    for eid, cols in by_ent.items():
        ent = ent_of[eid]
        P = np.asarray([topo.q[c] for c in cols])
        if hasattr(ent, "project_many") and hasattr(ent, "tangent_space_many"):
            feet = np.asarray(ent.project_many(P))
            tans = np.asarray(ent.tangent_space_many(feet))
        else:
            feet = np.array([ent.project(p) for p in P])
            tans = np.array([ent.tangent_space(p) for p in feet])
        for i, col in enumerate(cols):
            topo.q[col] = feet[i]
            t = tans[i].reshape(d, d - 1).T
            n = np.linalg.norm(t, axis=1, keepdims=True)
            t = np.where(n < 1e-12, 0.0, t / np.maximum(n, 1e-300))
            slide0[col] = t
    return slide0


def _attach_slide_wire(topo, wire: dict) -> None:
    """Attach the in-session sliding-frame wire: the reduction gets the
    tangential slide columns with CSR value-rewrite lists, entity records
    (+ shared arena), CAD wall faces, per-net seam flags, and per-block X0
    anchors — everything the C++ frame loop needs to run without Python."""
    d = topo.d
    grid = topo.grid
    slide0 = _initial_slide_frames(topo, d)
    nq, r_off, r_col, r_coef, entries = _reduction_csr(
        topo.R, topo.root_free, d, slide=slide0, record_slide=True
    )
    wire["nq"] = int(nq)
    wire["r_off"] = r_off
    wire["r_col"] = r_col
    wire["r_coef"] = r_coef
    # q was projected onto the entities above; ship the projected state.
    wire["C0"] = np.ascontiguousarray(topo.stacked_C(), dtype=np.float64).reshape(-1)

    arena: list[float] = []
    enc_cache: dict[int, tuple[int, np.ndarray]] = {}

    def enc(ent):
        if id(ent) not in enc_cache:
            enc_cache[id(ent)] = _encode_frame_entity(ent, d, arena)
        return enc_cache[id(ent)]

    from scipy import sparse as _sp

    if _sp.issparse(topo.R):
        Rr = topo.R.tocsr()
        Rc = topo.R.tocsc()
        row_nnz = np.diff(Rr.indptr)

        def column(col):
            lo, hi = Rc.indptr[col], Rc.indptr[col + 1]
            return Rc.indices[lo:hi], Rc.data[lo:hi]
    else:
        row_nnz = (topo.R != 0).sum(axis=1)

        def column(col):
            rows = np.flatnonzero(topo.R[:, col])
            return rows, topo.R[rows, col]

    slide_list = []
    for col, ent in topo.root_entity.items():
        rows, coefs = column(col)
        rep = [
            int(r)
            for r, v in zip(rows, coefs)
            if row_nnz[r] == 1 and abs(v - 1.0) < 1e-12
        ]
        if not rep:
            raise ValueError("sliding root without a pure representative control row")
        tag, params = enc(ent)
        e = entries[col]
        slide_list.append(
            {
                "rep_row": rep[0],
                "members": rows.astype(np.int32),
                "member_coef": np.asarray(coefs, dtype=np.float64),
                "tag": tag,
                "params": params,
                "r_entry": np.asarray([x[0] for x in e], dtype=np.int32),
                "comp": np.asarray([x[1] for x in e], dtype=np.int32),
                "kt": np.asarray([x[2] for x in e], dtype=np.int32),
                "base": np.asarray([x[3] for x in e], dtype=np.float64),
            }
        )
    wire["slide"] = slide_list

    wall_list = []
    for (bi, axis, side), ent in topo.wall_faces.items():
        tag, params = enc(ent)
        wall_list.append(
            {
                "block": int(bi),
                "axis": int(axis),
                "side": int(side),
                "tag": tag,
                "params": params,
            }
        )
    wire["wall_faces"] = wall_list

    seam_faces = {(s.a, s.axis_a, s.side_a) for s in topo.seams} | {
        (s.b, s.axis_b, s.side_b) for s in topo.seams
    }
    seam_flags = []
    for bi in range(len(grid.blocks)):
        fl = np.zeros(2 * d, dtype=np.int32)
        for k in range(d):
            for side in (0, 1):
                if (bi, k, side) in seam_faces:
                    fl[2 * k + side] = 1
        seam_flags.append(fl)
    wire["seam_face"] = seam_flags

    for bi, bd in enumerate(wire["blocks"]):
        bd["X0"] = np.ascontiguousarray(
            grid.blocks[bi].nodes, dtype=np.float64
        ).reshape(-1)
    if arena:
        wire["arena"] = np.asarray(arena, dtype=np.float64)


def _ortho_stacked_frame(topo, specs, h_mins):
    """Frozen seam-orthogonality frame in stacked-control terms.

    Mirrors ``control_topology._ortho_frame`` (same tangents, weights, and
    hard snap — parity source) but emits penalty rows over stacked control
    COMPONENTS (``kron(row_c, t̂)`` without pushing through R: identical
    residuals since C = R q) and hard entries as ``root -> signed n̂`` for the
    reduction rebuild. Hard mode mutates ``topo.q`` (the T snap), exactly as
    the reference does.
    """
    from itertools import product

    from egg.geometry.control_net import greville
    from egg.smoothing.control_topology import (
        _seam_normal,
        _spline_face_field,
        _t_row,
    )

    d = topo.d
    C = topo.stacked_C()
    pen, hard = [], {}
    for si, (s, spec) in enumerate(zip(topo.seams, specs)):
        if spec.ortho == "off":
            continue
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
            tangents = []
            for m in range(d - 1):
                orders = [0] * (d - 1) + [0]
                orders[m] = 1
                t = _spline_face_field(
                    topo, s.a, s.axis_a, s.side_a, np.asarray([t_pt]), orders
                )[0]
                tangents.append(t / np.linalg.norm(t))
            row_c = _t_row(topo, s, tang)
            T = row_c @ C
            if spec.ortho == "penalty":
                l2 = float(T @ T)
                if l2 <= 0.0:
                    continue
                nz = np.flatnonzero(row_c)
                for t_hat in tangents:
                    entries = [
                        (int(i) * d + c, float(row_c[i]) * float(t_hat[c]))
                        for i in nz
                        for c in range(d)
                    ]
                    pen.append((entries, float(w_arr[tang]) / l2))
            else:
                key = (si, tang)
                if key not in topo.t_root:
                    continue
                col, sgn = topo.t_root[key]
                if not topo.root_free[col]:
                    continue
                n_hat = _seam_normal(tangents, d)
                if float(n_hat @ T) < 0.0:
                    n_hat = -n_hat
                h = max(float(n_hat @ T), h_mins.get(key, 0.0))
                topo.q[col] = sgn * h * n_hat
                hard[col] = sgn * n_hat
    return pen, hard


def run_control_topo(
    grid,
    target_fn,
    ctrl_shapes=None,
    *,
    device: str = "cpu",
    max_outer: int = 30,
    grad_tol: float = 1e-8,
    alpha_min: float = 1e-4,
    lm0: float = 0.0,
    c1: bool = True,
    c2_weight: float = 0.0,
    ortho="off",
    ortho_weight=1.0,
    fan_c1_weight: float = 1.0,
    smooth_weight: float = 0.0,
    topo=None,
    pcg_max_iter: int = 200,
    pcg_rtol: float = 1e-10,
    pcg_forcing: bool = True,
    model_db: bool = True,
    phase: str = "barrier",
    session=None,
):
    """Device counterpart of
    :func:`egg.smoothing.control_topology.run_control_topo_ref` (D-general —
    the reference is restricted to the 2D NumPy metric layer).

    Builds the topology + wire, runs the reduced GN loop on the device
    session, and writes the accepted state back into ``grid`` and ``topo``.
    Seam-orthogonality modes drive the session one outer iteration at a time
    with a per-iteration frozen-frame rebuild (penalty rows / hard reduction
    columns re-uploaded between iterations), mirroring the reference cadence.
    Returns ``(topo, report)``.

    Pass ``session=report["session"]`` from a previous call (same ``topo``)
    to continue on the resident state — the pipeline chunks the phase this
    way so live views can animate the control iterations.
    """
    from egg.smoothing.control_topology import (
        SeamOrthoSpec,
        build_control_topology,
    )
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )
    from egg.smoothing.solver import build_sweep_context

    d = grid.topology.d
    if topo is None:
        topo = build_control_topology(grid, ctrl_shapes, c1=c1)
    sliding = bool(topo.root_entity)

    if isinstance(ortho, str):
        specs = [SeamOrthoSpec(ortho=ortho, weight=ortho_weight) for _ in topo.seams]
    else:
        specs = list(ortho)
        if len(specs) != len(topo.seams):
            raise ValueError("one SeamOrthoSpec per seam required")
    any_ortho = any(sp.ortho != "off" for sp in specs)

    if session is not None:
        # Continue on the resident state (same topo; chunked driving).
        sess = session
        cxx_frames = sess._ctrl_cxx_frames
        static_pen = sess._ctrl_static_pen
        X0_blocks_saved = sess._ctrl_x0_blocks
    else:
        wire = build_control_topo_wire(
            topo,
            c2_weight=c2_weight,
            fan_c1_weight=fan_c1_weight,
            smooth_weight=smooth_weight,
        )
        # Sliding-only runs get the in-session C++ frame loop (zero Python
        # per iteration); the ortho modes keep the Python per-iteration
        # frames, and entities the C++ projection cannot decode yet (3D
        # segmented surfaces) fall back to the Python frames too.
        cxx_frames = sliding and not any_ortho
        if cxx_frames:
            try:
                _attach_slide_wire(topo, wire)
            except NotImplementedError:
                cxx_frames = False
        static_pen = (
            wire.get("p_off", np.zeros(1, dtype=np.int32)),
            wire.get("p_col", np.zeros(0, dtype=np.int32)),
            wire.get("p_coef", np.zeros(0)),
            wire.get("p_w", np.zeros(0)),
        )

        ctx = build_sweep_context(grid, target_fn)
        bsc = build_block_structured_context(grid)
        sess = CppStructuredSweepSession(
            ctx, bsc, np.asarray(grid.global_nodes), device=device, control=wire
        )
        sess._ctrl_cxx_frames = cxx_frames
        sess._ctrl_static_pen = static_pen
        sess._ctrl_x0_blocks = [
            np.asarray(blk.nodes, dtype=float).copy() for blk in grid.blocks
        ]
        X0_blocks_saved = sess._ctrl_x0_blocks
    run_kw = dict(
        phase=phase,
        grad_tol=grad_tol,
        alpha_min=alpha_min,
        lm0=lm0,
        pcg_max_iter=pcg_max_iter,
        pcg_rtol=pcg_rtol,
        pcg_forcing=pcg_forcing,
        model_db=model_db,
    )

    # q recovery from the device C is exact (C stays in range(R)); the
    # sparse normal-equations factorization is computed once.
    from egg.smoothing.control_topology import lstsq_q_solver

    q_solve = lstsq_q_solver(topo.R)

    def sync_q_from_device():
        C = np.asarray(sess.get_C()).reshape(-1, d)
        topo.q = q_solve(C)

    # b re-extension anchors: the nodes the net was fitted to (NOT the
    # current grid — chunked calls mutate it between runs).
    X0_blocks = X0_blocks_saved
    wall_blocks = sorted({bi for (bi, _k, _s) in topo.wall_faces})

    def slide_frame():
        """Project sliding roots onto their entities and freeze tangent
        frames; returns the ``slide`` dict for the reduction rebuild.

        Batched per entity where the entity supports it — per-root scalar
        projections (a seeded Newton per composite segment each) dominated
        the frame-rebuild profile otherwise."""
        slide = {}
        by_ent: dict[int, list[int]] = {}
        ent_of: dict[int, object] = {}
        for col, ent in topo.root_entity.items():
            by_ent.setdefault(id(ent), []).append(col)
            ent_of[id(ent)] = ent
        for eid, cols in by_ent.items():
            ent = ent_of[eid]
            P = np.asarray([topo.q[c] for c in cols])
            batched = hasattr(ent, "project_many") and hasattr(
                ent, "tangent_space_many"
            )
            if batched:
                feet = np.asarray(ent.project_many(P))
                tans = np.asarray(ent.tangent_space_many(feet))
            else:
                feet = np.array([ent.project(p) for p in P])
                tans = np.array([ent.tangent_space(p) for p in feet])
            for i, col in enumerate(cols):
                topo.q[col] = feet[i]
                # tangent_space returns the tangents as COLUMNS (d, d-1).
                t = tans[i].reshape(d, d - 1).T
                norms = np.linalg.norm(t, axis=1)
                if np.any(norms < 1e-12):
                    # Degenerate chart point (e.g. a sphere-parameterization
                    # pole): rebuild the frame as the orthonormal complement
                    # of the entity normal.
                    n = np.asarray(ent.normal(feet[i]), dtype=float)
                    t = np.linalg.svd(n.reshape(1, -1), full_matrices=True)[2][1:]
                else:
                    t = t / norms[:, None]
                slide[col] = t
        return slide

    def device_mindet() -> float:
        return float(sess.run_control(0, **run_kw)["mindets"][0])

    def apply_frame(slide, hard):
        """Push a frozen frame (reprojected state, reduction, b) to the
        session, gated on validity: a rebuild that folds the fine grid is
        reverted (returns False) — the accept rule can never recover from an
        invalid baseline, so the caller stops at the last valid state."""
        q_prev = topo.q.copy()
        b_prev = [topo.b_fields[bi].copy() for bi in wall_blocks]
        sess.set_C(np.ascontiguousarray(topo.stacked_C()).reshape(-1))
        nq, r_off, r_col, r_coef = _reduction_csr(
            topo.R, topo.root_free, d, hard=hard, slide=slide
        )
        sess.set_control_reduction(nq, r_off, r_col, r_coef)
        if slide:
            from egg.smoothing.control_topology import wall_b_field

            for bi in wall_blocks:
                topo.b_fields[bi] = wall_b_field(topo, bi, X0_blocks[bi])
                sess.set_control_b(bi, topo.b_fields[bi].reshape(-1))
        if device_mindet() > 0.0:
            return True
        topo.q = q_prev
        for bi, b in zip(wall_blocks, b_prev):
            topo.b_fields[bi] = b
        sess.set_C(np.ascontiguousarray(topo.stacked_C()).reshape(-1))
        for bi in wall_blocks:
            sess.set_control_b(bi, topo.b_fields[bi].reshape(-1))
        return False

    energies, mindets, alphas, frame_jumps = [], [], [], []
    converged = False
    iters = 0
    if not any_ortho:
        # Plain run, or sliding with the in-session C++ frame loop.
        rep = sess.run_control(max_outer, **run_kw)
        energies = list(rep["energies"])
        mindets = list(rep["mindets"])
        alphas = list(rep["alphas"])
        frame_jumps = list(rep.get("frame_jumps", []))
        iters = int(rep["iters"])
        converged = bool(rep["converged"])
        sync_q_from_device()
        if cxx_frames:
            # The session re-extended b internally; mirror the final frame's
            # b on the host state (equals the device b whenever the closing
            # frame applied — the normal case).
            from egg.smoothing.control_topology import wall_b_field

            for bi in wall_blocks:
                topo.b_fields[bi] = wall_b_field(topo, bi, X0_blocks[bi])
    else:
        h_mins = {
            key: 1e-3 * float(np.linalg.norm(topo.q[col]))
            for key, (col, _s) in topo.t_root.items()
        }
        for _ in range(max_outer):
            pen, hard = _ortho_stacked_frame(topo, specs, h_mins)
            slide = slide_frame() if sliding else {}
            if (hard or slide) and not apply_frame(slide, hard):
                break  # the frame rebuild would fold; stop at the last valid state
            if pen:
                from egg.smoothing.control_topology import (
                    _c2_rows,
                    _fan_c1_rows,
                    _smooth_rows,
                )

                scalar_rows = []
                weights = []
                if c2_weight > 0.0:
                    rows = _c2_rows(topo, reduced=False)
                    if rows.shape[0]:
                        scalar_rows.append(rows)
                        weights.append(np.full(rows.shape[0], float(c2_weight)))
                if fan_c1_weight > 0.0:
                    rows = _fan_c1_rows(topo, reduced=False)
                    if rows.shape[0]:
                        scalar_rows.append(rows)
                        weights.append(np.full(rows.shape[0], float(fan_c1_weight)))
                P = (
                    np.vstack(scalar_rows)
                    if scalar_rows
                    else np.zeros((0, topo.R.shape[0]))
                )
                W = np.concatenate(weights) if weights else np.zeros(0)
                csr = _penalty_csr(P, W, d, extra=pen)
                if smooth_weight > 0.0:
                    csr = _append_penalty_rows(
                        csr, _smooth_rows(topo, reduced=False), float(smooth_weight)
                    )
                sess.set_control_penalty(*csr)
            rep = sess.run_control(1, **run_kw)
            if not energies:
                energies.append(rep["energies"][0])
                mindets.append(rep["mindets"][0])
            if rep["iters"] > 0:
                energies.extend(rep["energies"][1:])
                mindets.extend(rep["mindets"][1:])
                alphas.extend(rep["alphas"])
                frame_jumps.extend(rep.get("frame_jumps", []))
                iters += int(rep["iters"])
            sync_q_from_device()
            if rep["converged"]:
                converged = True
                break
            if rep["iters"] == 0:
                break

    if sliding and not cxx_frames:
        # Final consistency pass so the reported state is exactly on the
        # frame (reprojected sliding controls, fresh b) — mirrors the
        # single-block wall reference's closing rebuild. Same validity gate
        # as the per-iteration frames: a closing rebuild that folds is
        # reverted (the last valid state wins over frame exactness). (The
        # C++ frame loop runs its own closing frame.)
        sync_q_from_device()
        apply_frame(slide_frame(), {})

    # Final fine-only energy/min-det (penalties cleared for one evaluation).
    sess.set_control_penalty(
        np.zeros(1, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
        np.zeros(0),
    )
    rep0 = sess.run_control(0, **run_kw)
    final_fine_energy = float(rep0["energies"][0])
    final_mindet = float(rep0["mindets"][0])
    sess.set_control_penalty(*static_pen)

    # Write the accepted state back to the grid (device fine field).
    X = np.asarray(sess.get_X()).reshape(-1, d)
    grid.global_nodes[:] = X
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]

    report = {
        "energies": energies,
        "mindets": mindets,
        "alphas": alphas,
        "frame_jumps": frame_jumps,
        "final_fine_energy": final_fine_energy,
        "final_mindet": final_mindet,
        "iters": iters,
        "converged": converged,
        "session": sess,
    }
    return topo, report
