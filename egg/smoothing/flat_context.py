"""Dimension-generic construction of the flattened sweep context.

A structured hex/quad grid's per-sweep context is a fixed combinatorial
pattern: every cell contributes ``2**d`` corner samples, each sample is a
corner plus its ``d`` axis neighbours (with per-axis signs), and each cell node
"sees" all of the cell's samples with a role that depends only on the two local
corner indices. :func:`cell_stencil` builds that pattern for any dimension with
numpy array ops (no per-cell Python loops); :func:`build_flat_context` assembles
it into the ragged ``{"groups", "energy_stencil"}`` wire format the C++ core
consumes.

Local corner index ``b`` in ``0..2**d - 1`` packs the per-axis offset bits with
axis 0 most significant: ``b = sum_k o_k << (d - 1 - k)`` — the C-order of
``itertools.product((0, 1), repeat=d)``. Sample id ``= 2**d * cell + b`` over
the concatenated blocks.
"""

import numpy as np

from egg.geometry.entity_soa import group_entities_by_type


def _axbit(d):
    """Corner-index bit mask for each axis (axis 0 is the most significant)."""
    return np.array([1 << (d - 1 - k) for k in range(d)], dtype=np.int64)


def _role_table(d):
    """``role[a, b]``: how local corner ``a`` relates to the sample at corner ``b``.

    ``0`` when ``a`` is the sample's own corner, ``1 + k`` when ``a`` is the
    sample's axis-``k`` neighbour (``a ^ b`` is the axis-``k`` bit), and ``-1``
    when ``a`` shares the cell but is neither.
    """
    axbit = _axbit(d)
    nc = 1 << d
    b = np.arange(nc)
    role = np.full((nc, nc), -1, dtype=np.int32)
    role[b, b] = 0
    for k in range(d):
        role[b, b ^ axbit[k]] = 1 + k
    return role


def cell_stencil(blocks, d):
    """Vectorized per-(cell, corner) sample stencil and node->sample membership.

    ``blocks`` is a list of integer arrays of global node ids, each of shape
    ``(n_0, ..., n_{d-1})``. Returns a dict with, in energy-stencil order
    (sample id ``= 2**d * cell + corner``):

    - ``gc``   ``(ns,)`` corner global id per sample;
    - ``gn``   list of ``d`` arrays ``(ns,)`` — axis-``k`` neighbour global id;
    - ``s``    list of ``d`` arrays ``(ns,)`` — axis-``k`` sign (``+1``/``-1``);

    and the flattened node->sample membership (one row per cell node × cell
    sample), restricted later by the caller:

    - ``m_node`` ``(nm,)`` node global id;
    - ``m_sid``  ``(nm,)`` sample id;
    - ``m_role`` ``(nm,)`` role of the node in that sample;

    plus scalar ``nc`` (corners per cell), ``ns`` (samples) and ``ncell``.
    """
    nc = 1 << d
    axbit = _axbit(d)
    b_idx = np.arange(nc)
    role = _role_table(d)

    # Ccell[c, a] = global node id of local corner a of cell c (all blocks). For
    # corner a the cell grid is ids[o_0:o_0+n_0-1, ..., o_{d-1}:o_{d-1}+n_{d-1}-1].
    cell_cols = []
    for ids in blocks:
        shp = ids.shape
        if any(s < 2 for s in shp):
            continue
        cols = []
        for a in range(nc):
            sl = tuple(slice((a >> (d - 1 - ax)) & 1,
                             ((a >> (d - 1 - ax)) & 1) + shp[ax] - 1)
                       for ax in range(d))
            cols.append(ids[sl].reshape(-1))
        cell_cols.append(np.stack(cols, axis=1))  # (ncell_block, nc)
    Ccell = (np.concatenate(cell_cols) if cell_cols
             else np.zeros((0, nc), dtype=np.int64))
    ncell = Ccell.shape[0]

    # Per-sample corner / neighbour / sign arrays. Sample (c, b) -> id nc*c + b.
    sgn_b = np.where((b_idx[:, None] & axbit[None, :]) == 0, 1.0, -1.0)  # (nc, d)
    gc = Ccell.reshape(-1)
    gn = [Ccell[:, b_idx ^ axbit[k]].reshape(-1) for k in range(d)]
    s = [np.tile(sgn_b[:, k], ncell) for k in range(d)]
    ns = gc.shape[0]

    # Membership: in cell c, corner a (node Ccell[c, a]) meets sample b
    # (id nc*c + b) with role role[a, b]. Flatten the (cell, a, b) rows.
    cell_sid = nc * np.arange(ncell)[:, None] + b_idx[None, :]      # (ncell, nc)
    m_node = np.repeat(Ccell[:, :, None], nc, axis=2).reshape(-1)
    m_sid = np.repeat(cell_sid[:, None, :], nc, axis=1).reshape(-1)
    m_role = np.tile(role.reshape(-1), ncell)

    return {"gc": gc, "gn": gn, "s": s, "m_node": m_node, "m_sid": m_sid,
            "m_role": m_role, "nc": nc, "ns": ns, "ncell": ncell}


def build_flat_context(blocks, free_mask, dof_entities, d, *, w_inv):
    """Assemble the ragged ``{"groups", "energy_stencil"}`` wire format.

    ``blocks``       list of global-node-id arrays, one per structured block;
    ``free_mask``    bool ``(N,)`` — ``True`` marks a moving DOF, ``False`` fixed;
    ``dof_entities`` ``{node id -> entity}`` for every moving DOF (``None`` free);
    ``w_inv``        the inverse target metric, either one uniform ``(d, d)``
                     matrix or one ``(ns, d, d)`` matrix per sample (energy-
                     stencil order).

    A single group holds every moving DOF (under frozen halos every free DOF
    reads the previous snapshot, so the DOFs need no ordering). Each DOF
    contributes its incident samples contiguously with a ``P_of`` count (the C++
    core derives the sample offsets); sample order within a DOF is irrelevant, as
    the energy is a sum.
    """
    st = cell_stencil(blocks, d)
    ns = st["ns"]
    fixed = ~free_mask
    dd = d * d
    N = free_mask.shape[0]

    w_inv = np.asarray(w_inv, dtype=np.float64)
    uniform_w = w_inv.shape == (d, d)
    # The C++ interior-patch synthesis hard-codes an identity W_inv, so the
    # fast path is only valid when the target is a scalar multiple of the
    # identity everywhere (the shape metric is scale-invariant). For any
    # spatially-varying or anisotropic target (e.g. BoundaryLayerTarget),
    # interior DOFs must ship their real per-sample stencils — synthesizing
    # them would relax the interior towards isotropy while the reported
    # energy (true W) climbs.
    w0 = w_inv if uniform_w else (w_inv[0] if len(w_inv) else np.eye(d))
    synth_interior = bool(
        np.allclose(w0, w0[0, 0] * np.eye(d))
        and (uniform_w or np.allclose(w_inv, w0)))

    # Interior-eligible nodes: strictly inside a block (logical in [1, n-2] on every
    # axis), so the node AND its whole 2^d-cell patch sit inside that block. The C++
    # core synthesizes their patch off the block layout (identity target), so they
    # carry NO per-sample stencil on the wire — just their (block, logical). Every
    # other moving DOF (block faces, entity boundaries) keeps its stored patch.
    node_block = np.full(N, -1, dtype=np.int32)
    node_logical = np.zeros((N, d), dtype=np.int32)
    if synth_interior:
        for b, ids in enumerate(blocks):
            sh = np.asarray(ids).shape
            if any(s < 3 for s in sh):
                continue  # no strictly-interior node on a < 3-wide axis
            inner = np.asarray(ids)[tuple(slice(1, s - 1) for s in sh)].reshape(-1)
            axes = np.meshgrid(*[np.arange(1, s - 1, dtype=np.int32) for s in sh],
                               indexing="ij")
            node_block[inner] = b
            node_logical[inner] = np.stack([a.reshape(-1) for a in axes], axis=1)
    node_interior = node_block >= 0

    m_node_full = st["m_node"]

    # Keep membership for moving, non-interior DOFs only: fixed corners own no
    # patch, and interior DOFs are synthesized (they contribute no stored samples,
    # so their P_of is 0).
    keep = ~fixed[m_node_full] & ~node_interior[m_node_full]
    m_node = m_node_full[keep]
    m_sid = st["m_sid"][keep]
    m_role = st["m_role"][keep]

    def sample_fields(sid, *, uniform_out=False):
        gc = st["gc"][sid].astype(np.int32)
        out = {"gc": gc}
        for k in range(d):
            out[f"gn{k}"] = st["gn"][k][sid].astype(np.int32)
            out[f"s{k}"] = st["s"][k][sid].astype(np.float64)
        if uniform_w and uniform_out:
            # A single shared row: the C++ metric table reads a uniform W_inv with
            # stride 0. Avoids materializing (and uploading) one identical (d, d)
            # per sample — the largest wire field for a fixed/identity target.
            W = np.asarray(w_inv, dtype=np.float64).reshape(1, dd)
        elif uniform_w:
            W = np.broadcast_to(w_inv, (sid.shape[0], d, d)).reshape(-1, dd)
        else:
            W = w_inv[sid].reshape(-1, dd)
        out["W_inv"] = np.ascontiguousarray(W)
        return out

    # Sort membership by node so each DOF's samples are contiguous and the DOFs
    # ascend — one group over every moving DOF.
    perm = np.argsort(m_node, kind="stable")
    m_node, m_sid, m_role = m_node[perm], m_sid[perm], m_role[perm]

    groups = []
    dofs = np.flatnonzero(free_mask)
    if dofs.size:
        # P_of per DOF (0 for a DOF with no incident cell); m_node is ascending so
        # the concatenated sample rows are already in dof-ascending order.
        uniq, cnt = np.unique(m_node, return_counts=True)
        P_of = np.zeros(dofs.size, dtype=np.int32)
        P_of[np.searchsorted(dofs, uniq)] = cnt
        g = {"D": int(dofs.size), "role": m_role.astype(np.int32),
             "dof_idx": dofs.astype(np.int32), "P_of": P_of,
             "interior_block": np.ascontiguousarray(node_block[dofs]),
             "interior_logical": np.ascontiguousarray(node_logical[dofs]),
             "entities": group_entities_by_type(dofs.tolist(), dof_entities, d=d)}
        g.update(sample_fields(m_sid, uniform_out=True))
        groups.append(g)

    all_sid = np.arange(ns)
    energy_stencil = {"num_samples": int(ns)}
    # A uniform target ships as one shared W_inv row (the energy reduction reads it
    # with stride 0), same as the group table — avoids one (d, d) per sample.
    energy_stencil.update(sample_fields(all_sid, uniform_out=True))
    return {"groups": groups, "energy_stencil": energy_stencil}
