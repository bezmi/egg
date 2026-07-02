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


def greedy_colours(m_node_full, m_sid_full, nc, N):
    """Welsh-Powell greedy colouring of the share-a-cell graph.

    The graph's edges are the distinct within-cell corner pairs; nodes are
    coloured in descending-degree order (ties by ascending id). ``m_node_full``
    is the *unfiltered* membership node array (before restricting to DOFs), so
    the adjacency covers fixed nodes too — matching the reference partition.
    """
    # Rebuild each cell's nc corner ids from the (cell, a, b) membership: row
    # nc*nc*cell + nc*a + b has node = Ccell[cell, a]; take b == 0.
    ncell = m_node_full.size // (nc * nc)
    Ccell = m_node_full.reshape(ncell, nc, nc)[:, :, 0]
    ai, bi = np.triu_indices(nc, k=1)
    e_u = np.concatenate([Ccell[:, ai].reshape(-1), Ccell[:, bi].reshape(-1)])
    e_v = np.concatenate([Ccell[:, bi].reshape(-1), Ccell[:, ai].reshape(-1)])
    keep = e_u != e_v
    e_u, e_v = e_u[keep], e_v[keep]
    order_e = np.argsort(e_u, kind="stable")
    e_u, e_v = e_u[order_e], e_v[order_e]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(e_u, minlength=N), out=indptr[1:])
    deg = np.diff(indptr)

    colours = np.full(N, -1, dtype=np.int64)
    for v in np.argsort(-deg, kind="stable"):
        used = {int(x) for x in colours[e_v[indptr[v]:indptr[v + 1]]] if x >= 0}
        c = 0
        while c in used:
            c += 1
        colours[v] = c
    return colours


def build_flat_context(blocks, free_mask, dof_entities, d, *, w_inv,
                       colours=None):
    """Assemble the ragged ``{"groups", "energy_stencil"}`` wire format.

    ``blocks``       list of global-node-id arrays, one per structured block;
    ``free_mask``    bool ``(N,)`` — ``True`` marks a moving DOF, ``False`` fixed;
    ``dof_entities`` ``{node id -> entity}`` for every moving DOF (``None`` free);
    ``w_inv``        the inverse target metric, either one uniform ``(d, d)``
                     matrix or one ``(ns, d, d)`` matrix per sample (energy-
                     stencil order);
    ``colours``      optional ``(N,)`` colour per node; a Welsh-Powell greedy
                     colouring of the share-a-cell graph is used when omitted.

    Groups are one ragged partition per colour: a colour's moving DOFs in
    ascending id, each contributing its incident samples contiguously with a
    ``P_of`` count (the C++ core derives the sample offsets). Patch sample order
    within a DOF is irrelevant — the energy is a sum.
    """
    st = cell_stencil(blocks, d)
    nc, ns = st["nc"], st["ns"]
    N = free_mask.shape[0]
    fixed = ~free_mask
    dd = d * d

    m_node_full = st["m_node"]
    if colours is None:
        colours = greedy_colours(m_node_full, st["m_sid"], nc, N)
    colours = np.asarray(colours)

    # Restrict membership to moving DOFs (fixed corners never own a patch).
    keep = ~fixed[m_node_full]
    m_node = m_node_full[keep]
    m_sid = st["m_sid"][keep]
    m_role = st["m_role"][keep]

    w_inv = np.asarray(w_inv, dtype=np.float64)
    uniform_w = w_inv.shape == (d, d)

    def sample_fields(sid):
        gc = st["gc"][sid].astype(np.int32)
        out = {"gc": gc}
        for k in range(d):
            out[f"gn{k}"] = st["gn"][k][sid].astype(np.int32)
            out[f"s{k}"] = st["s"][k][sid].astype(np.float64)
        if uniform_w:
            W = np.broadcast_to(w_inv, (sid.shape[0], d, d)).reshape(-1, dd)
        else:
            W = w_inv[sid].reshape(-1, dd)
        out["W_inv"] = np.ascontiguousarray(W)
        return out

    # Sort membership by (colour, node): a colour's rows become contiguous and
    # its DOFs ascending — the reference partition, with each DOF's samples
    # grouped together.
    m_col = colours[m_node]
    perm = np.lexsort((m_node, m_col))
    m_col, m_node, m_sid, m_role = (m_col[perm], m_node[perm], m_sid[perm],
                                    m_role[perm])
    n_colours = int(colours.max()) + 1 if colours.size and colours.max() >= 0 else 0
    lo_all = np.searchsorted(m_col, np.arange(n_colours), side="left")
    hi_all = np.searchsorted(m_col, np.arange(n_colours), side="right")

    groups = []
    for c in range(n_colours):
        dofs = np.flatnonzero((colours == c) & free_mask)
        if dofs.size == 0:
            continue
        lo, hi = lo_all[c], hi_all[c]
        node_seg, sid_seg, role_seg = m_node[lo:hi], m_sid[lo:hi], m_role[lo:hi]
        # P_of per DOF (0 for a DOF with no incident cell); node_seg is ascending
        # so the concatenated sample rows are already in dof-ascending order.
        uniq, cnt = np.unique(node_seg, return_counts=True)
        P_of = np.zeros(dofs.size, dtype=np.int32)
        P_of[np.searchsorted(dofs, uniq)] = cnt
        g = {"D": int(dofs.size), "role": role_seg.astype(np.int32),
             "dof_idx": dofs.astype(np.int32), "P_of": P_of,
             "entities": group_entities_by_type(dofs.tolist(), dof_entities, d=d)}
        g.update(sample_fields(sid_seg))
        groups.append(g)

    all_sid = np.arange(ns)
    energy_stencil = {"num_samples": int(ns)}
    energy_stencil.update(sample_fields(all_sid))
    return {"groups": groups, "energy_stencil": energy_stencil}
