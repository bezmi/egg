// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file control_net.hpp
/// Device-side control-net machinery for the reduced Gauss-Newton smoother:
/// the fine grid is the fixed linear map X = M·C + b (M a tensor product of
/// per-axis clamped B-spline bases sampled at the block's node parameters,
/// b the Boolean-sum boundary correction), and the TMOP energy is minimised
/// over the free control points. This header owns the linear-map kernels:
///
///  - control_eval_kernel      C -> X into the halo-padded structured store
///  - control_sample_g/hz      per energy-stencil sample: wt·dmu (gradient)
///                             or wt·H_T·dT (GN Hessian action) into a z buffer
///  - control_gather_kernel    per-node y = sum Jb^T z over the sample
///                             membership (race-free CSR gather, no atomics)
///  - control_gather_hess      per-node contracted J^T H_T J (the Jacobi
///                             preconditioner input, NOT the GN matvec — the
///                             contracted form drops cross-node couplings)
///  - control_pullback_kernel  free-control gradient/matvec G = M_f^T y
///  - control_precond_*        per-control D×D Jacobi preconditioner
///
/// The GN matvec chains through the per-sample FULL metric Hessian H_T
/// (eval corr -> dT per sample -> H_T·dT -> J^T scatter -> M^T pullback);
/// the per-node contracted Hessian is over-stiff as a curvature model for a
/// globally coupled step — measured against the NumPy reference, it degrades
/// convergence to slow gradient descent — so it is used only to precondition.
///
/// No global CSR M: per-axis 1-D basis tables (span offset + k_supp weights
/// per node) are stored and kernels form the k_supp^D tensor weights in
/// registers; M^T ops are control-major gathers over contiguous support
/// intervals — race-free by construction.

#include "core.hpp"
#include "device.hpp"
#include "metric.hpp"
#include "patch.hpp"
#include "structured.hpp"
#include "sweep.hpp"  // StencilViewT / EnergyStencilHostT

#include <array>
#include <cstddef>
#include <sycl/sycl.hpp>
#include <vector>

namespace egg
{

/// One wall face on a CAD entity: sliding boundary controls + the
/// orthogonality dial (host wire; see egg/smoothing/control_wall.py for the
/// semantics — the device path mirrors the NumPy wall reference).
struct ControlWallHost {
    int axis = 0;
    int side = 0;
    int mode = 0;              ///< 0 off, 1 penalty, 2 hard
    std::vector<real> weight;  ///< [m] per-leg penalty weight
    std::vector<int> p0;       ///< [m] sliding boundary controls (flat ids)
    std::vector<int> p1;       ///< [m] matching first-interior controls
    int tag = 0;               ///< entity tag (fixed-size analytic types only)
    std::vector<real> params;  ///< entity record row (EntitySoA layout)
};

/// Unpacked control-net wire for ONE block (host side; validated in the
/// bindings). Control lattice is row-major over ctrl_shape with the D
/// coordinate components innermost, matching the node store convention.
template <int D> struct ControlNetHostT {
    int block = 0;                        ///< structured block the net drives
    std::array<int, D> ctrl_shape {};     ///< per-axis control counts
    int k_supp = 4;                       ///< support width per axis (degree+1)
    std::array<std::vector<int>, D> off;  ///< [n_k] first support control per node
    std::array<std::vector<real>, D> wt;  ///< [n_k * k_supp] basis values
    std::vector<real> C0;                 ///< [n_ctrl * D] initial control net
    std::vector<int> free_idx;            ///< flat free-control lattice ids
    std::vector<real> b;                  ///< [n_block_nodes * D] Boolean-sum offset
    std::vector<ControlWallHost> walls;   ///< wall faces (may be empty)
    std::vector<real> X0;                 ///< [n_block_nodes * D] initial exact
                                          ///< node field (required with walls)
};

/// One sliding wall root for the in-session frame loop: a shared boundary
/// control constrained to a CAD entity. The reduction CSR STRUCTURE is
/// frame-invariant (the root has D-1 tangential columns); only the coef
/// values change per frame, so the wire ships the entry rewrite lists
/// recorded when Python built the CSR:
///   r_coef[r_entry[e]] = base[e] * t[kt[e]][comp[e]]
/// with t the frozen tangent frame at the root's projected foot.
struct ControlSlideRootHost {
    int rep_row = 0;                ///< stacked control index carrying the position
    std::vector<int> members;       ///< stacked rows referencing the root
    std::vector<real> member_coef;  ///< scalar R coefficient per member row
    int tag = 0;                    ///< entity tag
    std::vector<real> params;       ///< device-layout record row
    std::vector<int> r_entry, comp, kt;
    std::vector<real> base;
};

/// One CAD wall face for the in-session b re-extension.
struct ControlWallFaceHost {
    int block = 0;
    int axis = 0;
    int side = 0;
    int tag = 0;
    std::vector<real> params;
};

template <int D> struct ControlTopoHostT {
    std::vector<ControlNetHostT<D>> nets;  ///< per block (free_idx/walls unused;
                                           ///< X0 anchors the b re-extension)
    std::vector<real> C0;                  ///< stacked initial controls
    std::size_t nq = 0;                    ///< reduced free DOF count
    std::vector<int> r_off, r_col;         ///< reduction CSR (N_stack·D rows)
    std::vector<real> r_coef;
    std::vector<int> p_off, p_col;  ///< penalty CSR (n_pen rows), may be empty
    std::vector<real> p_coef, p_w;
    // Sliding-wall frame loop (empty = frames driven externally, if at all).
    std::vector<real> arena;  ///< shared segmented entity payloads
    std::vector<ControlSlideRootHost> slide;
    std::vector<ControlWallFaceHost> wall_faces;
    std::vector<std::vector<std::uint8_t>> seam_face;  ///< per net, 2*D flags
};

/// Trivially-copyable per-kernel metadata (captured by value).
template <int D> struct ControlNetMetaT {
    std::array<int, D> node_shape;   ///< block interior node counts
    std::array<int, D> ctrl_shape;   ///< control lattice shape
    std::array<int, D> ctrl_stride;  ///< row-major control lattice strides
    std::array<int, D> node_stride;  ///< padded node-index strides (node units)
    int base_node;                   ///< padded node index of logical (0,…,0)
    int k_supp;
    int n_nodes;  ///< prod(node_shape)
    int n_ctrl;   ///< prod(ctrl_shape)
    int n_free;   ///< free control count
};

namespace control_detail
{

/// Structured node index of block-logical index `logical` under `m`.
template <int D> inline int node_slot(const ControlNetMetaT<D>& m, const int (&logical)[D])
{
    int slot = m.base_node;
    for (int k = 0; k < D; ++k) { slot += logical[k] * m.node_stride[k]; }
    return slot;
}

/// Unravel a row-major flat index over `shape` (last axis fastest).
template <int D> inline void unravel(int f, const std::array<int, D>& shape, int (&out)[D])
{
    for (int k = D - 1; k >= 0; --k) {
        out[k] = f % shape[k];
        f /= shape[k];
    }
}

/// Role-selected contraction of a vec(T)-space vector `z` onto one node's D
/// coordinates: c_i = (role part of J)^T z, i.e. the gradient contraction of
/// patch.hpp::accumulate_sample with z in place of dmu_dT.
template <int D>
inline void
  contract_role(const real* z, const real* w, int role, const real (&svals)[D], real (&c)[D])
{
    // dA[i][k] = sum_j z[i*D+j] * w[k*D+j]
    real dA[D][D];
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) {
            real acc = 0.0_r;
            for (int j = 0; j < D; ++j) { acc += z[(i * D) + j] * w[(k * D) + j]; }
            dA[i][k] = acc;
        }
    }
    for (int i = 0; i < D; ++i) { c[i] = 0.0_r; }
    if (role == 0) {
        for (int i = 0; i < D; ++i) {
            real acc = 0.0_r;
            for (int k = 0; k < D; ++k) { acc += svals[k] * dA[i][k]; }
            c[i] = -acc;
        }
    } else if (role >= 1) {
        const int ax = role - 1;
        for (int i = 0; i < D; ++i) { c[i] = svals[ax] * dA[i][ax]; }
    }
}

/// Index of the segment containing @p gid under the prefix array @p pfx
/// (n+1 entries, pfx[0] = 0) — the net lookup of the batched kernels.
inline int find_net(const int* pfx, int n, int gid)
{
    int lo = 0;
    int hi = n;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (pfx[mid] <= gid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Spline value (M·C)(n) at block-logical node @p logical — the shared body
/// of the per-net and batched eval kernels (odometer over the k_supp^D
/// tensor support).
template <int D>
inline void eval_spline_at(const ControlNetMetaT<D>& m,
                           const std::array<const int*, D>& off,
                           const std::array<const real*, D>& wt,
                           const real* C,
                           const int (&logical)[D],
                           real (&acc)[D])
{
    for (int c = 0; c < D; ++c) { acc[c] = 0.0_r; }
    int t[D];
    for (int k = 0; k < D; ++k) { t[k] = 0; }
    while (true) {
        real w = 1.0_r;
        int ci = 0;
        for (int k = 0; k < D; ++k) {
            const int a = off[k][logical[k]] + t[k];
            w *= wt[k][(logical[k] * m.k_supp) + t[k]];
            ci += a * m.ctrl_stride[k];
        }
        for (int c = 0; c < D; ++c) { acc[c] += w * C[(ci * D) + c]; }
        int k = D - 1;
        for (; k >= 0; --k) {
            if (++t[k] < m.k_supp) { break; }
            t[k] = 0;
        }
        if (k < 0) { break; }
    }
}

/// Control-major support gather Σ_{n ∈ supp(a)} w_n · y[slot(n)] for the
/// control with per-axis multi-index @p a — the shared pullback body.
template <int D>
inline void pullback_ctrl(const ControlNetMetaT<D>& m,
                          const std::array<const int*, D>& sup_off,
                          const std::array<const int*, D>& sup_node,
                          const std::array<const real*, D>& sup_w,
                          const int (&a)[D],
                          const real* y,
                          real (&acc)[D])
{
    int len[D];
    int base[D];
    for (int k = 0; k < D; ++k) {
        base[k] = sup_off[k][a[k]];
        len[k] = sup_off[k][a[k] + 1] - base[k];
    }
    for (int c = 0; c < D; ++c) { acc[c] = 0.0_r; }
    int t[D];
    for (int k = 0; k < D; ++k) { t[k] = 0; }
    bool any = true;
    for (int k = 0; k < D; ++k) { any = any && (len[k] > 0); }
    while (any) {
        real w = 1.0_r;
        int logical[D];
        for (int k = 0; k < D; ++k) {
            const int e = base[k] + t[k];
            logical[k] = sup_node[k][e];
            w *= sup_w[k][e];
        }
        const int slot = node_slot<D>(m, logical);
        for (int c = 0; c < D; ++c) { acc[c] += w * y[(slot * D) + c]; }
        int k = D - 1;
        for (; k >= 0; --k) {
            if (++t[k] < len[k]) { break; }
            t[k] = 0;
        }
        if (k < 0) { break; }
    }
}

/// Per-control D×D Jacobi block Σ_n w_n² H_node(n) over the control's tensor
/// support — the shared preconditioner-build body.
template <int D>
inline void precond_ctrl(const ControlNetMetaT<D>& m,
                         const std::array<const int*, D>& sup_off,
                         const std::array<const int*, D>& sup_node,
                         const std::array<const real*, D>& sup_w,
                         const int (&a)[D],
                         const real* h_buf,
                         real (&P)[D * D])
{
    int len[D];
    int base[D];
    for (int k = 0; k < D; ++k) {
        base[k] = sup_off[k][a[k]];
        len[k] = sup_off[k][a[k] + 1] - base[k];
    }
    for (int i = 0; i < D * D; ++i) { P[i] = 0.0_r; }
    int t[D];
    for (int k = 0; k < D; ++k) { t[k] = 0; }
    bool any = true;
    for (int k = 0; k < D; ++k) { any = any && (len[k] > 0); }
    while (any) {
        real w = 1.0_r;
        int logical[D];
        for (int k = 0; k < D; ++k) {
            const int e = base[k] + t[k];
            logical[k] = sup_node[k][e];
            w *= sup_w[k][e];
        }
        const int slot = node_slot<D>(m, logical);
        const real w2 = w * w;
        const real* Hn = h_buf + (static_cast<std::size_t>(slot) * D * D);
        for (int i = 0; i < D * D; ++i) { P[i] += w2 * Hn[i]; }
        int k = D - 1;
        for (; k >= 0; --k) {
            if (++t[k] < len[k]) { break; }
            t[k] = 0;
        }
        if (k < 0) { break; }
    }
}

/// Boolean-sum extension value of a boundary @p delta field (block-logical,
/// net-local) at node @p f: boundary rows pass the delta through, interior
/// rows use the closed form Σ_ax lerp_ax − (D−1)·corner blend — the shared
/// body of tfi_ext_add / tfi_ext_fill_block and their batched variants.
template <int D>
inline void tfi_ext_value(const ControlNetMetaT<D>& m,
                          const real* delta,
                          int f,
                          const int (&logical)[D],
                          real (&out)[D])
{
    int nstr[D];
    nstr[D - 1] = 1;
    for (int k = D - 2; k >= 0; --k) { nstr[k] = nstr[k + 1] * m.node_shape[k + 1]; }
    bool boundary = false;
    for (int k = 0; k < D; ++k) {
        if (logical[k] == 0 || logical[k] == m.node_shape[k] - 1) { boundary = true; }
    }
    if (boundary) {
        for (int c = 0; c < D; ++c) { out[c] = delta[(static_cast<std::size_t>(f) * D) + c]; }
        return;
    }
    real xi[D];
    for (int k = 0; k < D; ++k) {
        xi[k] = static_cast<real>(logical[k]) / static_cast<real>(m.node_shape[k] - 1);
    }
    for (int c = 0; c < D; ++c) { out[c] = 0.0_r; }
    for (int bits = 0; bits < (1 << D); ++bits) {
        real w = 1.0_r;
        int cf = 0;
        for (int k = 0; k < D; ++k) {
            if (bits & (1 << k)) {
                w *= xi[k];
                cf += (m.node_shape[k] - 1) * nstr[k];
            } else {
                w *= 1.0_r - xi[k];
            }
        }
        for (int c = 0; c < D; ++c) {
            out[c] -= static_cast<real>(D - 1) * w * delta[(static_cast<std::size_t>(cf) * D) + c];
        }
    }
    for (int ax = 0; ax < D; ++ax) {
        for (int side = 0; side < 2; ++side) {
            const int fp = f + ((side == 0 ? 0 : m.node_shape[ax] - 1) - logical[ax]) * nstr[ax];
            const real w = side == 0 ? 1.0_r - xi[ax] : xi[ax];
            for (int c = 0; c < D; ++c) {
                out[c] += w * delta[(static_cast<std::size_t>(fp) * D) + c];
            }
        }
    }
}

/// db/dC forward at one wall-face node: out = P_j · v(slot(f)).
template <int D>
inline void db_face_delta_at(
  const ControlNetMetaT<D>& m, int f, const real* Pj, const real* v, real (&out)[D])
{
    int logical[D];
    unravel<D>(f, m.node_shape, logical);
    const int slot = node_slot<D>(m, logical);
    for (int r = 0; r < D; ++r) {
        real acc = 0.0_r;
        for (int c = 0; c < D; ++c) {
            acc += Pj[(static_cast<std::size_t>(r) * D) + c] *
                   v[(static_cast<std::size_t>(slot) * D) + c];
        }
        out[r] = acc;
    }
}

/// db/dC adjoint at one wall-face node: y(slot(f)) += P_jᵀ · (extᵀ u)(f) —
/// the shared body of db_face_adjoint_kernel and its batched variant. Only
/// block-interior slots and the node's own slot are read.
template <int D>
inline void db_face_adjoint_at(const ControlNetMetaT<D>& m, int f, const real* Pj, const real* u, real* y)
{
    int logical[D];
    unravel<D>(f, m.node_shape, logical);
    const int slot = node_slot<D>(m, logical);
    int n_ext = 0;
    int ext_ax = -1;
    int ext_side = 0;
    for (int k = 0; k < D; ++k) {
        if (logical[k] == 0) {
            ++n_ext;
            ext_ax = k;
            ext_side = 0;
        } else if (logical[k] == m.node_shape[k] - 1) {
            ++n_ext;
            ext_ax = k;
            ext_side = 1;
        }
    }
    real dstar[D];
    for (int c = 0; c < D; ++c) { dstar[c] = u[(static_cast<std::size_t>(slot) * D) + c]; }
    if (n_ext == 1) {
        // Face-interior node: adjoint of its axis lerp over the line.
        int li[D];
        for (int k = 0; k < D; ++k) { li[k] = logical[k]; }
        for (int i = 1; i < m.node_shape[ext_ax] - 1; ++i) {
            li[ext_ax] = i;
            const real xi = static_cast<real>(i) / static_cast<real>(m.node_shape[ext_ax] - 1);
            const real w = ext_side == 0 ? 1.0_r - xi : xi;
            const int s = node_slot<D>(m, li);
            for (int c = 0; c < D; ++c) { dstar[c] += w * u[(static_cast<std::size_t>(s) * D) + c]; }
        }
    } else if (n_ext == D) {
        // Block corner: adjoint of the corner blend over the interior.
        int li[D];
        for (int k = 0; k < D; ++k) { li[k] = 1; }
        bool any = true;
        for (int k = 0; k < D; ++k) {
            if (m.node_shape[k] < 3) { any = false; }
        }
        while (any) {
            real w = 1.0_r;
            for (int k = 0; k < D; ++k) {
                const real xi = static_cast<real>(li[k]) / static_cast<real>(m.node_shape[k] - 1);
                w *= logical[k] == 0 ? 1.0_r - xi : xi;
            }
            const int s = node_slot<D>(m, li);
            for (int c = 0; c < D; ++c) {
                dstar[c] -= static_cast<real>(D - 1) * w * u[(static_cast<std::size_t>(s) * D) + c];
            }
            int k = D - 1;
            for (; k >= 0; --k) {
                if (++li[k] < m.node_shape[k] - 1) { break; }
                li[k] = 1;
            }
            if (k < 0) { break; }
        }
    }
    for (int r = 0; r < D; ++r) {
        real acc = 0.0_r;
        for (int c = 0; c < D; ++c) {
            acc += Pj[(static_cast<std::size_t>(c) * D) + r] * dstar[c];
        }
        y[(static_cast<std::size_t>(slot) * D) + r] += acc;
    }
}

}  // namespace control_detail

/// Per-net device record for the batched multi-net kernels: the metadata plus
/// every per-net table pointer one work item needs, uploaded once as an array
/// so a single launch covers all nets of a multi-block session. Pointers stay
/// valid for the session's lifetime (UsmBuffer::upload rewrites in place).
template <int D> struct NetRefT {
    ControlNetMetaT<D> meta;
    std::array<const int*, D> off;       ///< forward basis: first support control
    std::array<const real*, D> wt;       ///< forward basis: weights
    std::array<const int*, D> sup_off;   ///< support CSR (control -> nodes)
    std::array<const int*, D> sup_node;
    std::array<const real*, D> sup_w;
    const real* b = nullptr;  ///< Boolean-sum offset (block-logical)
    int c_off = 0;            ///< first stacked control of this net
    int node_off = 0;         ///< prefix sum of n_nodes (batched item spaces)
};

/// Evaluate the net into the structured store: X[slot(n)] = (M C)(n) [+ b(n)]
/// for every block-logical node n. With C a direction and b null this is the
/// node-space correction corr = M·dC (the map is linear).
template <int D>
inline void control_eval_kernel(sycl::queue& q,
                                const ControlNetMetaT<D>& meta,
                                std::array<const int*, D> off,
                                std::array<const real*, D> wt,
                                const real* C,
                                const real* b,
                                real* X)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_nodes)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        const int slot = control_detail::node_slot<D>(m, logical);
        real acc[D];
        control_detail::eval_spline_at<D>(m, off, wt, C, logical, acc);
        for (int c = 0; c < D; ++c) {
            real v = acc[c];
            if (b != nullptr) { v += b[(f * D) + c]; }
            X[(slot * D) + c] = v;
        }
    });
}

/// Batched eval over every net of a multi-block session: one launch over the
/// concatenated block-logical node space (net looked up by prefix search).
/// Identical per-item arithmetic to control_eval_kernel.
template <int D>
inline void control_eval_all_kernel(sycl::queue& q,
                                    const NetRefT<D>* nets,
                                    const int* node_pfx,
                                    int n_nets,
                                    std::size_t total_nodes,
                                    const real* C_stacked,
                                    bool with_b,
                                    real* X)
{
    q.parallel_for(sycl::range<1>(total_nodes), [=](sycl::id<1> idx) {
        const int gid = static_cast<int>(idx[0]);
        const int ni = control_detail::find_net(node_pfx, n_nets, gid);
        const NetRefT<D>& net = nets[ni];
        const ControlNetMetaT<D> m = net.meta;
        const int f = gid - node_pfx[ni];
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        const int slot = control_detail::node_slot<D>(m, logical);
        real acc[D];
        control_detail::eval_spline_at<D>(m,
                                          net.off,
                                          net.wt,
                                          C_stacked + (static_cast<std::size_t>(net.c_off) * D),
                                          logical,
                                          acc);
        for (int c = 0; c < D; ++c) {
            real v = acc[c];
            if (with_b) { v += net.b[(f * D) + c]; }
            X[(slot * D) + c] = v;
        }
    });
}

/// Per-sample gradient payload: z_p = wt_p · dmu/dvec(T)|_{T_p(X)}.
template <int D, class Obj>
inline void control_sample_g_kernel(
  sycl::queue& q, const StencilViewT<D>& es, const real* X, real* z_buf, Obj objective)
{
    constexpr int kVT = dim::vecT(D);
    q.parallel_for(sycl::range<1>(es.num_samples), [=](sycl::id<1> idx) {
        const int i = static_cast<int>(idx[0]);
        real detA;
        const VecTN<D> t = sample_vecT<D>(es, X, i, detA);
        const GradN<D> g = objective.grad(t);
        const real wt =
          es.weight.extent(0) > 0
            ? es.weight[static_cast<std::size_t>(i) * static_cast<std::size_t>(es.wt_stride)]
            : 1.0_r;
        real* z = z_buf + (static_cast<std::size_t>(i) * kVT);
        for (int a = 0; a < kVT; ++a) { z[a] = wt * g[a]; }
    });
}

/// Per-sample GN Hessian action: z_p = wt_p · H_T,p · dT_p with
/// dT_p = A(corr)·W_inv (the map X -> T is affine, so the directional change
/// of vec(T) along corr is exactly the sample assembly applied to corr).
/// H_T is recomputed from the CURRENT X per call — nothing is stored, which
/// is what keeps the 3D flagship memory footprint flat.
template <int D, class Obj>
inline void control_sample_hz_kernel(sycl::queue& q,
                                     const StencilViewT<D>& es,
                                     const real* X,
                                     const real* corr,
                                     real* z_buf,
                                     Obj objective)
{
    constexpr int kVT = dim::vecT(D);
    q.parallel_for(sycl::range<1>(es.num_samples), [=](sycl::id<1> idx) {
        const int i = static_cast<int>(idx[0]);
        real detA;
        const VecTN<D> t = sample_vecT<D>(es, X, i, detA);
        const HessN<D> H = objective.hess(t);
        real detc;
        const VecTN<D> dt = sample_vecT<D>(es, corr, i, detc);
        const real wt =
          es.weight.extent(0) > 0
            ? es.weight[static_cast<std::size_t>(i) * static_cast<std::size_t>(es.wt_stride)]
            : 1.0_r;
        real* z = z_buf + (static_cast<std::size_t>(i) * kVT);
        for (int a = 0; a < kVT; ++a) {
            real acc = 0.0_r;
            for (int bb = 0; bb < kVT; ++bb) { acc += H[(a * kVT) + bb] * dt[bb]; }
            z[a] = wt * acc;
        }
    });
}

/// Race-free membership gather: y[node] = Σ_{(p, role) ∋ node} Jb(role)^T z_p
/// over the node's CSR sample membership. With z from control_sample_g this
/// is the per-node energy gradient; with z from control_sample_hz it is the
/// node-space GN Hessian action.
template <int D>
inline void control_gather_kernel(sycl::queue& q,
                                  const StencilViewT<D>& es,
                                  const int* mem_nodes,
                                  const int* mem_off,
                                  const int* mem_sid,
                                  const int* mem_role,
                                  std::size_t n_member,
                                  const real* z_buf,
                                  real* y)
{
    constexpr int kVT = dim::vecT(D);
    q.parallel_for(sycl::range<1>(n_member), [=](sycl::id<1> idx) {
        const std::size_t mi = idx[0];
        const int node = mem_nodes[mi];
        real acc[D];
        for (int i = 0; i < D; ++i) { acc[i] = 0.0_r; }
        for (int e = mem_off[mi]; e < mem_off[mi + 1]; ++e) {
            const int p = mem_sid[e];
            const real* w =
              &es.W_inv[static_cast<std::size_t>(p) * static_cast<std::size_t>(es.w_stride)];
            real svals[D];
            for (int k = 0; k < D; ++k) { svals[k] = es.s[k][p]; }
            real c[D];
            control_detail::contract_role<D>(z_buf + (static_cast<std::size_t>(p) * kVT),
                                             w,
                                             mem_role[e],
                                             svals,
                                             c);
            for (int i = 0; i < D; ++i) { acc[i] += c[i]; }
        }
        for (int i = 0; i < D; ++i) { y[(static_cast<std::size_t>(node) * D) + i] = acc[i]; }
    });
}

/// Per-node contracted Hessian Σ Jb^T H_T Jb over the membership — the Jacobi
/// preconditioner input ONLY (it drops cross-node couplings; see file header).
template <int D, class Obj>
inline void control_gather_hess_kernel(sycl::queue& q,
                                       const StencilViewT<D>& es,
                                       const int* mem_nodes,
                                       const int* mem_off,
                                       const int* mem_sid,
                                       const int* mem_role,
                                       std::size_t n_member,
                                       const real* X,
                                       real* h_buf,
                                       Obj objective)
{
    constexpr int kVT = dim::vecT(D);
    q.parallel_for(sycl::range<1>(n_member), [=](sycl::id<1> idx) {
        const std::size_t mi = idx[0];
        const int node = mem_nodes[mi];
        real Hn[D * D];
        for (int i = 0; i < D * D; ++i) { Hn[i] = 0.0_r; }
        for (int e = mem_off[mi]; e < mem_off[mi + 1]; ++e) {
            const int p = mem_sid[e];
            const int role = mem_role[e];
            if (role < 0) { continue; }
            real detA;
            const VecTN<D> t = sample_vecT<D>(es, X, p, detA);
            const HessN<D> H = objective.hess(t);
            const real* w =
              &es.W_inv[static_cast<std::size_t>(p) * static_cast<std::size_t>(es.w_stride)];
            real svals[D];
            for (int k = 0; k < D; ++k) { svals[k] = es.s[k][p]; }
            real Jb[kVT * D];
            role_Jb<D>(role, svals, w, Jb);
            const real wt =
              es.weight.extent(0) > 0
                ? es.weight[static_cast<std::size_t>(p) * static_cast<std::size_t>(es.wt_stride)]
                : 1.0_r;
            real HJb[kVT][D];
            for (int a = 0; a < kVT; ++a) {
                for (int j = 0; j < D; ++j) {
                    real s = 0.0_r;
                    for (int bb = 0; bb < kVT; ++bb) { s += H[(a * kVT) + bb] * Jb[(bb * D) + j]; }
                    HJb[a][j] = s;
                }
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    real s = 0.0_r;
                    for (int a = 0; a < kVT; ++a) { s += Jb[(a * D) + i] * HJb[a][j]; }
                    Hn[(i * D) + j] += wt * s;
                }
            }
        }
        for (int i = 0; i < D * D; ++i) {
            h_buf[(static_cast<std::size_t>(node) * D * D) + i] = Hn[i];
        }
    });
}

/// Free-control pullback G_f = Σ_{n ∈ supp(f)} w_n · y[slot(n)] — a
/// control-major gather over the per-axis support CSR (race-free, no atomics).
template <int D>
inline void control_pullback_kernel(sycl::queue& q,
                                    const ControlNetMetaT<D>& meta,
                                    std::array<const int*, D> sup_off,
                                    std::array<const int*, D> sup_node,
                                    std::array<const real*, D> sup_w,
                                    const int* free_midx,
                                    const real* y,
                                    real* out)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_free)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int a[D];
        for (int k = 0; k < D; ++k) { a[k] = free_midx[(f * D) + k]; }
        real acc[D];
        control_detail::pullback_ctrl<D>(m, sup_off, sup_node, sup_w, a, y, acc);
        for (int c = 0; c < D; ++c) { out[(f * D) + c] = acc[c]; }
    });
}

/// Batched pullback over every net of a multi-block session: one launch over
/// the stacked control space (every lattice point of every net, matching the
/// multi-block convention that freedom lives in the reduction R). Identical
/// per-item arithmetic to control_pullback_kernel.
template <int D>
inline void control_pullback_all_kernel(sycl::queue& q,
                                        const NetRefT<D>* nets,
                                        const int* ctrl_pfx,
                                        int n_nets,
                                        std::size_t n_stack,
                                        const real* y,
                                        real* out)
{
    q.parallel_for(sycl::range<1>(n_stack), [=](sycl::id<1> idx) {
        const int gid = static_cast<int>(idx[0]);
        const int ni = control_detail::find_net(ctrl_pfx, n_nets, gid);
        const NetRefT<D>& net = nets[ni];
        const ControlNetMetaT<D> m = net.meta;
        int a[D];
        control_detail::unravel<D>(gid - ctrl_pfx[ni], m.ctrl_shape, a);
        real acc[D];
        control_detail::pullback_ctrl<D>(m, net.sup_off, net.sup_node, net.sup_w, a, y, acc);
        for (int c = 0; c < D; ++c) { out[(static_cast<std::size_t>(gid) * D) + c] = acc[c]; }
    });
}

/// Per-free-control D×D Jacobi preconditioner block P_f = Σ_n w_n² H_node(n)
/// over the control's tensor support (h_buf from control_gather_hess_kernel).
template <int D>
inline void control_precond_build_kernel(sycl::queue& q,
                                         const ControlNetMetaT<D>& meta,
                                         std::array<const int*, D> sup_off,
                                         std::array<const int*, D> sup_node,
                                         std::array<const real*, D> sup_w,
                                         const int* free_midx,
                                         const real* h_buf,
                                         real* prec)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_free)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int a[D];
        for (int k = 0; k < D; ++k) { a[k] = free_midx[(f * D) + k]; }
        real P[D * D];
        control_detail::precond_ctrl<D>(m, sup_off, sup_node, sup_w, a, h_buf, P);
        for (int i = 0; i < D * D; ++i) { prec[(static_cast<std::size_t>(f) * D * D) + i] = P[i]; }
    });
}

/// Batched preconditioner build over every net of a multi-block session (one
/// launch over the stacked control space). Identical per-item arithmetic to
/// control_precond_build_kernel.
template <int D>
inline void control_precond_build_all_kernel(sycl::queue& q,
                                             const NetRefT<D>* nets,
                                             const int* ctrl_pfx,
                                             int n_nets,
                                             std::size_t n_stack,
                                             const real* h_buf,
                                             real* prec)
{
    q.parallel_for(sycl::range<1>(n_stack), [=](sycl::id<1> idx) {
        const int gid = static_cast<int>(idx[0]);
        const int ni = control_detail::find_net(ctrl_pfx, n_nets, gid);
        const NetRefT<D>& net = nets[ni];
        const ControlNetMetaT<D> m = net.meta;
        int a[D];
        control_detail::unravel<D>(gid - ctrl_pfx[ni], m.ctrl_shape, a);
        real P[D * D];
        control_detail::precond_ctrl<D>(m, net.sup_off, net.sup_node, net.sup_w, a, h_buf, P);
        for (int i = 0; i < D * D; ++i) {
            prec[(static_cast<std::size_t>(gid) * D * D) + i] = P[i];
        }
    });
}

/// Preconditioner apply: z_f = (P_f + λI)^{-1} r_f per free control.
template <int D>
inline void control_precond_apply_kernel(
  sycl::queue& q, int n_free, const real* prec, real lam, const real* r, real* z)
{
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n_free)), [=](sycl::id<1> idx) {
        const std::size_t f = idx[0];
        MatN<D> P;
        for (int i = 0; i < D * D; ++i) { P[i] = prec[(f * D * D) + i]; }
        for (int i = 0; i < D; ++i) { P[(i * D) + i] += lam; }
        VecN<D> rhs;
        for (int i = 0; i < D; ++i) { rhs[i] = -r[(f * D) + i]; }
        // solveNxN solves H δ = −g, so negate the rhs to get P z = r.
        const VecN<D> sol = solveNxN<D>(P, rhs);
        for (int i = 0; i < D; ++i) { z[(f * D) + i] = sol[i]; }
    });
}

/// Scatter a free-control vector into the full control lattice (fixed entries
/// untouched — zero the destination first when a pure direction is needed).
template <int D>
inline void control_scatter_free_kernel(
  sycl::queue& q, int n_free, const int* free_idx, const real* v, real* C_full, real alpha = 1.0_r)
{
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n_free)), [=](sycl::id<1> idx) {
        const std::size_t f = idx[0];
        const std::size_t ci = static_cast<std::size_t>(free_idx[f]);
        for (int c = 0; c < D; ++c) { C_full[(ci * D) + c] += alpha * v[(f * D) + c]; }
    });
}

/// Generic CSR apply out[i] = Σ_e coef[e]·v[col[e]] — the frozen-frame
/// reduction R (reduced wall DOFs -> free-control displacements) and its
/// transpose are both stored as CSRs (≤ 2 controls per row) and applied with
/// this one kernel.
inline void csr_apply_kernel(sycl::queue& q,
                             std::size_t n_rows,
                             const int* off,
                             const int* col,
                             const real* coef,
                             const real* v,
                             real* out)
{
    if (n_rows == 0) { return; }
    q.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0];
        real acc = 0.0_r;
        for (int e = off[i]; e < off[i + 1]; ++e) { acc += coef[e] * v[col[e]]; }
        out[i] = acc;
    });
}

/// Weighted transpose-CSR accumulate out[i] += Σ_e coef[e]·w[row[e]]·r[row[e]]
/// — the Pᵀ·W·r half of a generic quadratic penalty ½‖W^½ P C‖² whose rows
/// may share components (the two-step r = P v, out += PᵀW r split keeps both
/// halves race-free where a fused scatter would race).
inline void csrT_weighted_accum_kernel(sycl::queue& q,
                                       std::size_t n_rows,
                                       const int* off,
                                       const int* row,
                                       const real* coef,
                                       const real* w,
                                       const real* r,
                                       real* out)
{
    if (n_rows == 0) { return; }
    q.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0];
        real acc = 0.0_r;
        for (int e = off[i]; e < off[i + 1]; ++e) { acc += coef[e] * w[row[e]] * r[row[e]]; }
        out[i] += acc;
    });
}

/// Reduce a per-node field over each alias group (all structured copies of
/// one fine node: owner interior + non-owner copies + ghosts): the total goes
/// to the OWNER slot (first member of each group) and every other member is
/// zeroed — the exact adjoint of the owner->copy broadcast. The fine field
/// has one owner per node (the broadcast overwrites copies/ghosts from it),
/// so only the owner slot carries sensitivity; crediting copies too would
/// double-count shared-node contributions in the pullback. Groups are
/// disjoint, so the write-back is race-free. @p ncomp is the per-node
/// component count (D for gradients, D·D for the contracted Hessian).
inline void alias_reduce_kernel(sycl::queue& q,
                                std::size_t n_groups,
                                const int* g_off,
                                const int* g_slot,  // node-slot ids, owner first
                                int ncomp,
                                real* buf)
{
    if (n_groups == 0) { return; }
    q.parallel_for(sycl::range<1>(n_groups), [=](sycl::id<1> idx) {
        const std::size_t g = idx[0];
        for (int c = 0; c < ncomp; ++c) {
            real acc = 0.0_r;
            for (int e = g_off[g]; e < g_off[g + 1]; ++e) {
                acc += buf[(static_cast<std::size_t>(g_slot[e]) * ncomp) + c];
            }
            buf[(static_cast<std::size_t>(g_slot[g_off[g]]) * ncomp) + c] = acc;
            for (int e = g_off[g] + 1; e < g_off[g + 1]; ++e) {
                buf[(static_cast<std::size_t>(g_slot[e]) * ncomp) + c] = 0.0_r;
            }
        }
    });
}

/// Wall-orthogonality penalty action out += Σ_j w̃_j Σ_kt row_{j,kt}(row_{j,kt}ᵀ v)
/// over the free-control displacement space, one row per frozen tangent
/// vector (D-1 of them per leg: walls are codim-1 facets). Each leg touches
/// exactly its own P0/P1 pair and no two legs share a control, so the scatter
/// is race-free; the tangent loop stays inside the leg's work-item.
template <int D>
inline void pen_apply_kernel(sycl::queue& q,
                             std::size_t n_pen,
                             const int* i0,  // [n] free-space row base of P0
                             const int* i1,  // [n] free-space row base of P1
                             const real* t,  // [n*(D-1)*D] frozen tangents
                             const real* w,  // [n] w̃ = weight / |leg|²
                             const real* v,
                             real* out)
{
    constexpr int kT = D - 1;
    if (n_pen == 0) { return; }
    q.parallel_for(sycl::range<1>(n_pen), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        for (int kt = 0; kt < kT; ++kt) {
            const real* tj = t + (((j * kT) + static_cast<std::size_t>(kt)) * D);
            real s = 0.0_r;
            for (int k = 0; k < D; ++k) { s += tj[k] * (v[i1[j] + k] - v[i0[j] + k]); }
            const real ws = w[j] * s;
            for (int k = 0; k < D; ++k) {
                out[i1[j] + k] += ws * tj[k];
                out[i0[j] + k] -= ws * tj[k];
            }
        }
    });
}

/// Per-leg per-tangent components at the current state and along a step:
/// out2[(j·(D-1)+kt)·2] = t̂ᵀL_j(Cfree), …+1 = t̂ᵀdL_j(dC) — downloaded
/// (tiny) so the α-ladder can add the penalty energy analytically
/// (quadratic in α).
template <int D>
inline void pen_residual_kernel(sycl::queue& q,
                                std::size_t n_pen,
                                const int* i0,
                                const int* i1,
                                const real* t,
                                const real* cfree,
                                const real* dc,
                                real* out2)
{
    constexpr int kT = D - 1;
    if (n_pen == 0) { return; }
    q.parallel_for(sycl::range<1>(n_pen), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        for (int kt = 0; kt < kT; ++kt) {
            const std::size_t e = (j * kT) + static_cast<std::size_t>(kt);
            const real* tj = t + (e * D);
            real sv = 0.0_r;
            real sr = 0.0_r;
            for (int k = 0; k < D; ++k) {
                sv += tj[k] * (cfree[i1[j] + k] - cfree[i0[j] + k]);
                sr += tj[k] * (dc[i1[j] + k] - dc[i0[j] + k]);
            }
            out2[e * 2] = sv;
            out2[(e * 2) + 1] = sr;
        }
    });
}

/// Gather the free controls' current positions into a packed [n_free*D] vector.
template <int D>
inline void
  gather_free_kernel(sycl::queue& q, int n_free, const int* free_idx, const real* C_full, real* out)
{
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n_free)), [=](sycl::id<1> idx) {
        const std::size_t f = idx[0];
        const std::size_t ci = static_cast<std::size_t>(free_idx[f]);
        for (int c = 0; c < D; ++c) { out[(f * D) + c] = C_full[(ci * D) + c]; }
    });
}

/// Scalar Jacobi diagonal of the reduced wall system: for reduced DOF c,
/// d_c = Σ_ctrl a_ctrlᵀ P_ctrl a_ctrl over the ≤2 controls its R column
/// touches (a_ctrl = the column's D coefficients on that control; P from
/// control_precond_build_kernel). The penalty contribution is omitted — it
/// only affects PCG iteration count, never the solution.
template <int D>
inline void qdiag_build_kernel(sycl::queue& q,
                               std::size_t nq,
                               const int* g_off,  // gather CSR: column -> rows
                               const int* g_row,  // free-space row indices
                               const real* g_coef,
                               const real* prec,  // [n_free*D*D] per-control blocks
                               real* diag)
{
    if (nq == 0) { return; }
    q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> idx) {
        const std::size_t c = idx[0];
        real d = 0.0_r;
        int e = g_off[c];
        const int e_end = g_off[c + 1];
        while (e < e_end) {
            // Group consecutive entries on the same control (rows are built
            // control-major, so a control's D components are adjacent).
            const int ctrl = g_row[e] / D;
            real a[D];
            for (int k = 0; k < D; ++k) { a[k] = 0.0_r; }
            while (e < e_end && g_row[e] / D == ctrl) {
                a[g_row[e] % D] = g_coef[e];
                ++e;
            }
            const real* P = prec + (static_cast<std::size_t>(ctrl) * D * D);
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) { d += a[i] * P[(i * D) + j] * a[j]; }
            }
        }
        diag[c] = d;
    });
}

/// Scalar-diagonal preconditioner apply: z = r / (d + λ), guarded.
inline void qdiag_apply_kernel(
  sycl::queue& q, std::size_t nq, const real* diag, real lam, const real* r, real* z)
{
    if (nq == 0) { return; }
    q.parallel_for(sycl::range<1>(nq), [=](sycl::id<1> idx) {
        const std::size_t c = idx[0];
        const real d = diag[c] + lam;
        z[c] = d > 0.0_r ? r[c] / d : r[c];
    });
}

// ---------------------------------------------------------------------------
// Small dense-vector helpers for the PCG driver (USM, one launch each).
// ---------------------------------------------------------------------------

inline void vec_fill(sycl::queue& q, real* x, std::size_t n, real v)
{
    if (n) { q.fill(x, v, n); }
}

/// y += a·x
inline void vec_axpy(sycl::queue& q, std::size_t n, real a, const real* x, real* y)
{
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i[0]] += a * x[i[0]]; });
}

/// p = z + beta·p
inline void vec_xpay(sycl::queue& q, std::size_t n, const real* z, real beta, real* p)
{
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { p[i[0]] = z[i[0]] + (beta * p[i[0]]); });
}

/// out[0] = x·y (USM scalar; caller downloads).
inline void vec_dot(sycl::queue& q, std::size_t n, const real* x, const real* y, real* out)
{
    auto red = sycl::reduction(out,
                               0.0_r,
                               std::plus<>(),
                               sycl::property::reduction::initialize_to_identity {});
    q.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> i, auto& sum) {
        sum += x[i[0]] * y[i[0]];
    });
}

/// out[0] = max_i |x_i|.
inline void vec_amax(sycl::queue& q, std::size_t n, const real* x, real* out)
{
    auto red = sycl::reduction(out,
                               0.0_r,
                               sycl::maximum<real>(),
                               sycl::property::reduction::initialize_to_identity {});
    q.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> i, auto& mx) {
        mx.combine(sycl::fabs(x[i[0]]));
    });
}

/// out[0] = Σ_f trace(P_f) over the [n_free · D·D] preconditioner blocks.
template <int D>
inline void vec_trace_sum(sycl::queue& q, std::size_t n_free, const real* prec, real* out)
{
    auto red = sycl::reduction(out,
                               0.0_r,
                               std::plus<>(),
                               sycl::property::reduction::initialize_to_identity {});
    q.parallel_for(sycl::range<1>(n_free), red, [=](sycl::id<1> i, auto& sum) {
        const std::size_t f = i[0];
        for (int k = 0; k < D; ++k) { sum += prec[(f * D * D) + (k * D) + k]; }
    });
}

// --------------------------------------------------------------------------- //
// Frozen-frame Boolean-sum linearization (d b / d C on sliding-wall blocks)
//
// Within a frozen frame the wall-face fine nodes sit at proj(spline): a
// control motion moves the spline face by dS = (M dC)|face and the projected
// foot by (I - n̂n̂ᵀ)·dS, so the face picks up db = -n̂n̂ᵀ·dS and the
// Boolean-sum extension carries the correction inward. Composing this into
// the GN Jacobian (J = (I + L)·M·R with L = ext ∘ (-n̂n̂ᵀ)|wall-face) removes
// the model error that feeds the frame-to-frame energy ratchet: without it a
// tangential wall-control slide looks like free normal fine-node motion.
// --------------------------------------------------------------------------- //

/// delta(f) = P_j · v(slot(f)) for every listed wall-face node (delta is a
/// block-logical field, zero-filled by the caller; P_j is the frozen
/// -n̂n̂ᵀ projector of face node j).
template <int D>
inline void db_face_delta_kernel(sycl::queue& q,
                                 const ControlNetMetaT<D>& meta,
                                 const int* face_flat,
                                 const real* face_P,
                                 std::size_t n_face,
                                 const real* v,
                                 real* delta)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(n_face), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        const int f = face_flat[j];
        real out[D];
        control_detail::db_face_delta_at<D>(m,
                                            f,
                                            face_P + (j * static_cast<std::size_t>(D * D)),
                                            v,
                                            out);
        for (int r = 0; r < D; ++r) { delta[(static_cast<std::size_t>(f) * D) + r] = out[r]; }
    });
}

/// Batched db/dC forward over every wall net of a multi-block session: one
/// launch over the concatenated wall-face node lists (net per face via
/// @p face_net); deltas land in the concatenated block-logical buffer at the
/// net's node prefix. Identical per-item arithmetic to db_face_delta_kernel.
template <int D>
inline void db_face_delta_all_kernel(sycl::queue& q,
                                     const NetRefT<D>* nets,
                                     const int* face_net,
                                     const int* face_flat,
                                     const real* face_P,
                                     std::size_t n_face_total,
                                     const real* v,
                                     real* delta)
{
    q.parallel_for(sycl::range<1>(n_face_total), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        const NetRefT<D>& net = nets[face_net[j]];
        const int f = face_flat[j];
        real out[D];
        control_detail::db_face_delta_at<D>(net.meta,
                                            f,
                                            face_P + (j * static_cast<std::size_t>(D * D)),
                                            v,
                                            out);
        real* d = delta + (static_cast<std::size_t>(net.node_off + f) * D);
        for (int r = 0; r < D; ++r) { d[r] = out[r]; }
    });
}

/// out(slot(n)) += ext(delta)(n): the Boolean-sum extension of a boundary
/// delta field added into a structured-store field. Boundary rows add the
/// delta itself; interior rows use the closed form Σ_ax lerp_ax − (D−1) ·
/// corner blend (exact for a face-supported delta — interior axis
/// projections never land on codim-2 facets, so corners enter only through
/// the corner blend).
template <int D>
inline void
  tfi_ext_add_kernel(sycl::queue& q, const ControlNetMetaT<D>& meta, const real* delta, real* out)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_nodes)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        const int slot = control_detail::node_slot<D>(m, logical);
        real add[D];
        control_detail::tfi_ext_value<D>(m, delta, f, logical, add);
        for (int c = 0; c < D; ++c) { out[(static_cast<std::size_t>(slot) * D) + c] += add[c]; }
    });
}

/// Batched Boolean-sum extension add over every wall net of a multi-block
/// session: one launch over the concatenated block-logical node space; nets
/// without wall-face nodes (@p net_flag zero) pass through untouched. The
/// delta buffer is concatenated at the nets' node prefixes. Identical
/// per-item arithmetic to tfi_ext_add_kernel.
template <int D>
inline void tfi_ext_add_all_kernel(sycl::queue& q,
                                   const NetRefT<D>* nets,
                                   const int* node_pfx,
                                   int n_nets,
                                   std::size_t total_nodes,
                                   const int* net_flag,
                                   const real* delta,
                                   real* out)
{
    q.parallel_for(sycl::range<1>(total_nodes), [=](sycl::id<1> idx) {
        const int gid = static_cast<int>(idx[0]);
        const int ni = control_detail::find_net(node_pfx, n_nets, gid);
        if (net_flag[ni] == 0) { return; }
        const NetRefT<D>& net = nets[ni];
        const ControlNetMetaT<D> m = net.meta;
        const int f = gid - node_pfx[ni];
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        const int slot = control_detail::node_slot<D>(m, logical);
        const real* dl = delta + (static_cast<std::size_t>(net.node_off) * D);
        real add[D];
        control_detail::tfi_ext_value<D>(m, dl, f, logical, add);
        for (int c = 0; c < D; ++c) { out[(static_cast<std::size_t>(slot) * D) + c] += add[c]; }
    });
}

/// b(f) = ext(delta)(f) over the BLOCK-LOGICAL layout: the Boolean-sum
/// extension of a boundary delta field assigned into a per-net b buffer
/// (the layout control_eval_kernel consumes). Same closed form as
/// tfi_ext_add_kernel; boundary rows pass the delta through.
template <int D>
inline void tfi_ext_fill_block_kernel(sycl::queue& q,
                                      const ControlNetMetaT<D>& meta,
                                      const real* delta,
                                      real* b_out)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_nodes)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        real out[D];
        control_detail::tfi_ext_value<D>(m, delta, f, logical, out);
        for (int c = 0; c < D; ++c) { b_out[(static_cast<std::size_t>(f) * D) + c] = out[c]; }
    });
}

/// y(slot(j)) += P_jᵀ · (extᵀ u)(j) for every listed wall-face node: the
/// adjoint of the composed correction. (extᵀ u)(j) is u(j) plus the axis-line
/// lerp sum for a face-interior node, u(j) − (D−1)·(interior corner moment)
/// for a block corner, and plain u(j) on codim-≥2 face nodes that are not
/// corners. Only block-interior slots and the node's own slot are read —
/// both single-owner — so this runs directly on the alias-reduced gradient
/// field.
template <int D>
inline void db_face_adjoint_kernel(sycl::queue& q,
                                   const ControlNetMetaT<D>& meta,
                                   const int* face_flat,
                                   const real* face_P,
                                   std::size_t n_face,
                                   const real* u,
                                   real* y)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(n_face), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        control_detail::db_face_adjoint_at<D>(m,
                                              face_flat[j],
                                              face_P + (j * static_cast<std::size_t>(D * D)),
                                              u,
                                              y);
    });
}

/// Batched db/dC adjoint over every wall net of a multi-block session: one
/// launch over the concatenated wall-face node lists. Race-free across nets
/// for the same reason the per-net launch is race-free across faces: each
/// face writes only its own slot and reads its own slot plus block-interior
/// slots, which are never face nodes. Identical per-item arithmetic to
/// db_face_adjoint_kernel.
template <int D>
inline void db_face_adjoint_all_kernel(sycl::queue& q,
                                       const NetRefT<D>* nets,
                                       const int* face_net,
                                       const int* face_flat,
                                       const real* face_P,
                                       std::size_t n_face_total,
                                       const real* u,
                                       real* y)
{
    q.parallel_for(sycl::range<1>(n_face_total), [=](sycl::id<1> idx) {
        const std::size_t j = idx[0];
        const NetRefT<D>& net = nets[face_net[j]];
        control_detail::db_face_adjoint_at<D>(net.meta,
                                              face_flat[j],
                                              face_P + (j * static_cast<std::size_t>(D * D)),
                                              u,
                                              y);
    });
}

}  // namespace egg
