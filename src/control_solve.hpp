// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file control_solve.hpp
/// Device-resident control-net reduced Gauss-Newton session: owns the control
/// buffers over one structured block and drives the outer loop
///
///   eval C -> X, per-node gradient (sample+gather), pullback G = M_f^T g,
///   matrix-free PCG on (Â + λI) dC = −G  with the per-sample full GN
///   Hessian action (control_net.hpp), corr = M dC, α-ladder line search on
///   the global fine energy with the codebase accept rule
///   (isfinite(e) && e <= e0 + tol::energy && accept_mindet), Levenberg
///   λ-bump on rejection.
///
/// The semantics mirror egg/smoothing/control_ref.py::run_control_ref — the
/// NumPy reference is the parity target — with the dense solve replaced by
/// Jacobi-preconditioned PCG.

#include "control_net.hpp"
#include "core.hpp"
#include "device.hpp"
#include "fas.hpp"       // detail::reduce_linesearch_pair
#include "geometry.hpp"  // host-side entity project / tangent_space (walls)
#include "structured.hpp"
#include "sweep.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <tuple>
#include <utility>
#include <vector>

namespace egg
{

namespace control_detail
{

/// Host-side entity projection through the D-templated dispatch. Segmented
/// entities (B-spline, composite) read their payload from @p arena.
template <int D>
inline PtN<D>
  project_entity(const PtN<D>& p, int tag, const real* params, const real* arena = nullptr)
{
    PtN<D> out {};
    dispatch_entity_type<D>(static_cast<EntityTag>(tag),
                            [&]<class E>() { out = decode_entity<E>(params, arena).project(p); });
    return out;
}

/// Wall tangent frame: the D-1 tangent vectors at @p p (walls are codim-1
/// facets, so the entity must have tdim == D-1; validated at wire unpack).
template <int D>
inline std::array<PtN<D>, D - 1>
  wall_tangent_basis(const PtN<D>& p, int tag, const real* params, const real* arena = nullptr)
{
    std::array<PtN<D>, D - 1> out {};
    dispatch_entity_type<D>(static_cast<EntityTag>(tag), [&]<class E>() {
        if constexpr (E::tdim == D - 1) {
            const auto tb = decode_entity<E>(params, arena).tangent_basis(p);
            for (int k = 0; k < D - 1; ++k) { out[static_cast<std::size_t>(k)] = tb[k]; }
        }
    });
    for (auto& t : out) {
        real n = 0.0_r;
        for (int k = 0; k < D; ++k) { n += t[k] * t[k]; }
        n = sycl::sqrt(n);
        if (n > 0.0_r) {
            for (int k = 0; k < D; ++k) { t[k] /= n; }
        }
    }
    return out;
}

/// tdim of the entity behind @p tag (host classification helper).
template <int D> inline int entity_tdim(int tag)
{
    int td = -1;
    dispatch_entity_type<D>(static_cast<EntityTag>(tag), [&]<class E>() { td = E::tdim; });
    return td;
}

/// Unit normal completing a codim-1 tangent frame: rot90 in 2D, cross in 3D.
template <int D> inline PtN<D> frame_normal(const std::array<PtN<D>, D - 1>& t)
{
    PtN<D> n {};
    if constexpr (D == 2) {
        n = {-t[0][1], t[0][0]};
    } else if constexpr (D == 3) {
        n = {(t[0][1] * t[1][2]) - (t[0][2] * t[1][1]),
             (t[0][2] * t[1][0]) - (t[0][0] * t[1][2]),
             (t[0][0] * t[1][1]) - (t[0][1] * t[1][0])};
    }
    real m = 0.0_r;
    for (int k = 0; k < D; ++k) { m += n[k] * n[k]; }
    m = sycl::sqrt(m);
    if (m > 0.0_r) {
        for (int k = 0; k < D; ++k) { n[k] /= m; }
    }
    return n;
}

}  // namespace control_detail

// --------------------------------------------------------------------------- //
// Device-resident frame rebuild kernels. The per-frame work — sliding-root
// projections + tangent frames, the damped member move, wall-node deltas and
// db/dC projectors — runs on the session queue over tables uploaded once, so
// a frame costs kernel launches plus one 2-scalar download for the validity
// gate (no host round-trip of C or b). The entity Newtons are the same
// device-capable dispatch the nodal sweep's boundary kernels use.
// --------------------------------------------------------------------------- //

/// Per sliding root: project the representative position onto its entity,
/// freeze the tangent frame, store (foot, pos) and rewrite the reduction /
/// gather CSR coefficient values through the per-root entry lists.
template <int D>
inline void frame_slide_kernel(sycl::queue& q,
                               std::size_t n_slide,
                               const int* sl_tag,
                               const int* sl_rep,
                               const real* sl_params,
                               std::size_t sl_stride,
                               const real* arena,
                               const int* entry_off,
                               const int* r_entry,
                               const int* kt,
                               const int* comp,
                               const real* base,
                               const int* g_entry_of_r,
                               const real* C,
                               real* fp_buf,
                               real* r_coef,
                               real* g_coef)
{
    q.parallel_for(sycl::range<1>(n_slide), [=](sycl::id<1> idx) {
        const std::size_t si = idx[0];
        PtN<D> pos {};
        for (int k = 0; k < D; ++k) { pos[k] = C[(static_cast<std::size_t>(sl_rep[si]) * D) + k]; }
        const real* params = sl_params + (si * sl_stride);
        const PtN<D> foot = control_detail::project_entity<D>(pos, sl_tag[si], params, arena);
        auto tb = control_detail::wall_tangent_basis<D>(foot, sl_tag[si], params, arena);
        // Degenerate chart point: freeze this root for the frame.
        for (auto& tv : tb) {
            real n = 0.0_r;
            for (int k = 0; k < D; ++k) { n += tv[k] * tv[k]; }
            if (n < 1e-24_r) {
                for (int k = 0; k < D; ++k) { tv[k] = 0.0_r; }
            }
        }
        for (int k = 0; k < D; ++k) {
            fp_buf[(si * 2 * D) + static_cast<std::size_t>(k)] = foot[k];
            fp_buf[(si * 2 * D) + static_cast<std::size_t>(D + k)] = pos[k];
        }
        for (int e = entry_off[si]; e < entry_off[si + 1]; ++e) {
            const real tv = tb[static_cast<std::size_t>(kt[e])][comp[e]];
            const real v = base[e] * tv;
            const auto re = static_cast<std::size_t>(r_entry[e]);
            r_coef[re] = v;
            g_coef[static_cast<std::size_t>(g_entry_of_r[re])] = v;
        }
    });
}

/// Damped member move: C = C_snap + alpha * coef * (foot - pos) accumulated
/// over every member row of every sliding root. The caller copies C_snap
/// into C first; the accumulation is atomic because a stacked row may in
/// principle reference more than one sliding root.
template <int D>
inline void frame_member_move_kernel(sycl::queue& q,
                                     std::size_t n_members,
                                     const int* members,
                                     const int* mem_root,
                                     const real* mem_coef,
                                     const real* fp_buf,
                                     real alpha,
                                     real* C)
{
    q.parallel_for(sycl::range<1>(n_members), [=](sycl::id<1> idx) {
        const std::size_t mi = idx[0];
        const auto row = static_cast<std::size_t>(members[mi]);
        const auto si = static_cast<std::size_t>(mem_root[mi]);
        for (int k = 0; k < D; ++k) {
            const real diff = fp_buf[(si * 2 * D) + static_cast<std::size_t>(k)] -
                              fp_buf[(si * 2 * D) + static_cast<std::size_t>(D + k)];
            sycl::atomic_ref<real,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
              ref(C[(row * D) + static_cast<std::size_t>(k)]);
            ref.fetch_add(alpha * mem_coef[mi] * diff);
        }
    });
}

/// Per wall net, over every block-logical node: the boundary delta of the b
/// re-extension (proj(spline) - spline on wall faces, X0 - spline on other
/// outer faces, zero on seams and in the interior), plus the frozen -n̂n̂ᵀ
/// db/dC projector at listed wall-face nodes. `Xs` is the pure spline field
/// in the structured store (eval_all without b).
template <int D>
inline void frame_wall_delta_kernel(sycl::queue& q,
                                    const ControlNetMetaT<D>& meta,
                                    const int* seam_flags,
                                    const real* X0,
                                    int net_block,
                                    std::size_t n_wf,
                                    const int* wf_block,
                                    const int* wf_axis,
                                    const int* wf_side,
                                    const int* wf_tag,
                                    const real* wf_params,
                                    std::size_t wf_stride,
                                    const real* arena,
                                    const real* Xs,
                                    bool with_db,
                                    const int* dbn_idx_of_flat,
                                    real* dbn_P,
                                    real* delta)
{
    const ControlNetMetaT<D> m = meta;
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(m.n_nodes)), [=](sycl::id<1> idx) {
        const int f = static_cast<int>(idx[0]);
        int logical[D];
        control_detail::unravel<D>(f, m.node_shape, logical);
        bool boundary = false;
        bool on_seam = false;
        for (int k = 0; k < D; ++k) {
            const int side = logical[k] == 0 ? 0 : (logical[k] == m.node_shape[k] - 1 ? 1 : -1);
            if (side >= 0) {
                boundary = true;
                if (seam_flags[(2 * k) + side] != 0) { on_seam = true; }
            }
        }
        if (!boundary || on_seam) {
            for (int c = 0; c < D; ++c) { delta[(static_cast<std::size_t>(f) * D) + c] = 0.0_r; }
            return;
        }
        const int slot = control_detail::node_slot<D>(m, logical);
        PtN<D> xs {};
        for (int c = 0; c < D; ++c) { xs[c] = Xs[(static_cast<std::size_t>(slot) * D) + c]; }
        PtN<D> exact {};
        for (int c = 0; c < D; ++c) { exact[c] = X0[(static_cast<std::size_t>(f) * D) + c]; }
        for (std::size_t w = 0; w < n_wf; ++w) {
            if (wf_block[w] != net_block) { continue; }
            const int end = wf_side[w] == 0 ? 0 : m.node_shape[wf_axis[w]] - 1;
            if (logical[wf_axis[w]] != end) { continue; }
            const real* params = wf_params + (w * wf_stride);
            const PtN<D> proj = control_detail::project_entity<D>(xs, wf_tag[w], params, arena);
            for (int c = 0; c < D; ++c) { exact[c] = proj[c]; }
            if (with_db && dbn_idx_of_flat != nullptr && dbn_idx_of_flat[f] >= 0) {
                // Frozen -n̂n̂ᵀ at the foot: orthonormalize the chart
                // tangents, then P = -(I - Σ t̂ t̂ᵀ); a degenerate chart
                // point keeps the zero projector.
                auto tb = control_detail::wall_tangent_basis<D>(proj, wf_tag[w], params, arena);
                real on[D - 1][D];
                int n_on = 0;
                for (auto& tv : tb) {
                    real v[D];
                    for (int k = 0; k < D; ++k) { v[k] = tv[k]; }
                    for (int uj = 0; uj < n_on; ++uj) {
                        real dp = 0.0_r;
                        for (int k = 0; k < D; ++k) { dp += on[uj][k] * v[k]; }
                        for (int k = 0; k < D; ++k) { v[k] -= dp * on[uj][k]; }
                    }
                    real nrm = 0.0_r;
                    for (int k = 0; k < D; ++k) { nrm += v[k] * v[k]; }
                    if (nrm > 1e-24_r) {
                        nrm = sycl::sqrt(nrm);
                        for (int k = 0; k < D; ++k) { on[n_on][k] = v[k] / nrm; }
                        ++n_on;
                    }
                }
                real* P = dbn_P + (static_cast<std::size_t>(dbn_idx_of_flat[f]) *
                                   static_cast<std::size_t>(D * D));
                for (int r = 0; r < D; ++r) {
                    for (int c = 0; c < D; ++c) {
                        real ttc = 0.0_r;
                        for (int uj = 0; uj < n_on; ++uj) { ttc += on[uj][r] * on[uj][c]; }
                        real p = n_on == 0 ? 0.0_r : ttc;
                        if (r == c && n_on > 0) { p -= 1.0_r; }
                        P[(static_cast<std::size_t>(r) * D) + c] = p;
                    }
                }
            }
        }
        for (int c = 0; c < D; ++c) {
            delta[(static_cast<std::size_t>(f) * D) + c] = exact[c] - xs[c];
        }
    });
}

struct ControlOptions {
    int n_outer = 30;         ///< max outer GN iterations
    real grad_tol = 1e-8_r;   ///< converged when max|G| drops below
    real alpha_min = 1e-4_r;  ///< line-search floor
    real lm0 = 0.0_r;         ///< initial Levenberg damping
    int lm_tries = 8;         ///< λ-ladder length per outer iteration
    int pcg_max_iter = 200;
    real pcg_rtol = 1e-10_r;  ///< PCG relative-residual floor (see pcg_forcing)
    /// Inexact-Newton forcing (Eisenstat–Walker flavour): solve the GN system
    /// only as tightly as the outer residual warrants — the effective PCG
    /// tolerance is max(pcg_rtol, min(0.5, sqrt(max|G|_k / max|G|_0))). A GN
    /// step needs 1–2 digits for the line search to get a good descent
    /// direction; the accept rule + λ-ladder own step quality. Disable for
    /// exact-solve parity runs (the NumPy reference solves densely).
    bool pcg_forcing = true;
    /// Model the frozen-frame Boolean-sum linearization d b/d C in the GN
    /// Jacobian on sliding-wall runs (J = (I + ext∘(-n̂n̂ᵀ)|wall)·M·R).
    /// Without it a tangential wall-control slide looks like free normal
    /// fine-node motion and the per-frame b re-extension injects energy —
    /// the sliding-frame ratchet. Applies to the in-session frame loop only.
    bool model_db = true;
};

struct ControlReport {
    std::vector<real> energies;  ///< accepted energies (initial state first)
    std::vector<real> mindets;
    std::vector<real> alphas;      ///< accepted step lengths
    std::vector<real> pcg_iters;   ///< PCG iterations per solve (diagnostics)
    std::vector<real> pcg_relres;  ///< PCG final relative residual per solve
    /// Energy injected by each frame rebuild (reprojection + b re-extension)
    /// relative to the previously accepted state — the ratchet's raw signal.
    std::vector<real> frame_jumps;
    int iters = 0;
    bool converged = false;
};

/// Control-net state resident next to a structured sweep session. Built once
/// from the (already structured-remapped) energy stencil + the control wire;
/// `run` operates in place on the session's packed X buffer.
template <int D> class ControlSessionT
{
  public:
    /// @param q        the session's queue (shared; in-order).
    /// @param layout   packed block layout (node slots / strides).
    /// @param es_host  energy stencil with STRUCTURED node indices (the
    ///                 membership tables are derived from it here).
    /// @param es       device stencil view over the same samples.
    /// @param host     validated control wire for one block.
    /// @param X        the session's packed coordinate buffer (USM).
    ControlSessionT(sycl::queue q,
                    const BlockLayout<D>& layout,
                    const EnergyStencilHostT<D>& es_host,
                    const StencilViewT<D>& es,
                    const ControlNetHostT<D>& host,
                    real* X) : q_(std::move(q)), es_(es), X_(X)
    {
        const auto b = static_cast<std::size_t>(host.block);
        const auto shape = layout.interior_shape(b);
        const auto nstr = layout.node_strides(b);

        meta_.k_supp = host.k_supp;
        meta_.n_nodes = 1;
        meta_.n_ctrl = 1;
        for (int k = 0; k < D; ++k) {
            const auto ks = static_cast<std::size_t>(k);
            meta_.node_shape[ks] = static_cast<int>(shape[ks]);
            meta_.ctrl_shape[ks] = host.ctrl_shape[ks];
            meta_.node_stride[ks] = static_cast<int>(nstr[ks] / static_cast<std::size_t>(D));
            meta_.n_nodes *= meta_.node_shape[ks];
            meta_.n_ctrl *= meta_.ctrl_shape[ks];
        }
        // Row-major control strides, last axis fastest.
        int acc = 1;
        for (int k = D - 1; k >= 0; --k) {
            meta_.ctrl_stride[static_cast<std::size_t>(k)] = acc;
            acc *= meta_.ctrl_shape[static_cast<std::size_t>(k)];
        }
        meta_.base_node =
          static_cast<int>(layout.interior_origin_offset(b) / static_cast<std::size_t>(D));
        meta_.n_free = static_cast<int>(host.free_idx.size());

        // Upload the per-axis basis tables + net state.
        for (int k = 0; k < D; ++k) {
            const auto ks = static_cast<std::size_t>(k);
            off_[ks] = UsmBuffer<int> {q_, host.off[ks]};
            wt_[ks] = UsmBuffer<real> {q_, host.wt[ks]};
        }
        C_ = UsmBuffer<real> {q_, host.C0};
        b_ = UsmBuffer<real> {q_, host.b};
        free_idx_ = UsmBuffer<int> {q_, host.free_idx};

        // Free-control multi-indices (row-major unravel of free_idx).
        {
            std::vector<int> midx(host.free_idx.size() * static_cast<std::size_t>(D));
            for (std::size_t f = 0; f < host.free_idx.size(); ++f) {
                int rem = host.free_idx[f];
                for (int k = D - 1; k >= 0; --k) {
                    midx[(f * D) + static_cast<std::size_t>(k)] =
                      rem % meta_.ctrl_shape[static_cast<std::size_t>(k)];
                    rem /= meta_.ctrl_shape[static_cast<std::size_t>(k)];
                }
            }
            free_midx_ = UsmBuffer<int> {q_, midx};
        }

        // Per-axis support CSR (control -> nodes with nonzero basis weight),
        // inverted from the forward tables host-side.
        for (int k = 0; k < D; ++k) {
            const auto ks = static_cast<std::size_t>(k);
            const int n_ctrl_k = host.ctrl_shape[ks];
            const int n_node_k = meta_.node_shape[ks];
            std::vector<std::vector<std::pair<int, real>>> lists(
              static_cast<std::size_t>(n_ctrl_k));
            for (int i = 0; i < n_node_k; ++i) {
                const int a0 = host.off[ks][static_cast<std::size_t>(i)];
                for (int t = 0; t < host.k_supp; ++t) {
                    const real w = host.wt[ks][static_cast<std::size_t>((i * host.k_supp) + t)];
                    if (w != 0.0_r) { lists[static_cast<std::size_t>(a0 + t)].emplace_back(i, w); }
                }
            }
            std::vector<int> soff(static_cast<std::size_t>(n_ctrl_k) + 1, 0);
            std::vector<int> snode;
            std::vector<real> sw;
            for (int a = 0; a < n_ctrl_k; ++a) {
                for (const auto& [i, w] : lists[static_cast<std::size_t>(a)]) {
                    snode.push_back(i);
                    sw.push_back(w);
                }
                soff[static_cast<std::size_t>(a) + 1] = static_cast<int>(snode.size());
            }
            sup_off_[ks] = UsmBuffer<int> {q_, soff};
            sup_node_[ks] = UsmBuffer<int> {q_, snode};
            sup_w_[ks] = UsmBuffer<real> {q_, sw};
        }

        // Sample-membership CSR (structured node slot -> (sample, role)),
        // derived from the remapped stencil. std::map keeps the node order
        // deterministic (fp reduction order stability across runs).
        {
            std::map<int, std::vector<std::pair<int, int>>> mem;
            const auto P = es_host.num_samples;
            for (std::size_t p = 0; p < P; ++p) {
                mem[es_host.gc[p]].emplace_back(static_cast<int>(p), 0);
                for (int k = 0; k < D; ++k) {
                    mem[es_host.gn[k][p]].emplace_back(static_cast<int>(p), k + 1);
                }
            }
            std::vector<int> nodes;
            std::vector<int> moff {0};
            std::vector<int> sid;
            std::vector<int> role;
            for (const auto& [node, entries] : mem) {
                nodes.push_back(node);
                for (const auto& [p, r] : entries) {
                    sid.push_back(p);
                    role.push_back(r);
                }
                moff.push_back(static_cast<int>(sid.size()));
            }
            n_member_ = nodes.size();
            mem_nodes_ = UsmBuffer<int> {q_, nodes};
            mem_off_ = UsmBuffer<int> {q_, moff};
            mem_sid_ = UsmBuffer<int> {q_, sid};
            mem_role_ = UsmBuffer<int> {q_, role};
        }

        // Work buffers.
        const std::size_t total_reals = layout.total_reals();
        const auto nfd = static_cast<std::size_t>(meta_.n_free) * static_cast<std::size_t>(D);
        const std::size_t total_nodes = total_reals / static_cast<std::size_t>(D);
        corr_ = UsmBuffer<real> {q_, std::vector<real>(total_reals, 0.0_r)};
        y_ = UsmBuffer<real> {q_, std::vector<real>(total_reals, 0.0_r)};
        h_node_ =
          UsmBuffer<real> {q_,
                           std::vector<real>(total_nodes * static_cast<std::size_t>(D * D), 0.0_r)};
        z_samples_ = UsmBuffer<real> {q_, es_.num_samples * static_cast<std::size_t>(dim::vecT(D))};
        C_dir_ = UsmBuffer<real> {
          q_,
          std::vector<real>(static_cast<std::size_t>(meta_.n_ctrl) * static_cast<std::size_t>(D),
                            0.0_r)};
        prec_ = UsmBuffer<real> {q_, nfd * static_cast<std::size_t>(D)};
        G_ = UsmBuffer<real> {q_, nfd};
        dC_ = UsmBuffer<real> {q_, nfd};
        pcg_r_ = UsmBuffer<real> {q_, nfd};
        pcg_z_ = UsmBuffer<real> {q_, nfd};
        pcg_p_ = UsmBuffer<real> {q_, nfd};
        pcg_ap_ = UsmBuffer<real> {q_, nfd};
        scal_ = UsmBuffer<real> {q_, 8};

        // Wall machinery (sliding boundary controls + orthogonality).
        if (!host.walls.empty()) { init_walls(host); }
    }

    /// Evaluate the resident net into the session's packed X (with b).
    void eval_to_x()
    {
        control_eval_kernel<D>(q_, meta_, off_ptrs(), wt_ptrs(), C_.data(), b_.data(), X_);
        q_.wait();
    }

    void get_c(real* out) { C_.download(out); }

    void set_c(const std::vector<real>& c)
    {
        if (c.size() != C_.size()) {
            throw std::invalid_argument("set_c: expected " + std::to_string(C_.size()) +
                                        " reals, got " + std::to_string(c.size()));
        }
        C_.upload(c);
        eval_to_x();
    }

    std::size_t c_size() const { return C_.size(); }

    /// Run the damped reduced-GN outer loop in place on the session's X.
    /// With walls the loop runs over the frozen-frame reduced DOFs (dC = R dq)
    /// with a per-iteration frame rebuild; without walls it is the plain
    /// free-control loop.
    template <class Obj> ControlReport run(Obj objective, const ControlOptions& opts)
    {
        if (has_walls_) { return run_walls(objective, opts); }
        return run_plain(objective, opts);
    }

  private:
    template <class Obj> ControlReport run_plain(Obj objective, const ControlOptions& opts)
    {
        constexpr real kLamFloorScale = 1e-8_r;
        ControlReport rep;
        const auto nfd = static_cast<std::size_t>(meta_.n_free) * static_cast<std::size_t>(D);
        real lam = opts.lm0;
        std::array<real, 8> s {};

        // The net owns X from here: evaluate the resident C (+ b) so the
        // optimisation starts exactly on the net's manifold.
        eval_to_x();
        reduce_energy_mindet<D>(q_, es_, X_, scal_.data(), scal_.data() + 1, objective);
        download(2, s);
        real e0 = s[0];
        real md0 = s[1];
        rep.energies.push_back(e0);
        rep.mindets.push_back(md0);

        real gmax0 = 0.0_r;
        for (int outer = 0; outer < opts.n_outer; ++outer) {
            // Reduced gradient G = M_f^T g (sample dmu -> node gather -> pullback).
            control_sample_g_kernel<D>(q_, es_, X_, z_samples_.data(), objective);
            gather_to(y_.data());
            // The directional soft-energy nodal gradient joins the fine
            // gradient before the pullback — the accept energies already
            // compose the term, so the model gradient must see it too.
            scatter_directional_grad<D>(q_, es_.dir, X_, y_.data());
            pullback_to(G_.data());
            vec_amax(q_, nfd, G_.data(), scal_.data());
            download(1, s);
            const real gmax = s[0];
            if (gmax < opts.grad_tol) {
                rep.converged = true;
                break;
            }
            if (outer == 0) { gmax0 = gmax; }
            const real pcg_tol =
              opts.pcg_forcing ? std::max(opts.pcg_rtol, std::min(0.5_r, sycl::sqrt(gmax / gmax0)))
                               : opts.pcg_rtol;

            // Per-node contracted Hessian -> Jacobi preconditioner + λ floor.
            control_gather_hess_kernel<D>(q_,
                                          es_,
                                          mem_nodes_.data(),
                                          mem_off_.data(),
                                          mem_sid_.data(),
                                          mem_role_.data(),
                                          n_member_,
                                          X_,
                                          h_node_.data(),
                                          objective);
            control_precond_build_kernel<D>(q_,
                                            meta_,
                                            sup_off_ptrs(),
                                            sup_node_ptrs(),
                                            sup_w_ptrs(),
                                            free_midx_.data(),
                                            h_node_.data(),
                                            prec_.data());
            vec_trace_sum<D>(q_,
                             static_cast<std::size_t>(meta_.n_free),
                             prec_.data(),
                             scal_.data());
            download(1, s);
            const real lam_floor = kLamFloorScale * s[0] / static_cast<real>(nfd);

            bool accepted = false;
            real alpha = 1.0_r;
            real e_new = e0;
            real md_new = md0;
            for (int lt = 0; lt < opts.lm_tries && !accepted; ++lt) {
                const auto [pcg_its, relres] =
                  pcg_solve(objective, lam, opts.pcg_max_iter, pcg_tol);
                rep.pcg_iters.push_back(static_cast<real>(pcg_its));
                rep.pcg_relres.push_back(relres);

                // corr = M dC (direction eval, no b), zero off-block by
                // construction (corr_ was zero-filled once; eval overwrites
                // exactly the block's node slots every call).
                scatter_dir(dC_.data());
                control_eval_kernel<D>(q_,
                                       meta_,
                                       off_ptrs(),
                                       wt_ptrs(),
                                       C_dir_.data(),
                                       nullptr,
                                       corr_.data());

                // α-ladder, two trials per fused reduction (fas.hpp pattern).
                alpha = 1.0_r;
                while (alpha >= opts.alpha_min) {
                    detail::reduce_linesearch_pair<D>(q_,
                                                      es_,
                                                      X_,
                                                      corr_.data(),
                                                      alpha,
                                                      alpha * 0.5_r,
                                                      scal_.data(),
                                                      objective);
                    download(4, s);
                    bool ok = false;
                    for (int t = 0; t < 2; ++t) {
                        const real e = s[(2 * t) + 0];
                        const real md = s[(2 * t) + 1];
                        const real a = t == 0 ? alpha : alpha * 0.5_r;
                        if (a >= opts.alpha_min && sycl::isfinite(e) && e <= e0 + tol::energy &&
                            objective.accept_mindet(md)) {
                            alpha = a;
                            e_new = e;
                            md_new = md;
                            ok = true;
                            break;
                        }
                    }
                    if (ok) {
                        accepted = true;
                        break;
                    }
                    alpha *= 0.25_r;
                }
                if (!accepted) { lam = std::max(10.0_r * lam, lam_floor); }
            }
            if (!accepted) { break; }  // damping exhausted; report what we have

            // Commit: X += α corr, C_free += α dC.
            vec_axpy(q_, x_reals(), alpha, corr_.data(), X_);
            control_scatter_free_kernel<D>(q_,
                                           meta_.n_free,
                                           free_idx_.data(),
                                           dC_.data(),
                                           C_.data(),
                                           alpha);
            e0 = e_new;
            md0 = md_new;
            rep.energies.push_back(e0);
            rep.mindets.push_back(md0);
            rep.alphas.push_back(alpha);
            rep.iters += 1;
            if (alpha == 1.0_r) { lam /= 3.0_r; }
        }
        q_.wait();
        return rep;
    }

  private:
    std::array<const int*, D> off_ptrs() const { return ptrs(off_); }
    std::array<const real*, D> wt_ptrs() const { return ptrs(wt_); }
    std::array<const int*, D> sup_off_ptrs() const { return ptrs(sup_off_); }
    std::array<const int*, D> sup_node_ptrs() const { return ptrs(sup_node_); }
    std::array<const real*, D> sup_w_ptrs() const { return ptrs(sup_w_); }

    template <class T> static std::array<const T*, D> ptrs(const std::array<UsmBuffer<T>, D>& bufs)
    {
        std::array<const T*, D> out {};
        for (int k = 0; k < D; ++k) { out[static_cast<std::size_t>(k)] = bufs[k].data(); }
        return out;
    }

    std::size_t x_reals() const { return corr_.size(); }

    void download(int n, std::array<real, 8>& s)
    {
        q_.wait();
        q_.memcpy(s.data(), scal_.data(), static_cast<std::size_t>(n) * sizeof(real)).wait();
    }

    /// z_samples_ -> per-node field via the membership gather.
    void gather_to(real* y)
    {
        control_gather_kernel<D>(q_,
                                 es_,
                                 mem_nodes_.data(),
                                 mem_off_.data(),
                                 mem_sid_.data(),
                                 mem_role_.data(),
                                 n_member_,
                                 z_samples_.data(),
                                 y);
    }

    /// y_ -> free-control vector.
    void pullback_to(real* out)
    {
        control_pullback_kernel<D>(q_,
                                   meta_,
                                   sup_off_ptrs(),
                                   sup_node_ptrs(),
                                   sup_w_ptrs(),
                                   free_midx_.data(),
                                   y_.data(),
                                   out);
    }

    /// Zero the full-lattice direction and scatter the free vector into it.
    void scatter_dir(const real* v_free)
    {
        vec_fill(q_, C_dir_.data(), C_dir_.size(), 0.0_r);
        control_scatter_free_kernel<D>(q_, meta_.n_free, free_idx_.data(), v_free, C_dir_.data());
    }

    /// Matrix-free (Â + λI)·v with the per-sample full GN Hessian chain:
    /// scatter v -> eval corr -> per-sample H_T·dT -> node gather -> pullback,
    /// then += λ·v. Writes into @p out.
    template <class Obj> void matvec(Obj objective, real lam, const real* v, real* out)
    {
        scatter_dir(v);
        control_eval_kernel<D>(q_,
                               meta_,
                               off_ptrs(),
                               wt_ptrs(),
                               C_dir_.data(),
                               nullptr,
                               corr_.data());
        control_sample_hz_kernel<D>(q_, es_, X_, corr_.data(), z_samples_.data(), objective);
        gather_to(y_.data());
        pullback_to(out);
        if (lam > 0.0_r) {
            vec_axpy(q_,
                     static_cast<std::size_t>(meta_.n_free) * static_cast<std::size_t>(D),
                     lam,
                     v,
                     out);
        }
    }

    /// Jacobi-preconditioned CG on (Â + λI) dC = −G; dC_ receives the step.
    /// Returns (iterations run, final relative residual) for diagnostics.
    template <class Obj>
    std::pair<int, real> pcg_solve(Obj objective, real lam, int max_iter, real rtol)
    {
        const auto n = static_cast<std::size_t>(meta_.n_free) * static_cast<std::size_t>(D);
        std::array<real, 8> s {};

        vec_fill(q_, dC_.data(), n, 0.0_r);
        // r = b = −G.
        q_.parallel_for(sycl::range<1>(n),
                        [r = pcg_r_.data(), g = G_.data()](sycl::id<1> i) { r[i[0]] = -g[i[0]]; });
        vec_dot(q_, n, pcg_r_.data(), pcg_r_.data(), scal_.data());
        download(1, s);
        const real b_norm2 = s[0];
        if (b_norm2 == 0.0_r) { return {0, 0.0_r}; }
        const real stop2 = rtol * rtol * b_norm2;
        real r_norm2 = b_norm2;

        control_precond_apply_kernel<D>(q_,
                                        meta_.n_free,
                                        prec_.data(),
                                        lam,
                                        pcg_r_.data(),
                                        pcg_z_.data());
        q_.memcpy(pcg_p_.data(), pcg_z_.data(), n * sizeof(real));
        vec_dot(q_, n, pcg_r_.data(), pcg_z_.data(), scal_.data());
        download(1, s);
        real rz = s[0];

        int it = 0;
        for (; it < max_iter; ++it) {
            matvec(objective, lam, pcg_p_.data(), pcg_ap_.data());
            vec_dot(q_, n, pcg_p_.data(), pcg_ap_.data(), scal_.data());
            download(1, s);
            const real p_ap = s[0];
            if (!(p_ap > 0.0_r)) { break; }  // negative curvature: stop at the last iterate
            const real a = rz / p_ap;
            vec_axpy(q_, n, a, pcg_p_.data(), dC_.data());
            vec_axpy(q_, n, -a, pcg_ap_.data(), pcg_r_.data());
            vec_dot(q_, n, pcg_r_.data(), pcg_r_.data(), scal_.data());
            download(1, s);
            r_norm2 = s[0];
            if (r_norm2 <= stop2) {
                ++it;
                break;
            }
            control_precond_apply_kernel<D>(q_,
                                            meta_.n_free,
                                            prec_.data(),
                                            lam,
                                            pcg_r_.data(),
                                            pcg_z_.data());
            vec_dot(q_, n, pcg_r_.data(), pcg_z_.data(), scal_.data());
            download(1, s);
            const real rz_new = s[0];
            vec_xpay(q_, n, pcg_z_.data(), rz_new / rz, pcg_p_.data());
            rz = rz_new;
        }
        return {it, sycl::sqrt(r_norm2 / b_norm2)};
    }

    // -----------------------------------------------------------------------
    // Walls: sliding boundary controls + orthogonality (mirrors the NumPy
    // reference in egg/smoothing/control_wall.py — see its module docstring).
    // The per-outer-iteration frame rebuild runs on the C++ host: control
    // counts are tiny, and it needs entity projection + the Boolean-sum
    // re-extension, both host-side today.
    // -----------------------------------------------------------------------

    void init_walls(const ControlNetHostT<D>& host)
    {
        has_walls_ = true;
        hostc_ = host;  // basis tables, X0, wall specs (host copies)
        C_host_ = host.C0;
        const std::size_t F = host.free_idx.size();
        row_of_ctrl_.assign(static_cast<std::size_t>(meta_.n_ctrl), -1);
        for (std::size_t f = 0; f < F; ++f) {
            row_of_ctrl_[static_cast<std::size_t>(host.free_idx[f])] = static_cast<int>(f);
        }
        // Classify controls + fix the (constant) reduced-DOF column layout:
        // hard-driven P1 -> 0 own columns + 1 shared h column; sliding P0 ->
        // 1 tangential column; every other free control -> D columns.
        std::size_t n_legs = 0;
        std::size_t n_pen_legs = 0;
        for (const auto& w : host.walls) {
            if (w.p0.size() != w.p1.size() || w.weight.size() != w.p0.size()) {
                throw std::invalid_argument("control walls: p0/p1/weight length mismatch");
            }
            // Walls are codim-1 facets: the entity's tangent dimension must
            // be D-1 (a curve in 2D, a surface in 3D).
            if (control_detail::entity_tdim<D>(w.tag) != D - 1) {
                throw std::invalid_argument(
                  "control walls: the wall entity's tangent dimension must be D-1");
            }
            for (std::size_t j = 0; j < w.p0.size(); ++j) {
                if (row_of_ctrl_[static_cast<std::size_t>(w.p0[j])] < 0 ||
                    row_of_ctrl_[static_cast<std::size_t>(w.p1[j])] < 0) {
                    throw std::invalid_argument(
                      "control walls: a wall control is not in the free set");
                }
            }
            n_legs += w.p0.size();
            if (w.mode == 1) { n_pen_legs += w.p0.size(); }
        }
        sliding_.assign(static_cast<std::size_t>(meta_.n_ctrl), 0);
        driven_.assign(static_cast<std::size_t>(meta_.n_ctrl), 0);
        for (const auto& w : hostc_.walls) {
            for (std::size_t j = 0; j < w.p0.size(); ++j) {
                sliding_[static_cast<std::size_t>(w.p0[j])] = 1;
                if (w.mode == 2) { driven_[static_cast<std::size_t>(w.p1[j])] = 1; }
            }
        }
        // Column layout: interior controls D columns, sliding controls D-1
        // tangential columns, hard-driven controls one shared h column.
        nq_ = 0;
        col_of_ctrl_.assign(static_cast<std::size_t>(meta_.n_ctrl), -1);
        for (int g : host.free_idx) {
            if (driven_[static_cast<std::size_t>(g)]) { continue; }
            col_of_ctrl_[static_cast<std::size_t>(g)] = static_cast<int>(nq_);
            nq_ += sliding_[static_cast<std::size_t>(g)] ? static_cast<std::size_t>(D - 1)
                                                         : static_cast<std::size_t>(D);
        }
        h_col_of_ctrl_.assign(static_cast<std::size_t>(meta_.n_ctrl), -1);
        for (const auto& w : hostc_.walls) {
            if (w.mode != 2) { continue; }
            for (int p1 : w.p1) {
                h_col_of_ctrl_[static_cast<std::size_t>(p1)] = static_cast<int>(nq_++);
            }
        }
        n_pen_ = n_pen_legs;

        // Fixed-size device buffers (structure constant; contents per
        // rebuild). Entries per free-space row: 1 (interior), D-1 (sliding),
        // D (driven: foot tangents + h) — cap at D.
        const std::size_t nfd = F * static_cast<std::size_t>(D);
        const std::size_t max_ent = static_cast<std::size_t>(D) * nfd;
        r_off_ = UsmBuffer<int> {q_, nfd + 1};
        r_col_ = UsmBuffer<int> {q_, max_ent};
        r_coef_ = UsmBuffer<real> {q_, max_ent};
        g_off_ = UsmBuffer<int> {q_, nq_ + 1};
        g_row_ = UsmBuffer<int> {q_, max_ent};
        g_coef_ = UsmBuffer<real> {q_, max_ent};
        if (n_pen_) {
            constexpr std::size_t kT = D - 1;
            pen_i0_ = UsmBuffer<int> {q_, n_pen_};
            pen_i1_ = UsmBuffer<int> {q_, n_pen_};
            pen_t_ = UsmBuffer<real> {q_, n_pen_ * kT * static_cast<std::size_t>(D)};
            pen_w_ = UsmBuffer<real> {q_, n_pen_};
            pen2_ = UsmBuffer<real> {q_, 2 * n_pen_ * kT};
        }
        qG_ = UsmBuffer<real> {q_, nq_};
        qdq_ = UsmBuffer<real> {q_, nq_};
        qr_ = UsmBuffer<real> {q_, nq_};
        qz_ = UsmBuffer<real> {q_, nq_};
        qp_ = UsmBuffer<real> {q_, nq_};
        qap_ = UsmBuffer<real> {q_, nq_};
        qdiag_ = UsmBuffer<real> {q_, nq_};
        cfree_ = UsmBuffer<real> {q_, nfd};
        vfull_ = UsmBuffer<real> {q_, nfd};
        // The h floor scale: the shortest initial hard leg (fallback 1).
        real lmin = std::numeric_limits<real>::infinity();
        for (const auto& w : hostc_.walls) {
            for (std::size_t j = 0; j < w.p0.size(); ++j) {
                real s = 0.0_r;
                for (int k = 0; k < D; ++k) {
                    const real dk = C_host_[(static_cast<std::size_t>(w.p1[j]) * D) + k] -
                                    C_host_[(static_cast<std::size_t>(w.p0[j]) * D) + k];
                    s += dk * dk;
                }
                lmin = std::min(lmin, sycl::sqrt(s));
            }
        }
        h_min_ = std::isfinite(lmin) ? 1e-3_r * lmin : 1e-3_r;
    }

    /// Host spline eval at a block-logical node (from the host basis tables).
    std::array<real, D> spline_at(const int (&logical)[D]) const
    {
        std::array<real, D> out {};
        int t[D];
        for (int k = 0; k < D; ++k) { t[k] = 0; }
        while (true) {
            real w = 1.0_r;
            int ci = 0;
            for (int k = 0; k < D; ++k) {
                const auto ks = static_cast<std::size_t>(k);
                const int a = hostc_.off[ks][static_cast<std::size_t>(logical[k])] + t[k];
                w *= hostc_.wt[ks][static_cast<std::size_t>((logical[k] * meta_.k_supp) + t[k])];
                ci += a * meta_.ctrl_stride[ks];
            }
            for (int c = 0; c < D; ++c) {
                out[static_cast<std::size_t>(c)] +=
                  w * C_host_[(static_cast<std::size_t>(ci) * D) + c];
            }
            int k = D - 1;
            for (; k >= 0; --k) {
                if (++t[k] < meta_.k_supp) { break; }
                t[k] = 0;
            }
            if (k < 0) { break; }
        }
        return out;
    }

    /// Boolean-sum interior fill of a boundary field (the host port of
    /// egg.init.tfi._tfi_point for the all-axes-interior case; boundary rows
    /// pass through). delta/b are [n_nodes*D], block-logical row-major.
    void tfi_extend(const std::vector<real>& delta, std::vector<real>& b) const
    {
        const auto& ns = meta_.node_shape;
        auto flat_of = [&](const int (&idx)[D]) {
            int f = 0;
            for (int k = 0; k < D; ++k) { f = (f * ns[static_cast<std::size_t>(k)]) + idx[k]; }
            return f;
        };
        // Multilinear corner blend of delta over the 2^D block corners, with
        // per-axis parameter xi (idx-based, matching egg.init.tfi).
        auto corner_interp = [&](const real(&xi)[D], std::array<real, D>& out) {
            for (int c = 0; c < D; ++c) { out[static_cast<std::size_t>(c)] = 0.0_r; }
            for (int bits = 0; bits < (1 << D); ++bits) {
                real w = 1.0_r;
                int cidx[D];
                for (int k = 0; k < D; ++k) {
                    if (bits & (1 << k)) {
                        w *= xi[k];
                        cidx[k] = ns[static_cast<std::size_t>(k)] - 1;
                    } else {
                        w *= 1.0_r - xi[k];
                        cidx[k] = 0;
                    }
                }
                const int f = flat_of(cidx);
                for (int c = 0; c < D; ++c) {
                    out[static_cast<std::size_t>(c)] +=
                      w * delta[(static_cast<std::size_t>(f) * D) + c];
                }
            }
        };
        int idx[D];
        for (int k = 0; k < D; ++k) { idx[k] = 0; }
        while (true) {
            bool boundary = false;
            for (int k = 0; k < D; ++k) {
                if (idx[k] == 0 || idx[k] == ns[static_cast<std::size_t>(k)] - 1) {
                    boundary = true;
                }
            }
            const int f = flat_of(idx);
            if (boundary) {
                for (int c = 0; c < D; ++c) {
                    b[(static_cast<std::size_t>(f) * D) + c] =
                      delta[(static_cast<std::size_t>(f) * D) + c];
                }
            } else {
                real xi[D];
                for (int k = 0; k < D; ++k) {
                    xi[k] = static_cast<real>(idx[k]) /
                            static_cast<real>(ns[static_cast<std::size_t>(k)] - 1);
                }
                std::array<real, D> acc {};
                corner_interp(xi, acc);
                for (int ax = 0; ax < D; ++ax) {
                    for (int side = 0; side < 2; ++side) {
                        int fidx[D];
                        for (int k = 0; k < D; ++k) { fidx[k] = idx[k]; }
                        fidx[ax] = side == 0 ? 0 : ns[static_cast<std::size_t>(ax)] - 1;
                        const int ff = flat_of(fidx);
                        real xif[D];
                        for (int k = 0; k < D; ++k) { xif[k] = xi[k]; }
                        xif[ax] = static_cast<real>(side);
                        std::array<real, D> cface {};
                        corner_interp(xif, cface);
                        const real w = side == 0 ? 1.0_r - xi[ax] : xi[ax];
                        for (int c = 0; c < D; ++c) {
                            acc[static_cast<std::size_t>(c)] +=
                              w * (delta[(static_cast<std::size_t>(ff) * D) + c] -
                                   cface[static_cast<std::size_t>(c)]);
                        }
                    }
                }
                for (int c = 0; c < D; ++c) {
                    b[(static_cast<std::size_t>(f) * D) + c] = acc[static_cast<std::size_t>(c)];
                }
            }
            int k = D - 1;
            for (; k >= 0; --k) {
                if (++idx[k] < ns[static_cast<std::size_t>(k)]) { break; }
                idx[k] = 0;
            }
            if (k < 0) { break; }
        }
    }

    /// Rebuild the frozen frame: reproject sliding controls onto their
    /// entities, re-snap hard legs (leg-oriented normal, h floored), rebuild
    /// the reduction CSRs + penalty rows, re-extend b, and re-evaluate X.
    void wall_rebuild()
    {
        constexpr int kT = D - 1;
        C_.download(C_host_.data());
        pen_i0_h_.clear();
        pen_i1_h_.clear();
        pen_t_h_.clear();
        pen_w_h_.clear();
        // Per-control frames for the R build: tangential frame of sliding
        // controls, (foot, normal) of driven controls.
        std::vector<std::array<PtN<D>, kT>> t_of(static_cast<std::size_t>(meta_.n_ctrl));
        std::vector<int> foot_of(static_cast<std::size_t>(meta_.n_ctrl), -1);
        std::vector<PtN<D>> n_of(static_cast<std::size_t>(meta_.n_ctrl));

        for (const auto& w : hostc_.walls) {
            for (std::size_t j = 0; j < w.p0.size(); ++j) {
                const auto c0 = static_cast<std::size_t>(w.p0[j]);
                const auto c1 = static_cast<std::size_t>(w.p1[j]);
                PtN<D> pos {};
                for (int k = 0; k < D; ++k) { pos[k] = C_host_[(c0 * D) + k]; }
                const PtN<D> proj = control_detail::project_entity<D>(pos, w.tag, w.params.data());
                for (int k = 0; k < D; ++k) { C_host_[(c0 * D) + k] = proj[k]; }
                t_of[c0] = control_detail::wall_tangent_basis<D>(proj, w.tag, w.params.data());
                PtN<D> leg {};
                for (int k = 0; k < D; ++k) { leg[k] = C_host_[(c1 * D) + k] - proj[k]; }
                if (w.mode == 2) {
                    // Hard leg: entity normals carry no inward convention —
                    // orient from the current leg, snap P1 = P0 + h·n (h
                    // floored so the orientation cannot flip mid-run).
                    PtN<D> n = control_detail::frame_normal<D>(t_of[c0]);
                    real h = 0.0_r;
                    for (int k = 0; k < D; ++k) { h += n[k] * leg[k]; }
                    if (h < 0.0_r) {
                        for (int k = 0; k < D; ++k) { n[k] = -n[k]; }
                        h = -h;
                    }
                    h = std::max(h, h_min_);
                    for (int k = 0; k < D; ++k) { C_host_[(c1 * D) + k] = proj[k] + (h * n[k]); }
                    foot_of[c1] = static_cast<int>(c0);
                    n_of[c1] = n;
                } else if (w.mode == 1 && w.weight[j] > 0.0_r) {
                    real l2 = 0.0_r;
                    for (int k = 0; k < D; ++k) { l2 += leg[k] * leg[k]; }
                    if (l2 > 0.0_r) {
                        pen_i0_h_.push_back(row_of_ctrl_[c0] * D);
                        pen_i1_h_.push_back(row_of_ctrl_[c1] * D);
                        for (int kt = 0; kt < kT; ++kt) {
                            for (int k = 0; k < D; ++k) {
                                pen_t_h_.push_back(t_of[c0][static_cast<std::size_t>(kt)][k]);
                            }
                        }
                        pen_w_h_.push_back(w.weight[j] / l2);
                    }
                }
            }
        }

        // Reduction CSRs: scatter (free-space row -> q columns) and its
        // transpose. ≤ 2 entries per row; the transpose is built alongside.
        const std::size_t F = hostc_.free_idx.size();
        const std::size_t nfd = F * static_cast<std::size_t>(D);
        std::vector<int> r_off(nfd + 1, 0);
        std::vector<int> r_col;
        std::vector<real> r_coef;
        std::vector<std::vector<std::pair<int, real>>> gcols(nq_);
        for (std::size_t f = 0; f < F; ++f) {
            const auto g = static_cast<std::size_t>(hostc_.free_idx[f]);
            for (int k = 0; k < D; ++k) {
                const std::size_t r = (f * D) + static_cast<std::size_t>(k);
                auto add = [&](int colid, real coef) {
                    r_col.push_back(colid);
                    r_coef.push_back(coef);
                    gcols[static_cast<std::size_t>(colid)].emplace_back(static_cast<int>(r), coef);
                };
                if (driven_[g]) {
                    const auto p0 = static_cast<std::size_t>(foot_of[g]);
                    for (int kt = 0; kt < D - 1; ++kt) {
                        add(col_of_ctrl_[p0] + kt, t_of[p0][static_cast<std::size_t>(kt)][k]);
                    }
                    add(h_col_of_ctrl_[g], n_of[g][static_cast<std::size_t>(k)]);
                } else if (sliding_[g]) {
                    for (int kt = 0; kt < D - 1; ++kt) {
                        add(col_of_ctrl_[g] + kt, t_of[g][static_cast<std::size_t>(kt)][k]);
                    }
                } else {
                    add(col_of_ctrl_[g] + k, 1.0_r);
                }
                r_off[r + 1] = static_cast<int>(r_col.size());
            }
        }
        std::vector<int> g_off(nq_ + 1, 0);
        std::vector<int> g_row;
        std::vector<real> g_coef;
        for (std::size_t c = 0; c < nq_; ++c) {
            for (const auto& [r, coef] : gcols[c]) {
                g_row.push_back(r);
                g_coef.push_back(coef);
            }
            g_off[c + 1] = static_cast<int>(g_row.size());
        }
        r_off_.upload(r_off);
        r_col_.upload(r_col);
        r_coef_.upload(r_coef);
        g_off_.upload(g_off);
        g_row_.upload(g_row);
        g_coef_.upload(g_coef);
        if (n_pen_) {
            // The penalty row count can shrink when a leg degenerates; keep
            // the live count and upload prefixes into the fixed buffers.
            n_pen_live_ = pen_w_h_.size();
            pen_i0_.upload(pen_i0_h_);
            pen_i1_.upload(pen_i1_h_);
            pen_t_.upload(pen_t_h_);
            pen_w_.upload(pen_w_h_);
        }

        // Push the reprojected/snapped controls back, re-extend b from the
        // projected spline boundary, and re-evaluate X on the net.
        C_.upload(C_host_);
        const auto& ns = meta_.node_shape;
        std::size_t n_nodes = 1;
        for (int k = 0; k < D; ++k) { n_nodes *= static_cast<std::size_t>(ns[k]); }
        std::vector<real> delta(n_nodes * static_cast<std::size_t>(D), 0.0_r);
        int idx[D];
        for (int k = 0; k < D; ++k) { idx[k] = 0; }
        while (true) {
            bool boundary = false;
            for (int k = 0; k < D; ++k) {
                if (idx[k] == 0 || idx[k] == ns[static_cast<std::size_t>(k)] - 1) {
                    boundary = true;
                }
            }
            if (boundary) {
                int f = 0;
                for (int k = 0; k < D; ++k) { f = (f * ns[static_cast<std::size_t>(k)]) + idx[k]; }
                const std::array<real, D> xs = spline_at(idx);
                std::array<real, D> exact {};
                for (int c = 0; c < D; ++c) {
                    exact[static_cast<std::size_t>(c)] =
                      hostc_.X0[(static_cast<std::size_t>(f) * D) + c];
                }
                for (const auto& w : hostc_.walls) {
                    const int end = w.side == 0 ? 0 : ns[static_cast<std::size_t>(w.axis)] - 1;
                    if (idx[w.axis] != end) { continue; }
                    PtN<D> xp {};
                    for (int c = 0; c < D; ++c) { xp[c] = xs[static_cast<std::size_t>(c)]; }
                    const PtN<D> proj =
                      control_detail::project_entity<D>(xp, w.tag, w.params.data());
                    for (int c = 0; c < D; ++c) { exact[static_cast<std::size_t>(c)] = proj[c]; }
                }
                for (int c = 0; c < D; ++c) {
                    delta[(static_cast<std::size_t>(f) * D) + c] =
                      exact[static_cast<std::size_t>(c)] - xs[static_cast<std::size_t>(c)];
                }
            }
            int k = D - 1;
            for (; k >= 0; --k) {
                if (++idx[k] < ns[static_cast<std::size_t>(k)]) { break; }
                idx[k] = 0;
            }
            if (k < 0) { break; }
        }
        std::vector<real> b_new(n_nodes * static_cast<std::size_t>(D));
        tfi_extend(delta, b_new);
        b_.upload(b_new);
        eval_to_x();
    }

    /// Wall-penalty energy at the current host mirror (frozen frame).
    real pen_energy_host() const
    {
        constexpr std::size_t kT = D - 1;
        real e = 0.0_r;
        for (std::size_t j = 0; j < pen_w_h_.size(); ++j) {
            const auto f0 = static_cast<std::size_t>(pen_i0_h_[j]);
            const auto f1 = static_cast<std::size_t>(pen_i1_h_[j]);
            // pen rows index the free-control layout; map back to lattice ids.
            const auto c0 = static_cast<std::size_t>(hostc_.free_idx[f0 / D]);
            const auto c1 = static_cast<std::size_t>(hostc_.free_idx[f1 / D]);
            for (std::size_t kt = 0; kt < kT; ++kt) {
                real s = 0.0_r;
                for (int k = 0; k < D; ++k) {
                    s += pen_t_h_[(((j * kT) + kt) * D) + static_cast<std::size_t>(k)] *
                         (C_host_[(c1 * D) + static_cast<std::size_t>(k)] -
                          C_host_[(c0 * D) + static_cast<std::size_t>(k)]);
                }
                e += 0.5_r * pen_w_h_[j] * s * s;
            }
        }
        return e;
    }

    /// Reduced matvec (Rᵀ(Â + A_pen)R + λI)·v over the wall DOFs.
    template <class Obj> void matvec_q(Obj objective, real lam, const real* v, real* out)
    {
        const std::size_t F = hostc_.free_idx.size();
        const std::size_t nfd = F * static_cast<std::size_t>(D);
        csr_apply_kernel(q_, nfd, r_off_.data(), r_col_.data(), r_coef_.data(), v, vfull_.data());
        scatter_dir(vfull_.data());
        control_eval_kernel<D>(q_,
                               meta_,
                               off_ptrs(),
                               wt_ptrs(),
                               C_dir_.data(),
                               nullptr,
                               corr_.data());
        control_sample_hz_kernel<D>(q_, es_, X_, corr_.data(), z_samples_.data(), objective);
        gather_to(y_.data());
        pullback_to(G_scratch());
        pen_apply_kernel<D>(q_,
                            n_pen_live_,
                            pen_i0_.data(),
                            pen_i1_.data(),
                            pen_t_.data(),
                            pen_w_.data(),
                            vfull_.data(),
                            G_scratch());
        csr_apply_kernel(q_, nq_, g_off_.data(), g_row_.data(), g_coef_.data(), G_scratch(), out);
        if (lam > 0.0_r) { vec_axpy(q_, nq_, lam, v, out); }
    }

    /// Full-space scratch for the matvec pullback (dC_ doubles as it: the
    /// PCG solution lives in q space until the final scatter).
    real* G_scratch() { return dC_.data(); }

    /// Jacobi-preconditioned CG on the reduced wall system.
    template <class Obj>
    std::pair<int, real> pcg_solve_q(Obj objective, real lam, int max_iter, real rtol)
    {
        std::array<real, 8> s {};
        vec_fill(q_, qdq_.data(), nq_, 0.0_r);
        q_.parallel_for(sycl::range<1>(nq_),
                        [r = qr_.data(), g = qG_.data()](sycl::id<1> i) { r[i[0]] = -g[i[0]]; });
        vec_dot(q_, nq_, qr_.data(), qr_.data(), scal_.data());
        download(1, s);
        const real b_norm2 = s[0];
        if (b_norm2 == 0.0_r) { return {0, 0.0_r}; }
        const real stop2 = rtol * rtol * b_norm2;
        real r_norm2 = b_norm2;

        qdiag_apply_kernel(q_, nq_, qdiag_.data(), lam, qr_.data(), qz_.data());
        q_.memcpy(qp_.data(), qz_.data(), nq_ * sizeof(real));
        vec_dot(q_, nq_, qr_.data(), qz_.data(), scal_.data());
        download(1, s);
        real rz = s[0];

        int it = 0;
        for (; it < max_iter; ++it) {
            matvec_q(objective, lam, qp_.data(), qap_.data());
            vec_dot(q_, nq_, qp_.data(), qap_.data(), scal_.data());
            download(1, s);
            const real p_ap = s[0];
            if (!(p_ap > 0.0_r)) { break; }
            const real a = rz / p_ap;
            vec_axpy(q_, nq_, a, qp_.data(), qdq_.data());
            vec_axpy(q_, nq_, -a, qap_.data(), qr_.data());
            vec_dot(q_, nq_, qr_.data(), qr_.data(), scal_.data());
            download(1, s);
            r_norm2 = s[0];
            if (r_norm2 <= stop2) {
                ++it;
                break;
            }
            qdiag_apply_kernel(q_, nq_, qdiag_.data(), lam, qr_.data(), qz_.data());
            vec_dot(q_, nq_, qr_.data(), qz_.data(), scal_.data());
            download(1, s);
            const real rz_new = s[0];
            vec_xpay(q_, nq_, qz_.data(), rz_new / rz, qp_.data());
            rz = rz_new;
        }
        return {it, sycl::sqrt(r_norm2 / b_norm2)};
    }

    /// The wall-mode outer loop: per-iteration frozen-frame rebuild, reduced
    /// gradient, PCG over the wall DOFs, α-ladder on the COMPOSED energy
    /// (fine + penalty; the penalty is quadratic in α, added analytically).
    template <class Obj> ControlReport run_walls(Obj objective, const ControlOptions& opts)
    {
        constexpr real kLamFloorScale = 1e-8_r;
        ControlReport rep;
        const std::size_t F = hostc_.free_idx.size();
        const std::size_t nfd = F * static_cast<std::size_t>(D);
        real lam = opts.lm0;
        real gmax0 = 0.0_r;
        std::array<real, 8> s {};
        constexpr std::size_t kT = D - 1;
        std::vector<real> pen2_h(2 * n_pen_ * kT);

        for (int outer = 0; outer < opts.n_outer; ++outer) {
            // Frozen frame (moves the state slightly: reprojection + snap +
            // b re-extension — so the baseline is re-evaluated after it).
            wall_rebuild();
            reduce_energy_mindet<D>(q_, es_, X_, scal_.data(), scal_.data() + 1, objective);
            download(2, s);
            real e0 = s[0] + pen_energy_host();
            real md0 = s[1];
            if (rep.energies.empty()) {
                rep.energies.push_back(e0);
                rep.mindets.push_back(md0);
            }

            // Reduced gradient Gq = Rᵀ(M_fᵀ g + A_pen·C_free).
            control_sample_g_kernel<D>(q_, es_, X_, z_samples_.data(), objective);
            gather_to(y_.data());
            scatter_directional_grad<D>(q_, es_.dir, X_, y_.data());
            pullback_to(G_.data());
            gather_free_kernel<D>(q_, meta_.n_free, free_idx_.data(), C_.data(), cfree_.data());
            pen_apply_kernel<D>(q_,
                                n_pen_live_,
                                pen_i0_.data(),
                                pen_i1_.data(),
                                pen_t_.data(),
                                pen_w_.data(),
                                cfree_.data(),
                                G_.data());
            csr_apply_kernel(q_,
                             nq_,
                             g_off_.data(),
                             g_row_.data(),
                             g_coef_.data(),
                             G_.data(),
                             qG_.data());
            vec_amax(q_, nq_, qG_.data(), scal_.data());
            download(1, s);
            const real gmax = s[0];
            if (gmax < opts.grad_tol) {
                rep.converged = true;
                break;
            }
            if (outer == 0) { gmax0 = gmax; }
            const real pcg_tol =
              opts.pcg_forcing ? std::max(opts.pcg_rtol, std::min(0.5_r, sycl::sqrt(gmax / gmax0)))
                               : opts.pcg_rtol;

            // Preconditioner: per-control blocks reduced to a scalar diagonal.
            control_gather_hess_kernel<D>(q_,
                                          es_,
                                          mem_nodes_.data(),
                                          mem_off_.data(),
                                          mem_sid_.data(),
                                          mem_role_.data(),
                                          n_member_,
                                          X_,
                                          h_node_.data(),
                                          objective);
            control_precond_build_kernel<D>(q_,
                                            meta_,
                                            sup_off_ptrs(),
                                            sup_node_ptrs(),
                                            sup_w_ptrs(),
                                            free_midx_.data(),
                                            h_node_.data(),
                                            prec_.data());
            qdiag_build_kernel<D>(q_,
                                  nq_,
                                  g_off_.data(),
                                  g_row_.data(),
                                  g_coef_.data(),
                                  prec_.data(),
                                  qdiag_.data());
            vec_trace_sum<D>(q_,
                             static_cast<std::size_t>(meta_.n_free),
                             prec_.data(),
                             scal_.data());
            download(1, s);
            const real lam_floor = kLamFloorScale * s[0] / static_cast<real>(nq_);

            bool accepted = false;
            real alpha = 1.0_r;
            real e_new = e0;
            real md_new = md0;
            for (int lt = 0; lt < opts.lm_tries && !accepted; ++lt) {
                const auto [pcg_its, relres] =
                  pcg_solve_q(objective, lam, opts.pcg_max_iter, pcg_tol);
                rep.pcg_iters.push_back(static_cast<real>(pcg_its));
                rep.pcg_relres.push_back(relres);

                // dC = R dq, corr = M dC; per-leg penalty components for the
                // analytic α-quadratic.
                csr_apply_kernel(q_,
                                 nfd,
                                 r_off_.data(),
                                 r_col_.data(),
                                 r_coef_.data(),
                                 qdq_.data(),
                                 dC_.data());
                scatter_dir(dC_.data());
                control_eval_kernel<D>(q_,
                                       meta_,
                                       off_ptrs(),
                                       wt_ptrs(),
                                       C_dir_.data(),
                                       nullptr,
                                       corr_.data());
                if (n_pen_live_) {
                    pen_residual_kernel<D>(q_,
                                           n_pen_live_,
                                           pen_i0_.data(),
                                           pen_i1_.data(),
                                           pen_t_.data(),
                                           cfree_.data(),
                                           dC_.data(),
                                           pen2_.data());
                    q_.wait();
                    q_.memcpy(pen2_h.data(), pen2_.data(), 2 * n_pen_live_ * kT * sizeof(real))
                      .wait();
                }
                auto pen_at = [&](real a) {
                    real e = 0.0_r;
                    for (std::size_t j = 0; j < n_pen_live_; ++j) {
                        for (std::size_t kt = 0; kt < kT; ++kt) {
                            const std::size_t ei = (j * kT) + kt;
                            const real v = pen2_h[ei * 2] + (a * pen2_h[(ei * 2) + 1]);
                            e += 0.5_r * pen_w_h_[j] * v * v;
                        }
                    }
                    return e;
                };

                alpha = 1.0_r;
                while (alpha >= opts.alpha_min) {
                    detail::reduce_linesearch_pair<D>(q_,
                                                      es_,
                                                      X_,
                                                      corr_.data(),
                                                      alpha,
                                                      alpha * 0.5_r,
                                                      scal_.data(),
                                                      objective);
                    download(4, s);
                    bool ok = false;
                    for (int t = 0; t < 2; ++t) {
                        const real a = t == 0 ? alpha : alpha * 0.5_r;
                        const real e = s[(2 * t) + 0] + pen_at(a);
                        const real md = s[(2 * t) + 1];
                        if (a >= opts.alpha_min && sycl::isfinite(e) && e <= e0 + tol::energy &&
                            objective.accept_mindet(md)) {
                            alpha = a;
                            e_new = e;
                            md_new = md;
                            ok = true;
                            break;
                        }
                    }
                    if (ok) {
                        accepted = true;
                        break;
                    }
                    alpha *= 0.25_r;
                }
                if (!accepted) { lam = std::max(10.0_r * lam, lam_floor); }
            }
            if (!accepted) { break; }

            vec_axpy(q_, x_reals(), alpha, corr_.data(), X_);
            control_scatter_free_kernel<D>(q_,
                                           meta_.n_free,
                                           free_idx_.data(),
                                           dC_.data(),
                                           C_.data(),
                                           alpha);
            rep.energies.push_back(e_new);
            rep.mindets.push_back(md_new);
            rep.alphas.push_back(alpha);
            rep.iters += 1;
            if (alpha == 1.0_r) { lam /= 3.0_r; }
        }
        // Final consistency pass so the reported state is exactly on the
        // frame (reprojected controls, fresh b).
        wall_rebuild();
        q_.wait();
        return rep;
    }

    sycl::queue q_;
    StencilViewT<D> es_;
    real* X_;  // the session's packed coordinate buffer (not owned)

    ControlNetMetaT<D> meta_ {};
    std::array<UsmBuffer<int>, D> off_;
    std::array<UsmBuffer<real>, D> wt_;
    std::array<UsmBuffer<int>, D> sup_off_;
    std::array<UsmBuffer<int>, D> sup_node_;
    std::array<UsmBuffer<real>, D> sup_w_;
    UsmBuffer<real> C_, b_, C_dir_;
    UsmBuffer<int> free_idx_, free_midx_;

    std::size_t n_member_ = 0;
    UsmBuffer<int> mem_nodes_, mem_off_, mem_sid_, mem_role_;

    UsmBuffer<real> corr_, y_, h_node_, z_samples_, prec_;
    UsmBuffer<real> G_, dC_, pcg_r_, pcg_z_, pcg_p_, pcg_ap_;
    UsmBuffer<real> scal_;

    // Wall machinery (engaged only when the wire carries walls).
    bool has_walls_ = false;
    ControlNetHostT<D> hostc_;  // host basis tables, X0, wall specs
    std::vector<real> C_host_;  // host mirror of the control lattice
    std::vector<int> row_of_ctrl_;
    std::vector<int> col_of_ctrl_, h_col_of_ctrl_;
    std::vector<std::uint8_t> sliding_, driven_;
    std::size_t nq_ = 0;
    real h_min_ = 0.0_r;
    UsmBuffer<int> r_off_, r_col_, g_off_, g_row_;
    UsmBuffer<real> r_coef_, g_coef_;
    std::size_t n_pen_ = 0;       // penalty-leg capacity (buffer size)
    std::size_t n_pen_live_ = 0;  // live rows this frame (degenerate legs drop)
    UsmBuffer<int> pen_i0_, pen_i1_;
    UsmBuffer<real> pen_t_, pen_w_, pen2_;
    std::vector<int> pen_i0_h_, pen_i1_h_;
    std::vector<real> pen_t_h_, pen_w_h_;
    UsmBuffer<real> qG_, qdq_, qr_, qz_, qp_, qap_, qdiag_, cfree_, vfull_;
};

/// Multi-block control-net session: per-block nets over one stacked control
/// vector, a global reduction CSR dC = R·dq built host-side by
/// egg/smoothing/control_topology.py (C0 sharing, S/T elimination, twists,
/// sliding/hard frames), and generic homogeneous penalty rows (soft C2, fan
/// C1, seam orthogonality) composed into both the GN system and the
/// line-search energy. Frozen-frame updates (seam-ortho tangents, wall
/// frames, b re-extension) happen in Python between run calls through the
/// set_reduction / set_penalty / set_b / set_c entry points — the device
/// never re-derives topology.
///
/// Cross-block coupling: every net eval is followed by the fused halo +
/// shared-node broadcast (ghost shells and non-owner copies see the fresh
/// spline surface — for both X and directions), and every node-space
/// gradient/Hessian gather is followed by the alias-group sum (the adjoint
/// of that copy: contributions landing on ghost/copy slots are folded back
/// so each block's pullback sees the full sample set).
template <int D> class ControlTopoSessionT
{
  public:
    /// @param alias_pairs (src, dst) node-slot pairs (node units, not reals)
    ///        of every halo/share/fan copy; grouped by union-find here.
    ControlTopoSessionT(sycl::queue q,
                        const BlockLayout<D>& layout,
                        const EnergyStencilHostT<D>& es_host,
                        const StencilViewT<D>& es,
                        const ControlTopoHostT<D>& host,
                        real* X,
                        HaloViewPtrs halo,
                        const std::vector<std::pair<int, int>>& alias_pairs) :
        q_(std::move(q)), es_(es), X_(X), halo_(halo)
    {
        nets_.resize(host.nets.size());
        n_stack_ = 0;
        for (std::size_t ni = 0; ni < host.nets.size(); ++ni) {
            const auto& hn = host.nets[ni];
            auto& net = nets_[ni];
            const auto b = static_cast<std::size_t>(hn.block);
            const auto shape = layout.interior_shape(b);
            const auto nstr = layout.node_strides(b);
            net.c_off = n_stack_;
            net.meta.k_supp = hn.k_supp;
            net.meta.n_nodes = 1;
            net.meta.n_ctrl = 1;
            for (int k = 0; k < D; ++k) {
                const auto ks = static_cast<std::size_t>(k);
                net.meta.node_shape[ks] = static_cast<int>(shape[ks]);
                net.meta.ctrl_shape[ks] = hn.ctrl_shape[ks];
                net.meta.node_stride[ks] = static_cast<int>(nstr[ks] / static_cast<std::size_t>(D));
                net.meta.n_nodes *= net.meta.node_shape[ks];
                net.meta.n_ctrl *= net.meta.ctrl_shape[ks];
            }
            int acc = 1;
            for (int k = D - 1; k >= 0; --k) {
                net.meta.ctrl_stride[static_cast<std::size_t>(k)] = acc;
                acc *= net.meta.ctrl_shape[static_cast<std::size_t>(k)];
            }
            net.meta.base_node =
              static_cast<int>(layout.interior_origin_offset(b) / static_cast<std::size_t>(D));
            net.meta.n_free = net.meta.n_ctrl;  // freedom lives in R, not here
            n_stack_ += static_cast<std::size_t>(net.meta.n_ctrl);

            for (int k = 0; k < D; ++k) {
                const auto ks = static_cast<std::size_t>(k);
                net.off[ks] = UsmBuffer<int> {q_, hn.off[ks]};
                net.wt[ks] = UsmBuffer<real> {q_, hn.wt[ks]};
            }
            net.b = UsmBuffer<real> {q_, hn.b};

            // Per-axis support CSR (control -> nodes), inverted host-side.
            for (int k = 0; k < D; ++k) {
                const auto ks = static_cast<std::size_t>(k);
                const int n_ctrl_k = hn.ctrl_shape[ks];
                const int n_node_k = net.meta.node_shape[ks];
                std::vector<std::vector<std::pair<int, real>>> lists(
                  static_cast<std::size_t>(n_ctrl_k));
                for (int i = 0; i < n_node_k; ++i) {
                    const int a0 = hn.off[ks][static_cast<std::size_t>(i)];
                    for (int t = 0; t < hn.k_supp; ++t) {
                        const real w = hn.wt[ks][static_cast<std::size_t>((i * hn.k_supp) + t)];
                        if (w != 0.0_r) {
                            lists[static_cast<std::size_t>(a0 + t)].emplace_back(i, w);
                        }
                    }
                }
                std::vector<int> soff(static_cast<std::size_t>(n_ctrl_k) + 1, 0);
                std::vector<int> snode;
                std::vector<real> sw;
                for (int a = 0; a < n_ctrl_k; ++a) {
                    for (const auto& [i, w] : lists[static_cast<std::size_t>(a)]) {
                        snode.push_back(i);
                        sw.push_back(w);
                    }
                    soff[static_cast<std::size_t>(a) + 1] = static_cast<int>(snode.size());
                }
                net.sup_off[ks] = UsmBuffer<int> {q_, soff};
                net.sup_node[ks] = UsmBuffer<int> {q_, snode};
                net.sup_w[ks] = UsmBuffer<real> {q_, sw};
            }
        }
        if (host.C0.size() != n_stack_ * static_cast<std::size_t>(D)) {
            throw std::invalid_argument("control topo: C0 size != total controls * D");
        }
        C_ = UsmBuffer<real> {q_, host.C0};

        // Batched-launch tables: one NetRefT per net plus the node/control
        // prefix arrays, so eval/pullback/precond/db run as single launches
        // over the concatenated item spaces instead of one launch per net
        // (the per-net launches made the GN matvec dispatch-bound on GPU).
        // Every pointer captured here is stable: UsmBuffer::upload rewrites
        // buffers in place.
        {
            std::vector<NetRefT<D>> refs(nets_.size());
            std::vector<int> npfx {0};
            std::vector<int> cpfx {0};
            total_nodes_ = 0;
            for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
                const auto& net = nets_[ni];
                auto& r = refs[ni];
                r.meta = net.meta;
                r.off = ptrs(net.off);
                r.wt = ptrs(net.wt);
                r.sup_off = ptrs(net.sup_off);
                r.sup_node = ptrs(net.sup_node);
                r.sup_w = ptrs(net.sup_w);
                r.b = net.b.data();
                r.c_off = static_cast<int>(net.c_off);
                r.node_off = static_cast<int>(total_nodes_);
                total_nodes_ += static_cast<std::size_t>(net.meta.n_nodes);
                npfx.push_back(static_cast<int>(total_nodes_));
                cpfx.push_back(static_cast<int>(net.c_off) + net.meta.n_ctrl);
            }
            net_refs_ = UsmBuffer<NetRefT<D>> {q_, refs};
            node_pfx_ = UsmBuffer<int> {q_, npfx};
            ctrl_pfx_ = UsmBuffer<int> {q_, cpfx};
        }

        // Sample-membership CSR over the whole stencil (see the single-block
        // constructor; std::map keeps fp reduction order deterministic).
        {
            std::map<int, std::vector<std::pair<int, int>>> mem;
            const auto P = es_host.num_samples;
            for (std::size_t p = 0; p < P; ++p) {
                mem[es_host.gc[p]].emplace_back(static_cast<int>(p), 0);
                for (int k = 0; k < D; ++k) {
                    mem[es_host.gn[k][p]].emplace_back(static_cast<int>(p), k + 1);
                }
            }
            std::vector<int> nodes;
            std::vector<int> moff {0};
            std::vector<int> sid;
            std::vector<int> role;
            for (const auto& [node, entries] : mem) {
                nodes.push_back(node);
                for (const auto& [p, r] : entries) {
                    sid.push_back(p);
                    role.push_back(r);
                }
                moff.push_back(static_cast<int>(sid.size()));
            }
            n_member_ = nodes.size();
            mem_nodes_ = UsmBuffer<int> {q_, nodes};
            mem_off_ = UsmBuffer<int> {q_, moff};
            mem_sid_ = UsmBuffer<int> {q_, sid};
            mem_role_ = UsmBuffer<int> {q_, role};
        }

        // Alias groups (union-find over the copy pairs), OWNER FIRST in each
        // group: the owner is the member that is never a copy destination
        // (the broadcast fills every other member from it), and the reduce
        // kernel credits the whole group's contribution to it.
        {
            std::map<int, int> parent;
            std::set<int> is_dst;
            std::function<int(int)> find = [&](int x) {
                auto it = parent.find(x);
                if (it == parent.end()) {
                    parent[x] = x;
                    return x;
                }
                if (it->second == x) { return x; }
                const int r = find(it->second);
                parent[x] = r;
                return r;
            };
            for (const auto& [s, d] : alias_pairs) {
                const int rs = find(s);
                const int rd = find(d);
                if (rs != rd) { parent[rd] = rs; }
                is_dst.insert(d);
            }
            std::map<int, std::vector<int>> groups;
            for (const auto& [slot, _] : parent) { groups[find(slot)].push_back(slot); }
            std::vector<int> g_off {0};
            std::vector<int> g_slot;
            for (auto& [root, members] : groups) {
                if (members.size() < 2) { continue; }
                std::sort(members.begin(), members.end());
                int owner = -1;
                for (int m : members) {
                    if (!is_dst.contains(m)) {
                        owner = m;
                        break;
                    }
                }
                if (owner < 0) {
                    throw std::invalid_argument(
                      "control topo: an alias group has no owner slot (every copy "
                      "is a destination); the halo tables are inconsistent");
                }
                g_slot.push_back(owner);
                for (int m : members) {
                    if (m != owner) { g_slot.push_back(m); }
                }
                g_off.push_back(static_cast<int>(g_slot.size()));
            }
            n_alias_ = g_off.size() - 1;
            alias_off_ = UsmBuffer<int> {q_, g_off};
            alias_slot_ = UsmBuffer<int> {q_, g_slot.empty() ? std::vector<int> {0} : g_slot};
        }

        // Work buffers.
        const std::size_t total_reals = layout.total_reals();
        const std::size_t total_nodes = total_reals / static_cast<std::size_t>(D);
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        corr_ = UsmBuffer<real> {q_, std::vector<real>(total_reals, 0.0_r)};
        y_ = UsmBuffer<real> {q_, std::vector<real>(total_reals, 0.0_r)};
        h_node_ =
          UsmBuffer<real> {q_,
                           std::vector<real>(total_nodes * static_cast<std::size_t>(D * D), 0.0_r)};
        z_samples_ = UsmBuffer<real> {q_, es_.num_samples * static_cast<std::size_t>(dim::vecT(D))};
        vfull_ = UsmBuffer<real> {q_, nsd};
        wfull_ = UsmBuffer<real> {q_, nsd};
        dCfull_ = UsmBuffer<real> {q_, nsd};
        prec_ = UsmBuffer<real> {q_, n_stack_ * static_cast<std::size_t>(D * D)};
        scal_ = UsmBuffer<real> {q_, 8};

        set_reduction(host.nq, host.r_off, host.r_col, host.r_coef);
        set_penalty(host.p_off, host.p_col, host.p_coef, host.p_w);

        // Sliding-wall frame loop: keep the host tables the per-iteration
        // rebuild needs (basis tables, X0 anchors, entity records, CSR
        // rewrite lists). No Python in the loop.
        has_slide_ = !host.slide.empty();
        if (has_slide_) {
            host_ = host;
            C_host_ = host.C0;
            b_host_.resize(nets_.size());
            for (std::size_t ni = 0; ni < nets_.size(); ++ni) { b_host_[ni] = host.nets[ni].b; }
            const auto nsd = static_cast<int>(n_stack_) * D;
            for (const auto& s : host.slide) {
                if (s.rep_row < 0 || s.rep_row >= static_cast<int>(n_stack_)) {
                    throw std::invalid_argument("control topo: slide rep_row out of range");
                }
                if (s.members.size() != s.member_coef.size() || s.r_entry.size() != s.comp.size() ||
                    s.comp.size() != s.kt.size() || s.kt.size() != s.base.size()) {
                    throw std::invalid_argument("control topo: slide list length mismatch");
                }
                for (int e : s.r_entry) {
                    if (e < 0 || std::cmp_greater_equal(e, r_coef_h_.size())) {
                        throw std::invalid_argument("control topo: slide r_entry out of range");
                    }
                }
                for (int m : s.members) {
                    if (m < 0 || m >= static_cast<int>(n_stack_)) {
                        throw std::invalid_argument("control topo: slide member out of range");
                    }
                }
                (void)nsd;
            }
            if (host.seam_face.size() != nets_.size()) {
                throw std::invalid_argument("control topo: seam_face flags must cover every net");
            }
            for (const auto& w : host.wall_faces) {
                if (w.block < 0 || std::cmp_greater_equal(w.block, nets_.size()) || w.axis < 0 ||
                    w.axis >= D || (w.side != 0 && w.side != 1)) {
                    throw std::invalid_argument("control topo: wall face out of range");
                }
            }

            // Wall-face node lists for the db/dC frame linearization: every
            // boundary node on a wall face of its net, excluding seam-face
            // nodes (exactly the nodes rebuild_wall_b projects). The node
            // sets are frame-invariant; only the projectors change per frame.
            dbn_flat_h_.resize(nets_.size());
            dbn_idx_of_flat_.resize(nets_.size());
            dbn_idx_d_.resize(nets_.size());
            std::size_t max_block_reals = 0;
            for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
                const auto& m = nets_[ni].meta;
                const auto& seam = host_.seam_face[ni];
                bool has_wall = false;
                for (const auto& w : host_.wall_faces) {
                    if (std::cmp_equal(w.block, ni)) { has_wall = true; }
                }
                if (!has_wall) { continue; }
                max_block_reals =
                  std::max(max_block_reals,
                           static_cast<std::size_t>(m.n_nodes) * static_cast<std::size_t>(D));
                dbn_idx_of_flat_[ni].assign(static_cast<std::size_t>(m.n_nodes), -1);
                int idx[D];
                for (int k = 0; k < D; ++k) { idx[k] = 0; }
                while (true) {
                    bool on_seam = false;
                    bool on_wall = false;
                    for (int k = 0; k < D; ++k) {
                        const int side =
                          idx[k] == 0
                            ? 0
                            : (idx[k] == m.node_shape[static_cast<std::size_t>(k)] - 1 ? 1 : -1);
                        if (side >= 0 && seam[static_cast<std::size_t>((2 * k) + side)]) {
                            on_seam = true;
                        }
                    }
                    for (const auto& w : host_.wall_faces) {
                        if (std::cmp_not_equal(w.block, ni)) { continue; }
                        const int end =
                          w.side == 0 ? 0 : m.node_shape[static_cast<std::size_t>(w.axis)] - 1;
                        if (idx[w.axis] == end) { on_wall = true; }
                    }
                    if (on_wall && !on_seam) {
                        int f = 0;
                        for (int k = 0; k < D; ++k) {
                            f = (f * m.node_shape[static_cast<std::size_t>(k)]) + idx[k];
                        }
                        dbn_idx_of_flat_[ni][static_cast<std::size_t>(f)] =
                          static_cast<int>(dbn_flat_h_[ni].size());
                        dbn_flat_h_[ni].push_back(f);
                    }
                    int k = D - 1;
                    for (; k >= 0; --k) {
                        if (++idx[k] < m.node_shape[static_cast<std::size_t>(k)]) { break; }
                        idx[k] = 0;
                    }
                    if (k < 0) { break; }
                }
                if (!dbn_flat_h_[ni].empty()) {
                    dbn_idx_d_[ni] = UsmBuffer<int> {q_, dbn_idx_of_flat_[ni]};
                }
            }
            if (max_block_reals) {
                db_delta_ = UsmBuffer<real> {q_, std::vector<real>(max_block_reals, 0.0_r)};
            }

            // Concatenated wall-face lists for the batched db kernels: face
            // -> net map, flat node ids, and one projector arena sliced per
            // net (rebuild_wall_b writes each net's frame projectors into
            // its slice through the per-net local face indices).
            {
                std::vector<int> fflat;
                std::vector<int> fnet;
                std::vector<int> fhas(nets_.size(), 0);
                face_pfx_h_.assign(nets_.size() + 1, 0);
                for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
                    face_pfx_h_[ni] = fflat.size();
                    fhas[ni] = dbn_flat_h_[ni].empty() ? 0 : 1;
                    for (const int f : dbn_flat_h_[ni]) {
                        fflat.push_back(f);
                        fnet.push_back(static_cast<int>(ni));
                    }
                }
                face_pfx_h_[nets_.size()] = fflat.size();
                n_face_total_ = fflat.size();
                dbn_net_ = UsmBuffer<int> {q_, fnet.empty() ? std::vector<int> {0} : fnet};
                dbn_flat_all_ = UsmBuffer<int> {q_, fflat.empty() ? std::vector<int> {0} : fflat};
                dbn_has_ = UsmBuffer<int> {q_, fhas};
                dbn_P_all_ =
                  UsmBuffer<real> {q_,
                                   std::vector<real>(std::max<std::size_t>(1, n_face_total_) *
                                                       static_cast<std::size_t>(D * D),
                                                     0.0_r)};
                if (n_face_total_) {
                    db_delta_all_ = UsmBuffer<real> {
                      q_,
                      std::vector<real>(total_nodes_ * static_cast<std::size_t>(D), 0.0_r)};
                }
            }

            // Device frame tables: the arena, packed wall-face and sliding-
            // root records, the per-root CSR entry / member lists, per-net
            // X0 anchors + seam flags, and the snapshot buffers. Everything
            // the per-frame kernels read; uploaded once.
            arena_d_ =
              UsmBuffer<real> {q_, host_.arena.empty() ? std::vector<real> {0.0_r} : host_.arena};
            n_wf_ = host_.wall_faces.size();
            {
                std::vector<int> wb, wa, ws, wt;
                wf_stride_ = 1;
                for (const auto& w : host_.wall_faces) {
                    wf_stride_ = std::max(wf_stride_, w.params.size());
                }
                std::vector<real> wp(std::max<std::size_t>(1, n_wf_ * wf_stride_), 0.0_r);
                for (std::size_t w = 0; w < n_wf_; ++w) {
                    const auto& wf = host_.wall_faces[w];
                    wb.push_back(wf.block);
                    wa.push_back(wf.axis);
                    ws.push_back(wf.side);
                    wt.push_back(wf.tag);
                    std::copy(wf.params.begin(), wf.params.end(), wp.begin() + (w * wf_stride_));
                }
                if (wb.empty()) { wb = wa = ws = wt = {0}; }
                wf_block_ = UsmBuffer<int> {q_, wb};
                wf_axis_ = UsmBuffer<int> {q_, wa};
                wf_side_ = UsmBuffer<int> {q_, ws};
                wf_tag_ = UsmBuffer<int> {q_, wt};
                wf_params_ = UsmBuffer<real> {q_, wp};
            }
            {
                const std::size_t nsl = host_.slide.size();
                std::vector<int> tag, rep, eoff {0}, rent, kt, comp, mem, mroot;
                std::vector<real> base, mcoef;
                sl_stride_ = 1;
                for (const auto& s : host_.slide) {
                    sl_stride_ = std::max(sl_stride_, s.params.size());
                }
                std::vector<real> sp(std::max<std::size_t>(1, nsl * sl_stride_), 0.0_r);
                for (std::size_t si = 0; si < nsl; ++si) {
                    const auto& s = host_.slide[si];
                    tag.push_back(s.tag);
                    rep.push_back(s.rep_row);
                    std::copy(s.params.begin(), s.params.end(), sp.begin() + (si * sl_stride_));
                    rent.insert(rent.end(), s.r_entry.begin(), s.r_entry.end());
                    kt.insert(kt.end(), s.kt.begin(), s.kt.end());
                    comp.insert(comp.end(), s.comp.begin(), s.comp.end());
                    base.insert(base.end(), s.base.begin(), s.base.end());
                    eoff.push_back(static_cast<int>(rent.size()));
                    for (std::size_t mi = 0; mi < s.members.size(); ++mi) {
                        mem.push_back(s.members[mi]);
                        mroot.push_back(static_cast<int>(si));
                        mcoef.push_back(s.member_coef[mi]);
                    }
                }
                n_members_ = mem.size();
                auto nz_i = [](std::vector<int>& v) {
                    if (v.empty()) { v = {0}; }
                };
                auto nz_r = [](std::vector<real>& v) {
                    if (v.empty()) { v = {0.0_r}; }
                };
                nz_i(tag);
                nz_i(rep);
                nz_i(rent);
                nz_i(kt);
                nz_i(comp);
                nz_i(mem);
                nz_i(mroot);
                nz_r(base);
                nz_r(mcoef);
                sl_tag_ = UsmBuffer<int> {q_, tag};
                sl_rep_ = UsmBuffer<int> {q_, rep};
                sl_entry_off_ = UsmBuffer<int> {q_, eoff};
                sl_r_entry_ = UsmBuffer<int> {q_, rent};
                sl_kt_ = UsmBuffer<int> {q_, kt};
                sl_comp_ = UsmBuffer<int> {q_, comp};
                sl_base_ = UsmBuffer<real> {q_, base};
                sl_params_ = UsmBuffer<real> {q_, sp};
                sl_members_ = UsmBuffer<int> {q_, mem};
                sl_mem_root_ = UsmBuffer<int> {q_, mroot};
                sl_mem_coef_ = UsmBuffer<real> {q_, mcoef};
                fp_buf_ =
                  UsmBuffer<real> {q_,
                                   std::vector<real>(std::max<std::size_t>(1, nsl * 2 * D), 0.0_r)};
            }
            g_entry_of_r_d_ =
              UsmBuffer<int> {q_, g_entry_of_r_.empty() ? std::vector<int> {0} : g_entry_of_r_};
            C_snap_ = UsmBuffer<real> {q_, std::vector<real>(nsd, 0.0_r)};
            b_snap_.resize(nets_.size());
            X0_d_.resize(nets_.size());
            seam_d_.resize(nets_.size());
            net_has_wall_.assign(nets_.size(), 0);
            for (const auto& w : host_.wall_faces) {
                net_has_wall_[static_cast<std::size_t>(w.block)] = 1;
            }
            for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
                if (!net_has_wall_[ni]) { continue; }
                b_snap_[ni] =
                  UsmBuffer<real> {q_, std::vector<real>(host_.nets[ni].b.size(), 0.0_r)};
                X0_d_[ni] = UsmBuffer<real> {q_, host_.nets[ni].X0};
                std::vector<int> fl(2 * static_cast<std::size_t>(D), 0);
                for (std::size_t j = 0; j < fl.size(); ++j) {
                    fl[j] = static_cast<int>(host_.seam_face[ni][j]);
                }
                seam_d_[ni] = UsmBuffer<int> {q_, fl};
            }
        }
    }

    /// Replace the reduction CSR (same stacked-component row space; column
    /// count may change when frames add/remove reduced DOFs).
    void set_reduction(std::size_t nq,
                       const std::vector<int>& off,
                       const std::vector<int>& col,
                       const std::vector<real>& coef)
    {
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        if (off.size() != nsd + 1) {
            throw std::invalid_argument("control topo: reduction row count != total controls * D");
        }
        for (int c : col) {
            if (c < 0 || std::cmp_greater_equal(c, nq)) {
                throw std::invalid_argument("control topo: reduction column out of range");
            }
        }
        nq_ = nq;
        r_off_ = UsmBuffer<int> {q_, off};
        r_col_ = UsmBuffer<int> {q_, col.empty() ? std::vector<int> {0} : col};
        r_coef_ = UsmBuffer<real> {q_, coef.empty() ? std::vector<real> {0.0_r} : coef};
        // Transpose (gather) CSR, rows ascending within each column. The
        // r-entry -> g-entry map lets the frame loop rewrite BOTH value
        // arrays in place (the structure is frame-invariant).
        std::vector<std::vector<std::tuple<int, real, int>>> gcols(nq_);
        for (std::size_t r = 0; r < nsd; ++r) {
            for (int e = off[r]; e < off[r + 1]; ++e) {
                gcols[static_cast<std::size_t>(col[static_cast<std::size_t>(e)])].emplace_back(
                  static_cast<int>(r),
                  coef[static_cast<std::size_t>(e)],
                  e);
            }
        }
        std::vector<int> g_off(nq_ + 1, 0);
        std::vector<int> g_row;
        std::vector<real> g_coef;
        g_entry_of_r_.assign(col.size(), -1);
        for (std::size_t c = 0; c < nq_; ++c) {
            for (const auto& [r, v, e] : gcols[c]) {
                g_entry_of_r_[static_cast<std::size_t>(e)] = static_cast<int>(g_row.size());
                g_row.push_back(r);
                g_coef.push_back(v);
            }
            g_off[c + 1] = static_cast<int>(g_row.size());
        }
        r_coef_h_ = coef;
        g_coef_h_ = g_coef;
        g_off_ = UsmBuffer<int> {q_, g_off};
        g_row_ = UsmBuffer<int> {q_, g_row.empty() ? std::vector<int> {0} : g_row};
        g_coef_ = UsmBuffer<real> {q_, g_coef.empty() ? std::vector<real> {0.0_r} : g_coef};
        // The frame kernels rewrite both value arrays through this map; keep
        // the device copy in step when the reduction is replaced.
        if (has_slide_) {
            g_entry_of_r_d_ =
              UsmBuffer<int> {q_, g_entry_of_r_.empty() ? std::vector<int> {0} : g_entry_of_r_};
        }
        qG_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qdq_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qr_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qz_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qp_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qap_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
        qdiag_ = UsmBuffer<real> {q_, nq_ ? nq_ : 1};
    }

    /// Replace the penalty rows (homogeneous: energy ½ Σ_j w_j (P_j C)²).
    void set_penalty(const std::vector<int>& off,
                     const std::vector<int>& col,
                     const std::vector<real>& coef,
                     const std::vector<real>& w)
    {
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        n_pen_ = off.empty() ? 0 : off.size() - 1;
        if (w.size() != n_pen_) {
            throw std::invalid_argument("control topo: penalty weight count != row count");
        }
        for (int c : col) {
            if (c < 0 || std::cmp_greater_equal(c, nsd)) {
                throw std::invalid_argument("control topo: penalty column out of range");
            }
        }
        if (n_pen_ == 0) {
            p_off_ = UsmBuffer<int> {};
            return;
        }
        p_off_ = UsmBuffer<int> {q_, off};
        p_col_ = UsmBuffer<int> {q_, col};
        p_coef_ = UsmBuffer<real> {q_, coef};
        p_w_ = UsmBuffer<real> {q_, w};
        p_w_h_ = w;
        // Transpose over the stacked component space.
        std::vector<std::vector<std::pair<int, real>>> trows(nsd);
        for (std::size_t j = 0; j < n_pen_; ++j) {
            for (int e = off[j]; e < off[j + 1]; ++e) {
                trows[static_cast<std::size_t>(col[static_cast<std::size_t>(e)])].emplace_back(
                  static_cast<int>(j),
                  coef[static_cast<std::size_t>(e)]);
            }
        }
        std::vector<int> pt_off(nsd + 1, 0);
        std::vector<int> pt_row;
        std::vector<real> pt_coef;
        for (std::size_t i = 0; i < nsd; ++i) {
            for (const auto& [j, v] : trows[i]) {
                pt_row.push_back(j);
                pt_coef.push_back(v);
            }
            pt_off[i + 1] = static_cast<int>(pt_row.size());
        }
        pt_off_ = UsmBuffer<int> {q_, pt_off};
        pt_row_ = UsmBuffer<int> {q_, pt_row.empty() ? std::vector<int> {0} : pt_row};
        pt_coef_ = UsmBuffer<real> {q_, pt_coef.empty() ? std::vector<real> {0.0_r} : pt_coef};
        r_pen_ = UsmBuffer<real> {q_, n_pen_};
        r_pen0_h_.resize(n_pen_);
        r_pen1_h_.resize(n_pen_);
    }

    /// Replace one block's Boolean-sum offset and re-evaluate X on the net.
    void set_b(int block_index, const std::vector<real>& b)
    {
        if (block_index < 0 || std::cmp_greater_equal(block_index, nets_.size())) {
            throw std::invalid_argument("control topo: set_b block index out of range");
        }
        auto& net = nets_[static_cast<std::size_t>(block_index)];
        if (b.size() != net.b.size()) {
            throw std::invalid_argument("control topo: set_b size mismatch");
        }
        net.b.upload(b);
        eval_to_x();
    }

    /// Evaluate the resident net into the session's packed X (with b) and
    /// refresh ghosts/copies.
    void eval_to_x()
    {
        eval_all(C_.data(), true, X_);
        q_.wait();
    }

    void get_c(real* out) { C_.download(out); }

    void set_c(const std::vector<real>& c)
    {
        if (c.size() != C_.size()) {
            throw std::invalid_argument("set_c: expected " + std::to_string(C_.size()) +
                                        " reals, got " + std::to_string(c.size()));
        }
        C_.upload(c);
        eval_to_x();
    }

    std::size_t c_size() const { return C_.size(); }
    std::size_t nq() const { return nq_; }

    /// Run the damped reduced-GN outer loop in place on the session's X.
    template <class Obj> ControlReport run(Obj objective, const ControlOptions& opts)
    {
        constexpr real kLamFloorScale = 1e-8_r;
        ControlReport rep;
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        real lam = opts.lm0;
        real gmax0 = 0.0_r;
        std::array<real, 8> s {};
        model_db_ = opts.model_db && has_slide_;

        eval_to_x();
        reduce_energy_mindet<D>(q_, es_, X_, scal_.data(), scal_.data() + 1, objective);
        download(2, s);
        real e0 = s[0] + pen_energy_current();
        real md0 = s[1];
        rep.energies.push_back(e0);
        rep.mindets.push_back(md0);
        if (nq_ == 0) { return rep; }

        for (int outer = 0; outer < opts.n_outer; ++outer) {
            if (has_slide_) {
                // Frozen sliding-wall frame: reproject the sliding roots,
                // rewrite the reduction values, re-extend b — α-damped and
                // validity-gated (a frame that folds reverts; total failure
                // stops at the last valid state).
                const real e_prev = e0;
                if (!rebuild_frame(objective)) { break; }
                reduce_energy_mindet<D>(q_, es_, X_, scal_.data(), scal_.data() + 1, objective);
                download(2, s);
                e0 = s[0] + pen_energy_current();
                md0 = s[1];
                rep.frame_jumps.push_back(e0 - e_prev);
            }
            // Reduced gradient qG = Rᵀ(M + db/dC)ᵀ g + penalty terms.
            control_sample_g_kernel<D>(q_, es_, X_, z_samples_.data(), objective);
            gather_alias_to(y_.data());
            // The directional soft-energy nodal gradient joins the fine
            // gradient before the pullback (accept energies compose the term
            // via the reductions; the GN model keeps the plain shape Hessian).
            scatter_directional_grad<D>(q_, es_.dir, X_, y_.data());
            db_adjoint_add(y_.data());
            pullback_all(wfull_.data());
            if (n_pen_) {
                csr_apply_kernel(q_,
                                 n_pen_,
                                 p_off_.data(),
                                 p_col_.data(),
                                 p_coef_.data(),
                                 C_.data(),
                                 r_pen_.data());
                csrT_weighted_accum_kernel(q_,
                                           nsd,
                                           pt_off_.data(),
                                           pt_row_.data(),
                                           pt_coef_.data(),
                                           p_w_.data(),
                                           r_pen_.data(),
                                           wfull_.data());
            }
            csr_apply_kernel(q_,
                             nq_,
                             g_off_.data(),
                             g_row_.data(),
                             g_coef_.data(),
                             wfull_.data(),
                             qG_.data());
            vec_amax(q_, nq_, qG_.data(), scal_.data());
            download(1, s);
            const real gmax = s[0];
            if (gmax < opts.grad_tol) {
                rep.converged = true;
                break;
            }
            if (outer == 0) { gmax0 = gmax; }
            const real pcg_tol =
              opts.pcg_forcing ? std::max(opts.pcg_rtol, std::min(0.5_r, sycl::sqrt(gmax / gmax0)))
                               : opts.pcg_rtol;

            // Preconditioner: per-node contracted Hessian (alias-summed) ->
            // per-control blocks -> scalar q diagonal through the gather CSR.
            vec_fill(q_, h_node_.data(), h_node_.size(), 0.0_r);
            control_gather_hess_kernel<D>(q_,
                                          es_,
                                          mem_nodes_.data(),
                                          mem_off_.data(),
                                          mem_sid_.data(),
                                          mem_role_.data(),
                                          n_member_,
                                          X_,
                                          h_node_.data(),
                                          objective);
            alias_reduce_kernel(q_,
                                n_alias_,
                                alias_off_.data(),
                                alias_slot_.data(),
                                D * D,
                                h_node_.data());
            control_precond_build_all_kernel<D>(q_,
                                                net_refs_.data(),
                                                ctrl_pfx_.data(),
                                                static_cast<int>(nets_.size()),
                                                n_stack_,
                                                h_node_.data(),
                                                prec_.data());
            qdiag_build_kernel<D>(q_,
                                  nq_,
                                  g_off_.data(),
                                  g_row_.data(),
                                  g_coef_.data(),
                                  prec_.data(),
                                  qdiag_.data());
            vec_trace_sum<D>(q_, n_stack_, prec_.data(), scal_.data());
            download(1, s);
            const real lam_floor = kLamFloorScale * s[0] / static_cast<real>(nsd);

            bool accepted = false;
            real alpha = 1.0_r;
            real e_new = e0;
            real md_new = md0;
            for (int lt = 0; lt < opts.lm_tries && !accepted; ++lt) {
                const auto [pcg_its, relres] =
                  pcg_solve_q(objective, lam, opts.pcg_max_iter, pcg_tol);
                rep.pcg_iters.push_back(static_cast<real>(pcg_its));
                rep.pcg_relres.push_back(relres);

                // dC = R dq (stacked), corr = M dC (+ halo), penalty
                // residuals for the analytic α-quadratic.
                csr_apply_kernel(q_,
                                 nsd,
                                 r_off_.data(),
                                 r_col_.data(),
                                 r_coef_.data(),
                                 qdq_.data(),
                                 dCfull_.data());
                eval_dir(dCfull_.data(), corr_.data());
                if (n_pen_) {
                    csr_apply_kernel(q_,
                                     n_pen_,
                                     p_off_.data(),
                                     p_col_.data(),
                                     p_coef_.data(),
                                     C_.data(),
                                     r_pen_.data());
                    q_.wait();
                    q_.memcpy(r_pen0_h_.data(), r_pen_.data(), n_pen_ * sizeof(real)).wait();
                    csr_apply_kernel(q_,
                                     n_pen_,
                                     p_off_.data(),
                                     p_col_.data(),
                                     p_coef_.data(),
                                     dCfull_.data(),
                                     r_pen_.data());
                    q_.wait();
                    q_.memcpy(r_pen1_h_.data(), r_pen_.data(), n_pen_ * sizeof(real)).wait();
                }
                auto pen_at = [&](real a) {
                    real e = 0.0_r;
                    for (std::size_t j = 0; j < n_pen_; ++j) {
                        const real v = r_pen0_h_[j] + (a * r_pen1_h_[j]);
                        e += 0.5_r * p_w_h_[j] * v * v;
                    }
                    return e;
                };

                alpha = 1.0_r;
                while (alpha >= opts.alpha_min) {
                    detail::reduce_linesearch_pair<D>(q_,
                                                      es_,
                                                      X_,
                                                      corr_.data(),
                                                      alpha,
                                                      alpha * 0.5_r,
                                                      scal_.data(),
                                                      objective);
                    download(4, s);
                    bool ok = false;
                    for (int t = 0; t < 2; ++t) {
                        const real a = t == 0 ? alpha : alpha * 0.5_r;
                        const real e = s[(2 * t) + 0] + pen_at(a);
                        const real md = s[(2 * t) + 1];
                        if (a >= opts.alpha_min && sycl::isfinite(e) && e <= e0 + tol::energy &&
                            objective.accept_mindet(md)) {
                            alpha = a;
                            e_new = e;
                            md_new = md;
                            ok = true;
                            break;
                        }
                    }
                    if (ok) {
                        accepted = true;
                        break;
                    }
                    alpha *= 0.25_r;
                }
                if (!accepted) { lam = std::max(10.0_r * lam, lam_floor); }
            }
            if (!accepted) { break; }

            vec_axpy(q_, x_reals(), alpha, corr_.data(), X_);
            vec_axpy(q_, nsd, alpha, dCfull_.data(), C_.data());
            e0 = e_new;
            md0 = md_new;
            rep.energies.push_back(e0);
            rep.mindets.push_back(md0);
            rep.alphas.push_back(alpha);
            rep.iters += 1;
            if (alpha == 1.0_r) { lam /= 3.0_r; }
        }
        if (has_slide_) {
            // Closing frame so the reported state ends exactly on the CAD
            // (same damping/gate as the per-iteration frames).
            rebuild_frame(objective);
        }
        q_.wait();
        return rep;
    }

  private:
    struct Net {
        ControlNetMetaT<D> meta {};
        std::array<UsmBuffer<int>, D> off;
        std::array<UsmBuffer<real>, D> wt;
        std::array<UsmBuffer<int>, D> sup_off;
        std::array<UsmBuffer<int>, D> sup_node;
        std::array<UsmBuffer<real>, D> sup_w;
        UsmBuffer<real> b;
        std::size_t c_off = 0;  // control offset into the stacked vector
    };

    template <class T> static std::array<const T*, D> ptrs(const std::array<UsmBuffer<T>, D>& bufs)
    {
        std::array<const T*, D> out {};
        for (int k = 0; k < D; ++k) { out[static_cast<std::size_t>(k)] = bufs[k].data(); }
        return out;
    }

    std::size_t x_reals() const { return corr_.size(); }

    void download(int n, std::array<real, 8>& s)
    {
        q_.wait();
        q_.memcpy(s.data(), scal_.data(), static_cast<std::size_t>(n) * sizeof(real)).wait();
    }

    /// Evaluate every net from a stacked control vector into @p Xout (per-block
    /// interior slots), optionally adding b, then refresh ghosts/copies so
    /// seam-adjacent samples read a consistent field.
    void eval_all(const real* C_stacked, bool with_b, real* Xout, bool bcast = true)
    {
        control_eval_all_kernel<D>(q_,
                                   net_refs_.data(),
                                   node_pfx_.data(),
                                   static_cast<int>(nets_.size()),
                                   total_nodes_,
                                   C_stacked,
                                   with_b,
                                   Xout);
        if (bcast) { fused_halo_broadcast<D>(q_, Xout, halo_); }
    }

    /// Direction-field evaluation for the GN chain: corr = M·dC plus, on
    /// db-modelled runs, the frozen-frame Boolean-sum correction (which
    /// broadcasts itself).
    void eval_dir(const real* dC_stacked, real* out)
    {
        eval_all(dC_stacked, false, out, /*bcast=*/!model_db_);
        db_forward_add(out);
    }

    /// z_samples_ -> zero-filled per-node field -> alias-group sum: every
    /// block's slots carry the FULL contribution of the fine node they alias.
    void gather_alias_to(real* y)
    {
        vec_fill(q_, y, x_reals(), 0.0_r);
        control_gather_kernel<D>(q_,
                                 es_,
                                 mem_nodes_.data(),
                                 mem_off_.data(),
                                 mem_sid_.data(),
                                 mem_role_.data(),
                                 n_member_,
                                 z_samples_.data(),
                                 y);
        alias_reduce_kernel(q_, n_alias_, alias_off_.data(), alias_slot_.data(), D, y);
    }

    /// Per-block pullback of y_ into the stacked control space.
    void pullback_all(real* out_stacked)
    {
        control_pullback_all_kernel<D>(q_,
                                       net_refs_.data(),
                                       ctrl_pfx_.data(),
                                       static_cast<int>(nets_.size()),
                                       n_stack_,
                                       y_.data(),
                                       out_stacked);
    }

    /// Penalty energy at the current resident C (device apply + tiny download).
    real pen_energy_current()
    {
        if (!n_pen_) { return 0.0_r; }
        csr_apply_kernel(q_,
                         n_pen_,
                         p_off_.data(),
                         p_col_.data(),
                         p_coef_.data(),
                         C_.data(),
                         r_pen_.data());
        q_.wait();
        q_.memcpy(r_pen0_h_.data(), r_pen_.data(), n_pen_ * sizeof(real)).wait();
        real e = 0.0_r;
        for (std::size_t j = 0; j < n_pen_; ++j) {
            e += 0.5_r * p_w_h_[j] * r_pen0_h_[j] * r_pen0_h_[j];
        }
        return e;
    }

    /// Reduced matvec (Rᵀ(Â + PᵀWP)R + λI)·v.
    template <class Obj> void matvec_q(Obj objective, real lam, const real* v, real* out)
    {
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        csr_apply_kernel(q_, nsd, r_off_.data(), r_col_.data(), r_coef_.data(), v, vfull_.data());
        eval_dir(vfull_.data(), corr_.data());
        control_sample_hz_kernel<D>(q_, es_, X_, corr_.data(), z_samples_.data(), objective);
        gather_alias_to(y_.data());
        db_adjoint_add(y_.data());
        pullback_all(wfull_.data());
        if (n_pen_) {
            csr_apply_kernel(q_,
                             n_pen_,
                             p_off_.data(),
                             p_col_.data(),
                             p_coef_.data(),
                             vfull_.data(),
                             r_pen_.data());
            csrT_weighted_accum_kernel(q_,
                                       nsd,
                                       pt_off_.data(),
                                       pt_row_.data(),
                                       pt_coef_.data(),
                                       p_w_.data(),
                                       r_pen_.data(),
                                       wfull_.data());
        }
        csr_apply_kernel(q_, nq_, g_off_.data(), g_row_.data(), g_coef_.data(), wfull_.data(), out);
        if (lam > 0.0_r) { vec_axpy(q_, nq_, lam, v, out); }
    }

    /// Jacobi-preconditioned CG on the reduced system; qdq_ receives the step.
    template <class Obj>
    std::pair<int, real> pcg_solve_q(Obj objective, real lam, int max_iter, real rtol)
    {
        std::array<real, 8> s {};
        vec_fill(q_, qdq_.data(), nq_, 0.0_r);
        q_.parallel_for(sycl::range<1>(nq_),
                        [r = qr_.data(), g = qG_.data()](sycl::id<1> i) { r[i[0]] = -g[i[0]]; });
        vec_dot(q_, nq_, qr_.data(), qr_.data(), scal_.data());
        download(1, s);
        const real b_norm2 = s[0];
        if (b_norm2 == 0.0_r) { return {0, 0.0_r}; }
        const real stop2 = rtol * rtol * b_norm2;
        real r_norm2 = b_norm2;

        qdiag_apply_kernel(q_, nq_, qdiag_.data(), lam, qr_.data(), qz_.data());
        q_.memcpy(qp_.data(), qz_.data(), nq_ * sizeof(real));
        vec_dot(q_, nq_, qr_.data(), qz_.data(), scal_.data());
        download(1, s);
        real rz = s[0];

        int it = 0;
        for (; it < max_iter; ++it) {
            matvec_q(objective, lam, qp_.data(), qap_.data());
            vec_dot(q_, nq_, qp_.data(), qap_.data(), scal_.data());
            download(1, s);
            const real p_ap = s[0];
            if (!(p_ap > 0.0_r)) { break; }
            const real a = rz / p_ap;
            vec_axpy(q_, nq_, a, qp_.data(), qdq_.data());
            vec_axpy(q_, nq_, -a, qap_.data(), qr_.data());
            vec_dot(q_, nq_, qr_.data(), qr_.data(), scal_.data());
            download(1, s);
            r_norm2 = s[0];
            if (r_norm2 <= stop2) {
                ++it;
                break;
            }
            qdiag_apply_kernel(q_, nq_, qdiag_.data(), lam, qr_.data(), qz_.data());
            vec_dot(q_, nq_, qr_.data(), qz_.data(), scal_.data());
            download(1, s);
            const real rz_new = s[0];
            vec_xpay(q_, nq_, qz_.data(), rz_new / rz, qp_.data());
            rz = rz_new;
        }
        return {it, sycl::sqrt(r_norm2 / b_norm2)};
    }

    sycl::queue q_;
    StencilViewT<D> es_;
    real* X_;  // the session's packed coordinate buffer (not owned)
    HaloViewPtrs halo_;

    std::vector<Net> nets_;
    std::size_t n_stack_ = 0;
    UsmBuffer<real> C_;

    // Batched-launch tables (one record per net; item-space prefixes).
    std::size_t total_nodes_ = 0;
    UsmBuffer<NetRefT<D>> net_refs_;
    UsmBuffer<int> node_pfx_, ctrl_pfx_;

    std::size_t n_member_ = 0;
    UsmBuffer<int> mem_nodes_, mem_off_, mem_sid_, mem_role_;

    std::size_t n_alias_ = 0;
    UsmBuffer<int> alias_off_, alias_slot_;

    std::size_t nq_ = 0;
    UsmBuffer<int> r_off_, r_col_, g_off_, g_row_;
    UsmBuffer<real> r_coef_, g_coef_;

    std::size_t n_pen_ = 0;
    UsmBuffer<int> p_off_, p_col_, pt_off_, pt_row_;
    UsmBuffer<real> p_coef_, p_w_, pt_coef_, r_pen_;
    std::vector<real> p_w_h_, r_pen0_h_, r_pen1_h_;

    UsmBuffer<real> corr_, y_, h_node_, z_samples_, prec_;
    UsmBuffer<real> vfull_, wfull_, dCfull_;
    UsmBuffer<real> qG_, qdq_, qr_, qz_, qp_, qap_, qdiag_;
    UsmBuffer<real> scal_;

    // -----------------------------------------------------------------------
    // Sliding-wall frame loop (mirrors egg/smoothing/control_backend.py's
    // Python driver frames, ported so the loop has zero interpreter time and
    // one host<->device C round-trip per outer iteration).
    // -----------------------------------------------------------------------

    bool has_slide_ = false;
    ControlTopoHostT<D> host_;               // basis tables, X0, slide/wall wire
    std::vector<real> C_host_;               // host mirror of the stacked lattice
    std::vector<std::vector<real>> b_host_;  // per-net Boolean-sum offsets
    std::vector<int> g_entry_of_r_;
    std::vector<real> r_coef_h_, g_coef_h_;

    // db/dC frame linearization (see ControlOptions::model_db): frame-
    // invariant wall-face node lists per net + per-frame -n̂n̂ᵀ projectors
    // (written on device by frame_wall_delta_kernel). The batched db kernels
    // read the concatenated face lists; rebuild_wall_b writes each net's
    // projectors into its slice of the shared arena (face_pfx_h_).
    bool model_db_ = false;  // armed per run (opts.model_db && has_slide_)
    std::vector<std::vector<int>> dbn_flat_h_;
    std::vector<std::vector<int>> dbn_idx_of_flat_;
    std::vector<UsmBuffer<int>> dbn_idx_d_;
    std::size_t n_face_total_ = 0;
    std::vector<std::size_t> face_pfx_h_;
    UsmBuffer<int> dbn_net_, dbn_flat_all_, dbn_has_;
    UsmBuffer<real> dbn_P_all_;
    UsmBuffer<real> db_delta_;      // block-logical scratch (max wall net)
    UsmBuffer<real> db_delta_all_;  // concatenated block-logical delta (batched fwd)

    // Device-resident frame tables (uploaded once; only the CSR coefficient
    // VALUES and the projectors change per frame, and those are written by
    // the frame kernels in place).
    UsmBuffer<real> arena_d_;
    std::size_t n_wf_ = 0;
    std::size_t wf_stride_ = 0;
    UsmBuffer<int> wf_block_, wf_axis_, wf_side_, wf_tag_;
    UsmBuffer<real> wf_params_;
    std::size_t sl_stride_ = 0;
    std::size_t n_members_ = 0;
    UsmBuffer<int> sl_tag_, sl_rep_, sl_entry_off_, sl_r_entry_, sl_kt_, sl_comp_;
    UsmBuffer<real> sl_params_, sl_base_;
    UsmBuffer<int> sl_members_, sl_mem_root_;
    UsmBuffer<real> sl_mem_coef_;
    UsmBuffer<int> g_entry_of_r_d_;
    UsmBuffer<real> fp_buf_;               // per root: foot then pos (2D reals)
    UsmBuffer<real> C_snap_;               // frame-base / revert snapshot
    std::vector<UsmBuffer<real>> b_snap_;  // per wall net, revert snapshot
    std::vector<UsmBuffer<real>> X0_d_;    // per net (wall nets only)
    std::vector<UsmBuffer<int>> seam_d_;   // per net, 2*D outer-face flags
    std::vector<std::uint8_t> net_has_wall_;

    /// Re-extend one net's b on the DEVICE against the resident C: wall
    /// faces take the projection of the spline boundary onto their entity,
    /// other outer faces keep X0, seam faces stay zero (the recipe of
    /// egg.smoothing.control_topology.wall_b_field). Also refreshes the
    /// frozen db/dC projectors of the net's listed wall-face nodes.
    /// `Xs` must hold the pure spline field (eval_all without b).
    void rebuild_wall_b(std::size_t ni, const real* Xs)
    {
        const bool with_db = model_db_ && !dbn_flat_h_[ni].empty();
        frame_wall_delta_kernel<D>(q_,
                                   nets_[ni].meta,
                                   seam_d_[ni].data(),
                                   X0_d_[ni].data(),
                                   static_cast<int>(ni),
                                   n_wf_,
                                   wf_block_.data(),
                                   wf_axis_.data(),
                                   wf_side_.data(),
                                   wf_tag_.data(),
                                   wf_params_.data(),
                                   wf_stride_,
                                   arena_d_.data(),
                                   Xs,
                                   with_db,
                                   with_db ? dbn_idx_d_[ni].data() : nullptr,
                                   with_db ? dbn_P_all_.data() +
                                               (face_pfx_h_[ni] * static_cast<std::size_t>(D * D))
                                           : nullptr,
                                   db_delta_.data());
        tfi_ext_fill_block_kernel<D>(q_, nets_[ni].meta, db_delta_.data(), nets_[ni].b.data());
    }

    /// Forward db/dC: field += ext(P·field|wall-face) per wall net, then a
    /// fresh halo broadcast (the correction touches owner slots only).
    void db_forward_add(real* field)
    {
        if (!model_db_ || n_face_total_ == 0) { return; }
        vec_fill(q_, db_delta_all_.data(), db_delta_all_.size(), 0.0_r);
        db_face_delta_all_kernel<D>(q_,
                                    net_refs_.data(),
                                    dbn_net_.data(),
                                    dbn_flat_all_.data(),
                                    dbn_P_all_.data(),
                                    n_face_total_,
                                    field,
                                    db_delta_all_.data());
        tfi_ext_add_all_kernel<D>(q_,
                                  net_refs_.data(),
                                  node_pfx_.data(),
                                  static_cast<int>(nets_.size()),
                                  total_nodes_,
                                  dbn_has_.data(),
                                  db_delta_all_.data(),
                                  field);
        fused_halo_broadcast<D>(q_, field, halo_);
    }

    /// Adjoint db/dC on the alias-reduced node gradient field: y += Lᵀ y
    /// (reads block-interior and own-face slots only — all single-owner).
    void db_adjoint_add(real* y)
    {
        if (!model_db_ || n_face_total_ == 0) { return; }
        db_face_adjoint_all_kernel<D>(q_,
                                      net_refs_.data(),
                                      dbn_net_.data(),
                                      dbn_flat_all_.data(),
                                      dbn_P_all_.data(),
                                      n_face_total_,
                                      y,
                                      y);
    }

    /// One frozen-frame rebuild, fully device-resident: reproject the
    /// sliding roots and rewrite the reduction values (frame_slide_kernel),
    /// then walk the damped move (frame_member_move_kernel + per-net wall
    /// delta / b kernels) under the codebase validity gate — one 2-scalar
    /// download per damping try is the only host traffic. Returns false
    /// when even the smallest damped move folds; the previous (valid)
    /// state is restored from the device snapshots.
    template <class Obj> bool rebuild_frame(Obj objective)
    {
        const std::size_t nsd = n_stack_ * static_cast<std::size_t>(D);
        frame_slide_kernel<D>(q_,
                              host_.slide.size(),
                              sl_tag_.data(),
                              sl_rep_.data(),
                              sl_params_.data(),
                              sl_stride_,
                              arena_d_.data(),
                              sl_entry_off_.data(),
                              sl_r_entry_.data(),
                              sl_kt_.data(),
                              sl_comp_.data(),
                              sl_base_.data(),
                              g_entry_of_r_d_.data(),
                              C_.data(),
                              fp_buf_.data(),
                              r_coef_.data(),
                              g_coef_.data());
        // Snapshot the pre-frame state (device-to-device) for damping/revert.
        q_.memcpy(C_snap_.data(), C_.data(), nsd * sizeof(real));
        for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
            if (net_has_wall_[ni]) {
                q_.memcpy(b_snap_[ni].data(),
                          nets_[ni].b.data(),
                          nets_[ni].b.size() * sizeof(real));
            }
        }
        std::array<real, 8> s {};
        for (const real alpha : {1.0_r, 0.5_r, 0.25_r}) {
            q_.memcpy(C_.data(), C_snap_.data(), nsd * sizeof(real));
            frame_member_move_kernel<D>(q_,
                                        n_members_,
                                        sl_members_.data(),
                                        sl_mem_root_.data(),
                                        sl_mem_coef_.data(),
                                        fp_buf_.data(),
                                        alpha,
                                        C_.data());
            // Pure spline field for the wall deltas (the delta kernel reads
            // owner slots only, so no halo broadcast is needed here).
            eval_all(C_.data(), false, corr_.data(), /*bcast=*/false);
            for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
                if (net_has_wall_[ni]) { rebuild_wall_b(ni, corr_.data()); }
            }
            eval_all(C_.data(), true, X_);
            reduce_energy_mindet<D>(q_, es_, X_, scal_.data(), scal_.data() + 1, objective);
            download(2, s);
            if (objective.accept_mindet(s[1])) { return true; }
        }
        // Even the smallest damped move folds: restore the previous state.
        q_.memcpy(C_.data(), C_snap_.data(), nsd * sizeof(real));
        for (std::size_t ni = 0; ni < nets_.size(); ++ni) {
            if (net_has_wall_[ni]) {
                q_.memcpy(nets_[ni].b.data(),
                          b_snap_[ni].data(),
                          nets_[ni].b.size() * sizeof(real));
            }
        }
        eval_all(C_.data(), true, X_);
        q_.wait();
        return false;
    }
};

}  // namespace egg
