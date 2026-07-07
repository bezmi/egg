// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file fas.hpp
/// FAS (Full Approximation Scheme) nonlinear geometric-multigrid driver in
/// the safeguarded-minimization (MG/OPT) framing: the coarse-grid correction
/// is a search direction, applied through a backtracking line search on the
/// GLOBAL fine energy with the same acceptance rule as the smoother
/// (finite, non-increasing within tol::energy, accept_mindet) — so monotone
/// energy decrease is preserved and the shape barrier is never crossed.
///
/// One V-cycle: ν₁ Jacobi pre-smooths (jacobi_sweep_once) → restrict state
/// (injection) and gradient (full weighting) → τ = ∇E_c(Rx) − R̂∇E_f(x) →
/// recursive coarse solve of E_c(x) − τ·x (the HasTau kernel pair; the same
/// smooth/restrict/τ/correct leg repeats down the hierarchy, ν_c Newton
/// smooths on the coarsest level) → prolong the correction (multilinear,
/// zeroed at non-Free fine nodes) → safeguarded apply → ν₂ post-smooths.
/// The safeguard evaluates the α ladder two trials per stencil pass with
/// α = 0 riding along as the pre-state energy (reduce_linesearch_pair), so
/// the common accept-at-α=1 cycle costs one host sync; the per-cycle report
/// reduces into a device slot and is downloaded once after the last cycle.
/// Shape phase only: the untangle δ-continuation landscape is not
/// elliptic-like, so callers reject it before dispatching here.
///
/// Recursion is what buys the grid-size-independent rate: a single
/// coarse level is itself too large to solve with a fixed sweep budget once
/// fine grids pass ~129 nodes per axis (the two-level rate saturates), so
/// intermediate levels re-coarsen until a handful of sweeps is exact.
/// Coarse levels relax interior free DOFs only, so they need no halo
/// topology (mg_level.hpp).

#include "device.hpp"
#include "mg_level.hpp"
#include "mg_transfer.hpp"
#include "structured.hpp"
#include "structured_halo.hpp"
#include "sweep.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace egg
{

/// Knobs of one run_fas call. Defaults give a V(2,2) cycle with a
/// well-converged coarsest solve (coarse sweeps are cheap: 2^-D of the fine
/// DOF count per level).
struct FasOptions {
    int n_cycles = 10;                    ///< V-cycles to run (one (energy, mindet) report each)
    int nu_pre = 2;                       ///< pre-smooth sweeps per level per cycle
    int nu_post = 2;                      ///< post-smooth sweeps per level per cycle
    int nu_coarse = 32;                   ///< cap on Newton sweeps on the COARSEST level per
                                          ///< cycle; the driver runs min(nu_coarse, twice the
                                          ///< coarsest level's interior diameter) — see
                                          ///< detail::coarse_sweep_budget
    real omega = 1.0_r;                   ///< Jacobi relaxation weight (all levels)
    int max_backtracks = kMaxBacktracks;  ///< α halvings before the correction is dropped
    int max_levels = 32;                  ///< V-cycle depth cap, fine level included; the cycle
                                          ///< recurses through min(built, max_levels − 1) coarse
                                          ///< levels (the hierarchy build stops on its own when
                                          ///< blocks stop coarsening, so the default is
                                          ///< effectively unbounded)
};

/// The fine level as run_fas consumes it — a thin non-owning view so the
/// driver is decoupled from SweepDeviceContextT (the executor assembles this
/// from its context; tests assemble it from hand-built pieces).
template <int D> struct FasFineView {
    std::size_t num_nodes = 0;  ///< padded node count
    real* X = nullptr;          ///< canonical packed positions [num_nodes * D]
    real* seeds = nullptr;      ///< warm-start cache (null = cold projections)
    GroupViewT<D> group;        ///< merged Free-first group view
    StencilViewT<D> stencil;    ///< global energy stencil (the reduction)
};

/// Reconstruct the fine BlockLayout from the node-unit layout a structured
/// sweep context carries (block_off/nstride are per block, node units,
/// innermost stride 1): padded[k>0] = nstride[k-1]/nstride[k], padded[0] =
/// block node count / nstride[0], interior = padded − 2. Empty (unstructured
/// context) returns an empty layout.
template <int D> BlockLayout<D> layout_from_context(const SweepContextHostT<D>& host)
{
    const std::size_t nb = host.block_off.size();
    if (nb == 0) { return BlockLayout<D> {}; }
    std::vector<typename BlockLayout<D>::Shape> shapes(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        const auto off = static_cast<std::size_t>(host.block_off[b]);
        const std::size_t end =
          (b + 1 < nb) ? static_cast<std::size_t>(host.block_off[b + 1]) : host.num_nodes;
        const std::size_t count = end - off;
        typename BlockLayout<D>::Shape padded {};
        const auto stride = [&](int k) {
            return static_cast<std::size_t>(host.nstride[(b * D) + static_cast<std::size_t>(k)]);
        };
        padded[0] = count / stride(0);
        for (int k = 1; k < D; ++k) {
            padded[static_cast<std::size_t>(k)] = stride(k - 1) / stride(k);
        }
        for (int k = 0; k < D; ++k) {
            shapes[b][static_cast<std::size_t>(k)] = padded[static_cast<std::size_t>(k)] - 2;
        }
    }
    return BlockLayout<D> {shapes};
}

/// Fine-level relaxability masks (per packed node id) from the host context:
/// `free` marks every Free-partition DOF's node, `free_interior` the subset
/// the structured build flagged interior-eligible. Feed to
/// build_mg_hierarchy (mg_level.hpp).
template <int D> MgMasks masks_from_context(const SweepContextHostT<D>& host)
{
    MgMasks m;
    m.free.assign(host.num_nodes, 0);
    m.free_interior.assign(host.num_nodes, 0);
    for (const auto& g : host.groups) {
        for (const auto& rec : g.soa) {
            if (rec.tag != EntityTag::Free) { continue; }
            for (const int dl : rec.dof_local) {
                const auto d = static_cast<std::size_t>(dl);
                const auto node = static_cast<std::size_t>(g.dof_idx[d]);
                m.free[node] = 1;
                if (!g.interior_block.empty() && g.interior_block[d] >= 0) {
                    m.free_interior[node] = 1;
                }
            }
        }
    }
    return m;
}

namespace detail
{

/// Newton-sweep budget that converges the coarsest solve: relaxation
/// propagates information roughly one cell per sweep, so the budget scales
/// with the level's interior diameter (largest interior extent over blocks
/// and axes), doubled for slack. The caller caps it with
/// FasOptions::nu_coarse, so a genuinely large coarsest level (a shallow
/// hierarchy or a big carried block) keeps the full fixed budget while the
/// deep-ladder tiny-grid case — where each sweep is almost pure launch
/// overhead — stops iterating on an already-converged state.
template <int D> int coarse_sweep_budget(const BlockLayout<D>& layout)
{
    std::size_t ext = 0;
    for (std::size_t b = 0; b < layout.num_blocks(); ++b) {
        const auto shape = layout.interior_shape(b);
        for (int k = 0; k < D; ++k) { ext = std::max(ext, shape[static_cast<std::size_t>(k)]); }
    }
    return static_cast<int>(2 * ext);
}

/// Energy and min det of the two line-search trials x + a·corr, a ∈ {a0, a1},
/// in ONE stencil pass (four reductions, one launch): each sample loads its
/// nodes from x and corr once and assembles both trial states in registers.
/// @p corr must already be zeroed outside the Free set and halo-broadcast, so
/// every trial inherits x's consistent halo by linearity of the (pure-copy)
/// broadcast; a = 0 evaluates x itself exactly. out receives
/// {e(a0), min det(a0), e(a1), min det(a1)}.
template <int D, class Obj>
inline void reduce_linesearch_pair(sycl::queue& q,
                                   const StencilViewT<D>& es,
                                   const real* X,
                                   const real* corr,
                                   real a0,
                                   real a1,
                                   real* out,
                                   Obj objective)
{
    const auto init = sycl::property::reduction::initialize_to_identity {};
    auto e0_red = sycl::reduction(out + 0, 0.0_r, std::plus<>(), init);
    auto m0_red =
      sycl::reduction(out + 1, std::numeric_limits<real>::infinity(), sycl::minimum<real>(), init);
    auto e1_red = sycl::reduction(out + 2, 0.0_r, std::plus<>(), init);
    auto m1_red =
      sycl::reduction(out + 3, std::numeric_limits<real>::infinity(), sycl::minimum<real>(), init);

    q.parallel_for(sycl::range<1>(es.num_samples),
                   e0_red,
                   m0_red,
                   e1_red,
                   m1_red,
                   [=](sycl::id<1> idx, auto& e0_sum, auto& m0_min, auto& e1_sum, auto& m1_min) {
                       const std::size_t i = idx[0];
                       const real* w = &es.W_inv[i * static_cast<std::size_t>(es.w_stride)];
                       const PtN<D> xc = load_pt<D>(X, es.gc[i]);
                       const PtN<D> cc = load_pt<D>(corr, es.gc[i]);
                       std::array<PtN<D>, D> xn;
                       std::array<PtN<D>, D> cn;
                       std::array<real, D> sc;
                       for (int k = 0; k < D; ++k) {
                           xn[k] = load_pt<D>(X, es.gn[k][i]);
                           cn[k] = load_pt<D>(corr, es.gn[k][i]);
                           sc[k] = es.s[k][i];
                       }
                       const auto eval = [&](real a, auto& e_sum, auto& m_min) {
                           PtN<D> corner;
                           std::array<PtN<D>, D> nbr;
                           for (int k = 0; k < D; ++k) {
                               corner[k] = xc[k] + (a * cc[k]);
                               for (int j = 0; j < D; ++j) {
                                   nbr[j][k] = xn[j][k] + (a * cn[j][k]);
                               }
                           }
                           real detA;
                           const VecTN<D> t = assemble_vecT<D>(corner, nbr, sc, w, detA);
                           e_sum += objective.value(t);
                           m_min.combine(detA);
                       };
                       eval(a0, e0_sum, m0_min);
                       eval(a1, e1_sum, m1_min);
                   });
}

/// One V-cycle leg on coarse level @p li of @p levels, in place. On entry
/// levels[li].x AND .x_swap hold the restricted parent state (frozen nodes
/// must be present in both Jacobi buffers) and .tau the FAS shift; on exit
/// levels[li].x holds the level's approximate minimiser of E_ℓ(x) − τ_ℓ·x.
/// The coarsest level runs opts.nu_coarse Newton sweeps; intermediate levels
/// recurse — nu_pre smooths, restrict state + shifted gradient, child τ,
/// recurse, apply the child's prolonged correction at free nodes with α = 1,
/// nu_post smooths.
///
/// Intermediate corrections are deliberately unsafeguarded: the fine-level
/// backtracking line search still gates the only correction that reaches the
/// real grid, and every coarse state is rebuilt from restriction next cycle,
/// so a bad intermediate step costs at most one rejected cycle — while the
/// per-DOF acceptance inside interior_update_kernel keeps each smooth itself
/// monotone in the shifted energy.
template <int D, class Obj>
void fas_coarse_vcycle(sycl::queue& q,
                       std::span<MgLevel<D>> levels,
                       std::size_t li,
                       Obj objective,
                       const FasOptions& opts)
{
    MgLevel<D>& lvl = levels[li];
    const GroupViewT<D> cg = lvl.group();
    const std::size_t cbuf = lvl.num_nodes * static_cast<std::size_t>(D);

    real* cx_in = lvl.x.data();
    real* cx_out = lvl.x_swap.data();
    // The level's halo hook: refresh face ghosts + non-owner copies of
    // shared interface nodes on the read buffer, so interface DOFs relax from
    // a current frozen halo exactly like the fine cadence. A no-op (empty
    // topology) when no interface DOF relaxes on this level.
    const auto hook = [&](real* buf) { fused_halo_broadcast<D>(q, buf, lvl.topo); };
    const auto smooth = [&](int sweeps) {
        for (int s = 0; s < sweeps; ++s) {
            hook(cx_in);
            metric_kernel<D, Obj, true>(q,
                                        cg,
                                        cx_in,
                                        lvl.grad.data(),
                                        lvl.hess.data(),
                                        lvl.e0.data(),
                                        objective,
                                        lvl.tau.data());
            interior_update_kernel<D, Obj, true>(q,
                                                 cg,
                                                 cx_in,
                                                 cx_out,
                                                 opts.omega,
                                                 lvl.grad.data(),
                                                 lvl.hess.data(),
                                                 lvl.e0.data(),
                                                 objective,
                                                 lvl.tau.data());
            std::swap(cx_in, cx_out);
        }
    };

    if (li + 1 == levels.size()) {
        smooth(opts.nu_coarse);
    } else {
        MgLevel<D>& next = levels[li + 1];
        const std::size_t nbuf = next.num_nodes * static_cast<std::size_t>(D);
        const std::size_t nb = lvl.layout.num_blocks();

        smooth(opts.nu_pre);

        // The same restrict/τ leg the fine driver runs, one level down: the
        // gradient the child must be coherent with is the SHIFTED one,
        // g = ∇E_ℓ − τ_ℓ, which the HasTau metric kernel already stores. The
        // hooks keep the post-smooth state and the gradient consistent at
        // interfaces before either is read across blocks (the FW stencil of a
        // child interface DOF crosses this level's ghost plane).
        hook(cx_in);
        metric_kernel<D, Obj, true>(q,
                                    cg,
                                    cx_in,
                                    lvl.grad.data(),
                                    lvl.hess.data(),
                                    lvl.e0.data(),
                                    objective,
                                    lvl.tau.data());
        hook(lvl.grad.data());
        for (std::size_t b = 0; b < nb; ++b) {
            restrict_inject<D>(q,
                               interior_view<D>(cx_in, lvl.layout, b),
                               interior_view<D>(next.x.data(), next.layout, b),
                               next.factor[b]);
            restrict_full_weight<D>(q,
                                    halo_view<D>(lvl.grad.data(), lvl.layout, b),
                                    interior_view<D>(next.tau.data(), next.layout, b),
                                    interior_mask_view<D>(next.free_mask.data(), next.layout, b),
                                    next.factor[b]);
        }
        // Injection fills the child interior; its ghosts/copies come from the
        // child hook so rx and both Jacobi buffers inherit a consistent halo.
        fused_halo_broadcast<D>(q, next.x.data(), next.topo);
        q.memcpy(next.rx.data(), next.x.data(), nbuf * sizeof(real));
        q.memcpy(next.x_swap.data(), next.x.data(), nbuf * sizeof(real));
        metric_kernel<D, Obj>(q,
                              next.group(),
                              next.rx.data(),
                              next.grad.data(),
                              next.hess.data(),
                              next.e0.data(),
                              objective);
        vec_sub(q, next.tau.data(), next.grad.data(), next.tau.data(), nbuf);

        fas_coarse_vcycle<D>(q, levels, li + 1, objective, opts);

        // Child correction, prolonged into this level's zero-ghosted corr
        // buffer and applied in place at free nodes (frozen nodes keep their
        // Dirichlet data; next.grad reused as x_c − Rx scratch).
        vec_sub(q, next.grad.data(), next.x.data(), next.rx.data(), nbuf);
        for (std::size_t b = 0; b < nb; ++b) {
            prolong_correction<D>(q,
                                  interior_view<D>(next.grad.data(), next.layout, b),
                                  interior_view<D>(lvl.corr.data(), lvl.layout, b),
                                  next.factor[b]);
        }
        masked_axpy<D>(q,
                       cx_in,
                       cx_in,
                       lvl.corr.data(),
                       1.0_r,
                       lvl.free_mask.data(),
                       lvl.num_nodes);

        smooth(opts.nu_post);
    }

    if (cx_in != lvl.x.data()) { q.memcpy(lvl.x.data(), cx_in, cbuf * sizeof(real)); }
    // Leave the solution with a consistent halo: the parent's correction
    // (x − rx, prolonged per block) reads interface copies of BOTH blocks.
    hook(lvl.x.data());
}

}  // namespace detail

/// Run n_cycles of the FAS V-cycle described in the file header.
/// The result is left in fine.X (like run_block_jacobi's canonical buffer);
/// returns per-cycle (energy, min det) over the fine stencil.
///
/// @param q           in-order queue (same discipline as run_block_jacobi).
/// @param fine        fine-level view (positions, group, stencil, seeds).
/// @param fine_layout fine BlockLayout (transfer views).
/// @param fine_free   device mask, 1 byte per fine node: correction applies
///                    only at Free nodes (constrained nodes must not be
///                    pushed off their geometry).
/// @param levels      the coarse hierarchy, fine→coarsest, non-empty; the
///                    cycle recurses through all of it (callers clip to
///                    opts.max_levels − 1 coarse levels).
/// @param objective   shape objective (untangle is rejected by callers).
/// @param before_sweep per-sweep fine halo hook (also run on the prolonged
///                    correction and before the report reduction so
///                    ghosts/shared copies are current). Must be a pure copy
///                    gather (linear in the buffer): the line search
///                    broadcasts the masked correction once and forms every
///                    trial as x + α·corr without per-trial hooks.
/// @param scratch     reusable fine-level device buffers (lazily sized here;
///                    the resident session passes its own so per-chunk calls
///                    do not re-allocate).
template <int D, class Obj, class PreSweep>
std::pair<std::vector<real>, std::vector<real>> run_fas(sycl::queue& q,
                                                        const FasFineView<D>& fine,
                                                        const BlockLayout<D>& fine_layout,
                                                        const std::uint8_t* fine_free,
                                                        std::span<MgLevel<D>> levels,
                                                        Obj objective,
                                                        const FasOptions& opts,
                                                        PreSweep before_sweep,
                                                        SweepScratchT<D>& scratch)
{
    MgLevel<D>& lvl = levels.front();
    const std::size_t nn = fine.num_nodes;
    const std::size_t nbuf = nn * static_cast<std::size_t>(D);
    const std::size_t cbuf = lvl.num_nodes * static_cast<std::size_t>(D);
    const std::size_t nb = fine_layout.num_blocks();

    // Fine-level scratch: double buffer, smoother metric buffers, prolonged
    // correction, 4-slot line-search readback (all reused via @p scratch);
    // per-cycle report slots are per-call (sized by n_cycles, downloaded once
    // after the loop). corr is zeroed ONCE PER CALL so every slot is defined
    // before the first cycle's zero_unmasked/broadcast pass touches it
    // (prolongation refills the interior slots each cycle).
    scratch.ensure_fas(q, nn, nbuf);
    UsmBuffer<real>& x_new = scratch.x_new;
    UsmBuffer<real>& grad = scratch.grad;
    UsmBuffer<real>& hess = scratch.hess;
    UsmBuffer<real>& e0 = scratch.e0;
    UsmBuffer<real>& corr = scratch.corr;
    UsmBuffer<real>& d_ls = scratch.ls;
    UsmBuffer<real> d_e {q, static_cast<std::size_t>(opts.n_cycles)};
    UsmBuffer<real> d_m {q, static_cast<std::size_t>(opts.n_cycles)};
    q.memset(corr.data(), 0, nbuf * sizeof(real));

    real* canonical = fine.X;
    real* x_in = canonical;
    real* x_out = x_new.data();
    q.memcpy(x_out, x_in, nbuf * sizeof(real)).wait();

    const GroupViewT<D>& mg = fine.group;
    const GroupViewT<D> cg = lvl.group();

    const auto fine_smooth = [&](int sweeps, bool& first_sweep) {
        for (int s = 0; s < sweeps; ++s) {
            jacobi_sweep_once<D, Obj>(q,
                                      mg,
                                      objective,
                                      before_sweep,
                                      x_in,
                                      x_out,
                                      opts.omega,
                                      first_sweep,
                                      fine.seeds,
                                      grad.data(),
                                      hess.data(),
                                      e0.data());
            first_sweep = false;
            std::swap(x_in, x_out);
        }
    };

    // Two ladder trials per launch, one waited readback into @p ls — the only
    // host sync of a cycle. The wait also bounds the in-order queue's growth
    // (run_block_jacobi syncs every kSyncEvery sweeps for the same reason).
    real ls[4];
    const auto ladder = [&](real a0, real a1) {
        detail::reduce_linesearch_pair<D>(q,
                                          fine.stencil,
                                          x_in,
                                          corr.data(),
                                          a0,
                                          a1,
                                          d_ls.data(),
                                          objective);
        q.memcpy(ls, d_ls.data(), 4 * sizeof(real)).wait();
    };

    // Coarsest-solve budget adapted to the hierarchy actually cycled: the
    // fixed nu_coarse is a cap (see coarse_sweep_budget). Only the recursive
    // leg reads nu_coarse, so the copy leaves the fine-level knobs alone.
    FasOptions vopts = opts;
    vopts.nu_coarse =
      std::min(opts.nu_coarse, detail::coarse_sweep_budget<D>(levels.back().layout));

    bool first_sweep = true;

    for (int cyc = 0; cyc < opts.n_cycles; ++cyc) {
        // 1. Pre-smooth.
        fine_smooth(opts.nu_pre, first_sweep);

        // 2. Fresh fine gradient at the smoothed x (per-node, into grad).
        //    The second hook runs the SAME halo/broadcast on the gradient
        //    buffer: a coarse interface DOF's full-weighting stencil reads
        //    the fine gradient across its face (ghost slot) and at non-owner
        //    face copies.
        before_sweep(q, x_in);
        metric_kernel<D, Obj>(q,
                              mg.dof_subrange(0, mg.n_free),
                              x_in,
                              grad.data(),
                              hess.data(),
                              e0.data(),
                              objective);
        before_sweep(q, grad.data());

        // 3. Restrict state (injection, all coarse nodes — frozen nodes get
        //    their Dirichlet data) and gradient (full weighting, free coarse
        //    nodes only; τ buffer holds R̂∇E_f for now). The coarse hook then
        //    gives rx and both Jacobi buffers a consistent interface halo.
        for (std::size_t b = 0; b < nb; ++b) {
            restrict_inject<D>(q,
                               interior_view<D>(x_in, fine_layout, b),
                               interior_view<D>(lvl.x.data(), lvl.layout, b),
                               lvl.factor[b]);
            restrict_full_weight<D>(q,
                                    halo_view<D>(grad.data(), fine_layout, b),
                                    interior_view<D>(lvl.tau.data(), lvl.layout, b),
                                    interior_mask_view<D>(lvl.free_mask.data(), lvl.layout, b),
                                    lvl.factor[b]);
        }
        fused_halo_broadcast<D>(q, lvl.x.data(), lvl.topo);
        q.memcpy(lvl.rx.data(), lvl.x.data(), cbuf * sizeof(real));
        q.memcpy(lvl.x_swap.data(), lvl.x.data(), cbuf * sizeof(real));

        // 4. τ = ∇E_c(R x) − R̂ ∇E_f(x): raw coarse gradient at Rx, then
        //    subtract in place. Frozen τ slots end up as stale-minus-zero,
        //    which is fine — the kernels only ever index τ at free DOFs.
        metric_kernel<D, Obj>(q,
                              cg,
                              lvl.rx.data(),
                              lvl.grad.data(),
                              lvl.hess.data(),
                              lvl.e0.data(),
                              objective);
        vec_sub(q, lvl.tau.data(), lvl.grad.data(), lvl.tau.data(), cbuf);

        // 5. Coarse solve of E_c(x) − τ·x: the recursive V-cycle leg —
        //    smooth/re-coarsen down the hierarchy, ν_c Newton sweeps on the
        //    coarsest level (frozen nodes hold Rx in both Jacobi buffers).
        detail::fas_coarse_vcycle<D>(q, levels, 0, objective, vopts);

        // 6. Prolong the coarse correction x_c − Rx (lvl.grad reused as
        //    scratch) onto the fine correction buffer.
        vec_sub(q, lvl.grad.data(), lvl.x.data(), lvl.rx.data(), cbuf);
        for (std::size_t b = 0; b < nb; ++b) {
            prolong_correction<D>(q,
                                  interior_view<D>(lvl.grad.data(), lvl.layout, b),
                                  interior_view<D>(corr.data(), fine_layout, b),
                                  lvl.factor[b]);
        }

        // 7. Safeguarded application: backtracking on the global fine energy
        //    with the smoother's acceptance rule; all-fail drops the
        //    correction — FAS still converges through the smooths. The
        //    correction is stripped to the Free set and halo-broadcast ONCE
        //    (before_sweep is a pure copy gather, so every x + α·corr trial
        //    inherits x's consistent halo — x was broadcast in step 2 and is
        //    untouched since); each launch reduces two ladder entries, and
        //    the first doubles α = 0 as the pre-state energy, so the common
        //    accept-at-α=1 cycle costs a single host sync.
        zero_unmasked<D>(q, corr.data(), fine_free, nn);
        before_sweep(q, corr.data());
        const auto ok = [&](real e_pre, real e_t, real m_t) {
            return sycl::isfinite(e_t) && (e_t <= e_pre + tol::energy) &&
                   objective.accept_mindet(m_t);
        };
        real alpha = 0.0_r;
        if (opts.max_backtracks > 0) {
            ladder(0.0_r, 1.0_r);
            const real e_pre = ls[0];
            if (ok(e_pre, ls[2], ls[3])) { alpha = 1.0_r; }
            real a = 0.5_r;
            for (int it = 1; alpha == 0.0_r && it < opts.max_backtracks; it += 2, a *= 0.25_r) {
                ladder(a, a * 0.5_r);
                if (ok(e_pre, ls[0], ls[1])) {
                    alpha = a;
                } else if (it + 1 < opts.max_backtracks && ok(e_pre, ls[2], ls[3])) {
                    alpha = a * 0.5_r;
                }
            }
        }
        // In-place apply at Free nodes only; copies/ghosts go stale until the
        // next hook, which runs before any read (post-smooth and the report).
        if (alpha != 0.0_r) { masked_axpy<D>(q, x_in, x_in, corr.data(), alpha, fine_free, nn); }

        // 8. Post-smooth, then the per-cycle report — reduced into this
        //    cycle's device slot, downloaded once after the loop.
        fine_smooth(opts.nu_post, first_sweep);
        before_sweep(q, x_in);
        reduce_energy_mindet<D>(q,
                                fine.stencil,
                                x_in,
                                d_e.data() + static_cast<std::size_t>(cyc),
                                d_m.data() + static_cast<std::size_t>(cyc),
                                objective);
    }
    q.wait();

    if (x_in != canonical) { q.memcpy(canonical, x_in, nbuf * sizeof(real)).wait(); }
    std::vector<real> energies(static_cast<std::size_t>(opts.n_cycles));
    std::vector<real> mindets(static_cast<std::size_t>(opts.n_cycles));
    d_e.download(energies.data());
    d_m.download(mindets.data());
    return {energies, mindets};
}

/// Convenience overload with one-shot scratch (tests / single calls); resident
/// sessions pass their own SweepScratchT to reuse the buffers across calls.
template <int D, class Obj, class PreSweep>
std::pair<std::vector<real>, std::vector<real>> run_fas(sycl::queue& q,
                                                        const FasFineView<D>& fine,
                                                        const BlockLayout<D>& fine_layout,
                                                        const std::uint8_t* fine_free,
                                                        std::span<MgLevel<D>> levels,
                                                        Obj objective,
                                                        const FasOptions& opts,
                                                        PreSweep before_sweep)
{
    SweepScratchT<D> scratch;
    return run_fas<D>(q,
                      fine,
                      fine_layout,
                      fine_free,
                      levels,
                      objective,
                      opts,
                      before_sweep,
                      scratch);
}

}  // namespace egg
