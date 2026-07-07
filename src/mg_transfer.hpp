// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file mg_transfer.hpp
/// Inter-grid transfer kernels for the FAS geometric multigrid: injection
/// restriction of node positions, full-weighting restriction of per-node
/// gradients (for the τ coherence term), multilinear prolongation of the
/// coarse correction, and the masked axpy the safeguarded line search applies
/// corrections with. All node addressing goes through the BlockLayout mdspan
/// views (structured.hpp); per-axis coarsening factors of 1 (semi-coarsened /
/// uncoarsened axis) or 2 are carried explicitly, so every kernel is generic
/// over partial coarsening. Kernels capture the trivially-copyable views by
/// value; one launch per block (block counts are small — fusing across blocks
/// is a possible later optimization, not done here).

#include "core.hpp"
#include "structured.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <utility>

namespace egg
{

/// Per-axis fine→coarse coarsening factor of one block: 2 on a coarsened
/// axis, 1 on a semi-coarsened (kept) axis.
template <int D> using CoarsenFactor = std::array<int, D>;

/// Rank-D node-unit view over a per-node byte mask laid out like a
/// BlockLayout store (see @ref interior_mask_view). layout_stride because the
/// interior window of a padded block is strided.
template <int D>
using MaskView =
  stdex::mdspan<const std::uint8_t, stdex::dextents<std::size_t, D>, stdex::layout_stride>;

namespace detail
{

template <int D, class View, std::size_t... K>
decltype(auto) node_at_impl(const View& v,
                            const std::array<std::size_t, D>& idx,
                            std::size_t comp,
                            std::index_sequence<K...> /*seq*/)
{ return v[idx[K]..., comp]; }

template <int D, std::size_t... K>
std::uint8_t mask_at_impl(const MaskView<D>& m,
                          const std::array<std::size_t, D>& idx,
                          std::index_sequence<K...> /*seq*/)
{ return m[idx[K]...]; }

}  // namespace detail

/// Component @p comp of the node at spatial index @p idx of a rank-(D+1) view.
template <int D, class View>
decltype(auto) node_at(const View& v, const std::array<std::size_t, D>& idx, std::size_t comp)
{ return detail::node_at_impl<D>(v, idx, comp, std::make_index_sequence<D> {}); }

/// Mask byte at spatial index @p idx of a rank-D mask view.
template <int D> std::uint8_t mask_at(const MaskView<D>& m, const std::array<std::size_t, D>& idx)
{ return detail::mask_at_impl<D>(m, idx, std::make_index_sequence<D> {}); }

/// Row-major delinearization of a flat index over a D-axis shape.
template <int D>
inline std::array<std::size_t, D> delinearize(std::size_t i,
                                              const std::array<std::size_t, D>& shape)
{
    std::array<std::size_t, D> idx {};
    for (int k = D; k-- > 0;) {
        idx[static_cast<std::size_t>(k)] = i % shape[static_cast<std::size_t>(k)];
        i /= shape[static_cast<std::size_t>(k)];
    }
    return idx;
}

/// Interior (ghost-excluded) node-unit mask view of block `b`: the mask buffer
/// holds ONE byte per node of the packed padded store (size
/// layout.total_reals()/D), so its strides are the coordinate-unit node
/// strides divided by D.
template <int D>
[[nodiscard]] MaskView<D>
  interior_mask_view(const std::uint8_t* base, const BlockLayout<D>& layout, std::size_t b)
{
    const auto shape = layout.interior_shape(b);
    const auto nstr = layout.node_strides(b);
    std::array<std::size_t, D> strides {};
    for (int k = 0; k < D; ++k) {
        strides[static_cast<std::size_t>(k)] = nstr[static_cast<std::size_t>(k)] / D;
    }
    using Ext = stdex::dextents<std::size_t, D>;
    const Ext ext {shape};
    return MaskView<D> {base + (layout.interior_origin_offset(b) / D),
                        stdex::layout_stride::mapping<Ext> {ext, strides}};
}

// The kernels below are NAMED functor types, not lambdas, so their kernel
// names are concrete instantiations (e.g. RestrictInject<2>) — stable across
// compilers and readable in runtime logs/profiles, instead of a lambda name
// dragging the enclosing function-template signature (with its dependent
// dextents<std::size_t, D + 1> alias) into the mangling. NB: these kernels
// must only be LAUNCHED from single-TU binaries for now — the ACPP SSCP flow
// mis-resolves the per-TU HCF object id when kernels launch from a multi-TU
// binary ("Could not obtain hcf kernel info"); see test_multigrid_device.cpp.

/// Kernel body of @ref restrict_inject.
template <int D> struct RestrictInject {
    InteriorView<D> fine, coarse;
    CoarsenFactor<D> factor;
    std::array<std::size_t, D> cs;  ///< coarse interior shape

    void operator()(sycl::id<1> it) const
    {
        const std::array<std::size_t, D> I = delinearize<D>(it[0], cs);
        std::array<std::size_t, D> F {};
        for (int k = 0; k < D; ++k) {
            F[static_cast<std::size_t>(k)] =
              I[static_cast<std::size_t>(k)] * static_cast<std::size_t>(factor[k]);
        }
        for (std::size_t k = 0; k < D; ++k) { node_at<D>(coarse, I, k) = node_at<D>(fine, F, k); }
    }
};

/// Injection restriction of node positions: coarse(I) = fine(I ⊙ factor) for
/// every coarse interior node (frozen coarse nodes included — their restricted
/// positions are the Dirichlet data near-face free DOFs read). Exact (a copy),
/// so coarse values are bitwise equal to their fine images.
template <int D>
inline sycl::event restrict_inject(sycl::queue& q,
                                   InteriorView<D> fine,
                                   InteriorView<D> coarse,
                                   CoarsenFactor<D> factor)
{
    std::array<std::size_t, D> cs {};
    std::size_t n = 1;
    for (int k = 0; k < D; ++k) {
        cs[static_cast<std::size_t>(k)] = coarse.extent(static_cast<std::size_t>(k));
        n *= cs[static_cast<std::size_t>(k)];
    }
    return q.parallel_for(sycl::range<1>(n), RestrictInject<D> {fine, coarse, factor, cs});
}

/// Full-weighting restriction of a per-node D-vector field (the fine
/// gradient), for the τ coherence term: a tensor-product (¼, ½, ¼) stencil on
/// each coarsened axis, identity on factor-1 axes. Only *free* coarse nodes
/// (per @p coarse_free) receive a value — frozen coarse nodes get exact zeros,
/// so the τ buffer is deterministic and no fine slot the metric kernels never
/// wrote (constrained/boundary nodes) is ever read. The fine field is the
/// PADDED (ghost-inclusive) view: a free coarse INTERFACE node's stencil
/// crosses its face into the ghost plane (refreshed on the gradient buffer by
/// the level's halo hook before restriction), while a free interior node's
/// stencil spans fine logical [2I−1, 2I+1] ⊂ [1, n−2] — in both cases every
/// slot read is one the hierarchy build's stencil-of-free/interface guards
/// verified is written (mg_level.hpp).
/// Kernel body of @ref restrict_full_weight.
template <int D> struct RestrictFullWeight {
    HaloView<D> fine;  ///< padded fine view (ghosts included)
    InteriorView<D> coarse;
    MaskView<D> coarse_free;
    CoarsenFactor<D> factor;
    std::array<std::size_t, D> cs;  ///< coarse interior shape

    void operator()(sycl::id<1> it) const
    {
        constexpr int kOffsets = [] {
            int p = 1;
            for (int k = 0; k < D; ++k) { p *= 3; }
            return p;
        }();
        const std::array<std::size_t, D> I = delinearize<D>(it[0], cs);
        if (mask_at<D>(coarse_free, I) == 0) {
            for (std::size_t k = 0; k < D; ++k) { node_at<D>(coarse, I, k) = 0.0_r; }
            return;
        }
        real acc[D] = {};
        for (int e = 0; e < kOffsets; ++e) {
            // Digits of e in base 3 give the per-axis offset in {-1, 0, +1};
            // non-zero offsets on factor-1 axes are skipped (identity there).
            int rem = e;
            real w = 1.0_r;
            std::array<std::size_t, D> F {};  // PADDED fine index (logical + 1)
            bool valid = true;
            for (int k = 0; k < D; ++k) {
                const int o = (rem % 3) - 1;
                rem /= 3;
                const std::size_t fk =
                  (I[static_cast<std::size_t>(k)] * static_cast<std::size_t>(factor[k])) + 1;
                if (factor[k] == 1) {
                    if (o != 0) {
                        valid = false;
                        break;
                    }
                    F[static_cast<std::size_t>(k)] = fk;
                } else {
                    w *= (o == 0) ? 0.5_r : 0.25_r;
                    // fk ≥ 1 in padded coords, so o = −1 lands on the ghost
                    // plane (index 0) at a face node — never out of range.
                    F[static_cast<std::size_t>(k)] = fk + static_cast<std::size_t>(o);
                }
            }
            if (!valid) { continue; }
            for (int k = 0; k < D; ++k) {
                acc[k] += w * node_at<D>(fine, F, static_cast<std::size_t>(k));
            }
        }
        for (std::size_t k = 0; k < D; ++k) { node_at<D>(coarse, I, k) = acc[k]; }
    }
};

template <int D>
inline sycl::event restrict_full_weight(sycl::queue& q,
                                        HaloView<D> fine,
                                        InteriorView<D> coarse,
                                        MaskView<D> coarse_free,
                                        CoarsenFactor<D> factor)
{
    std::array<std::size_t, D> cs {};
    std::size_t n = 1;
    for (int k = 0; k < D; ++k) {
        cs[static_cast<std::size_t>(k)] = coarse.extent(static_cast<std::size_t>(k));
        n *= cs[static_cast<std::size_t>(k)];
    }
    return q.parallel_for(sycl::range<1>(n),
                          RestrictFullWeight<D> {fine, coarse, coarse_free, factor, cs});
}

/// Kernel body of @ref prolong_correction: multilinear prolongation of the
/// coarse correction onto every fine interior node — fine nodes even on all
/// coarsened axes copy their coarse image, odd components average the 2^m
/// bracketing coarse nodes (m = number of odd axes). Exact for fields linear
/// in the logical index. The whole fine buffer is written (frozen coarse
/// nodes carry zero correction, so values taper toward frozen boundaries);
/// zeroing at non-Free fine nodes is the caller's job via @ref masked_axpy.
template <int D> struct ProlongCorrection {
    InteriorView<D> coarse, fine;
    CoarsenFactor<D> factor;
    std::array<std::size_t, D> fs;  ///< fine interior shape

    void operator()(sycl::id<1> it) const
    {
        constexpr int kCorners = 1 << D;
        const std::array<std::size_t, D> J = delinearize<D>(it[0], fs);
        std::array<std::size_t, D> c {};
        std::array<std::size_t, D> r {};
        real w0 = 1.0_r;  // weight of each contributing corner
        for (int k = 0; k < D; ++k) {
            const auto f = static_cast<std::size_t>(factor[k]);
            c[static_cast<std::size_t>(k)] = J[static_cast<std::size_t>(k)] / f;
            r[static_cast<std::size_t>(k)] = J[static_cast<std::size_t>(k)] % f;
            if (r[static_cast<std::size_t>(k)] != 0) { w0 *= 0.5_r; }
        }
        real acc[D] = {};
        for (int e = 0; e < kCorners; ++e) {
            std::array<std::size_t, D> C = c;
            bool valid = true;
            for (int k = 0; k < D; ++k) {
                if ((e >> k & 1) == 0) { continue; }
                if (r[static_cast<std::size_t>(k)] == 0) {
                    valid = false;  // no second corner on an even/uncoarsened axis
                    break;
                }
                C[static_cast<std::size_t>(k)] += 1;  // odd fine J < n-1 ⇒ c+1 ≤ n_c-1
            }
            if (!valid) { continue; }
            for (int k = 0; k < D; ++k) {
                acc[k] += w0 * node_at<D>(coarse, C, static_cast<std::size_t>(k));
            }
        }
        for (std::size_t k = 0; k < D; ++k) { node_at<D>(fine, J, k) = acc[k]; }
    }
};

template <int D>
inline sycl::event prolong_correction(sycl::queue& q,
                                      InteriorView<D> coarse,
                                      InteriorView<D> fine,
                                      CoarsenFactor<D> factor)
{
    std::array<std::size_t, D> fs {};
    std::size_t n = 1;
    for (int k = 0; k < D; ++k) {
        fs[static_cast<std::size_t>(k)] = fine.extent(static_cast<std::size_t>(k));
        n *= fs[static_cast<std::size_t>(k)];
    }
    return q.parallel_for(sycl::range<1>(n), ProlongCorrection<D> {coarse, fine, factor, fs});
}

/// out = x + α·c at nodes with a non-zero mask byte, out = x elsewhere — the
/// safeguarded line-search trial over the flat packed store (@p num_nodes
/// padded nodes of D coords each). @p out may alias @p x for in-place apply.
template <int D>
inline sycl::event masked_axpy(sycl::queue& q,
                               real* out,
                               const real* x,
                               const real* c,
                               real alpha,
                               const std::uint8_t* node_free,
                               std::size_t num_nodes)
{
    return q.parallel_for(sycl::range<1>(num_nodes), [=](sycl::id<1> it) {
        const std::size_t i = it[0];
        const real a = (node_free[i] != 0) ? alpha : 0.0_r;
        for (int k = 0; k < D; ++k) {
            out[(i * D) + static_cast<std::size_t>(k)] =
              x[(i * D) + static_cast<std::size_t>(k)] +
              (a * c[(i * D) + static_cast<std::size_t>(k)]);
        }
    });
}

/// Zero the D coords of every node with a zero mask byte, in place — strips a
/// prolonged correction to the Free set so one halo broadcast of the buffer
/// makes every line-search trial x + α·corr halo-consistent (the broadcast is
/// a pure copy gather, hence linear in the buffer).
template <int D>
inline sycl::event
  zero_unmasked(sycl::queue& q, real* buf, const std::uint8_t* node_free, std::size_t num_nodes)
{
    return q.parallel_for(sycl::range<1>(num_nodes), [=](sycl::id<1> it) {
        const std::size_t i = it[0];
        if (node_free[i] != 0) { return; }
        for (int k = 0; k < D; ++k) { buf[(i * D) + static_cast<std::size_t>(k)] = 0.0_r; }
    });
}

/// out = a − b over a flat buffer of @p n reals (the coarse correction
/// x_c − R x_f and the τ difference are both plain elementwise subtractions).
inline sycl::event vec_sub(sycl::queue& q, real* out, const real* a, const real* b, std::size_t n)
{
    return q.parallel_for(sycl::range<1>(n),
                          [=](sycl::id<1> it) { out[it[0]] = a[it[0]] - b[it[0]]; });
}

}  // namespace egg
