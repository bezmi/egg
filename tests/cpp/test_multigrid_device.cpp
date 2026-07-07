// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_multigrid_device.cpp — FAS transfer kernels on EVERY visible device
// (GPU + OpenMP host), own binary with its own main like the other
// kernel-launching tests (the SSCP flow mis-resolves per-TU HCF object ids
// when kernels launch from a multi-TU binary, so kernel tests stay
// single-TU). Injection restriction is bitwise-exact, full weighting
// reproduces linears at free coarse nodes and zeros frozen ones, multilinear
// prolongation is exact on index-linear fields, restrict∘prolong is the
// coarse identity, masked axpy respects its mask. Algebraically-exact
// assertions use tight tolerances via real_tol (identity in fp64, floored in
// fp32).

#include "fas.hpp"
#include "mg_level.hpp"
#include "mg_transfer.hpp"
#include "real_tol.hpp"
#include "structured.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace boost::ut;
using namespace egg;

namespace
{

// Distinct per-slot values so any indexing slip shows up.
void fill_pattern(std::vector<real>& v)
{
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<real>(i % 977) + 0.25_r; }
}

// Fine masks for one all-interior block (faces constrained): boundary ring
// nodes are constrained, everything else Free and interior-eligible.
template <int D> MgMasks single_block_masks(const BlockLayout<D>& layout)
{
    MgMasks m;
    const std::size_t nn = layout.total_reals() / D;
    m.free.assign(nn, 0);
    m.free_interior.assign(nn, 0);
    const auto shape = layout.interior_shape(0);
    std::size_t count = 1;
    for (int k = 0; k < D; ++k) { count *= shape[static_cast<std::size_t>(k)]; }
    for (std::size_t i = 0; i < count; ++i) {
        const std::array<std::size_t, D> I = delinearize<D>(i, shape);
        bool interior = true;
        for (int k = 0; k < D; ++k) {
            const std::size_t v = I[static_cast<std::size_t>(k)];
            interior = interior && v >= 1 && v + 1 < shape[static_cast<std::size_t>(k)];
        }
        if (!interior) { continue; }
        const std::size_t id = layout.interior_node_offset(0, I) / D;
        m.free[id] = 1;
        m.free_interior[id] = 1;
    }
    return m;
}

void test_restrict_inject(sycl::queue& q)
{
    const BlockLayout<2> fine {{{9, 9}}};
    const BlockLayout<2> coarse {{{5, 5}}};
    std::vector<real> hf(fine.total_reals());
    fill_pattern(hf);
    UsmBuffer<real> df {q, hf};
    UsmBuffer<real> dc {q, std::vector<real>(coarse.total_reals(), -1.0_r)};
    restrict_inject<2>(q,
                       interior_view<2>(df.data(), fine, 0),
                       interior_view<2>(dc.data(), coarse, 0),
                       {2, 2})
      .wait();
    std::vector<real> hc(coarse.total_reals());
    dc.download(hc.data());
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                const real got = hc[coarse.interior_node_offset(0, {i, j}) + k];
                const real want = hf[fine.interior_node_offset(0, {2 * i, 2 * j}) + k];
                expect(got == want) << "coarse" << i << j << k;
            }
        }
    }
}

void test_full_weight(sycl::queue& q)
{
    const BlockLayout<2> fine {{{9, 9}}};
    const auto masks = single_block_masks(fine);
    auto levels = build_mg_hierarchy<2>(q, fine, masks, 2);
    expect(fatal(levels.size() == 1_u));
    auto& lv = levels[0];

    // f(i, j, k) = (3 + k)·i − 2·j: linear in the logical index, so the
    // symmetric FW stencil returns the centre value exactly (up to rounding).
    std::vector<real> hf(fine.total_reals(), 0.0_r);
    for (std::size_t i = 0; i < 9; ++i) {
        for (std::size_t j = 0; j < 9; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                hf[fine.interior_node_offset(0, {i, j}) + k] =
                  ((3.0_r + static_cast<real>(k)) * static_cast<real>(i)) -
                  (2.0_r * static_cast<real>(j));
            }
        }
    }
    UsmBuffer<real> df {q, hf};
    UsmBuffer<real> dc {q, std::vector<real>(lv.layout.total_reals(), -7.0_r)};
    restrict_full_weight<2>(q,
                            halo_view<2>(df.data(), fine, 0),
                            interior_view<2>(dc.data(), lv.layout, 0),
                            interior_mask_view<2>(lv.free_mask.data(), lv.layout, 0),
                            lv.factor[0])
      .wait();
    std::vector<real> hc(lv.layout.total_reals());
    dc.download(hc.data());
    const double tol = egg_test::real_tol(1e-12);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            const std::size_t id = lv.layout.interior_node_offset(0, {i, j}) / 2;
            for (std::size_t k = 0; k < 2; ++k) {
                const real got = hc[lv.layout.interior_node_offset(0, {i, j}) + k];
                if (lv.masks.free[id] != 0) {
                    const real want = hf[fine.interior_node_offset(0, {2 * i, 2 * j}) + k];
                    expect(std::abs(static_cast<double>(got - want)) < tol)
                      << "free coarse" << i << j << k << got << "vs" << want;
                } else {
                    expect(got == 0.0_r) << "frozen coarse" << i << j << k;
                }
            }
        }
    }
}

void test_prolong_linear(sycl::queue& q)
{
    const BlockLayout<2> fine {{{9, 9}}};
    const BlockLayout<2> coarse {{{5, 5}}};
    std::vector<real> hc(coarse.total_reals(), 0.0_r);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                hc[coarse.interior_node_offset(0, {i, j}) + k] = (1.5_r * static_cast<real>(i)) +
                                                                 (0.5_r * static_cast<real>(j)) +
                                                                 static_cast<real>(k);
            }
        }
    }
    UsmBuffer<real> dc {q, hc};
    UsmBuffer<real> df {q, std::vector<real>(fine.total_reals(), -3.0_r)};
    prolong_correction<2>(q,
                          interior_view<2>(dc.data(), coarse, 0),
                          interior_view<2>(df.data(), fine, 0),
                          {2, 2})
      .wait();
    std::vector<real> hf(fine.total_reals());
    df.download(hf.data());
    const double tol = egg_test::real_tol(1e-12);
    for (std::size_t i = 0; i < 9; ++i) {
        for (std::size_t j = 0; j < 9; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                // Linear in the fractional coarse index i/2, j/2.
                const real want = (1.5_r * static_cast<real>(i) / 2.0_r) +
                                  (0.5_r * static_cast<real>(j) / 2.0_r) + static_cast<real>(k);
                const real got = hf[fine.interior_node_offset(0, {i, j}) + k];
                expect(std::abs(static_cast<double>(got - want)) < tol)
                  << "fine" << i << j << k << got << "vs" << want;
            }
        }
    }
}

void test_restrict_prolong_identity(sycl::queue& q)
{
    const BlockLayout<2> fine {{{9, 9}}};
    const BlockLayout<2> coarse {{{5, 5}}};
    std::vector<real> hc(coarse.total_reals());
    fill_pattern(hc);
    UsmBuffer<real> dc {q, hc};
    UsmBuffer<real> df {q, std::vector<real>(fine.total_reals(), 0.0_r)};
    UsmBuffer<real> dr {q, std::vector<real>(coarse.total_reals(), -1.0_r)};
    prolong_correction<2>(q,
                          interior_view<2>(dc.data(), coarse, 0),
                          interior_view<2>(df.data(), fine, 0),
                          {2, 2});
    restrict_inject<2>(q,
                       interior_view<2>(df.data(), fine, 0),
                       interior_view<2>(dr.data(), coarse, 0),
                       {2, 2})
      .wait();
    std::vector<real> hr(coarse.total_reals());
    dr.download(hr.data());
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            for (std::size_t k = 0; k < 2; ++k) {
                const std::size_t off = coarse.interior_node_offset(0, {i, j}) + k;
                expect(hr[off] == hc[off]) << "coarse" << i << j << k;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic single-block fine level for the FAS driver tests, generic over D:
// an n^D-node block on the unit cube, boundary shell fixed, interior nodes
// free and interior-eligible (synth patches, identity target), displaced by a
// smooth low-frequency perturbation — the error component plain Jacobi damps
// slowest and the coarse grid is supposed to kill. Group and energy stencil
// are hand-built (no SweepContextHostT needed): the group is synth-only like
// MgLevel::group(), the stencil is one (cell, corner) sample per cell corner
// with edge vectors signed into the cell (det > 0 on a valid grid) and a
// uniform identity W_inv row (w_stride = 0).
template <int D> struct SyntheticFine {
    BlockLayout<D> layout;
    MgMasks masks;
    std::size_t num_nodes = 0;
    std::size_t n_free = 0;
    std::size_t num_samples = 0;
    UsmBuffer<real> X;
    UsmBuffer<std::uint8_t> free_dev;
    UsmBuffer<int> dof_idx, iblock, ilog, block_off, nstride, per_dof_zero;
    UsmBuffer<int> s_gc;
    UsmBuffer<int> s_gn[D];
    UsmBuffer<real> s_s[D];
    UsmBuffer<real> s_w;

    [[nodiscard]] GroupViewT<D> group() const
    {
        GroupViewT<D> gv;
        gv.ndof = n_free;
        gv.n_free = n_free;
        gv.dof_idx = View1<const int> {dof_idx.data(), n_free};
        // Valid (all-zero) per-DOF arrays so dof_subrange pointer arithmetic
        // in jacobi_sweep_once stays defined; never dereferenced (all DOFs
        // take the synth path and the boundary tail is empty).
        gv.P_of = View1<const int> {per_dof_zero.data(), n_free};
        gv.sample_offset = View1<const int> {per_dof_zero.data(), n_free};
        gv.part_of = View1<const int> {per_dof_zero.data(), n_free};
        gv.row_of = View1<const int> {per_dof_zero.data(), n_free};
        gv.interior_block = iblock.data();
        gv.interior_logical = ilog.data();
        gv.block_off = block_off.data();
        gv.nstride = nstride.data();
        return gv;
    }

    [[nodiscard]] StencilViewT<D> stencil() const
    {
        StencilViewT<D> sv;
        sv.num_samples = num_samples;
        sv.gc = View1<const int> {s_gc.data(), num_samples};
        for (int k = 0; k < D; ++k) {
            sv.gn[k] = View1<const int> {s_gn[k].data(), num_samples};
            sv.s[k] = View1<const real> {s_s[k].data(), num_samples};
        }
        sv.W_inv = View1<const real> {s_w.data(), dim::wInv(D)};
        sv.w_stride = 0;  // uniform identity target: one shared row
        return sv;
    }
};

template <int D>
void init_synthetic_fine(SyntheticFine<D>& f, sycl::queue& q, std::size_t n, real amp)
{
    typename BlockLayout<D>::Shape shape {};
    for (int k = 0; k < D; ++k) { shape[static_cast<std::size_t>(k)] = n; }
    f.layout = BlockLayout<D> {{shape}};
    f.masks = single_block_masks<D>(f.layout);
    f.num_nodes = f.layout.total_reals() / D;

    // Positions: unit grid + smooth interior displacement (boundary exact) —
    // a product-of-sines bump with distinct per-component amplitudes.
    constexpr real kComp[3] = {1.0_r, 0.7_r, 0.55_r};
    std::vector<real> hx(f.layout.total_reals(), 0.0_r);
    const real h = 1.0_r / static_cast<real>(n - 1);
    const auto pi = static_cast<real>(std::numbers::pi);
    std::size_t count = 1;
    for (int k = 0; k < D; ++k) { count *= n; }
    for (std::size_t i = 0; i < count; ++i) {
        std::array<std::size_t, D> I {};
        std::size_t rem = i;
        for (int k = D; k-- > 0;) {
            I[static_cast<std::size_t>(k)] = rem % n;
            rem /= n;
        }
        real bump = 1.0_r;
        for (int k = 0; k < D; ++k) {
            bump *= sycl::sin(pi * static_cast<real>(I[static_cast<std::size_t>(k)]) * h);
        }
        const std::size_t off = f.layout.interior_node_offset(0, I);
        for (int k = 0; k < D; ++k) {
            hx[off + static_cast<std::size_t>(k)] =
              (static_cast<real>(I[static_cast<std::size_t>(k)]) * h) + (kComp[k] * amp * bump);
        }
    }
    f.X = UsmBuffer<real> {q, hx};
    f.free_dev = UsmBuffer<std::uint8_t> {q, f.masks.free};

    // Synth-only group over the free interior nodes.
    std::vector<int> dof_idx, ilog;
    for (std::size_t i = 0; i < count; ++i) {
        std::array<std::size_t, D> I {};
        std::size_t rem = i;
        for (int k = D; k-- > 0;) {
            I[static_cast<std::size_t>(k)] = rem % n;
            rem /= n;
        }
        bool interior = true;
        for (int k = 0; k < D; ++k) {
            const std::size_t v = I[static_cast<std::size_t>(k)];
            interior = interior && v >= 1 && v + 1 < n;
        }
        if (!interior) { continue; }
        dof_idx.push_back(static_cast<int>(f.layout.interior_node_offset(0, I) / D));
        for (int k = 0; k < D; ++k) {
            ilog.push_back(static_cast<int>(I[static_cast<std::size_t>(k)]));
        }
    }
    f.n_free = dof_idx.size();
    f.dof_idx = UsmBuffer<int> {q, dof_idx};
    f.iblock = UsmBuffer<int> {q, std::vector<int>(f.n_free, 0)};
    f.ilog = UsmBuffer<int> {q, ilog};
    f.block_off = UsmBuffer<int> {q, std::vector<int> {0}};
    const auto nstr = f.layout.node_strides(0);
    std::vector<int> nstride_host;
    for (int k = 0; k < D; ++k) {
        nstride_host.push_back(static_cast<int>(nstr[static_cast<std::size_t>(k)] / D));
    }
    f.nstride = UsmBuffer<int> {q, nstride_host};
    f.per_dof_zero = UsmBuffer<int> {q, std::vector<int>(f.n_free, 0)};

    // Energy stencil: 2^D corner samples per cell, edges signed into the cell.
    std::vector<int> gc;
    std::vector<int> gn[D];
    std::vector<real> ss[D];
    const std::size_t ncells = n - 1;
    std::size_t cell_count = 1;
    for (int k = 0; k < D; ++k) { cell_count *= ncells; }
    for (std::size_t c = 0; c < cell_count; ++c) {
        std::array<std::size_t, D> C {};
        std::size_t rem = c;
        for (int k = D; k-- > 0;) {
            C[static_cast<std::size_t>(k)] = rem % ncells;
            rem /= ncells;
        }
        for (std::size_t a = 0; a < (1U << D); ++a) {
            std::array<std::size_t, D> corner = C;
            for (int k = 0; k < D; ++k) { corner[static_cast<std::size_t>(k)] += (a >> k) & 1U; }
            gc.push_back(static_cast<int>(f.layout.interior_node_offset(0, corner) / D));
            for (int k = 0; k < D; ++k) {
                std::array<std::size_t, D> nbr = corner;
                const bool hi = ((a >> k) & 1U) != 0;
                nbr[static_cast<std::size_t>(k)] = hi ? (nbr[static_cast<std::size_t>(k)] - 1)
                                                      : (nbr[static_cast<std::size_t>(k)] + 1);
                gn[k].push_back(static_cast<int>(f.layout.interior_node_offset(0, nbr) / D));
                ss[k].push_back(hi ? -1.0_r : 1.0_r);
            }
        }
    }
    f.num_samples = gc.size();
    f.s_gc = UsmBuffer<int> {q, gc};
    for (int k = 0; k < D; ++k) {
        f.s_gn[k] = UsmBuffer<int> {q, gn[k]};
        f.s_s[k] = UsmBuffer<real> {q, ss[k]};
    }
    std::vector<real> wI(dim::wInv(D), 0.0_r);
    for (int k = 0; k < D; ++k) { wI[static_cast<std::size_t>((k * D) + k)] = 1.0_r; }
    f.s_w = UsmBuffer<real> {q, wI};
}

// τ-coherence identity — validates the whole FAS plumbing (transfers, masks,
// HasTau sign): with τ = ∇E_c(Rx) − R̂∇E_f(x), the shifted coarse gradient at
// Rx must equal the restricted fine gradient, ∇(E_c − τ·x)(Rx) = R̂∇E_f(x).
template <int D> void test_tau_coherence(sycl::queue& q, std::size_t n, real amp)
{
    SyntheticFine<D> f;
    init_synthetic_fine<D>(f, q, n, amp);
    const ShapeObjectiveT<D> obj {};

    const std::size_t nbuf = f.num_nodes * D;
    UsmBuffer<real> grad {q, nbuf};
    UsmBuffer<real> hess {q, f.num_nodes * static_cast<std::size_t>(D * D)};
    UsmBuffer<real> e0 {q, f.num_nodes};
    metric_kernel<D>(q, f.group(), f.X.data(), grad.data(), hess.data(), e0.data(), obj);

    auto levels = build_mg_hierarchy<D>(q, f.layout, f.masks, 2);
    expect(fatal(levels.size() == 1_u));
    auto& lvl = levels[0];

    restrict_inject<D>(q,
                       interior_view<D>(f.X.data(), f.layout, 0),
                       interior_view<D>(lvl.x.data(), lvl.layout, 0),
                       lvl.factor[0]);
    restrict_full_weight<D>(q,
                            halo_view<D>(grad.data(), f.layout, 0),
                            interior_view<D>(lvl.tau.data(), lvl.layout, 0),
                            interior_mask_view<D>(lvl.free_mask.data(), lvl.layout, 0),
                            lvl.factor[0])
      .wait();
    const std::size_t cbuf = lvl.num_nodes * D;
    std::vector<real> rg(cbuf);  // R̂∇E_f, before the τ subtraction reuses the buffer
    lvl.tau.download(rg.data());

    const GroupViewT<D> cg = lvl.group();
    metric_kernel<D>(q, cg, lvl.x.data(), lvl.grad.data(), lvl.hess.data(), lvl.e0.data(), obj);
    vec_sub(q, lvl.tau.data(), lvl.grad.data(), lvl.tau.data(), cbuf);
    metric_kernel<D, ShapeObjectiveT<D>, true>(q,
                                               cg,
                                               lvl.x.data(),
                                               lvl.grad.data(),
                                               lvl.hess.data(),
                                               lvl.e0.data(),
                                               obj,
                                               lvl.tau.data());
    q.wait();
    std::vector<real> sg(cbuf);
    lvl.grad.download(sg.data());

    std::vector<int> cdof(lvl.n_free);
    lvl.dof_idx.download(cdof.data());
    const double tol = egg_test::real_tol(1e-10);
    for (const int d : cdof) {
        for (std::size_t k = 0; k < D; ++k) {
            const auto slot = (static_cast<std::size_t>(d) * D) + k;
            const double want = static_cast<double>(rg[slot]);
            const double got = static_cast<double>(sg[slot]);
            expect(std::abs(got - want) <= tol * std::max(1.0, std::abs(want)))
              << "coarse dof" << d << k << got << "vs" << want;
        }
    }
}

// End-to-end FAS on the smooth perturbation: per-cycle energies
// non-increasing (the safeguard's guarantee), min det stays positive (the
// barrier is never crossed), and FAS beats plain Jacobi given the same fine
// sweep budget — the whole point of the coarse grid. @p build_depth = 2
// exercises the two-level cycle (the recursion base case), a larger depth the
// full recursive V-cycle; @p min_levels asserts the hierarchy is actually as
// deep as the case intends.
template <int D>
void test_fas_end_to_end(
  sycl::queue& q, std::size_t n, real amp, int build_depth, std::size_t min_levels)
{
    SyntheticFine<D> f;
    init_synthetic_fine<D>(f, q, n, amp);
    const ShapeObjectiveT<D> obj {};
    const std::size_t nbuf = f.num_nodes * D;

    // Snapshot the initial X for the plain-Jacobi reference run.
    UsmBuffer<real> xj {q, nbuf};
    q.memcpy(xj.data(), f.X.data(), nbuf * sizeof(real)).wait();

    auto levels = build_mg_hierarchy<D>(q, f.layout, f.masks, build_depth);
    expect(fatal(levels.size() >= min_levels)) << "levels" << levels.size();

    FasOptions opts;
    opts.n_cycles = 5;
    opts.nu_pre = 2;
    opts.nu_post = 2;
    opts.nu_coarse = 32;
    opts.omega = 0.8_r;

    FasFineView<D> fv;
    fv.num_nodes = f.num_nodes;
    fv.X = f.X.data();
    fv.seeds = nullptr;
    fv.group = f.group();
    fv.stencil = f.stencil();

    auto noop = [](sycl::queue&, real*) {};
    auto [fe, fm] =
      run_fas<D>(q, fv, f.layout, f.free_dev.data(), std::span {levels}, obj, opts, noop);

    expect(fatal(fe.size() == 5_u));
    for (std::size_t c = 0; c < fe.size(); ++c) {
        expect(fm[c] > 0.0_r) << "cycle" << c << "min det" << fm[c];
        if (c > 0) {
            const double slack =
              egg_test::real_tol(1e-10) * std::max(1.0, std::abs(static_cast<double>(fe[c - 1])));
            expect(static_cast<double>(fe[c]) <= static_cast<double>(fe[c - 1]) + slack)
              << "cycle" << c << fe[c] << "vs" << fe[c - 1];
        }
    }

    // Plain Jacobi with the same fine-sweep budget (nu_pre + nu_post per
    // cycle), same group/omega, then the same energy reduction.
    const GroupViewT<D> g = f.group();
    UsmBuffer<real> xswap {q, nbuf};
    q.memcpy(xswap.data(), xj.data(), nbuf * sizeof(real)).wait();
    UsmBuffer<real> grad {q, nbuf};
    UsmBuffer<real> hess {q, f.num_nodes * static_cast<std::size_t>(D * D)};
    UsmBuffer<real> e0 {q, f.num_nodes};
    real* x_in = xj.data();
    real* x_out = xswap.data();
    const int n_sweeps = opts.n_cycles * (opts.nu_pre + opts.nu_post);
    for (int s = 0; s < n_sweeps; ++s) {
        metric_kernel<D>(q, g, x_in, grad.data(), hess.data(), e0.data(), obj);
        interior_update_kernel<D>(q,
                                  g,
                                  x_in,
                                  x_out,
                                  opts.omega,
                                  grad.data(),
                                  hess.data(),
                                  e0.data(),
                                  obj);
        std::swap(x_in, x_out);
    }
    UsmBuffer<real> d_e {q, 1};
    UsmBuffer<real> d_m {q, 1};
    reduce_energy_mindet<D>(q, f.stencil(), x_in, d_e.data(), d_m.data(), obj);
    q.wait();
    real ej = 0.0_r;
    real mj = 0.0_r;
    q.memcpy(&ej, d_e.data(), sizeof(real)).wait();
    q.memcpy(&mj, d_m.data(), sizeof(real)).wait();

    expect(mj > 0.0_r);
    // The coarse grid must at least halve what Jacobi leaves behind on this
    // smooth mode (in practice it does far better); the floor guards the
    // both-converged-to-noise case.
    const double floor = egg_test::real_tol(1e-9);
    expect(static_cast<double>(fe.back()) <= (0.5 * static_cast<double>(ej)) + floor)
      << "FAS" << fe.back() << "vs Jacobi" << ej;
}

void test_zero_unmasked(sycl::queue& q)
{
    constexpr std::size_t kN = 6;
    std::vector<real> buf(kN * 2);
    fill_pattern(buf);
    const std::vector<real> orig = buf;
    const std::vector<std::uint8_t> mask {1, 0, 1, 0, 0, 1};
    UsmBuffer<real> db {q, buf};
    UsmBuffer<std::uint8_t> dm {q, mask};
    zero_unmasked<2>(q, db.data(), dm.data(), kN).wait();
    db.download(buf.data());
    for (std::size_t i = 0; i < kN; ++i) {
        for (std::size_t k = 0; k < 2; ++k) {
            const real want = mask[i] != 0 ? orig[(i * 2) + k] : 0.0_r;
            expect(buf[(i * 2) + k] == want) << "node" << i << k;
        }
    }
}

// The fused line-search reduction must agree with the reference path (form
// the trial with masked_axpy, reduce with reduce_energy_mindet) for both α
// slots, and the α = 0 slot must reproduce the pre-state energy exactly.
void test_linesearch_pair(sycl::queue& q)
{
    SyntheticFine<2> f;
    init_synthetic_fine<2>(f, q, 9, 0.03_r);
    const ShapeObjectiveT<2> obj {};
    const std::size_t nbuf = f.num_nodes * 2;

    // A small smooth correction on the Free set only, zero elsewhere — the
    // shape the driver hands the kernel after zero_unmasked + broadcast.
    std::vector<real> hc(nbuf, 0.0_r);
    for (std::size_t i = 0; i < f.num_nodes; ++i) {
        if (f.masks.free[i] == 0) { continue; }
        for (std::size_t k = 0; k < 2; ++k) {
            hc[(i * 2) + k] = 1e-3_r * (static_cast<real>((((i * 2) + k) % 7)) - 3.0_r);
        }
    }
    UsmBuffer<real> corr {q, hc};
    UsmBuffer<real> d_ls {q, 4};
    egg::detail::reduce_linesearch_pair<2>(q,
                                           f.stencil(),
                                           f.X.data(),
                                           corr.data(),
                                           0.0_r,
                                           0.5_r,
                                           d_ls.data(),
                                           obj);
    q.wait();
    real ls[4];
    q.memcpy(ls, d_ls.data(), 4 * sizeof(real)).wait();

    UsmBuffer<real> trial {q, nbuf};
    UsmBuffer<real> d_e {q, 1};
    UsmBuffer<real> d_m {q, 1};
    const double tol = egg_test::real_tol(1e-10);
    const real alphas[2] = {0.0_r, 0.5_r};
    for (std::size_t t = 0; t < 2; ++t) {
        masked_axpy<2>(q,
                       trial.data(),
                       f.X.data(),
                       corr.data(),
                       alphas[t],
                       f.free_dev.data(),
                       f.num_nodes);
        reduce_energy_mindet<2>(q, f.stencil(), trial.data(), d_e.data(), d_m.data(), obj);
        q.wait();
        real e_ref = 0.0_r;
        real m_ref = 0.0_r;
        q.memcpy(&e_ref, d_e.data(), sizeof(real)).wait();
        q.memcpy(&m_ref, d_m.data(), sizeof(real)).wait();
        const auto e_got = static_cast<double>(ls[2 * t]);
        const auto m_got = static_cast<double>(ls[(2 * t) + 1]);
        expect(std::abs(e_got - static_cast<double>(e_ref)) <=
               tol * std::max(1.0, std::abs(static_cast<double>(e_ref))))
          << "alpha" << alphas[t] << "energy" << e_got << "vs" << e_ref;
        expect(std::abs(m_got - static_cast<double>(m_ref)) <=
               tol * std::max(1.0, std::abs(static_cast<double>(m_ref))))
          << "alpha" << alphas[t] << "min det" << m_got << "vs" << m_ref;
    }
}

void test_masked_axpy(sycl::queue& q)
{
    constexpr std::size_t kN = 6;
    const std::vector<real> x {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::vector<real> c(kN * 2, 2.0_r);
    const std::vector<std::uint8_t> mask {1, 0, 1, 0, 0, 1};
    UsmBuffer<real> dx {q, x};
    UsmBuffer<real> dcv {q, c};
    UsmBuffer<real> dout {q, std::vector<real>(kN * 2, -1.0_r)};
    UsmBuffer<std::uint8_t> dm {q, mask};
    masked_axpy<2>(q, dout.data(), dx.data(), dcv.data(), 0.5_r, dm.data(), kN).wait();
    std::vector<real> out(kN * 2);
    dout.download(out.data());
    for (std::size_t i = 0; i < kN; ++i) {
        for (std::size_t k = 0; k < 2; ++k) {
            const real want = x[(i * 2) + k] + (mask[i] != 0 ? 1.0_r : 0.0_r);
            expect(out[(i * 2) + k] == want) << "node" << i << k;
        }
    }
}

}  // namespace

int main()
{
    const auto devices = egg_test::usable_devices();
    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    for (const auto& dev : devices) {
        // In-order, like the executor's queue (structured_sweep.hpp): the FAS
        // driver and the smoother kernels chain without per-launch waits.
        sycl::queue q(dev, {sycl::property::queue::in_order()});
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("restrict_inject bitwise on " + name) = [&] { test_restrict_inject(q); };
        test("restrict_full_weight linear on " + name) = [&] { test_full_weight(q); };
        test("prolong_correction linear on " + name) = [&] { test_prolong_linear(q); };
        test("restrict∘prolong identity on " + name) = [&] { test_restrict_prolong_identity(q); };
        test("masked_axpy mask on " + name) = [&] { test_masked_axpy(q); };
        test("zero_unmasked mask on " + name) = [&] { test_zero_unmasked(q); };
        test("line-search pair reduction on " + name) = [&] { test_linesearch_pair(q); };
        test("tau coherence identity 2D on " + name) = [&] { test_tau_coherence<2>(q, 9, 0.03_r); };
        test("tau coherence identity 3D on " + name) = [&] { test_tau_coherence<3>(q, 9, 0.02_r); };
        test("FAS end-to-end vs Jacobi 2D on " + name) = [&] {
            test_fas_end_to_end<2>(q, 17, 0.02_r, 2, 1);
        };
        test("FAS end-to-end vs Jacobi 3D on " + name) = [&] {
            test_fas_end_to_end<3>(q, 13, 0.015_r, 2, 1);
        };
        // Full-depth recursion: 17 interior nodes ladder 9 → 5 → 3,
        // so the cycle takes at least one genuinely recursive leg.
        test("FAS recursive V-cycle 2D on " + name) = [&] {
            test_fas_end_to_end<2>(q, 17, 0.02_r, 8, 2);
        };
        test("FAS recursive V-cycle 3D on " + name) = [&] {
            test_fas_end_to_end<3>(q, 13, 0.015_r, 8, 2);
        };
    }
}
