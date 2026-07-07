// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_multigrid.cpp — FAS hierarchy build (host): coarsening factors,
// semi-coarsening, stop conditions, free-mask and DOF-list derivation.
//
// No kernels are launched here — build_mg_hierarchy only allocates/uploads
// USM. The transfer-kernel tests live in test_multigrid_device.cpp, a
// SEPARATE single-TU binary like every other kernel-launching test: the
// AdaptiveCpp SSCP flow mis-resolves the per-TU HCF object id in multi-TU
// binaries ("Could not obtain hcf kernel info" on kernel submission), so
// kernels must not be launched from a TU of the multi-TU cpp_tests runner.

#include "fas.hpp"
#include "mg_level.hpp"
#include "structured.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <array>
#include <cstddef>
#include <vector>

using namespace boost::ut;
using namespace egg;

namespace
{

// One long-lived, DELIBERATELY LEAKED queue for the whole binary. ACPP's
// runtime (backend_manager) is refcounted from live queues, and destroying
// the last queue tears the whole runtime down (CUDA context included) —
// which deadlocks in ~backend_manager on the CUDA backend here when no CUDA
// kernel ever ran (this binary only does USM copies). A per-test queue hits
// that mid-run; a static queue hits it again at static destruction. Leaking
// keeps the runtime alive until the OS reclaims the process.
sycl::queue& shared_queue()
{
    static auto* q = new sycl::queue {egg_test::usable_devices().front()};
    return *q;
}

// Fine masks for one all-interior block: physical boundary (face) nodes are
// constrained, everything else Free and interior-eligible — the single-block
// analogue of what the executor derives from the host sweep context.
MgMasks single_block_masks(const BlockLayout<2>& layout)
{
    MgMasks m;
    const std::size_t nn = layout.total_reals() / 2;
    m.free.assign(nn, 0);
    m.free_interior.assign(nn, 0);
    const auto shape = layout.interior_shape(0);
    for (std::size_t i = 1; i + 1 < shape[0]; ++i) {
        for (std::size_t j = 1; j + 1 < shape[1]; ++j) {
            const std::size_t id = layout.interior_node_offset(0, {i, j}) / 2;
            m.free[id] = 1;
            m.free_interior[id] = 1;
        }
    }
    return m;
}

// Hand-built conforming interface between block 0's axis-0 HIGH face and
// block 1's axis-0 LOW face of a {{9,9},{9,9}} layout, mirroring what the
// Python builder emits: one ghost fill per (face node, direction) plus one
// owner→copy share pair per face node (block 0 owns every shared node).
// `flip` reverses the tangential axis across the interface — the reversed-
// orientation case the odd-count parity rule must survive.
HaloTablesHost<2> two_block_tables(const BlockLayout<2>& fine, bool flip)
{
    HaloTablesHost<2> t;
    const std::size_t n = fine.interior_shape(0)[1];
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t jb = flip ? (n - 1 - j) : j;  // twin tangential index
        t.src_block.push_back(1);                       // b0 ghost ← b1 layer 1
        t.src_padded.push_back({2, jb + 1});
        t.dst_block.push_back(0);
        t.dst_padded.push_back({10, j + 1});
        t.src_block.push_back(0);  // b1 ghost ← b0 layer 1
        t.src_padded.push_back({8, jb + 1});
        t.dst_block.push_back(1);
        t.dst_padded.push_back({0, j + 1});
        t.share_src_off.push_back(fine.interior_node_offset(0, {8, jb}));
        t.share_dst_off.push_back(fine.interior_node_offset(1, {0, j}));
    }
    return t;
}

// Masks for the two-block interface layout: both interiors free/interior-
// eligible, block 0's face nodes Free (the owner side; the tangential ends sit
// on the domain boundary and stay constrained), block 1's copies not DOFs.
MgMasks interface_masks(const BlockLayout<2>& fine)
{
    MgMasks m;
    const std::size_t nn = fine.total_reals() / 2;
    m.free.assign(nn, 0);
    m.free_interior.assign(nn, 0);
    for (std::size_t b = 0; b < 2; ++b) {
        const auto shape = fine.interior_shape(b);
        for (std::size_t i = 1; i + 1 < shape[0]; ++i) {
            for (std::size_t j = 1; j + 1 < shape[1]; ++j) {
                const std::size_t id = fine.interior_node_offset(b, {i, j}) / 2;
                m.free[id] = 1;
                m.free_interior[id] = 1;
            }
        }
    }
    for (std::size_t j = 1; j + 1 < fine.interior_shape(0)[1]; ++j) {
        m.free[fine.interior_node_offset(0, {8, j}) / 2] = 1;  // owner face DOFs
    }
    return m;
}

}  // namespace

static const suite<"multigrid hierarchy"> hierarchy_suite = [] {
    "9x9 coarsens to 5x5 with factor (2,2)"_test = [] {
        const BlockLayout<2> fine {{{9, 9}}};
        sycl::queue& q = shared_queue();
        const auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 2);
        expect(fatal(levels.size() == 1_u));
        expect(levels[0].layout.interior_shape(0) == std::array<std::size_t, 2> {5, 5});
        expect(levels[0].factor[0] == CoarsenFactor<2> {2, 2});
    };

    "even axis is semi-coarsened"_test = [] {
        const BlockLayout<2> fine {{{9, 8}}};
        sycl::queue& q = shared_queue();
        const auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 2);
        expect(fatal(levels.size() == 1_u));
        expect(levels[0].layout.interior_shape(0) == std::array<std::size_t, 2> {5, 8});
        expect(levels[0].factor[0] == CoarsenFactor<2> {2, 1});
    };

    "block with no coarsenable axis stops the hierarchy"_test = [] {
        const BlockLayout<2> fine {{{4, 4}}};
        sycl::queue& q = shared_queue();
        const auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 4);
        expect(levels.empty());
    };

    // An uncoarsenable block does not veto the level: it is CARRIED unchanged
    // (factor 1 everywhere, identity transfers, full DOF set) while the other
    // block ladders down; the hierarchy stops when nothing halves.
    "uncoarsenable block is carried while others halve"_test = [] {
        const BlockLayout<2> fine {{{9, 9}, {4, 4}}};
        sycl::queue& q = shared_queue();
        MgMasks masks;
        const std::size_t nn = fine.total_reals() / 2;
        masks.free.assign(nn, 0);
        masks.free_interior.assign(nn, 0);
        for (std::size_t b = 0; b < 2; ++b) {
            const auto shape = fine.interior_shape(b);
            for (std::size_t i = 1; i + 1 < shape[0]; ++i) {
                for (std::size_t j = 1; j + 1 < shape[1]; ++j) {
                    const std::size_t id = fine.interior_node_offset(b, {i, j}) / 2;
                    masks.free[id] = 1;
                    masks.free_interior[id] = 1;
                }
            }
        }
        const auto levels = build_mg_hierarchy<2>(q, fine, masks, 4);
        expect(fatal(levels.size() == 2_u));  // 9 → 5 → 3; then nothing halves
        expect(levels[0].layout.interior_shape(0) == std::array<std::size_t, 2> {5, 5});
        expect(levels[0].layout.interior_shape(1) == std::array<std::size_t, 2> {4, 4});
        expect(levels[0].factor[1] == CoarsenFactor<2> {1, 1});
        // Block 0 contributes its 3×3 coarse interior, the carried block its
        // full 2×2 free interior — at every level.
        expect(levels[0].n_free == 13_u) << levels[0].n_free;
        expect(levels[1].layout.interior_shape(0) == std::array<std::size_t, 2> {3, 3});
        expect(levels[1].layout.interior_shape(1) == std::array<std::size_t, 2> {4, 4});
        expect(levels[1].n_free == 5_u) << levels[1].n_free;
    };

    // The coarsest-solve budget is twice the largest interior extent over
    // blocks and axes — a carried block's full size counts, so a big carried
    // block keeps the budget at the fixed nu_coarse cap.
    "coarse sweep budget scales with the interior diameter"_test = [] {
        expect(egg::detail::coarse_sweep_budget<2>(BlockLayout<2> {{{3, 3}}}) == 6_i);
        expect(egg::detail::coarse_sweep_budget<2>(BlockLayout<2> {{{3, 3}, {4, 4}}}) == 8_i);
        expect(egg::detail::coarse_sweep_budget<2>(BlockLayout<2> {{{3, 9}}}) == 18_i);
    };

    "small odd axis below kMinCoarsenNodes is kept"_test = [] {
        const BlockLayout<2> fine {{{3, 9}}};
        sycl::queue& q = shared_queue();
        const auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 2);
        expect(fatal(levels.size() == 1_u));
        expect(levels[0].layout.interior_shape(0) == std::array<std::size_t, 2> {3, 5});
        expect(levels[0].factor[0] == CoarsenFactor<2> {1, 2});
    };

    // 9×9 with a frozen face ring: coarse 5×5 free DOFs are exactly the 3×3
    // interior I ∈ [1,3]² (image at 2I, full-weighting stencil ⊂ [1,7]², all
    // Free); coarse faces frozen.
    "single-block free mask and DOF list"_test = [] {
        const BlockLayout<2> fine {{{9, 9}}};
        sycl::queue& q = shared_queue();
        auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 2);
        expect(fatal(levels.size() == 1_u));
        auto& lv = levels[0];
        expect(lv.n_free == 9_u);
        for (std::size_t i = 0; i < 5; ++i) {
            for (std::size_t j = 0; j < 5; ++j) {
                const std::size_t id = lv.layout.interior_node_offset(0, {i, j}) / 2;
                const bool expect_free = i >= 1 && i <= 3 && j >= 1 && j <= 3;
                expect(lv.masks.free[id] == (expect_free ? 1 : 0)) << "coarse node" << i << j;
            }
        }
        // Spot-check the device DOF arrays round-trip: first DOF is (1,1).
        std::vector<int> iblock(lv.n_free), ilog(lv.n_free * 2);
        lv.interior_block.download(iblock.data());
        lv.interior_logical.download(ilog.data());
        expect(iblock[0] == 0_i);
        expect(ilog[0] == 1_i && ilog[1] == 1_i);
    };

    // Three-level recursion: 9×9 → 5×5 → 3×3 with a single free DOF at (1,1).
    "recursive coarsening derives masks per level"_test = [] {
        const BlockLayout<2> fine {{{9, 9}}};
        sycl::queue& q = shared_queue();
        auto levels = build_mg_hierarchy<2>(q, fine, single_block_masks(fine), 3);
        expect(fatal(levels.size() == 2_u));
        expect(levels[1].layout.interior_shape(0) == std::array<std::size_t, 2> {3, 3});
        expect(levels[1].n_free == 1_u);
        std::vector<int> ilog(2);
        levels[1].interior_logical.download(ilog.data());
        expect(ilog[0] == 1_i && ilog[1] == 1_i);
    };

    // Coarse-table derivation exactness: every coarse ghost entry is the parent entry
    // at the even tangential index with the twin's coordinates divided by its
    // block's factors — same-orientation and reversed interfaces.
    "coarse interface tables derive by even-index lookup"_test = [] {
        const BlockLayout<2> fine {{{9, 9}, {9, 9}}};
        const BlockLayout<2> coarse {{{5, 5}, {5, 5}}};
        const std::vector<CoarsenFactor<2>> factor {{2, 2}, {2, 2}};
        for (const bool flip : {false, true}) {
            const auto ct =
              coarsen_halo_tables<2>(two_block_tables(fine, flip), fine, coarse, factor);
            expect(fatal(ct.src_block.size() == 10_u)) << "flip" << flip;
            expect(fatal(ct.share_src_off.size() == 5_u)) << "flip" << flip;
            for (std::size_t e = 0; e < ct.src_block.size(); ++e) {
                const std::size_t T = ct.dst_padded[e][1] - 1;
                const std::size_t Tb = flip ? (4 - T) : T;
                if (ct.dst_block[e] == 0) {
                    expect(ct.dst_padded[e] == std::array<std::size_t, 2> {6, T + 1});
                    expect(ct.src_block[e] == 1_i);
                    expect(ct.src_padded[e] == std::array<std::size_t, 2> {2, Tb + 1})
                      << "flip" << flip << "T" << T;
                } else {
                    expect(ct.dst_padded[e] == std::array<std::size_t, 2> {0, T + 1});
                    expect(ct.src_block[e] == 0_i);
                    // COARSE padded index of b0's coarse layer-1 (logical 3).
                    expect(ct.src_padded[e] == std::array<std::size_t, 2> {4, Tb + 1})
                      << "flip" << flip << "T" << T;
                }
            }
            for (std::size_t p = 0; p < ct.share_src_off.size(); ++p) {
                const auto [db, dp] = egg::detail::locate_node<2>(coarse, ct.share_dst_off[p] / 2);
                const auto [sb, sp] = egg::detail::locate_node<2>(coarse, ct.share_src_off[p] / 2);
                expect(sb == 0_u && db == 1_u);
                expect(sp[0] == 5_u && dp[0] == 1_u);  // padded: owner face 4, copy face 0
                const std::size_t J = dp[1] - 1;
                expect(sp[1] - 1 == (flip ? (4 - J) : J)) << "flip" << flip << "J" << J;
            }
        }
    };

    // Interface free-mask guards: owner face-interior nodes unfreeze, tangential edge
    // nodes and non-owner copies stay frozen, and a constrained neighbour
    // across the interface re-freezes exactly the face DOFs that would read
    // its gradient slot. Chaining: the 3×3 level re-derives its own face DOF.
    "interface face DOFs unfreeze with guards"_test = [] {
        const BlockLayout<2> fine {{{9, 9}, {9, 9}}};
        sycl::queue& q = shared_queue();
        auto levels =
          build_mg_hierarchy<2>(q, fine, interface_masks(fine), 3, two_block_tables(fine, false));
        expect(fatal(levels.size() == 2_u));
        auto& lv = levels[0];
        // 9 interior DOFs per block + 3 owner face DOFs (J ∈ [1,3]).
        expect(lv.n_free == 21_u) << lv.n_free;
        for (std::size_t j = 0; j < 5; ++j) {
            const std::size_t own = lv.layout.interior_node_offset(0, {4, j}) / 2;
            const std::size_t cpy = lv.layout.interior_node_offset(1, {0, j}) / 2;
            expect(lv.masks.free[own] == ((j >= 1 && j <= 3) ? 1 : 0)) << "owner" << j;
            expect(lv.masks.free_interior[own] == 0_u) << "owner fi" << j;
            expect(lv.masks.free[cpy] == 0_u) << "copy" << j;
        }
        expect(lv.topo.num_entries() == 10_u);
        expect(lv.topo.num_share() == 5_u);
        // Level 2 (3×3 blocks): 1 interior DOF per block + 1 face DOF.
        expect(levels[1].n_free == 3_u) << levels[1].n_free;
        expect(levels[1].masks.free[levels[1].layout.interior_node_offset(0, {2, 1}) / 2] == 1_u);

        // Constrain block 1's layer-1 node (1, 4): the fine gradient behind
        // the face at tangential 4 is never written, so the face DOF at
        // J = 2 (whose stencil reads it) must freeze; J = 1 and 3 survive.
        MgMasks masks = interface_masks(fine);
        masks.free[fine.interior_node_offset(1, {1, 4}) / 2] = 0;
        masks.free_interior[fine.interior_node_offset(1, {1, 4}) / 2] = 0;
        auto guarded = build_mg_hierarchy<2>(q, fine, masks, 2, two_block_tables(fine, false));
        expect(fatal(guarded.size() == 1_u));
        const auto& gm = guarded[0].masks;
        expect(gm.free[guarded[0].layout.interior_node_offset(0, {4, 1}) / 2] == 1_u);
        expect(gm.free[guarded[0].layout.interior_node_offset(0, {4, 2}) / 2] == 0_u);
        expect(gm.free[guarded[0].layout.interior_node_offset(0, {4, 3}) / 2] == 1_u);
    };

    "two blocks coarsen independently"_test = [] {
        const BlockLayout<2> fine {{{9, 9}, {13, 8}}};
        sycl::queue& q = shared_queue();
        MgMasks masks;
        const std::size_t nn = fine.total_reals() / 2;
        masks.free.assign(nn, 0);
        masks.free_interior.assign(nn, 0);
        for (std::size_t b = 0; b < 2; ++b) {
            const auto shape = fine.interior_shape(b);
            for (std::size_t i = 1; i + 1 < shape[0]; ++i) {
                for (std::size_t j = 1; j + 1 < shape[1]; ++j) {
                    const std::size_t id = fine.interior_node_offset(b, {i, j}) / 2;
                    masks.free[id] = 1;
                    masks.free_interior[id] = 1;
                }
            }
        }
        const auto levels = build_mg_hierarchy<2>(q, fine, masks, 2);
        expect(fatal(levels.size() == 1_u));
        expect(levels[0].layout.interior_shape(0) == std::array<std::size_t, 2> {5, 5});
        expect(levels[0].layout.interior_shape(1) == std::array<std::size_t, 2> {7, 8});
        expect(levels[0].factor[1] == CoarsenFactor<2> {2, 1});
    };
};
