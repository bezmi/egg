#pragma once

/// @file structured_patch.hpp
/// Structured patch-evaluation bridge. Over a BlockField<D>'s halo-padded
/// buffer every node lives at a fixed (block, padded index) slot, the store is
/// D-contiguous per node with no gaps, and each block base is a multiple of D
/// — so a node's flat index in node units is its double-offset / D, and
/// patch_eval (which reads X[D*i + k]) is a drop-in once the stencil's gc/gn
/// carry these structured indices instead of global ids. That is the
/// coalescing win: consecutive interior nodes on the fastest axis are D
/// doubles apart. This header is the single place converting BlockLayout
/// offsets into patch_eval node indices; SYCL-free and host-unit-testable.

#include "patch.hpp"
#include "structured.hpp"

#include <array>
#include <cstddef>

namespace egg
{

/// Flat node index (in node units, for load_pt/store_pt) of the node at PADDED
/// index `padded_idx` in block `b`. Equals the double-offset / D because the
/// packed buffer is D-contiguous per node with no gaps and every block base is a
/// multiple of D — so `patch_eval(buf, …)` reading `buf[D*i + k]` lands exactly
/// on that node's coordinates.
template <int D>
[[nodiscard]] inline int padded_node_index(const BlockLayout<D>& layout,
                                           std::size_t b,
                                           const std::array<std::size_t, D>& padded_idx)
{ return static_cast<int>(layout.padded_node_offset(b, padded_idx) / static_cast<std::size_t>(D)); }

/// Flat node index of an INTERIOR node (logical index in `[0, n_k)`), shifted
/// +1 per axis into the padded array. The structured analogue of a global DOF id
/// for patch_eval's gc/gn over a BlockField buffer.
template <int D>
[[nodiscard]] inline int interior_node_index(const BlockLayout<D>& layout,
                                             std::size_t b,
                                             const std::array<std::size_t, D>& logical_idx)
{
    return static_cast<int>(layout.interior_node_offset(b, logical_idx) /
                            static_cast<std::size_t>(D));
}

// Implicit interior patch synthesis. An interior structured node's patch is a
// fixed stencil — corner of 2^D cells x 2^D corner samples = 4^D occurrences —
// derivable from the logical index alone, with no stored per-occurrence
// gc/gn/sample_id/role/sign arrays. Occurrence ORDER reproduces the host
// build_context exactly (outer: 2^D cells row-major; inner: 2^D corner offsets
// row-major), so accumulation is bit-identical to the stored path, not merely
// the same set. s[k]/role depend only on the occurrence bit pattern; only
// gc/gn carry the node's absolute indices.

/// Number of metric occurrences in an interior structured node's patch: 4^D.
template <int D> [[nodiscard]] constexpr int interior_patch_size() { return 1 << (2 * D); }

/// One synthesized patch occurrence over the halo-padded structured buffer:
/// the corner node index, its D axis-neighbour indices, the D axis signs, and
/// the DOF's role (0 = corner, 1..D = axis-(role−1) neighbour, −1 = absent).
template <int D> struct InteriorOccurrence {
    int gc;
    int gn[D];
    real s[D];
    int role;
};

/// Synthesize occurrence `occ` (∈ [0, 4^D)) of an interior patch, given the
/// node's integer `logical` index and a node-indexing callable `idx` mapping a
/// logical index (`const int (&)[D]`) to its flat node index. Fills `gc`, the D
/// axis neighbours `gn`, the D axis signs `s`, and the DOF's `role` (0 = corner,
/// 1..D = axis-(role−1) neighbour, −1 = absent).
///
/// `occ` splits as `cell = occ / 2^D` and `o = occ % 2^D`, each a row-major bit
/// pattern over D axes (axis 0 most significant), matching product((0,1),
/// repeat=D) order: the cell base is `logical − 1 + cell_bits` (the 2^D cells the
/// node corners), the corner within that cell is `base + o`, and the neighbour on
/// axis k flips bit k of `o`, stepping the corner by `s[k] = (o[k]==0 ? +1 : −1)`.
/// This is the single source the host (BlockLayout) and device (flat stride
/// arrays) callers share; only `idx` differs. Pure integer/sign math, no reads.
template <int D, class IdxFn>
inline void interior_occurrence_raw(int occ,
                                    const int (&logical)[D],
                                    const IdxFn& idx,
                                    int& gc,
                                    int (&gn)[D],
                                    real (&s)[D],
                                    int& role)
{
    constexpr int kC = 1 << D;  // corners(D)
    const int cell_bits = occ / kC;
    const int o = occ % kC;

    int corner[D];
    int o_bit[D];
    for (int k = 0; k < D; ++k) {
        const int shift = D - 1 - k;
        o_bit[k] = (o >> shift) & 1;
        corner[k] = logical[k] - 1 + ((cell_bits >> shift) & 1) + o_bit[k];
    }

    gc = idx(corner);
    bool is_corner = true;
    for (int k = 0; k < D; ++k) {
        if (corner[k] != logical[k]) { is_corner = false; }
    }
    role = is_corner ? 0 : -1;
    for (int k = 0; k < D; ++k) {
        s[k] = (o_bit[k] == 0) ? 1.0_r : -1.0_r;
        int nbr[D];
        for (int j = 0; j < D; ++j) { nbr[j] = corner[j]; }
        nbr[k] = corner[k] + (o_bit[k] == 0 ? 1 : -1);
        gn[k] = idx(nbr);
        if (role < 0) {
            bool eq = true;
            for (int j = 0; j < D; ++j) {
                if (nbr[j] != logical[j]) { eq = false; }
            }
            if (eq) { role = k + 1; }
        }
    }
}

/// BlockLayout-backed synthesis of occurrence `occ` for node `logical` in block
/// `b`. The std::array/host convenience wrapper over `interior_occurrence_raw`.
/// Valid when `logical`'s whole 4^D patch stays inside block `b` (each axis index
/// in `[1, interior_shape - 2]`, so every corner resolves without a ghost slot).
template <int D>
[[nodiscard]] inline InteriorOccurrence<D> interior_patch_occurrence(
  const BlockLayout<D>& layout, std::size_t b, const std::array<std::size_t, D>& logical, int occ)
{
    int li[D];
    for (int k = 0; k < D; ++k) { li[k] = static_cast<int>(logical[static_cast<std::size_t>(k)]); }
    const auto idx = [&](const int (&l)[D]) {
        std::array<std::size_t, D> a;
        for (int k = 0; k < D; ++k) {
            a[static_cast<std::size_t>(k)] = static_cast<std::size_t>(l[k]);
        }
        return interior_node_index<D>(layout, b, a);
    };
    InteriorOccurrence<D> out {};
    interior_occurrence_raw<D>(occ, li, idx, out.gc, out.gn, out.s, out.role);
    return out;
}

/// Flat node index from device-resident stride arrays: the block's node base +
/// Σ (logical_k + 1)·stride_k. The device analogue of interior_node_index —
/// `nstride` holds the padded node strides in NODE units (innermost == 1) and
/// `block_off` the per-block node base, both packed per block. Lets a kernel
/// resolve a synthesized stencil node without a host BlockLayout.
template <int D>
[[nodiscard]] inline int
  dev_node_index(const int* block_off, const int* nstride, int b, const int (&logical)[D])
{
    int idx = block_off[b];
    for (int k = 0; k < D; ++k) { idx += (logical[k] + 1) * nstride[(b * D) + k]; }
    return idx;
}

/// Evaluate an interior structured patch from SYNTHESIZED indices: the same pass
/// as patch_eval, but each of the 4^D occurrences' gc/gn/sign/role is rebuilt
/// in-kernel from the block layout (`block_off`/`nstride` at block `b`, logical
/// index `logical`) and W_inv is the identity — nothing is read from per-DOF
/// arrays. Shares accumulate_sample with patch_eval, so the math is identical.
/// Valid only for DOFs the structured build flagged interior-eligible: their
/// stored patch reproduces this occurrence for occurrence with identity W_inv.
template <int D, ObjectiveD<D> M = ShapeObjectiveT<D>>
inline PatchResultT<D> patch_eval_synth(const int* block_off,
                                        const int* nstride,
                                        int b,
                                        const int (&logical)[D],
                                        const real* X,
                                        M objective = {})
{
    real wI[D * D];
    for (int i = 0; i < D * D; ++i) { wI[i] = 0.0_r; }
    for (int k = 0; k < D; ++k) { wI[(k * D) + k] = 1.0_r; }

    PatchResultT<D> r {};
    r.grad = VecN<D> {};
    r.hess = MatN<D> {};
    r.energy = 0.0_r;
    r.mindet = std::numeric_limits<real>::infinity();

    const auto idx = [&](const int (&l)[D]) { return dev_node_index<D>(block_off, nstride, b, l); };
    for (int occ = 0; occ < interior_patch_size<D>(); ++occ) {
        int gc;
        int gn[D];
        real s[D];
        int role;
        interior_occurrence_raw<D>(occ, logical, idx, gc, gn, s, role);
        const PtN<D> corner = load_pt<D>(X, gc);
        std::array<PtN<D>, D> nbr;
        std::array<real, D> sa;
        for (int k = 0; k < D; ++k) {
            nbr[static_cast<std::size_t>(k)] = load_pt<D>(X, gn[k]);
            sa[static_cast<std::size_t>(k)] = s[k];
        }
        real detA;
        const VecTN<D> t = assemble_vecT<D>(corner, nbr, sa, wI, detA);
        if (detA < r.mindet) { r.mindet = detA; }
        accumulate_sample<D>(objective, t, wI, role, s, r);
    }
    return r;
}

/// Trial energy + min det over a SYNTHESIZED interior patch, substituting `trial`
/// for the moving node `dof` (all other nodes read from `X`). The line-search
/// analogue of patch_eval_synth: same synthesized indices + identity W_inv, but
/// only the scalar energy and det are needed, so the role/grad/hess work is
/// skipped. Mirrors the stored trial loop in the interior update kernel.
template <int D, class Obj>
inline void synth_trial_energy_mindet(const int* block_off,
                                      const int* nstride,
                                      int b,
                                      const int (&logical)[D],
                                      const real* X,
                                      int dof,
                                      const PtN<D>& trial,
                                      Obj& objective,
                                      real& e_new,
                                      real& mdet)
{
    real wI[D * D];
    for (int i = 0; i < D * D; ++i) { wI[i] = 0.0_r; }
    for (int k = 0; k < D; ++k) { wI[(k * D) + k] = 1.0_r; }

    e_new = 0.0_r;
    mdet = std::numeric_limits<real>::infinity();
    const auto idx = [&](const int (&l)[D]) { return dev_node_index<D>(block_off, nstride, b, l); };
    for (int occ = 0; occ < interior_patch_size<D>(); ++occ) {
        int gc;
        int gn[D];
        real s[D];
        int role;
        interior_occurrence_raw<D>(occ, logical, idx, gc, gn, s, role);
        const auto node = [&](int ni) -> PtN<D> { return ni == dof ? trial : load_pt<D>(X, ni); };
        const PtN<D> corner = node(gc);
        std::array<PtN<D>, D> nbr;
        std::array<real, D> sa;
        for (int k = 0; k < D; ++k) {
            nbr[static_cast<std::size_t>(k)] = node(gn[k]);
            sa[static_cast<std::size_t>(k)] = s[k];
        }
        real detA;
        const VecTN<D> t = assemble_vecT<D>(corner, nbr, sa, wI, detA);
        e_new += objective.value(t);
        mdet = std::min(detA, mdet);
    }
}

}  // namespace egg
