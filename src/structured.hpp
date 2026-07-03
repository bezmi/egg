// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file structured.hpp
/// Host-side geometry of halo-padded structured blocks. Deliberately SYCL- and
/// allocation-free: pure index arithmetic consumed by BlockField<D> (device
/// mdspans) and the halo builder (face slices), so the device side never
/// re-derives a stride.
///
/// Conventions (shared with core.hpp and egg/topology/block_topology.py):
///  - "interior shape" = node count per axis (BlockSpec.logical_shape), NOT cells.
///  - Each block is padded by one ghost node per face: padded shape (n_k + 2);
///    interior logical index l maps to padded l + 1 on every axis.
///  - Storage is row-major over the padded axes, the D coordinate components
///    innermost and contiguous (matches load_pt/store_pt X[D*i + k]) —
///    consecutive nodes on the fastest axis are D doubles apart, the property
///    the coalesced stencil reads rely on.
///  - All offsets and strides are in units of DOUBLES.

#include "core.hpp"  // egg::real (packed node store element type)

#include <array>
#include <cstddef>
#include <experimental/mdspan>
#include <utility>
#include <vector>

// Same alias as device.hpp (redeclaring to the same namespace is well-formed);
// mdspan needs no SYCL, so this header stays SYCL-free.
namespace stdex = std::experimental;

namespace egg
{

/// Packed layout of `num_blocks()` halo-padded structured blocks concatenated
/// back-to-back into one flat coordinate buffer. Holds each block's interior
/// shape and its base offset; padded shapes, node strides, and node offsets are
/// derived on demand. Pure host-side math — see the file header for conventions.
template <int D> class BlockLayout
{
  public:
    static_assert(D >= 1, "BlockLayout requires a positive spatial dimension");

    /// Per-axis index/shape tuple (interior or padded), e.g. node counts.
    using Shape = std::array<std::size_t, D>;

    BlockLayout() = default;

    /// Build the packed layout from every block's interior (node-count) shape.
    /// Blocks are laid out in the given order; block `b` occupies the doubles
    /// `[block_offset(b), block_offset(b+1))`.
    explicit BlockLayout(std::vector<Shape> interior_shapes) : interior_(std::move(interior_shapes))
    {
        offsets_.reserve(interior_.size() + 1);
        std::size_t acc = 0;
        for (const Shape& s : interior_) {
            offsets_.push_back(acc);
            acc += padded_doubles(s);
        }
        offsets_.push_back(acc);  // sentinel == total size in doubles
    }

    [[nodiscard]] std::size_t num_blocks() const { return interior_.size(); }

    /// Size (doubles) of the packed coordinate buffer for all blocks together.
    [[nodiscard]] std::size_t total_doubles() const
    { return offsets_.empty() ? 0 : offsets_.back(); }

    /// Interior (node-count) shape of block `b`.
    [[nodiscard]] const Shape& interior_shape(std::size_t b) const { return interior_[b]; }

    /// Padded shape of block `b`: interior + 2 ghost nodes per axis.
    [[nodiscard]] Shape padded_shape(std::size_t b) const
    {
        Shape p = interior_[b];
        for (std::size_t k = 0; k < D; ++k) { p[k] += 2; }
        return p;
    }

    /// Number of padded nodes (interior + ghosts) in block `b`.
    [[nodiscard]] std::size_t padded_node_count(std::size_t b) const
    {
        const Shape p = padded_shape(b);
        std::size_t n = 1;
        for (std::size_t k = 0; k < D; ++k) { n *= p[k]; }
        return n;
    }

    /// Offset (doubles) of block `b`'s first element in the packed buffer.
    [[nodiscard]] std::size_t block_offset(std::size_t b) const { return offsets_[b]; }

    /// Per-axis node strides (doubles) of block `b`'s padded array, row-major
    /// with the coordinate axis innermost. `strides[D-1] == D` (consecutive
    /// nodes on the fastest axis are D doubles apart); the step between the D
    /// coordinate components of one node is 1.
    [[nodiscard]] Shape node_strides(std::size_t b) const
    {
        const Shape p = padded_shape(b);
        Shape s {};
        std::size_t acc = D;  // D coordinate components per node, innermost
        for (std::size_t k = D; k-- > 0;) {
            s[k] = acc;
            acc *= p[k];
        }
        return s;
    }

    /// Offset (doubles) of a node addressed by its PADDED index (ghost layer
    /// included: each component in `[0, n_k + 2)`). Points at coordinate
    /// component 0 of that node.
    [[nodiscard]] std::size_t padded_node_offset(std::size_t b, const Shape& padded_idx) const
    {
        const Shape s = node_strides(b);
        std::size_t off = offsets_[b];
        for (std::size_t k = 0; k < D; ++k) { off += padded_idx[k] * s[k]; }
        return off;
    }

    /// Offset (doubles) of an INTERIOR node (logical index in `[0, n_k)`),
    /// shifted +1 per axis into the padded array. Points at coordinate
    /// component 0 of that node.
    [[nodiscard]] std::size_t interior_node_offset(std::size_t b, const Shape& logical_idx) const
    {
        Shape padded = logical_idx;
        for (std::size_t k = 0; k < D; ++k) { padded[k] += 1; }
        return padded_node_offset(b, padded);
    }

    /// Offset (doubles) of block `b`'s interior origin — logical (0,…,0), i.e.
    /// padded (1,…,1). The base of the centred interior view.
    [[nodiscard]] std::size_t interior_origin_offset(std::size_t b) const
    { return interior_node_offset(b, Shape {}); }

  private:
    /// Doubles occupied by one block's padded array: D coords × ∏(n_k + 2).
    static std::size_t padded_doubles(const Shape& interior)
    {
        std::size_t n = D;  // coordinate components per node
        for (std::size_t k = 0; k < D; ++k) { n *= (interior[k] + 2); }
        return n;
    }

    std::vector<Shape> interior_;       // per-block interior (node-count) shape
    std::vector<std::size_t> offsets_;  // size num_blocks()+1; back() == total
};

// mdspan views over a (host or device) coordinate buffer laid out by a
// BlockLayout<D>. Rank D+1: D spatial axes plus an innermost length-D
// coordinate axis, so v(i, ..., k) is component k of node (i, ...). They carry
// the BlockLayout strides — pass a base pointer (USM in kernels) + layout.

/// Whole padded block including the ghost layer: extents (n0+2, …, n_{D-1}+2, D),
/// contiguous row-major (layout_right) since a block occupies one packed run.
template <int D> using HaloView = stdex::mdspan<real, stdex::dextents<std::size_t, D + 1>>;

/// Interior only (ghosts excluded): extents (n0, …, n_{D-1}, D). A centred,
/// strided window into the padded block, so it is layout_stride.
template <int D>
using InteriorView = stdex::mdspan<real, stdex::dextents<std::size_t, D + 1>, stdex::layout_stride>;

namespace detail
{
template <int D, std::size_t... I>
HaloView<D>
  make_halo_view(real* p, const std::array<std::size_t, D>& padded, std::index_sequence<I...>)
{
    // (padded[0], …, padded[D-1], D) integral extents for the rank-(D+1) view.
    return HaloView<D> {p, padded[I]..., static_cast<std::size_t>(D)};
}

template <int D, std::size_t... I>
InteriorView<D> make_interior_view(real* p,
                                   const std::array<std::size_t, D>& interior,
                                   const std::array<std::size_t, D>& node_strides,
                                   std::index_sequence<I...>)
{
    using Ext = stdex::dextents<std::size_t, D + 1>;
    const Ext ext {std::array<std::size_t, D + 1> {interior[I]..., static_cast<std::size_t>(D)}};
    // Spatial axes use the padded node strides; the innermost coordinate axis is
    // unit-stride (the D components of a node are contiguous).
    const std::array<std::size_t, D + 1> strides {node_strides[I]..., static_cast<std::size_t>(1)};
    return InteriorView<D> {p, stdex::layout_stride::mapping<Ext> {ext, strides}};
}
}  // namespace detail

/// Halo (whole padded block) view of block `b` over `base` (doubles).
template <int D>
[[nodiscard]] HaloView<D> halo_view(real* base, const BlockLayout<D>& layout, std::size_t b)
{
    return detail::make_halo_view<D>(base + layout.block_offset(b),
                                     layout.padded_shape(b),
                                     std::make_index_sequence<D> {});
}

/// Interior (ghost-excluded) view of block `b` over `base` (doubles).
template <int D>
[[nodiscard]] InteriorView<D> interior_view(real* base, const BlockLayout<D>& layout, std::size_t b)
{
    return detail::make_interior_view<D>(base + layout.interior_origin_offset(b),
                                         layout.interior_shape(b),
                                         layout.node_strides(b),
                                         std::make_index_sequence<D> {});
}

}  // namespace egg
