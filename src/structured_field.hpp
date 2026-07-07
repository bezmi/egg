// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file structured_field.hpp
/// Device-resident storage of halo-padded structured blocks. BlockField<D>
/// owns ONE UsmBuffer holding every block's padded node coordinates
/// back-to-back (the BlockLayout<D> packing) and hands out mdspan views on the
/// single device base pointer — no per-block allocation, no raw index math.
/// Storage layer only: the global-DOF <-> structured correspondence is built
/// host-side and ghosts are populated by halo_exchange. SYCL lives here, kept
/// out of structured.hpp so the layout math stays device-free and testable.

#include "device.hpp"
#include "structured.hpp"

#include <vector>

namespace egg
{

/// Owns the packed padded-coordinate buffer for all blocks of a structured
/// mesh and the BlockLayout<D> describing it. Move-only (it owns USM). Views
/// returned by interior()/with_halo() are non-owning and trivially copyable, so
/// they can be captured by value into a kernel.
template <int D> class BlockField
{
  public:
    BlockField() = default;

    /// Allocate the packed buffer (zero-initialised is NOT guaranteed; callers
    /// upload coordinates and let halo_exchange fill ghosts).
    BlockField(sycl::queue q, BlockLayout<D> layout) :
        layout_(std::move(layout)), buf_(q, layout_.total_reals())
    {
    }

    [[nodiscard]] const BlockLayout<D>& layout() const { return layout_; }
    [[nodiscard]] std::size_t num_blocks() const { return layout_.num_blocks(); }

    /// Device base pointer of the whole packed buffer (reals).
    [[nodiscard]] real* data() const { return buf_.data(); }
    /// Size of the packed buffer in reals.
    [[nodiscard]] std::size_t size() const { return buf_.size(); }

    /// Upload a host buffer of exactly total_reals() values into the device
    /// store (padded layout, including ghost slots).
    void upload(const std::vector<real>& host) { buf_.upload(host); }

    /// Download the whole packed buffer into `host` (resized to size()).
    void download(std::vector<real>& host)
    {
        host.resize(buf_.size());
        buf_.download(host.data());
    }

    /// Interior (ghost-excluded) view of block `b`: extents (n0,…,n_{D-1}, D).
    [[nodiscard]] InteriorView<D> interior(std::size_t b) const
    { return interior_view<D>(buf_.data(), layout_, b); }

    /// Whole padded block (ghost layer included) view of block `b`:
    /// extents (n0+2,…,n_{D-1}+2, D).
    [[nodiscard]] HaloView<D> with_halo(std::size_t b) const
    { return halo_view<D>(buf_.data(), layout_, b); }

  private:
    BlockLayout<D> layout_;
    UsmBuffer<real> buf_;
};

}  // namespace egg
