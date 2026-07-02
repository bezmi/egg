#pragma once

// structured_field.hpp — device-resident storage of halo-padded structured
// blocks (Phase 1.1b of gpu-performance-improvement.md).
//
// BlockField<D> owns ONE UsmBuffer<double> holding every block's padded node
// coordinates back-to-back (the packing described by BlockLayout<D>), plus the
// layout itself. It hands out mdspan views — interior(b) and with_halo(b) —
// built on the single device base pointer, so a kernel addresses any block by
// base pointer + strides with no per-block allocation and no raw index math.
//
// This is the storage layer only: the global-DOF <-> structured correspondence
// (which host node fills which padded slot) is built host-side in 1.1c, and the
// ghost layer is populated by the halo_exchange kernel (1.2). Here the buffer is
// just an addressable, RAII-owned slab with typed, bounds-checked views.
//
// SYCL lives here (UsmBuffer), deliberately kept out of structured.hpp so the
// pure layout/view math stays SYCL-free and unit-testable without a device.

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
        layout_(std::move(layout)), buf_(q, layout_.total_doubles())
    {
    }

    [[nodiscard]] const BlockLayout<D>& layout() const { return layout_; }
    [[nodiscard]] std::size_t num_blocks() const { return layout_.num_blocks(); }

    /// Device base pointer of the whole packed buffer (doubles).
    [[nodiscard]] real* data() const { return buf_.data(); }
    /// Size of the packed buffer in doubles.
    [[nodiscard]] std::size_t size() const { return buf_.size(); }

    /// Upload a host buffer of exactly total_doubles() values into the device
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
    {
        return interior_view<D>(buf_.data(), layout_, b);
    }

    /// Whole padded block (ghost layer included) view of block `b`:
    /// extents (n0+2,…,n_{D-1}+2, D).
    [[nodiscard]] HaloView<D> with_halo(std::size_t b) const
    {
        return halo_view<D>(buf_.data(), layout_, b);
    }

  private:
    BlockLayout<D> layout_;
    UsmBuffer<real> buf_;
};

}  // namespace egg
