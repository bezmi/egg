// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file structured_smoother.hpp
/// The structured field interface: the StructuredField concept (read/write
/// node store with one-call halo synchronisation) and its concrete model
/// BlockFieldStructured<D> (BlockField<D> store + BlockTopologyDevice<D>
/// halo tables). BlockFieldStructured is host-side orchestration — it owns
/// USM and is never captured into a kernel; kernels capture the
/// trivially-copyable interior(b)/with_halo(b) views by value.

#include "device.hpp"
#include "structured.hpp"
#include "structured_field.hpp"
#include "structured_halo.hpp"

#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

namespace egg
{

/// A read/write field of node positions stored as halo-padded structured
/// blocks, addressable through strided mdspan views, with a one-call halo
/// synchronisation. The relaxation reads neighbours by fixed logical stride
/// (interior view) or from the ghost shell (halo view); exchange_halos()
/// refreshes the ghost shell from neighbouring blocks.
template <class F, int D>
concept StructuredField = requires(F f, std::size_t b) {
    { f.num_blocks() } -> std::convertible_to<std::size_t>;
    { f.interior(b) } -> std::same_as<InteriorView<D>>;
    { f.with_halo(b) } -> std::same_as<HaloView<D>>;
    { f.exchange_halos() };
};

/// Concrete StructuredField: BlockField<D> store + BlockTopologyDevice<D> halo
/// tables on a shared queue. Move-only (owns USM); exchange_halos() is the
/// per-sweep synchronisation the block-Jacobi sweep calls.
template <int D> class BlockFieldStructured
{
  public:
    BlockFieldStructured() = default;

    BlockFieldStructured(sycl::queue q, BlockField<D> field, BlockTopologyDevice<D> topo) :
        q_(q), field_(std::move(field)), topo_(std::move(topo))
    {
    }

    [[nodiscard]] std::size_t num_blocks() const { return field_.num_blocks(); }
    [[nodiscard]] const BlockLayout<D>& layout() const { return field_.layout(); }
    [[nodiscard]] real* data() const { return field_.data(); }

    [[nodiscard]] InteriorView<D> interior(std::size_t b) const { return field_.interior(b); }
    [[nodiscard]] HaloView<D> with_halo(std::size_t b) const { return field_.with_halo(b); }

    /// Refresh every block's ghost shell from its neighbours' interior layers.
    /// Returns the launch event (the caller waits / chains as the cadence needs).
    sycl::event exchange_halos() { return halo_exchange<D>(q_, field_.data(), topo_); }

    void upload(const std::vector<real>& host) { field_.upload(host); }
    void download(std::vector<real>& host) { field_.download(host); }

  private:
    sycl::queue q_ {};
    BlockField<D> field_;
    BlockTopologyDevice<D> topo_;
};

// Compile-time check that the storage adapter satisfies the swap seam.
static_assert(StructuredField<BlockFieldStructured<2>, 2>);
static_assert(StructuredField<BlockFieldStructured<3>, 3>);

}  // namespace egg
