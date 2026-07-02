#pragma once

// structured_patch.hpp — the structured patch-evaluation bridge (Phase 1.3 of
// gpu-performance-improvement.md).
//
// The colored-GS hot path (patch.hpp::patch_eval / sample_vecT) reads a stencil
// node by index `i` as `X[D*i + k]` (load_pt/store_pt). In the UNSTRUCTURED path
// `i` is an arbitrary global node id, so the gather is random. In the STRUCTURED
// path the same evaluation runs over a BlockField<D>'s halo-padded buffer: every
// node lives at a fixed (block, padded index) slot, the store is D-contiguous per
// node with NO gaps (BlockLayout strides: innermost == D), and a block occupies
// one packed run whose base offset is a multiple of D. Therefore a node's flat
// index in node units is simply its double-offset / D — and patch_eval is a
// drop-in over the padded buffer once the stencil's gc/gn carry these structured
// indices instead of global ids.
//
// That is the whole coalescing win: consecutive interior nodes along the fastest
// axis are D doubles apart, so consecutive work-items read consecutive memory.
// This header is the single place that converts BlockLayout offsets into the
// node indices patch_eval consumes; nothing here re-derives a stride.
//
// SYCL-free: pure index arithmetic over BlockLayout<D>, host-unit-testable
// alongside structured.hpp. The structured sweep kernel (which captures the
// resulting PatchViewT<D> over the device buffer) lands on top of this.

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
[[nodiscard]] inline int padded_node_index(const BlockLayout<D>& layout, std::size_t b,
                                           const std::array<std::size_t, D>& padded_idx)
{
    return static_cast<int>(layout.padded_node_offset(b, padded_idx) / static_cast<std::size_t>(D));
}

/// Flat node index of an INTERIOR node (logical index in `[0, n_k)`), shifted
/// +1 per axis into the padded array. The structured analogue of a global DOF id
/// for patch_eval's gc/gn over a BlockField buffer.
template <int D>
[[nodiscard]] inline int interior_node_index(const BlockLayout<D>& layout, std::size_t b,
                                             const std::array<std::size_t, D>& logical_idx)
{
    return static_cast<int>(layout.interior_node_offset(b, logical_idx) /
                            static_cast<std::size_t>(D));
}

}  // namespace egg
