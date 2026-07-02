#pragma once

// structured_halo.hpp — device-side block interface topology and the
// halo_exchange kernel (Phase 1.2 of gpu-performance-improvement.md).
//
// BlockTopologyDevice<D> is the device image of the host halo tables built by
// egg.smoothing.cpp_backend.build_block_structured_context (1.1c): for each
// interface node it names one source interior node and one destination ghost
// node. It carries them as flat double-offsets into a BlockField's packed
// buffer — the host turns the padded (block, index) pairs into offsets through
// BlockLayout<D>, so the offset math stays single-sourced and the kernel is a
// pure gather/scatter with no per-entry index arithmetic.
//
// halo_exchange then copies, per entry, the D coordinates of the source node
// into the destination ghost slot. Entries are face-major (the host emits them
// walking a face's free axis), so consecutive work-items touch consecutive
// memory — the copy is coalesced and cheap.
//
// A conforming interface node is duplicated: it is interior to every block that
// touches it, but only its owner block (lowest index) relaxes it. The owner ->
// non-owner *broadcast* table (interior -> interior, distinct from the ghost
// halo's interior -> ghost) refreshes the non-owner copies from the owner each
// sweep so a non-owner block's patch reads the up-to-date shared value. Both are
// the same coordinate gather/scatter (broadcast_shared shares halo_exchange's
// kernel body) and are run together before each sweep (cadence 1.4b).
//
// Singularities (O-grid poles, valence != 2D) are carried as a separate node
// list for the explicit patch the structured smoother applies after the regular
// exchange (1.3); the regular table above covers every conforming face node.

#include "device.hpp"
#include "structured.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace egg
{

/// Device tables describing the cross-block halo copies for one structured
/// mesh. Move-only (owns USM). Build once from the host padded-index tables and
/// a BlockLayout<D>; reuse every sweep.
template <int D> class BlockTopologyDevice
{
  public:
    using Index = std::array<std::size_t, D>;

    BlockTopologyDevice() = default;

    /// Build from the host halo + singularity tables.
    ///
    /// @param layout       packing geometry (turns padded/logical indices into
    ///                     buffer offsets).
    /// @param src_block    (E,) block of each source interior node.
    /// @param src_padded   (E,) padded index of each source interior node.
    /// @param dst_block    (E,) block of each destination ghost node.
    /// @param dst_padded   (E,) padded index of each destination ghost node.
    /// @param sing_block   (S,) block of each singular node.
    /// @param sing_logical (S,) interior logical index of each singular node.
    /// @param share_src_block  (K,) owner block of each shared-node broadcast.
    /// @param share_src_padded (K,) owner interior padded index of each broadcast.
    /// @param share_dst_block  (K,) non-owner block of each shared-node broadcast.
    /// @param share_dst_padded (K,) non-owner interior padded index of each broadcast.
    /// @param fan_src_off  (F,) extra interior->ghost copies as already-computed
    ///                     double offsets (singular-fan neighbours mirrored into
    ///                     spare ghost-ring slots; appended to the ghost halo).
    /// @param fan_dst_off  (F,) destination ghost double offsets for the above.
    BlockTopologyDevice(sycl::queue q, const BlockLayout<D>& layout,
                        const std::vector<int>& src_block,
                        const std::vector<Index>& src_padded,
                        const std::vector<int>& dst_block,
                        const std::vector<Index>& dst_padded,
                        const std::vector<int>& sing_block,
                        const std::vector<Index>& sing_logical,
                        const std::vector<int>& share_src_block = {},
                        const std::vector<Index>& share_src_padded = {},
                        const std::vector<int>& share_dst_block = {},
                        const std::vector<Index>& share_dst_padded = {},
                        const std::vector<std::size_t>& fan_src_off = {},
                        const std::vector<std::size_t>& fan_dst_off = {}) :
        n_entries_(src_block.size() + fan_src_off.size()), n_sing_(sing_block.size()),
        n_share_(share_src_block.size())
    {
        std::vector<std::size_t> hsrc(n_entries_), hdst(n_entries_);
        for (std::size_t e = 0; e < src_block.size(); ++e) {
            hsrc[e] = layout.padded_node_offset(static_cast<std::size_t>(src_block[e]),
                                                src_padded[e]);
            hdst[e] = layout.padded_node_offset(static_cast<std::size_t>(dst_block[e]),
                                                dst_padded[e]);
        }
        // Singular-fan mirrors come pre-resolved to offsets (the host allocated
        // their spare ghost slots), so they simply extend the ghost halo table.
        for (std::size_t f = 0; f < fan_src_off.size(); ++f) {
            hsrc[src_block.size() + f] = fan_src_off[f];
            hdst[src_block.size() + f] = fan_dst_off[f];
        }
        src_off_ = UsmBuffer<std::size_t> {q, hsrc};
        dst_off_ = UsmBuffer<std::size_t> {q, hdst};

        // Owner -> non-owner shared-node copies (interior -> interior). Both
        // endpoints are interior padded indices, so the same padded_node_offset
        // path applies; the kernel is the same coordinate copy as the ghost halo.
        std::vector<std::size_t> hssrc(n_share_), hsdst(n_share_);
        for (std::size_t k = 0; k < n_share_; ++k) {
            hssrc[k] = layout.padded_node_offset(static_cast<std::size_t>(share_src_block[k]),
                                                 share_src_padded[k]);
            hsdst[k] = layout.padded_node_offset(static_cast<std::size_t>(share_dst_block[k]),
                                                 share_dst_padded[k]);
        }
        share_src_off_ = UsmBuffer<std::size_t> {q, hssrc};
        share_dst_off_ = UsmBuffer<std::size_t> {q, hsdst};

        std::vector<std::size_t> hsing(n_sing_);
        for (std::size_t s = 0; s < n_sing_; ++s) {
            hsing[s] = layout.interior_node_offset(static_cast<std::size_t>(sing_block[s]),
                                                   sing_logical[s]);
        }
        sing_off_ = UsmBuffer<std::size_t> {q, hsing};
    }

    [[nodiscard]] std::size_t num_entries() const { return n_entries_; }
    [[nodiscard]] const std::size_t* src_off() const { return src_off_.data(); }
    [[nodiscard]] const std::size_t* dst_off() const { return dst_off_.data(); }

    [[nodiscard]] std::size_t num_share() const { return n_share_; }
    [[nodiscard]] const std::size_t* share_src_off() const { return share_src_off_.data(); }
    [[nodiscard]] const std::size_t* share_dst_off() const { return share_dst_off_.data(); }

    [[nodiscard]] std::size_t num_singular() const { return n_sing_; }
    [[nodiscard]] const std::size_t* sing_off() const { return sing_off_.data(); }

  private:
    std::size_t n_entries_ = 0;
    std::size_t n_sing_ = 0;
    std::size_t n_share_ = 0;
    UsmBuffer<std::size_t> src_off_;        // (E,) source-node coord-0 offsets (doubles)
    UsmBuffer<std::size_t> dst_off_;        // (E,) destination ghost-node offsets
    UsmBuffer<std::size_t> share_src_off_;  // (K,) owner interior-node offsets
    UsmBuffer<std::size_t> share_dst_off_;  // (K,) non-owner interior-node offsets
    UsmBuffer<std::size_t> sing_off_;       // (S,) singular interior-node offsets
};

/// Copy the D coordinates of each `src[e]` node into `dst[e]`. One work-item per
/// entry; each copies D contiguous doubles. The single source for both the ghost
/// halo and the shared-node broadcast — they differ only in which offset tables
/// they pass. Returns the launch event (empty if `count == 0`).
template <int D>
inline sycl::event copy_nodes(sycl::queue& q, real* buf, const std::size_t* src,
                              const std::size_t* dst, std::size_t count)
{
    if (count == 0) { return {}; }
    return q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
        const std::size_t e = idx[0];
        const std::size_t s = src[e];
        const std::size_t d = dst[e];
        for (int k = 0; k < D; ++k) { buf[d + k] = buf[s + k]; }
    });
}

/// Fill every destination ghost slot with its source interior node's D
/// coordinates (the cross-block halo, interior -> ghost).
template <int D>
inline sycl::event halo_exchange(sycl::queue& q, real* buf,
                                 const BlockTopologyDevice<D>& topo)
{
    return copy_nodes<D>(q, buf, topo.src_off(), topo.dst_off(), topo.num_entries());
}

/// Refresh every non-owner copy of a shared interface node from its owner block
/// (interior -> interior). Run alongside @ref halo_exchange before each sweep so a
/// non-owner block's patch reads the owner's up-to-date shared value (cadence
/// 1.4b). The two write disjoint slots (ghost shell vs. non-owner interior) and
/// read genuinely-interior sources, so their order is immaterial.
template <int D>
inline sycl::event broadcast_shared(sycl::queue& q, real* buf,
                                    const BlockTopologyDevice<D>& topo)
{
    return copy_nodes<D>(q, buf, topo.share_src_off(), topo.share_dst_off(), topo.num_share());
}

/// Fused halo exchange + shared-node broadcast in a single launch. The two
/// passes share the @ref copy_nodes body and write disjoint slots (ghost shell
/// vs. non-owner interior) with genuinely-interior sources, so fusing is
/// behaviour-preserving (their order is immaterial, see broadcast_shared above).
/// One work-item per (halo entry || share entry); the per-item index selects
/// the offset table — all-halo and all-share work-groups run straight paths,
/// with at most one work-group straddling the boundary. Cuts one launch per
/// sweep on the halo-padded structured path.
template <int D>
inline sycl::event fused_halo_broadcast(sycl::queue& q, real* buf,
                                        const BlockTopologyDevice<D>& topo)
{
    const std::size_t n_halo = topo.num_entries();
    const std::size_t n_share = topo.num_share();
    const std::size_t total = n_halo + n_share;
    if (total == 0) { return {}; }
    const std::size_t* halo_src = topo.src_off();
    const std::size_t* halo_dst = topo.dst_off();
    const std::size_t* share_src = topo.share_src_off();
    const std::size_t* share_dst = topo.share_dst_off();
    return q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
        const std::size_t e = idx[0];
        const bool is_halo = (e < n_halo);
        const std::size_t off = is_halo ? 0 : (e - n_halo);
        const std::size_t s = is_halo ? halo_src[e] : share_src[off];
        const std::size_t d = is_halo ? halo_dst[e] : share_dst[off];
        for (int k = 0; k < D; ++k) { buf[d + k] = buf[s + k]; }
    });
}

}  // namespace egg
