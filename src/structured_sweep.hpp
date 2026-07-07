// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file structured_sweep.hpp
/// Block-Jacobi sweep over a halo-padded structured store.
/// StructuredExecutorT<D> composes SweepDeviceContextT<D>, run_block_jacobi
/// and reduce_energy_mindet, with STRUCTURED node indices: gc/gn index the
/// packed BlockField buffer (built via structured_patch.hpp) and X IS that
/// buffer (interior slots filled through the block scatter map, ghosts by the
/// halo exchange). A per-sweep fused halo_exchange + broadcast_shared runs as
/// the before-sweep hook — additive Schwarz / frozen halos: every free DOF
/// relaxes from the same snapshot.

#include "fas.hpp"
#include "mg_level.hpp"
#include "structured_halo.hpp"
#include "sweep.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <variant>
#include <vector>

namespace egg
{

/// Block-Jacobi executor over a halo-padded structured store. Owns its in-order
/// queue, the device sweep context (structured indices), and the block halo
/// topology; refreshes ghost shells once per sweep before relaxing.
template <int D> class StructuredExecutorT
{
  public:
    /// @param queue  device to run on (a fresh in-order queue is made on it).
    /// @param host   sweep context with STRUCTURED node indices and a packed
    ///               (halo-padded) X buffer — built host-side from a
    ///               BlockStructuredContext.
    /// @param topo   block halo tables over the same packed layout; an empty
    ///               topology makes the per-sweep exchange a no-op (single block).
    /// @param halo_host  host copies of @p topo's conforming face tables, kept
    ///               so the FAS hierarchy can derive coarse interface topology
    ///               (mg_topology.hpp). Empty keeps every coarse interface
    ///               node frozen at its restricted position.
    StructuredExecutorT(const sycl::queue& queue,
                        const SweepContextHostT<D>& host,
                        BlockTopologyDevice<D> topo,
                        HaloTablesHost<D> halo_host = {}) :
        q_(sycl::queue(queue.get_context(),
                       queue.get_device(),
                       {sycl::property::queue::in_order(),
                        sycl::property::queue::AdaptiveCpp_coarse_grained_events {}})),
        ctx_(q_, host), topo_(std::move(topo)),
        // FAS host-side inputs (cheap; the device hierarchy is built lazily
        // on the first run_fas call). Empty on an unstructured context.
        fine_layout_(layout_from_context<D>(host)), fine_masks_(masks_from_context<D>(host)),
        fine_halo_(std::move(halo_host))
    {
    }

    SweepDeviceContextT<D>& ctx() { return ctx_; }

    /// Run n_sweeps of double-buffered block-Jacobi (one merged launch per sweep),
    /// returns (energies, min-dets). Under the frozen-halo cadence Jacobi
    /// converges to the minimiser.
    /// @p omega is the SOR/damping weight (1.0 = undamped).
    /// @p report_every throttles the energy/min-det reduction cadence
    /// (see @ref run_block_jacobi); default preserves the legacy per-sweep contract.
    std::pair<std::vector<real>, std::vector<real>>
      run_jacobi(int n_sweeps,
                 const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 real omega = 1.0_r,
                 int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              return run_block_jacobi<D>(q_,
                                         ctx_,
                                         scratch_,
                                         objective,
                                         n_sweeps,
                                         halo_hook(),
                                         omega,
                                         report_every);
          },
          kind);
    }

    /// Run FAS V-cycles (fas.hpp) on the resident context. Shape phase only —
    /// the untangle objective is rejected (its δ-continuation landscape is
    /// not elliptic-like, so the coarse-grid smooth-error argument fails).
    /// The cycle recurses through min(built, opts.max_levels − 1) coarse
    /// levels; falls back to plain block-Jacobi with the equivalent
    /// fine-sweep budget when no coarse level can be built (or max_levels
    /// disables the hierarchy), keeping the per-cycle report cadence.
    std::pair<std::vector<real>, std::vector<real>>
      run_fas(const FasOptions& opts, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {})
    {
        if (std::holds_alternative<UntangleObjectiveT<D>>(kind)) {
            throw std::invalid_argument(
              "run_fas: FAS supports the shape phase only, not \"untangle\"");
        }
        ensure_hierarchy();
        const int per_cycle = opts.nu_pre + opts.nu_post;
        const std::size_t n_coarse =
          opts.max_levels > 1
            ? std::min(levels_.size(), static_cast<std::size_t>(opts.max_levels - 1))
            : 0;
        if (n_coarse == 0) {
            return run_jacobi(opts.n_cycles * per_cycle, kind, opts.omega, per_cycle);
        }
        return std::visit(
          [&](auto objective) {
              FasFineView<D> fine;
              fine.num_nodes = ctx_.num_nodes();
              fine.X = ctx_.X();
              fine.seeds = ctx_.seeds();
              fine.group = ctx_.merged_group_view();
              fine.stencil = ctx_.stencil_view();
              // Qualified: the member function shadows the fas.hpp driver.
              return egg::run_fas<D>(q_,
                                     fine,
                                     fine_layout_,
                                     fine_free_.data(),
                                     std::span {levels_}.first(n_coarse),
                                     objective,
                                     opts,
                                     halo_hook(),
                                     scratch_);
          },
          kind);
    }

    /// Per-level per-block coarse interior shapes (introspection/bindings).
    std::vector<std::vector<typename BlockLayout<D>::Shape>> mg_level_shapes()
    {
        ensure_hierarchy();
        std::vector<std::vector<typename BlockLayout<D>::Shape>> out;
        out.reserve(levels_.size());
        for (const auto& lvl : levels_) {
            std::vector<typename BlockLayout<D>::Shape> shapes;
            shapes.reserve(lvl.layout.num_blocks());
            for (std::size_t b = 0; b < lvl.layout.num_blocks(); ++b) {
                shapes.push_back(lvl.layout.interior_shape(b));
            }
            out.push_back(std::move(shapes));
        }
        return out;
    }

  private:
    /// Per-sweep hook: refresh the cross-block ghost shell and the non-owner
    /// copies of shared interface nodes on the read buffer (frozen for the
    /// sweep) in one launch (see @ref fused_halo_broadcast).
    auto halo_hook()
    {
        return [this](sycl::queue& q, real* X) { fused_halo_broadcast<D>(q, X, topo_); };
    }

    /// Build the full coarse hierarchy + the fine free mask once, on first
    /// use. The build stops on its own when blocks stop coarsening, so the
    /// depth cap only bounds pathological cases; each run_fas call clips the
    /// ladder it actually cycles through via FasOptions::max_levels.
    static constexpr int kMaxHierarchyDepth = 32;
    void ensure_hierarchy()
    {
        if (fas_built_) { return; }
        fas_built_ = true;
        if (fine_layout_.num_blocks() == 0) { return; }  // unstructured — no FAS
        levels_ =
          build_mg_hierarchy<D>(q_, fine_layout_, fine_masks_, kMaxHierarchyDepth, fine_halo_);
        fine_free_ = UsmBuffer<std::uint8_t> {q_, fine_masks_.free};
    }

    sycl::queue q_;
    SweepDeviceContextT<D> ctx_;
    BlockTopologyDevice<D> topo_;
    // Reusable smoother scratch (x_new/metric/FAS buffers), lazily sized by
    // the drivers so per-chunk run/run_fas calls do not re-allocate USM.
    SweepScratchT<D> scratch_;

    // FAS state: host inputs built in the ctor, device hierarchy lazily.
    BlockLayout<D> fine_layout_;
    MgMasks fine_masks_;
    HaloTablesHost<D> fine_halo_;
    std::vector<MgLevel<D>> levels_;
    UsmBuffer<std::uint8_t> fine_free_;
    bool fas_built_ = false;
};

}  // namespace egg
