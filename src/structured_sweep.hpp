#pragma once

// structured_sweep.hpp — the colored Gauss-Seidel sweep driven over a
// halo-padded structured store (Phase 1.3 of gpu-performance-improvement.md).
//
// StructuredExecutorT<D> reuses the exact unstructured sweep machinery —
// SweepDeviceContextT<D> (the uploaded per-colour patch tables + X store),
// sweep_colour_kernel, and reduce_energy_mindet — but the SweepContextHostT it
// carries uses STRUCTURED node indices: gc/gn index into the packed BlockField
// buffer (built host-side via structured_patch.hpp), and X IS that packed buffer
// (interior slots filled through the block scatter map, ghost slots refreshed by
// the halo exchange). The only thing the structured path adds over ExecutorT is
// a per-sweep halo_exchange + broadcast_shared, injected as the run_colored_gs
// before-sweep hook — cadence 1.4b (additive Schwarz / frozen halos): cross-block
// neighbours read the previous sweep's halo and non-owner copies of shared
// interface nodes are refreshed from their owner, so only within-block colour
// independence is needed.
//
// Because both executors compose the same run_colored_gs driver, the sweep,
// backtracking, entity dispatch, and reduction paths are single-sourced; this
// header contributes only the halo interleave and the topology ownership.

#include "structured_halo.hpp"
#include "sweep.hpp"

#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace egg
{

/// Colored-GS executor over a halo-padded structured store. Owns its in-order
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
    StructuredExecutorT(const sycl::queue& queue, const SweepContextHostT<D>& host,
                        BlockTopologyDevice<D> topo) :
        q_(sycl::queue(queue.get_context(), queue.get_device(),
          {sycl::property::queue::in_order(),
           sycl::property::queue::AdaptiveCpp_coarse_grained_events{}})),
        ctx_(q_, host), topo_(std::move(topo))
    {
    }

    SweepDeviceContextT<D>& ctx() { return ctx_; }

    /// Run n_sweeps with a per-sweep halo exchange; returns (energies, min-dets).
    /// @p report_every throttles the energy/min-det reduction cadence
    /// (see @ref run_colored_gs); default preserves the legacy per-sweep contract.
    std::pair<std::vector<double>, std::vector<double>>
      run_sweeps(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              return run_colored_gs<D>(q_, ctx_, objective, n_sweeps, halo_hook(), report_every);
          },
          kind);
    }

    /// Run n_sweeps of double-buffered block-Jacobi (one merged launch per sweep),
    /// returns (energies, min-dets). Same per-sweep halo hook as the colored-GS
    /// path — under the frozen-halo cadence Jacobi converges to the same minimiser.
    /// @p omega is the SOR/damping weight (1.0 = undamped).
    /// @p report_every throttles the energy/min-det reduction cadence
    /// (see @ref run_block_jacobi); default preserves the legacy per-sweep contract.
    std::pair<std::vector<double>, std::vector<double>>
      run_jacobi(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 double omega = 1.0, int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              return run_block_jacobi<D>(q_, ctx_, objective, n_sweeps, halo_hook(),
                                        omega, report_every);
          },
          kind);
    }

  private:
    /// The shared per-sweep hook: refresh the cross-block ghost shell and the
    /// non-owner copies of shared interface nodes on the read buffer (frozen for
    /// the sweep — cadence 1.4b). Used by both colored-GS and block-Jacobi.
    /// Fused into a single launch — the two passes write disjoint slots, so
    /// fusion preserves behaviour (see @ref fused_halo_broadcast).
    auto halo_hook()
    {
        return [this](sycl::queue& q, double* X) {
            fused_halo_broadcast<D>(q, X, topo_);
        };
    }

    sycl::queue q_;
    SweepDeviceContextT<D> ctx_;
    BlockTopologyDevice<D> topo_;
};

}  // namespace egg
