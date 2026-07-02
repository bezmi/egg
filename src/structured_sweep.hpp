#pragma once

// structured_sweep.hpp — the block-Jacobi sweep driven over a halo-padded
// structured store.
//
// StructuredExecutorT<D> composes SweepDeviceContextT<D> (the uploaded patch
// tables + X store), run_block_jacobi, and reduce_energy_mindet, but the
// SweepContextHostT it carries uses STRUCTURED node indices: gc/gn index into the
// packed BlockField buffer (built host-side via structured_patch.hpp), and X IS
// that packed buffer (interior slots filled through the block scatter map, ghost
// slots refreshed by the halo exchange). The structured path adds a per-sweep
// halo_exchange + broadcast_shared, injected as the run_block_jacobi before-sweep
// hook — cadence 1.4b (additive Schwarz / frozen halos): cross-block neighbours
// read the previous sweep's halo and non-owner copies of shared interface nodes
// are refreshed from their owner, so every free DOF relaxes from the same
// snapshot.

#include "structured_halo.hpp"
#include "sweep.hpp"

#include <sycl/sycl.hpp>
#include <utility>
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
    StructuredExecutorT(const sycl::queue& queue, const SweepContextHostT<D>& host,
                        BlockTopologyDevice<D> topo) :
        q_(sycl::queue(queue.get_context(), queue.get_device(),
          {sycl::property::queue::in_order(),
           sycl::property::queue::AdaptiveCpp_coarse_grained_events{}})),
        ctx_(q_, host), topo_(std::move(topo))
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
      run_jacobi(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 real omega = 1.0_r, int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              return run_block_jacobi<D>(q_, ctx_, objective, n_sweeps, halo_hook(),
                                        omega, report_every);
          },
          kind);
    }

  private:
    /// The per-sweep hook: refresh the cross-block ghost shell and the non-owner
    /// copies of shared interface nodes on the read buffer (frozen for the sweep —
    /// cadence 1.4b).
    /// Fused into a single launch — the two passes write disjoint slots, so
    /// fusion preserves behaviour (see @ref fused_halo_broadcast).
    auto halo_hook()
    {
        return [this](sycl::queue& q, real* X) {
            fused_halo_broadcast<D>(q, X, topo_);
        };
    }

    sycl::queue q_;
    SweepDeviceContextT<D> ctx_;
    BlockTopologyDevice<D> topo_;
};

}  // namespace egg
