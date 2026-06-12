// test_sweep_device.cpp — SYCL device parity of the colored Gauss-Seidel
// barrier sweep (src/sweep.hpp): sweep_group_kernel, compute_energy_mindet,
// and Executor.run_sweeps, run on EVERY visible device (the AMD GPU and the
// OpenMP host/CPU device) against the JAX golden sweep data.
//
// Proves the full sweep path — including the colour-ordered kernel sequence,
// backtracking, and energy/min-det reduction — is device-callable and
// numerically correct on both backends.
//
// Requires the acpp toolchain (SYCL).
#include "golden_sweep.hpp"
#include "sweep.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <cstring>
#include <format>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

using namespace boost::ut;
using namespace egg;
using egg_test::usable_devices;

namespace
{
bool close(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

// Build a SweepContextHost from the golden data.
SweepContextHost build_context_from_golden()
{
    SweepContextHost host;
    host.num_nodes = golden::kNumNodes;
    host.X.assign(golden::kX0.begin(), golden::kX0.end());

    // Groups
    for (std::size_t g = 0; g < golden::kNumGroups; ++g) {
        const auto& gg = golden::kSweepGroups[g];
        SweepGroupHost sg;
        sg.ndof = gg.D;

        // The golden groups are uniform-P (one per (colour,P)); express each in the
        // ragged per-DOF layout SweepGroupHost now uses (P_of[d]=P, sample_offset the
        // prefix sum). The flat DOF-major arrays carry over unchanged. (The runtime
        // #1 merge to one group per colour happens in the Python flatten_context
        // path; the Executor runs identically over either grouping.)
        const std::size_t DP = gg.D * gg.P;
        sg.total_samples = DP;
        sg.gc.assign(gg.gc.begin(), gg.gc.begin() + DP);
        sg.gn[0].assign(gg.gn0.begin(), gg.gn0.begin() + DP);
        sg.gn[1].assign(gg.gn1.begin(), gg.gn1.begin() + DP);
        sg.s[0].assign(gg.s0.begin(), gg.s0.begin() + DP);
        sg.s[1].assign(gg.s1.begin(), gg.s1.begin() + DP);
        sg.W_inv.assign(gg.W_inv.begin(), gg.W_inv.begin() + DP * 4);
        sg.role.assign(gg.role.begin(), gg.role.begin() + DP);
        sg.J.assign(gg.J.begin(), gg.J.begin() + DP * 24);
        sg.dof_idx.assign(gg.dof_idx.begin(), gg.dof_idx.begin() + gg.D);
        sg.tag.assign(gg.tag.begin(), gg.tag.begin() + gg.D);
        sg.params.assign(gg.params.begin(), gg.params.begin() + gg.D * 12);

        sg.P_of.assign(gg.D, gg.P);
        sg.sample_offset.resize(gg.D);
        std::exclusive_scan(sg.P_of.begin(), sg.P_of.end(), sg.sample_offset.begin(), 0);

        host.groups.push_back(std::move(sg));
    }

    // Energy stencil
    const auto& es = golden::kEnergyStencil;
    constexpr std::size_t es_num = golden::SweepEnergyStencil::num_samples;
    host.energy_stencil.num_samples = es_num;
    host.energy_stencil.gc.assign(es.gc.begin(), es.gc.begin() + es_num);
    host.energy_stencil.gn[0].assign(es.gn0.begin(), es.gn0.begin() + es_num);
    host.energy_stencil.gn[1].assign(es.gn1.begin(), es.gn1.begin() + es_num);
    host.energy_stencil.s[0].assign(es.s0.begin(), es.s0.begin() + es_num);
    host.energy_stencil.s[1].assign(es.s1.begin(), es.s1.begin() + es_num);
    host.energy_stencil.W_inv.assign(es.W_inv.begin(), es.W_inv.begin() + es_num * 4);

    return host;
}

// Test one sweep on one device: energy, mindet, and final X.
void test_sweep_on_device(sycl::queue& q, const std::string& name)
{
    boost::ut::log << std::format("  device: {}\n", name);

    SweepContextHost host = build_context_from_golden();
    Executor exec(q, host);

    auto [energies, mindets] = exec.run_sweeps(golden::kNumSweeps);

    // Per-sweep energy and mindet
    for (int s = 0; s < golden::kNumSweeps; ++s) {
        expect(close(energies[s], golden::kEnergies[s], 1e-10))
          << std::format("sweep {} energy: {} vs golden {}", s, energies[s], golden::kEnergies[s]);
        expect(close(mindets[s], golden::kMindets[s], 1e-10))
          << std::format("sweep {} mindet: {} vs golden {}", s, mindets[s], golden::kMindets[s]);
    }

    // Final X
    std::vector<double> X_final(golden::kNumNodes * 2);
    exec.ctx().download_X(X_final.data());

    double max_x_diff = 0.0;
    for (std::size_t i = 0; i < golden::kNumNodes * 2; ++i) {
        const double diff = std::abs(X_final[i] - golden::kXFinal[i]);
        if (diff > max_x_diff) max_x_diff = diff;
        expect(close(X_final[i], golden::kXFinal[i], 1e-9))
          << std::format("X[{}]: {} vs golden {}", i, X_final[i], golden::kXFinal[i]);
    }
    boost::ut::log << std::format("  max X diff: {:.2e}\n", max_x_diff);
}

}  // namespace

int main()
{
    const auto devices = usable_devices();
    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    for (const auto& dev : devices) {
        sycl::queue q(dev);
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("sweep on " + name) = [&] { test_sweep_on_device(q, name); };
    }
}
