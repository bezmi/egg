// bench_sweep.cpp — CPU vs GPU comparison for the colored GS sweep.
//
// Measures warm sweep-only ms/sweep (the project's preferred metric — not
// wall clock) of Executor.run_sweeps on a fixed demo-grid context, reported
// per device:
//
//   CPU serial          single-thread host baseline.
//   CPU SYCL (omp)      sweep on the OpenMP host device, data resident
//                       (honours OMP_NUM_THREADS).
//   GPU SYCL resident   sweep on the GPU, data uploaded once, kernel+
//                       wait only (the realistic on-device hot-loop cost).
//
// Built only with -DEGG_BUILD_BENCH=ON; not a CTest. Uses the golden
// sweep context (a small 4x4 grid); for production benchmarks, generate a
// larger golden context or load from a file.
#include "sweep.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <numeric>
#include <nanobench.h>
#include <optional>
#include <print>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

// Pull the golden context from gen_golden.py's output.
#include "golden_soa.hpp"
#include "golden_sweep.hpp"

using namespace egg;

namespace
{

std::optional<sycl::queue> try_queue(const auto& selector)
{
    try {
        return sycl::queue {selector};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

SweepContextHost build_context_from_golden()
{
    SweepContextHost host;
    host.num_nodes = golden::kNumNodes;
    host.X.assign(golden::kX0.begin(), golden::kX0.end());

    for (std::size_t g = 0; g < golden::kNumGroups; ++g) {
        const auto& gg = golden::kSweepGroups[g];
        SweepGroupHost sg;
        sg.ndof = gg.D;

        // Express the uniform-P golden group in the ragged per-DOF layout
        // SweepGroupHostT now uses (P_of[d]=P, sample_offset the prefix sum); the
        // flat DOF-major arrays carry over unchanged.
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

        sg.P_of.assign(gg.D, gg.P);
        sg.sample_offset.resize(gg.D);
        std::exclusive_scan(sg.P_of.begin(), sg.P_of.end(), sg.sample_offset.begin(), 0);

        // Decode the golden positional blob (per-DOF tag + params + shared arena)
        // into the typed SoA records the device ctx consumes.
        const std::vector<int> tag(gg.tag.begin(), gg.tag.begin() + gg.D);
        const std::vector<double> params(gg.params.begin(), gg.params.begin() + gg.D * 12);
        const std::vector<double> arena(golden::kArena.begin(), golden::kArena.end());
        sg.soa = egg_test::build_soa_from_blob(tag.data(), params.data(), arena, sg.ndof);

        host.groups.push_back(std::move(sg));
    }

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

}  // namespace

int main()
{
    auto cpu_q = try_queue(sycl::cpu_selector_v);
    auto gpu_q = try_queue(sycl::gpu_selector_v);

    std::println("CPU SYCL device : {}",
                 cpu_q ? cpu_q->get_device().get_info<sycl::info::device::name>() : "<none>");
    std::println("GPU SYCL device : {}",
                 gpu_q ? gpu_q->get_device().get_info<sycl::info::device::name>() : "<none>");

    SweepContextHost host = build_context_from_golden();
    constexpr int N_SWEEPS = 10;

    ankerl::nanobench::Bench bench;
    bench.title("colored GS sweep: CPU vs GPU")
      .unit("sweep")
      .relative(true)
      .warmup(3)
      .minEpochIterations(11);

    // --- CPU SYCL (OpenMP host device), data resident ---
    if (cpu_q) {
        SweepContextHost ctx_copy = host;
        Executor exec(*cpu_q, ctx_copy);

        bench.run("CPU SYCL (omp)", [&] {
            auto [energies, mindets] = exec.run_sweeps(N_SWEEPS);
            ankerl::nanobench::doNotOptimizeAway(energies.back());
        });
    }

    // --- GPU SYCL (device-resident), data uploaded once ---
    if (gpu_q) {
        SweepContextHost ctx_copy = host;
        Executor exec(*gpu_q, ctx_copy);

        bench.run("GPU SYCL resident", [&] {
            auto [energies, mindets] = exec.run_sweeps(N_SWEEPS);
            ankerl::nanobench::doNotOptimizeAway(energies.back());
        });
    }

    return 0;
}
