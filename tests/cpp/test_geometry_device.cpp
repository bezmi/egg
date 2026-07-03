// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_geometry_device.cpp — SYCL device parity of project / tangent_space
// (src/geometry.hpp), run on EVERY visible device (the AMD GPU and the OpenMP
// host/CPU device), against the JAX golden table. This proves the geometry path
// is device-callable and numerically correct on both backends.
//
// Requires the acpp toolchain (SYCL).
#include "geometry.hpp"
#include "golden_geometry.hpp"
#include "real_tol.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>
#include <sycl/sycl.hpp>
#include <vector>

using namespace boost::ut;
using namespace egg;
using egg_test::usable_devices;

namespace
{
bool close(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

// Run project + tangent_space for every golden sample on one device, returning
// proj/tang in host SoA arrays (each length M*2).
void run_geometry(sycl::queue& q,
                  std::size_t M,
                  std::vector<egg::real>& proj,
                  std::vector<egg::real>& tang)
{
    // Upload inputs (tag, params, p) into device USM.
    int* d_tag = sycl::malloc_device<int>(M, q);
    egg::real* d_params = sycl::malloc_device<egg::real>(M * 12, q);
    egg::real* d_p = sycl::malloc_device<egg::real>(M * 2, q);
    egg::real* d_proj = sycl::malloc_device<egg::real>(M * 2, q);
    egg::real* d_tang = sycl::malloc_device<egg::real>(M * 2, q);

    std::vector<int> h_tag(M);
    std::vector<egg::real> h_params(M * 12), h_p(M * 2);
    for (std::size_t k = 0; k < M; ++k) {
        const auto& s = golden::kGeoSamples[k];
        h_tag[k] = s.tag;
        for (int i = 0; i < 12; ++i) h_params[k * 12 + i] = static_cast<egg::real>(s.params[i]);
        h_p[k * 2 + 0] = static_cast<egg::real>(s.p[0]);
        h_p[k * 2 + 1] = static_cast<egg::real>(s.p[1]);
    }
    q.memcpy(d_tag, h_tag.data(), M * sizeof(int));
    q.memcpy(d_params, h_params.data(), M * 12 * sizeof(egg::real));
    q.memcpy(d_p, h_p.data(), M * 2 * sizeof(egg::real));
    q.wait();

    q.parallel_for(sycl::range<1>(M),
                   [=](sycl::id<1> idx) {
                       const std::size_t k = idx[0];
                       const Pt p {d_p[k * 2 + 0], d_p[k * 2 + 1]};
                       const Pt pr = project(p, d_tag[k], &d_params[k * 12]);
                       const Pt tg = tangent_space(p, d_tag[k], &d_params[k * 12]);
                       d_proj[k * 2 + 0] = pr[0];
                       d_proj[k * 2 + 1] = pr[1];
                       d_tang[k * 2 + 0] = tg[0];
                       d_tang[k * 2 + 1] = tg[1];
                   })
      .wait();

    q.memcpy(proj.data(), d_proj, M * 2 * sizeof(egg::real));
    q.memcpy(tang.data(), d_tang, M * 2 * sizeof(egg::real));
    q.wait();

    sycl::free(d_tag, q);
    sycl::free(d_params, q);
    sycl::free(d_p, q);
    sycl::free(d_proj, q);
    sycl::free(d_tang, q);
}
}  // namespace

int main()
{
    const auto devices = usable_devices();
    const std::size_t M = golden::kGeoSamples.size();
    constexpr double kTol = 1e-12;

    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    for (const auto& dev : devices) {
        sycl::queue q(dev);
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("geometry on " + name) = [&] {
            boost::ut::log << std::format("  device: {}\n", name);
            std::vector<egg::real> proj(M * 2), tang(M * 2);
            run_geometry(q, M, proj, tang);
            for (std::size_t k = 0; k < M; ++k) {
                const auto& s = golden::kGeoSamples[k];
                for (int i = 0; i < 2; ++i) {
                    expect(close(proj[k * 2 + i], s.proj[i], egg_test::real_tol(kTol)))
                      << std::format("sample {} (tag {}) proj[{}]: {} vs {}",
                                     k,
                                     s.tag,
                                     i,
                                     proj[k * 2 + i],
                                     s.proj[i]);
                    expect(close(tang[k * 2 + i], s.tang[i], egg_test::real_tol(kTol)))
                      << std::format("sample {} (tag {}) tang[{}]: {} vs {}",
                                     k,
                                     s.tag,
                                     i,
                                     tang[k * 2 + i],
                                     s.tang[i]);
                }
            }
        };
    }
}
