// test_geometry.cpp — host-side parity of the closed-form geometry project /
// tangent_space (src/geometry.hpp) vs the JAX oracle (egg.geometry.
// jax_project), over curated points per entity (on-curve, off-curve, degenerate
// p == center). Golden values from tests/cpp/golden_geometry.hpp.
//
// This suite runs under the default compiler in the `cpp_tests` executable; the
// SYCL device variant (CPU + GPU) lives in test_geometry_device.cpp.
#include "geometry.hpp"
#include "golden_geometry.hpp"
#include "real_tol.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{
constexpr double kTol = 1e-12;
bool close(double a, double b, double tol)
{
    tol = egg_test::real_tol(tol);
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}
}  // namespace

static const suite<"geometry"> geometry_suite = [] {
    const std::size_t n = golden::kGeoSamples.size();

    "project matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking project over {} golden samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kGeoSamples[k];
            const auto prm = egg_test::to_real(s.params);
            const Pt pr = project(Pt {egg::real(s.p[0]), egg::real(s.p[1])}, s.tag, prm.data());
            for (int i = 0; i < 2; ++i)
                expect(close(pr[i], s.proj[i], kTol))
                  << std::format("sample {} (tag {}) proj[{}]: {} vs {}",
                                 k,
                                 s.tag,
                                 i,
                                 pr[i],
                                 s.proj[i]);
        }
    };

    "tangent_space matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking tangent over {} golden samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kGeoSamples[k];
            const auto prm = egg_test::to_real(s.params);
            const Pt tg = tangent_space(Pt {egg::real(s.p[0]), egg::real(s.p[1])}, s.tag, prm.data());
            for (int i = 0; i < 2; ++i)
                expect(close(tg[i], s.tang[i], kTol))
                  << std::format("sample {} (tag {}) tang[{}]: {} vs {}",
                                 k,
                                 s.tag,
                                 i,
                                 tg[i],
                                 s.tang[i]);
        }
    };
};
