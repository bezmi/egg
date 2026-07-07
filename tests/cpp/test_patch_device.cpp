// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_patch_device.cpp — SYCL device parity of per-DOF patch evaluation
// (src/patch.hpp): patch_eval, patch_energy_mindet, and newton_delta, run on
// EVERY visible device (the AMD GPU and the OpenMP host/CPU device) against the
// JAX golden patches. Proves the patch path — including the sample loop, chain-J
// Hessian assembly, and the tangent-reduced Newton step — is device-callable and
// correct on both backends.
//
// Requires the acpp toolchain (SYCL).
#include "golden_patch.hpp"
#include "patch.hpp"
#include "real_tol.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <cstdint>
#include <format>
#include <sycl/sycl.hpp>
#include <vector>

using namespace boost::ut;
using namespace egg;
using egg_test::usable_devices;

namespace
{
bool close(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

constexpr int kMaxP = golden::kMaxP;
constexpr int kMaxNodes = golden::kMaxNodes;

// Device outputs per patch, copied back to the host for comparison.
struct Outputs {
    std::vector<egg::real> grad;    // M*2
    std::vector<egg::real> hess;    // M*4
    std::vector<egg::real> energy;  // M
    std::vector<egg::real> mindet;  // M
    std::vector<egg::real> e2;      // M  (patch_energy_mindet)
    std::vector<egg::real> md2;     // M
    std::vector<egg::real> delta;   // M*2
};

// Evaluate every golden patch on one device. Each work-item handles one patch:
// builds a PatchView into its padded slice of the uploaded SoA arrays, then runs
// patch_eval / patch_energy_mindet / newton_delta.
void run_patch(sycl::queue& q, std::size_t M, Outputs& out)
{
    // Host SoA staging.
    std::vector<int> h_P(M), h_tag(M);
    std::vector<egg::real> h_X(M * kMaxNodes * 2);
    std::vector<int> h_gc(M * kMaxP), h_gn0(M * kMaxP), h_gn1(M * kMaxP), h_role(M * kMaxP);
    // Identity sample_id: each patch's gc/gn arrays are its own (local) metric
    // table, so occurrence p reads table row p.
    std::vector<int> h_sid(kMaxP);
    for (int i = 0; i < kMaxP; ++i) { h_sid[static_cast<std::size_t>(i)] = i; }
    std::vector<std::int8_t> h_s0(M * kMaxP), h_s1(M * kMaxP);  // ±1 signs (int8 store)
    std::vector<egg::real> h_W(M * kMaxP * 4), h_J(M * kMaxP * 24), h_params(M * 12);
    for (std::size_t k = 0; k < M; ++k) {
        const auto& s = golden::kPatchSamples[k];
        h_P[k] = s.P;
        h_tag[k] = s.tag;
        for (int i = 0; i < kMaxNodes * 2; ++i)
            h_X[k * kMaxNodes * 2 + i] = static_cast<egg::real>(s.X[i]);
        for (int i = 0; i < kMaxP; ++i) {
            h_gc[k * kMaxP + i] = s.gc[i];
            h_gn0[k * kMaxP + i] = s.gn0[i];
            h_gn1[k * kMaxP + i] = s.gn1[i];
            h_role[k * kMaxP + i] = s.role[i];
            h_s0[k * kMaxP + i] = static_cast<std::int8_t>(s.s0[i]);
            h_s1[k * kMaxP + i] = static_cast<std::int8_t>(s.s1[i]);
        }
        for (int i = 0; i < kMaxP * 4; ++i)
            h_W[k * kMaxP * 4 + i] = static_cast<egg::real>(s.W_inv[i]);
        for (int i = 0; i < kMaxP * 24; ++i)
            h_J[k * kMaxP * 24 + i] = static_cast<egg::real>(s.J[i]);
        for (int i = 0; i < 12; ++i)
            h_params[k * 12 + i] = static_cast<egg::real>(s.params[i]);
    }

    // Device USM.
    auto* d_P = sycl::malloc_device<int>(M, q);
    auto* d_tag = sycl::malloc_device<int>(M, q);
    auto* d_X = sycl::malloc_device<egg::real>(M * kMaxNodes * 2, q);
    auto* d_gc = sycl::malloc_device<int>(M * kMaxP, q);
    auto* d_gn0 = sycl::malloc_device<int>(M * kMaxP, q);
    auto* d_gn1 = sycl::malloc_device<int>(M * kMaxP, q);
    auto* d_role = sycl::malloc_device<int>(M * kMaxP, q);
    auto* d_sid = sycl::malloc_device<int>(kMaxP, q);
    auto* d_s0 = sycl::malloc_device<std::int8_t>(M * kMaxP, q);
    auto* d_s1 = sycl::malloc_device<std::int8_t>(M * kMaxP, q);
    auto* d_W = sycl::malloc_device<egg::real>(M * kMaxP * 4, q);
    auto* d_J = sycl::malloc_device<egg::real>(M * kMaxP * 24, q);
    auto* d_params = sycl::malloc_device<egg::real>(M * 12, q);
    auto* d_grad = sycl::malloc_device<egg::real>(M * 2, q);
    auto* d_hess = sycl::malloc_device<egg::real>(M * 4, q);
    auto* d_energy = sycl::malloc_device<egg::real>(M, q);
    auto* d_mindet = sycl::malloc_device<egg::real>(M, q);
    auto* d_e2 = sycl::malloc_device<egg::real>(M, q);
    auto* d_md2 = sycl::malloc_device<egg::real>(M, q);
    auto* d_delta = sycl::malloc_device<egg::real>(M * 2, q);

    q.memcpy(d_P, h_P.data(), M * sizeof(int));
    q.memcpy(d_tag, h_tag.data(), M * sizeof(int));
    q.memcpy(d_X, h_X.data(), h_X.size() * sizeof(egg::real));
    q.memcpy(d_gc, h_gc.data(), h_gc.size() * sizeof(int));
    q.memcpy(d_gn0, h_gn0.data(), h_gn0.size() * sizeof(int));
    q.memcpy(d_gn1, h_gn1.data(), h_gn1.size() * sizeof(int));
    q.memcpy(d_role, h_role.data(), h_role.size() * sizeof(int));
    q.memcpy(d_sid, h_sid.data(), h_sid.size() * sizeof(int));
    q.memcpy(d_s0, h_s0.data(), h_s0.size() * sizeof(std::int8_t));
    q.memcpy(d_s1, h_s1.data(), h_s1.size() * sizeof(std::int8_t));
    q.memcpy(d_W, h_W.data(), h_W.size() * sizeof(egg::real));
    q.memcpy(d_J, h_J.data(), h_J.size() * sizeof(egg::real));
    q.memcpy(d_params, h_params.data(), h_params.size() * sizeof(egg::real));
    q.wait();

    q.parallel_for(sycl::range<1>(M),
                   [=](sycl::id<1> idx) {
                       const std::size_t k = idx[0];
                    PatchView v {d_P[k],
                                 d_sid,
                                 &d_role[k * kMaxP],
                                 &d_gc[k * kMaxP],
                                 &d_gn0[k * kMaxP],
                                 &d_gn1[k * kMaxP],
                                 &d_s0[k * kMaxP],
                                 &d_s1[k * kMaxP],
                                 &d_W[k * kMaxP * 4]};
                    const egg::real* X = &d_X[k * kMaxNodes * 2];

                    const PatchResult r = patch_eval(v, X);
                    d_grad[k * 2 + 0] = r.grad[0];
                    d_grad[k * 2 + 1] = r.grad[1];
                    d_hess[k * 4 + 0] = r.hess[0];
                    d_hess[k * 4 + 1] = r.hess[1];
                    d_hess[k * 4 + 2] = r.hess[2];
                    d_hess[k * 4 + 3] = r.hess[3];
                    d_energy[k] = r.energy;
                    d_mindet[k] = r.mindet;

                    egg::real e2, md2;
                    patch_energy_mindet<2>(v, X, e2, md2);
                    d_e2[k] = e2;
                    d_md2[k] = md2;

                    const Pt pos {X[0], X[1]};
                       const Vec2 del =
                         newton_delta<2>(r.grad, r.hess, pos, d_tag[k], &d_params[k * 12]);
                       d_delta[k * 2 + 0] = del[0];
                       d_delta[k * 2 + 1] = del[1];
                   })
      .wait();

    out.grad.resize(M * 2);
    out.hess.resize(M * 4);
    out.energy.resize(M);
    out.mindet.resize(M);
    out.e2.resize(M);
    out.md2.resize(M);
    out.delta.resize(M * 2);
    q.memcpy(out.grad.data(), d_grad, M * 2 * sizeof(egg::real));
    q.memcpy(out.hess.data(), d_hess, M * 4 * sizeof(egg::real));
    q.memcpy(out.energy.data(), d_energy, M * sizeof(egg::real));
    q.memcpy(out.mindet.data(), d_mindet, M * sizeof(egg::real));
    q.memcpy(out.e2.data(), d_e2, M * sizeof(egg::real));
    q.memcpy(out.md2.data(), d_md2, M * sizeof(egg::real));
    q.memcpy(out.delta.data(), d_delta, M * 2 * sizeof(egg::real));
    q.wait();

    for (void* p : {(void*)d_P,
                    (void*)d_tag,
                    (void*)d_X,
                    (void*)d_gc,
                    (void*)d_gn0,
                    (void*)d_gn1,
                    (void*)d_role,
                    (void*)d_sid,
                    (void*)d_s0,
                    (void*)d_s1,
                    (void*)d_W,
                    (void*)d_J,
                    (void*)d_params,
                    (void*)d_grad,
                    (void*)d_hess,
                    (void*)d_energy,
                    (void*)d_mindet,
                    (void*)d_e2,
                    (void*)d_md2,
                    (void*)d_delta})
        sycl::free(p, q);
}
}  // namespace

int main()
{
    const auto devices = usable_devices();
    const std::size_t M = golden::kPatchSamples.size();
    // grad/hess/delta: closed-form vs autodiff (+ GPU FMA) -> ~1e-9; energy/mindet
    // closed-form -> tighter, but allow GPU lane-math slack.
    constexpr double kTolGH = 1e-9;
    constexpr double kTolE = 1e-10;

    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    for (const auto& dev : devices) {
        sycl::queue q(dev);
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("patch on " + name) = [&] {
            boost::ut::log << std::format("  device: {}\n", name);
            Outputs out;
            run_patch(q, M, out);
            for (std::size_t k = 0; k < M; ++k) {
                const auto& s = golden::kPatchSamples[k];
                for (int i = 0; i < 2; ++i)
                    expect(close(out.grad[k * 2 + i], s.grad[i], egg_test::real_tol(kTolGH)))
                      << std::format("patch {} grad[{}]: {} vs {}",
                                     k,
                                     i,
                                     out.grad[k * 2 + i],
                                     s.grad[i]);
                for (int i = 0; i < 4; ++i)
                    expect(close(out.hess[k * 4 + i], s.hess[i], egg_test::real_tol(kTolGH)))
                      << std::format("patch {} hess[{}]: {} vs {}",
                                     k,
                                     i,
                                     out.hess[k * 4 + i],
                                     s.hess[i]);
                expect(close(out.energy[k], s.energy, egg_test::real_tol(kTolE)))
                  << std::format("patch {} energy: {} vs {}", k, out.energy[k], s.energy);
                expect(close(out.mindet[k], s.mindet, egg_test::real_tol(kTolE)))
                  << std::format("patch {} mindet: {} vs {}", k, out.mindet[k], s.mindet);
                expect(close(out.e2[k], s.energy, egg_test::real_tol(kTolE)))
                  << std::format("patch {} energy(cheap): {} vs {}", k, out.e2[k], s.energy);
                expect(close(out.md2[k], s.mindet, egg_test::real_tol(kTolE)))
                  << std::format("patch {} mindet(cheap): {} vs {}", k, out.md2[k], s.mindet);
                for (int i = 0; i < 2; ++i)
                    expect(close(out.delta[k * 2 + i], s.delta[i], egg_test::real_tol(kTolGH)))
                      << std::format("patch {} (tag {}) delta[{}]: {} vs {}",
                                     k,
                                     s.tag,
                                     i,
                                     out.delta[k * 2 + i],
                                     s.delta[i]);
            }
        };
    }
}
