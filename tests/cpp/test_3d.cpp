// test_3d.cpp — milestone-B checks for the dimension-specific 3D math:
// det<3>, solveNxN<3>, the condition-number barrier mu_cond3 (+ untangle
// surrogate) and its dual-AD derivatives, the 3D surface/edge parametrizations
// (plane, sphere, cylinder, line), the K=2 Gram–Schmidt branch of
// orthonormalize, the Trim<2> UV polygon, and the surface (k=2) arm of the
// k-reduced constrained newton_delta. Self-contained (no oracle goldens).
#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"
#include "solve.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{
constexpr double kTol = 1e-12;

bool close(double a, double b, double tol = kTol)
{ return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

double norm3(const VecN<3>& v) { return std::sqrt(dot(v, v)); }
}  // namespace

static const suite<"scalar3d"> scalar3d_suite = [] {
    "det<3> goldens"_test = [] {
        expect(close(det<3>(MatN<3> {1, 0, 0, 0, 1, 0, 0, 0, 1}), 1.0));
        expect(close(det<3>(MatN<3> {2, 0, 0, 0, 3, 0, 0, 0, 4}), 24.0));
        // numpy.linalg.det([[2,-1,0.5],[1,3,-2],[0.25,1,1]]) = 11.625
        expect(close(det<3>(MatN<3> {2, -1, 0.5, 1, 3, -2, 0.25, 1, 1}), 11.625, 1e-12));
    };

    "solveNxN<3> residual on a well-conditioned system"_test = [] {
        const MatN<3> H {4, 1, 0.5, 1, 3, -1, 0.5, -1, 5};
        const VecN<3> g {1.0, -2.0, 0.5};
        const VecN<3> x = solveNxN<3>(H, g);
        for (int i = 0; i < 3; ++i) {
            double r = g[i];
            for (int j = 0; j < 3; ++j) { r += H[(i * 3) + j] * x[j]; }
            expect(close(r, 0.0, 1e-10)) << std::format("residual row {} = {}", i, r);
        }
    };

    "solveNxN<3> fallbacks"_test = [] {
        const VecN<3> zero = solveNxN<3>(MatN<3> {}, VecN<3> {1e-16, 0, 0});
        expect(close(zero[0], 0.0) && close(zero[1], 0.0) && close(zero[2], 0.0));
        // Singular H -> non-finite candidate -> -0.1 g/|g|.
        const VecN<3> g {3.0, 0.0, 4.0};
        const VecN<3> fb = solveNxN<3>(MatN<3> {}, g);
        expect(close(fb[0], -0.1 * 3.0 / 5.0) && close(fb[2], -0.1 * 4.0 / 5.0));
    };
};

static const suite<"metric3d"> metric3d_suite = [] {
    "mu_cond3 is zero at the identity and under rotation"_test = [] {
        expect(close(mu_cond3(VecTN<3> {1, 0, 0, 0, 1, 0, 0, 0, 1}), 0.0, 1e-14));
        const double c = std::cos(0.7), s = std::sin(0.7);
        expect(close(mu_cond3(VecTN<3> {c, -s, 0, s, c, 0, 0, 0, 1}), 0.0, 1e-14));
    };

    "mu_cond3 is scale invariant"_test = [] {
        const VecTN<3> t {2, -1, 0.5, 1, 3, -2, 0.25, 1, 1};
        VecTN<3> t2;
        for (int i = 0; i < 9; ++i) { t2[i] = 3.7 * t[i]; }
        expect(close(mu_cond3(t), mu_cond3(t2), 1e-12));
    };

    "mu_cond3 golden on a diagonal stretch"_test = [] {
        // T = diag(2, 0.5, 1): |T|^2 = 5.25, |cof|^2 = 5.25, det = 1
        // mu = 5.25 * 5.25 / 9 - 1.
        expect(close(mu_cond3(VecTN<3> {2, 0, 0, 0, 0.5, 0, 0, 0, 1}),
                     (5.25 * 5.25 / 9.0) - 1.0,
                     1e-14));
    };

    "untangle surrogate approaches the barrier as delta -> 0"_test = [] {
        const VecTN<3> t {2, -1, 0.5, 1, 3, -2, 0.25, 1, 1};
        expect(close(mu_cond3_untangle(t, 1e-12), mu_cond3(t), 1e-9));
        // Finite on a folded (negative-det) cell.
        const VecTN<3> folded {-1, 0, 0, 0, 1, 0, 0, 0, 1};
        expect(std::isfinite(mu_cond3_untangle(folded, 0.5)));
    };

    "ShapeObjectiveT<3> grad matches finite differences"_test = [] {
        const ShapeObjectiveT<3> obj;
        const VecTN<3> t {1.2, -0.3, 0.1, 0.2, 0.9, -0.1, 0.05, 0.15, 1.1};
        const GradN<3> g = obj.grad(t);
        const double h = 1e-6;
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const double fd = (obj.value(tp) - obj.value(tm)) / (2 * h);
            expect(close(g[i], fd, 1e-6)) << std::format("grad[{}] {} vs FD {}", i, g[i], fd);
        }
    };

    "ShapeObjectiveT<3> hess matches finite differences of grad"_test = [] {
        const ShapeObjectiveT<3> obj;
        const VecTN<3> t {1.2, -0.3, 0.1, 0.2, 0.9, -0.1, 0.05, 0.15, 1.1};
        const HessN<3> H = obj.hess(t);
        const double h = 1e-6;
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const GradN<3> gp = obj.grad(tp), gm = obj.grad(tm);
            for (int j = 0; j < 9; ++j) {
                const double fd = (gp[j] - gm[j]) / (2 * h);
                expect(close(H[(i * 9) + j], fd, 1e-5))
                  << std::format("hess[{},{}] {} vs FD {}", i, j, H[(i * 9) + j], fd);
            }
        }
    };

    "UntangleObjectiveT<3> grad matches finite differences on a folded cell"_test = [] {
        const UntangleObjectiveT<3> obj {.delta = 0.3};
        const VecTN<3> t {-0.8, 0.2, 0.1, 0.3, 1.1, -0.2, 0.0, 0.1, 0.9};
        const GradN<3> g = obj.grad(t);
        const double h = 1e-6;
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const double fd = (obj.value(tp) - obj.value(tm)) / (2 * h);
            expect(close(g[i], fd, 1e-6)) << std::format("grad[{}] {} vs FD {}", i, g[i], fd);
        }
    };
};

static const suite<"geometry3d"> geometry3d_suite = [] {
    "plane projection lands on the plane, basis spans it"_test = [] {
        const auto ent =
          TrimmedEntity<PlaneParam> {.param = {.o = {1, 0, 0}, .ax = {0, 1, 0}, .ay = {0, 0, 1}},
                                     .trim = {}};
        const PtN<3> q {5.0, 2.0, -3.0};
        const auto f = ent.project_frame(q);
        expect(close(f.pos[0], 1.0) && close(f.pos[1], 2.0) && close(f.pos[2], -3.0));
        expect(f.eff_tdim == 2_i);
        expect(close(norm3(f.basis[0]), 1.0) && close(norm3(f.basis[1]), 1.0));
        expect(close(dot(f.basis[0], f.basis[1]), 0.0));
        // Both columns perpendicular to the plane normal (1,0,0).
        expect(close(f.basis[0][0], 0.0) && close(f.basis[1][0], 0.0));
    };

    "sphere projection is radial; frame orthonormal and tangent"_test = [] {
        const SphereParam sp {.c = {1, 2, 3},
                              .ax = {1, 0, 0},
                              .ay = {0, 1, 0},
                              .az = {0, 0, 1},
                              .r = 2.0};
        const auto ent = TrimmedEntity<SphereParam> {.param = sp, .trim = {}};
        const PtN<3> q {4.0, 4.0, 5.0};
        const auto f = ent.project_frame(q);
        expect(close(norm3(f.pos - PtN<3> {1, 2, 3}), 2.0, 1e-12));
        const VecN<3> n = normalize(f.pos - PtN<3> {1, 2, 3});
        expect(close(dot(f.basis[0], f.basis[1]), 0.0, 1e-12));
        expect(close(dot(f.basis[0], n), 0.0, 1e-12) && close(dot(f.basis[1], n), 0.0, 1e-12));
        expect(close(norm3(f.basis[0]), 1.0, 1e-12) && close(norm3(f.basis[1]), 1.0, 1e-12));
    };

    "cylinder projection lands at radius r from the axis"_test = [] {
        const CylinderParam cy {.o = {0, 0, 0},
                                .ax = {1, 0, 0},
                                .ay = {0, 1, 0},
                                .az = {0, 0, 1},
                                .r = 1.5};
        const auto ent = TrimmedEntity<CylinderParam> {.param = cy, .trim = {}};
        const PtN<3> q {3.0, 4.0, 2.5};
        const PtN<3> p = ent.project(q);
        expect(close(std::hypot(p[0], p[1]), 1.5, 1e-12));
        expect(close(p[2], 2.5));  // height preserved
    };

    "3D line: on-line projection, trim clamp drops eff_tdim"_test = [] {
        const auto ent =
          TrimmedEntity<Line3Param> {.param = {.p0 = {0, 0, 0}, .p1 = {1, 1, 1}},
                                     .trim = {.t0 = 0.0, .t1 = 1.0, .closed = false}};
        const auto f = ent.project_frame({0.5, 0.5, 0.0});
        expect(f.eff_tdim == 1_i);
        const VecN<3> u = normalize(VecN<3> {1, 1, 1});
        expect(close(std::abs(dot(f.basis[0], u)), 1.0, 1e-12));
        const auto fc = ent.project_frame({2.0, 2.0, 2.0});  // beyond t1 -> clamp
        expect(fc.eff_tdim == 0_i);
        expect(close(fc.pos[0], 1.0) && close(fc.pos[1], 1.0) && close(fc.pos[2], 1.0));
    };

    "orthonormalize<3,2> Gram-Schmidt on a skewed basis"_test = [] {
        const auto b =
          orthonormalize<3, 2>(std::array<VecN<3>, 2> {VecN<3> {2, 0, 0}, VecN<3> {1, 1, 0}});
        expect(close(norm3(b[0]), 1.0) && close(norm3(b[1]), 1.0));
        expect(close(dot(b[0], b[1]), 0.0));
    };

    "Trim<2> polygon contains/clamp (unit square with a hole)"_test = [] {
        // Outer CCW unit square, square hole [0.4,0.6]^2.
        const std::array<PtN<2>, 8> verts {PtN<2> {0, 0},
                                           PtN<2> {1, 0},
                                           PtN<2> {1, 1},
                                           PtN<2> {0, 1},
                                           PtN<2> {0.4, 0.4},
                                           PtN<2> {0.6, 0.4},
                                           PtN<2> {0.6, 0.6},
                                           PtN<2> {0.4, 0.6}};
        const std::array<int, 3> loops {0, 4, 8};
        const Trim<2> trim {.verts = verts, .loops = loops};
        expect(trim.contains({0.2, 0.2}));
        expect(!trim.contains({0.5, 0.5}));  // inside the hole
        expect(!trim.contains({1.5, 0.5}));  // outside the outer loop
        const Param<2> c = trim.clamp({1.5, 0.5});
        expect(close(c[0], 1.0) && close(c[1], 0.5));
        // Untrimmed (empty) always contains.
        expect(Trim<2> {}.contains({123.0, -456.0}));
    };

    "make_entity<3> decodes the surface blobs"_test = [] {
        std::array<double, kParamPad> blob {};
        // Sphere: [c(3), r, ax(3), ay(3)].
        blob = {1, 2, 3, 2.0, 1, 0, 0, 0, 1, 0};
        const auto ent = make_entity<3>(TAG_SPHERE, blob.data());
        const PtN<3> p =
          std::visit([&](const auto& e) { return e.project(PtN<3> {4, 4, 5}); }, ent);
        expect(close(norm3(p - PtN<3> {1, 2, 3}), 2.0, 1e-12));
        // Free fallback.
        const auto fr = make_entity<3>(TAG_FREE, blob.data());
        expect(std::holds_alternative<Free<3>>(fr));
    };
};

static const suite<"newton3d"> newton3d_suite = [] {
    "Free<3>: delta = -g with H = I"_test = [] {
        const MatN<3> H {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const VecN<3> g {1.0, -2.0, 0.5};
        const VecN<3> d = newton_delta<3>(g, H, PtN<3> {0, 0, 0}, Free<3> {});
        expect(close(d[0], -1.0) && close(d[1], 2.0) && close(d[2], -0.5));
    };

    "plane (tdim=2): delta is the in-plane part of -g"_test = [] {
        const auto ent =
          TrimmedEntity<PlaneParam> {.param = {.o = {0, 0, 0}, .ax = {1, 0, 0}, .ay = {0, 1, 0}},
                                     .trim = {}};
        const MatN<3> H {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const VecN<3> g {1.0, -2.0, 0.5};
        const VecN<3> d = newton_delta<3>(g, H, PtN<3> {0.3, 0.4, 0.0}, ent);
        // delta = -(g - (g.n)n) with n = (0,0,1): (-1, 2, 0).
        expect(close(d[0], -1.0) && close(d[1], 2.0) && close(d[2], 0.0));
    };

    "line (tdim=1): delta parallel to the direction"_test = [] {
        const auto ent =
          TrimmedEntity<Line3Param> {.param = {.p0 = {0, 0, 0}, .p1 = {0, 0, 2}},
                                     .trim = {.t0 = 0.0, .t1 = 1.0, .closed = false}};
        const MatN<3> H {1, 0, 0, 0, 1, 0, 0, 0, 1};
        const VecN<3> g {1.0, -2.0, 0.5};
        const VecN<3> d = newton_delta<3>(g, H, PtN<3> {0, 0, 0.5}, ent);
        expect(close(d[0], 0.0) && close(d[1], 0.0) && close(d[2], -0.5));
    };
};
