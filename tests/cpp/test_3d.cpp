// test_3d.cpp — checks for the dimension-specific 3D math:
// det<3>, solveNxN<3>, the condition-number barrier mu_cond3 (+ untangle
// surrogate) and its dual-AD derivatives, the 3D surface/edge parametrizations
// (plane, sphere, cylinder, line), the K=2 Gram–Schmidt branch of
// orthonormalize, the Trim<2> UV polygon, and the surface (k=2) arm of the
// k-reduced constrained newton_delta. Self-contained (no oracle goldens).
#include "real_tol.hpp"
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
{
    tol = egg_test::real_tol(tol);
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

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
        const egg::real c = std::cos(0.7), s = std::sin(0.7);
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
        const double h = egg_test::fd_step();
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const double fd = (obj.value(tp) - obj.value(tm)) / (2 * h);
            expect(close(g[i], fd, egg_test::fd_tol(1e-6))) << std::format("grad[{}] {} vs FD {}", i, g[i], fd);
        }
    };

    "ShapeObjectiveT<3> hess matches finite differences of grad"_test = [] {
        const ShapeObjectiveT<3> obj;
        const VecTN<3> t {1.2, -0.3, 0.1, 0.2, 0.9, -0.1, 0.05, 0.15, 1.1};
        const HessN<3> H = obj.hess(t);
        const double h = egg_test::fd_step();
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const GradN<3> gp = obj.grad(tp), gm = obj.grad(tm);
            for (int j = 0; j < 9; ++j) {
                const double fd = (gp[j] - gm[j]) / (2 * h);
                expect(close(H[(i * 9) + j], fd, egg_test::fd_tol(1e-5)))
                  << std::format("hess[{},{}] {} vs FD {}", i, j, H[(i * 9) + j], fd);
            }
        }
    };

    "UntangleObjectiveT<3> grad matches finite differences on a folded cell"_test = [] {
        const UntangleObjectiveT<3> obj {.delta = 0.3};
        const VecTN<3> t {-0.8, 0.2, 0.1, 0.3, 1.1, -0.2, 0.0, 0.1, 0.9};
        const GradN<3> g = obj.grad(t);
        const double h = egg_test::fd_step();
        for (int i = 0; i < 9; ++i) {
            VecTN<3> tp = t, tm = t;
            tp[i] += h;
            tm[i] -= h;
            const double fd = (obj.value(tp) - obj.value(tm)) / (2 * h);
            expect(close(g[i], fd, egg_test::fd_tol(1e-6))) << std::format("grad[{}] {} vs FD {}", i, g[i], fd);
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

    "decode_entity decodes the surface blobs"_test = [] {
        std::array<egg::real, kParamPad> blob {};
        // Sphere: [c(3), r, ax(3), ay(3)].
        blob = {1, 2, 3, 2.0, 1, 0, 0, 0, 1, 0};
        const auto ent =
          decode_entity<TrimmedEntity<SphereParam>>(blob.data());
        const PtN<3> p = ent.project(PtN<3> {4, 4, 5});
        expect(close(norm3(p - PtN<3> {1, 2, 3}), 2.0, 1e-12));
        // Free fallback.
        const auto fr = decode_entity<Free<3>>(blob.data());
        (void)fr;  // Free<3> is trivially constructed
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

static const suite<"bsplinesurf"> bsplinesurf_suite = [] {
    // Bilinear 2x2 patch on the plane z = 0 over [0,1]^2 — S(u,v) must be the
    // bilinear interpolant, the frame must span the plane, and invert must be
    // the identity map (u,v) = (x, y).
    constexpr auto bilinear = [] {
        static const std::array<egg::real, 4> ku {0, 0, 1, 1};
        static const std::array<egg::real, 12> ctrl {0,
                                                  0,
                                                  0,
                                                  0,
                                                  1,
                                                  0,  // (iu=0, iv=0..1)
                                                  1,
                                                  0,
                                                  0,
                                                  1,
                                                  1,
                                                  0};  // (iu=1, iv=0..1)
        return BSplineSurfaceParam {.pu = 1,
                                    .pv = 1,
                                    .nu = 2,
                                    .nv = 2,
                                    .knots_u = ku,
                                    .knots_v = ku,
                                    .ctrl = ctrl};
    };

    "bilinear patch reproduces the plane"_test = [bilinear] {
        const BSplineSurfaceParam s = bilinear();
        for (const auto& [u, v] :
             std::array<std::pair<egg::real, egg::real>, 3> {{{0.3, 0.7}, {0.0, 1.0}, {0.5, 0.5}}}) {
            const PtN<3> p = s.eval({u, v});
            expect(close(p[0], u) && close(p[1], v) && close(p[2], 0.0))
              << std::format("eval({},{}) = ({},{},{})", u, v, p[0], p[1], p[2]);
        }
        const auto fr = s.frame({0.3, 0.7});
        expect(close(fr[0][0], 1.0) && close(fr[0][1], 0.0) && close(fr[0][2], 0.0));
        expect(close(fr[1][0], 0.0) && close(fr[1][1], 1.0) && close(fr[1][2], 0.0));
        const Param<2> q = s.invert({0.25, 0.6, 3.0});
        expect(close(q[0], 0.25, 1e-9) && close(q[1], 0.6, 1e-9));
    };

    // A curved bicubic patch (z = bump): FD-check the frame and recover the
    // parameters of an on-surface query through the seeded Newton inverse.
    constexpr auto bump = [] {
        static const std::array<egg::real, 8> k3 {0, 0, 0, 0, 1, 1, 1, 1};
        static std::array<egg::real, 48> ctrl {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const egg::real x = static_cast<egg::real>(i) / 3.0_r,
                              y = static_cast<egg::real>(j) / 3.0_r;
                ctrl[3 * ((i * 4) + j) + 0] = x;
                ctrl[3 * ((i * 4) + j) + 1] = y;
                // Interior control z lifts the middle of the patch.
                ctrl[3 * ((i * 4) + j) + 2] = (i == 0 || i == 3 || j == 0 || j == 3) ? 0.0 : 1.0;
            }
        }
        return BSplineSurfaceParam {.pu = 3,
                                    .pv = 3,
                                    .nu = 4,
                                    .nv = 4,
                                    .knots_u = k3,
                                    .knots_v = k3,
                                    .ctrl = ctrl};
    };

    "bicubic frame matches finite differences"_test = [bump] {
        const BSplineSurfaceParam s = bump();
        const egg::real h = static_cast<egg::real>(egg_test::fd_step());
        for (const auto& [u, v] :
             std::array<std::pair<egg::real, egg::real>, 3> {{{0.3, 0.4}, {0.6, 0.2}, {0.5, 0.5}}}) {
            const auto fr = s.frame({u, v});
            const PtN<3> up = s.eval({u + h, v}), um = s.eval({u - h, v});
            const PtN<3> vp = s.eval({u, v + h}), vm = s.eval({u, v - h});
            for (int i = 0; i < 3; ++i) {
                expect(close(fr[0][i], (up[i] - um[i]) / (2 * h), egg_test::fd_tol(1e-5)))
                  << std::format("S_u[{}] at ({},{})", i, u, v);
                expect(close(fr[1][i], (vp[i] - vm[i]) / (2 * h), egg_test::fd_tol(1e-5)))
                  << std::format("S_v[{}] at ({},{})", i, u, v);
            }
        }
    };

    "Newton inverse recovers the parameters of an on-surface point"_test = [bump] {
        const BSplineSurfaceParam s = bump();
        for (const auto& [u, v] :
             std::array<std::pair<egg::real, egg::real>, 3> {{{0.3, 0.4}, {0.7, 0.6}, {0.2, 0.8}}}) {
            const Param<2> q = s.invert(s.eval({u, v}));
            expect(close(q[0], u, 1e-7) && close(q[1], v, 1e-7))
              << std::format("recovered ({},{}) vs ({},{})", q[0], q[1], u, v);
        }
    };

    "TrimmedEntity surface frame is orthonormal and tangent"_test = [bump] {
        const TrimmedEntity<BSplineSurfaceParam> ent {.param = bump(), .trim = {}};
        const PtN<3> q {0.4, 0.5, 2.0};
        const auto f = ent.project_frame(q);
        expect(f.eff_tdim == 2_i);
        expect(close(std::sqrt(dot(f.basis[0], f.basis[0])), 1.0, 1e-12));
        expect(close(std::sqrt(dot(f.basis[1], f.basis[1])), 1.0, 1e-12));
        expect(close(dot(f.basis[0], f.basis[1]), 0.0, 1e-12));
        // Projection is a stationary foot: (S - q) ⟂ both tangent columns.
        const VecN<3> d = f.pos - q;
        expect(close(dot(d, f.basis[0]), 0.0, 1e-7) && close(dot(d, f.basis[1]), 0.0, 1e-7));
    };

    // Rational (NURBS) quarter-cylinder: rational quadratic quarter circle in u
    // (weights {1, 1/sqrt2, 1}) extruded linearly in v — exact unit radius.
    constexpr auto quarter_cyl = [] {
        static const std::array<egg::real, 6> ku {0, 0, 0, 1, 1, 1};
        static const std::array<egg::real, 4> kv {0, 0, 1, 1};
        static const std::array<egg::real, 18> ctrl {1,
                                                  0,
                                                  0,
                                                  1,
                                                  0,
                                                  1,  // circle ctrl (1,0), v = 0/1
                                                  1,
                                                  1,
                                                  0,
                                                  1,
                                                  1,
                                                  1,  // circle ctrl (1,1)
                                                  0,
                                                  1,
                                                  0,
                                                  0,
                                                  1,
                                                  1};  // circle ctrl (0,1)
        static const std::array<egg::real, 6> w {1.0,
                                              1.0,
                                              1.0 / std::numbers::sqrt2,
                                              1.0 / std::numbers::sqrt2,
                                              1.0,
                                              1.0};
        return BSplineSurfaceParam {.pu = 2,
                                    .pv = 1,
                                    .nu = 3,
                                    .nv = 2,
                                    .knots_u = ku,
                                    .knots_v = kv,
                                    .ctrl = ctrl,
                                    .weights = w};
    };

    "rational quarter-cylinder has exact unit radius"_test = [quarter_cyl] {
        const BSplineSurfaceParam s = quarter_cyl();
        for (const egg::real u : {0.0_r, 0.25_r, 0.5_r, 0.75_r, 1.0_r}) {
            for (const egg::real v : {0.0_r, 0.5_r, 1.0_r}) {
                const PtN<3> p = s.eval({u, v});
                expect(close(std::hypot(p[0], p[1]), 1.0, 1e-14))
                  << std::format("radius at ({},{})", u, v);
                expect(close(p[2], v, 1e-14));  // linear extrusion preserved
            }
        }
    };

    "projection onto the rational cylinder is radial"_test = [quarter_cyl] {
        const TrimmedEntity<BSplineSurfaceParam> ent {.param = quarter_cyl(), .trim = {}};
        const PtN<3> q {2.0, 2.0, 0.5};  // 45°, mid-height
        const PtN<3> p = ent.project(q);
        expect(close(std::hypot(p[0], p[1]), 1.0, 1e-9));
        expect(close(p[0], p[1], 1e-9));  // stays at 45°
        expect(close(p[2], 0.5, 1e-9));   // height preserved
    };

    "decode_entity slices the surface from the arena"_test = [quarter_cyl] {
        const BSplineSurfaceParam ref = quarter_cyl();
        std::vector<egg::real> arena;
        const auto ku_off = arena.size();
        arena.insert(arena.end(), ref.knots_u.begin(), ref.knots_u.end());
        const auto kv_off = arena.size();
        arena.insert(arena.end(), ref.knots_v.begin(), ref.knots_v.end());
        const auto ctrl_off = arena.size();
        arena.insert(arena.end(), ref.ctrl.begin(), ref.ctrl.end());
        const auto w_off = arena.size();
        arena.insert(arena.end(), ref.weights.begin(), ref.weights.end());

        std::array<egg::real, kParamPad> blob {};
        blob[0] = 2;  // pu
        blob[1] = 1;  // pv
        blob[2] = 3;  // nu
        blob[3] = 2;  // nv
        blob[4] = static_cast<double>(ku_off);
        blob[5] = static_cast<double>(kv_off);
        blob[6] = static_cast<double>(ctrl_off);
        blob[7] = static_cast<double>(w_off);
        blob[8] = 1.0;  // has_w
        const auto ent =
          decode_entity<TrimmedEntity<BSplineSurfaceParam>>(blob.data(), arena.data());
        const PtN<3> p = ent.project(PtN<3> {2.0, 2.0, 0.5});
        expect(close(std::hypot(p[0], p[1]), 1.0, 1e-9) && close(p[2], 0.5, 1e-9));
    };
};
