// test_curves.cpp — host-side checks for the curved 2D parametrizations (circular /
// elliptical arc, quadratic / cubic Bézier) and the shared seeded-Newton foot
// projector in src/geometry.hpp. Self-contained (no oracle goldens): projections
// must land on the curve, tangents must be unit and parallel to C', the inverse
// must recover the parameter of an on-curve query, and the interval trim must
// clamp out-of-range feet and drop the effective tangent dimension.
#include "geometry.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>
#include <numbers>
#include <vector>

using namespace boost::ut;
using namespace egg;

namespace
{
constexpr double kTol = 1e-9;

bool close(double a, double b, double tol = kTol)
{ return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

// Foot-of-perpendicular residual: a true nearest foot has (C(t) - q) · C'(t) = 0.
template <class C> double foot_residual(const C& c, const PtN<2>& q, double t)
{
    const VecN<2> d = c.eval({t}) - q;
    return dot(d, c.deriv({t}));
}
}  // namespace

static const suite<"curves"> curves_suite = [] {
    "circular arc projects onto the circle"_test = [] {
        const PtN<2> centre {0.5, -0.3};
        const double r = 2.0;
        const CircleArcParam arc {.c = centre, .r = r};
        for (const PtN<2> q : {PtN<2> {3.0, 2.0}, PtN<2> {-1.0, -1.0}, PtN<2> {0.5, 5.0}}) {
            const PtN<2> pr = arc.eval(arc.invert(q));
            const double dist = std::sqrt(dot(pr - centre, pr - centre));
            expect(close(dist, r)) << std::format("on-circle: |pr - c| = {} vs {}", dist, r);
        }
    };

    "tangent is unit and parallel to C'"_test = [] {
        const QuadBezierParam bez {.p = {{{0.0, 0.0}, {1.0, 2.0}, {3.0, 0.0}}}};
        for (const double t : {0.1, 0.5, 0.9}) {
            const std::array<VecN<2>, 1> raw = bez.frame({t});
            const std::array<VecN<2>, 1> on = orthonormalize<2, 1>(raw);
            expect(close(std::sqrt(dot(on[0], on[0])), 1.0)) << "tangent not unit";
            // unit tangent is parallel to the raw derivative ⇒ |cross| ≈ 0.
            const double cross = (on[0][0] * raw[0][1]) - (on[0][1] * raw[0][0]);
            expect(close(cross, 0.0, 1e-9)) << std::format("not parallel: cross = {}", cross);
        }
    };

    "Newton foot recovers the parameter of an on-curve point"_test = [] {
        const CubicBezierParam bez {.p = {{{0.0, 0.0}, {1.0, 3.0}, {2.0, -1.0}, {4.0, 1.0}}}};
        for (const double t_true : {0.2, 0.45, 0.8}) {
            const PtN<2> q = bez.eval({t_true});  // query is exactly on the curve
            const double t = bez.invert(q)[0];
            expect(close(t, t_true, 1e-7)) << std::format("recovered t = {} vs {}", t, t_true);
            expect(close(foot_residual(bez, q, t), 0.0, 1e-7)) << "foot residual nonzero";
        }
    };

    "Newton foot satisfies the stationarity condition off-curve"_test = [] {
        const EllipseArcParam ell {.c = {0.0, 0.0}, .a = 3.0, .b = 1.0, .phi = 0.4};
        for (const PtN<2> q : {PtN<2> {2.0, 2.0}, PtN<2> {-1.5, 0.5}}) {
            const double t = ell.invert(q)[0];
            expect(close(foot_residual(ell, q, t), 0.0, 1e-6))
              << std::format("residual = {}", foot_residual(ell, q, t));
        }
    };

    "interval trim clamps and drops eff_tdim"_test = [] {
        // Quarter circle arc, t in [0, pi/2]. A query at angle pi (outside) must
        // clamp onto the t1 endpoint and report eff_tdim == 0; an in-range query
        // stays eff_tdim == 1.
        const TrimmedEntity<CircleArcParam> arc {
          .param = {.c = {0.0, 0.0}, .r = 1.0},
          .trim = {.t0 = 0.0, .t1 = std::numbers::pi / 2.0, .closed = false}};

        const auto inside = arc.project_frame(PtN<2> {0.7, 0.7});  // angle ~ pi/4
        expect(inside.eff_tdim == 1_i) << "in-range query should keep eff_tdim 1";

        const auto outside = arc.project_frame(PtN<2> {-1.0, 0.0});  // angle pi, clamps to t1
        expect(outside.eff_tdim == 0_i) << "out-of-range query should drop eff_tdim to 0";
        const PtN<2> endpoint = arc.param.eval({arc.trim.t1});
        expect(close(outside.pos[0], endpoint[0]) && close(outside.pos[1], endpoint[1]))
          << "clamped projection should land on the t1 endpoint";
    };

    "project dispatches the curve tags"_test = [] {
        // Cubic Bézier through the flat upload blob, full [0,1] range.
        std::array<double, kParamPad> blob {};
        const std::array<PtN<2>, 4> cp {{{0.0, 0.0}, {1.0, 2.0}, {3.0, 2.0}, {4.0, 0.0}}};
        for (int i = 0; i < 4; ++i) {
            blob[2 * i] = cp[i][0];
            blob[(2 * i) + 1] = cp[i][1];
        }
        blob[8] = 0.0;  // t0
        blob[9] = 1.0;  // t1

        const CubicBezierParam ref {.p = cp};
        const PtN<2> q {2.0, 3.0};
        const PtN<2> pr = project(q, TAG_CUBICBEZIER, blob.data());
        const PtN<2> expref = ref.eval(ref.invert(q));
        expect(close(pr[0], expref[0], 1e-7) && close(pr[1], expref[1], 1e-7))
          << std::format("project cubic Bézier proj ({}, {}) vs ({}, {})",
                         pr[0],
                         pr[1],
                         expref[0],
                         expref[1]);
    };
};

static const suite<"bspline"> bspline_suite = [] {
    // A degree-3 B-spline on the Bézier knot vector {0,0,0,0,1,1,1,1} with four
    // control points is exactly the cubic Bézier over the same points — a strong
    // cross-check of the de Boor evaluation and its derivatives.
    "degree-3 B-spline on Bézier knots equals the cubic Bézier"_test = [] {
        const std::array<double, 8> knots {0, 0, 0, 0, 1, 1, 1, 1};
        const std::array<double, 8> ctrl {0, 0, 1, 3, 2, -1, 4, 1};  // (x,y) per control point
        const BSplineCurveParam bs {.degree = 3, .n_ctrl = 4, .knots = knots, .ctrl = ctrl};
        const CubicBezierParam bz {.p = {{{0, 0}, {1, 3}, {2, -1}, {4, 1}}}};
        for (const double u : {0.0, 0.1, 0.5, 0.9, 1.0}) {
            const PtN<2> a = bs.eval({u}), b = bz.eval({u});
            expect(close(a[0], b[0]) && close(a[1], b[1]))
              << std::format("eval u={}: ({},{}) vs ({},{})", u, a[0], a[1], b[0], b[1]);
            const VecN<2> da = bs.deriv({u}), db = bz.deriv({u});
            expect(close(da[0], db[0]) && close(da[1], db[1])) << std::format("deriv u={}", u);
            const VecN<2> d2a = bs.deriv2({u}), d2b = bz.deriv2({u});
            expect(close(d2a[0], d2b[0]) && close(d2a[1], d2b[1])) << std::format("deriv2 u={}", u);
        }
    };

    // Clamped knot vectors interpolate the first and last control points.
    "clamped endpoints interpolate the end control points"_test = [] {
        const std::array<double, 8> knots {0, 0, 0, 1, 2, 3, 3, 3};         // degree 2, clamped
        const std::array<double, 10> ctrl {0, 0, 1, 2, 3, -1, 4, 2, 6, 0};  // 5 control points
        const BSplineCurveParam bs {.degree = 2, .n_ctrl = 5, .knots = knots, .ctrl = ctrl};
        const PtN<2> p0 = bs.eval({0.0});  // u = knots[degree]
        const PtN<2> pn = bs.eval({3.0});  // u = knots[n_ctrl]
        expect(close(p0[0], 0.0) && close(p0[1], 0.0)) << "start endpoint";
        expect(close(pn[0], 6.0) && close(pn[1], 0.0)) << "end endpoint";
    };

    // Analytic derivatives must match central finite differences on a uniform spline.
    "derivatives match finite differences"_test = [] {
        const std::array<double, 8> knots {0, 0, 0, 1, 2, 3, 3, 3};
        const std::array<double, 10> ctrl {0, 0, 1, 2, 3, -1, 4, 2, 6, 0};
        const BSplineCurveParam bs {.degree = 2, .n_ctrl = 5, .knots = knots, .ctrl = ctrl};
        const double h = 1e-6;
        for (const double u : {0.5, 1.0, 1.7, 2.4}) {
            const PtN<2> fp = bs.eval({u + h}), fm = bs.eval({u - h});
            const VecN<2> fd {(fp[0] - fm[0]) / (2 * h), (fp[1] - fm[1]) / (2 * h)};
            const VecN<2> an = bs.deriv({u});
            expect(close(an[0], fd[0], 1e-5) && close(an[1], fd[1], 1e-5))
              << std::format("deriv FD u={}: ({},{}) vs ({},{})", u, an[0], an[1], fd[0], fd[1]);
        }
    };

    // Seeded-Newton inverse recovers the parameter of an on-curve query point.
    "Newton foot recovers the parameter of an on-curve point"_test = [] {
        const std::array<double, 8> knots {0, 0, 0, 1, 2, 3, 3, 3};
        const std::array<double, 10> ctrl {0, 0, 1, 2, 3, -1, 4, 2, 6, 0};
        const BSplineCurveParam bs {.degree = 2, .n_ctrl = 5, .knots = knots, .ctrl = ctrl};
        for (const double u_true : {0.6, 1.5, 2.3}) {
            const PtN<2> q = bs.eval({u_true});
            const double u = bs.invert(q)[0];
            expect(close(u, u_true, 1e-6)) << std::format("recovered u={} vs {}", u, u_true);
        }
    };

    // decode_entity must slice the knot/control spans out of the arena using the
    // offsets in the flat blob, matching a directly-constructed entity.
    "decode_entity slices the B-spline arena"_test = [] {
        const std::array<double, 8> knots {0, 0, 0, 1, 2, 3, 3, 3};
        const std::array<double, 10> ctrl {0, 0, 1, 2, 3, -1, 4, 2, 6, 0};
        std::vector<double> arena;
        const auto knot_off = arena.size();
        arena.insert(arena.end(), knots.begin(), knots.end());
        const auto ctrl_off = arena.size();
        arena.insert(arena.end(), ctrl.begin(), ctrl.end());

        std::array<double, kParamPad> blob {};
        blob[0] = 2;  // degree
        blob[1] = 5;  // n_ctrl
        blob[2] = static_cast<double>(knot_off);
        blob[3] = static_cast<double>(ctrl_off);
        blob[4] = 0.0;  // t0
        blob[5] = 3.0;  // t1

        const auto ent = decode_entity<TrimmedEntity<BSplineCurveParam>>(blob.data(), arena.data());
        const TrimmedEntity<BSplineCurveParam> ref {
          .param = {.degree = 2, .n_ctrl = 5, .knots = knots, .ctrl = ctrl},
          .trim = {.t0 = 0.0, .t1 = 3.0, .closed = false}};
        const PtN<2> q {2.5, 0.5};
        const PtN<2> pr = ent.project(q);
        const PtN<2> expref = ref.project(q);
        expect(close(pr[0], expref[0], 1e-7) && close(pr[1], expref[1], 1e-7))
          << std::format("arena B-spline proj ({},{}) vs ({},{})",
                         pr[0],
                         pr[1],
                         expref[0],
                         expref[1]);
    };
};

static const suite<"composite"> composite_suite = [] {
    // Pack one segment record [tag, params(kParamPad)] into the arena.
    constexpr auto push_rec =
      [](std::vector<double>& arena, Tag tag, std::initializer_list<double> params) {
          arena.push_back(static_cast<double>(tag));
          std::size_t n = 0;
          for (const double v : params) {
              arena.push_back(v);
              ++n;
          }
          for (; n < kParamPad; ++n) { arena.push_back(0.0); }
      };

    // L-shaped path: (0,0)->(1,0) then (1,0)->(1,1).
    constexpr auto make_L = [push_rec](std::vector<double>& arena) {
        push_rec(arena, TAG_LINESEG, {0.0, 0.0, 1.0, 0.0});
        push_rec(arena, TAG_LINESEG, {1.0, 0.0, 1.0, 1.0});
        return CompositePath {.n_segs = 2,
                              .recs = {arena.data(), arena.size()},
                              .arena = arena.data()};
    };

    "nearest-segment selection picks the right sub-chart"_test = [make_L] {
        std::vector<double> arena;
        const CompositePath path = make_L(arena);
        const PtN<2> a = path.project({0.4, -0.5});
        expect(close(a[0], 0.4) && close(a[1], 0.0)) << std::format("({},{})", a[0], a[1]);
        const PtN<2> b = path.project({1.5, 0.6});
        expect(close(b[0], 1.0) && close(b[1], 0.6)) << std::format("({},{})", b[0], b[1]);
    };

    "tangent is the matched segment's"_test = [make_L] {
        std::vector<double> arena;
        const CompositePath path = make_L(arena);
        const VecN<2> t0 = path.tangent_basis({0.4, -0.5})[0];
        expect(close(t0[0], 1.0) && close(t0[1], 0.0)) << "horizontal segment tangent";
        const VecN<2> t1 = path.tangent_basis({1.5, 0.6})[0];
        expect(close(t1[0], 0.0) && close(t1[1], 1.0)) << "vertical segment tangent";
    };

    "interior joint stays slidable (eff_tdim == 1)"_test = [make_L] {
        std::vector<double> arena;
        const CompositePath path = make_L(arena);
        // Outside the corner: both segments clamp to the joint (1,0).
        const auto f = path.project_frame({1.4, -0.5});
        expect(close(f.pos[0], 1.0) && close(f.pos[1], 0.0))
          << std::format("joint ({},{})", f.pos[0], f.pos[1]);
        expect(f.eff_tdim == 1_i);
    };

    "mixed segment types: line + circular arc"_test = [push_rec] {
        // Line (0,0)->(1,0), then a quarter arc centred at (1,1), radius 1,
        // from angle -pi/2 (point (1,0)) to 0 (point (2,1)).
        std::vector<double> arena;
        push_rec(arena, TAG_LINESEG, {0.0, 0.0, 1.0, 0.0});
        push_rec(arena, TAG_CIRCLEARC, {1.0, 1.0, 1.0, -std::numbers::pi / 2, 0.0, 0.0});
        const CompositePath path {.n_segs = 2,
                                  .recs = {arena.data(), arena.size()},
                                  .arena = arena.data()};
        // Query near the arc: radially projects onto the circle.
        const PtN<2> q {2.5, 0.0};
        const PtN<2> pr = path.project(q);
        const double rr = std::hypot(pr[0] - 1.0, pr[1] - 1.0);
        expect(close(rr, 1.0)) << std::format("on-arc radius {}", rr);
        // Query near the line.
        const PtN<2> pl = path.project({0.3, 0.2});
        expect(close(pl[0], 0.3) && close(pl[1], 0.0));
    };

    "decode_entity decodes the composite blob from the arena"_test = [push_rec] {
        std::vector<double> arena;
        arena.push_back(99.0);  // padding so rec_off != 0
        const auto rec_off = arena.size();
        push_rec(arena, TAG_LINESEG, {0.0, 0.0, 1.0, 0.0});
        push_rec(arena, TAG_LINESEG, {1.0, 0.0, 1.0, 1.0});

        std::array<double, kParamPad> blob {};
        blob[0] = 2.0;
        blob[1] = static_cast<double>(rec_off);
        const auto ent = decode_entity<CompositePath>(blob.data(), arena.data());
        const PtN<2> q {1.5, 0.6};
        const PtN<2> pr = ent.project(q);
        expect(close(pr[0], 1.0) && close(pr[1], 0.6)) << std::format("({},{})", pr[0], pr[1]);
    };
};

// EntitySoA<CompositePath> reconstruction (the device SoA load path). A composite
// owns one self-contained arena slice — any variable-length sub-segment data
// (B-spline knots/ctrl) first, then the fixed-stride segment records at rec_off —
// exactly the per-composite layout the Python wire builds. These tests rebuild a
// CompositePath via EntitySoA<CompositePath>::load from such a slice and verify it
// projects identically to the directly-constructed entity, for fixed-size segments
// AND a B-spline sub-segment (the NURBS-relevant case the blob path never carried).
static const suite<"composite_soa"> composite_soa_suite = [] {
    constexpr auto push_rec =
      [](std::vector<double>& arena, Tag tag, std::initializer_list<double> params) {
          arena.push_back(static_cast<double>(tag));
          std::size_t n = 0;
          for (const double v : params) {
              arena.push_back(v);
              ++n;
          }
          for (; n < kParamPad; ++n) { arena.push_back(0.0); }
      };

    // Rebuild a CompositePath from a self-contained arena slice via the device
    // SoA load path: records = {n_segs, rec_off}, one CSR slot = the whole slice.
    constexpr auto soa_load =
      [](const std::vector<double>& arena, int n_segs, int rec_off) -> CompositePath {
          using SoA = EntitySoA<CompositePath>;
          const std::array<double, 2> fields {static_cast<double>(n_segs),
                                              static_cast<double>(rec_off)};
          const std::array<int, 2> off {0, static_cast<int>(arena.size())};
          const SoAView<const double> recs_view {fields.data(), 1, SoA::kFields};
          const SegmentedView<double> seg {arena.data(), off.data()};
          return SoA::load(SoA::tie_view(recs_view, &seg), 0);
      };

    "fixed-size segments (lines, arc) round-trip through SoA load"_test = [push_rec, soa_load] {
        // L-shaped line path + a quarter arc: records only, rec_off == 0.
        std::vector<double> arena;
        push_rec(arena, TAG_LINESEG, {0.0, 0.0, 1.0, 0.0});
        push_rec(arena, TAG_LINESEG, {1.0, 0.0, 1.0, 1.0});
        push_rec(arena, TAG_CIRCLEARC, {1.0, 2.0, 1.0, -std::numbers::pi / 2, 0.0, 0.0});
        const CompositePath ref {.n_segs = 3,
                                 .recs = {arena.data(), arena.size()},
                                 .arena = arena.data()};
        const CompositePath got = soa_load(arena, 3, 0);
        for (const PtN<2> q : {PtN<2> {0.4, -0.5}, PtN<2> {1.5, 0.6}, PtN<2> {2.0, 2.0}}) {
            const PtN<2> a = ref.project(q);
            const PtN<2> b = got.project(q);
            expect(close(a[0], b[0]) && close(a[1], b[1]))
              << std::format("SoA load mismatch q=({},{}): ({},{}) vs ({},{})",
                             q[0], q[1], b[0], b[1], a[0], a[1]);
        }
    };

    "B-spline sub-segment round-trips through SoA load"_test = [push_rec, soa_load] {
        // Self-contained slice: [knots | ctrl | records]. A degree-2 B-spline arch
        // (ctrl (0,0),(1,1),(2,0), clamped knots) joined to a return line.
        const std::array<double, 6> knots {0, 0, 0, 1, 1, 1};
        const std::array<double, 6> ctrl {0, 0, 1, 1, 2, 0};
        std::vector<double> arena;
        const auto knot_off = arena.size();
        arena.insert(arena.end(), knots.begin(), knots.end());
        const auto ctrl_off = arena.size();
        arena.insert(arena.end(), ctrl.begin(), ctrl.end());
        const auto rec_off = arena.size();
        push_rec(arena, TAG_BSPLINE,
                 {2, 3, static_cast<double>(knot_off), static_cast<double>(ctrl_off), 0.0, 1.0});
        push_rec(arena, TAG_LINESEG, {2.0, 0.0, 0.0, 0.0});

        const CompositePath ref {
          .n_segs = 2,
          .recs = {arena.data() + rec_off, 2 * static_cast<std::size_t>(kCompositeRecSize)},
          .arena = arena.data()};
        const CompositePath got = soa_load(arena, 2, static_cast<int>(rec_off));

        // The standalone B-spline, to confirm the sub-segment is reconstructed
        // (not just that ref == got): a query above the arch projects onto it.
        const TrimmedEntity<BSplineCurveParam> bs {
          .param = {.degree = 2, .n_ctrl = 3, .knots = knots, .ctrl = ctrl},
          .trim = {.t0 = 0.0, .t1 = 1.0, .closed = false}};

        const PtN<2> q {1.0, 2.0};  // above the arch's apex (1, 0.5)
        const PtN<2> a = ref.project(q);
        const PtN<2> b = got.project(q);
        const PtN<2> s = bs.project(q);
        expect(close(a[0], b[0]) && close(a[1], b[1]))
          << std::format("SoA load mismatch: ({},{}) vs ({},{})", b[0], b[1], a[0], a[1]);
        expect(close(b[0], s[0]) && close(b[1], s[1]))
          << std::format("composite did not project onto its B-spline: ({},{}) vs ({},{})",
                         b[0], b[1], s[0], s[1]);
        expect(b[1] > 0.1_d) << "B-spline arch projection should sit above the return line";
    };
};
