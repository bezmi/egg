// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

#include "core.hpp"  // egg::real (SoA wire element type)

#include <concepts>
#include <cstddef>
#include <experimental/mdspan>
#include <span>
#include <vector>

namespace stdex = std::experimental;

namespace egg
{

// Strongly-typed, data-oriented entity storage: the transport layer replacing
// the positional (int tag, real params[kParamPad]) blob — strong entity tag,
// one segmented-array (CSR) idiom for variable-length payloads, and the
// per-entity SoA trait. Geometry-agnostic: the concrete EntitySoA<E>
// specializations live in geometry.hpp with the entity types.

/// Strongly-typed entity kind tag. Integer values are FROZEN — the wire
/// contract with the Python encoder and the geometry golden tables, locked to
/// the legacy TAG_* constants by a static_assert in geometry.hpp. 3D-only
/// kinds share the one tag space (they map to Free in the 2D set). Base type
/// int is the frozen wire width (int32 across the Python boundary), not a
/// size to optimise — hence the suppressed enum-size lint.
// NOLINTNEXTLINE(performance-enum-size)
enum class EntityTag : int {
    Free = 0,
    LineSeg = 1,
    Circle = 2,
    Ellipse = 3,
    Sphere = 4,  ///< 3D surface; no 2D entity (falls through to Free in the 2D set).
    Plane = 5,   ///< 3D surface; no 2D entity.
    CircleArc = 6,
    EllipseArc = 7,
    QuadBezier = 8,
    CubicBezier = 9,
    BSpline = 10,
    Composite = 11,
    Cylinder = 12,        ///< 3D surface.
    Line3 = 13,           ///< 3D edge curve.
    BSplineSurface = 14,  ///< 3D tensor-product B-spline/NURBS surface.
};

/// Frozen integer value of @p t (wire / Python / golden boundary).
[[nodiscard]] constexpr int to_int(EntityTag t) { return static_cast<int>(t); }

// Segmented (CSR) payload — the single idiom for every variable-length entity
// payload (B-spline knots/nets/weights, composite segment records), mirroring
// the sweep's ragged patch samples (P_of/sample_offset). Entity i owns
// data[off[i] .. off[i+1]); off has length count+1, off[0] == 0.

/// Host-side segmented array: flat @c data + CSR @c off table.
template <class T> struct SegmentedHost {
    std::vector<T> data;   ///< Concatenated payload of every entity, in entity order.
    std::vector<int> off;  ///< CSR offsets, length @c count+1 (@c off[0]==0).

    /// Number of entities described by the offset table.
    [[nodiscard]] std::size_t count() const { return off.empty() ? 0 : off.size() - 1; }

    /// Append the next entity's payload @p s (lazily seeds the leading 0, then
    /// concatenates and records the running length).
    void push_back(std::span<const T> s)
    {
        if (off.empty()) { off.push_back(0); }
        data.insert(data.end(), s.begin(), s.end());
        off.push_back(static_cast<int>(data.size()));
    }
};

/// Trivially-copyable device view over a @ref SegmentedHost.
template <class T> struct SegmentedView {
    const T* data = nullptr;
    const int* off = nullptr;
    /// The contiguous slice owned by entity @p i: data[off[i] .. off[i+1]).
    [[nodiscard]] std::span<const T> operator[](std::size_t i) const
    {
        const auto b = static_cast<std::size_t>(off[i]);
        const auto e = static_cast<std::size_t>(off[i + 1]);
        return {data + b, e - b};
    }
};

// Per-entity SoA trait — the single field schema. EntitySoA<E> (specialized in
// geometry.hpp) composes storage from packed records (real[count*kFields],
// stride kFields) plus SegmentedHost/SegmentedView for ragged fields, and
// supplies: load(View, i) -> E (the ONE device-side builder, replacing that
// entity's make_entity arm); load_into(Host&, i, E) (the host-side inverse);
// tie_view(SoAView, SegmentedView*) -> View (typed View from PartitionView's
// generic slots, once per launch). Uniform shapes across specializations:
//   Host: { std::vector<real> records; std::size_t count;
//           std::vector<SegmentedHost<real>> seg; }   (seg empty if fixed-size)
//   View: { SoAView<const real> records;
//           SegmentedView<real> seg[kMaxSoASeg]{}; }  (null if fixed-size)
// kSeg = ragged slots used (0 fixed-size / 2 B-spline / 4 B-spline surface).
// Primary template undefined: a missing specialization fails HasEntitySoA at
// compile time. records is a (count, kFields) mdspan — trivially copyable,
// shared as the one typed SoAView<const real> so PartitionView holds it
// without const void* erasure.

/// Non-owning (count, kFields) mdspan over packed per-entity records; device-
/// safe and trivially copyable. Row i = entity i's kFields contiguous reals.
template <class T> using SoAView = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;

/// Max segmented (CSR) fields any EntitySoA<E> uses — sized for
/// BSplineSurfaceParam (knots_u, knots_v, ctrl, weights).
inline constexpr int kMaxSoASeg = 4;

/// Per-entity SoA storage trait; specialize per entity type (see above).
template <class E> struct EntitySoA;

/// E models the `EntitySoA` trait: tag, Host, View, load, load_into,
/// tie_view with the exact types. A mis-shaped specialization fails here
/// rather than deep inside a kernel.
template <class E>
concept HasEntitySoA = requires(const typename EntitySoA<E>::View v,
                                typename EntitySoA<E>::Host h,
                                std::size_t i,
                                const E e,
                                SoAView<const real> soa,
                                const SegmentedView<real>* seg) {
    typename EntitySoA<E>::Host;
    typename EntitySoA<E>::View;
    { EntitySoA<E>::tag } -> std::convertible_to<EntityTag>;
    { EntitySoA<E>::load(v, i) } -> std::same_as<E>;
    { EntitySoA<E>::load_into(h, i, e) };
    { EntitySoA<E>::tie_view(soa, seg) } -> std::same_as<typename EntitySoA<E>::View>;
};

/// F is invocable as `f.template operator()<E>()` — gates
/// `dispatch_entity_type` so a malformed callable fails at the concept
/// rather than deep in instantiation.
template <class F, class E>
concept EntityDispatchFn = requires(F f) {
    { f.template operator()<E>() };
};

}  // namespace egg
