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

// ===========================================================================
// Strongly-typed, data-oriented entity storage primitives.
//
// This header holds the *transport* layer that replaces the positional
// `(int tag, real params[kParamPad])` blob: the strong entity tag, the single
// segmented-array (CSR) idiom for variable-length payloads, and the per-entity
// structure-of-arrays trait. It is deliberately geometry-agnostic — the concrete
// `EntitySoA<E>` specializations (which name the entity types) live in
// geometry.hpp, where those types are defined. Nothing here depends on the math.
// ===========================================================================

/// @brief Strongly-typed entity kind tag (replaces the loose `using Tag = int`).
///
/// The underlying integer values are FROZEN: they are the cross-language wire
/// contract shared with the Python encoder and the geometry golden tables, and
/// must stay identical to the legacy `TAG_*` constants in geometry.hpp (a
/// `static_assert` there locks the two together). 3D-only kinds are listed so the
/// 2D/3D entity sets share one tag space; in the 2D set they map to `Free`.
///
/// The base type is fixed to `int` deliberately: it is the frozen wire width
/// (the Python boundary passes int32 tags; the golden tables store `int`), not a
/// size to be optimised — hence the enum-size lint is suppressed here.
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

/// @brief Underlying-integer accessor, for the wire / Python / golden boundary.
/// @param t The entity tag.
/// @return The frozen integer value of @p t.
[[nodiscard]] constexpr int to_int(EntityTag t) { return static_cast<int>(t); }

// ---------------------------------------------------------------------------
// Segmented (CSR) payload: flat concatenated data + a per-entity offset table.
//
// This is the single idiom for every variable-length entity payload — B-spline
// knot vectors / control nets / weights, surface UV-trim polygons, and composite
// segment records — mirroring the layout the sweep already uses for ragged patch
// samples (`P_of` / `sample_offset` in sweep.hpp). Entity `i` owns the slice
// `data[off[i] .. off[i + 1])`; `off` has length `count + 1` with `off[0] == 0`.
// ---------------------------------------------------------------------------

/// @brief Host-side concatenated segmented array: flat @c data + CSR @c off table.
/// @tparam T Element type of the payload (e.g. `real`).
template <class T> struct SegmentedHost {
    std::vector<T> data;   ///< Concatenated payload of every entity, in entity order.
    std::vector<int> off;  ///< CSR offsets, length @c count+1 (@c off[0]==0).

    /// @brief Number of entities described by the offset table.
    [[nodiscard]] std::size_t count() const { return off.empty() ? 0 : off.size() - 1; }

    /// @brief Append entity's slice @p s, extending the concatenated data + CSR table.
    ///
    /// Lazily seeds `off` with the leading `0` on the first push, then concatenates
    /// @p s and records the new running length — so a sequence of `push_back`s
    /// builds the CSR `(data, off)` pair incrementally in entity order.
    /// @param s The variable-length payload of the next entity.
    void push_back(std::span<const T> s)
    {
        if (off.empty()) { off.push_back(0); }
        data.insert(data.end(), s.begin(), s.end());
        off.push_back(static_cast<int>(data.size()));
    }
};

/// @brief Trivially-copyable device view over a @ref SegmentedHost (raw ptr + offsets).
/// @tparam T Element type of the payload.
template <class T> struct SegmentedView {
    const T* data = nullptr;
    const int* off = nullptr;
    /// @brief The contiguous slice owned by entity @p i.
    /// @param i Entity index.
    /// @return A span over @c data[off[i] .. off[i+1]).
    [[nodiscard]] std::span<const T> operator[](std::size_t i) const
    {
        const auto b = static_cast<std::size_t>(off[i]);
        const auto e = static_cast<std::size_t>(off[i + 1]);
        return {data + b, e - b};
    }
};

// ---------------------------------------------------------------------------
// Per-entity structure-of-arrays trait — the single field schema.
//
// `EntitySoA<E>` is specialized per entity type (in geometry.hpp). Each
// specialization composes the entity's storage out of named field arrays —
// packed contiguous records (one flat `real[count*kFields]` per partition,
// stride `kFields` per entity) for fixed-size fields, SegmentedHost/
// SegmentedView for variable-length ones — and supplies:
//
// - `load(View, i) -> E`:  the ONE device-side builder that reconstructs a
//   typed `E` from its SoA view at index `i` (replacing that entity's
//   `make_entity` arm).
// - `load_into(Host&, i, const E&)`:  the inverse — scatters a typed `E` into
//   the host build target at row `i` (fills `records` + appends to `seg[j]`).
// - `tie_view(SoAView, SegmentedView*) -> View`:  constructs the typed `View`
//   from `PartitionView`'s generic slots, called once per launch (cold).
//
// All specializations share one uniform `Host`/`View` shape:
//   Host: { std::vector<real> records; std::size_t count;
//           std::vector<SegmentedHost<real>> seg; }  (seg empty for fixed-size)
//   View: { SoAView<const real> records;
//           SegmentedView<real> seg[kMaxSoASeg]{}; }  (null for fixed-size)
//
// `kSeg` (0 for fixed-size, 2 for B-spline, 4 for B-spline surface) is the
// number of segmented slots the specialization actually uses. The primary
// template is left undefined: an entity lacking a specialization fails the
// HasEntitySoA concept at compile time.
//
// The fixed-size View's `records` is a 2-D mdspan with extents (count,
// kFields): non-owning, trivially-copyable (captured by value into SYCL
// kernels), and indexed by `view.records(i, FIELD)` — no raw pointers. All
// specializations share the one typed `SoAView<const real>` for `records`
// so `PartitionView` can hold it directly without `const void*` type erasure.
// ---------------------------------------------------------------------------

/// @brief Non-owning 2-D mdspan over packed per-entity records: `(count, kFields)`.
///
/// SYCL-device-safe (mdspan, not `std::span`); trivially copyable, captured by
/// value into kernels. Row `i` is entity `i`'s `kFields` contiguous reals.
template <class T>
using SoAView = stdex::mdspan<T, stdex::dextents<std::size_t, 2>>;

/// @brief Maximum number of segmented (CSR) fields any `EntitySoA<E>` can use.
///
/// Sized for the 3D `BSplineSurfaceParam` (4 ragged fields: knots_u, knots_v,
/// ctrl, weights) at the `feat_3d_math` merge. All `EntitySoA<E>::View` structs
/// carry `SegmentedView<real> seg[kMaxSoASeg]{};` — null for fixed-size types
/// (kSeg == 0), filled for B-spline (kSeg == 2) and B-spline surface (kSeg == 4).
inline constexpr int kMaxSoASeg = 4;

/// @brief Per-entity SoA storage trait. Specialize per entity type @p E.
/// @tparam E The entity type whose field schema this describes.
template <class E> struct EntitySoA;

/// @brief Constrains a type @p E to model the @ref EntitySoA trait.
///
/// Requires the storage tag, a host build target (`Host`), a trivially-copyable
/// device view (`View`), the device-side builder `load(view, i) -> E`, the
/// host-side scatter `load_into(host, i, e)`, and the per-type View factory
/// `tie_view(soa, seg) -> View`. The `same_as<E>` on `load` ties the schema
/// back to its entity type; the `same_as<View>` on `tie_view` ensures the
/// factory produces the right view type. A mis-shaped specialization fails
/// here rather than deep inside a kernel.
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

/// @brief Constrains a callable @p F to accept a compile-time entity type @p E.
///
/// The counterpart to @ref HasEntitySoA for @ref dispatch_entity_type: a
/// callable models `EntityDispatchFn<F, E>` if it can be invoked as
/// `f.template operator()<E>()`. Gate `dispatch_entity_type`'s `F` with this so
/// a malformed callable fails at the concept rather than deep in instantiation.
template <class F, class E>
concept EntityDispatchFn = requires(F f) {
    { f.template operator()<E>() };
};

}  // namespace egg
