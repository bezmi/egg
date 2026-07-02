// golden_soa.hpp — host-side blob→SoA transform for the golden sweep context.
//
// The device sweep context (SweepGroupHost) carries only the typed per-
// (colour,EntityTag) SoA records (`soa`); the positional `(tag, params, arena)`
// blob was retired from it in Phase 4. The golden header (gen_golden.py output)
// still stores the blob — it is the frozen cross-language wire contract — so the
// C++ golden test and the bench decode it here into the same SoA the Python wire
// produces in `egg.geometry.entity_soa.group_entities_by_type`. Test/bench only.
#pragma once

#include "sweep.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace egg_test
{

// Partition a group's `ndof` DOFs by EntityTag (first-seen order) and decode each
// DOF's positional blob into its type's packed SoA records + segmented CSR slots,
// returning one egg::SoAHostRecord per present entity type. `dof_local` is the
// group-local DOF list of each partition.
inline std::vector<egg::SoAHostRecord> build_soa_from_blob(const int* tag,
                                                           const egg::real* params,
                                                           const std::vector<egg::real>& arena,
                                                           std::size_t ndof)
{
    using namespace egg;
    std::vector<EntityTag> order;
    std::vector<std::vector<int>> lists;
    for (std::size_t d = 0; d < ndof; ++d) {
        const auto t = static_cast<EntityTag>(tag[d]);
        auto it = std::find(order.begin(), order.end(), t);
        if (it == order.end()) {
            order.push_back(t);
            lists.emplace_back();
            lists.back().push_back(static_cast<int>(d));
        } else {
            lists[static_cast<std::size_t>(it - order.begin())].push_back(static_cast<int>(d));
        }
    }

    const egg::real* arena_ptr = arena.empty() ? nullptr : arena.data();
    std::vector<SoAHostRecord> out;
    out.reserve(order.size());
    for (std::size_t pi = 0; pi < order.size(); ++pi) {
        const auto& dofs = lists[pi];
        SoAHostRecord rec;
        rec.tag = order[pi];
        rec.dof_local = dofs;
        rec.count = dofs.size();
        rec.k_fields = 0;
        dispatch_entity_type<2>(order[pi], [&]<class E>() {
            if constexpr (HasEntitySoA<E>) {
                using SoA = EntitySoA<E>;
                rec.k_fields = SoA::kFields;
                typename SoA::Host buf;
                buf.records.resize(dofs.size() * static_cast<std::size_t>(SoA::kFields));
                buf.seg.resize(SoA::kSeg);
                for (std::size_t i = 0; i < dofs.size(); ++i) {
                    const E ent =
                      decode_entity<E>(params + (static_cast<std::size_t>(dofs[i]) * kParamPad),
                                       arena_ptr);
                    SoA::load_into(buf, i, ent);
                }
                rec.records = std::move(buf.records);
                rec.seg.resize(SoA::kSeg);
                for (int j = 0; j < SoA::kSeg; ++j) {
                    rec.seg[j].data = std::move(buf.seg[j].data);
                    rec.seg[j].off = std::move(buf.seg[j].off);
                }
            }
        });
        out.push_back(std::move(rec));
    }
    return out;
}

}  // namespace egg_test
