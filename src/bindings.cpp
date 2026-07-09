// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file bindings.cpp
/// pybind11 boundary: context unpack + validation, the resident structured
/// block-Jacobi session, and SYCL-free oracle wrappers for golden tests.
/// Python passes float64; precision is narrowed to egg::real here.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"
#include "solve.hpp"
#include "structured_patch.hpp"
#include "structured_sweep.hpp"
#include "sweep.hpp"

#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace
{
using std::invalid_argument;

sycl::queue select_queue(const std::string& device)
{
    // cpu_selector_v / gpu_selector_v throw sycl::exception with a terse message
    // when no matching device is present; rewrap with a clear, actionable error.
    try {
        // Coarse-grained events: elide per-op backend event creation on the
        // in-order HIP/CUDA stream — lower per-launch latency. Safe here because
        // the sweep loop never inspects returned events; all sync is via
        // q.wait() at chunk boundaries. See AdaptiveCpp doc/extensions.md
        // ACPP_EXT_COARSE_GRAINED_EVENTS, and performance.md "Strong-scaling/
        // latency-bound problems".
        constexpr auto coarse = sycl::property::queue::AdaptiveCpp_coarse_grained_events {};
        if (device == "cpu") {
            return sycl::queue {sycl::cpu_selector_v, {coarse}};
        } else if (device == "gpu") {
            return sycl::queue {sycl::gpu_selector_v, {coarse}};
        }
        return sycl::queue {{coarse}};  // "auto" — default selector
    } catch (const sycl::exception& e) {
        throw std::runtime_error("select_queue: no SYCL device available for device='" + device +
                                 "' (" + e.what() +
                                 "). Try device='cpu', or check the AdaptiveCpp install / "
                                 "ACPP_VISIBILITY_MASK.");
    }
}

// num_nodes from a flat X array, with a hard shape check: X must be 1-D with
// length a multiple of d. Catches the silent-UB case where an (N, d) array is
// passed and num_nodes is miscomputed.
std::size_t
  checked_num_nodes(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
                    int d)
{
    auto buf = X_arr.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument(
          "X must be a flat 1-D array of length d*N; got ndim=" + std::to_string(buf.ndim) +
          " (did you pass an (N, d) array? ravel() it first).");
    }
    if (buf.shape[0] % d != 0) {
        throw std::invalid_argument("X length must be a multiple of d (" + std::to_string(d) +
                                    " coordinates per node); got " + std::to_string(buf.shape[0]) +
                                    ".");
    }
    return static_cast<std::size_t>(buf.shape[0]) / d;
}

// Host-side validation of an unpacked context: array-length invariants, the
// ragged sum(P_of) == total_samples contract, and index bounds for every
// gc/gn/dof_idx. Throws before any device work, so a malformed context never
// reaches the kernels as silent OOB/UB — the single biggest guard for a
// memory-safe core.
template <int D> void validate_context(const egg::SweepContextHostT<D>& host)
{
    const std::size_t N = host.num_nodes;

    auto check_index = [N](const std::vector<int>& idx, const char* name, const std::string& grp) {
        for (int v : idx) {
            if (std::cmp_less(v, 0) || std::cmp_greater_equal(v, N)) {
                throw std::invalid_argument("validate_context: " + grp + " " + name + " index " +
                                            std::to_string(v) + " out of range [0, " +
                                            std::to_string(N) + ")");
            }
        }
    };

    if (host.X.size() != N * D) {
        throw std::invalid_argument("validate_context: X size " + std::to_string(host.X.size()) +
                                    " != d*num_nodes " + std::to_string(N * D));
    }

    for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
        const auto& g = host.groups[gi];
        const std::string grp = "group[" + std::to_string(gi) + "]";
        const std::size_t ts = g.total_samples;
        auto eq = [&](std::size_t got, std::size_t want, const std::string& what) {
            if (got != want) {
                // NOLINTNEXTLINE(performance-inefficient-string-concatenation): cold throw path
                throw invalid_argument("validate_context: " + grp + " " + what + " size " +
                                       std::to_string(got) + " != " + std::to_string(want));
            }
        };
        eq(g.gc.size(), ts, "gc");
        for (int k = 0; k < D; ++k) {
            eq(g.gn[k].size(), ts, "gn" + std::to_string(k));
            eq(g.s[k].size(), ts, "s" + std::to_string(k));
        }
        eq(g.role.size(), ts, "role");
        if (g.w_uniform) {
            eq(g.W_inv.size(), egg::dim::wInv(D), "W_inv (uniform row)");
        } else {
            eq(g.W_inv.size(), ts * egg::dim::wInv(D), "W_inv");
        }
        // Weight is optional (unit when omitted); one shared value or one per sample.
        if (!g.weight.empty()) {
            eq(g.weight.size(), g.wt_uniform ? std::size_t {1} : ts, "weight");
        }
        // J is optional (recomputed in-kernel); only size-check it when supplied.
        if (!g.J.empty()) { eq(g.J.size(), ts * egg::dim::jSize(D), "J"); }
        eq(g.dof_idx.size(), g.ndof, "dof_idx");
        eq(g.P_of.size(), g.ndof, "P_of");

        // role selects the per-corner stencil branch (corner / neighbour axis /
        // absent) and, via c0 = D*role, the J columns in patch_eval — an
        // out-of-range role would read past jCols into the next sample's J.
        for (int r : g.role) {
            if (r < -1 || r > egg::dim::nbrs(D)) {
                throw std::invalid_argument("validate_context: " + grp + " role " +
                                            std::to_string(r) + " out of range [-1, " +
                                            std::to_string(egg::dim::nbrs(D)) + "]");
            }
        }

        std::size_t sumP = 0;
        for (int p : g.P_of) {
            if (p < 0) {
                throw std::invalid_argument("validate_context: " + grp + " negative P_of");
            }
            sumP += static_cast<std::size_t>(p);
        }
        if (sumP != ts) {
            throw std::invalid_argument("validate_context: " + grp + " sum(P_of) " +
                                        std::to_string(sumP) + " != total_samples " +
                                        std::to_string(ts));
        }

        check_index(g.gc, "gc", grp);
        for (int k = 0; k < D; ++k) {
            check_index(g.gn[k], ("gn" + std::to_string(k)).c_str(), grp);
        }
        check_index(g.dof_idx, "dof_idx", grp);

        // Per-type SoA record + segmented field validation. Also verify the
        // SoA partitions cover every group-local DOF exactly once: entity data
        // is carried solely by `soa`, so an uncovered DOF would silently never
        // be swept.
        std::vector<int> covered(g.ndof, 0);
        for (std::size_t si = 0; si < g.soa.size(); ++si) {
            const auto& s = g.soa[si];
            const std::string snm = grp + " soa[" + std::to_string(si) +
                                    "] (tag=" + std::to_string(egg::to_int(s.tag)) + ")";
            if (s.records.size() != s.count * static_cast<std::size_t>(s.k_fields)) {
                throw std::invalid_argument(
                  "validate_context: " + snm + " records size " + std::to_string(s.records.size()) +
                  " != count*kFields " +
                  std::to_string(s.count * static_cast<std::size_t>(s.k_fields)));
            }
            if (s.dof_local.size() != s.count) {
                throw std::invalid_argument("validate_context: " + snm + " dof_local size " +
                                            std::to_string(s.dof_local.size()) + " != count " +
                                            std::to_string(s.count));
            }
            for (std::size_t j = 0; j < s.seg.size(); ++j) {
                if (s.seg[j].off.size() != s.count + 1) {
                    throw std::invalid_argument("validate_context: " + snm + " seg[" +
                                                std::to_string(j) + "].off size " +
                                                std::to_string(s.seg[j].off.size()) +
                                                " != count+1 " + std::to_string(s.count + 1));
                }
            }
            for (int dl : s.dof_local) {
                if (dl < 0 || static_cast<std::size_t>(dl) >= g.ndof) {
                    throw std::invalid_argument("validate_context: " + snm + " dof_local " +
                                                std::to_string(dl) + " out of range [0, " +
                                                std::to_string(g.ndof) + ")");
                }
                if (covered[static_cast<std::size_t>(dl)]++ != 0) {
                    throw std::invalid_argument("validate_context: " + snm + " dof_local " +
                                                std::to_string(dl) +
                                                " covered by more than one SoA partition");
                }
            }
        }
        for (std::size_t d = 0; d < g.ndof; ++d) {
            if (covered[d] == 0) {
                throw std::invalid_argument("validate_context: " + grp + " DOF " +
                                            std::to_string(d) +
                                            " not covered by any SoA partition");
            }
        }
    }

    const auto& es = host.energy_stencil;
    const std::size_t ns = es.num_samples;
    auto eqs = [&](std::size_t got, std::size_t want, const std::string& what) {
        if (got != want) {
            throw std::invalid_argument("validate_context: energy_stencil " + what + " size " +
                                        std::to_string(got) + " != " + std::to_string(want));
        }
    };
    eqs(es.gc.size(), ns, "gc");
    for (int k = 0; k < D; ++k) {
        eqs(es.gn[k].size(), ns, "gn" + std::to_string(k));
        eqs(es.s[k].size(), ns, "s" + std::to_string(k));
    }
    // W_inv is either one row per sample or a single shared row (uniform target,
    // read with stride 0 by the energy reduction — mirrors the group metric table).
    if (es.W_inv.size() != egg::dim::wInv(D)) {
        eqs(es.W_inv.size(), ns * egg::dim::wInv(D), "W_inv");
    }
    // Weight is optional; one shared value or one per sample.
    if (!es.weight.empty() && es.weight.size() != 1) {
        eqs(es.weight.size(), ns, "weight");
    }
    check_index(es.gc, "gc", "energy_stencil");
    for (int k = 0; k < D; ++k) {
        check_index(es.gn[k], ("gn" + std::to_string(k)).c_str(), "energy_stencil");
    }
}

// Extract a contiguous int32 numpy array from a dict key, returning a vector.
std::vector<int> extract_int(const py::dict& d, const std::string& key)
{
    auto arr = d[key.c_str()].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    return std::vector<int>(arr.data(), arr.data() + arr.size());
}

// Extract a contiguous float64 numpy array from a dict key, narrowing into the
// device value type egg::real. The Python API stays float64; precision is
// converted here at the boundary (a no-op copy when real == double).
std::vector<egg::real> extract_real(const py::dict& d, const std::string& key)
{
    auto arr =
      d[key.c_str()].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    const double* p = arr.data();
    std::vector<egg::real> v(static_cast<std::size_t>(arr.size()));
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<egg::real>(p[i]); }
    return v;
}

// Widen an egg::real span back to a freshly-allocated float64 numpy array (the
// download side of the boundary; a plain copy when real == double).
py::array_t<double> to_f64(const egg::real* src, py::ssize_t n)
{
    py::array_t<double> out(n);
    double* dst = out.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i) { dst[i] = static_cast<double>(src[i]); }
    return out;
}

// Narrow a contiguous float64 numpy array into an egg::real buffer (for the
// SYCL-free math bindings that take const real* / real-typed views). The
// returned vector must outlive the consuming call.
std::vector<egg::real>
  narrow(const py::array_t<double, py::array::c_style | py::array::forcecast>& a)
{
    const double* p = a.data();
    std::vector<egg::real> v(static_cast<std::size_t>(a.size()));
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<egg::real>(p[i]); }
    return v;
}

template <int D>
egg::SweepContextHostT<D>
  unpack_context(const py::dict& ctx_arrays, const double* X_data, std::size_t num_nodes)
{
    using namespace egg;

    SweepContextHostT<D> host;
    host.num_nodes = num_nodes;
    host.X.assign(X_data, X_data + (num_nodes * D));

    // Groups — one ragged group over all free DOFs. Per-sample arrays are flat
    // over the concatenated DOFs (DOF-major, variable P per DOF); per-DOF
    // arrays carry P_of, and sample_offset is derived here as its prefix sum.
    // The wire keeps per-axis keys gn0../s0.., mapped into gn[k]/s[k].
    auto groups_list = ctx_arrays["groups"].cast<py::list>();
    for (auto g_item : groups_list) {
        auto gd = g_item.cast<py::dict>();
        SweepGroupHostT<D> sg;
        sg.ndof = gd["D"].cast<std::size_t>();

        sg.gc = extract_int(gd, "gc");
        for (int k = 0; k < D; ++k) {
            sg.gn[k] = extract_int(gd, "gn" + std::to_string(k));
            sg.s[k] = extract_real(gd, "s" + std::to_string(k));
        }
        sg.W_inv = extract_real(gd, "W_inv");
        // A single-row W_inv (length wInv, not total_samples*wInv) is a uniform
        // target: every sample shares it, read with stride 0 in the metric table.
        sg.w_uniform = (sg.W_inv.size() == egg::dim::wInv(D));
        // Per-sample energy weight (composed interface-orthogonality term). Optional:
        // a single-value array is a uniform weight (read with stride 0); omission
        // means unit weight everywhere.
        if (gd.contains("weight")) {
            sg.weight = extract_real(gd, "weight");
            sg.wt_uniform = (sg.weight.size() == 1);
        }
        sg.role = extract_int(gd, "role");
        // J is optional: no kernel reads it (the role-selected chain-Jacobian
        // block is recomputed in-kernel from s + W_inv). Contexts may omit it.
        if (gd.contains("J")) { sg.J = extract_real(gd, "J"); }
        sg.dof_idx = extract_int(gd, "dof_idx");
        sg.P_of = extract_int(gd, "P_of");
        // Interior DOFs come pre-classified by build_flat_context: their patch is
        // synthesized off the block layout (identity target), so they carry no
        // stored stencil (P_of == 0). interior_block[d] >= 0 names the block,
        // interior_logical the (D-wide) 0-based logical index; both empty on a wire
        // that predates the classification (then nothing is synthesized).
        if (gd.contains("interior_block")) {
            sg.interior_block = extract_int(gd, "interior_block");
            sg.interior_logical = extract_int(gd, "interior_logical");
        }
        // Block-interface C2 curvature window table (2D): per-DOF fixed-K arrays
        // (curv_nodes [ndof*K*4], curv_slot [ndof*K], curv_weight [ndof*K]). The
        // node ids are global here; apply_structured_remap rewrites them to the
        // structured owner slots (like the energy stencil).
        if (gd.contains("curv_k")) {
            sg.curv_k = gd["curv_k"].cast<int>();
            if (sg.curv_k > 0) {
                sg.curv_nodes = extract_int(gd, "curv_nodes");
                sg.curv_slot = extract_int(gd, "curv_slot");
                sg.curv_weight = extract_real(gd, "curv_weight");
            }
        }

        // Typed per-entity-type SoA sub-dicts (the sole carrier of entity
        // data). Each present type contributes {tag, kFields, dof_local,
        // records, [seg]}: records is a packed (count, kFields) row-major
        // array, seg a list of {data, off} CSR fields (absent for fixed-size
        // types). Every DOF is covered by exactly one sub-dict (validated).
        if (gd.contains("entities")) {
            auto entities = gd["entities"].cast<py::dict>();
            for (auto [name, val] : entities) {
                auto ed = val.cast<py::dict>();
                if (!ed.contains("tag")) {  // "__blob__" marker: unsupported
                    throw std::invalid_argument(
                      "unpack_context: entities carries a '__blob__' group (an entity "
                      "with no SoA encoder, e.g. a composite with a nested B-spline); "
                      "the positional blob path was retired");
                }
                egg::SoAHostRecord rec;
                rec.tag = static_cast<egg::EntityTag>(ed["tag"].cast<int>());
                rec.k_fields = ed["kFields"].cast<int>();
                rec.records = extract_real(ed, "records");
                rec.dof_local = extract_int(ed, "dof_local");  // group-local DOF indices
                rec.count = rec.dof_local.size();
                // Ingest segmented CSR fields (B-spline knots/ctrl). Each slot
                // is {data: f64, off: i32[count+1]}. Absent for fixed-size types.
                if (ed.contains("seg")) {
                    auto seg_list = ed["seg"].cast<py::list>();
                    for (auto seg_val : seg_list) {
                        auto sd = seg_val.cast<py::dict>();
                        egg::SoAHostRecord::SegmentedField sf;
                        sf.data = extract_real(sd, "data");
                        sf.off = extract_int(sd, "off");
                        rec.seg.push_back(std::move(sf));
                    }
                }
                sg.soa.push_back(std::move(rec));
            }
        }

        sg.total_samples = sg.gc.size();

        // sample_offset[d] = Σ_{k<d} P_of[k] (exclusive prefix sum).
        sg.sample_offset.resize(sg.ndof);
        std::exclusive_scan(sg.P_of.begin(), sg.P_of.end(), sg.sample_offset.begin(), 0);

        host.groups.push_back(std::move(sg));
    }

    // Energy stencil
    auto es = ctx_arrays["energy_stencil"].cast<py::dict>();
    host.energy_stencil.num_samples = es["num_samples"].cast<std::size_t>();
    host.energy_stencil.gc = extract_int(es, "gc");
    for (int k = 0; k < D; ++k) {
        host.energy_stencil.gn[k] = extract_int(es, "gn" + std::to_string(k));
        host.energy_stencil.s[k] = extract_real(es, "s" + std::to_string(k));
    }
    host.energy_stencil.W_inv = extract_real(es, "W_inv");
    // Optional per-sample energy weight; omission ⇒ unit weight (one shared row).
    if (es.contains("weight")) { host.energy_stencil.weight = extract_real(es, "weight"); }

    // Fail fast on a malformed context before any device allocation / kernel.
    validate_context<D>(host);

    return host;
}

// Extract a (rows, D) int32 numpy array as a vector of per-row D-index arrays.
// An empty (0, D) array yields an empty vector (no entries to exchange).
template <int D>
std::vector<std::array<std::size_t, D>> extract_index_rows(const py::dict& d,
                                                           const std::string& key)
{
    auto arr = d[key.c_str()].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    const auto info = arr.request();
    const std::size_t rows =
      (info.ndim >= 1 && info.shape[0] > 0) ? static_cast<std::size_t>(info.shape[0]) : 0;
    std::vector<std::array<std::size_t, D>> out(rows);
    const int* p = arr.data();
    for (std::size_t r = 0; r < rows; ++r) {
        for (int k = 0; k < D; ++k) { out[r][k] = static_cast<std::size_t>(p[(r * D) + k]); }
    }
    return out;
}

// The result of re-homing a global SweepContextHostT onto a halo-padded
// structured store: the packing layout and the global-DOF -> structured-node-index
// map (also used to gather the padded X back to global node order after the run).
template <int D> struct StructuredRemap {
    egg::BlockLayout<D> layout;
    std::vector<int> g2s;          // global node id -> owner-interior (padded) node index
    std::size_t global_num_nodes;  // M (pre-remap host.num_nodes)
    // Singular-fan mirrors: extra interior->ghost copies (already as real offsets)
    // the host allocated when a patch node fell outside the regular interior+face
    // ghost shell. Appended to the device halo so they refresh every sweep.
    std::vector<std::size_t> fan_src_off;
    std::vector<std::size_t> fan_dst_off;
    // Owner -> non-owner shared-node broadcast, derived here from the per-block
    // interior maps (lowest-index block owns each global node; the others hold a
    // copy refreshed from the owner every sweep). Element offsets into packed X.
    std::vector<std::size_t> share_src_off;
    std::vector<std::size_t> share_dst_off;
};

// Re-index a global-indexed SweepContextHostT onto the halo-padded structured
// store described by `structured` (interior shapes, per-block global-DOF maps,
// halo table + owner map, built host-side by build_block_structured_context).
// Mutates `host` in place: gc/gn/dof_idx and the energy-stencil indices are
// remapped global -> structured node index, host.X becomes the packed buffer
// (interiors scattered, ghosts 0), and host.num_nodes becomes the padded node
// count. BlockLayout owns all offset math (no padded-index arithmetic in Python).
//
// Conforming multiblock (step 2): a shared interface node is duplicated across
// blocks but relaxed only by its OWNER block (owner_block, lowest index). Each
// group's DOF is therefore remapped relative to its owner block: own-block nodes
// resolve to the owner's interior slot, cross-block neighbours to the owner's
// GHOST copy (frozen per sweep). The energy stencil + the X gather-back use the
// single owner-interior slot per global node (g2s).
template <int D>
StructuredRemap<D> apply_structured_remap(egg::SweepContextHostT<D>& host,
                                          const py::dict& structured)
{
    using namespace egg;

    auto shapes_list = structured["interior_shapes"].cast<py::list>();
    std::vector<typename BlockLayout<D>::Shape> shapes;
    shapes.reserve(py::len(shapes_list));
    for (auto s : shapes_list) {
        auto arr = s.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        if (arr.size() != D) {
            throw std::invalid_argument("structured: each interior shape must have D entries");
        }
        typename BlockLayout<D>::Shape sh;
        for (int k = 0; k < D; ++k) { sh[k] = static_cast<std::size_t>(arr.data()[k]); }
        shapes.push_back(sh);
    }
    BlockLayout<D> layout {shapes};
    const std::size_t num_blocks = shapes.size();

    const std::size_t M = host.num_nodes;
    const std::vector<egg::real>& Xg = host.X;

    // Unravel a row-major flat logical index (last axis fastest), matching the
    // C-order build_block_structured_context flattens block_global_dof in.
    auto unravel = [](std::size_t f,
                      const BlockLayout<D>::Shape& shape) -> std::array<std::size_t, D> {
        std::array<std::size_t, D> logical {};
        std::size_t rem = f;
        for (int k = D - 1; k >= 0; --k) {
            logical[static_cast<std::size_t>(k)] = rem % shape[static_cast<std::size_t>(k)];
            rem /= shape[static_cast<std::size_t>(k)];
        }
        return logical;
    };
    auto ravel = [](const std::array<std::size_t, D>& logical,
                    const BlockLayout<D>::Shape& shape) -> std::size_t {
        std::size_t f = 0;
        for (int k = 0; k < D; ++k) { f = (f * shape[static_cast<std::size_t>(k)]) + logical[k]; }
        return f;
    };

    // Cache the per-block global-DOF maps.
    auto bgd_list = structured["block_global_dof"].cast<py::list>();
    if (py::len(bgd_list) != num_blocks) {
        throw std::invalid_argument("structured: block_global_dof count != interior_shapes count");
    }
    std::vector<std::vector<int>> bgd(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        auto arr = bgd_list[b].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        std::size_t n = 1;
        for (int k = 0; k < D; ++k) { n *= shapes[b][static_cast<std::size_t>(k)]; }
        if (std::cmp_not_equal(arr.size(), n)) {
            throw std::invalid_argument("structured: block_global_dof length != prod(shape)");
        }
        bgd[b].assign(arr.data(), arr.data() + arr.size());
    }

    // Per-block map: global node id -> padded node index in that block. Interior
    // slots come from block_global_dof; the ghost shell from the halo table (a
    // dst-block ghost holds the src-block's interior node, keyed by its global id).
    std::vector<std::unordered_map<int, int>> block_map(num_blocks);
    std::vector<egg::real> Xp(layout.total_reals(), egg::real {0});
    // Owner block + owner interior slot of each global node, derived in this same
    // interior pass: the lowest-index block that contains a node owns (relaxes)
    // it; every later block holds a copy refreshed by an owner->copy broadcast.
    std::vector<int> owner_block(M, -1);
    std::vector<int> owner_slot(M, -1);
    std::vector<std::size_t> share_src_off, share_dst_off;
    // Pass 1: interior map, lowest-index owner, and the initial-position scatter.
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t n = bgd[b].size();
        for (std::size_t f = 0; f < n; ++f) {
            const std::array<std::size_t, D> logical = unravel(f, shapes[b]);
            const int gid = bgd[b][f];
            const int sidx = interior_node_index<D>(layout, b, logical);
            block_map[b][gid] = sidx;
            if (owner_block[static_cast<std::size_t>(gid)] < 0) {
                owner_block[static_cast<std::size_t>(gid)] = static_cast<int>(b);
                owner_slot[static_cast<std::size_t>(gid)] = sidx;
            }
            // Scatter the initial position into every block's copy (owner +
            // non-owner) of the node; the broadcast keeps them in sync thereafter.
            for (int k = 0; k < D; ++k) {
                Xp[(static_cast<std::size_t>(sidx) * D) + k] =
                  Xg[(static_cast<std::size_t>(gid) * D) + k];
            }
        }
    }

    // A singular node is a shared corner of every block in its fan and may be
    // relaxed in any of them. Python (build_block_structured_context) picks a fan
    // block with spare ghost-ring slots as its host (sing_block/sing_logical), so
    // its fan mirror has room even when the lowest-index block is a full interior
    // one. Override the owner with that choice before the broadcasts are built.
    {
        const std::vector<int> sblk = extract_int(structured, "sing_block");
        const auto slog = extract_index_rows<D>(structured, "sing_logical");
        for (std::size_t s = 0; s < sblk.size(); ++s) {
            const auto ob = static_cast<std::size_t>(sblk[s]);
            std::array<std::size_t, D> lg {};
            for (int k = 0; k < D; ++k) { lg[static_cast<std::size_t>(k)] = slog[s][k]; }
            const int gid = bgd[ob][ravel(lg, shapes[ob])];
            owner_block[static_cast<std::size_t>(gid)] = static_cast<int>(ob);
            owner_slot[static_cast<std::size_t>(gid)] = interior_node_index<D>(layout, ob, lg);
        }
    }

    // Pass 2: owner -> non-owner-copy broadcasts, built after the owner overrides
    // so each copy sources from the final owner (copy order is immaterial, see
    // fused_halo_broadcast).
    for (std::size_t b = 0; b < num_blocks; ++b) {
        const std::size_t n = bgd[b].size();
        for (std::size_t f = 0; f < n; ++f) {
            const int gid = bgd[b][f];
            if (owner_block[static_cast<std::size_t>(gid)] == static_cast<int>(b)) { continue; }
            const std::array<std::size_t, D> logical = unravel(f, shapes[b]);
            share_src_off.push_back(
              static_cast<std::size_t>(owner_slot[static_cast<std::size_t>(gid)]) * D);
            share_dst_off.push_back(
              static_cast<std::size_t>(interior_node_index<D>(layout, b, logical)) * D);
        }
    }

    const std::vector<int> halo_src_block = extract_int(structured, "halo_src_block");
    const std::vector<int> halo_dst_block = extract_int(structured, "halo_dst_block");
    const auto halo_src_padded = extract_index_rows<D>(structured, "halo_src_padded");
    const auto halo_dst_padded = extract_index_rows<D>(structured, "halo_dst_padded");
    for (std::size_t e = 0; e < halo_src_block.size(); ++e) {
        const auto sb = static_cast<std::size_t>(halo_src_block[e]);
        const auto db = static_cast<std::size_t>(halo_dst_block[e]);
        // The halo source is an INTERIOR padded index (1-based per axis); recover
        // the neighbour's global id, then register db's ghost slot under it.
        std::array<std::size_t, D> src_logical {};
        for (int k = 0; k < D; ++k) {
            src_logical[static_cast<std::size_t>(k)] = halo_src_padded[e][k] - 1;
        }
        const int gid = bgd[sb][ravel(src_logical, shapes[sb])];
        block_map[db][gid] = padded_node_index<D>(layout, db, halo_dst_padded[e]);
    }

    // owner_block[] was derived in the interior pass above (the only block that
    // relaxes each global DOF).

    // g2s[g] = the owner-interior slot of global node g (energy stencil + gather-back).
    std::vector<int> g2s(M, -1);
    for (std::size_t g = 0; g < M; ++g) {
        const int ob = owner_block[g];
        if (ob < 0) { continue; }  // node in no block's interior (never referenced)
        const auto it = block_map[static_cast<std::size_t>(ob)].find(static_cast<int>(g));
        if (it == block_map[static_cast<std::size_t>(ob)].end()) {
            throw std::invalid_argument("structured: owner_block names a block that does not "
                                        "contain its global node");
        }
        g2s[g] = it->second;
    }

    // Spare ghost-ring slots per block, for foreign patch nodes (singular fans).
    // A few of a singular node's fan neighbours cross non-axis-aligned interfaces,
    // so they are not in any face-ghost slot; we mirror each into an otherwise
    // unused ghost-ring slot of the owner block and refresh it every sweep from the
    // node's owner interior (frozen halo). The pool excludes interior slots (never
    // on the ring) and the regular face-ghost destinations.
    std::vector<std::unordered_set<int>> used_ghost(num_blocks);
    for (std::size_t e = 0; e < halo_dst_block.size(); ++e) {
        used_ghost[static_cast<std::size_t>(halo_dst_block[e])].insert(
          padded_node_index<D>(layout,
                               static_cast<std::size_t>(halo_dst_block[e]),
                               halo_dst_padded[e]));
    }
    std::vector<std::vector<int>> free_ghosts(num_blocks);
    std::vector<std::size_t> free_cursor(num_blocks, 0);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        std::array<std::size_t, D> ext {};
        std::size_t total = 1;
        for (int k = 0; k < D; ++k) {
            ext[static_cast<std::size_t>(k)] = shapes[b][static_cast<std::size_t>(k)] + 2;
            total *= ext[static_cast<std::size_t>(k)];
        }
        for (std::size_t f = 0; f < total; ++f) {
            std::array<std::size_t, D> padded {};
            std::size_t rem = f;
            bool ring = false;
            for (int k = D - 1; k >= 0; --k) {
                padded[static_cast<std::size_t>(k)] = rem % ext[static_cast<std::size_t>(k)];
                rem /= ext[static_cast<std::size_t>(k)];
            }
            for (int k = 0; k < D; ++k) {
                if (padded[static_cast<std::size_t>(k)] == 0 ||
                    padded[static_cast<std::size_t>(k)] == ext[static_cast<std::size_t>(k)] - 1) {
                    ring = true;
                }
            }
            if (!ring) { continue; }
            const int nidx = padded_node_index<D>(layout, b, padded);
            if (!used_ghost[b].contains(nidx)) { free_ghosts[b].push_back(nidx); }
        }
    }

    std::vector<std::size_t> fan_src_off, fan_dst_off;

    // Remap one global node id relative to block b's interior-or-ghost map. A node
    // outside it is a singular-fan neighbour: allocate it a spare ghost slot in b,
    // sourced from its owner interior, and record the extra halo copy.
    auto lookup = [&](std::size_t b, int gid, const char* what) -> int {
        const auto it = block_map[b].find(gid);
        if (it != block_map[b].end()) { return it->second; }
        const int src = g2s[static_cast<std::size_t>(gid)];  // owner-interior node index
        if (src < 0) {
            throw std::invalid_argument(std::string("structured: ") + what +
                                        " references global node " + std::to_string(gid) +
                                        " absent from every block's interior map");
        }
        if (free_cursor[b] >= free_ghosts[b].size()) {
            throw std::invalid_argument(
              "structured: block " + std::to_string(b) +
              " ran out of spare ghost slots for singular-fan / foreign patch nodes");
        }
        const int slot = free_ghosts[b][free_cursor[b]++];
        block_map[b][gid] = slot;
        fan_src_off.push_back(static_cast<std::size_t>(src) * D);
        fan_dst_off.push_back(static_cast<std::size_t>(slot) * D);
        return slot;
    };

    // Each group's DOF is remapped relative to its OWNER block: the relaxed node
    // and all its patch samples resolve to that block's interior or ghost slots.
    for (auto& g : host.groups) {
        for (std::size_t d = 0; d < g.ndof; ++d) {
            const auto ob =
              static_cast<std::size_t>(owner_block[static_cast<std::size_t>(g.dof_idx[d])]);
            const auto off = static_cast<std::size_t>(g.sample_offset[d]);
            const auto P = static_cast<std::size_t>(g.P_of[d]);
            for (std::size_t p = off; p < off + P; ++p) {
                g.gc[p] = lookup(ob, g.gc[p], "gc");
                for (int k = 0; k < D; ++k) { g.gn[k][p] = lookup(ob, g.gn[k][p], "gn"); }
            }
            g.dof_idx[d] = lookup(ob, g.dof_idx[d], "dof_idx");
        }
        // soa[*].dof_local are GROUP-LOCAL DOF positions, not global ids — never remapped.
    }

    // Interior DOFs (patch synthesized off the block layout) are classified by
    // build_flat_context and arrive on the wire as interior_block/interior_logical
    // (see the unpack above); nothing to compute here.

    auto remap_stencil = [&](std::vector<int>& v) {
        for (int& x : v) {
            const int s = g2s[static_cast<std::size_t>(x)];
            if (s < 0) {
                throw std::invalid_argument(
                  "structured: an energy-stencil index references a global node absent from "
                  "every block's interior map");
            }
            x = s;
        }
    };
    remap_stencil(host.energy_stencil.gc);
    for (int k = 0; k < D; ++k) { remap_stencil(host.energy_stencil.gn[k]); }

    // Curvature-window nodes resolve to their OWNER structured slot (g2s): the
    // packed X is global-readable and read frozen per sweep, so every DOF reads
    // the same node value regardless of block. Pad slots (-1) pass through.
    for (auto& g : host.groups) {
        for (int& x : g.curv_nodes) {
            if (x < 0) { continue; }
            const int s = g2s[static_cast<std::size_t>(x)];
            if (s < 0) {
                throw std::invalid_argument(
                  "structured: a curvature-window node references a global node absent from "
                  "every block's interior map");
            }
            x = s;
        }
    }

    host.X = std::move(Xp);
    host.num_nodes = layout.total_reals() / static_cast<std::size_t>(D);

    // Flatten the block layout into node-unit arrays the device synthesis reads
    // (block base + padded node strides per block). Doubles/D are exact: a block
    // base and every stride is a whole multiple of D (D coords per node).
    host.block_off.assign(num_blocks, 0);
    host.nstride.assign(num_blocks * static_cast<std::size_t>(D), 0);
    for (std::size_t b = 0; b < num_blocks; ++b) {
        host.block_off[b] = static_cast<int>(layout.block_offset(b) / static_cast<std::size_t>(D));
        const auto ns = layout.node_strides(b);
        for (int k = 0; k < D; ++k) {
            host.nstride[(b * static_cast<std::size_t>(D)) + static_cast<std::size_t>(k)] =
              static_cast<int>(ns[static_cast<std::size_t>(k)] / static_cast<std::size_t>(D));
        }
    }

    return StructuredRemap<D> {std::move(layout),
                               std::move(g2s),
                               M,
                               std::move(fan_src_off),
                               std::move(fan_dst_off),
                               std::move(share_src_off),
                               std::move(share_dst_off)};
}

// Build the device halo topology from the host-side padded-index tables, plus
// any singular-fan mirror copies the remap allocated (pre-resolved to offsets).
template <int D>
egg::BlockTopologyDevice<D> build_topology(sycl::queue& q,
                                           const egg::BlockLayout<D>& layout,
                                           const py::dict& structured,
                                           const std::vector<std::size_t>& share_src_off,
                                           const std::vector<std::size_t>& share_dst_off,
                                           const std::vector<std::size_t>& fan_src_off,
                                           const std::vector<std::size_t>& fan_dst_off)
{
    return egg::BlockTopologyDevice<D> {q,
                                        layout,
                                        extract_int(structured, "halo_src_block"),
                                        extract_index_rows<D>(structured, "halo_src_padded"),
                                        extract_int(structured, "halo_dst_block"),
                                        extract_index_rows<D>(structured, "halo_dst_padded"),
                                        extract_int(structured, "sing_block"),
                                        extract_index_rows<D>(structured, "sing_logical"),
                                        share_src_off,
                                        share_dst_off,
                                        fan_src_off,
                                        fan_dst_off};
}

// Runtime dispatch on the input's actual dimension d: 2 and 3 are built.
int require_supported_dim(int d)
{
    if (d != 2 && d != 3) {
        throw std::invalid_argument("egg C++ core: dimension must be 2 or 3, got " +
                                    std::to_string(d));
    }
    return d;
}

// A device-resident structured session: the executor (owning the packed X +
// halo topology) and the gather-back remap for one dimension.
template <int D> struct StructuredSession {
    egg::StructuredExecutorT<D> exec;
    StructuredRemap<D> rm;
};

// Build a device-resident structured session: re-home the global context onto
// the halo-padded store once, upload it, and keep it resident.
template <int D>
StructuredSession<D> make_structured_session(const py::dict& ctx_arrays,
                                             const py::dict& structured,
                                             const double* X,
                                             std::size_t num_nodes,
                                             const std::string& device)
{
    sycl::queue q = select_queue(device);
    auto host = unpack_context<D>(ctx_arrays, X, num_nodes);
    StructuredRemap<D> rm = apply_structured_remap<D>(host, structured);
    auto topo = build_topology<D>(q,
                                  rm.layout,
                                  structured,
                                  rm.share_src_off,
                                  rm.share_dst_off,
                                  rm.fan_src_off,
                                  rm.fan_dst_off);
    // Host copies of the conforming face tables (ghost fills + owner→copy
    // broadcasts; fans/singularities stay fine-level-only), so the FAS
    // hierarchy can derive coarse interface topology (mg_topology.hpp).
    egg::HaloTablesHost<D> halo_host {extract_int(structured, "halo_src_block"),
                                      extract_index_rows<D>(structured, "halo_src_padded"),
                                      extract_int(structured, "halo_dst_block"),
                                      extract_index_rows<D>(structured, "halo_dst_padded"),
                                      rm.share_src_off,
                                      rm.share_dst_off};
    egg::StructuredExecutorT<D> exec(q, host, std::move(topo), std::move(halo_host));
    return StructuredSession<D> {std::move(exec), std::move(rm)};
}

// Persistent device-resident structured block-Jacobi session. The context is
// re-homed onto the halo-padded store and uploaded once; X stays resident
// (packed) across .run() calls, so the per-sweep halo exchange + coalesced
// stencil reads are measured warm with no re-staging.
class CppStructuredSweepSession
{
  public:
    CppStructuredSweepSession(
      const py::dict& ctx_arrays,
      const py::dict& structured,
      const py::array_t<double, py::array::c_style | py::array::forcecast>& X0,
      const std::string& device,
      int dim) :
        dim_(require_supported_dim(dim)), num_nodes_(checked_num_nodes(X0, dim_)),
        sess_(dim_ == 2 ? SessVariant {std::in_place_type<StructuredSession<2>>,
                                       make_structured_session<2>(
                                         ctx_arrays, structured, X0.data(), num_nodes_, device)}
                        : SessVariant {std::in_place_type<StructuredSession<3>>,
                                       make_structured_session<3>(
                                         ctx_arrays, structured, X0.data(), num_nodes_, device)})
    {
    }

    // Run n_sweeps of block-Jacobi on the resident packed X (per-sweep halo +
    // broadcast included); omega is the SOR/damping weight.
    // @p report_every throttles the energy/min-det reduction cadence; default 0
    // (chunk-end) returns a single (energy, min_det) pair for the final sweep,
    // cutting up to (n_sweeps - 1) reduction launches per call. Pass 1 for the
    // legacy per-sweep contract.
    std::tuple<py::array_t<double>, py::array_t<double>>
      run(int n_sweeps, const std::string& phase, double delta, double omega, int report_every)
    {
        auto [energies, mindets] = std::visit(
          [&]<int D>(StructuredSession<D>& s) {
              const egg::ObjectiveKindT<D> objective = egg::make_objective<D>(phase, delta);
              return s.exec.run_jacobi(n_sweeps, objective, omega, report_every);
          },
          sess_);
        const auto n_reported = static_cast<py::ssize_t>(energies.size());
        return std::make_tuple(to_f64(energies.data(), n_reported),
                               to_f64(mindets.data(), n_reported));
    }

    // Run n_cycles of FAS V-cycles (fas.hpp) on the resident packed X. Shape
    // phase only: phase="untangle" raises ValueError. Returns per-cycle
    // (energies, mindets); falls back to plain block-Jacobi with the same
    // fine-sweep budget (per-cycle report cadence) when no coarse level can
    // be built.
    std::tuple<py::array_t<double>, py::array_t<double>> run_fas(int n_cycles,
                                                                 int nu_pre,
                                                                 int nu_post,
                                                                 int nu_coarse,
                                                                 double omega,
                                                                 int max_levels,
                                                                 const std::string& phase)
    {
        auto [energies, mindets] = std::visit(
          [&]<int D>(StructuredSession<D>& s) {
              egg::FasOptions opts;
              opts.n_cycles = n_cycles;
              opts.nu_pre = nu_pre;
              opts.nu_post = nu_post;
              opts.nu_coarse = nu_coarse;
              opts.omega = static_cast<egg::real>(omega);
              opts.max_levels = max_levels;
              const egg::ObjectiveKindT<D> objective = egg::make_objective<D>(phase, 0.0);
              return s.exec.run_fas(opts, objective);
          },
          sess_);
        const auto n_reported = static_cast<py::ssize_t>(energies.size());
        return std::make_tuple(to_f64(energies.data(), n_reported),
                               to_f64(mindets.data(), n_reported));
    }

    // Coarse-hierarchy shape: one list per level, one (n0, ..., n_{D-1})
    // interior-node-count tuple per block. Empty when no level can be built.
    py::list mg_levels()
    {
        return std::visit(
          [&]<int D>(StructuredSession<D>& s) {
              py::list levels;
              for (const auto& shapes : s.exec.mg_level_shapes()) {
                  py::list blocks;
                  for (const auto& shape : shapes) {
                      py::tuple t(D);
                      for (int k = 0; k < D; ++k) { t[k] = shape[static_cast<std::size_t>(k)]; }
                      blocks.append(t);
                  }
                  levels.append(blocks);
              }
              return levels;
          },
          sess_);
    }

    // Download the packed buffer and gather it back to global node order.
    py::array_t<double> get_X()
    {
        return std::visit(
          [&]<int D>(StructuredSession<D>& s) {
              std::vector<egg::real> X_pad(s.rm.layout.total_reals());
              s.exec.ctx().download_X(X_pad.data());
              std::vector<double> X_out(s.rm.global_num_nodes * D);
              for (std::size_t g = 0; g < s.rm.global_num_nodes; ++g) {
                  const auto slot = static_cast<std::size_t>(s.rm.g2s[g]);
                  for (int k = 0; k < D; ++k) {
                      X_out[(g * D) + k] = static_cast<double>(X_pad[(slot * D) + k]);
                  }
              }
              return py::array_t<double>(static_cast<py::ssize_t>(s.rm.global_num_nodes * D),
                                         X_out.data());
          },
          sess_);
    }

  private:
    using SessVariant = std::variant<StructuredSession<2>, StructuredSession<3>>;
    int dim_;
    std::size_t num_nodes_;
    SessVariant sess_;
};

}  // namespace

PYBIND11_MODULE(cpp_core, m)
{
    m.doc() = "egg C++ compute core (AdaptiveCpp).";
    m.def("ping", [] { return 0; }, "Liveness check; returns 0.");

    // The compiled de Boor degree caps, so host-side encoders can reject an
    // over-degree B-spline with a clean error instead of letting it index the
    // fixed device work arrays out of bounds (a segfault). Curve and surface
    // caps are decoupled (see geometry.hpp): the surface cap drives the 3D
    // boundary kernel's VGPR budget, the curve cap carries 2D-profile headroom.
    m.attr("MAX_BSPLINE_DEGREE") = egg::kMaxBSplineDegree;
    m.attr("MAX_BSPLINE_CURVE_DEGREE") = egg::kMaxBSplineCurveDegree;

    // The compiled precision of egg::real (float vs double). Cross-precision
    // parity tests read this to floor their double-tuned tolerances at the fp32
    // level in the float build (see tests/real_tol.py, the mirror of
    // tests/cpp/real_tol.hpp). True when built with -DEGG_REAL_IS_FP32=ON.
    m.attr("REAL_IS_FLOAT") = (sizeof(egg::real) == 4);

    py::class_<CppStructuredSweepSession>(
      m,
      "CppStructuredSweepSession",
      "Persistent device-resident structured smoothing session.\n\n"
      "Runs structured block-Jacobi: the context is re-homed onto\n"
      "the halo-padded per-block store and uploaded once, and the packed X stays\n"
      "device-resident across .run() calls, so the coalesced stencil + per-sweep\n"
      "halo exchange are measured warm with no re-staging.")
      .def(py::init<py::dict,
                    py::dict,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    const std::string&,
                    int>(),
           py::arg("ctx_arrays"),
           py::arg("structured"),
           py::arg("X"),
           py::kw_only(),
           py::arg("device") = "auto",
           py::arg("dim") = 2)
      .def("run",
           &CppStructuredSweepSession::run,
           py::arg("n_sweeps"),
           py::kw_only(),
           py::arg("phase") = "barrier",
           py::arg("delta") = 0.0,
           py::arg("omega") = 1.0,
           py::arg("report_every") = 0,
           "Run n_sweeps of block-Jacobi on the resident packed X; returns\n"
           "(energies, mindets). omega is the SOR/damping weight.\n\n"
           "report_every: 0 (default) reports a single (energy, min_det) pair for\n"
           "the final sweep of this call (chunk-end cadence, minimal reduction\n"
           "launches); 1 reports every sweep (legacy per-sweep contract); k > 1\n"
           "reports every k-th sweep plus the final one.")
      .def("run_fas",
           &CppStructuredSweepSession::run_fas,
           py::arg("n_cycles"),
           py::kw_only(),
           py::arg("nu_pre") = 2,
           py::arg("nu_post") = 2,
           py::arg("nu_coarse") = 32,
           py::arg("omega") = 1.0,
           py::arg("max_levels") = egg::FasOptions {}.max_levels,
           py::arg("phase") = "barrier",
           "Run n_cycles of FAS (nonlinear geometric multigrid) V-cycles on the\n"
           "resident packed X; returns per-cycle (energies, mindets). Each cycle is\n"
           "nu_pre Jacobi sweeps, a tau-corrected coarse solve — recursing down the\n"
           "factor-2 hierarchy (at most max_levels levels, fine included) to a\n"
           "coarsest level solved with Newton sweeps scaled to its interior\n"
           "diameter (capped by nu_coarse) — a safeguarded coarse-grid\n"
           "correction (backtracking on the global energy), and nu_post sweeps.\n"
           "Shape phase only: phase=\"untangle\" raises ValueError. Falls back to\n"
           "plain block-Jacobi with the equivalent fine-sweep budget when the block\n"
           "shapes admit no coarse level (or max_levels < 2).")
      .def("mg_levels",
           &CppStructuredSweepSession::mg_levels,
           "Coarse-hierarchy shape: one list per level of per-block interior\n"
           "node-count tuples; empty when no coarse level can be built.")
      .def("get_X",
           &CppStructuredSweepSession::get_X,
           "Download X gathered back to global node order, shape (N*dim,).");

    m.def(
      "cpp_structured_sweep",
      [](const py::dict& ctx_arrays,
         const py::dict& structured,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         int n_sweeps,
         const std::string& device,
         const std::string& phase,
         double delta,
         int dim,
         double omega,
         int report_every)
        -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> {
          require_supported_dim(dim);

          const auto run = [&]<int D>() {
              sycl::queue q = select_queue(device);

              // Build the global-indexed context (validated here), then re-home
              // it onto the halo-padded structured store: indices become
              // structured node indices, X the packed buffer, num_nodes the
              // padded count. The structured executor then runs block-Jacobi
              // with a per-sweep halo exchange.
              const std::size_t num_nodes = checked_num_nodes(X_arr, dim);
              auto host_ctx = unpack_context<D>(ctx_arrays, X_arr.data(), num_nodes);
              StructuredRemap<D> rm = apply_structured_remap<D>(host_ctx, structured);
              egg::BlockTopologyDevice<D> topo = build_topology<D>(q,
                                                                   rm.layout,
                                                                   structured,
                                                                   rm.share_src_off,
                                                                   rm.share_dst_off,
                                                                   rm.fan_src_off,
                                                                   rm.fan_dst_off);

              egg::StructuredExecutorT<D> exec(q, host_ctx, std::move(topo));
              const egg::ObjectiveKindT<D> objective = egg::make_objective<D>(phase, delta);
              auto [energies, mindets] = exec.run_jacobi(n_sweeps, objective, omega, report_every);

              // Download the padded buffer and gather it back to global node order.
              std::vector<egg::real> X_pad(rm.layout.total_reals());
              exec.ctx().download_X(X_pad.data());
              std::vector<double> X_out(rm.global_num_nodes * D);
              for (std::size_t g = 0; g < rm.global_num_nodes; ++g) {
                  const auto s = static_cast<std::size_t>(rm.g2s[g]);
                  for (int k = 0; k < D; ++k) {
                      X_out[(g * D) + k] = static_cast<double>(X_pad[(s * D) + k]);
                  }
              }

              const auto n_reported = static_cast<py::ssize_t>(energies.size());
              py::array_t<double> X_ret(static_cast<py::ssize_t>(rm.global_num_nodes * D),
                                        X_out.data());
              py::array_t<double> e_ret = to_f64(energies.data(), n_reported);
              py::array_t<double> m_ret = to_f64(mindets.data(), n_reported);
              return std::make_tuple(X_ret, e_ret, m_ret);
          };
          return dim == 2 ? run.template operator()<2>() : run.template operator()<3>();
      },
      py::arg("ctx_arrays"),
      py::arg("structured"),
      py::arg("X"),
      py::arg("n_sweeps"),
      py::kw_only(),
      py::arg("device") = "auto",
      py::arg("phase") = "barrier",
      py::arg("delta") = 0.0,
      py::arg("dim") = 2,
      py::arg("omega") = 1.0,
      py::arg("report_every") = 0,
      "Run n_sweeps of block-Jacobi over a halo-padded structured store.\n\n"
      "The context is re-homed onto per-block halo-padded arrays (coalesced stencil\n"
      "reads) with a per-sweep halo exchange and shared-node broadcast; omega is the\n"
      "SOR/damping weight. Conforming multiblock (shared interfaces) is supported via\n"
      "owner-per-shared-node + cross-block-neighbour ghost remap (frozen-halo additive\n"
      "Schwarz, cadence 1.4b).\n\n"
      "Parameters\n"
      "----------\n"
      "ctx_arrays : dict\n"
      "    Flattened context from cpp_backend.flatten_context (global indices).\n"
      "structured : dict\n"
      "    Halo-padded structured tables from build_block_structured_context\n"
      "    (interior_shapes, block_global_dof, halo_*, sing_*).\n"
      "X : ndarray, shape (N*d,)\n"
      "    Initial node positions, flat (global node order).\n"
      "X : ndarray, shape (N*d,)\n"
      "    Initial node positions, flat (global node order).\n\n"
      "Returns\n"
      "-------\n"
      "(X_out, energies, mindets) — X_out in global node order.");

    // --------------------------------------------------------------------------
    // Oracle primitives — host-side wrappers for golden-generator reference values.
    // SYCL-free; these call the inline C++ header functions directly.
    // --------------------------------------------------------------------------

    m.def(
      "metric_eval",
      [](
        const py::array_t<double, py::array::c_style | py::array::forcecast>& t_arr,
        const std::string& metric) -> std::tuple<double, py::array_t<double>, py::array_t<double>> {
          const auto* t = t_arr.data();
          egg::VecT vt {static_cast<egg::real>(t[0]),
                        static_cast<egg::real>(t[1]),
                        static_cast<egg::real>(t[2]),
                        static_cast<egg::real>(t[3])};
          if (metric != "shape_2d" && metric != "shape_size") {
              throw py::value_error("metric_eval: unknown metric '" + metric + "'");
          }
          const egg::ObjectiveKind objective = egg::make_objective<2>(metric, 0.0);
          return std::visit(
            [&](const auto& obj) {
                const auto mu = static_cast<double>(obj.value(vt));
                auto g = obj.grad(vt);
                auto h = obj.hess(vt);
                return std::make_tuple(mu, to_f64(g.data(), 4), to_f64(h.data(), 16));
            },
            objective);
      },
      py::arg("t"),
      py::arg("metric") = "shape_2d",
      "Evaluate a named 2D metric (\"shape_2d\" or \"shape_size\"). "
      "Returns (mu, grad(4,), hess(16,)).");

    m.def(
      "metric_eval_3d",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& t_arr)
        -> std::tuple<double, py::array_t<double>, py::array_t<double>> {
          const std::vector<egg::real> t = narrow(t_arr);
          egg::MuCond3Eval r = egg::mu_cond3_eval(t.data());
          return std::make_tuple(static_cast<double>(r.val), to_f64(r.grad, 9), to_f64(r.hess, 81));
      },
      py::arg("t"),
      "Evaluate mu_cond3 metric (3D). vec(T) row-major [t00..t22]. "
      "Returns (mu, grad(9,), hess(81,)).");

    m.def(
      "geometry_project",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& p_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          egg::Pt p {static_cast<egg::real>(p_arr.data()[0]),
                     static_cast<egg::real>(p_arr.data()[1])};
          const std::vector<egg::real> params = narrow(params_arr);
          egg::Pt proj = egg::project(p, tag, params.data());
          return to_f64(proj.data(), 2);
      },
      py::arg("p"),
      py::arg("tag"),
      py::arg("params"),
      "Project point onto entity. Returns proj(2,).");

    m.def(
      "geometry_tangent",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& p_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          egg::Pt p {static_cast<egg::real>(p_arr.data()[0]),
                     static_cast<egg::real>(p_arr.data()[1])};
          const std::vector<egg::real> params = narrow(params_arr);
          egg::Pt tang = egg::tangent_space(p, tag, params.data());
          return to_f64(tang.data(), 2);
      },
      py::arg("p"),
      py::arg("tag"),
      py::arg("params"),
      "Tangent direction at point on entity. Returns tang(2,).");

    m.def(
      "patch_eval",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gc_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn0_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s0_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& W_inv_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& role_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& J_arr)
        -> std::tuple<py::array_t<double>, py::array_t<double>, double, double> {
          // Narrow the real-typed views at the boundary (no-ops when real==double).
          const std::vector<egg::real> X = narrow(X_arr);
          // s is ±1; PatchView stores it as int8, so narrow the signs to match.
          const auto to_sign = [](const py::array_t<double>& a) {
              std::vector<std::int8_t> out(static_cast<std::size_t>(a.size()));
              const double* p = a.data();
              for (std::size_t i = 0; i < out.size(); ++i) {
                  out[i] = static_cast<std::int8_t>(p[i]);
              }
              return out;
          };
          const std::vector<std::int8_t> s0 = to_sign(s0_arr);
          const std::vector<std::int8_t> s1 = to_sign(s1_arr);
          const std::vector<egg::real> W_inv = narrow(W_inv_arr);
          // J_arr is accepted for API stability but ignored: patch_eval recomputes
          // the role-selected chain-Jacobian block from s + W_inv (role_Jb).
          static_cast<void>(J_arr);
          egg::PatchView pv;
          pv.P = static_cast<int>(gc_arr.size());
          // Identity sample_id: this single patch's arrays are its own metric table.
          std::vector<int> sid(static_cast<std::size_t>(pv.P));
          for (int p = 0; p < pv.P; ++p) { sid[static_cast<std::size_t>(p)] = p; }
          pv.sample_id = sid.data();
          pv.gc = gc_arr.data();
          pv.gn[0] = gn0_arr.data();
          pv.gn[1] = gn1_arr.data();
          pv.s[0] = s0.data();
          pv.s[1] = s1.data();
          pv.W_inv = W_inv.data();
          pv.role = role_arr.data();
          auto result = egg::patch_eval(pv, X.data());
          return std::make_tuple(to_f64(result.grad.data(), 2),
                                 to_f64(result.hess.data(), 4),
                                 static_cast<double>(result.energy),
                                 static_cast<double>(result.mindet));
      },
      py::arg("X"),
      py::arg("gc"),
      py::arg("gn0"),
      py::arg("gn1"),
      py::arg("s0"),
      py::arg("s1"),
      py::arg("W_inv"),
      py::arg("role"),
      py::arg("J"),
      "Per-DOF patch evaluation. Returns (grad(2,), hess(4,), energy, mindet).");

    m.def(
      "energy_mindet",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& X_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gc_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn0_arr,
         const py::array_t<int, py::array::c_style | py::array::forcecast>& gn1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s0_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& s1_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& W_inv_arr)
        -> std::tuple<double, double> {
          const std::vector<egg::real> X = narrow(X_arr);
          const std::vector<egg::real> s0 = narrow(s0_arr);
          const std::vector<egg::real> s1 = narrow(s1_arr);
          const std::vector<egg::real> W_inv = narrow(W_inv_arr);
          egg::StencilSampleView sv;
          sv.P = static_cast<int>(gc_arr.size());
          sv.gc = gc_arr.data();
          sv.gn[0] = gn0_arr.data();
          sv.gn[1] = gn1_arr.data();
          sv.s[0] = s0.data();
          sv.s[1] = s1.data();
          sv.W_inv = W_inv.data();
          egg::real energy {0};
          egg::real mindet {0};
          egg::patch_energy_mindet<2>(sv, X.data(), energy, mindet);
          return std::make_tuple(static_cast<double>(energy), static_cast<double>(mindet));
      },
      py::arg("X"),
      py::arg("gc"),
      py::arg("gn0"),
      py::arg("gn1"),
      py::arg("s0"),
      py::arg("s1"),
      py::arg("W_inv"),
      "Cheap patch energy + min-det only. Returns (energy, mindet).");

    m.def(
      "newton_step",
      [](const py::array_t<double, py::array::c_style | py::array::forcecast>& grad_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& hess_arr,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& pos_arr,
         int tag,
         const py::array_t<double, py::array::c_style | py::array::forcecast>& params_arr)
        -> py::array_t<double> {
          const auto* g = grad_arr.data();
          const auto* h = hess_arr.data();
          egg::Vec2 gv {static_cast<egg::real>(g[0]), static_cast<egg::real>(g[1])};
          egg::Mat2 hv {static_cast<egg::real>(h[0]),
                        static_cast<egg::real>(h[1]),
                        static_cast<egg::real>(h[2]),
                        static_cast<egg::real>(h[3])};
          egg::Pt pos {static_cast<egg::real>(pos_arr.data()[0]),
                       static_cast<egg::real>(pos_arr.data()[1])};
          const std::vector<egg::real> params = narrow(params_arr);
          egg::Vec2 delta = egg::newton_delta<2>(gv, hv, pos, tag, params.data());
          return to_f64(delta.data(), 2);
      },
      py::arg("grad"),
      py::arg("hess"),
      py::arg("pos"),
      py::arg("tag"),
      py::arg("params"),
      "Tangent-reduced Newton step. Returns delta(2,).");
}
