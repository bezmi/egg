// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_structured.cpp — host-side unit tests for BlockLayout<D> (src/structured.hpp),
// the padded-shape / offset / stride math of the structured store. No SYCL: this suite
// builds under the default compiler in the `cpp_tests` executable.
//
// The layout packs halo-padded structured blocks (one ghost node per face) into
// a single flat double buffer, row-major over the padded axes with the D
// coordinate components innermost. These tests pin the derived shapes, offsets,
// strides, and node addresses against hand-computed values for 2D and 3D.
#include "structured.hpp"
#include "structured_patch.hpp"
#include "ut_cfg.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace boost::ut;
using namespace egg;

static const suite<"structured"> structured_suite = [] {
    "empty layout is zero-sized"_test = [] {
        const BlockLayout<2> layout {};
        expect(layout.num_blocks() == 0_u);
        expect(layout.total_doubles() == 0_u);
    };

    // Two 2D blocks: interior (3,4) and (2,2). Padded shapes (5,6) and (4,4).
    // Block 0: 5*6=30 nodes * 2 coords = 60 doubles, offset 0.
    // Block 1: 4*4=16 nodes * 2 coords = 32 doubles, offset 60. Total 92.
    "2D packing: shapes, offsets, total"_test = [] {
        const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};
        expect(layout.num_blocks() == 2_u);

        expect(layout.padded_shape(0) == std::array<std::size_t, 2> {5, 6});
        expect(layout.padded_shape(1) == std::array<std::size_t, 2> {4, 4});
        expect(layout.padded_node_count(0) == 30_u);
        expect(layout.padded_node_count(1) == 16_u);

        expect(layout.block_offset(0) == 0_u);
        expect(layout.block_offset(1) == 60_u);
        expect(layout.total_doubles() == 92_u);
    };

    // Strides are row-major over padded axes with D coords innermost: for padded
    // shape (5,6), D=2 -> stride[1]=2 (=D), stride[0]=2*6=12.
    "2D node strides are row-major, innermost == D"_test = [] {
        const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};
        expect(layout.node_strides(0) == std::array<std::size_t, 2> {12, 2});
        expect(layout.node_strides(1) == std::array<std::size_t, 2> {8, 2});
    };

    // Interior node = padded index shifted +1 per axis. Block 0, logical (0,0)
    // -> padded (1,1) -> 12*1 + 2*1 = 14. Max interior (2,3) -> padded (3,4) ->
    // 12*3 + 2*4 = 44 (< 60, in block 0). Block 1 is offset by 60.
    "2D interior/padded node offsets"_test = [] {
        const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};
        expect(layout.interior_node_offset(0, {0, 0}) == 14_u);
        expect(layout.padded_node_offset(0, {1, 1}) == 14_u);
        expect(layout.interior_node_offset(0, {2, 3}) == 44_u);
        // Ghost corner (0,0) of block 0 is the buffer base.
        expect(layout.padded_node_offset(0, {0, 0}) == 0_u);
        // Block 1 interior (0,0) -> base 60 + 8*1 + 2*1 = 70.
        expect(layout.interior_node_offset(1, {0, 0}) == 70_u);
    };

    // 3D block interior (2,3,4) -> padded (4,5,6): 120 nodes * 3 coords = 360
    // doubles; strides stride[2]=3 (=D), stride[1]=3*6=18, stride[0]=18*5=90.
    "3D shapes, strides, offsets"_test = [] {
        const BlockLayout<3> layout {{{{2, 3, 4}}}};
        expect(layout.padded_shape(0) == std::array<std::size_t, 3> {4, 5, 6});
        expect(layout.padded_node_count(0) == 120_u);
        expect(layout.total_doubles() == 360_u);
        expect(layout.node_strides(0) == std::array<std::size_t, 3> {90, 18, 3});
        // Interior (0,0,0) -> padded (1,1,1) -> 90+18+3 = 111.
        expect(layout.interior_node_offset(0, {0, 0, 0}) == 111_u);
        // Interior (1,2,3) -> padded (2,3,4) -> 180+54+12 = 246 (< 360).
        expect(layout.interior_node_offset(0, {1, 2, 3}) == 246_u);
    };

    // Every interior node addresses a distinct in-range slot; the +1 ghost shift
    // never collides with another block. Sweep all interior nodes of a 2-block
    // layout and assert offsets are unique and within the packed buffer.
    "interior node offsets are unique and in-range"_test = [] {
        const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};
        const std::size_t total = layout.total_doubles();
        std::array<int, 92> hit {};  // total_doubles() == 92
        bool ok = true;
        for (std::size_t b = 0; b < layout.num_blocks(); ++b) {
            const auto shape = layout.interior_shape(b);
            for (std::size_t i = 0; i < shape[0]; ++i) {
                for (std::size_t j = 0; j < shape[1]; ++j) {
                    const std::size_t off = layout.interior_node_offset(b, {i, j});
                    if (off + 1 >= total || hit[off] != 0) { ok = false; }
                    hit[off] = 1;
                }
            }
        }
        expect(ok) << "interior node offsets must be unique and leave room for D coords";
    };

    // The mdspan view builders must address the same doubles as the offset math.
    // Fill a host buffer so buf[off] == off, then assert v(idx…, k) reads `off`.
    // (Same builders are used on the device base pointer in test_field_device.)
    "2D halo/interior views read the packed offsets"_test = [] {
        const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};
        std::vector<egg::real> buf(layout.total_doubles());
        for (std::size_t i = 0; i < buf.size(); ++i) { buf[i] = static_cast<double>(i); }

        for (std::size_t b = 0; b < layout.num_blocks(); ++b) {
            const HaloView<2> hv = halo_view<2>(buf.data(), layout, b);
            const InteriorView<2> iv = interior_view<2>(buf.data(), layout, b);
            const auto padded = layout.padded_shape(b);
            const auto interior = layout.interior_shape(b);

            // Halo view spans the whole padded block including ghosts.
            expect(hv.extent(0) == padded[0] && hv.extent(1) == padded[1] && hv.extent(2) == 2_u);
            for (std::size_t i = 0; i < padded[0]; ++i) {
                for (std::size_t j = 0; j < padded[1]; ++j) {
                    for (std::size_t k = 0; k < 2; ++k) {
                        expect(hv[i, j, k]
                               == static_cast<double>(layout.padded_node_offset(b, {i, j}) + k));
                    }
                }
            }

            // Interior view is the centred (+1) window; ghosts excluded.
            expect(iv.extent(0) == interior[0] && iv.extent(1) == interior[1]
                   && iv.extent(2) == 2_u);
            for (std::size_t i = 0; i < interior[0]; ++i) {
                for (std::size_t j = 0; j < interior[1]; ++j) {
                    for (std::size_t k = 0; k < 2; ++k) {
                        const double expected =
                          static_cast<double>(layout.interior_node_offset(b, {i, j}) + k);
                        expect(iv[i, j, k] == expected);
                        // interior(i,j) aliases halo(i+1,j+1) — same storage.
                        expect(iv[i, j, k] == (hv[i + 1, j + 1, k]));
                    }
                }
            }
        }
    };

    "3D interior view strides match the layout"_test = [] {
        const BlockLayout<3> layout {{{{2, 3, 4}}}};
        std::vector<egg::real> buf(layout.total_doubles());
        for (std::size_t i = 0; i < buf.size(); ++i) { buf[i] = static_cast<double>(i); }

        const InteriorView<3> iv = interior_view<3>(buf.data(), layout, 0);
        const auto interior = layout.interior_shape(0);
        expect(iv.extent(0) == 2_u && iv.extent(1) == 3_u && iv.extent(2) == 4_u
               && iv.extent(3) == 3_u);
        for (std::size_t i = 0; i < interior[0]; ++i) {
            for (std::size_t j = 0; j < interior[1]; ++j) {
                for (std::size_t l = 0; l < interior[2]; ++l) {
                    for (std::size_t k = 0; k < 3; ++k) {
                        expect(iv[i, j, l, k]
                               == static_cast<double>(
                                 layout.interior_node_offset(0, {i, j, l}) + k));
                    }
                }
            }
        }
    };

    // patch_eval over a halo-padded BlockField buffer (structured node indices
    // from structured_patch.hpp) is bit-identical to patch_eval over a compact
    // global array. Proves the padded store is a drop-in for the global X and
    // that the structured gc/gn read the correct 3x3 (+/-1 per axis)
    // neighbourhood (the structured patch-eval bridge). The two views
    // share the same W_inv/J/role/s; only the node ids differ (compact global vs
    // structured padded), so any divergence is an indexing bug.
    "structured patch_eval matches compact patch_eval"_test = [] {
        constexpr std::size_t n = 3;  // (3,3) interior block; mover (1,1) is interior
        const BlockLayout<2> layout {{{{n, n}}}};

        auto pos = [](std::size_t i, std::size_t j) {
            return PtN<2> {static_cast<egg::real>(i) + 0.05_r * static_cast<egg::real>(i * j),
                           static_cast<egg::real>(j) + 0.03_r * static_cast<egg::real>(i * i)};
        };
        std::vector<egg::real> Xc(n * n * 2);                // compact: global idx = i*n + j
        std::vector<egg::real> buf(layout.total_doubles());  // padded BlockField store
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const PtN<2> p = pos(i, j);
                const std::size_t gc = ((i * n) + j) * 2;
                Xc[gc + 0] = p[0];
                Xc[gc + 1] = p[1];
                const std::size_t off = layout.interior_node_offset(0, {i, j});
                buf[off + 0] = p[0];
                buf[off + 1] = p[1];
            }
        }

        // Mover (1,1)'s patch: its 4 incident cells x 4 corner samples. The W_inv
        // (identity) and J (a fixed nonzero pattern so the hessian check bites)
        // are shared; gc/gn are built twice (compact vs structured padded).
        const std::array<std::size_t, 2> mover {1, 1};
        const std::array<std::array<std::size_t, 2>, 4> cell_bases {
          {{0, 0}, {0, 1}, {1, 0}, {1, 1}}};
        const std::array<std::array<int, 2>, 4> corner_offs {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};

        std::vector<int> gc_c, gn0_c, gn1_c, gc_p, gn0_p, gn1_p, role;
        std::vector<std::int8_t> s0, s1;  // per-axis sign ±1, matches PatchViewT::s
        std::vector<egg::real> W_inv, J;
        double jfill = 0.0;
        for (const auto& base : cell_bases) {
            for (const auto& co : corner_offs) {
                const std::array<std::size_t, 2> corner {base[0] + co[0], base[1] + co[1]};
                const int sx = co[0] == 0 ? 1 : -1;
                const int sy = co[1] == 0 ? 1 : -1;
                const std::array<std::size_t, 2> nb0 {
                  static_cast<std::size_t>(static_cast<int>(corner[0]) + sx), corner[1]};
                const std::array<std::size_t, 2> nb1 {
                  corner[0], static_cast<std::size_t>(static_cast<int>(corner[1]) + sy)};

                auto gidx = [&](const std::array<std::size_t, 2>& l) {
                    return static_cast<int>((l[0] * n) + l[1]);
                };
                gc_c.push_back(gidx(corner));
                gn0_c.push_back(gidx(nb0));
                gn1_c.push_back(gidx(nb1));
                gc_p.push_back(interior_node_index<2>(layout, 0, corner));
                gn0_p.push_back(interior_node_index<2>(layout, 0, nb0));
                gn1_p.push_back(interior_node_index<2>(layout, 0, nb1));
                s0.push_back(static_cast<std::int8_t>(sx));
                s1.push_back(static_cast<std::int8_t>(sy));
                int r = -1;
                if (corner == mover) { r = 0; }
                else if (nb0 == mover) { r = 1; }
                else if (nb1 == mover) { r = 2; }
                role.push_back(r);
                W_inv.insert(W_inv.end(), {1.0, 0.0, 0.0, 1.0});
                for (int t = 0; t < dim::jSize(2); ++t) { J.push_back((jfill += 0.013)); }
            }
        }

        // Identity sample_id: these compact arrays are their own metric table,
        // so occurrence p reads table row p.
        std::vector<int> sid(gc_c.size());
        for (std::size_t i = 0; i < sid.size(); ++i) { sid[i] = static_cast<int>(i); }

        auto make_view = [&](const std::vector<int>& gc, const std::vector<int>& gn0,
                             const std::vector<int>& gn1) {
            PatchViewT<2> pv;
            pv.P = static_cast<int>(gc.size());
            pv.sample_id = sid.data();
            pv.gc = gc.data();
            pv.gn[0] = gn0.data();
            pv.gn[1] = gn1.data();
            pv.s[0] = s0.data();
            pv.s[1] = s1.data();
            pv.W_inv = W_inv.data();
            pv.role = role.data();
            return pv;
        };

        const PatchResultT<2> rc = patch_eval<2>(make_view(gc_c, gn0_c, gn1_c), Xc.data());
        const PatchResultT<2> rp = patch_eval<2>(make_view(gc_p, gn0_p, gn1_p), buf.data());

        expect(rc.energy == rp.energy) << "energy";
        expect(rc.mindet == rp.mindet) << "mindet";
        for (int k = 0; k < 2; ++k) { expect(rc.grad[k] == rp.grad[k]) << "grad component"; }
        for (int k = 0; k < 4; ++k) { expect(rc.hess[k] == rp.hess[k]) << "hess component"; }
    };

    // The synthesized interior patch must reproduce, occurrence for occurrence,
    // the explicit per-DOF patch the host builder emits (cells in row-major base
    // order, corners in row-major offset order). Reference here mirrors the
    // hand-assembled build_context loop: iterate every cell in C-order, and for
    // each cell that has the mover as a corner emit its 2^D corner samples.
    "interior patch synthesis matches the explicit builder order"_test = [] {
        constexpr int D = 3;
        constexpr std::size_t n = 4;  // interior (4,4,4); movers (1,1,1)/(2,2,2) eligible
        const BlockLayout<D> layout {{{{n, n, n}}}};
        const std::array<std::size_t, D> mover {1, 1, 1};

        auto id = [&](const std::array<std::size_t, D>& l) {
            return interior_node_index<D>(layout, 0, l);
        };
        // Build the reference occurrence list exactly as the host builder does.
        std::vector<int> e_gc, e_role;
        std::array<std::vector<int>, D> e_gn;
        std::array<std::vector<egg::real>, D> e_s;
        for (std::size_t ci = 0; ci < n - 1; ++ci) {
            for (std::size_t cj = 0; cj < n - 1; ++cj) {
                for (std::size_t ck = 0; ck < n - 1; ++ck) {
                    const std::array<std::size_t, D> base {ci, cj, ck};
                    bool has_mover = false;
                    for (int o = 0; o < (1 << D); ++o) {
                        std::array<std::size_t, D> corner = base;
                        for (int k = 0; k < D; ++k) { corner[k] += (o >> (D - 1 - k)) & 1; }
                        if (corner == mover) { has_mover = true; }
                    }
                    if (!has_mover) { continue; }
                    for (int o = 0; o < (1 << D); ++o) {
                        std::array<std::size_t, D> corner = base;
                        int ob[D];
                        for (int k = 0; k < D; ++k) {
                            ob[k] = (o >> (D - 1 - k)) & 1;
                            corner[k] += ob[k];
                        }
                        e_gc.push_back(id(corner));
                        int role = (corner == mover) ? 0 : -1;
                        for (int k = 0; k < D; ++k) {
                            const int sk = ob[k] == 0 ? 1 : -1;
                            std::array<std::size_t, D> nbr = corner;
                            nbr[k] = ob[k] == 0 ? nbr[k] + 1 : nbr[k] - 1;
                            e_gn[k].push_back(id(nbr));
                            e_s[k].push_back(static_cast<egg::real>(sk));
                            if (role < 0 && nbr == mover) { role = k + 1; }
                        }
                        e_role.push_back(role);
                    }
                }
            }
        }

        expect(interior_patch_size<D>() == 1 << (2 * D));  // 4^D == 64
        expect(e_gc.size() == static_cast<std::size_t>(interior_patch_size<D>()));

        for (int occ = 0; occ < interior_patch_size<D>(); ++occ) {
            const InteriorOccurrence<D> oc =
              interior_patch_occurrence<D>(layout, 0, mover, occ);
            expect(oc.gc == e_gc[static_cast<std::size_t>(occ)]) << "gc occ" << occ;
            expect(oc.role == e_role[static_cast<std::size_t>(occ)]) << "role occ" << occ;
            for (int k = 0; k < D; ++k) {
                expect(oc.gn[k] == e_gn[k][static_cast<std::size_t>(occ)]) << "gn occ" << occ;
                expect(oc.s[k] == e_s[k][static_cast<std::size_t>(occ)]) << "s occ" << occ;
            }
        }
    };

    // patch_eval_synth (the in-kernel synthesized path) must reproduce the stored
    // patch_eval bit-for-bit when fed the same interior patch — this locks the
    // device synthesis + identity-W_inv evaluation to the audited stored math.
    "synthesized patch_eval matches the stored evaluation"_test = [] {
        constexpr int D = 3;
        constexpr std::size_t n = 5;
        const BlockLayout<D> layout {{{{n, n, n}}}};
        const std::array<std::size_t, D> mover {2, 2, 2};

        auto pos = [](std::size_t i, std::size_t j, std::size_t k) {
            return PtN<3> {static_cast<egg::real>(i) + 0.03_r * static_cast<egg::real>(j),
                           static_cast<egg::real>(j) + 0.02_r * static_cast<egg::real>(k),
                           static_cast<egg::real>(k) + 0.01_r * static_cast<egg::real>(i)};
        };
        std::vector<egg::real> buf(layout.total_doubles());
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t k = 0; k < n; ++k) {
                    const std::size_t off = layout.interior_node_offset(0, {i, j, k});
                    const PtN<3> p = pos(i, j, k);
                    for (int c = 0; c < D; ++c) { buf[off + static_cast<std::size_t>(c)] = p[c]; }
                }
            }
        }

        // Stored arrays for the mover, built from the (trusted) synthesizer with
        // identity W_inv and identity sample_id — what the stored path consumes.
        std::vector<int> gc, role;
        std::array<std::vector<int>, D> gn;
        std::array<std::vector<std::int8_t>, D> s;  // per-axis sign ±1, matches PatchViewT::s
        std::vector<egg::real> W_inv;
        std::vector<int> sid;
        for (int occ = 0; occ < interior_patch_size<D>(); ++occ) {
            const InteriorOccurrence<D> o = interior_patch_occurrence<D>(layout, 0, mover, occ);
            gc.push_back(o.gc);
            role.push_back(o.role);
            for (int k = 0; k < D; ++k) {
                gn[k].push_back(o.gn[k]);
                s[k].push_back(static_cast<std::int8_t>(o.s[k]));
            }
            for (int a = 0; a < D; ++a) {
                for (int c = 0; c < D; ++c) { W_inv.push_back(a == c ? 1.0_r : 0.0_r); }
            }
            sid.push_back(occ);
        }
        PatchViewT<D> pv;
        pv.P = static_cast<int>(gc.size());
        pv.sample_id = sid.data();
        pv.role = role.data();
        pv.gc = gc.data();
        for (int k = 0; k < D; ++k) {
            pv.gn[k] = gn[k].data();
            pv.s[k] = s[k].data();
        }
        pv.W_inv = W_inv.data();
        const PatchResultT<D> rc = patch_eval<D>(pv, buf.data());

        // Synthesized path: node-unit block base + strides from the layout.
        const int block_off[1] = {0};
        const auto ns = layout.node_strides(0);  // doubles
        const int nstride[D] = {static_cast<int>(ns[0] / D), static_cast<int>(ns[1] / D),
                                static_cast<int>(ns[2] / D)};
        const int logical[D] = {2, 2, 2};
        const PatchResultT<D> rs =
          patch_eval_synth<D>(block_off, nstride, 0, logical, buf.data());

        expect(rc.energy == rs.energy) << "energy";
        expect(rc.mindet == rs.mindet) << "mindet";
        for (int k = 0; k < D; ++k) { expect(rc.grad[k] == rs.grad[k]) << "grad"; }
        for (int k = 0; k < D * D; ++k) { expect(rc.hess[k] == rs.hess[k]) << "hess"; }
    };

    // interior_node_index shifts each logical axis +1 into the padded array, so it
    // must equal padded_node_index at logical+1; consecutive interior nodes are one
    // node apart along the fastest axis (the coalescing invariant the structured
    // store is built for). Two unequal blocks so the per-block base offset bites.
    "interior and padded node indices agree on the +1 shift"_test = [] {
        const BlockLayout<3> layout {{{{2, 3, 4}}, {{3, 2, 2}}}};
        for (std::size_t b = 0; b < layout.num_blocks(); ++b) {
            const auto shape = layout.interior_shape(b);
            for (std::size_t i = 0; i < shape[0]; ++i) {
                for (std::size_t j = 0; j < shape[1]; ++j) {
                    for (std::size_t k = 0; k < shape[2]; ++k) {
                        const std::array<std::size_t, 3> logical {i, j, k};
                        const std::array<std::size_t, 3> padded {i + 1, j + 1, k + 1};
                        expect(interior_node_index<3>(layout, b, logical) ==
                               padded_node_index<3>(layout, b, padded));
                    }
                }
            }
        }
        // Fastest axis: consecutive interior nodes are one node apart.
        expect((interior_node_index<3>(layout, 0, {1, 1, 2}) -
                interior_node_index<3>(layout, 0, {1, 1, 1})) == 1);
    };
};
