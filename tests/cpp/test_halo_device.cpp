// test_halo_device.cpp — SYCL device test of halo_exchange + BlockTopologyDevice
// (src/structured_halo.hpp, Phase 1.2), on EVERY visible device (GPU + CPU).
//
// Two conforming blocks share a face. Each block's interior is seeded with a
// GLOBAL coordinate field that is continuous across the shared face: block 0
// occupies x in [0,3] (node i -> x=i), block 1 occupies x in [3,6] (node i ->
// x=3+i), y = j. The regular face exchange must then make each block's ghost
// shell hold the neighbour's first interior layer — which, because the field is
// globally continuous, equals each block's OWN coordinate extrapolated one node
// past its face. That gives an exact, geometry-checked expected value per ghost.
//
// Requires the acpp toolchain (SYCL).
#include "structured_field.hpp"
#include "structured_halo.hpp"
#include "sycl_devices.hpp"
#include "ut_cfg.hpp"

#include <array>
#include <cstddef>
#include <format>
#include <sycl/sycl.hpp>
#include <vector>

using namespace boost::ut;
using namespace egg;
using egg_test::usable_devices;

namespace
{
using Index = std::array<std::size_t, 2>;

// Node counts: 4 along the shared (x) axis, 3 along the free (y) axis, per block.
constexpr std::size_t kN0 = 4;
constexpr std::size_t kN1 = 3;

// Seed block b's interior so node (i,j) holds coordinate (x0 + i, j).
void seed_interior(const BlockLayout<2>& layout, std::vector<double>& host, std::size_t b,
                   double x0)
{
    const auto shape = layout.interior_shape(b);
    for (std::size_t i = 0; i < shape[0]; ++i) {
        for (std::size_t j = 0; j < shape[1]; ++j) {
            const std::size_t off = layout.interior_node_offset(b, {i, j});
            host[off + 0] = x0 + static_cast<double>(i);
            host[off + 1] = static_cast<double>(j);
        }
    }
}

// Build the regular face-exchange tables for the two-block case:
//   block 0 high-x face (axis 0, side 1)  <->  block 1 low-x face (axis 0, side 0).
void build_face_tables(std::vector<int>& src_block, std::vector<Index>& src_padded,
                       std::vector<int>& dst_block, std::vector<Index>& dst_padded)
{
    for (std::size_t j = 0; j < kN1; ++j) {
        // block 0's ghost just past its high-x face <- block 1's first interior layer (i=1).
        dst_block.push_back(0);
        dst_padded.push_back({kN0 + 1, j + 1});  // padded ghost outside high-x face
        src_block.push_back(1);
        src_padded.push_back({2, j + 1});  // block 1 interior (1,j) -> padded (2,j+1)

        // block 1's ghost just before its low-x face <- block 0's last interior layer (i=N0-2).
        dst_block.push_back(1);
        dst_padded.push_back({0, j + 1});  // padded ghost outside low-x face
        src_block.push_back(0);
        src_padded.push_back({kN0 - 1, j + 1});  // block 0 interior (N0-2,j) -> padded (N0-1,j+1)
    }
}
}  // namespace

int main()
{
    const auto devices = usable_devices();
    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    const BlockLayout<2> layout {{{{kN0, kN1}}, {{kN0, kN1}}}};

    for (const auto& dev : devices) {
        sycl::queue q(dev);
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("halo_exchange on " + name) = [&] {
            boost::ut::log << std::format("  device: {}\n", name);

            BlockField<2> field {q, layout};
            std::vector<double> host(field.size(), 0.0);
            seed_interior(layout, host, 0, 0.0);  // block 0: x in [0,3]
            seed_interior(layout, host, 1, 3.0);  // block 1: x in [3,6]
            field.upload(host);

            std::vector<int> src_block, dst_block;
            std::vector<Index> src_padded, dst_padded;
            build_face_tables(src_block, src_padded, dst_block, dst_padded);

            const BlockTopologyDevice<2> topo {q,         layout,     src_block, src_padded,
                                               dst_block, dst_padded, {},        {}};
            expect(topo.num_entries() == 2 * kN1);
            expect(topo.num_singular() == 0_u);

            halo_exchange<2>(q, field.data(), topo).wait();

            std::vector<double> out;
            field.download(out);

            for (std::size_t j = 0; j < kN1; ++j) {
                // block 0 ghost past high-x face: should carry block 1's (1,j) = (4, j),
                // which equals block 0's own field extrapolated to i = N0 (x = 4).
                const std::size_t g0 = layout.padded_node_offset(0, {kN0 + 1, j + 1});
                expect(out[g0 + 0] == 4.0)
                  << std::format("block0 ghost x at j={}: {} (want 4)", j, out[g0 + 0]);
                expect(out[g0 + 1] == static_cast<double>(j))
                  << std::format("block0 ghost y at j={}: {}", j, out[g0 + 1]);

                // block 1 ghost before low-x face: block 0's (2,j) = (2, j); block 1's
                // own field extrapolated to i = -1 is x = 3 + (-1) = 2.
                const std::size_t g1 = layout.padded_node_offset(1, {0, j + 1});
                expect(out[g1 + 0] == 2.0)
                  << std::format("block1 ghost x at j={}: {} (want 2)", j, out[g1 + 0]);
                expect(out[g1 + 1] == static_cast<double>(j))
                  << std::format("block1 ghost y at j={}: {}", j, out[g1 + 1]);
            }

            // Interior face nodes are untouched by the exchange.
            for (std::size_t j = 0; j < kN1; ++j) {
                const std::size_t f0 = layout.interior_node_offset(0, {kN0 - 1, j});
                expect(out[f0 + 0] == 3.0) << "block0 face node must be unchanged";
                const std::size_t f1 = layout.interior_node_offset(1, {0, j});
                expect(out[f1 + 0] == 3.0) << "block1 face node must be unchanged";
            }
        };
    }
}
