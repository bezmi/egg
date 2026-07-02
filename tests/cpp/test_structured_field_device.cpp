// test_structured_field_device.cpp — SYCL device test of BlockFieldStructured<D>
// (src/structured_smoother.hpp, Phase 1.3): the StructuredField adapter that
// composes the BlockField store with the BlockTopologyDevice halo tables and
// exposes exchange_halos(). Runs on EVERY visible device (GPU + CPU).
//
// Mirrors the standalone halo test, but drives the exchange through the adapter
// (and the StructuredField concept) instead of the free halo_exchange — proving
// the composition Phase 2 swaps behind works end to end: seed interiors, call
// field.exchange_halos(), and check each block's ghost shell carries the
// neighbour's first interior layer.
//
// Requires the acpp toolchain (SYCL).
#include "structured_smoother.hpp"
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
constexpr std::size_t kN0 = 4;
constexpr std::size_t kN1 = 3;

// A generic helper that only sees the StructuredField interface — it must
// compile and run for any model of the concept.
template <int D, class Field>
    requires StructuredField<Field, D>
void sync(Field& f)
{
    f.exchange_halos().wait();
}

void seed_interior(const BlockLayout<2>& layout, std::vector<egg::real>& host, std::size_t b,
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

void build_face_tables(std::vector<int>& src_block, std::vector<Index>& src_padded,
                       std::vector<int>& dst_block, std::vector<Index>& dst_padded)
{
    for (std::size_t j = 0; j < kN1; ++j) {
        dst_block.push_back(0);
        dst_padded.push_back({kN0 + 1, j + 1});
        src_block.push_back(1);
        src_padded.push_back({2, j + 1});

        dst_block.push_back(1);
        dst_padded.push_back({0, j + 1});
        src_block.push_back(0);
        src_padded.push_back({kN0 - 1, j + 1});
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

        test("BlockFieldStructured on " + name) = [&] {
            boost::ut::log << std::format("  device: {}\n", name);

            std::vector<int> src_block, dst_block;
            std::vector<Index> src_padded, dst_padded;
            build_face_tables(src_block, src_padded, dst_block, dst_padded);

            BlockField<2> field {q, layout};
            BlockTopologyDevice<2> topo {q,         layout,     src_block, src_padded,
                                         dst_block, dst_padded, {},        {}};
            BlockFieldStructured<2> sfield {q, std::move(field), std::move(topo)};

            expect(sfield.num_blocks() == 2_u);

            std::vector<egg::real> host(sfield.layout().total_doubles(), 0.0_r);
            seed_interior(layout, host, 0, 0.0);
            seed_interior(layout, host, 1, 3.0);
            sfield.upload(host);

            sync<2>(sfield);  // exchange_halos() via the StructuredField concept

            std::vector<egg::real> out;
            sfield.download(out);

            for (std::size_t j = 0; j < kN1; ++j) {
                const std::size_t g0 = layout.padded_node_offset(0, {kN0 + 1, j + 1});
                expect(out[g0 + 0] == 4.0)
                  << std::format("block0 ghost x at j={}: {}", j, out[g0 + 0]);
                const std::size_t g1 = layout.padded_node_offset(1, {0, j + 1});
                expect(out[g1 + 0] == 2.0)
                  << std::format("block1 ghost x at j={}: {}", j, out[g1 + 0]);
            }
        };
    }
}
