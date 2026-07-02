// test_field_device.cpp — SYCL device test of BlockField<D>
// (src/structured_field.hpp): the UsmBuffer-owning halo-padded coordinate store and its
// interior(b) / with_halo(b) mdspan views. Runs on EVERY visible device (the
// AMD GPU and the OpenMP host/CPU device).
//
// Strategy: fill the packed buffer so buf[off] == off, then in a kernel read
// every interior node through BOTH views and assert (a) the value equals the
// offset BlockLayout predicts and (b) interior[i,…] aliases halo[i+1,…] (same
// storage). A second kernel writes a marker through the interior view only; we
// download and assert the ghost layer is untouched and the interior is marked —
// proving the views address exactly the strided slots BlockLayout describes,
// on device, and that writes go through.
//
// Requires the acpp toolchain (SYCL).
#include "structured_field.hpp"
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
// Two 2D blocks: interior (3,4) and (2,2) -> padded (5,6) and (4,4). Matches the
// host BlockLayout tests so the hand-computed numbers carry over.
constexpr std::size_t kNB = 2;

// Total interior doubles across both blocks: (3*4 + 2*2) * 2 = 32.
constexpr std::size_t kInteriorDoubles = (3 * 4 + 2 * 2) * 2;

// Run the read + write probes on one device. `read_int`/`read_halo` come back
// with the per-interior-node values seen through each view (packed block-major);
// `buf_after` is the whole padded buffer after the marker write.
void run_field(sycl::queue& q,
               std::vector<egg::real>& read_int,
               std::vector<egg::real>& read_halo,
               std::vector<egg::real>& buf_after)
{
    BlockField<2> field {q, BlockLayout<2> {{{{3, 4}}, {{2, 2}}}}};

    // Seed every double with its own offset so a view read is self-describing.
    std::vector<egg::real> host(field.size());
    for (std::size_t i = 0; i < host.size(); ++i) { host[i] = static_cast<egg::real>(i); }
    field.upload(host);

    // Views are non-owning + trivially copyable -> capture by value.
    std::array<InteriorView<2>, kNB> ivs {field.interior(0), field.interior(1)};
    std::array<HaloView<2>, kNB> hvs {field.with_halo(0), field.with_halo(1)};

    auto* d_int = sycl::malloc_device<egg::real>(kInteriorDoubles, q);
    auto* d_halo = sycl::malloc_device<egg::real>(kInteriorDoubles, q);

    // One serial task walks both blocks; this is a correctness probe of the view
    // index math, not a performance kernel.
    q.single_task([=]() {
         std::size_t pos = 0;
         for (std::size_t b = 0; b < kNB; ++b) {
             const InteriorView<2> iv = ivs[b];
             const HaloView<2> hv = hvs[b];
             for (std::size_t i = 0; i < iv.extent(0); ++i) {
                 for (std::size_t j = 0; j < iv.extent(1); ++j) {
                     for (std::size_t k = 0; k < 2; ++k) {
                         d_int[pos] = iv[i, j, k];
                         d_halo[pos] = hv[i + 1, j + 1, k];  // interior aliases padded+1
                         ++pos;
                     }
                 }
             }
         }
     }).wait();

    read_int.resize(kInteriorDoubles);
    read_halo.resize(kInteriorDoubles);
    q.memcpy(read_int.data(), d_int, kInteriorDoubles * sizeof(egg::real));
    q.memcpy(read_halo.data(), d_halo, kInteriorDoubles * sizeof(egg::real));
    q.wait();

    // Write a marker through the interior view only, then read the whole buffer
    // back to confirm ghosts are untouched and interior slots changed.
    q.single_task([=]() {
         for (std::size_t b = 0; b < kNB; ++b) {
             const InteriorView<2> iv = ivs[b];
             for (std::size_t i = 0; i < iv.extent(0); ++i) {
                 for (std::size_t j = 0; j < iv.extent(1); ++j) {
                     for (std::size_t k = 0; k < 2; ++k) { iv[i, j, k] = iv[i, j, k] + 1000.0; }
                 }
             }
         }
     }).wait();

    field.download(buf_after);

    sycl::free(d_int, q);
    sycl::free(d_halo, q);
}
}  // namespace

int main()
{
    const auto devices = usable_devices();
    expect(!devices.empty()) << "no fp64/USM SYCL device found";

    // Reference layout on the host to predict offsets the device should reproduce.
    const BlockLayout<2> layout {{{{3, 4}}, {{2, 2}}}};

    for (const auto& dev : devices) {
        sycl::queue q(dev);
        const std::string name = dev.get_info<sycl::info::device::name>();

        test("BlockField on " + name) = [&] {
            boost::ut::log << std::format("  device: {}\n", name);
            std::vector<egg::real> read_int, read_halo, buf_after;
            run_field(q, read_int, read_halo, buf_after);

            // Mark which buffer slots the interior view legitimately wrote to, so
            // we can assert every *other* slot (ghost layer) was left untouched.
            std::vector<char> is_interior(layout.total_doubles(), 0);

            std::size_t pos = 0;
            for (std::size_t b = 0; b < layout.num_blocks(); ++b) {
                const auto interior = layout.interior_shape(b);
                for (std::size_t i = 0; i < interior[0]; ++i) {
                    for (std::size_t j = 0; j < interior[1]; ++j) {
                        const std::size_t off = layout.interior_node_offset(b, {i, j});
                        for (std::size_t k = 0; k < 2; ++k) {
                            const egg::real expected = static_cast<egg::real>(off + k);
                            expect(read_int[pos] == expected) << std::format(
                              "block {} interior[{},{},{}]: {} vs {}", b, i, j, k, read_int[pos],
                              expected);
                            expect(read_halo[pos] == expected) << std::format(
                              "block {} halo alias [{},{},{}]: {} vs {}", b, i, j, k,
                              read_halo[pos], expected);
                            is_interior[off + k] = 1;
                            // Interior slot got the +1000 marker.
                            expect(buf_after[off + k] == expected + 1000.0) << std::format(
                              "block {} marker[{},{},{}]: {}", b, i, j, k, buf_after[off + k]);
                            ++pos;
                        }
                    }
                }
            }

            // Ghost slots (everything not flagged interior) keep their seed value.
            for (std::size_t off = 0; off < layout.total_doubles(); ++off) {
                if (!is_interior[off]) {
                    expect(buf_after[off] == static_cast<egg::real>(off))
                      << std::format("ghost slot {} was modified: {}", off, buf_after[off]);
                }
            }
        };
    }
}
