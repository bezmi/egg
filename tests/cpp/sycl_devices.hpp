// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// sycl_devices.hpp — enumerate the SYCL devices a device test should cover.
//
// The device unit tests must exercise BOTH the GPU (HIP) and the CPU (the
// AdaptiveCpp OpenMP host device), so we run each kernel on every visible device
// that supports the f64 + device-USM the core needs, rather than the single
// default-selector queue. On this box that is {AMD gfx1030, OpenMP host}; on a
// CPU-only box it is just the host device (the GPU rows are simply skipped).
#pragma once

#include <sycl/sycl.hpp>
#include <vector>

namespace egg_test
{

// All visible SYCL devices supporting fp64 and device USM allocations.
inline std::vector<sycl::device> usable_devices()
{
    std::vector<sycl::device> out;
    for (const auto& plat : sycl::platform::get_platforms()) {
        for (const auto& dev : plat.get_devices()) {
            if (dev.has(sycl::aspect::fp64) && dev.has(sycl::aspect::usm_device_allocations)) {
                out.push_back(dev);
            }
        }
    }
    return out;
}

}  // namespace egg_test
