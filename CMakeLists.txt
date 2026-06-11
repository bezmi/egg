cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# AdaptiveCpp is required. SYCL sources are compiled through the AdaptiveCpp
# CMake integration (find_package(AdaptiveCpp) + add_sycl_to_target) using the
# normal host compiler — we do NOT set CMAKE_CXX_COMPILER=acpp. Select device
# target(s) with -DACPP_TARGETS (e.g. "generic" for a single CPU+GPU binary)
# and locate the install with -DAdaptiveCpp_DIR=<prefix>/lib/cmake/AdaptiveCpp
# (or CMAKE_PREFIX_PATH).
# ---------------------------------------------------------------------------

project(egg_cpp LANGUAGES CXX VERSION 0.1.0)

# ---------------------------------------------------------------------------
# AdaptiveCpp source pin (single source of truth, shared with the devcontainer
# Containerfile). The released acpp tags do not build against LLVM 21, so we pin
# a commit that does. Read it from ./ACPP_VERSION so there is exactly one place
# to bump it.
# ---------------------------------------------------------------------------
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/ACPP_VERSION" EGG_ACPP_VERSION
     LIMIT_COUNT 1)
string(STRIP "${EGG_ACPP_VERSION}" EGG_ACPP_VERSION)

# ---------------------------------------------------------------------------
# Opt-in superbuild: build AdaptiveCpp from source, then build egg against
# it. OFF by default — the normal path is find_package(AdaptiveCpp) against a
# pre-installed acpp (the devcontainer, or a system/brew install). Never enabled
# on the `uv pip install` path. See cmake/Superbuild.cmake.
# ---------------------------------------------------------------------------
option(EGG_BUILD_ADAPTIVECPP
       "Build AdaptiveCpp from source (pinned) as a superbuild" OFF)
option(EGG_SUPERBUILD_INNER "Internal: inner build of the superbuild" OFF)
set(EGG_ACPP_WITH_CUDA OFF CACHE BOOL "Superbuild: acpp CUDA backend")
set(EGG_ACPP_WITH_ROCM OFF CACHE BOOL "Superbuild: acpp ROCm/HIP backend")
set(EGG_ACPP_WITH_METAL OFF CACHE BOOL "Superbuild: acpp Metal backend (Apple)")
if(EGG_BUILD_ADAPTIVECPP AND NOT EGG_SUPERBUILD_INNER)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Superbuild.cmake)
  return()  # the inner ExternalProject does the real egg build
endif()

# ---------------------------------------------------------------------------
# C++23: std::print / std::format for logging, C++20 concepts, std::variant.
# No compiler extensions (portable, AdaptiveCpp-friendly).
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# ccache: transparently cache compilations if it is on PATH. Used as a compiler
# launcher so it also wraps the host-compile portion of AdaptiveCpp's SYCL
# targets (acpp attaches its own per-target launch rule on top).
# ---------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "ccache: ${CCACHE_PROGRAM}")
endif()

# ---------------------------------------------------------------------------
# mold: use it as the linker if present (CMAKE_LINKER_TYPE needs CMake >= 3.29).
# An explicit -DCMAKE_LINKER_TYPE on the command line takes precedence.
# ---------------------------------------------------------------------------
if(NOT CMAKE_LINKER_TYPE AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.29)
  find_program(MOLD_PROGRAM mold)
  if(MOLD_PROGRAM)
    set(CMAKE_LINKER_TYPE MOLD)
    message(STATUS "mold: ${MOLD_PROGRAM}")
  endif()
endif()

option(EGG_BUILD_TESTS "Build the C++ unit tests (Boost.UT)" ON)
option(EGG_BUILD_BENCH "Build the C++ micro-benchmarks (nanobench)" OFF)

# ---------------------------------------------------------------------------
# AdaptiveCpp CMake integration. The compute core uses explicit SYCL (not
# stdpar). find_package(AdaptiveCpp) requires ACPP_TARGETS to be set (via the
# cache/-D or the environment) before it is called; we default it to "omp" (CPU
# only) — pass -DACPP_TARGETS=generic for a single CPU+GPU binary, or e.g.
# "omp.accelerated;hip:gfx1030" for explicit multi-target. The ';'-escaping
# launcher hack the compiler-driver build needed is gone: add_sycl_to_target()
# attaches the toolchain per-target and consumes ACPP_TARGETS natively.
# ---------------------------------------------------------------------------
set(ACPP_TARGETS "generic" CACHE STRING "Semicolon-separated acpp targets")
# clang-scan-deps does not understand acpp's flags; disable C++20 module
# dependency scanning so it does not choke on them.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
find_package(AdaptiveCpp CONFIG REQUIRED)
message(STATUS "AdaptiveCpp: targets=${ACPP_TARGETS}")

# ---------------------------------------------------------------------------
# CPM.cmake dependency manager (vendored in ./cmake). Cache downloads under the
# source tree so reconfigures are offline-friendly.
# ---------------------------------------------------------------------------
set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm_cache"
    CACHE PATH "Directory CPM caches downloaded dependencies in")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

# ---------------------------------------------------------------------------
# External dependencies via CPM.
#   mdspan   — Kokkos reference mdspan (GPU-capable, unlike std::mdspan); used
#              for non-owning multidimensional views into device-resident USM.
#              Exports the INTERFACE target std::mdspan; we alias the namespace
#              as `stdex` in src/core.hpp (include <experimental/mdspan>).
#   pybind11 — Python binding layer.
#   Boost.UT — header-only unit-test framework (tests only).
# ---------------------------------------------------------------------------
# EXCLUDE_FROM_ALL so these deps' own install() rules never run — otherwise a
# `cmake --install` (the wheel step AND the editable.rebuild import hook, which
# is NOT component-scoped) would dump mdspan's headers + lib/cmake configs into
# site-packages root. They are header-only and already compiled into cpp_core,
# so they are dead weight. SYSTEM silences warnings from their headers.
# (install.components=["egg"] in pyproject.toml is a belt-and-suspenders
# backstop for the wheel; EXCLUDE_FROM_ALL is what also keeps editable installs
# clean, since the rebuild hook ignores install.components.)
CPMAddPackage(
  NAME mdspan
  GITHUB_REPOSITORY kokkos/mdspan
  GIT_TAG mdspan-0.6.0
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)

set(PYBIND11_FINDPYTHON ON)
CPMAddPackage(
  NAME pybind11
  GITHUB_REPOSITORY pybind/pybind11
  GIT_TAG v2.13.6
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
)

# ---------------------------------------------------------------------------
# Core compute library.
# ---------------------------------------------------------------------------
add_library(egg_core INTERFACE)
target_include_directories(egg_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(egg_core INTERFACE cxx_std_23)
target_link_libraries(egg_core INTERFACE std::mdspan)

# ---------------------------------------------------------------------------
# Python extension module (egg._cpp.cpp_core).
# ---------------------------------------------------------------------------
pybind11_add_module(cpp_core src/bindings.cpp)
target_link_libraries(cpp_core PRIVATE egg_core)
# Route bindings.cpp through the AdaptiveCpp toolchain (also links
# AdaptiveCpp::acpp-rt).
add_sycl_to_target(TARGET cpp_core SOURCES src/bindings.cpp)

# ---------------------------------------------------------------------------
# Where the extension lands, and how it finds the AdaptiveCpp runtime libs.
#
# Distribution (scikit-build-core drives this): install into the wheel under
# egg/_cpp and bake an rpath so libacpp-rt/libacpp-common resolve WITHOUT
# the user having to set LD_LIBRARY_PATH. The rpath token is platform-specific:
# $ORIGIN on Linux, @loader_path on macOS. We point it both at the directory
# next to the extension (for a future bundled/--acpp-deploy runtime) and at the
# AdaptiveCpp install's lib dir we built against.
#
# Editable dev workflow (-DEGG_INPLACE=ON): also drop the .so into the
# source tree at egg/_cpp so `uv run` picks it up without an install.
# ---------------------------------------------------------------------------
if(APPLE)
  set(_origin "@loader_path")
  set(CMAKE_MACOSX_RPATH ON)
else()
  set(_origin "$ORIGIN")
endif()

# AdaptiveCpp_DIR is <prefix>/lib/cmake/AdaptiveCpp -> the runtime libs are in
# <prefix>/lib. Add it to the install rpath so the bundled extension finds them.
get_filename_component(_acpp_libdir "${AdaptiveCpp_DIR}/../.." ABSOLUTE)
set_target_properties(cpp_core PROPERTIES
  INSTALL_RPATH "${_origin};${_origin}/acpp_runtime;${_acpp_libdir}"
  INSTALL_RPATH_USE_LINK_PATH ON
  BUILD_WITH_INSTALL_RPATH OFF)

option(EGG_INPLACE "Also write cpp_core into the source tree (editable dev)" OFF)
if(EGG_INPLACE)
  set_target_properties(cpp_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/egg/_cpp")
endif()

# scikit-build-core installs whatever lands here into the wheel. Tag it with a
# component so the wheel build can install ONLY this (see install.components in
# pyproject.toml) — otherwise `cmake --install` would also run the CPM deps'
# install rules (e.g. mdspan's headers + lib/cmake configs), dumping them into
# site-packages root. They are header-only and already compiled into cpp_core,
# so they are dead weight in the wheel.
install(TARGETS cpp_core LIBRARY DESTINATION egg/_cpp COMPONENT egg)

# ---------------------------------------------------------------------------
# Optional: vendor the AdaptiveCpp runtime into the wheel so it installs
# STANDALONE (no system acpp / no LD_LIBRARY_PATH needed). Uses acpp's own
# deployment mechanism (doc/deployment.md): `acpp --acpp-deploy=<component>:<dir>`
# populates <dir> with the runtime libs (acpp-rt/common, the generic-JIT LLVM
# infra + bitcode, and any requested backends). We deploy into the install tree
# at egg/_cpp/acpp_runtime, which the INSTALL_RPATH above already searches
# ($ORIGIN/acpp_runtime, @loader_path/acpp_runtime).
#
# Trade-offs (straight from deployment.md): pulls in libLLVM.so (~150 MB; `upx`
# ~halves it), the acpp runtime has NO stable ABI so the wheel is pinned to this
# exact acpp version, and HIP has no forward-compat (locked to its ROCm). So the
# result is a platform/arch-specific wheel for a controlled target — NOT a
# manylinux PyPI wheel. macOS standalone deploy is unverified (the deploy tool
# is Linux-centric).
#
# Enable for a wheel build with:
#   uv pip install . -C cmake.define.EGG_BUNDLE_RUNTIME=ON \
#       -C cmake.define.EGG_DEPLOY_COMPONENTS=core   # or "core;hip" / "all"
# ---------------------------------------------------------------------------
option(EGG_BUNDLE_RUNTIME
       "Vendor the AdaptiveCpp runtime into the install via acpp --acpp-deploy" OFF)
set(EGG_DEPLOY_COMPONENTS "core"
    CACHE STRING "acpp deploy components (core|cuda|hip|ocl|all; ';'-separated)")
if(EGG_BUNDLE_RUNTIME)
  # acpp lives in <prefix>/bin next to the <prefix>/lib we link against.
  find_program(ACPP_EXECUTABLE acpp HINTS "${_acpp_libdir}/../bin")
  if(NOT ACPP_EXECUTABLE)
    message(FATAL_ERROR
      "EGG_BUNDLE_RUNTIME=ON but `acpp` was not found near ${_acpp_libdir}. "
      "Pass -DACPP_EXECUTABLE=/path/to/acpp.")
  endif()
  message(STATUS "Bundling AdaptiveCpp runtime: ${EGG_DEPLOY_COMPONENTS} (acpp=${ACPP_EXECUTABLE})")
  # Run the deploy at install time, into the (staged) wheel prefix. Same
  # component as cpp_core so it is included under install.components=["egg"].
  install(CODE "
    set(_dest \"\${CMAKE_INSTALL_PREFIX}/egg/_cpp/acpp_runtime\")
    file(MAKE_DIRECTORY \"\${_dest}\")
    foreach(_comp ${EGG_DEPLOY_COMPONENTS})
      message(STATUS \"acpp --acpp-deploy=\${_comp}:\${_dest}\")
      execute_process(
        COMMAND \"${ACPP_EXECUTABLE}\" --acpp-deploy=\${_comp}:\"\${_dest}\"
        RESULT_VARIABLE _rc)
      if(_rc)
        message(FATAL_ERROR \"acpp --acpp-deploy=\${_comp} failed: \${_rc}\")
      endif()
    endforeach()
  " COMPONENT egg)
endif()

# ---------------------------------------------------------------------------
# C++ unit tests (Boost.UT), registered with CTest.
# ---------------------------------------------------------------------------
if(EGG_BUILD_TESTS)
  CPMAddPackage(
    NAME ut
    GITHUB_REPOSITORY boost-ext/ut
    GIT_TAG v2.3.1
    OPTIONS "BOOST_UT_BUILD_EXAMPLES OFF" "BOOST_UT_BUILD_TESTS OFF"
            "BOOST_UT_BUILD_BENCHMARKS OFF"
  )
  enable_testing()

  # Metric + solve parity tests build under the default compiler (no stdpar /
  # par_unseq backend needed). Golden reference values come from the C++
  # regression goldens via tests/cpp/golden_*.hpp (regenerate with gen_golden.py).
  add_executable(cpp_tests
    tests/cpp/test_main.cpp
    tests/cpp/test_metric.cpp
    tests/cpp/test_solve.cpp
    tests/cpp/test_geometry.cpp
    tests/cpp/test_patch.cpp)
  target_link_libraries(cpp_tests PRIVATE egg_core Boost::ut)
  target_compile_definitions(cpp_tests PRIVATE BOOST_UT_DISABLE_MODULE)
  target_include_directories(cpp_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
  add_test(NAME cpp_tests COMMAND cpp_tests)

  # The spike test runs the explicit-SYCL kernel (spike_launch) on the default
  # device, so it must be compiled by acpp (SYCL). No TBB needed (we no longer
  # use <execution>/par_unseq).
  add_executable(cpp_spike_tests tests/cpp/test_spike.cpp)
  target_link_libraries(cpp_spike_tests PRIVATE egg_core Boost::ut)
  target_compile_definitions(cpp_spike_tests PRIVATE BOOST_UT_DISABLE_MODULE)
  target_include_directories(cpp_spike_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
  add_sycl_to_target(TARGET cpp_spike_tests SOURCES tests/cpp/test_spike.cpp)
  add_test(NAME cpp_spike_tests COMMAND cpp_spike_tests)

  # Geometry + patch device tests run the closed-form project/tangent and the
  # per-DOF patch evaluation as SYCL kernels on EVERY visible device (the AMD
  # GPU and the OpenMP host/CPU device), against the C++ regression golden tables.
  foreach(_t geometry patch)
    add_executable(cpp_${_t}_tests tests/cpp/test_${_t}_device.cpp)
    target_link_libraries(cpp_${_t}_tests PRIVATE egg_core Boost::ut)
    target_compile_definitions(cpp_${_t}_tests PRIVATE BOOST_UT_DISABLE_MODULE)
    target_include_directories(cpp_${_t}_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
    add_sycl_to_target(TARGET cpp_${_t}_tests SOURCES tests/cpp/test_${_t}_device.cpp)
    add_test(NAME cpp_${_t}_tests COMMAND cpp_${_t}_tests)
  endforeach()

  # Sweep device test runs the full colored GS sweep on every visible device
  # (CPU + GPU) against the C++ golden sweep data (per-sweep energy, mindet,
  # and final X). Uses golden_sweep.hpp generated by gen_golden.py.
  add_executable(cpp_sweep_tests tests/cpp/test_sweep_device.cpp)
  target_link_libraries(cpp_sweep_tests PRIVATE egg_core Boost::ut)
  target_compile_definitions(cpp_sweep_tests PRIVATE BOOST_UT_DISABLE_MODULE)
  target_include_directories(cpp_sweep_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
  add_sycl_to_target(TARGET cpp_sweep_tests SOURCES tests/cpp/test_sweep_device.cpp)
  add_test(NAME cpp_sweep_tests COMMAND cpp_sweep_tests)
endif()

# ---------------------------------------------------------------------------
# Micro-benchmarks (nanobench). Deliberately NOT registered with CTest: timing
# builds stay -O3 -DNDEBUG and never run in the correctness suite. The benchmark
# uses SYCL directly, so it requires the acpp toolchain (build with
# -DACPP_TARGETS=generic, or omp;hip:gfxNNN, to include the GPU rows).
# ---------------------------------------------------------------------------
if(EGG_BUILD_BENCH)
  CPMAddPackage(
    NAME nanobench
    GITHUB_REPOSITORY martinus/nanobench
    GIT_TAG v4.3.11
  )
  # CPU-vs-GPU comparison at matched N (explicit per-device SYCL queues).
  add_executable(cpp_bench_compare bench/bench_compare.cpp)
  target_link_libraries(cpp_bench_compare PRIVATE egg_core nanobench)
  add_sycl_to_target(TARGET cpp_bench_compare SOURCES bench/bench_compare.cpp)

  # CPU-vs-GPU sweep benchmark: warm sweep-only ms/sweep of the colored GS
  # sweep on a fixed demo-grid context (golden data). Reports per device
  # (CPU omp, GPU resident). Uses golden_sweep.hpp from gen_golden.py.
  add_executable(cpp_bench_sweep bench/bench_sweep.cpp)
  target_link_libraries(cpp_bench_sweep PRIVATE egg_core nanobench)
  target_include_directories(cpp_bench_sweep PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
  add_sycl_to_target(TARGET cpp_bench_sweep SOURCES bench/bench_sweep.cpp)
endif()
