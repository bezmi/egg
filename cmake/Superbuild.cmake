# ---------------------------------------------------------------------------
# Superbuild: build AdaptiveCpp from source, then build egg against it.
#
# This file is included from the top-level CMakeLists.txt ONLY when
#   -DEGG_BUILD_ADAPTIVECPP=ON
# is passed *and* we are not already inside the inner egg build
# (EGG_SUPERBUILD_INNER). It turns the configure into a two-project driver:
#
#   1. ExternalProject "AdaptiveCpp" — clone + build + install acpp into the
#      build tree (NOT system-wide), pinned to ${EGG_ACPP_VERSION}, against
#      a pre-existing LLVM (we do not build LLVM here).
#   2. ExternalProject "egg"     — re-invoke this same source tree with
#      EGG_SUPERBUILD_INNER=ON and AdaptiveCpp_DIR pointed at the install
#      from step 1, so the normal find_package(AdaptiveCpp) path is taken.
#
# This is an explicit, opt-in DEVELOPER convenience — it is heavy (builds a
# clang-based SYCL toolchain) and is deliberately never on the `uv pip install`
# path. acpp must match its LLVM: the pin in ./ACPP_VERSION exists because the
# released tags do not compile against LLVM 21 (see .devcontainer/Containerfile,
# which builds the same pin the same way). The devcontainer remains the
# recommended way to get a known-good acpp because it also pins the LLVM version.
# ---------------------------------------------------------------------------
include(ExternalProject)

# LLVM/Clang to build acpp against. On Linux we expect the distro -dev packages
# (the devcontainer uses LLVM 21 under /usr/lib/llvm-21); on macOS, Homebrew LLVM
# (acpp does NOT work with Apple Clang — no full LLVM). Callers may override
# -DLLVM_DIR / -DClang_DIR explicitly.
if(APPLE AND NOT DEFINED LLVM_DIR)
  message(WARNING
    "Superbuild on macOS: pass -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm "
    "(and -DClang_DIR=.../lib/cmake/clang). Only the CPU/OpenMP backend is "
    "supported; the generic JIT compiler may require AdaptiveCpp's 2-stage "
    "LLVM build (doc/installing.md) which this superbuild does NOT perform.")
endif()

set(_acpp_prefix "${CMAKE_BINARY_DIR}/adaptivecpp")
set(_acpp_install "${_acpp_prefix}/install")

# GPU backends are opt-in and forwarded straight through, mirroring the
# Containerfile build args. Off by default => CPU/OpenMP + generic SSCP only.
set(_acpp_cmake_args
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=${_acpp_install}
  -DWITH_SSCP_COMPILER=ON
  -DWITH_CPU_BACKEND=ON
  -DWITH_CUDA_BACKEND=${EGG_ACPP_WITH_CUDA}
  -DWITH_ROCM_BACKEND=${EGG_ACPP_WITH_ROCM}
  -DWITH_METAL_BACKEND=${EGG_ACPP_WITH_METAL})  # Apple GPUs (experimental)
# WITH_ACCELERATED_CPU (the OpenMP "library-only" fast CPU path) is a Linux
# feature; leave it off on Apple where it is unsupported.
if(NOT APPLE)
  list(APPEND _acpp_cmake_args -DWITH_ACCELERATED_CPU=ON)
endif()
# Pass-through for the backend-specific locations a developer may need to set
# (names per AdaptiveCpp's doc/install-*.md):
#   LLVM_DIR               — the LLVM acpp builds against (must match acpp);
#                            acpp derives clang from this. Override clang only
#                            with CLANG_EXECUTABLE_PATH / CLANG_INCLUDE_PATH.
#   ROCM_PATH              — ROCm prefix for the HIP backend (default /opt/rocm)
#   CUDA_TOOLKIT_ROOT_DIR  — CUDA toolkit root for the CUDA backend
#   METAL_INCLUDE_DIR      — Apple's metal-cpp headers for the Metal backend
foreach(_v LLVM_DIR CLANG_EXECUTABLE_PATH CLANG_INCLUDE_PATH
           ROCM_PATH CUDA_TOOLKIT_ROOT_DIR METAL_INCLUDE_DIR
           CMAKE_C_COMPILER CMAKE_CXX_COMPILER)
  if(DEFINED ${_v})
    list(APPEND _acpp_cmake_args -D${_v}=${${_v}})
  endif()
endforeach()

ExternalProject_Add(AdaptiveCpp
  GIT_REPOSITORY https://github.com/AdaptiveCpp/AdaptiveCpp.git
  GIT_TAG        ${EGG_ACPP_VERSION}
  GIT_SHALLOW    OFF                     # tag is a bare commit SHA; no shallow
  PREFIX         ${_acpp_prefix}
  CMAKE_ARGS     ${_acpp_cmake_args}
  BUILD_ALWAYS   OFF
  USES_TERMINAL_DOWNLOAD ON
  USES_TERMINAL_BUILD    ON
  USES_TERMINAL_INSTALL  ON)

# Inner egg build: same sources, but with acpp located and the superbuild
# guard tripped so it takes the normal find_package path. Forward the knobs a
# developer is likely to set.
set(_inner_args
  -DEGG_SUPERBUILD_INNER=ON
  -DAdaptiveCpp_DIR=${_acpp_install}/lib/cmake/AdaptiveCpp
  -DACPP_TARGETS=${ACPP_TARGETS}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DEGG_BUILD_TESTS=${EGG_BUILD_TESTS}
  -DEGG_BUILD_BENCH=${EGG_BUILD_BENCH})
foreach(_v CMAKE_PREFIX_PATH Python_EXECUTABLE Python3_EXECUTABLE
           CMAKE_INSTALL_PREFIX)
  if(DEFINED ${_v})
    list(APPEND _inner_args -D${_v}=${${_v}})
  endif()
endforeach()

ExternalProject_Add(egg
  SOURCE_DIR   ${CMAKE_SOURCE_DIR}
  BINARY_DIR   ${CMAKE_BINARY_DIR}/egg-build
  DEPENDS      AdaptiveCpp
  CMAKE_ARGS   ${_inner_args}
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""                     # inner build installs on its own terms
  USES_TERMINAL_CONFIGURE ON
  USES_TERMINAL_BUILD     ON)
