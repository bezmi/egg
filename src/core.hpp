#pragma once

#include <experimental/mdspan>

namespace egg
{

// for kokkos mdspan
namespace stdex = std::experimental;

inline constexpr int kVecT = 4;  // length of vec(T) in 2D

}  // namespace egg
