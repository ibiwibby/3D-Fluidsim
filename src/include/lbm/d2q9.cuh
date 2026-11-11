#pragma once

#include <cuda_runtime.h>
#include "lbm/config.hpp"

namespace lbm {

// Rest, E, N, W, S, NE, NW, SW, SE. Local constexpr arrays are usable
// from both host and device functions without separate device symbols.
__host__ __device__ inline int cx(int i) {
    constexpr int values[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    return values[i];
}

__host__ __device__ inline int cy(int i) {
    constexpr int values[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    return values[i];
}

__host__ __device__ inline Real weight(int i) {
    return i == 0 ? 4.0 / 9.0 : (i < 5 ? 1.0 / 9.0 : 1.0 / 36.0);
}

// Structure of arrays: all cells for one direction are contiguous.
__host__ __device__ inline std::size_t population_index(
    int direction, std::size_t cell, std::size_t cells) {
    return static_cast<std::size_t>(direction) * cells + cell;
}

} // namespace lbm
