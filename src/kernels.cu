#include "lbm/kernels.cuh"
#include "lbm/d2q9.cuh"
#include "lbm/cuda_utils.cuh"

#include <cmath>
#include <cstdio>

namespace lbm {

namespace {

__global__ void initialize_rest_kernel(Real* current, std::size_t cells) {
    const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
    if (cell >= cells) return;

    // rho = 1 and u = (0, 0): each population equals its lattice weight.
    for (int i = 0; i < Q; ++i) {
        current[population_index(i, cell, cells)] = weight(i);
    }
}

// Fail the CUDA operation rather than letting invalid state silently propagate.
// The caller's synchronization reports this device failure through CUDA_CHECK.
__device__ void invalid_collision_state(std::size_t cell) {
    printf("Invalid collision state at cell %llu\n",
           static_cast<unsigned long long>(cell));
    asm volatile("trap;");
}

__global__ void collide_kernel(Real* current, std::size_t cells, Real omega) {
    const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
    if (cell >= cells) return;

    Real populations[Q];
    Real rho = 0.0;
    Real momentum_x = 0.0;
    Real momentum_y = 0.0;
    // Read the complete local state before any in-place writes.
    for (int i = 0; i < Q; ++i) {
        const Real f = current[population_index(i, cell, cells)];
        if (!isfinite(f)) {
            invalid_collision_state(cell);
            return;
        }
        populations[i] = f;
        rho += f;
        momentum_x += f * cx(i);
        momentum_y += f * cy(i);
    }
    if (!isfinite(rho) || rho <= 0.0) {
        invalid_collision_state(cell);
        return;
    }

    const Real ux = momentum_x / rho;
    const Real uy = momentum_y / rho;
    const Real speed_squared = ux * ux + uy * uy;
    if (!isfinite(speed_squared)) {
        invalid_collision_state(cell);
        return;
    }
    for (int i = 0; i < Q; ++i) {
        const Real cu = cx(i) * ux + cy(i) * uy;
        const Real equilibrium = weight(i) * rho
            * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared);
        populations[i] -= omega * (populations[i] - equilibrium);
        if (!isfinite(populations[i])) {
            invalid_collision_state(cell);
            return;
        }
    }
    // Each thread owns these nine entries; no atomics or barriers are needed.
    for (int i = 0; i < Q; ++i) {
        current[population_index(i, cell, cells)] = populations[i];
    }
}

__global__ void stream_periodic_kernel(const Real* current, Real* next,
                                       std::size_t cells, int nx, int ny) {
    const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                           + threadIdx.x;
    if (cell >= cells) return;

    const int x = static_cast<int>(cell % nx);
    const int y = static_cast<int>(cell / nx);
    for (int i = 0; i < Q; ++i) {
        // Pull from the upstream neighbor. Branch wrapping avoids negative
        // modulo results and overflow from adding a large grid dimension.
        int source_x = x - cx(i);
        int source_y = y - cy(i);
        if (source_x < 0) source_x = nx - 1;
        else if (source_x >= nx) source_x = 0;
        if (source_y < 0) source_y = ny - 1;
        else if (source_y >= ny) source_y = 0;
        const std::size_t source = static_cast<std::size_t>(source_y) * nx + source_x;
        next[population_index(i, cell, cells)] = current[population_index(i, source, cells)];
    }
}

} // namespace

void initialize_rest(Real* current, const Config& config, cudaStream_t stream) {
    if (current == nullptr || config.nx <= 0 || config.ny <= 0) {
        throw std::invalid_argument("Initialization requires a device buffer and positive dimensions");
    }
    constexpr unsigned int threads = 128;
    const std::size_t blocks = (config.cells() + threads - 1) / threads;
    // CUDA's one-dimensional grid limit; avoid narrowing an oversized launch.
    if (blocks > 2147483647ULL) {
        throw std::invalid_argument("Grid is too large for the initialization launch");
    }
    initialize_rest_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        current, config.cells());
    CUDA_CHECK(cudaGetLastError());
}

void collide(Real* current, const Config& config, cudaStream_t stream) {
    if (current == nullptr || config.nx <= 0 || config.ny <= 0
        || !std::isfinite(config.tau) || config.tau <= 0.5) {
        throw std::invalid_argument(
            "Collision requires a device buffer, positive dimensions, and finite tau > 0.5");
    }
    constexpr unsigned int threads = 128;
    const std::size_t blocks = (config.cells() + threads - 1) / threads;
    if (blocks > 2147483647ULL) {
        throw std::invalid_argument("Grid is too large for the collision launch");
    }
    collide_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        current, config.cells(), 1.0 / config.tau);
    CUDA_CHECK(cudaGetLastError());
}

void stream_periodic(const Real* current, Real* next, const Config& config,
                     cudaStream_t stream) {
    if (current == nullptr || next == nullptr || current == next
        || config.nx <= 0 || config.ny <= 0) {
        throw std::invalid_argument(
            "Streaming requires distinct device buffers and positive dimensions");
    }
    constexpr unsigned int threads = 128;
    const std::size_t blocks = (config.cells() + threads - 1) / threads;
    if (blocks > 2147483647ULL) {
        throw std::invalid_argument("Grid is too large for the streaming launch");
    }
    stream_periodic_kernel<<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        current, next, config.cells(), config.nx, config.ny);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace lbm
