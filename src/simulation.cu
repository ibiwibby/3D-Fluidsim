#include "lbm/simulation.cuh"
#include "lbm/cuda_utils.cuh"
#include "lbm/kernels.cuh"

#include <cmath>
#include <limits>
#include <utility>

namespace lbm {

Simulation::Simulation(Config config) : config_(config) {
    if (config.nx <= 0 || config.ny <= 0 || config.steps < 0
        || !std::isfinite(config.tau) || config.tau <= 0.5) {
        throw std::invalid_argument("Require nx, ny > 0; steps >= 0; finite tau > 0.5");
    }
    const auto max_cells = std::numeric_limits<std::size_t>::max() / (Q * sizeof(Real));
    if (static_cast<std::size_t>(config.nx) > max_cells / config.ny) {
        throw std::invalid_argument("Grid allocation size overflows size_t");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&current_), config_.bytes()));
    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&next_), config_.bytes()));
    } catch (...) {
        cudaFree(current_);
        throw;
    }
}

Simulation::~Simulation() {
    // Destructors must not throw, including during exception cleanup.
    cudaFree(next_);
    cudaFree(current_);
}

void Simulation::initialize() {
    initialized_ = false;
    initialize_rest(current_, config_);
    CUDA_CHECK(cudaDeviceSynchronize());
    initialized_ = true;
}

void Simulation::upload(const std::vector<Real>& populations) {
    if (populations.size() != config_.populations()) {
        throw std::invalid_argument("Upload must contain exactly 9 populations per cell");
    }
    initialized_ = false;
    CUDA_CHECK(cudaMemcpy(current_, populations.data(), config_.bytes(), cudaMemcpyHostToDevice));
    initialized_ = true;
}

void Simulation::collide_only() {
    if (!initialized_) throw std::logic_error("Initialize or upload before collision");
    initialized_ = false;
    collide(current_, config_);
    CUDA_CHECK(cudaDeviceSynchronize()); // Deliberate checkpoint debugging boundary.
    initialized_ = true;
}

void Simulation::stream_only() {
    if (!initialized_) throw std::logic_error("Initialize or upload before streaming");
    initialized_ = false;
    stream_periodic(current_, next_, config_);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(current_, next_);
    initialized_ = true;
}

void Simulation::step() {
    collide_only();
    stream_only();
}

std::vector<Real> Simulation::download() const {
    if (!initialized_) throw std::logic_error("Initialize before downloading");
    std::vector<Real> result(config_.populations());
    CUDA_CHECK(cudaMemcpy(result.data(), current_, config_.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

} // namespace lbm
