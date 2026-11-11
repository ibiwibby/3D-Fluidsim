#pragma once

#include <cstddef>

namespace lbm {

using Real = double;
inline constexpr int Q = 9;

struct Config {
    int nx = 16;
    int ny = 16;
    int steps = 1000;
    Real tau = 1.0;
    Real tolerance = 1e-12;
    int disturbance_steps = 100;
    int disturbance_x = 8;
    int disturbance_y = 8;
    Real density_increase = 0.10; // Added to rho=1 at the disturbance cell, with u=0.

    std::size_t cells() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    }
    std::size_t populations() const { return Q * cells(); }
    std::size_t bytes() const { return populations() * sizeof(Real); }
};

} // namespace lbm
