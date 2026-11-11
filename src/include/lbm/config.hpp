#pragma once

#include <cstddef>

namespace lbm {

using Real = double;
inline constexpr int Q = 9;

struct Config {
    int nx = 2;
    int ny = 2;
    int steps = 1000;
    Real tau = 1.0;
    Real tolerance = 1e-12;
    int disturbance_steps = 100;
    Real density_increase = 0.01; // Added to rho=1 at cell (0,0), with u=0.

    std::size_t cells() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    }
    std::size_t populations() const { return Q * cells(); }
    std::size_t bytes() const { return populations() * sizeof(Real); }
};

} // namespace lbm
