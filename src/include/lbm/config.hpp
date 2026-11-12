#pragma once

#include <cstddef>
#include <array>

namespace lbm {

using Real = double;
inline constexpr int Q = 9;

struct Config {
    int nx = 64;
    int ny = 64;
    int steps = 1000;
    Real tau = 1.0;
    Real tolerance = 1e-12;
    int disturbance_steps = 50;
    std::array<std::array<int, 2>, 16> disturbance_centers = {{
        {{30, 30}}, {{31, 30}}, {{32, 30}}, {{33, 30}},
        {{30, 31}}, {{31, 31}}, {{32, 31}}, {{33, 31}},
        {{30, 32}}, {{31, 32}}, {{32, 32}}, {{33, 32}},
        {{30, 33}}, {{31, 33}}, {{32, 33}}, {{33, 33}}
    }};
    Real density_increase = 4.0; // rho=5 in the selected 4x4 square; rho=1 elsewhere.

    std::size_t cells() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    }
    std::size_t populations() const { return Q * cells(); }
    std::size_t bytes() const { return populations() * sizeof(Real); }
};

} // namespace lbm
