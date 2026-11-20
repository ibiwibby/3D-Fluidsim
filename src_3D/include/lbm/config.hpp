#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <initializer_list>
#include <stdexcept>
namespace lbm {
using Real = double;
inline constexpr int Q = 19;
struct Config {
    int nx = 32, ny = 32, nz = 32;
    int steps = 50;
    int output_every = 1;
    Real tau = 1.0;
    Real tolerance = 1e-12;
    bool channel = false;
    bool closed_ends = false;
    bool obstacle = true;
    Real obstacle_radius = 4.0;
    Real force_x = 1e-5; // Constant force per unit volume, active only in channel mode.
    int bump_width = 4;
    Real density_increase = 0.10;
    std::size_t cells() const { return std::size_t(nx) * ny * nz; }
    std::size_t populations() const { return Q * cells(); }
    std::size_t bytes() const { return populations() * sizeof(Real); }
    void validate() const {
        if(closed_ends&&!channel)throw std::invalid_argument("Closed ends require channel mode");
        if (nx < 1 || ny < 1 || nz < 1 || steps < 0 || output_every < 1
            || !std::isfinite(tau) || tau <= 0.5
            || !std::isfinite(tolerance) || tolerance <= 0
            || !std::isfinite(density_increase) || density_increase < 0
            || bump_width < 1 || bump_width > nx || bump_width > ny || bump_width > nz)
            throw std::invalid_argument("Invalid dimensions, steps, output interval, tau, tolerance or bump");
        if (channel && (nx < 6 || ny < 6 || nz < 6 || !std::isfinite(force_x)
            || !std::isfinite(obstacle_radius) || obstacle_radius <= 0
            || obstacle_radius >= (nx-1)*0.5 || obstacle_radius >= (ny-3)*0.5
            || obstacle_radius >= (nz-3)*0.5))
            throw std::invalid_argument("Invalid channel dimensions, force or obstacle radius");
        std::size_t size = nx;
        const auto limit = std::numeric_limits<std::size_t>::max() / (Q * sizeof(Real));
        for (int dimension : {ny, nz}) {
            if (size > limit / dimension) throw std::overflow_error("Grid too large");
            size *= dimension;
        }
        if ((size + 127) / 128 > 2147483647ULL) throw std::overflow_error("CUDA grid too large");
    }
};
}
