#pragma once

#include "lbm/config.hpp"

namespace lbm {

// Sharp initial condition: only the selected cells have elevated density.
// Repeated coordinates do not add the disturbance more than once.
inline Real initial_density(int x, int y, const Config& config) {
    for (const auto& center : config.disturbance_centers) {
        if (x == center[0] && y == center[1]) {
            return 1.0 + config.density_increase;
        }
    }
    return 1.0;
}

} // namespace lbm
