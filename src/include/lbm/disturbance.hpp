#pragma once

#include "lbm/config.hpp"
#include <string>

namespace lbm {

// Run the density-disturbance demonstration and write all snapshots to one CSV.
// Throws on invalid state, mass drift, or output failure.
void run_disturbance(Config config, const std::string& output_path);

} // namespace lbm
