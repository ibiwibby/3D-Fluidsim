#include "lbm/disturbance.hpp"
#include "lbm/simulation.cuh"
#include "lbm/d2q9.cuh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <vector>

namespace lbm {
namespace {

struct CellFields {
    Real density;
    Real ux;
    Real uy;
};

// Validate a complete snapshot before writing any of its rows.
void write_snapshot(std::ofstream& output, const std::vector<Real>& populations,
                    const Config& config, int step, Real expected_mass) {
    if (populations.size() != config.populations()) {
        throw std::invalid_argument("Snapshot population count does not match the grid");
    }
    std::vector<CellFields> fields(config.cells());
    Real mass = 0.0;
    for (std::size_t cell = 0; cell < config.cells(); ++cell) {
        Real rho = 0.0, mx = 0.0, my = 0.0;
        for (int i = 0; i < Q; ++i) {
            const Real f = populations[population_index(i, cell, config.cells())];
            if (!std::isfinite(f)) {
                throw std::runtime_error("Nonfinite population at step " + std::to_string(step)
                    + ", cell " + std::to_string(cell));
            }
            rho += f;
            mx += f * cx(i);
            my += f * cy(i);
        }
        if (!std::isfinite(rho) || rho <= 0.0
            || !std::isfinite(mx / rho) || !std::isfinite(my / rho)) {
            throw std::runtime_error("Invalid fluid fields at step " + std::to_string(step)
                + ", cell " + std::to_string(cell));
        }
        fields[cell] = {rho, mx / rho, my / rho};
        mass += rho;
    }
    const Real relative_error = std::abs(mass - expected_mass) / expected_mass;
    if (!std::isfinite(mass) || relative_error > config.tolerance) {
        throw std::runtime_error("Mass check failed at step " + std::to_string(step)
            + "; expected mass=" + std::to_string(expected_mass));
    }
    for (std::size_t cell = 0; cell < config.cells(); ++cell) {
        const auto& field = fields[cell];
        output << step << ',' << cell % config.nx << ',' << cell / config.nx << ','
               << field.density << ',' << field.ux << ',' << field.uy << '\n';
    }
    output.flush(); // Preserve completed snapshots if a later step fails.
    if (step == 0 || step == config.disturbance_steps || step % 10 == 0) {
        std::cout << std::setprecision(15) << "Snapshot " << step << ": mass=" << mass
                  << " relative_mass_error=" << relative_error << '\n';
    }
}

} // namespace

void run_disturbance(Config config, const std::string& output_path) {
    if (config.disturbance_steps < 0 || !std::isfinite(config.density_increase)
        || config.density_increase <= 0.0 || !std::isfinite(config.tolerance)
        || config.tolerance < 0.0) {
        throw std::invalid_argument("Require nonnegative disturbance steps/tolerance and a finite positive density increase");
    }
    config.steps = config.disturbance_steps;
    if (config.disturbance_x < 0 || config.disturbance_x >= config.nx
        || config.disturbance_y < 0 || config.disturbance_y >= config.ny) {
        throw std::invalid_argument("Disturbance coordinates must lie inside the grid");
    }
    const std::size_t disturbance_cell =
        static_cast<std::size_t>(config.disturbance_y) * config.nx + config.disturbance_x;
    Simulation simulation(config);
    simulation.initialize();
    auto initial = simulation.download();
    // Prescribed initial condition only; subsequent fluid updates remain on GPU.
    // At zero velocity, equilibrium is w_i * rho for every direction.
    for (int i = 0; i < Q; ++i) {
        initial[population_index(i, disturbance_cell, config.cells())] =
            weight(i) * (1.0 + config.density_increase);
    }
    simulation.upload(initial);
    const Real expected_mass = static_cast<Real>(config.cells()) + config.density_increase;

    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.imbue(std::locale::classic());
    output.open(output_path, std::ios::out | std::ios::trunc);
    output << std::setprecision(std::numeric_limits<Real>::max_digits10)
           << "step,x,y,density,ux,uy\n";
    std::cout << "Density disturbance: rho(" << config.disturbance_x << ','
              << config.disturbance_y << ")=" << 1.0 + config.density_increase
              << ", steps=" << config.disturbance_steps << ", output=" << output_path << '\n';
    write_snapshot(output, simulation.download(), config, 0, expected_mass);
    for (int step = 0; step < config.disturbance_steps; ++step) {
        simulation.step();
        write_snapshot(output, simulation.download(), config, step + 1, expected_mass);
    }
    output.close(); // Report any final file errors before announcing success.
    std::cout << "Saved " << static_cast<long long>(config.disturbance_steps) + 1
              << " snapshots; all finite-state and mass checks passed.\n";
}

} // namespace lbm
