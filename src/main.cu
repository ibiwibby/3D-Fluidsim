#include "lbm/simulation.cuh"
#include "lbm/cuda_utils.cuh"
#include "lbm/d2q9.cuh"
#include "lbm/disturbance.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace {

bool check_rest_state(const std::vector<lbm::Real>& populations,
                      const lbm::Config& config, int step) {
    if (populations.size() != config.populations()) {
        throw std::invalid_argument("Rest-state check received an incorrect population count");
    }
    if (!std::isfinite(config.tolerance) || config.tolerance < 0.0) {
        throw std::invalid_argument("Rest-state tolerance must be finite and nonnegative");
    }
    lbm::Real max_population_error = 0.0, max_density_error = 0.0;
    lbm::Real max_ux = 0.0, max_uy = 0.0, total_mass = 0.0;
    bool valid = true;
    // Map NaN/Inf to infinite error, so max() cannot conceal invalid values.
    const auto error = [](lbm::Real value) {
        return std::isfinite(value) ? std::abs(value)
                                   : std::numeric_limits<lbm::Real>::infinity();
    };
    for (std::size_t cell = 0; cell < config.cells(); ++cell) {
        lbm::Real rho = 0.0, mx = 0.0, my = 0.0;
        for (int i = 0; i < lbm::Q; ++i) {
            const auto f = populations[lbm::population_index(i, cell, config.cells())];
            max_population_error = std::max(max_population_error, error(f - lbm::weight(i)));
            valid = valid && std::isfinite(f);
            rho += f;
            mx += f * lbm::cx(i);
            my += f * lbm::cy(i);
        }
        max_density_error = std::max(max_density_error, error(rho - 1.0));
        if (!std::isfinite(rho) || rho <= 0.0) {
            valid = false;
            max_ux = max_uy = std::numeric_limits<lbm::Real>::infinity();
        } else {
            max_ux = std::max(max_ux, error(mx / rho));
            max_uy = std::max(max_uy, error(my / rho));
        }
        total_mass += rho;
    }
    const auto expected_mass = static_cast<lbm::Real>(config.cells());
    const auto mass_error = error(total_mass - expected_mass) / expected_mass;
    const bool passed = valid && max_population_error <= config.tolerance
        && max_density_error <= config.tolerance && max_ux <= config.tolerance
        && max_uy <= config.tolerance && mass_error <= config.tolerance;
    std::cout << std::setprecision(15)
              << "Rest-state checkpoint step " << step << ": " << (passed ? "PASS" : "FAIL")
              << "\n  max_population_error=" << max_population_error
              << " max_density_error=" << max_density_error
              << " max_abs_ux=" << max_ux << " max_abs_uy=" << max_uy
              << "\n  mass=" << total_mass << " expected_mass=" << expected_mass
              << " relative_mass_error=" << mass_error
              << " tolerance=" << config.tolerance << '\n';
    if (!passed) {
        std::cerr << "Stopping at failed rest-state checkpoint " << step << '\n';
    }
    return passed;
}

int check_streaming() {
    // Fixed, hand-calculated 2x2 oracle, independent of kernel routing logic.
    lbm::Config config;
    config.nx = config.ny = 2;
    constexpr int source_cells[lbm::Q][4] = {
        {0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1},
        {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0},
        {3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}
    };
    std::vector<lbm::Real> labels(config.populations());
    for (int i = 0; i < lbm::Q; ++i) {
        for (std::size_t cell = 0; cell < config.cells(); ++cell) {
            labels[lbm::population_index(i, cell, config.cells())] = 100 * i + cell + 1;
        }
    }
    lbm::Simulation simulation(config);
    simulation.upload(labels); // Diagnostic labels must never go through collision.
    bool passed = true;
    std::cout << "Streaming-only check: fixed 2x2 periodic grid, no collision\n"
              << "step cell direction expected actual\n";
    for (int step = 1; step <= 2; ++step) {
        simulation.stream_only();
        const auto actual = simulation.download();
        for (int i = 0; i < lbm::Q; ++i) {
            for (std::size_t cell = 0; cell < config.cells(); ++cell) {
                const auto index = lbm::population_index(i, cell, config.cells());
                const auto expected = step == 1 ? 100 * i + source_cells[i][cell] + 1
                                                : labels[index];
                if (actual[index] != expected) {
                    passed = false;
                    std::cerr << "Streaming mismatch at step " << step << ", cell " << cell
                              << ", direction " << i << '\n';
                }
                std::cout << step << ' ' << cell << ' ' << i << ' '
                          << expected << ' ' << actual[index] << '\n';
            }
        }
    }
    // Every expected label is unique, so exact agreement also checks that
    // no population was lost or duplicated. Step 2 checks the second swap.
    std::cout << "Streaming check: " << (passed ? "PASS" : "FAIL") << '\n'
              << "2x2 cannot distinguish opposite spatial directions; test 3x3 next.\n";
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "--help";
        if (argc > (mode == "--disturbance" ? 3 : 2)
            || (mode != "--help" && mode != "--device-info" && mode != "--run"
                         && mode != "--collision-check" && mode != "--streaming-check"
                         && mode != "--simulate" && mode != "--disturbance")) {
            std::cerr << "Usage: " << argv[0]
                      << " [--help|--device-info|--run|--collision-check|--streaming-check|--simulate]\n"
                      << "       " << argv[0] << " --disturbance [output.csv]\n";
            return 2;
        }
        if (mode == "--help") {
            std::cout << "D2Q9 CUDA checkpoint\n"
                      << "  --device-info  Check CUDA access on an allocated GPU\n"
                      << "  --run          Initialize, read back, and check the rest state\n"
                      << "  --collision-check  Perturb cell 0 and check one collision at tau=1\n"
                      << "  --streaming-check  Check labels after one and two streams on 2x2\n"
                      << "  --simulate     Run from rest; check steps 0, 1, 2, 10, 1000 and final\n"
                      << "  --disturbance [output.csv]  Evolve density bump; save every step\n"
                      << "Disturbance defaults: 100 steps, cell (8,8), density increase 0.10, disturbance.csv.\n"
                      << "Defaults: 16x16 grid, double precision, 1000 simulation timesteps.\n"
                      << "Edit include/lbm/config.hpp to change defaults.\n";
            return 0;
        }

        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count == 0) throw std::runtime_error("No CUDA GPU visible");
        CUDA_CHECK(cudaSetDevice(0)); // First GPU visible to this job.
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::cout << "GPU: " << properties.name << " (compute capability "
                  << properties.major << '.' << properties.minor << ")\n";
        if (mode == "--device-info") return 0;
        if (mode == "--streaming-check") return check_streaming();
        if (mode == "--disturbance") {
            lbm::run_disturbance(lbm::Config{}, argc == 3 ? argv[2] : "disturbance.csv");
            return 0;
        }

        const bool collision_check = mode == "--collision-check";
        const bool simulate = mode == "--simulate";
        lbm::Config config;
        // The one-collision return-to-equilibrium expectation requires tau=1.
        if (collision_check) config.tau = 1.0;
        lbm::Simulation simulation(config);
        simulation.initialize();
        std::vector<lbm::Real> perturbed;
        if (collision_check) {
            perturbed = simulation.download();
            constexpr lbm::Real delta = 0.01;
            perturbed[lbm::population_index(1, 0, config.cells())] += delta; // East
            perturbed[lbm::population_index(3, 0, config.cells())] += delta; // West
            perturbed[lbm::population_index(0, 0, config.cells())] -= 2.0 * delta;
            simulation.upload(perturbed);
            simulation.collide_only();
        }
        if (simulate) {
            std::cout << "Simulating " << config.steps << " timesteps on "
                      << config.nx << 'x' << config.ny << " from rest, tau="
                      << config.tau << '\n';
            if (!check_rest_state(simulation.download(), config, 0)) return 1;
            // Read back only at checkpoints. Each full step includes a swap.
            for (int step = 0; step < config.steps; ++step) {
                simulation.step();
                const int completed = step + 1;
                if (completed == 1 || completed == 2 || completed == 10
                    || completed == 1000 || completed == config.steps) {
                    if (!check_rest_state(simulation.download(), config, completed)) return 1;
                }
            }
            // The final step is checked once, including when it is step zero.
            std::cout << "Completed " << config.steps << " timesteps; all rest-state checkpoints passed.\n";
            return 0;
        }
        const auto populations = simulation.download();
        std::cout << std::setprecision(15);
        if (collision_check) {
            std::cout << "Collision-only check: cell 0, delta=0.01, tau=1; no streaming\n"
                      << "direction before after expected\n";
            for (int i = 0; i < lbm::Q; ++i) {
                const auto index = lbm::population_index(i, 0, config.cells());
                std::cout << i << ' ' << perturbed[index] << ' ' << populations[index]
                          << ' ' << lbm::weight(i) << '\n';
            }
        }
        // All physical modes here should end at the uniform rest equilibrium.
        std::cout << (collision_check ? "After one collision, " : "Initial state (step 0), ");
        std::cout << config.nx << 'x' << config.ny
                  << ", " << populations.size() << " populations\n"
                  << std::setprecision(15) << "cell x y density ux uy\n";
        bool passed = true;
        lbm::Real total_mass = 0.0;
        for (std::size_t cell = 0; cell < config.cells(); ++cell) {
            lbm::Real rho = 0.0, momentum_x = 0.0, momentum_y = 0.0;
            for (int i = 0; i < lbm::Q; ++i) {
                const auto f = populations[lbm::population_index(i, cell, config.cells())];
                if (!std::isfinite(f) || std::abs(f - lbm::weight(i)) > config.tolerance) {
                    std::cerr << "Population mismatch: cell=" << cell << " direction=" << i
                              << " expected=" << lbm::weight(i) << " actual=" << f << '\n';
                    passed = false;
                }
                rho += f;
                momentum_x += f * lbm::cx(i);
                momentum_y += f * lbm::cy(i);
            }
            if (!std::isfinite(rho) || rho <= 0.0) {
                throw std::runtime_error("Invalid density at cell " + std::to_string(cell));
            }
            const auto ux = momentum_x / rho;
            const auto uy = momentum_y / rho;
            passed = passed && std::abs(rho - 1.0) <= config.tolerance
                     && std::isfinite(ux) && std::isfinite(uy)
                     && std::abs(ux) <= config.tolerance && std::abs(uy) <= config.tolerance;
            total_mass += rho;
            std::cout << cell << ' ' << cell % config.nx << ' ' << cell / config.nx
                      << ' ' << rho << ' ' << ux << ' ' << uy << '\n';
        }
        const auto expected_mass = static_cast<lbm::Real>(config.cells());
        const auto relative_mass_error = std::abs(total_mass - expected_mass) / expected_mass;
        passed = passed && std::isfinite(total_mass) && relative_mass_error <= config.tolerance;
        std::cout << "Total mass: " << total_mass << " (expected " << expected_mass << ")\n"
                  << "Relative mass error: " << relative_mass_error << '\n'
                  << "Tolerance: " << config.tolerance << '\n'
                  << (collision_check ? "Collision check: " : "Initialization check: ")
                  << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Checkpoint incomplete or failed: " << error.what() << '\n';
        return 1;
    }
}
