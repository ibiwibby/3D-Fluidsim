#pragma once

#include "lbm/config.hpp"
#include <vector>

namespace lbm {

// Owns two persistent device buffers. No copies: ownership must stay unique.
class Simulation {
public:
    explicit Simulation(Config config);
    ~Simulation();
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;

    void initialize();
    // Upload a complete SoA state, e.g. a hand-constructed test fixture.
    void upload(const std::vector<Real>& populations);
    // Collision only: no streaming or buffer swap.
    void collide_only();
    // Streaming only: fill next, synchronize, then swap buffers.
    void stream_only();
    void step();
    std::vector<Real> download() const;

private:
    Config config_;
    Real* current_ = nullptr;
    Real* next_ = nullptr;
    bool initialized_ = false;
};

} // namespace lbm
