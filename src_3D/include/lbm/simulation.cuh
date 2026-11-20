#pragma once
#include "config.hpp"
#include <vector>
namespace lbm {
struct Fields { std::vector<Real> density, velocity; }; // velocity: xyz interleaved
class Simulation {
public:
    explicit Simulation(Config config);
    ~Simulation();
    Simulation(const Simulation&)=delete;
    Simulation& operator=(const Simulation&)=delete;
    void initialize(bool disturbance=false,Real ux=0,Real uy=0,Real uz=0);
    void upload(const std::vector<Real>& values);
    std::vector<Real> download() const;
    Fields fields() const;
    void collide_only();
    void stream_only();
    void step();
private:
    Config c_;
    Real *current_=nullptr,*next_=nullptr,*fields_=nullptr;
    bool ready_=false;
};
}
