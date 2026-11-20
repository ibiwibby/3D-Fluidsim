#pragma once
#include "simulation.cuh"
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
namespace lbm {
void write_vti(const std::filesystem::path& path,const Config& c,const Fields& fields,int step);
void write_pvd(const std::filesystem::path& path,const std::vector<std::pair<int,std::string>>& frames);
}
