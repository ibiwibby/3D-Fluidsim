#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace lbm {

inline void check_cuda(cudaError_t result, const char* expression,
                       const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line)
            + ": " + expression + ": " + cudaGetErrorString(result));
    }
}

} // namespace lbm

#define CUDA_CHECK(expression) \
    ::lbm::check_cuda((expression), #expression, __FILE__, __LINE__)
