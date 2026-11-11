#pragma once

#include <cuda_runtime.h>
#include "lbm/config.hpp"

namespace lbm {

// Host launch wrappers. Each will launch a kernel on the supplied stream.
// Initialization, collision, and periodic pull streaming are implemented.
// Launches are asynchronous: synchronize before inspecting results/errors.
void initialize_rest(Real* current, const Config& config, cudaStream_t stream = nullptr);
// Invalid density or nonfinite state triggers a device trap; terminate the run
// on the resulting CUDA error rather than attempting to reuse the failed context.
void collide(Real* current, const Config& config, cudaStream_t stream = nullptr);
// current and next must refer to separate, non-overlapping allocations.
void stream_periodic(const Real* current, Real* next, const Config& config,
                     cudaStream_t stream = nullptr);

} // namespace lbm
