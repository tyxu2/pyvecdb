#pragma once

#include <cstddef>

namespace pyvecdb {

void compute_l2_distance(int d, int n, const float* x, int m, const float* y, float* distances);

// Helper for single vector distance (mostly for CPU HNSW/IVF scanning)
float l2_sq(const float* x, const float* y, int d);

// Check if compiled with CUDA support
bool is_cuda_enabled();

} // namespace pyvecdb

