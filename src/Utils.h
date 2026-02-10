#pragma once
#include <vector>
#include <algorithm>
#include <numeric>

namespace pyvecdb {

// Helper to find top-k smallest values
// distances: n * num_points
// indices: n * k (output)
// out_dists: n * k (output)
void find_top_k(int k, int n, int num_points, const float* all_distances, long* indices, float* out_dists);

} // namespace pyvecdb
