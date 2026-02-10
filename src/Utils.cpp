#include "Utils.h"
#include <vector>
#include <algorithm>

namespace pyvecdb {

void find_top_k(int k, int n, int num_points, const float* all_distances, long* indices, float* out_dists) {
    for (int i = 0; i < n; i++) {
        // For each query
        std::vector<std::pair<float, int>> dist_idx(num_points);
        for (int j = 0; j < num_points; j++) {
            dist_idx[j] = {all_distances[i * num_points + j], j};
        }

        // We want smallest distances
        if (k < num_points) {
            std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end());
        } else {
             std::sort(dist_idx.begin(), dist_idx.end());
        }

        for (int j = 0; j < k && j < num_points; j++) {
            out_dists[i * k + j] = dist_idx[j].first;
            indices[i * k + j] = dist_idx[j].second; // This is the index in the database
        }
        // Fill the rest if k > num_points (edge case)
        for (int j = num_points; j < k; j++) {
             out_dists[i * k + j] = -1;
             indices[i * k + j] = -1;
        }
    }
}

} // namespace pyvecdb
