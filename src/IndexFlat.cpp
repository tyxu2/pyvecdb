#include "IndexFlat.h"
#include "Distance.h"
#include "Utils.h"
#include <vector>

namespace pyvecdb {

IndexFlat::IndexFlat(int d) : Index(d) {}

void IndexFlat::add(int n, const float* x) {
    data.insert(data.end(), x, x + n * d);
    ntotal += n;
}

void IndexFlat::search(int n, const float* x, int k, float* distances, long* labels) {
    if (ntotal == 0) return;

    // Compute all pairwise distances
    std::vector<float> all_dists(n * ntotal);
    compute_l2_distance(d, n, x, ntotal, data.data(), all_dists.data());

    // Find top-k
    find_top_k(k, n, ntotal, all_dists.data(), labels, distances);
}

void IndexFlat::reset() {
    data.clear();
    ntotal = 0;
}

} // namespace pyvecdb
