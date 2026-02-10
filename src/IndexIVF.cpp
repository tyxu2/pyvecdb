#include "IndexIVF.h"
#include "Distance.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <numeric>

namespace pyvecdb {

IndexIVF::IndexIVF(int d, int nlist) : Index(d), nlist(nlist), quantizer(d) {
    ids.resize(nlist);
    codes.resize(nlist);
}

void IndexIVF::train(int n, const float* x) {
    if (n < nlist) {
        return;
    }

    // 1. Initialize centroids randomly
    std::vector<float> centroids(nlist * d);
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937{std::random_device{}()});

    for (int i = 0; i < nlist; ++i) {
        std::memcpy(centroids.data() + i * d, x + perm[i] * d, d * sizeof(float));
    }

    // 2. K-Means iterations
    int iterations = 10;

    for (int iter = 0; iter < iterations; ++iter) {
         quantizer.reset();
         quantizer.add(nlist, centroids.data());

         std::vector<float> dists(n);
         std::vector<long> assigns(n);
         // quantizer.search uses K=1
         quantizer.search(n, x, 1, dists.data(), assigns.data());

         std::vector<float> new_centroids(nlist * d, 0.0f);
         std::vector<int> counts(nlist, 0);

         for (int i = 0; i < n; i++) {
             int c = assigns[i];
             if (c < 0) continue;
             for (int j = 0; j < d; j++) {
                 new_centroids[c * d + j] += x[i * d + j];
             }
             counts[c]++;
         }

         for (int i = 0; i < nlist; i++) {
             if (counts[i] > 0) {
                 for (int j = 0; j < d; j++) {
                     new_centroids[i * d + j] /= counts[i];
                 }
             } else {
                 std::memcpy(new_centroids.data() + i * d, centroids.data() + i * d, d * sizeof(float));
             }
         }
         centroids = new_centroids;
    }

    quantizer.reset();
    quantizer.add(nlist, centroids.data());
    is_trained = true;
}

void IndexIVF::add(int n, const float* x) {
    if (!is_trained) {
        return;
    }

    std::vector<float> dists(n);
    std::vector<long> assigns(n);
    quantizer.search(n, x, 1, dists.data(), assigns.data());

    for (int i = 0; i < n; i++) {
        int list_no = assigns[i];
        if (list_no < 0) continue;

        ids[list_no].push_back(ntotal + i);
        codes[list_no].insert(codes[list_no].end(), x + i * d, x + (i + 1) * d);
    }
    ntotal += n;
}

void IndexIVF::search(int n, const float* x, int k, float* distances, long* labels) {
    if (!is_trained) return;

    std::vector<float> coarse_dists(n * nprobe);
    std::vector<long> coarse_ids(n * nprobe);
    quantizer.search(n, x, nprobe, coarse_dists.data(), coarse_ids.data());

    for (int i = 0; i < n; i++) {
         std::vector<float> candidates_vec;
         std::vector<long> candidates_ids;

         for (int p = 0; p < nprobe; p++) {
             long list_no = coarse_ids[i * nprobe + p];
             if (list_no < 0 || list_no >= nlist) continue;

             const auto& list_codes = codes[list_no];
             const auto& list_ids = ids[list_no];
             if (list_ids.empty()) continue;

             candidates_vec.insert(candidates_vec.end(), list_codes.begin(), list_codes.end());
             candidates_ids.insert(candidates_ids.end(), list_ids.begin(), list_ids.end());
         }

         int n_cand = candidates_ids.size();
         if (n_cand == 0) {
              for (int j = 0; j < k; j++) {
                   distances[i * k + j] = -1;
                   labels[i * k + j] = -1;
              }
              continue;
         }

         std::vector<float> cand_dists(n_cand);

         compute_l2_distance(d, 1, x + i * d, n_cand, candidates_vec.data(), cand_dists.data());

         std::vector<float> top_dists(k);
         std::vector<long> local_indices(k);

         find_top_k(k, 1, n_cand, cand_dists.data(), local_indices.data(), top_dists.data());

         for (int j = 0; j < k; j++) {
             if (local_indices[j] != -1) {
                 distances[i * k + j] = top_dists[j];
                 labels[i * k + j] = candidates_ids[local_indices[j]];
             } else {
                 distances[i * k + j] = -1;
                 labels[i * k + j] = -1;
             }
         }
    }
}

} // namespace pyvecdb
