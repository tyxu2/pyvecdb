#include "IndexHNSW.h"
#include "Distance.h"
#include <cmath>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <limits>

namespace pyvecdb {

IndexHNSW::IndexHNSW(int d, int M, int efConstruction)
    : Index(d), M(M), efConstruction(efConstruction), distribution(0.0, 1.0) {
    M_max = M;
    M_max0 = M * 2;
    level_mult = 1 / log(1.0 * M);
}

int IndexHNSW::random_level() {
    double r = distribution(generator);
    return (int)(-log(r) * level_mult);
}

float IndexHNSW::dist_func(int i, const float* x) {
    return l2_sq(data.data() + i * d, x, d);
}

float IndexHNSW::dist_func_stored(int i, int j) {
    return l2_sq(data.data() + i * d, data.data() + j * d, d);
}

void IndexHNSW::search_layer(const float* q, std::vector<std::pair<float, int>>& res, int ep, int ef, int level) {
    std::unordered_set<int> visited;
    visited.insert(ep);

    using P = std::pair<float, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> candidates; // min heap
    std::priority_queue<P> top_candidates; // max heap (furthest at top)

    float dist = dist_func(ep, q);
    candidates.push({dist, ep});
    top_candidates.push({dist, ep});

    while (!candidates.empty()) {
        P c = candidates.top();
        candidates.pop();

        float dist_c = c.first;
        if (dist_c > top_candidates.top().first && top_candidates.size() == ef) break;

        int u = c.second;
        // Check neighbors
        if (level >= nodes[u].neighbors.size()) continue; // Should not happen if logic correct

        const auto& neighbors = nodes[u].neighbors[level];
        for (int v : neighbors) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                float dist_v = dist_func(v, q);

                if (top_candidates.size() < ef || dist_v < top_candidates.top().first) {
                    candidates.push({dist_v, v});
                    top_candidates.push({dist_v, v});

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }

    // Copy result
    while (!top_candidates.empty()) {
        res.push_back(top_candidates.top());
        top_candidates.pop();
    }
}

void IndexHNSW::add(int n, const float* x) {
    for (int i = 0; i < n; ++i) {
        int id = ntotal + i;
        const float* vec = x + i * d;
        data.insert(data.end(), vec, vec + d);

        int level = random_level();
        nodes.push_back(Node{});
        nodes.back().neighbors.resize(level + 1);

        if (enter_point == -1) {
            enter_point = id;
            max_level = level;
            continue;
        }

        int curr_obj = enter_point;
        float curr_dist = dist_func(curr_obj, vec);

        // Search from top to new element's level
        for (int l = max_level; l > level; l--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const auto& neighbors = nodes[curr_obj].neighbors[l];
                for (int neighbor : neighbors) {
                    float d = dist_func(neighbor, vec);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Insert in all levels from level down to 0
        for (int l = std::min(level, max_level); l >= 0; l--) {
            // search layer to find nearest neighbors
            std::vector<std::pair<float, int>> neighbors;
            search_layer(vec, neighbors, curr_obj, efConstruction, l);

            // Connect
            // neighbors contains top efConstruction elements.
            // We select M nearest and connect bidirectional
            std::sort(neighbors.begin(), neighbors.end()); // sort by distance

            int M_curr = (l == 0) ? M_max0 : M_max;
            // Add connections from id to neighbors
            for (int j = 0; j < neighbors.size() && j < M_curr; j++) {
                int neighbor_id = neighbors[j].second;
                nodes[id].neighbors[l].push_back(neighbor_id);
                nodes[neighbor_id].neighbors[l].push_back(id);
            }

            // Prune connections of neighbors if too many
            for (int j = 0; j < neighbors.size() && j < M_curr; j++) {
                int neighbor_id = neighbors[j].second;
                if (nodes[neighbor_id].neighbors[l].size() > M_curr) {
                    // Simple shrinking: compute distances to neighbor_id and keep closest
                    // This is slow but correct.
                    auto& conn = nodes[neighbor_id].neighbors[l];
                    std::vector<std::pair<float, int>> conn_dists;
                    for (int n_idx : conn) {
                        conn_dists.push_back({dist_func_stored(neighbor_id, n_idx), n_idx});
                    }
                    std::sort(conn_dists.begin(), conn_dists.end());
                    conn.clear();
                    for (int k = 0; k < M_curr; k++) {
                        conn.push_back(conn_dists[k].second);
                    }
                }
            }

            // Update entry point for next layer (closest one)
            curr_obj = neighbors[0].second;
        }

        if (level > max_level) {
            max_level = level;
            enter_point = id;
        }
    }
    ntotal += n;
}

void IndexHNSW::search(int n, const float* x, int k, float* distances, long* labels) {
    if (ntotal == 0) return;

    for (int i = 0; i < n; i++) {
        const float* q = x + i * d;
        int curr_obj = enter_point;
        float curr_dist = dist_func(curr_obj, q);

        // Go down
        for (int l = max_level; l > 0; l--) {
            bool changed = true;
            while(changed) {
                changed = false;
                const auto& neighbors = nodes[curr_obj].neighbors[l];
                for (int neighbor : neighbors) {
                    float d = dist_func(neighbor, q);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr_obj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Layer 0
        std::vector<std::pair<float, int>> candidates; // vector<dist, id>
        search_layer(q, candidates, curr_obj, efSearch, 0);

        std::sort(candidates.begin(), candidates.end());

        for (int j = 0; j < k; j++) {
            if (j < candidates.size()) {
                distances[i * k + j] = candidates[j].first;
                labels[i * k + j] = candidates[j].second;
            } else {
                distances[i * k + j] = -1;
                labels[i * k + j] = -1;
            }
        }
    }
}

} // namespace pyvecdb
