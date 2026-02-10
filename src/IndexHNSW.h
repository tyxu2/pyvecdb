#pragma once
#include "Index.h"
#include <vector>
#include <random>

namespace pyvecdb {

class IndexHNSW : public Index {
public:
    IndexHNSW(int d, int M = 16, int efConstruction = 200);

    void add(int n, const float* x) override;
    void search(int n, const float* x, int k, float* distances, long* labels) override;

    void set_ef(int ef) { efSearch = ef; }

private:
    int M;
    int efConstruction;
    int efSearch = 50;

    int M_max;
    int M_max0;
    double level_mult;

    std::vector<float> data;

    struct Node {
         std::vector<std::vector<int>> neighbors;
    };
    std::vector<Node> nodes;

    int enter_point = -1;
    int max_level = -1;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

    int random_level();

    // search for nearest neighbors in a specific layer
    // returns candidates (min-heap)
    void search_layer(const float* q, std::vector<std::pair<float, int>>& res, int ep, int ef, int level);

    float dist_func(int i, const float* x); // distance between stored vector i and query x
    float dist_func_stored(int i, int j); // distance between two stored vectors
};

} // namespace pyvecdb
