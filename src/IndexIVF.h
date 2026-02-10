#pragma once
#include "Index.h"
#include "IndexFlat.h"
#include <vector>

namespace pyvecdb {

class IndexIVF : public Index {
public:
    IndexIVF(int d, int nlist);

    void train(int n, const float* x) override;
    void add(int n, const float* x) override;
    void search(int n, const float* x, int k, float* distances, long* labels) override;

    void set_nprobe(int nprobe) { this->nprobe = nprobe; }

private:
    int nlist;
    int nprobe = 1;
    bool is_trained = false;
    IndexFlat quantizer;

    // Inverted lists
    std::vector<std::vector<long>> ids;
    std::vector<std::vector<float>> codes; // stored vectors
};

} // namespace pyvecdb
