#pragma once
#include "Index.h"
#include <vector>
namespace pyvecdb {
class IndexFlat : public Index {
public:
    IndexFlat(int d);
    void add(int n, const float* x) override;
    void search(int n, const float* x, int k, float* distances, long* labels) override;
    void reset();
private:
    std::vector<float> data;
};
} // namespace pyvecdb
