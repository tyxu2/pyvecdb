#pragma once
#include <vector>
#include <string>
#include <memory>

namespace pyvecdb {

class Index {
public:
    Index(int d) : d(d) {}
    virtual ~Index() = default;

    virtual void add(int n, const float* x) = 0;
    virtual void search(int n, const float* x, int k, float* distances, long* labels) = 0;

    int get_d() const { return d; }
    int get_ntotal() const { return ntotal; }

    virtual void train(int n, const float* x) {}

protected:
    int d;
    int ntotal = 0;
};

} // namespace pyvecdb
