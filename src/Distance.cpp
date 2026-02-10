#include "Distance.h"
#include <cmath>

#ifdef USE_CUDA
void compute_l2_distance_cuda(int d, int n, const float* x, int m, const float* y, float* distances);
#endif

namespace pyvecdb {

float l2_sq(const float* x, const float* y, int d) {
    float res = 0;
    for (int i = 0; i < d; i++) {
        float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

void compute_l2_distance_cpu(int d, int n, const float* x, int m, const float* y, float* distances) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            distances[i * m + j] = l2_sq(x + i * d, y + j * d, d);
        }
    }
}

void compute_l2_distance(int d, int n, const float* x, int m, const float* y, float* distances) {
#ifdef USE_CUDA
    // Dispatch to CUDA implementation
    // For simplicity, we just assume if compiled with CUDA we try to use it.
    // In a real app we might check if device is available or user requested it.
    compute_l2_distance_cuda(d, n, x, m, y, distances);
#else
    compute_l2_distance_cpu(d, n, x, m, y, distances);
#endif
}

bool is_cuda_enabled() {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}

} // namespace pyvecdb
