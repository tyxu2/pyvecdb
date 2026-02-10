#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return; \
        } \
    } while (0)

__global__ void l2_distance_kernel(int d, int n, const float* x, int m, const float* y, float* dists) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // index in x (0..n-1)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // index in y (0..m-1)

    if (row < n && col < m) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = x[row * d + i] - y[col * d + i];
            sum += diff * diff;
        }
        dists[row * m + col] = sum;
    }
}

namespace pyvecdb {

void compute_l2_distance_cuda(int d, int n, const float* x, int m, const float* y, float* distances) {
    float *d_x, *d_y, *d_dists;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, m * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dists, n * m * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_x, x, n * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, m * d * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    l2_distance_kernel<<<grid, block>>>(d, n, d_x, m, d_y, d_dists);

    // Copy result back
    CHECK_CUDA(cudaMemcpy(distances, d_dists, n * m * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_dists));
}

} // namespace pyvecdb
