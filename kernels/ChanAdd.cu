#include "ChanAdd.cuh"

__global__ void sq_add(int n, const float *x, float *y) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = offset; i < n; i += stride)
        y[i] = pow(x[i], 2) + pow(y[i], 2);
}

__global__ void sq_diff(int n, const float *x, float *y) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = offset; i < n; i += stride)
        y[i] = pow(x[i] - y[i], 2);
}
