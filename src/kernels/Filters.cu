#include "Filters.cuh"

__global__ void ndiff2_i16(int N, const short *in, short *out, int nchans) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (auto i = offset; i < N; i += stride) {
        if (i < nchans || i >= N - 2*nchans) {
            out[i] = 0;
        } else {
            out[i] = -in[i - nchans] - 2 * in[i]  + 2 * in[i + nchans] + in[i + 2 * nchans];
        }
    }
}

__global__ void ndiff2_f32(int N, const float *in, float *out, int nchans) {
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (auto i = offset; i < N; i += stride) {
        if (i < nchans || i >= N - 2*nchans) {
            out[i] = 0;
        } else {
            out[i] = -in[i - nchans] - 2 * in[i]  + 2 * in[i + nchans] + in[i + 2 * nchans];
        }
    }
}