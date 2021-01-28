#include "thresholds.cuh"

template<class T>
__global__ void find_crossings(int n_samples, int n_channels, const T *in,
                               const float *thresholds, bool *out) {
  unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (auto i = offset; i < n_samples; i += stride) {
    out[i] = abs(in[i]) > thresholds[i % n_channels];
  }
}

template __global__
void find_crossings<short>(int n_samples, int n_channels, const short *in,
                           const float *thresholds, bool *out);
template __global__
void find_crossings<float>(int n_samples, int n_channels, const float *in,
                           const float *thresholds, bool *out);