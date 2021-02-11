#include "thresholds.cuh"

template<class T>
__global__ void find_crossings_(int n_samples, int n_channels, const T *in,
                                const float *thresholds, unsigned char *out) {
  unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (auto i = offset; i < n_samples; i += stride) {
    out[i] = in[i] < -thresholds[i % n_channels];
  }
}

template<class T>
void find_crossings(int n_samples, int n_channels, const T *in,
                    const float *thresholds, unsigned char *out,
                    long n_blocks, long n_threads) {
  find_crossings_<<<n_blocks, n_threads>>>(n_samples,
                                           n_channels, in, thresholds, out);
  cudaDeviceSynchronize();
}

template __global__
void find_crossings_<short>(int n_samples, int n_channels, const short *in,
                            const float *thresholds, unsigned char *out);
template __global__
void find_crossings_<float>(int n_samples, int n_channels, const float *in,
                            const float *thresholds, unsigned char *out);

template
void find_crossings<short>(int n_samples, int n_channels, const short *in,
                           const float *thresholds, unsigned char *out,
                           long n_blocks, long n_threads);

template
void find_crossings<float>(int n_samples, int n_channels, const float *in,
                           const float *thresholds, unsigned char *out,
                           long n_blocks, long n_threads);