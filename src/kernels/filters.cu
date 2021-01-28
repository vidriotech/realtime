#include "./filters.cuh"

template<class T>
__global__ void ndiff2(int n_samples, int n_channels, const T *in, T *out) {
  unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (auto i = offset; i < n_samples; i += stride) {
    if (i < n_channels || i >= n_samples - 2 * n_channels) {
      out[i] = 0;
    } else {
      out[i] = -in[i - n_channels] - 2 * in[i] + 2 * in[i + n_channels]
          + in[i + 2 * n_channels];
    }
  }
}

template __global__
void ndiff2<short>(int n_samples, int n_chans, const short *in, short *out);

template __global__
void ndiff2<float>(int n_samples, int n_chans, const float *in, float *out);