#ifndef RTS_2_SRC_KERNELS_THRESHOLDS_CUH_
#define RTS_2_SRC_KERNELS_THRESHOLDS_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<class T>
__global__ void find_crossings(int n_samples, int n_channels, const T *in,
                               const float *thresholds, bool *out) {
  unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;

  for (auto i = offset; i < n_samples; i += stride) {
    out[i] = abs(in[i]) > thresholds[i % n_channels];
  }
}

#endif //RTS_2_SRC_KERNELS_THRESHOLDS_CUH_
