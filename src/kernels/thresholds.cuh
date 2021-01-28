#ifndef RTS_2_SRC_KERNELS_THRESHOLDS_CUH_
#define RTS_2_SRC_KERNELS_THRESHOLDS_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<class T>
__global__ void find_crossings(int n_samples, int n_channels, const T *in,
                               const float *thresholds, bool *out);

#endif //RTS_2_SRC_KERNELS_THRESHOLDS_CUH_
