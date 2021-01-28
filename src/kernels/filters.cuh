#ifndef RTS_2_FILTERS_CUH
#define RTS_2_FILTERS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<class T>
__global__ void ndiff2(int n_samples, int n_channels, const T *in, T *out);

#endif //RTS_2_FILTERS_CUH
