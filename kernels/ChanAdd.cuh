#ifndef RTS_2_CHANADD_CUH
#define RTS_2_CHANADD_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sq_add(int n, const float *x, float *y);
__global__ void sq_diff(int n, const float *x, float *y);
__global__ void ndiff2(int n, const float *x, float *y);

#endif //RTS_2_CHANADD_CUH
