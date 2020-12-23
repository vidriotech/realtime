//
// Created by alan on 12/21/20.
//

#ifndef RTS_2_FILTERS_CUH
#define RTS_2_FILTERS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void ndiff2_i16(int N, const short *in, short *out, int nchans);
__global__ void ndiff2_f32(int N, const float *in, float *out, int nchans);

#endif //RTS_2_FILTERS_CUH
