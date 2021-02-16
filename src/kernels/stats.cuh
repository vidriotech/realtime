#ifndef RTS_2_SRC_KERNELS_STATS_CUH_
#define RTS_2_SRC_KERNELS_STATS_CUH_

template<class T>
__global__
void mean_var(size_t N, size_t d, const T *in, float *mean_out);

#endif //RTS_2_SRC_KERNELS_STATS_CUH_
