#include "./filters.cuh"

template __global__
void ndiff2<short>(int n_samples, int n_chans, const short *in, short *out);

template __global__
void ndiff2<float>(int n_samples, int n_chans, const float *in, float *out);