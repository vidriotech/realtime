#include "thresholds.cuh"

template __global__
void find_crossings<short>(int n_samples, int n_channels, const short *in,
                           const float *thresholds, bool *out);
template __global__
void find_crossings<float>(int n_samples, int n_channels, const float *in,
                           const float *thresholds, bool *out);