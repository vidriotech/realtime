#include "stats.cuh"

/**
 * @brief Compute the mean and variance of a set of observations.
 * @param N Number of observations
 * @param d Dimensionality of observations.
 * @param in Array of observations, stored as stacked observation vectors.
 * @param means Array for storing observation means.
 */
template<class T>
__global__
void mean_var(size_t N, size_t d, const T *in, float *means, float *vars) {
  unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;
}

