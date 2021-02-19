#ifndef RTS_SRC_KERNELS_OPERATORS_CUH_
#define RTS_SRC_KERNELS_OPERATORS_CUH_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// convert a linear index to a row index
struct idx_to_row_idx : public thrust::unary_function<int, int> {
  int C; // number of columns

  __host__ __device__
  explicit idx_to_row_idx(int C) : C(C) {}

  __host__ __device__
  int operator()(int i) const {
    auto row = i / C;
    return row;
  }
};

// divide value by number of observations
struct mean_functor {
  float N; // number of observations

  __host__ __device__
  explicit mean_functor(float N) : N(N) {};

  __host__ __device__
  float operator()(const float &x) const {
    return x / N;
  }
};

struct mean_subtract {
  float *means;

  __host__ __device__
  explicit mean_subtract(float *means)
      : means(means) {};

  __host__ __device__
  float operator()(const float &x, const int &row) const {
    auto mean = means == nullptr ? 0 : means[row];
    return x - mean;
  }
};

struct transpose : public thrust::unary_function<int, int> {
  int R; // number of rows
  int C; // number of columns

  __host__ __device__
  explicit transpose(int R, int C) : R(R), C(C) {}

  __host__ __device__
  int operator()(int k) const {
    return (k % R) * C + k / R;
  }
};

#endif //RTS_SRC_KERNELS_OPERATORS_CUH_
