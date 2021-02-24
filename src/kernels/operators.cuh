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
struct div_by_mean {
  float N; // number of observations

  __host__ __device__
  explicit div_by_mean(float N) : N(N) {};

  __host__ __device__
  float operator()(const float &x) const {
    return x / N;
  }
};

struct subtract_mean {
  float *means;

  __host__ __device__
  explicit subtract_mean(float *means)
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

 struct abs_dev : public thrust::unary_function<float, float> {
   float center_;

   __host__ __device__
   explicit abs_dev(float center)
   : center_(center) {};

   __host__ __device__
   float operator()(float x) const {
     auto v = x - center_;
     return v < 0 ? -v : v;
   }
 };

struct med_trans : public thrust::binary_function<float, int, float> {
  int idx[2];

  __host__ __device__
  explicit med_trans(int n) {
    idx[0] = n / 2;
    idx[1] = n % 2 == 1 ? n / 2 - 1 : -1;
  }

  __host__ __device__
  float operator()(float x, int i) const {
    if (i == idx[0] || i == idx[1]) {
      float coef = idx[1] > -1 ? 0.5f : 1.0f;
      return coef * x;
    } else {
      return 0.0f;
    }
  }
};

#endif //RTS_SRC_KERNELS_OPERATORS_CUH_
