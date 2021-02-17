#ifndef RTS_2_SRC_STRUCTURES_SAMPLE_MATRIX_CUH_
#define RTS_2_SRC_STRUCTURES_SAMPLE_MATRIX_CUH_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * @brief A class representing samples in a an observation matrix, each row a
 * single sample.
 */
template<class T>
class SampleMatrix {
 public:
  explicit SampleMatrix(thrust::host_vector<T> samples, uint32_t n_dims)
      : h_samples_(samples), d_samples_(samples), n_dims_(n_dims) {};

  void Update(thrust::host_vector<T> samples);

  void ComputeMean();

 private:
  thrust::host_vector<T> h_samples_;
  thrust::device_vector<T> d_samples_;

  uint32_t n_dims_; /*!< dimension of each observation */
};

#endif //RTS_2_SRC_STRUCTURES_SAMPLE_MATRIX_CUH_
