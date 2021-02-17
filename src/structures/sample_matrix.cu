#include "sample_matrix.cuh"

template<class T>
void SampleMatrix<T>::Update(thrust::host_vector<T> samples) {
  h_samples_ = samples;
  d_samples_ = samples;
}

/**
 * @brief Compute the mean of each sample
 * @tparam T
 */
template<class T>
void SampleMatrix<T>::ComputeMean() {

}

template
class SampleMatrix<short>;

template
class SampleMatrix<float>;