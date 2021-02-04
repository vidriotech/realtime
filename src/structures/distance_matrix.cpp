#include "distance_matrix.h"

template<class T>
DistanceMatrix<T>::DistanceMatrix(uint32_t n_observations)
    : data(n_observations * (n_observations - 1) / 2) {
  this->n_observations_ = n_observations;
}

template<class T>
uint32_t DistanceMatrix<T>::index_at(uint32_t i, uint32_t j) const {
  // private and accessed only from at, so we can skip a bounds check
  if (j < i) {
    return index_at(j, i);
  }

  // the index in the data_ array of the (i, j) element
  return i * n_observations_ - (i + 1) * (i + 2) / 2 + j;
}

template<class T>
uint32_t DistanceMatrix<T>::n_cols() {
  return n_observations_;
}

template<class T>
T DistanceMatrix<T>::at(uint32_t i, uint32_t j) const {
  if (i >= n_observations_ || j >= n_observations_) {
    throw std::out_of_range("Index is out of bounds for this size matrix.");
  }
  if (i == j) {
    return (T) 0;
  }

  auto idx = index_at(i, j);
  return data[idx];
}

template<class T>
void DistanceMatrix<T>::set_at(uint32_t i, uint32_t j, T val) {
  if (i >= n_observations_ || j >= n_observations_) {
    throw std::out_of_range("Index is out of bounds for this size matrix.");
  }
  if (i == j) {
    throw std::domain_error("Setting a diagonal element is forbidden.");
  }

  auto idx = index_at(i, j);
  data[idx] = val;
}

template
class DistanceMatrix<float>;

template
class DistanceMatrix<double>;