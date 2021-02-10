#include "distance_matrix.h"

/**
 * @brief Get the value of the (`i`, `j`) element in the distance matrix.
 * @param i The row of the element.
 * @param j The column of the element.
 * @return The value of the (`i`, `j`) element in the distance matrix.
 */
template<class T>
T DistanceMatrix<T>::at(uint32_t i, uint32_t j) const {
  if (i >= n_observations_ || j >= n_observations_) {
    throw std::out_of_range("Index is out of bounds for this size matrix.");
  }
  if (i == j) {
    return (T) 0;
  }

  auto idx = index_at(i, j);
  return data.at(idx);
}

/**
 * @brief Set the value at the (`i`, `j`) coordinate to `val`.
 * @param i The row in the distance matrix.
 * @param j The column in the distance matrix.
 * @param val The value to insert.
 */
template<class T>
void DistanceMatrix<T>::set_at(uint32_t i, uint32_t j, T val) {
  if (i >= n_observations_ || j >= n_observations_) {
    throw std::out_of_range("Index is out of bounds for this size matrix.");
  }
  if (i == j) {
    throw std::domain_error("Setting a diagonal element is forbidden.");
  }

  auto idx = index_at(i, j);
  data.at(idx) = val;
}

/**
 * @brief Get the index into the data vector of the (`i`, `j`) element in the
 * distance matrix.
 * @param i The row of the element.
 * @param j The column of the element.
 * @return The index into the data vector of the (`i`, `j`) element.
 */
template<class T>
uint32_t DistanceMatrix<T>::index_at(uint32_t i, uint32_t j) const {
  // accessed only from at, so we can skip a bounds check
  if (j < i) {
    return index_at(j, i);
  }

  // the index in the data_ array of the (i, j) element
  return i * n_observations_ - (i + 1) * (i + 2) / 2 + j;
}

template
class DistanceMatrix<float>;

template
class DistanceMatrix<double>;