#ifndef RTS_2_DISTANCEMATRIX_H
#define RTS_2_DISTANCEMATRIX_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../utilities.h"

template<class T>
class DistanceMatrix {
 public:
  /**
   * @brief Matrix of distances between observations.
   * @param n_observations The number of elements whose distances are being
   * stored.
   */
  explicit DistanceMatrix(uint32_t n_observations)
      : data(n_observations * (n_observations - 1) / 2),
        n_observations_(n_observations) {};

  // getters
  T at(uint32_t i, uint32_t j) const;  // get the element at the (i,j) index
  /**
   * @brief Get the number of columns/rows in the distance matrix.
   * @return The number of columns/rows in the distance matrix.
   */
  [[nodiscard]] uint32_t n_cols() const { return n_observations_; };
  std::vector<uint32_t> closest(uint32_t i, uint32_t n);

  // setters
  void set_at(uint32_t i, uint32_t j, T val);

 private:
  uint32_t n_observations_; /*!< Number of observations/rows/columns. */
  std::vector<T> data; /*!< Compressed form of distance matrix. */

  [[nodiscard]] uint32_t index_at(uint32_t i, uint32_t j) const;
};

#endif //RTS_2_DISTANCEMATRIX_H
