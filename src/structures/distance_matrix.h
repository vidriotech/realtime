#ifndef RTS_2_DISTANCEMATRIX_H
#define RTS_2_DISTANCEMATRIX_H

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// T: the type of value being stored
// n_observations_: the number of elements whose distances are being stored
template<class T>
class DistanceMatrix {
 public:
  explicit DistanceMatrix(uint32_t n_observations);

  uint32_t n_cols();  // column count (also row count)

  // getters
  T at(uint32_t i, uint32_t j) const;  // get the element at the (i,j) index

  // setters
  void set_at(uint32_t i,
              uint32_t j,
              T val);  // set the element at the (i,j) index

 private:
  uint32_t n_observations_{};
  std::vector<T> data;

  uint32_t index_at(uint32_t i, uint32_t j) const;
};

#endif //RTS_2_DISTANCEMATRIX_H
