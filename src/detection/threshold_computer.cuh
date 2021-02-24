#ifndef RTS_2_THRESHOLDCOMPUTER_H
#define RTS_2_THRESHOLDCOMPUTER_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../kernels/operators.cuh"
#include "../utilities.cuh"

template<class T>
class ThresholdComputer {
 public:
  explicit ThresholdComputer(unsigned buf_size) {};

  void UpdateBuffer(std::vector<T> buf);
  float ComputeMedian();
  float ComputeThreshold(float multiplier);
  void Clear() { device_data_.clear(); };

  // getters
  /**
   * @brief Get the size of the data.
   * @return The size of the data.
   */
  [[nodiscard]] uint32_t buffer_size() const { return device_data_.size(); };

 private:
  thrust::device_vector<float> device_data_;
  float med = std::numeric_limits<float>::infinity();
  float mad = std::numeric_limits<float>::infinity(); /*!< cached median absolute deviation from the ComputeMedian. */
  bool is_sorted = false; /*!< true iff the data is already sorted. */
  bool is_cached = false; /*!< true iff the mad is already computed. */
};

#endif //RTS_2_THRESHOLDCOMPUTER_H
