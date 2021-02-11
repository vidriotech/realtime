#ifndef RTS_2_THRESHOLDCOMPUTER_H
#define RTS_2_THRESHOLDCOMPUTER_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

#include "../utilities.h"

template<class T>
class ThresholdComputer {
 public:
  explicit ThresholdComputer(unsigned buf_size)
      : data_(buf_size), abs_dev_(buf_size) {};

  void UpdateBuffer(std::vector<T> buf);
  float ComputeThreshold(float multiplier);

  // getters
  /**
   * @brief Get the size of the data.
   * @return The size of the data.
   */
  [[nodiscard]] uint32_t buffer_size() const { return data_.size(); };
  /**
   * @brief Get the underlying data data.
   * @return The underlying data data.
   */
  const std::vector<T>& data() { return data_; };
  double median();

 private:
  std::vector<T> data_;
  std::vector<double> abs_dev_;
  double mad = 0.0; /*!< cached median absolute deviation from the median. */
  bool is_sorted = false; /*!< true iff the data is already sorted. */
  bool is_cached = false; /*!< true iff the mad is already computed. */
};

#endif //RTS_2_THRESHOLDCOMPUTER_H
