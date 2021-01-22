#ifndef RTS_2_THRESHOLDCOMPUTER_H
#define RTS_2_THRESHOLDCOMPUTER_H

#include <cmath>
#include <cstring>
#include <algorithm>

#include "../utilities.h"

template<class T>
class ThresholdComputer {
 public:
  ThresholdComputer(unsigned bufsize)
      : data_(bufsize) {};

  void UpdateBuffer(T *buf, int n = -1);
  double ComputeThreshold(double multiplier);

  // getters
  /**
   * @brief Get the size of the buffer.
   * @return The size of the buffer.
   */
  unsigned buffer_size() const { return data_.size(); };
  /**
   * @brief Get the underlying data buffer.
   * @return The underlying data buffer.
   */
  const std::vector<T>& data() { return data_; };
  double median();

 private:
  std::vector<T> data_;
  double mad = 0.0; /*!< cached median absolute deviation from the median. */
  bool is_sorted = false; /*!< true iff the data is already sorted. */
  bool is_cached = false; /*!< true iff the mad is already computed. */
};

template<class T>
void ThresholdComputer<T>::UpdateBuffer(T *buf, int n) {
  if (n == -1)
    n = buffer_size();

  std::memcpy(data_.data(), buf, n * sizeof(T));
  is_sorted = false;
  is_cached = false;
}

template<class T>
double ThresholdComputer<T>::ComputeThreshold(double multiplier) {
  if (is_cached) {
    return multiplier * mad / 0.6745;
  }

  auto med = median();
  std::vector<double> abs_dev(data_.size());

  // absolute deviation from the median
  for (auto i = 0; i < data_.size(); i++) {
    abs_dev[i] = std::abs(data_[i] - med);
  }

  // median absolute deviation from the median (i.e., the MAD)
  mad = utilities::median(abs_dev, false);
  is_cached = true;

  return multiplier * mad / 0.6745;
}

/**
 * @brief Compute and return the median of the data.
 * @return The median of the data.
 */
template<class T>
double ThresholdComputer<T>::median() {
  auto med = utilities::median<T>(data_, is_sorted);
  is_sorted = true;

  return med;
}

#endif //RTS_2_THRESHOLDCOMPUTER_H
