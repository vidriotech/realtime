#include "threshold_computer.h"

template<class T>
void ThresholdComputer<T>::UpdateBuffer(std::vector<T> buf) {
  data_.assign(buf.begin(), buf.end());
  abs_dev_.assign(buf.size(), 0); // fill with zeros

  is_sorted = false;
  is_cached = false;
}

template<class T>
float ThresholdComputer<T>::ComputeThreshold(float multiplier) {
  if (is_cached) {
    return multiplier * mad / 0.6745;
  }

  auto med = median();

  // absolute deviation from the median
  for (auto i = 0; i < data_.size(); i++) {
    abs_dev_.at(i) = std::abs(data_.at(i) - med);
  }

  // median absolute deviation from the median (i.e., the MAD)
  mad = utilities::median(abs_dev_, false);
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

template
class ThresholdComputer<short>;