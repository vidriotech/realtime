#include "threshold_computer.h"

template<class T>
void ThresholdComputer<T>::UpdateBuffer(std::shared_ptr<T[]> buf,
                                        uint32_t buf_size) {
  if (buf_size > 0 && data_.size() != buf_size) {
    data_.resize(buf_size);
  }

  std::memcpy(data_.data(), buf.get(), buf_size * sizeof(T));

  is_sorted = false;
  is_cached = false;
}

template<class T>
float ThresholdComputer<T>::ComputeThreshold(float multiplier) {
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

template
class ThresholdComputer<short>;