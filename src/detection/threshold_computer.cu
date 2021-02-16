#include "threshold_computer.cuh"

template<class T>
void ThresholdComputer<T>::UpdateBuffer(std::vector<T> buf) {
  data_.assign(buf.begin(), buf.end());
  host_data_.assign(buf.begin(), buf.end());
  dev_data_ = host_data_; // copy data to device

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
  for (auto i = 0; i < host_data_.size(); i++) {
    host_data_[i] = std::abs(host_data_[i] - med);
  }

  // median absolute deviation from the median (i.e., the MAD)
  dev_data_ = host_data_;
  mad = utilities::median(dev_data_, false);
  is_cached = true;

  return multiplier * mad / 0.6745;
}

/**
 * @brief Compute and return the median of the data.
 * @return The median of the data.
 */
template<class T>
double ThresholdComputer<T>::median() {
  auto med = utilities::median<T>(dev_data_, is_sorted);
  is_sorted = true;

  host_data_ = dev_data_;

  return med;
}

template
class ThresholdComputer<short>;