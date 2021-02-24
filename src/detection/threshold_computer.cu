#include "threshold_computer.cuh"

template<class T>
void ThresholdComputer<T>::UpdateBuffer(std::vector<T> buf) {
  thrust::host_vector<float> host_data_;

  host_data_.assign(buf.begin(), buf.end());
  device_data_ = host_data_; // copy data to device

  is_sorted = false;
  is_cached = false;
}

template<class T>
float ThresholdComputer<T>::ComputeThreshold(float multiplier) {
  if (is_cached) {
    return multiplier * mad / 0.6745;
  }

  ComputeMedian(); // sorts device_data_

  // absolute deviation from the ComputeMedian
  thrust::transform(device_data_.begin(), device_data_.end(),
                    device_data_.begin(), abs_dev(med));

  // median absolute deviation from the ComputeMedian (i.e., the MAD)
  mad = utilities::median(device_data_, false);
  is_cached = true;

  return multiplier * mad / 0.6745;
}

/**
 * @brief Compute and return the ComputeMedian of the data.
 * @return The ComputeMedian of the data.
 */
template<class T>
float ThresholdComputer<T>::ComputeMedian() {
  if (med < std::numeric_limits<float>::infinity()) {
    return med;
  }

  med = utilities::median(device_data_, is_sorted);
  is_sorted = true;

  return med;
}

template
class ThresholdComputer<short>;