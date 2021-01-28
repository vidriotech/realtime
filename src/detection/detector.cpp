#include "detector.h"

template<class T>
Detector<T>::Detector(Params &params, Probe &probe)
    : params_(params), probe_(probe), thresholds_(probe.n_total()) {
  n_frames_ = (int) std::ceil(
      params_.acquire.n_seconds * probe_.sample_rate());

  for (auto i = 0; i < probe.n_active(); i++) {
    threshold_computers.push_back(ThresholdComputer<T>(n_frames_));
  }
}

/**
 * @brief Update ThresholdDetector buffers.
 * @param buf Incoming data, n_total x n_frames, column-major.
 * @param n The *total* number of samples (n_total * n_frames) in `buf`.
 */
template<class T>
void Detector<T>::UpdateBuffers(T *buf, int n) {
  auto n_frames = n == -1 ? (int) std::ceil(
      params_.acquire.n_seconds * probe_.sample_rate()) :
                  n / probe_.n_total();

  T *threshold_buffer = new short[n_frames];

  auto site_idx = 0;
  for (auto j = 0; j < probe_.n_total(); ++j) {
    if (!probe_.is_active(j)) {
      continue;
    }

    for (auto i = 0; i < n_frames; ++i) {
      threshold_buffer[i] = buf[j + i * probe_.n_total()];
    }

    threshold_computers[site_idx++].UpdateBuffer(threshold_buffer, n_frames);
  }

  delete[] threshold_buffer; // cleanup
}

/**
 * @brief Compute thresholds for each active site.
 * @param multiplier Multiple of MAD to serve as threshold.
 */
template<class T>
void Detector<T>::ComputeThresholds(float multiplier) {
  auto site_idx = 0;
  for (auto i = 0; i < probe_.n_total(); i++) {
    if (!probe_.is_active(i)) {
      thresholds_[i] = std::numeric_limits<float>::infinity();
    } else {
      thresholds_[i] = threshold_computers[site_idx++].ComputeThreshold(multiplier);
    }
  }
}
template<class T>
std::vector<bool> Detector<T>::FindCrossings() {
  auto n_samples = n_frames_ * probe_.n_total();
  std::vector<bool> crossings(n_samples);

  return crossings;
}

template class Detector<short>;
