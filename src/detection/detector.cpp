#include "detector.h"

template<class T>
Detector<T>::Detector(Params &params, Probe &probe)
    : params_(params),
      probe_(probe),
      thresholds_(probe.n_total()) {
  auto nf = (int) std::ceil(params_.acquire.n_seconds * probe_.sample_rate());
  buf_size_ = nf * probe.n_total();

  for (auto i = 0; i < probe.n_active(); i++) {
    threshold_computers.push_back(ThresholdComputer<T>(nf));
  }
}

/**
 * @brief Update ThresholdDetector buffers.
 * @param buf Incoming data, n_total x n_frames_buf, column-major.
 * @param n The *total* number of samples (n_total * n_frames_buf) in `buf_`.
 */
template<class T>
void Detector<T>::UpdateBuffer(std::shared_ptr<T[]> buf, uint32_t buf_size) {
  buf_ = buf;
  buf_size_ = buf_size;

  std::shared_ptr<T[]> threshold_buffer(new T[n_frames()]);

  auto site_idx = 0;
  for (auto j = 0; j < probe_.n_total(); ++j) {
    if (!probe_.is_active(j)) {
      continue;
    }

    for (auto i = 0; i < n_frames(); ++i) {
      threshold_buffer[i] = buf_[j + i * probe_.n_total()];
    }

    threshold_computers[site_idx++].UpdateBuffer(threshold_buffer.get(),
                                                 n_frames());
  }
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
      thresholds_[i] =
          threshold_computers[site_idx++].ComputeThreshold(multiplier);
    }
  }
}

template<class T>
std::vector<bool> Detector<T>::FindCrossings() {
  auto n_samples = buf_size_;
  std::vector<bool> crossings(n_samples);

  return crossings;
}

template
class Detector<short>;
