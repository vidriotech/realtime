#include "detector.cuh"

template<class T>
Detector<T>::Detector(Params &params, Probe &probe)
    : params_(params), probe_(probe), thresholds_(probe.n_total()) {
  for (auto i = 0; i < probe.n_active(); i++) {
    threshold_computers.push_back(ThresholdComputer<T>());
  }
}

/**
 * @brief Update ThresholdDetector buffers.
 * @param buf Incoming data, n_total x n_frames_buf, column-major.
 */
template<class T>
void Detector<T>::UpdateBuffer(std::vector<T> &buf) {
  data_ = std::move(buf);
  crossings_.resize(data_.size());
}

/**
 * @brief Filter samples.
 */
template<class T>
void Detector<T>::Filter() {
  if (data_.empty()) {
    return;
  }

  auto n_samples = data_.size();
  auto n_bytes = n_samples * sizeof(T);
  auto n_blocks = params_.device.n_blocks(n_samples);
  auto n_threads = params_.device.n_threads;

  // allocate device memory and copy raw samples to device
  T *cu_raw, *cu_filtered;

  cudaMallocManaged(&cu_raw, n_bytes);
  cudaMallocManaged(&cu_filtered, n_bytes);
  cudaMemcpy(cu_raw, data_.data(), n_bytes, cudaMemcpyHostToDevice);

  // do the filtering
  ndiff2<T>(n_samples, probe_.n_total(),
            cu_raw, cu_filtered, n_blocks, n_threads);

  // copy device memory back to host
  cudaMemcpy(data_.data(), cu_filtered, n_bytes, cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(cu_raw);
  cudaFree(cu_filtered);

  UpdateThresholdComputers();
}

/**
 * @brief Update channel-wise buffers for each ThresholdComputer.
 */
template<class T>
void Detector<T>::UpdateThresholdComputers() {
  std::vector<T> buf(n_frames());

  auto site_idx = 0;
  for (auto j = 0; j < probe_.n_total(); ++j) {
    if (!probe_.is_active(j)) {
      continue;
    }

    for (auto i = 0; i < n_frames(); ++i) {
      buf.at(i) = data_[j + i * probe_.n_total()];
    }

    threshold_computers[site_idx++].UpdateBuffer(buf);
  }
}

/**
 * @brief Compute thresholds for each active site.
 * @param multiplier Multiple of MAD to serve as detect.
 */
template<class T>
void Detector<T>::ComputeThresholds() {
  auto multiplier = params_.detect.thresh_multiplier;

  auto site_idx = 0;
  for (auto i = 0; i < probe_.n_total(); i++) {
    if (!probe_.is_active(i)) {
      thresholds_.at(i) = std::numeric_limits<float>::infinity();
    } else {
      thresholds_.at(i) =
          threshold_computers[site_idx].ComputeThreshold(multiplier);
      threshold_computers[site_idx++].Clear();
    }
  }
}

/**
 * @brief Find threshold crossings in the filtered data.
 */
template<class T>
void Detector<T>::FindCrossings() {
  auto n_samples = data_.size();
  auto n_bytes = n_samples * sizeof(T);
  auto n_blocks = params_.device.n_blocks(n_samples);
  auto n_threads = params_.device.n_threads;

  // allocate device memory and copy thresholds and filtered samples to device
  T *cu_filtered;
  uint8_t *cu_crossings;
  float *cu_thresholds;

  cudaMallocManaged(&cu_filtered, n_bytes);
  cudaMallocManaged(&cu_crossings, n_samples * sizeof(uint8_t));
  cudaMallocManaged(&cu_thresholds, probe_.n_total() * sizeof(float));

  cudaMemcpy(cu_filtered, data_.data(), n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(cu_thresholds, thresholds_.data(),
             probe_.n_total() * sizeof(float), cudaMemcpyHostToDevice);

  find_crossings<T>(n_samples, probe_.n_total(),
                    cu_filtered, cu_thresholds, cu_crossings,
                    n_blocks, n_threads);

  cudaMemcpy(crossings_.data(), cu_crossings, n_samples * sizeof(uint8_t),
             cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(cu_filtered);
  cudaFree(cu_crossings);
  cudaFree(cu_thresholds);
}

/**
 * @brief Detect and remove duplicate crossings.
 */
template<class T>
void Detector<T>::DedupePeaks() {
  DedupePeaksTime();
  DedupePeaksSpace();
}

/**
 * @brief Remove duplicate threshold crossings in time to find temporally
 * local peaks.
 */
template<class T>
void Detector<T>::DedupePeaksTime() {
  std::vector<uint64_t> chan_crossings;
  std::vector<uint64_t> chan_offsets;

  auto thresh =
      (uint64_t) (params_.detect.dedupe_ms * probe_.sample_rate() / 1000);
  for (auto i = 0; i < probe_.n_total(); ++i) {
    if (!probe_.is_active(i)) {
      continue;
    }

    chan_crossings.clear();
    chan_offsets.clear();
    chan_offsets.clear();

    for (auto j = 0; j < n_frames(); ++j) {
      auto k = j * probe_.n_total() + i;

      if (crossings_.at(k)) {
        chan_crossings.push_back(j);
        crossings_.at(k) = 0; // clear out the crossing at this point
      }
    }

    if (chan_crossings.empty()) {
      continue;
    }

    // partition crossings into neighborhoods and take the one with the
    // largest absolute value
    auto groups = utilities::part_nearby(chan_crossings, thresh);
    for (auto &group : groups) {
      if (group.size() == 1) {
        chan_offsets.push_back(group.at(0));
      } else {
        std::vector<uint64_t> cross_values;
        for (auto &idx : group) {
          auto val = data_[idx * probe_.n_total() + i];
          cross_values.push_back(std::abs(val));
        }

        auto best_idx = utilities::argmax(cross_values);
        chan_offsets.push_back(group.at(best_idx));
      }
    }

    // replace crossings *only* at peak sites
    for (auto &j : chan_offsets) {
      auto k = i + probe_.n_total() * j;
      crossings_.at(k) = 1;
    }
  }
}

/**
 * @brief Remove duplicate threshold crossings in time to find spatially
 * local peaks.
 */
template<class T>
void Detector<T>::DedupePeaksSpace() {
  std::vector<uint64_t> peaks;

  auto time_thresh =
      (uint64_t) (params_.detect.dedupe_ms * probe_.sample_rate() / 1000);
  auto space_thresh = params_.detect.dedupe_um;

  for (auto chan = 0; chan < probe_.n_total(); ++chan) {
    if (!probe_.is_active(chan)) {
      continue;
    }

    peaks.clear();

    // get indices of channel crossings
    for (auto frame = 0; frame < n_frames(); ++frame) {
      auto k = frame * probe_.n_total() + chan;

      if (crossings_.at(k)) {
        peaks.push_back(frame);
      }
    }

    if (peaks.empty()) {
      continue;
    }

    auto site_idx = probe_.site_index(chan);
    auto nearest_neighbors = probe_.NearestNeighbors(site_idx, 5);
    for (auto &site2 : nearest_neighbors) {
      if (peaks.empty()) {
        break;
      }

      auto chan2 = probe_.chan_index(site2);

      if (chan2 <= chan ||
          probe_.dist_between(site_idx, site2) > space_thresh) {
        continue;
      }

      /* for each peak on the current channel, check if there are any peaks
       * on neighboring channels which are also nearby in time.
       * Peaks which are larger in magnitude are preferred.
       */
      for (auto &peak : peaks) {
        auto k = chan + peak * probe_.n_total();
        auto peak_val = std::abs(data_[k]);

        for (auto frame = peak - std::min(peak, time_thresh);
             frame < std::min((uint64_t) n_frames(), peak + time_thresh);
             ++frame) {
          auto k2 = chan2 + frame * probe_.n_total();
          if (!crossings_.at(k2)) {
            continue;
          }

          auto peak_val2 = std::abs(data_[k2]);
          if (peak_val >= peak_val2) {
            crossings_.at(k2) = 0;
          } else {
            crossings_.at(k) = 0;
            break;
          }
        }
      }
    }
  }
}

template<class T>
unsigned Detector<T>::n_frames() const {
  return data_.size() / probe_.n_total();
}

template
class Detector<short>;
