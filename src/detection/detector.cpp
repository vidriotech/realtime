#include "detector.h"

template<class T>
Detector<T>::Detector(Params &params, Probe &probe)
    : params_(params),
      probe_(probe),
      thresholds_(probe.n_total()) {
  auto nf = (int) std::ceil(params_.acquire.n_seconds * probe_.sample_rate());
  buf_.resize(nf);

  for (auto i = 0; i < probe.n_active(); i++) {
    threshold_computers.push_back(ThresholdComputer<T>(nf));
  }

  cudaMalloc(&cu_thresh, probe_.n_total() * sizeof(float));
  Realloc();
}


template<class T>
Detector<T>::~Detector() {
  if (cu_in != nullptr) {
    cudaFree(cu_in);
  }

  if (cu_out != nullptr) {
    cudaFree(cu_out);
  }

  if (cu_thresh != nullptr) {
    cudaFree(cu_thresh);
  }
}

/**
 * @brief Update ThresholdDetector buffers.
 * @param buf Incoming data, n_total x n_frames_buf, column-major.
 */
template<class T>
void Detector<T>::UpdateBuffer(std::vector<T> buf) {
  auto do_realloc = buf.size() != buf_.size();
  buf_ = buf;

  // (re)allocate memory on GPU
  if (do_realloc) {
    Realloc();
  }

  // copy buffer over
  cudaMemcpy(cu_in, buf_.data(), buf_.size() * sizeof(T),
             cudaMemcpyHostToDevice);
}

/**
 * @brief Filter samples.
 */
template<class T>
void Detector<T>::Filter() {
  if (cu_in == nullptr || cu_out == nullptr) {
    return;
  }

  auto n_blocks = params_.device.n_blocks(buf_.size());
  auto n_threads = params_.device.n_threads;
  ndiff2<T>(buf_.size(), probe_.n_total(), cu_in, cu_out, n_blocks,
            n_threads);

  cudaMemcpy(buf_.data(), cu_out, buf_.size() * sizeof(T),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(cu_in, cu_out, buf_.size() * sizeof(T),
             cudaMemcpyDeviceToDevice);

  UpdateThresholdComputers();
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
      thresholds_.at(i) = std::numeric_limits<float>::infinity();
    } else {
      thresholds_.at(i) =
          threshold_computers[site_idx++].ComputeThreshold(multiplier);
    }
  }
}

/**
 * @brief Find threshold crossings_.
 * @return A vector of threshold crossings_.
 */
template<class T>
std::vector<uint8_t> Detector<T>::FindCrossings() {
  auto n_blocks = params_.device.n_blocks(buf_.size());
  auto n_threads = params_.device.n_threads;

  cudaMemcpy(cu_thresh, thresholds_.data(),
             probe_.n_total() * sizeof (float), cudaMemcpyHostToDevice);

  unsigned char *cu_crossings;
  cudaMalloc(&cu_crossings, buf_.size() * sizeof(char));

  std::vector<T> vec_buf(buf_.size());
  cudaMemcpy(vec_buf.data(), cu_in, buf_.size() * sizeof(T),
             cudaMemcpyDeviceToHost);

  std::vector<T> vec_buf_out(buf_.size());
  cudaMemcpy(vec_buf_out.data(), cu_out, buf_.size() * sizeof(T),
             cudaMemcpyDeviceToHost);

  find_crossings(buf_.size(), probe_.n_total(), cu_in, cu_thresh,
                 (unsigned char *) cu_out, n_blocks, n_threads);
  cudaMemcpy(crossings_.data(), cu_out, buf_.size() * sizeof(char),
             cudaMemcpyDeviceToHost);

  return crossings_;
}

/**
 * @brief (Re)allocate pointers to in/out GPU memory buffers.
 */
template<class T>
void Detector<T>::Realloc() {
  if (cu_in != nullptr) {
    cudaFree(cu_in);
    cu_in = nullptr;
  }
  if (cu_out != nullptr) {
    cudaFree(cu_out);
    cu_out = nullptr;
  }

  crossings_.resize(buf_.size());

  if (buf_.empty()) {
    return;
  }

  cudaMalloc(&cu_in, buf_.size() * sizeof(T));
  cudaMalloc(&cu_out, buf_.size() * sizeof(T));
}

/**
 * @brief Update channel-wise buffers for each ThresholdComputer.
 */
template<class T>
void Detector<T>::UpdateThresholdComputers() {
  std::vector<T> threshold_buffer(n_frames());
  auto site_idx = 0;
  for (auto j = 0; j < probe_.n_total(); ++j) {
    if (!probe_.is_active(j)) {
      continue;
    }

    for (auto i = 0; i < n_frames(); ++i) {
      threshold_buffer.at(i) = buf_.at(j + i * probe_.n_total());
    }

    threshold_computers[site_idx++].UpdateBuffer(threshold_buffer);
  }
}

template
class Detector<short>;
