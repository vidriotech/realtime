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

  CudaRealloc();
}


template<class T>
Detector<T>::~Detector() {
  if (cubuf_in != nullptr) {
    cudaFree(cubuf_in);
  }

  if (cubuf_out != nullptr) {
    cudaFree(cubuf_out);
  }
}

/**
 * @brief Update ThresholdDetector buffers.
 * @param buf Incoming data, n_total x n_frames_buf, column-major.
 * @param n The *total* number of samples (n_total * n_frames_buf) in `buf_`.
 */
template<class T>
void Detector<T>::UpdateBuffer(std::shared_ptr<T[]> buf, uint32_t buf_size) {
  auto do_realloc = buf_size != buf_size_;

  buf_ = buf;
  buf_size_ = buf_ == nullptr ? 0 : buf_size;

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

  // (re)allocate memory on GPU
  if (do_realloc) {
    CudaRealloc();
  }

  // copy buffer over
  cudaMemcpy(cubuf_in, buf_.get(), buf_size_ * sizeof(T),
             cudaMemcpyHostToDevice);
}

/**
 * @brief Filter samples.
 */
template<class T>
void Detector<T>::Filter() {
  if (cubuf_in == nullptr || cubuf_out == nullptr) {
    return;
  }

  ndiff2<T>(buf_size_, probe_.n_total(), cubuf_in, cubuf_out);
  cudaMemcpy(buf_.get(), cubuf_out, buf_size_ * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(cubuf_in, cubuf_out, buf_size_ * sizeof(T),
             cudaMemcpyDeviceToDevice);
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

/**
 * @brief Find threshold crossings.
 * @return A vector of threshold crossings.
 */
template<class T>
std::shared_ptr<bool[]> Detector<T>::FindCrossings() {
  std::shared_ptr<bool[]> crossings(new bool[buf_size_]);

  find_crossings(buf_size_, probe_.n_total(), cubuf_in, thresholds_.data(),
                 (bool *) cubuf_out);
  cudaMemcpy(crossings.get(), cubuf_out, buf_size_ * sizeof(bool),
             cudaMemcpyDeviceToHost);

  return crossings;
}

/**
 * @brief (Re)allocate pointers to in/out GPU memory buffers.
 */
template<class T>
void Detector<T>::CudaRealloc() {
  if (cubuf_in != nullptr) {
    cudaFree(cubuf_in);
    cubuf_in = nullptr;
  }
  if (cubuf_out != nullptr) {
    cudaFree(cubuf_out);
    cubuf_out = nullptr;
  }

  if (buf_size_ == 0) {
    return;
  }

  cudaMalloc(&cubuf_in, buf_size_ * sizeof(T));
  cudaMalloc(&cubuf_out, buf_size_ * sizeof(T));
}

template
class Detector<short>;
