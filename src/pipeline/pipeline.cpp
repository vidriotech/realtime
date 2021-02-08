#include <iostream>
#include "pipeline.h"

/**
 * @brief Get the number of frames in the buffer.
 * @return The number of frames in the buffer.
 */
template<class T>
uint32_t Pipeline<T>::n_frames_buf() const {
  return buf_size_ / probe_.n_total();
}

/**
 * @brief Update the data buffer, buffer size, and frame offset.
 * @param buf Shared pointer wrapping the new data buffer.
 * @param buf_size Size of new data buffer.
 * @param frame_offset Timestep at the beginning of the new data buffer.
 */
template<class T>
void Pipeline<T>::Update(std::shared_ptr<T[]> buf,
                         uint32_t buf_size,
                         uint64_t frame_offset) {
  buf_ = buf;
  buf_size_ = buf == nullptr ? 0 : buf_size;
  frame_offset_ = frame_offset;
}

/**
 * @brief Process the data in the buffer.
 */
template<class T>
void Pipeline<T>::Process() {
  if (buf_size_ == 0 || buf_ == nullptr) {
    return;
  }

  // detect
  Detector<T> detector(params_, probe_);
  detector.UpdateBuffer(buf_, buf_size_);
  detector.Filter();
  detector.ComputeThresholds(params_.threshold.multiplier);
  auto crossings = detector.FindCrossings();

  uint32_t n_crossings = 0;
  for (auto i = 0; i < buf_size_; ++i) {
    if (crossings[i]) n_crossings++;
  }

  std::cout << n_crossings << "/" << buf_size_ << " found" << std::endl;

//  Extractor<T> extractor(params_, probe_);
//  extractor.Update(detector.buffer(), n_samples);
//  extractor.MakeSnippets();
//  auto features = extractor.ExtractFeatures();
}

template
class Pipeline<short>;
