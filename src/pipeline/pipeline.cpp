#include <iostream>
#include "pipeline.h"

/**
 * @brief Get the number of frames in the data.
 * @return The number of frames in the data.
 */
template<class T>
uint32_t Pipeline<T>::n_frames_buf() const {
  return buf_.size() / probe_.n_total();
}

/**
 * @brief Update the data buffer, data size, and frame offset.
 * @param buf The new data data.
 * @param frame_offset Timestep at the beginning of the new data data.
 */
template<class T>
void Pipeline<T>::Update(std::vector<T> buf, uint64_t frame_offset) {
  buf_ = buf;
  frame_offset_ = frame_offset;
}

/**
 * @brief Process the data in the data.
 */
template<class T>
void Pipeline<T>::Process() {
  if (buf_.empty()) {
    return;
  }

  // detect
  Detector<T> detector(params_, probe_);
  detector.UpdateBuffer(buf_);
  detector.Filter();
  detector.ComputeThresholds(params_.detect.thresh_multiplier);
  detector.FindCrossings();

  uint32_t n_crossings = 0;
  for (auto i = 0; i < buf_.size(); ++i) {
    if (detector.crossings().at(i)) n_crossings++;
  }
  std::cout << n_crossings << "/" << buf_.size() << " found before; ";

  detector.DedupePeaks();

  n_crossings = 0;
  for (auto i = 0; i < buf_.size(); ++i) {
    if (detector.crossings().at(i)) n_crossings++;
  }
  std::cout << n_crossings << "/" << buf_.size() << " after" << std::endl;

  Extractor<T> extractor(params_, probe_);
  extractor.Update(detector.data(), detector.crossings(), frame_offset_);
  extractor.MakeSnippets();
  auto x = 1 + 1;
//  auto features = extractor.ExtractFeatures();
}

template
class Pipeline<short>;
