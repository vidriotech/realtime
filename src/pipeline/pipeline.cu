#include <iostream>
#include "pipeline.cuh"

/**
 * @brief Get the number of frames in the data.
 * @return The number of frames in the data.
 */
template<class T>
uint32_t Pipeline<T>::n_frames_buf() const {
  return samples_.size() / probe_.n_total();
}

/**
 * @brief Update the data data, data size, and frame offset.
 * @param buf The new data data.
 * @param frame_offset Timestep at the beginning of the new data data.
 */
template<class T>
void Pipeline<T>::Update(std::vector<T> &buf, uint64_t frame_offset) {
  samples_ = std::move(buf);
  frame_offset_ = frame_offset;
}

/**
 * @brief Process the data in the data.
 */
template<class T>
void Pipeline<T>::Process() {
  if (samples_.empty()) {
    return;
  }

  // detect crossings
  detector_.UpdateBuffer(samples_);
  detector_.Filter();
  detector_.ComputeThresholds();
  detector_.FindCrossings();
  detector_.DedupePeaks();

  // extract snippets
  extractor_.Update(detector_.data(), detector_.crossings(), frame_offset_);
  extractor_.MakeSnippets();

  // cluster extracted snippets or switch to classification task, depending
  // on where we are in the recording
  auto n_secs = frame_offset_ / probe_.sample_rate();
  if (n_secs < params_.classify.n_secs_cluster) {
    ProcessClustering();
  } else {
    ProcessClassification();
  }

  auto tid = std::this_thread::get_id();
  std::cout << "thread " << tid << " finished processing " << frame_offset_ << std::endl;
}

/**
 * @brief Process a clustering: extract features and cluster them.
 * @param extractor
 */
template<class T>
void Pipeline<T>::ProcessClustering() {
  extractor_.ExtractFeatures();
}

template<class T>
void Pipeline<T>::ProcessClassification() {

}

template
class Pipeline<short>;
