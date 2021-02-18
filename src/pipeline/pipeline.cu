#include <iostream>
#include "pipeline.cuh"

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
 * @brief Process the data in the buffer.
 */
template<class T>
void Pipeline<T>::Process() {
  if (buf_.empty()) {
    return;
  }

  // detect crossings
  Detector<T> detector(params_, probe_);
  detector.UpdateBuffer(buf_);
  detector.Filter();
  detector.ComputeThresholds(params_.detect.thresh_multiplier);
  detector.FindCrossings();
  detector.DedupePeaks();

  // extract snippets
  Extractor<T> extractor(params_, probe_);
  extractor.Update(detector.data(), detector.crossings(), frame_offset_);
  extractor.MakeSnippets();

  //
  auto n_secs = frame_offset_ / probe_.sample_rate();
  if (n_secs < params_.classify.n_secs_cluster) {
    ProcessClustering(extractor);
  } else {
    ProcessClassification(extractor);
  }

  auto tid = std::this_thread::get_id();
  std::cout << "thread " << tid << " finished processing " << frame_offset_ << std::endl;
}

template<class T>
void Pipeline<T>::ProcessClustering(Extractor<T> &extractor) {
  extractor.ExtractFeatures();
}

template<class T>
void Pipeline<T>::ProcessClassification(Extractor<T> &extractor) {

}

template
class Pipeline<short>;
