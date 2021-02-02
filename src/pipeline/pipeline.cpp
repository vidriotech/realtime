#include "pipeline.h"

template<class T>
Pipeline<T>::Pipeline(Params &params, Probe &probe)
    : params_(params), probe_(probe) {
  auto n_frames = (uint32_t)
      std::ceil(probe_.sample_rate() * params_.acquire.n_seconds);
  buf_size_ = n_frames * probe_.n_total();

  buf_.reset(new T[buf_size_]);
}

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
 * @param buf Pointer to new data buffer.
 * @param buf_size Size of new data buffer.
 * @param frame_offset Timestep at the beginning of the new data buffer.
 */
template<class T>
void Pipeline<T>::Update(T *buf, uint32_t buf_size, uint64_t frame_offset) {
  buf_.reset(buf);
  buf_size_ = buf_size;
  frame_offset_ = frame_offset;
}

/**
 * @brief Process the data in the buffer.
 */
template<class T>
void Pipeline<T>::Process() {
  Detector<T> detector(params_, probe_);

  detector.UpdateBuffer(buf_, buf_size_);
  //    detector.Filter();
  detector.FindCrossings();

//    extractor.Update(detector.buffer(), n_samples);
//    extractor.MakeSnippets();
//    auto features = extractor.ExtractFeatures();
}

template
class Pipeline<short>;
