#include "offline_pipeline.h"

template<class T>
void OfflinePipeline<T>::Run() {
  auto frame_offset_ = this->frame_offset_;
  auto n_frames_ = this->n_frames();

  reader_.AcquireFrames(frame_offset_, n_frames_, this->buf_.get());

  1 + 1;
}

template
class OfflinePipeline<short>;