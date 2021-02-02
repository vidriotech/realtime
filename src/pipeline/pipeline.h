#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include <cmath>
#include <memory>

#include "../params/params.h"
#include "../acquisition/reader.h"
#include "../detection/detector.h"

template<class T>
class Pipeline {
 public:
  Pipeline(Params &params, Probe &probe);

  void Update(T *buf, uint32_t buf_size, uint64_t frame_offset);
  void Process();

  // getters
  [[nodiscard]] uint64_t frame_offset() const { return frame_offset_; };
  [[nodiscard]] uint32_t n_frames_buf() const;

 protected:
  Params params_;
  Probe probe_;

  std::shared_ptr<T[]> buf_;
  uint32_t buf_size_ = 0;
  uint64_t frame_offset_ = 0;
};

#endif //RTS_2_SRC_PIPELINE_H_
