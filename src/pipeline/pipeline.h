#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include <cmath>
#include <memory>
#include <thread>

#include "../params/params.h"
#include "../detection/detector.h"

template<class T>
class Pipeline {
 public:
  Pipeline(Params &params, Probe &probe)
      : params_(params), probe_(probe) {};

  void Update(std::shared_ptr<T[]> buf, uint32_t buf_size,
              uint64_t frame_offset);
  void Process();

  // getters
  std::shared_ptr<T[]> buffer() const { return buf_; };
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
