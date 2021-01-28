#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include <cmath>
#include <memory>

#include "../params/params.h"
#include "../acquisition/reader.h"

template<class T>
class Pipeline {
 public:
  Pipeline(Params &params, Probe &probe);

  virtual void Run() = 0;

  // getters
  [[nodiscard]] unsigned n_frames() const;
  [[nodiscard]] unsigned frame_offset() const { return frame_offset_; };

 protected:
  Params params_;
  Probe probe_;
  std::unique_ptr<T[]> buf_;

  unsigned frame_offset_ = 0;
};

#endif //RTS_2_SRC_PIPELINE_H_
