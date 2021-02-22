#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include <cmath>
#include <memory>
#include <thread>

#include "../params/params.h"
#include "../detection/detector.cuh"
#include "../extraction/extractor.cuh"

template<class T>
class Pipeline {
 public:
  Pipeline(Params &params, Probe &probe)
      : params_(params), probe_(probe), detector_(params, probe),
        extractor_(params, probe) {};

  void Update(std::vector<T> buf, uint64_t frame_offset);
  void Process();

  // getters
  std::vector<T> buffer() const { return buf_; };
  [[nodiscard]] uint64_t frame_offset() const { return frame_offset_; };
  [[nodiscard]] uint32_t n_frames_buf() const;

 private:
  Params &params_;
  Probe &probe_;

  std::vector<T> buf_;
  uint64_t frame_offset_ = 0;

  Detector<T> detector_;
  Extractor<T> extractor_;

  void ProcessClustering(Extractor<T> &extractor);
  void ProcessClassification(Extractor<T> &extractor);
};

#endif //RTS_2_SRC_PIPELINE_H_
