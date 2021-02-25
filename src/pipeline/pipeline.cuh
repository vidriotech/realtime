#ifndef RTS_2_SRC_PIPELINE_H_
#define RTS_2_SRC_PIPELINE_H_

#include <cmath>
#include <memory>
#include <thread>

#include "../params/params.cuh"
#include "../detection/detector.cuh"
#include "../extraction/extractor.cuh"

template<class T>
class Pipeline {
 public:
  Pipeline(Params &params, Probe &probe)
      : params_(params), probe_(probe),
        detector_(params, probe),
        extractor_(params, probe) {};

  void Update(std::vector<T> &buf, uint64_t frame_offset);
  void Process();

  // getters
  std::vector<T> &data() { return samples_; };
  [[nodiscard]] uint64_t frame_offset() const { return frame_offset_; };
  [[nodiscard]] uint32_t n_frames_buf() const;

 private:
  Params &params_;
  Probe &probe_;

  std::vector<T> samples_; /*!< raw (or filtered) data buffer */
  std::vector<uint8_t> crossings_; /*!< threshold crossings per sample */
  uint64_t frame_offset_ = 0;

  Detector<T> detector_;
  Extractor<T> extractor_;

  void ProcessClustering();
  void ProcessClassification();
};

#endif //RTS_2_SRC_PIPELINE_H_
