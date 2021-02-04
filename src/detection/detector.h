#ifndef RTS_2_SRC_DETECTION_DETECTOR_H_
#define RTS_2_SRC_DETECTION_DETECTOR_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../params/params.h"
#include "../probe/probe.h"
#include "threshold_computer.h"
#include "../kernels/filters.cuh"
#include "../kernels/thresholds.cuh"

template<class T>
class Detector {
 public:
  // rule of 5
  Detector(Params &params, Probe &probe);
  ~Detector();

  // detection sub-pipeline
  void UpdateBuffer(std::shared_ptr<T[]> buf, uint32_t buf_size);
  void Filter();
  void ComputeThresholds(float multiplier);
  std::shared_ptr<bool[]> FindCrossings();

  // getters
  std::shared_ptr<T[]> buffer() const { return buf_; };
  std::vector<float> &thresholds() { return thresholds_; };
  [[nodiscard]] unsigned n_frames() const { return buf_size_ / probe_.n_total(); };
  [[nodiscard]] uint32_t buffer_size() const { return buf_size_; }

 private:
  Params params_;
  Probe probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;
  std::vector<float> thresholds_;

  uint32_t buf_size_;
  std::shared_ptr<T[]> buf_;

  // CUDA buffers
  T *cubuf_in = nullptr;
  T *cubuf_out = nullptr;

  void CudaRealloc();
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
