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
  void UpdateBuffer(std::vector<T> buf);
  void Filter();
  void UpdateThresholdComputers();
  void ComputeThresholds(float multiplier);
  std::vector<uint8_t> FindCrossings();

  // getters
  std::vector<T> buffer() const { return buf_; };
  std::vector<float> &thresholds() { return thresholds_; };
  std::vector<uint8_t> &crossings() { return crossings_; };
  [[nodiscard]] unsigned n_frames() const { return buf_.size() / probe_.n_total()
  ; };
  [[nodiscard]] uint32_t buffer_size() const { return buf_.size(); }

 private:
  Params params_;
  Probe probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;

  std::vector<T> buf_;
  std::vector<float> thresholds_;
  std::vector<uint8_t> crossings_;

  // CUDA buffers
  T *cu_in = nullptr;
  T *cu_out = nullptr;
  float *cu_thresh = nullptr;

  void Realloc();
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
