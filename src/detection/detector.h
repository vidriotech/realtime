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
#include "../kernels/thresholds.cuh"

template<class T>
class Detector {
 public:
  Detector(Params &params, Probe &probe);

  void UpdateBuffer(std::unique_ptr<T[]> buf, int n = -1);
  void ComputeThresholds(float multiplier);
  std::vector<bool> FindCrossings();

  // getters
  std::vector<float> &thresholds() { return thresholds_; };
  [[nodiscard]] unsigned n_frames() const { return n_frames_; };

 private:
  unsigned n_frames_;
  Params params_;
  Probe probe_;

  std::unique_ptr<T[]> buf_;
  unsigned buf_size_;
  std::vector<ThresholdComputer<T>> threshold_computers;
  std::vector<float> thresholds_;
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
