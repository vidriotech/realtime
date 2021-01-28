#ifndef RTS_2_SRC_DETECTION_DETECTOR_H_
#define RTS_2_SRC_DETECTION_DETECTOR_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "../params/params.h"
#include "../probe/probe.h"
#include "threshold_computer.h"

template<class T>
class Detector {
 public:
  explicit Detector(Params &params, Probe &probe);

  void UpdateBuffers(T *buf, int n = -1);
  void ComputeThresholds(float multiplier);
  std::vector<bool> FindCrossings();

  // getters
  std::vector<float> &thresholds() { return thresholds_; };
  [[nodiscard]] unsigned n_frames() const { return n_frames_; };

 private:
  unsigned n_frames_;
  Params params_;
  Probe probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;
  std::vector<float> thresholds_;
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
