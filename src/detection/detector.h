#ifndef RTS_2_SRC_DETECTION_DETECTOR_H_
#define RTS_2_SRC_DETECTION_DETECTOR_H_

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../utilities.h"
#include "../kernels/filters.cuh"
#include "../kernels/thresholds.cuh"
#include "../params/params.h"
#include "../probe/probe.h"
#include "threshold_computer.h"

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
  void FindCrossings();
  void DedupePeaks();

  // getters
  std::vector<T> &data() { return data_; };
  std::vector<float> &thresholds() { return thresholds_; };
  std::vector<uint8_t> &crossings() { return crossings_; };
  [[nodiscard]] unsigned n_frames() const;

 private:
  Params params_;
  Probe probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;

  std::vector<T> data_;
  std::vector<float> thresholds_;
  std::vector<uint8_t> crossings_;

  // CUDA buffers
  T *cu_in = nullptr; /*<! GPU input data */
  T *cu_out = nullptr; /*<! GPU output data */
  float *cu_thresh = nullptr; /*<! GPU detect data */

  void DedupePeaksTime();
  void DedupePeaksSpace();

  void Realloc();
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
