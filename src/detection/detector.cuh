#ifndef RTS_2_SRC_DETECTION_DETECTOR_CUH_
#define RTS_2_SRC_DETECTION_DETECTOR_CUH_

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "../utilities.cuh"
#include "../kernels/filters.cuh"
#include "../kernels/thresholds.cuh"
#include "../params/params.cuh"
#include "../probe/probe.cuh"
#include "threshold_computer.cuh"

template<class T>
class Detector {
 public:
  // rule of 5
  Detector(Params &params, Probe &probe);

  // detection sub-pipeline
  void UpdateBuffer(std::vector<T> buf);
  void Filter();
  void UpdateThresholdComputers();
  void ComputeThresholds();
  void FindCrossings();
  void DedupePeaks();

  // getters
  std::vector<T> &data() { return data_; };
  thrust::host_vector<float> &thresholds() { return thr_; };
  thrust::host_vector<uint8_t> &crossings() { return host_crossings_; };
  [[nodiscard]] unsigned n_frames() const;

 private:
  Params &params_;
  Probe &probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;

  std::vector<T> data_;
  thrust::host_vector<uint8_t> host_crossings_;
  thrust::host_vector<float> thr_;

  thrust::host_vector<T> host_buffer_;
  thrust::device_vector<T> device_buffer_;

  void DedupePeaksTime();
  void DedupePeaksSpace();
};

#endif //RTS_2_SRC_DETECTION_DETECTOR_CUH_
