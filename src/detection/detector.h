#ifndef RTS_2_SRC_DETECTION_DETECTOR_H_
#define RTS_2_SRC_DETECTION_DETECTOR_H_

#include <cmath>
#include <cstring>
#include <vector>

#include "../params/params.h"
#include "../probe/probe.h"
#include "threshold_computer.h"

template<class T>
class Detector {
 public:
  explicit Detector(Params &params, Probe &probe);

  // getters
  std::vector<double> &thresholds() { return thresholds_; };

 private:
  Params params_;
  Probe probe_;
  std::vector<ThresholdComputer<T>> threshold_computers;
  std::vector<double> thresholds_;
};

template<class T>
Detector<T>::Detector(Params &params, Probe &probe)
    : params_(params), probe_(probe), thresholds_(probe.n_total()) {
  auto n_samples = std::ceil(params.acquire.n_seconds * probe.sample_rate());
  for (auto i = 0; i < probe.n_active(); i++) {
    threshold_computers.push_back(ThresholdComputer<T>(n_samples));
  }
}

#endif //RTS_2_SRC_DETECTION_DETECTOR_H_
