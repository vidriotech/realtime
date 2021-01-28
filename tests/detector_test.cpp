#include "gtest/gtest.h"

#include <limits>
#include <vector>

#include "../src/detection/detector.h"
#include "./test_utilities/test_utilities.h"

/*
 * GIVEN a Params `params_` and a Probe `probe_`
 * DO construct a Detector `detector` AND
 * TEST THAT `thresholds` is a vector of zeros.
 */
TEST(DetectorTest, InitialState) {
  Params params;
  Probe probe = probe_from_env();

  Detector<short> detector(params, probe);

  EXPECT_EQ(probe.n_total(), detector.thresholds().size());
  for (auto &t : detector.thresholds()) {
    EXPECT_EQ(0, t);
  }
}

/*
 * GIVEN a ThresholdDetector `detector` and test data buffer `buf`
 * DO update `detector`'s buffers with `buf` AND
 *    compute the thresholds of each channel AND
 * TEST THAT each threshold is as expected.
 */
TEST(DetectorTest, DetectThresholds) {
  Params params;
  auto probe = probe_from_env();

  Detector<short> detector(params, probe);

  auto n_frames = detector.n_frames();
  auto n_samples = n_frames * probe.n_total();

  auto data = new short[n_samples];
  for (auto i = 0; i < n_frames; ++i) {
    for (auto j = 0; j < probe.n_total(); ++j) {
      auto k = i * probe.n_total() + j;

      /* Multiply each element of the sequence -1, 0, 1, -1, 0, 1, ... by
       * j + 1. The median should be 0, so the absolute
       * deviation from the median should go j, 0, j, j, 0, j, j, 0, j, ...
       * and, since ~2/3 of the absolute deviations are j-valued, j will be
       * the MAD value.
       */
      data[k] = (short) ((j + 1) * ((i % 3) - 1));
    }
  }
  std::vector<float> thresholds;

  // update buffers
  detector.UpdateBuffers(data, n_samples);
  // compute thresholds
  detector.ComputeThresholds(1.0);

  for (auto i = 0; i < probe.n_active(); ++i) {
    EXPECT_FLOAT_EQ((i + 1) / 0.6745, detector.thresholds().at(i));
  }

  for (auto i = probe.n_active(); i < probe.n_total(); ++i) {
    EXPECT_FLOAT_EQ(std::numeric_limits<float>::infinity(),
                    detector.thresholds().at(i));
  }

  delete[] data;
}