#include "gtest/gtest.h"

#include "../src/detection/detector.h"
#include "./test_utilities/test_utilities.h"

/*
 * GIVEN a Params `params` and a Probe `probe`
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
