#include "gtest/gtest.h"

#include "../src/params/params.cuh"

/*
 * GIVEN a Params with the standard values and a sample rate of 30000 Hz
 * TEST THAT the number of frames to extract is 31.
 */
TEST(ParamsTest, ExtractNFrames) {
  Params params;

  EXPECT_EQ(31, params.extract.n_frames(30000));
}