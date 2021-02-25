#include "gtest/gtest.h"

#include "../src/detection/threshold_computer.cuh"

/*
 * DO construct a ThresholdComputer `computer` AND
 * TEST THAT buffer_size is 0.
 */
TEST(ThresholdComputerTest, InitialState) {
  ThresholdComputer<short> computer;

  EXPECT_EQ(0, computer.buffer_size());
}

/*
 * GIVEN a ThresholdComputer `computer` and a buffer `data`
 * DO update data underlying `computer` with the samples_ from `data_` AND
 * TEST THAT the samples_ in the `computer` data matches that in `data_`.
 */
TEST(ThresholdComputerTest, UpdateData) {
  unsigned bufsize = 1024;
  ThresholdComputer<short> computer;

  std::vector<short> data(bufsize);
  for (auto i = 0; i < bufsize; i++)
    data.at(i) = i;

  // perform the update
  computer.UpdateBuffer(data);

  EXPECT_EQ(bufsize, computer.buffer_size());
  auto computer_data = computer.data();
  ASSERT_EQ(data.size(), computer_data.size());

  for (auto i = 0; i < data.size(); ++i) {
    EXPECT_EQ(data.at(i), computer_data[i]);
  }
}

/*
 * GIVEN a ThresholdComputer `computer` with underlying data and a thresh_multiplier
 *       `thresh_multiplier`
 * DO compute the detect associated with the data and thresh_multiplier AND
 * TEST THAT the detect as computed is as expected; AND
 *           the cached value is also as expected.
 */
TEST(ThresholdComputerTest, ComputeThreshold) {
  unsigned bufsize = 100;
  ThresholdComputer<short> computer;

  std::vector<short> buf(bufsize);
  for (auto i = 0; i < bufsize; i++)
    buf.at(i) = i;

  computer.UpdateBuffer(buf);

  EXPECT_FLOAT_EQ(185.32246108228318, computer.ComputeThreshold(5));

  // should be cached now, check that it hasn't changed
  EXPECT_FLOAT_EQ(185.32246108228318, computer.ComputeThreshold(5));
}
