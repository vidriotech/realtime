#include "gtest/gtest.h"

#include "../src/detection/threshold_computer.h"

/*
 * GIVEN a buffer size `bufsize`
 * DO construct a ThresholdComputer `computer` AND
 * TEST THAT buffer_size is equal to `bufsize`.
 */
TEST(ThresholdComputerTest, InitialState) {
  unsigned bufsize = 100;
  ThresholdComputer<short> computer(bufsize);

  EXPECT_EQ(bufsize, computer.buffer_size());
}

/*
 * GIVEN a ThresholdComputer `computer` and a buffer `buf_`
 * DO update buffer underlying `computer` with the data_ from `buf_` AND
 * TEST THAT the data_ in the `computer` buffer matches that in `buf_`.
 */
TEST(ThresholdComputerTest, UpdateData) {
  unsigned bufsize = 100;
  ThresholdComputer<short> computer(bufsize);

  std::vector<short> buf(bufsize);
  for (auto i = 0; i < bufsize; i++)
    buf.at(i) = i;

  // perform the update
  computer.UpdateBuffer(buf);

  for (auto i = 0; i < bufsize; i++)
    EXPECT_EQ(buf.at(i), computer.data().at(i));
}

/*
 * GIVEN a ThresholdComputer `computer`
 * DO construct a copy `computer_copy` AND
 * TEST THAT buffer_size()s match; AND
 *           data_ is equal between the underlying buffers.
 */
TEST(ThresholdComputerTest, CopyConstructor) {
  unsigned bufsize = 100;
  ThresholdComputer<short> computer(bufsize);

  std::vector<short> buf(bufsize);
  for (auto i = 0; i < bufsize; i++)
    buf.at(i) = i;

  computer.UpdateBuffer(buf);

  // establish preconditions for the test
  for (auto i = 0; i < bufsize; i++)
    EXPECT_EQ(buf.at(i), computer.data().at(i));

  // perform the copy
  ThresholdComputer<short> computer_copy(computer);

  EXPECT_EQ(computer.buffer_size(), computer_copy.buffer_size());
  for (auto i = 0; i < bufsize; i++)
    EXPECT_EQ(computer.data().at(i), computer_copy.data().at(i));
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
  ThresholdComputer<short> computer(bufsize);

  std::vector<short> buf(bufsize);
  for (auto i = 0; i < bufsize; i++)
    buf.at(i) = i;

  computer.UpdateBuffer(buf);

  EXPECT_FLOAT_EQ(185.32246, computer.ComputeThreshold(5));

  // should be cached now, check that it hasn't changed
  EXPECT_FLOAT_EQ(185.32246, computer.ComputeThreshold(5));
}
