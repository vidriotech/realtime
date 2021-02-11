#include "gtest/gtest.h"

#include <limits>

#include "../src/extraction/snippet.h"

/*
 * GIVEN a buffer `buf`, a channel count `n_chans_` and a frame count `n_frames_`
 * DO construct a Snippet `snippet` AND
 * TEST THAT the channel count and frame count are as expected; AND
 *           each value is indexed in column-major order.
 */
TEST(SnippetTest, InitialState) {
  auto n_chans = 5, n_frames = 53;

  std::vector<short> buf(n_chans * n_frames);
  for (auto i = 0; i < buf.size(); ++i) {
    buf.at(i) = i;
  }
  Snippet<short> snippet(buf, n_chans, n_frames);

  // test channel and frame counts
  EXPECT_EQ(n_chans, snippet.n_chans());
  EXPECT_EQ(n_frames, snippet.n_frames());

  for (auto frame = 0; frame < n_frames; ++frame) {
    for (auto chan = 0; chan < n_chans; ++chan) {
      auto k = frame * n_chans + chan;
      EXPECT_EQ(buf.at(k), snippet.at(chan, frame));
    }
  }
}

/*
 * GIVEN a Snippet `snippet`
 * DO compute the squared Euclidean distance between `snippet` and itself AND
 * TEST THAT the distance is 0.
 */
TEST(SnippetTest, SelfSqDist) {
  auto n_chans = 5, n_frames = 53;

  std::vector<short> buf(n_chans * n_frames);
  for (auto i = 0; i < buf.size(); ++i) {
    buf.at(i) = i;
  }
  Snippet<short> snippet(buf, n_chans, n_frames);

  EXPECT_EQ(0, snippet.SqDist(snippet));
}

/*
 * GIVEN a Snippet `snippet` and a Snippet `zero` with all zeros
 * DO compute the squared Euclidean distance between `snippet` and `zero` AND
 * TEST THAT the distance is the sum of the squares of the values in
 *           `snippet`.
 */
TEST(SnippetTest, ZeroSqDist) {
  auto n_chans = 5, n_frames = 53;

  std::vector<short> buf(n_chans * n_frames);
  double expected_dist = 0.0;
  for (auto i = 0; i < buf.size(); ++i) {
    buf.at(i) = i;
    expected_dist += i * i;
  }

  Snippet<short> snippet(buf, n_chans, n_frames);

  std::vector<short> buf0(buf.size(), 0);
  Snippet<short> zero(buf0, n_chans, n_frames);

  EXPECT_EQ(expected_dist, snippet.SqDist(zero));
}

/*
 * GIVEN a Snippet `snippet` and a Snippet `mismatch` with dimensions not
 *       equal to those of `snippet`
 * DO compute the squared Euclidean distance between `snippet` and `mismatch`
 * AND TEST THAT the distance is infinite.
 */
TEST(SnippetTest, MismatchSqDist) {
  auto n_chans = 5, n_frames = 53;

  std::vector<short> buf(n_chans * n_frames);
  for (auto i = 0; i < buf.size(); ++i) {
    buf.at(i) = i;
  }
  Snippet<short> snippet(buf, n_chans, n_frames);

  // same number of channels, but different frame count
  std::vector<short> mismatched_buf(n_chans * (n_frames + 1), 0);
  Snippet<short> mismatch(mismatched_buf, n_chans, n_frames + 1);

  EXPECT_DOUBLE_EQ(std::numeric_limits<double>::infinity(),
                   snippet.SqDist(mismatch));
}