#include "gtest/gtest.h"
#include "../src/kernels/kernels.cuh"

TEST(KernelTestSuite, TestSqAdd) {
  float *x, *y;

  auto N = 1 << 20;

  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = -1.0;
  }

  int block_size = 256;
  int nblocks = (N + block_size - 1) / block_size;

  sq_add<<<nblocks, block_size>>>(N, x, y);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(2.0, y[i]);
  }

  cudaFree(x);
  cudaFree(y);
}

TEST(KernelTestSuite, TestSqDiff) {
  float *x, *y;

  auto N = 1 << 20;

  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0;
    y[i] = -1.0;
  }

  int block_size = 256;
  int nblocks = (N + block_size - 1) / block_size;

  sq_diff<<<nblocks, block_size>>>(N, x, y);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(4.0, y[i]);
  }

  cudaFree(x);
  cudaFree(y);
}

TEST(KernelTestSuite, TestNdiff2Short) {
  auto nchans = 64;

  short *data, *filtered;
  cudaMallocManaged(&data, 4 * nchans * sizeof(short));
  cudaMallocManaged(&filtered, 4 * nchans * sizeof(short));

  /*
   * channel values: 1 1 2 2 -> (-1 * 1) + (-2 * 1) + (2 * 2) + (1 * 2) = 3
   */
  for (auto i = 0; i < 4 * nchans; i++) {
    if (i < 2 * nchans) {
      data[i] = 1;
    } else {
      data[i] = 2;
    }

    filtered[i] = 0;
  }

  auto nthreads = 256;
  auto nblocks = (4 * nchans + nthreads - 1) / nthreads;

  ndiff2<<<nblocks, nthreads>>>(4 * nchans, nchans, data, filtered);
  cudaDeviceSynchronize();

  /*
   * filtered values at indices 0, 2, and 3 get 0, while channel value at index 1 gets 3
   */
  for (auto i = 0; i < 4 * nchans; i++) {
    EXPECT_EQ((i >= nchans && i < 2 * nchans) ? 3 : 0, filtered[i]);
  }

  // cleanup
  cudaFree(data);
  cudaFree(filtered);
}

TEST(KernelTestSuite, TestNdiff2Float) {
  auto nchans = 64;

  float *data, *filtered;
  cudaMallocManaged(&data, 4 * nchans * sizeof(float));
  cudaMallocManaged(&filtered, 4 * nchans * sizeof(float));

  /*
   * channel values: 1 1 2 2 -> (-1 * 1) + (-2 * 1) + (2 * 2) + (1 * 2) = 3
   */
  for (auto i = 0; i < 4 * nchans; i++) {
    if (i < 2 * nchans) {
      data[i] = 1.0f;
    } else {
      data[i] = 2.0f;
    }

    filtered[i] = 0;
  }

  auto nthreads = 256;
  auto nblocks = (4 * nchans + nthreads - 1) / nthreads;

  ndiff2<<<nblocks, nthreads>>>(4 * nchans, nchans, data, filtered);
  cudaDeviceSynchronize();

  /*
   * filtered values at indices 0, 2, and 3 get 0, while channel value at index 1 gets 3
   */
  for (auto i = 0; i < 4 * nchans; i++) {
    EXPECT_EQ((i >= nchans && i < 2 * nchans) ? 3.0f : 0.0f, filtered[i]);
  }

  // cleanup
  cudaFree(data);
  cudaFree(filtered);
}

/*
 * GIVEN a buffer `buf` of int16 and a constant threshold `const_thresh`
 * TEST THAT values in `buf` which exceed `const_thresh` correspond to true
 *           values in a boolean buffer `crossings`.
 */
TEST(KernelTestSuite, FindCrossingsShort) {
  auto n_channels = 100;
  auto n_frames = 100;
  auto n_samples = n_channels * n_frames;
  auto const_thresh = 9.0f;

  short *data;
  bool *crossings;
  float *thresholds;

  cudaMallocManaged(&data, n_samples * sizeof(short));
  cudaMallocManaged(&crossings, n_samples * sizeof(bool));
  cudaMallocManaged(&thresholds, n_channels * sizeof(float));

  for (auto i = 0; i < n_channels; ++i) {
    thresholds[i] = const_thresh;
  }

  // column j gets all j's
  for (auto k = 0; k < n_samples; ++k) {
    data[k] = (short) (k / n_channels);
  }

  // establish preconditions for the test
  for (auto k = 0; k < n_samples; k++) {
    EXPECT_FALSE(crossings[k]);

    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(data[k] > const_thresh);
    } else {
      EXPECT_TRUE(data[k] > const_thresh);
    }
  }

  // perform the thresholding
  auto n_threads = 256;
  auto n_bloacks = (n_samples + n_threads - 1) / n_threads;
  find_crossings<<<n_bloacks, n_threads>>>(n_samples, n_channels, data,
                                           thresholds, crossings);
  cudaDeviceSynchronize();

  // test crossings detected correctly
  for (auto k = 0; k < n_samples; k++) {
    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(crossings[k]);
    } else {
      EXPECT_TRUE(crossings[k]);
    }
  }

  // clean up
  cudaFree(data);
  cudaFree(crossings);
  cudaFree(thresholds);
}

/*
* GIVEN a buffer `buf` of float32 and a constant threshold `const_thresh`
* TEST THAT values in `buf` which exceed `const_thresh` correspond to true
*           values in a boolean buffer `crossings`.
*/
TEST(KernelTestSuite, FindCrossingsFloat) {
  auto n_channels = 100;
  auto n_frames = 100;
  auto n_samples = n_channels * n_frames;
  auto const_thresh = 9.0f;

  float *data;
  bool *crossings;
  float *thresholds;

  cudaMallocManaged(&data, n_samples * sizeof(float));
  cudaMallocManaged(&crossings, n_samples * sizeof(bool));
  cudaMallocManaged(&thresholds, n_channels * sizeof(float));

  for (auto i = 0; i < n_channels; ++i) {
    thresholds[i] = const_thresh;
  }

  // column j gets all j's
  for (auto k = 0; k < n_samples; ++k) {
    data[k] = (float) (k / n_channels); // NOLINT(bugprone-integer-division)
  }

  // establish preconditions for the test
  for (auto k = 0; k < n_samples; k++) {
    EXPECT_FALSE(crossings[k]);

    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(data[k] > const_thresh);
    } else {
      EXPECT_TRUE(data[k] > const_thresh);
    }
  }

  // perform the thresholding
  auto n_threads = 256;
  auto n_bloacks = (n_samples + n_threads - 1) / n_threads;
  find_crossings<<<n_bloacks, n_threads>>>(n_samples, n_channels, data,
                                           thresholds, crossings);
  cudaDeviceSynchronize();

  // test crossings detected correctly
  for (auto k = 0; k < n_samples; k++) {
    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(crossings[k]);
    } else {
      EXPECT_TRUE(crossings[k]);
    }
  }

  // clean up
  cudaFree(data);
  cudaFree(crossings);
  cudaFree(thresholds);
}
