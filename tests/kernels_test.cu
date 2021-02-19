#include "gtest/gtest.h"

#include <thrust/device_vector.h>

#include "../src/kernels/kernels.cuh"

TEST(KernelTest, TestNdiff2KernelShort) {
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

  ndiff2_<<<nblocks, nthreads>>>(4 * nchans, nchans, data, filtered);
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

TEST(KernelTest, TestNdiff2KernelFloat) {
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

  ndiff2_<<<nblocks, nthreads>>>(4 * nchans, nchans, data, filtered);
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

TEST(KernelTest, TestNdiff2Short) {
  auto nchans = 64;

  auto n_frames = 4;
  short *data, *filtered;
  cudaMallocManaged(&data, n_frames * nchans * sizeof(short));
  cudaMallocManaged(&filtered, n_frames * nchans * sizeof(short));

  /*
   * channel values: 1 1 2 2 -> (-1 * 1) + (-2 * 1) + (2 * 2) + (1 * 2) = 3
   */
  for (auto i = 0; i < n_frames * nchans; i++) {
    if (i < 2 * nchans) {
      data[i] = 1;
    } else {
      data[i] = 2;
    }

    filtered[i] = 0;
  }

  auto nthreads = 256;
  auto nblocks = (n_frames * nchans + nthreads - 1) / nthreads;

  ndiff2(n_frames * nchans, nchans, data, filtered, nblocks, nthreads);

  /*
   * filtered values at indices 0, 2, and 3 get 0, while channel value at index 1 gets 3
   */
  for (auto i = 0; i < n_frames * nchans; i++) {
    EXPECT_EQ((i >= nchans && i < 2 * nchans) ? 3 : 0, filtered[i]);
  }

  // cleanup
  cudaFree(data);
  cudaFree(filtered);
}

/*
 * GIVEN a buffer `data_` of int16 and a constant detect `const_thresh`
 * TEST THAT values in `data_` which exceed `const_thresh` correspond to true
 *           values in a boolean data `crossings_`.
 */
TEST(KernelTest, FindCrossingsKernelShort) {
  auto n_channels = 100;
  auto n_frames = 100;
  auto n_samples = n_channels * n_frames;
  auto const_thresh = 9.0f;

  short *data;
  uint8_t *crossings;
  float *thresholds;

  cudaMallocManaged(&data, n_samples * sizeof(short));
  cudaMallocManaged(&crossings, n_samples * sizeof(bool));
  cudaMallocManaged(&thresholds, n_channels * sizeof(float));

  for (auto i = 0; i < n_channels; ++i) {
    thresholds[i] = const_thresh;
  }

  // column j gets all j's
  for (auto k = 0; k < n_samples; ++k) {
    data[k] = (short) (-k / n_channels);
  }

  // establish preconditions for the test
  for (auto k = 0; k < n_samples; k++) {
    EXPECT_FALSE(crossings[k]);

    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(data[k] < -const_thresh);
    } else {
      EXPECT_TRUE(data[k] < -const_thresh);
    }
  }

  // perform the thresholding
  auto n_threads = 256;
  auto n_blocks = (n_samples + n_threads - 1) / n_threads;
  find_crossings_<<<n_blocks, n_threads>>>(n_samples, n_channels, data,
                                           thresholds, crossings);
  cudaDeviceSynchronize();

  // test crossings_ detected correctly
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
* GIVEN a buffer `data_` of float32 and a constant detect `const_thresh`
* TEST THAT values in `data_` which exceed `const_thresh` correspond to true
*           values in a boolean data `crossings_`.
*/
TEST(KernelTest, FindCrossingsKernelFloat) {
  auto n_channels = 100;
  auto n_frames = 100;
  auto n_samples = n_channels * n_frames;
  auto const_thresh = 9.0f;

  float *data;
  uint8_t *crossings;
  float *thresholds;

  cudaMallocManaged(&data, n_samples * sizeof(float));
  cudaMallocManaged(&crossings, n_samples * sizeof(bool));
  cudaMallocManaged(&thresholds, n_channels * sizeof(float));

  for (auto i = 0; i < n_channels; ++i) {
    thresholds[i] = const_thresh;
  }

  // column j gets all j's
  for (auto k = 0; k < n_samples; ++k) {
    data[k] = (float) (-k / n_channels); // NOLINT(bugprone-integer-division)
  }

  // establish preconditions for the test
  for (auto k = 0; k < n_samples; k++) {
    EXPECT_FALSE(crossings[k]);

    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(data[k] < -const_thresh);
    } else {
      EXPECT_TRUE(data[k] < -const_thresh);
    }
  }

  // perform the thresholding
  auto n_threads = 256;
  auto n_blocks = (n_samples + n_threads - 1) / n_threads;
  find_crossings_<<<n_blocks, n_threads>>>(n_samples, n_channels, data,
                                           thresholds, crossings);
  cudaDeviceSynchronize();

  // test crossings_ detected correctly
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

TEST(KernelTest, FindCrossingsShort) {
  auto n_channels = 100;
  auto n_frames = 100;
  auto n_samples = n_channels * n_frames;
  auto const_thresh = 9.0f;

  short *data;
  uint8_t *crossings;
  float *thresholds;

  cudaMallocManaged(&data, n_samples * sizeof(short));
  cudaMallocManaged(&crossings, n_samples * sizeof(bool));
  cudaMallocManaged(&thresholds, n_channels * sizeof(float));

  for (auto i = 0; i < n_channels; ++i) {
    thresholds[i] = const_thresh;
  }

  // column j gets all j's
  for (auto k = 0; k < n_samples; ++k) {
    data[k] = (short) (-k / n_channels);
  }

  // establish preconditions for the test
  for (auto k = 0; k < n_samples; k++) {
    EXPECT_FALSE(crossings[k]);

    if (k < n_channels * (const_thresh + 1)) {
      EXPECT_FALSE(data[k] < -const_thresh);
    } else {
      EXPECT_TRUE(data[k] < -const_thresh);
    }
  }

  // perform the thresholding
  auto n_threads = 256;
  auto n_blocks = (n_samples + n_threads - 1) / n_threads;
  find_crossings(n_samples, n_channels, data, thresholds, crossings,
                 n_blocks, n_threads);

  // test crossings_ detected correctly
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

TEST(KernelTest, TestMakeCovMatrix) {
  unsigned long n_obs = 11;
  unsigned int n_feats = 5;

  float *features;
  float *cov;

  cudaMallocManaged(&features, n_obs * n_feats * sizeof(float));
  cudaMallocManaged(&cov, n_feats * n_feats * sizeof(float));

  // store each observation in a row (row-major order)
  for (auto i = 0; i < n_feats; ++i) {
    for (auto j = 0; j < n_obs; ++j) {
      auto k = i * n_obs + j;

      features[k] = (float) (i + 1);
    }
  }

  // compute the covariance matrix
  CovMatrixArgs args{n_obs, n_feats, features, cov};
  make_cov_matrix(args);

  auto base_val = (float) n_obs / ((float) n_obs - 1);
  for (auto i = 0; i < n_feats; ++i) {
    for (auto j = 0; j < n_feats; ++j) {
      auto k = i * n_obs + j;

      EXPECT_FLOAT_EQ((i + 1) * (j + 1) * base_val, args.cov_matrix[k]);
    }
  }

  cudaFree(features);
  cudaFree(cov);
}

TEST(KernelTest, CenterFeatures) {
  unsigned long n_obs = 11;
  unsigned int n_feats = 5;

  thrust::device_vector<float> features(n_feats * n_obs);
  thrust::sequence(features.begin(), features.end());

  // center the features matrix
  CenterFeaturesArgs args{n_obs, n_feats, features};
  center_features(args);

  for (auto i = 0; i < n_feats; ++i) {
    for (auto j = 0; j < n_obs; ++j) {
      auto k = i * n_obs + j;

      EXPECT_LT(std::abs(args.features[k] - (i - 2) * 11), 1e-5);
    }
  }
}