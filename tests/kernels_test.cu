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

TEST(KernelTest, MakePVs) {
  uint32_t n_feats = 7;
  thrust::device_vector<float> mat(n_feats * n_feats);

  // generate an n = 7 Wilkinson eigenvalue test matrix
  // https://en.wikipedia.org/wiki/Wilkinson_matrix
  thrust::fill(mat.begin(), mat.end(), 0);
  for (auto i = 0; i < n_feats; ++i) {
    if (i < n_feats - 1) {
      auto j = i + 1;
      mat[i * n_feats + j] = 1.0; // i, j entry
      mat[j * n_feats + i] = 1.0; // j, i entry
    }

    auto diag = i * n_feats + i;
    if (i == 0 || i == n_feats - 1) {
      mat[diag] = 3.0;
    } else if (i == 1 || i == n_feats - 2) {
      mat[diag] = 2.0;
    } else if (i == 2 || i == n_feats - 3) {
      mat[diag] = 1.0;
    }
  }

  MakePVArgs args{n_feats, n_feats, mat};
  make_principal_vectors(args);

  // principal vectors are stored in mat in column-major order,
  // but may differ by a sign
  EXPECT_LT(std::abs(std::abs(-0.036139846) - std::abs(mat[0])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.14907272) - std::abs(mat[1])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4296953) - std::abs(mat[2])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.76398057) - std::abs(mat[3])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4296953) - std::abs(mat[4])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.14907272) - std::abs(mat[5])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.036139846) - std::abs(mat[6])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.14942925) - std::abs(mat[7])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4082483) - std::abs(mat[8])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.5576775) - std::abs(mat[9])), 1e-5);
  EXPECT_LT(std::abs(std::abs(3.0957322e-15) - std::abs(mat[10])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5576775) - std::abs(mat[11])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4082483) - std::abs(mat[12])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.14942925) - std::abs(mat[13])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.25) - std::abs(mat[14])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5) - std::abs(mat[15])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.25) - std::abs(mat[16])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.5) - std::abs(mat[17])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.25) - std::abs(mat[18])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5) - std::abs(mat[19])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.25) - std::abs(mat[20])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4082483) - std::abs(mat[21])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4082483) - std::abs(mat[22])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4082483) - std::abs(mat[23])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-9.064933e-16) - std::abs(mat[24])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4082483) - std::abs(mat[25])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4082483) - std::abs(mat[26])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4082483) - std::abs(mat[27])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.37990108) - std::abs(mat[28])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.24187228) - std::abs(mat[29])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.46778008) - std::abs(mat[30])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.39586553) - std::abs(mat[31])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.46778008) - std::abs(mat[32])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.24187228) - std::abs(mat[33])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.37990108) - std::abs(mat[34])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.5576775) - std::abs(mat[35])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.4082483) - std::abs(mat[36])), 1e-5);
  EXPECT_LT(std::abs(std::abs(-0.14942925) - std::abs(mat[37])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0) - std::abs(mat[38])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.14942925) - std::abs(mat[39])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4082483) - std::abs(mat[40])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5576775) - std::abs(mat[41])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5402491) - std::abs(mat[42])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4114306) - std::abs(mat[43])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.1845094) - std::abs(mat[44])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.09810267) - std::abs(mat[45])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.1845094) - std::abs(mat[46])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.4114306) - std::abs(mat[47])), 1e-5);
  EXPECT_LT(std::abs(std::abs(0.5402491) - std::abs(mat[48])), 1e-5);
}

TEST(KernelTest, ProjectOntoPVs) {
  // 3 principal components, 5 dimensions, 10 observations
  uint32_t q = 3, d = 5, N = 10;

  // observations matrix
  thrust::device_vector<float> obs(d * N);
  thrust::sequence(obs.begin(), obs.end());

  // principal vectors
  thrust::device_vector<float> pvs(q * d);
  pvs[0] = 0.1f;
  pvs[1] = 0.6f;
  pvs[2] = -0.9f;
  pvs[3] = -1.0f;
  pvs[4] = 0.8f;
  pvs[5] = 0.4f;
  pvs[6] = -0.3f;
  pvs[7] = -0.3f;
  pvs[8] = 0.5f;
  pvs[9] = 0.5f;
  pvs[10] = 0.5f;
  pvs[11] = -0.7f;
  pvs[12] = 0.7f;
  pvs[13] = -0.3f;
  pvs[14] = 0.0f;

  ProjectOntoPVsArgs args{q, d, N, pvs, obs};
  project_onto_pvs(args);

  EXPECT_LT(std::abs(-1.0 - obs[0]), 1e-5);
  EXPECT_LT(std::abs(2.6 - obs[1]), 1e-5);
  EXPECT_LT(std::abs(-0.2 - obs[2]), 1e-5);
  EXPECT_LT(std::abs(-3.0 - obs[3]), 1e-5);
  EXPECT_LT(std::abs(6.6 - obs[4]), 1e-5);
  EXPECT_LT(std::abs(0.8 - obs[5]), 1e-5);
  EXPECT_LT(std::abs(-5.0 - obs[6]), 1e-5);
  EXPECT_LT(std::abs(10.6 - obs[7]), 1e-5);
  EXPECT_LT(std::abs(1.8 - obs[8]), 1e-5);
  EXPECT_LT(std::abs(-7.0 - obs[9]), 1e-5);
  EXPECT_LT(std::abs(14.6 - obs[10]), 1e-5);
  EXPECT_LT(std::abs(2.8 - obs[11]), 1e-5);
  EXPECT_LT(std::abs(-9.0 - obs[12]), 1e-5);
  EXPECT_LT(std::abs(18.6 - obs[13]), 1e-5);
  EXPECT_LT(std::abs(3.8 - obs[14]), 1e-5);
  EXPECT_LT(std::abs(-11.0 - obs[15]), 1e-5);
  EXPECT_LT(std::abs(22.6 - obs[16]), 1e-5);
  EXPECT_LT(std::abs(4.8 - obs[17]), 1e-5);
  EXPECT_LT(std::abs(-13.0 - obs[18]), 1e-5);
  EXPECT_LT(std::abs(26.6 - obs[19]), 1e-5);
  EXPECT_LT(std::abs(5.8 - obs[20]), 1e-5);
  EXPECT_LT(std::abs(-15.0 - obs[21]), 1e-5);
  EXPECT_LT(std::abs(30.6 - obs[22]), 1e-5);
  EXPECT_LT(std::abs(6.8 - obs[23]), 1e-5);
  EXPECT_LT(std::abs(-17.0 - obs[24]), 1e-5);
  EXPECT_LT(std::abs(34.6 - obs[25]), 1e-5);
  EXPECT_LT(std::abs(7.8 - obs[26]), 1e-5);
  EXPECT_LT(std::abs(-19.0 - obs[27]), 1e-5);
  EXPECT_LT(std::abs(38.6 - obs[28]), 1e-5);
  EXPECT_LT(std::abs(8.8 - obs[29]), 1e-5);
}
