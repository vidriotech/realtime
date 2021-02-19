#ifndef RTS_2_SRC_KERNELS_PCA_CUH_
#define RTS_2_SRC_KERNELS_PCA_CUH_

#include <cublas_v2.h>
#include <thrust/device_vector.h>

#include "operators.cuh"

struct CenterFeaturesArgs {
  unsigned long n_obs; /*!< number of observations (rows in the matrix) */
  unsigned int n_feats; /*!< number of features (columns in the matrix) */
  thrust::device_vector<float> &features; /*!< input, feature matrix */
};

void center_features(CenterFeaturesArgs &args);

struct CovMatrixArgs {
  unsigned long n_obs; /*!< number of observations (rows in the matrix) */
  unsigned int n_feats; /*!< number of features (columns in the matrix) */
  float *features; /*!< input, centered feature matrix */
  float *cov_matrix; /*!< output, covariance matrix */
};

void make_cov_matrix(CovMatrixArgs &args);

#endif //RTS_2_SRC_KERNELS_PCA_CUH_
