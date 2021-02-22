#ifndef RTS_2_SRC_KERNELS_PCA_CUH_
#define RTS_2_SRC_KERNELS_PCA_CUH_

#include <cublas_v2.h>
#include <cusolverDn.h>

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

struct MakePVArgs {
  unsigned int n_feats; /*!< number of features (columns/rows of cov_matrix) */
  unsigned int n_pcs; /*!< desired number of principal vectors */
  thrust::device_vector<float> &cov_matrix; /*!< covariance matrix of data */
};

void make_principal_vectors(MakePVArgs &args);

struct ProjectOntoPVsArgs {
  unsigned int n_pcs;
  unsigned int n_feats;
  unsigned int n_obs;
  thrust::device_vector<float> &pvs;
  thrust::device_vector<float> &observations;
  thrust::device_vector<float> &projections;
};

void project_onto_pvs(ProjectOntoPVsArgs &args);

#endif //RTS_2_SRC_KERNELS_PCA_CUH_
