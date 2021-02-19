#include "pca.cuh"

/**
 * @brief
 * @param args
 */
void cov_matrix(CovMatrixArgs &args) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  auto n_obs = args.n_obs;
  auto n_feats = args.n_feats;
  auto features = args.features;
  auto cov_matrix = args.cov_matrix;

  auto alpha = 1.0f / ((float) n_obs - 1);
  auto beta = 0.f;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_feats, n_feats, n_obs, &alpha,
              features, n_obs, features, n_obs, &beta, cov_matrix, n_obs);

  cudaDeviceSynchronize();

  cublasDestroy(handle);
}

