#include "pca.cuh"

/**
 * @brief
 * @param args
 */
void center_features(CenterFeaturesArgs &args) {
  auto n_obs = args.n_obs;
  auto n_feats = args.n_feats;
  auto features = args.features;

  auto transpose_idx =
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      transpose(n_feats, n_obs));

  auto features_t =
      thrust::make_permutation_iterator(features.begin(), transpose_idx);

  auto mean_iter =
      thrust::make_transform_iterator(features_t,
                                      mean_functor((float) n_feats));

  auto row_iter =
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      idx_to_row_idx(n_feats));

  // allocate storage for row sums and indices
  thrust::device_vector<float> row_means(n_obs);
  thrust::device_vector<int> row_indices(n_obs);

  // compute the means for each column (row in the transposed matrix)
  thrust::reduce_by_key
      (row_iter, row_iter + features.size(),
       mean_iter,
       row_indices.begin(),
       row_means.begin(),
       thrust::equal_to<int>(),
       thrust::plus<float>());

  // subtract the means from each column (row in the transposed matrix)
  // copy back into features array
  thrust::device_ptr<float> means_ptr = row_means.data();
  thrust::transform(features_t,
                    features_t + (n_feats * n_obs),
                    row_iter,
                    features.begin(),
                    mean_subtract(thrust::raw_pointer_cast(means_ptr)));

  // transpose the subtracted matrix back to its original form
  transpose_idx =
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      transpose(n_obs, n_feats));

  features_t =
      thrust::make_permutation_iterator(features.begin(), transpose_idx);

  // assign back into arg.features
  args.features.assign(features_t,
                       features_t + (n_obs * n_feats));
}

/**
 * @brief Compute a covariance matrix from a matrix of centered observations.
 *
 * In the observation matrix, it's expected that each row of the matrix is an
 * observation, and since the matrix is stored in row-major order (per C/C++
 * convention), the corresponding array is stored as one observation stacked
 * after another. Moreover, the observations are expected to be centered,
 * i.e., the mean of each feature (namely, each column) should be
 * approximately zero.
 *
 * @param args Struct of arguments.
 */
void make_cov_matrix(CovMatrixArgs &args) {
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

