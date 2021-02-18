#include "feature_extractor.cuh"

template<class T>
void FeatureExtractor<T>::Update(std::vector<Snippet<T>> &snippets) {
  if (snippets.empty()) {
    dims = 0;
    return;
  }

  dims = snippets.at(0).size();
  auto n_snippets = snippets.size();

  host_snippets_.resize(dims * n_snippets);

  // store data in host snippets in order of feature, i.e., store the first
  // sample in every snippet, then move on to the second sample, etc.
  for (auto i = 0; i < n_snippets; ++i) {
    auto snip = snippets.at(i).data();
    for (auto j = 0; j < dims; ++j) {
      auto k = j * n_snippets + i;
      host_snippets_[k] = snip.at(j);
    }
  }
}

template<class T>
void FeatureExtractor<T>::ComputeCovarianceMatrix() {
  CenterSnippets();

//  // compute transpose of centered device_snippets_ matrix
//  thrust::device_vector<float> ds_transpose;
//
  auto n_obs = device_snippets_.size() / dims;
//  auto transpose_iter = thrust::make_transform_iterator
//      (thrust::counting_iterator<int>(0), transpose(n_obs, dims));
//  auto trans_perm = thrust::make_permutation_iterator(device_snippets_.begin(),
//                                                      transpose_iter);
//
//  ds_transpose.assign(trans_perm, trans_perm + device_snippets_.size());
//
//  std::vector<float> ds, dst;
//  ds.assign(device_snippets_.begin(), device_snippets_.end());
//  dst.assign(ds_transpose.begin(), ds_transpose.end());
//
//  // now multiply ds_transpose by device_snippets_
  thrust::device_ptr<float> snippets_ptr = device_snippets_.data();
  auto X = thrust::raw_pointer_cast(snippets_ptr);

  thrust::device_vector<float> cov(dims * dims);
  thrust::device_ptr<float> cov_ptr = cov.data();
  auto Sigma = thrust::raw_pointer_cast(cov_ptr);

  cublasHandle_t handle;
  cublasCreate(&handle);

  auto alpha = 1.0f / ((float) n_obs - 1);
  auto beta = 0.f;

  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dims, dims, n_obs, &alpha,
              X, n_obs, X, n_obs, &beta, Sigma, dims);
  cudaDeviceSynchronize();

  cublasDestroy(handle);
}

/**
 * @brief Compute the mean of each feature and subtract it, centering the new
 * means at 0.
 */
template<class T>
void FeatureExtractor<T>::CenterSnippets() {
  if (dims == 0) {
    return;
  }
  // copy host data to device
  device_snippets_ = host_snippets_;

  auto n_obs = host_snippets_.size() / dims;
  auto mean_iter =
      thrust::make_transform_iterator(device_snippets_.begin(),
                                      mean_functor((float) n_obs));
  auto row_iter =
      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                      idx_to_row_idx(n_obs));

  // allocate storage for row sums and indices
  thrust::device_vector<float> row_means(dims);
  thrust::device_vector<int> row_indices(dims);

  thrust::reduce_by_key
      (row_iter, row_iter + device_snippets_.size(),
       mean_iter,
       row_indices.begin(),
       row_means.begin(),
       thrust::equal_to<int>(),
       thrust::plus<float>());

  // TODO: is this the best solution?
  thrust::device_ptr<float> means_ptr = row_means.data();
  thrust::transform(device_snippets_.begin(),
                    device_snippets_.end(),
                    row_iter,
                    device_snippets_.begin(),
                    mean_subtract(thrust::raw_pointer_cast(means_ptr)));

  host_snippets_ = device_snippets_;
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;
