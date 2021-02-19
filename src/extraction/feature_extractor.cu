#include "feature_extractor.cuh"
#include "operators.cuh"

/**
 * @brief
 * @tparam T
 * @param snippets
 */
template<class T>
void FeatureExtractor<T>::Update(std::vector<Snippet<T>> &snippets) {
  if (snippets.empty()) {
    n_feats = 0;
    return;
  }

  n_feats = snippets.at(0).size();
  auto n_snippets = snippets.size();

  // concatenate snippets into host_snippets_
  host_snippets_.clear();
  for (auto i = 0; i < n_snippets; ++i) {
    auto snip = snippets.at(i).data();
    host_snippets_.insert(host_snippets_.end(), snip.begin(), snip.end());
  }
}

/**
 * @brief
 * @tparam T
 */
template<class T>
void FeatureExtractor<T>::ComputeCovarianceMatrix() {
  if (n_feats == 0) {
    return;
  }

  CenterSnippets();

  auto n_obs = device_snippets_.size() / n_feats;

  thrust::device_ptr<float> snip_dev_ptr = device_snippets_.data();
  auto snip_ptr = thrust::raw_pointer_cast(snip_dev_ptr);

  cov_matrix_.resize(n_feats * n_feats);
  thrust::device_ptr<float> cov_dev_ptr = cov_matrix_.data();
  auto cov_ptr = thrust::raw_pointer_cast(cov_dev_ptr);

  CovMatrixArgs args{n_obs, n_feats, snip_ptr, cov_ptr};
  make_cov_matrix(args);
}

/**
 * @brief Compute the mean of each feature and subtract it, centering the new
 * means at 0.
 */
template<class T>
void FeatureExtractor<T>::CenterSnippets() {
  if (n_feats == 0) {
    return;
  }
  // copy host data to device
  device_snippets_ = host_snippets_;

  auto n_obs = host_snippets_.size() / n_feats;

  CenterFeaturesArgs args{n_obs, n_feats, device_snippets_};
  center_features(args);

  host_snippets_ = device_snippets_;
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;
