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
 * @brief Compute the covariance matrix of snippet features.
 * @tparam T
 */
template<class T>
void FeatureExtractor<T>::ComputeCovarianceMatrix() {
  if (n_feats == 0) {
    return;
  }

  CenterSnippets();
  features_.resize(n_feats * n_feats);

  float *snip_ptr = thrust::raw_pointer_cast(device_snippets_.data());
  float *cov_ptr = thrust::raw_pointer_cast(features_.data());

  CovMatrixArgs args{device_snippets_.size() / n_feats,
                     n_feats, snip_ptr, cov_ptr};
  make_cov_matrix(args);
}

template<class T>
void FeatureExtractor<T>::ProjectSnippets() {
  uint32_t n_obs = (uint32_t) host_snippets_.size() / n_feats;
  MakePVArgs pv_args{n_feats, params_.extract.n_pcs, features_};
  make_principal_vectors(pv_args);

  ProjectOntoPVsArgs project_args{params_.extract.n_pcs,
                                  n_feats,
                                  n_obs,
                                  features_,
                                  device_snippets_};
  project_onto_pvs(project_args);

  // projected snippets live in features_
  features_.resize(params_.extract.n_pcs * n_obs);
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

  CenterFeaturesArgs args{host_snippets_.size() / n_feats,
                          n_feats, device_snippets_};
  center_features(args);

  std::vector<float> feats;
  feats.assign(device_snippets_.begin(), device_snippets_.end());

  host_snippets_ = device_snippets_;
}

template
class FeatureExtractor<short>;

template
class FeatureExtractor<float>;
